# -*- coding: utf-8 -*-
"""稳定版求解器客户端（E3 起使用）。

旧 Serve 的脆性（已定位）：
  1) 求解器内部用 ProcessPoolExecutor；某个 worker 被终止时它返回
     {"error": "A process in the process pool was terminated abruptly ..."}。
  2) 旧代码 p.kill() 只杀父进程，Windows 不会连带杀进程池子进程 -> 孤儿 worker 残留，
     随后重建池时内存/句柄叠加，新池更易再次崩溃。
  3) stderr=PIPE 从不读取，子进程写满管道后会阻塞。
  4) stdout.readline() 无超时，一旦子进程卡住，主程序永久挂起（表现为"进程消失/无输出"）。

本客户端：
  - stderr 写文件（保留崩溃证据，不阻塞）；
  - 读 stdout 用后台线程 + 队列 + 超时；
  - 失败时 taskkill /T 杀整棵进程树，重建池并重试整批（评估是确定性的，重试安全）；
  - serve 连续失败后降级为逐点独立进程（--stdin），只让真正有毒的点失败；
  - 所有重试事件写入 self.events，供状态文件记录。
"""
import json
import os
import queue
import subprocess
import threading
import time

import numpy as np

ROOT = os.environ.get("SDM_WORKSPACE", os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "RampingSolver.exe")
STDERR_LOG = os.path.join(ROOT, "optimization", "E3_solver_stderr.log")
CREATE_NEW_PROCESS_GROUP = 0x00000200


class RobustServe:
    def __init__(self, workers=4, batch=20, read_timeout=300):
        self.workers = workers
        self.batch = batch
        self.read_timeout = read_timeout
        self.p = None
        self.q = None
        self._errf = None
        self.events = []

    # ---------- 进程管理 ----------
    def _start(self):
        if self.p is not None:
            self._kill_tree()
        with open(STDERR_LOG, "a", encoding="utf-8") as f:
            f.write("\n=== solver start %s workers=%d ===\n"
                    % (time.strftime("%Y-%m-%d %H:%M:%S"), self.workers))
        self._errf = open(STDERR_LOG, "a", encoding="utf-8", buffering=1)
        self.p = subprocess.Popen(
            [EXE, "--serve", "--workers", str(self.workers)],
            cwd=os.path.dirname(EXE), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._errf, text=True, encoding="utf-8", bufsize=1,
            creationflags=CREATE_NEW_PROCESS_GROUP)
        self.q = queue.Queue()
        threading.Thread(target=self._reader, args=(self.p, self.q), daemon=True).start()

    @staticmethod
    def _reader(p, q):
        try:
            for line in p.stdout:
                q.put(line)
        except Exception as e:
            q.put("__ERR__%s" % e)
        finally:
            q.put(None)

    def _readline(self):
        try:
            line = self.q.get(timeout=self.read_timeout)
        except queue.Empty:
            raise RuntimeError("读 stdout 超时（%ds 无响应，疑似 worker 卡死）" % self.read_timeout)
        if line is None:
            raise RuntimeError("serve 进程已退出（stdout EOF）")
        if isinstance(line, str) and line.startswith("__ERR__"):
            raise RuntimeError("读取失败: " + line)
        return line

    def _kill_tree(self):
        if self.p is None:
            return
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.p.pid)],
                           capture_output=True, timeout=30)
        except Exception:
            pass
        try:
            self.p.kill()
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except Exception:
            pass
        self.p = None
        self.q = None
        try:
            if self._errf is not None:
                self._errf.close()
        except Exception:
            pass
        self._errf = None

    # ---------- 评估 ----------
    def evaluate(self, Wlist):
        out = []
        for s in range(0, len(Wlist), self.batch):
            out.extend(self._eval_chunk([np.asarray(w, float) for w in Wlist[s:s + self.batch]]))
        return out

    def _eval_chunk(self, chunk, tries=3):
        last = None
        for attempt in range(tries):
            try:
                if self.p is None or self.p.poll() is not None:
                    self._start()
                self.p.stdin.write("".join(
                    json.dumps(np.asarray(w, float).ravel().tolist()) + "\n" for w in chunk))
                self.p.stdin.flush()
                res = []
                for _ in chunk:
                    line = self._readline()
                    d = json.loads(line)
                    if "R" not in d:
                        raise RuntimeError("响应异常: " + line[:200])
                    res.append(float(d["R"]))
                return res
            except Exception as e:
                last = e
                self.events.append({"t": time.strftime("%H:%M:%S"), "kind": "serve_retry",
                                    "attempt": attempt + 1, "n": len(chunk),
                                    "reason": str(e)[:220]})
                print("  [重试] serve 第%d次（%d 点批）原因: %s" % (attempt + 1, len(chunk), e),
                      flush=True)
                self._kill_tree()
                time.sleep(2 + 3 * attempt)
        # 降级：逐点独立进程
        print("  [降级] serve 连续 %d 次失败，改为逐点独立进程（%d 点）" % (tries, len(chunk)),
              flush=True)
        res = []
        for w in chunk:
            r = None
            for k in range(2):
                try:
                    r = self._one(w)
                    break
                except Exception as e:
                    self.events.append({"t": time.strftime("%H:%M:%S"), "kind": "oneshot_retry",
                                        "attempt": k + 1, "reason": str(e)[:220]})
                    print("  [降级重试] 第%d次原因: %s" % (k + 1, e), flush=True)
                    time.sleep(2)
            res.append(r)
        return res

    def _one(self, w):
        p = subprocess.run([EXE, "--stdin", "--workers", "1"],
                           cwd=os.path.dirname(EXE),
                           input=json.dumps(np.asarray(w, float).ravel().tolist()),
                           capture_output=True, text=True, timeout=self.read_timeout,
                           encoding="utf-8")
        if p.returncode != 0:
            raise RuntimeError("stdin 模式退出码 %d: %s" % (p.returncode, (p.stderr or "")[-300:]))
        line = (p.stdout or "").strip().splitlines()
        if not line:
            raise RuntimeError("stdin 模式无输出")
        d = json.loads(line[-1])
        return float(d["R"]) if isinstance(d, dict) else float(d)

    def close(self):
        self._kill_tree()
