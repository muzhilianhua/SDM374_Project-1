# -*- coding: utf-8 -*-
"""
random_simplex_search.py — 方案A：3000 点均匀随机撒网 + top 候选精细坐标下降（总预算 ≤ 5000）

调用 RampingSolver.exe 的正确方式：
  - 先用 PowerShell .NET 反射从 solver.exe 提取嵌入的 RampingSolver.exe 到 %TEMP%
  - 用 --serve --workers 4 启动后台进程
  - 每次输入: json.dumps([21 floats]) + "\\n"
  - 每次输出: {"R": number}\\n

预算：3000（随机撒网） + ~2000（精修） ≤ 5000
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

# ============================ 集中配置 ============================
SEED = 20261113
N_B, N_F = 3, 7
RANDOM_SAMPLE = 1500          # 阶段1：均匀随机撒网（按用户决定截断到 1500）
SPENT_BEFORE = 840            # 上一轮已消耗的真实评估（4 serve × 210 点），计入总预算
BUDGET = 5000
REFINE_BUDGET = BUDGET - SPENT_BEFORE - RANDOM_SAMPLE   # = 2660
TOP_N_REFINE = 15
STEP = 0.002
TOP_K_FEAT = 4
IMPROVE_PATIENCE = 3
MAX_ROUNDS = 8

EPS = 1e-9
WORKERS = 2
NPROC = 4

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(ROOT, "random_search_run1")
os.makedirs(RUN_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "eval_log.jsonl")
RESULT_PATH = os.path.join(RUN_DIR, "result.json")

SOLVER_EXE_ORIG = os.path.join(ROOT, "solver", "solver.exe")
BASE_EXE = os.path.join(tempfile.gettempdir(), "RampingSolver.exe")

np.random.seed(SEED)


# ============================ 清理 PyInstaller 残留 ============================
def cleanup_pyi():
    for d in os.listdir(tempfile.gettempdir()):
        if d.startswith("_MEI") and os.path.isdir(os.path.join(tempfile.gettempdir(), d)):
            try:
                shutil.rmtree(os.path.join(tempfile.gettempdir(), d), ignore_errors=True)
            except Exception:
                pass


# ============================ 提取真实求解器 ============================
def ensure_solver():
    if os.path.exists(BASE_EXE) and os.path.getsize(BASE_EXE) > 1e7:
        return BASE_EXE
    if os.path.exists(BASE_EXE):
        os.remove(BASE_EXE)
    import base64
    ps_code = (
        "$ErrorActionPreference = 'Stop'\n"
        "$src = '{src}'\n"
        "$dst = '{dst}'\n"
        "$a = [System.Reflection.Assembly]::LoadFile($src)\n"
        "$s = $a.GetManifestResourceStream('RampingSolver.exe')\n"
        "$fs = [System.IO.File]::Create($dst)\n"
        "$s.CopyTo($fs)\n"
        "$fs.Close(); $s.Close()\n"
    ).format(src=SOLVER_EXE_ORIG.replace("\\", "/"), dst=BASE_EXE.replace("\\", "/"))
    ps_b64 = base64.b64encode(ps_code.encode("utf-16-le")).decode()
    subprocess.run(["powershell", "-NoProfile", "-EncodedCommand", ps_b64], check=True)
    assert os.path.exists(BASE_EXE) and os.path.getsize(BASE_EXE) > 1e7, "提取失败"
    print("  已提取 RampingSolver.exe size=%d" % os.path.getsize(BASE_EXE))
    return BASE_EXE


# ============================ Serve 基础设施 ============================
class Serve:
    """单个 --serve 进程。"""

    _START_LOCK = threading.Lock()

    def __init__(self):
        self.p = None
        self.lock = threading.Lock()

    def _start(self):
        exe = ensure_solver()
        with Serve._START_LOCK:
            self.p = subprocess.Popen(
                [exe, "--serve", "--workers", str(WORKERS)],
                cwd=os.path.dirname(exe),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0,  # unbuffered
            )
            time.sleep(0.6)   # 错开多个 serve 同时解压 PyInstaller 的竞争

    def _stop(self):
        if self.p is not None:
            try:
                self.p.kill()
            except Exception:
                pass
            try:
                self.p.stdout.close()
                self.p.stderr.close()
                self.p.stdin.close()
            except Exception:
                pass
            self.p = None

    def evaluate(self, w, attempts=5):
        """评估单个权重数组（21 floats），返回 R 值。失败按指数退避重建进程。"""
        payload = json.dumps(np.asarray(w, float).ravel().tolist()) + "\n"
        last = None
        with self.lock:
            for k in range(attempts):
                try:
                    if self.p is None or self.p.poll() is not None:
                        self._stop()
                        self._start()
                    self.p.stdin.write(payload.encode())
                    self.p.stdin.flush()
                    line = self.p.stdout.readline()
                    if not line:
                        err = b""
                        try:
                            err = self.p.stderr.read1(4096) or b""
                        except Exception:
                            pass
                        raise RuntimeError("serve 退出 rc=%s %s"
                                           % (self.p.poll(), err.decode("utf-8", "ignore")[:200]))
                    return float(json.loads(line)["R"])
                except Exception as e:
                    last = e
                    print("  [serve 重试 %d/%d] %s" % (k + 1, attempts, e), flush=True)
                    self._stop()
                    time.sleep(min(8.0, 1.0 * (2 ** k)))
            raise RuntimeError("serve 连续失败 %d 次: %s" % (attempts, last))

    def evaluate_batch(self, Wlist, label="", print_every=0, on_result=None):
        """顺序评估一批，带进度；on_result(w, R) 用于逐点增量落盘。"""
        out = []
        for k, w in enumerate(Wlist):
            r = self.evaluate(w)
            out.append(r)
            if on_result is not None:
                on_result(w, r)
            if print_every > 0 and (k + 1) % print_every == 0:
                best = min(out)
                print("    %s %d/%d  best_R=%.4f" % (label, k + 1, len(Wlist), best), flush=True)
        return out

    def close(self):
        with self.lock:
            if self.p is not None:
                try:
                    self.p.kill()
                except Exception:
                    pass
                self.p = None


class ParallelEval:
    """多进程并行：交错分配候选给 NPROC 个 serve 进程。"""

    def __init__(self, n=NPROC):
        self.serves = [Serve() for _ in range(n)]
        print("  已启动 %d 个 --serve 进程" % n, flush=True)

    def evaluate(self, Wlist, label="", print_every=0, log_phase=None):
        def make_logger():
            if not log_phase:
                return None

            def _log(w, r):
                rows = np.asarray(w, float).reshape(N_B, N_F)
                append_log([{"eval_id": next_eval_id(), "phase": log_phase,
                             "weights": [float(x) for x in np.ravel(w)], "R": float(r),
                             "dominant": dominant_of(rows),
                             "sig_count": sig_count(rows, 0.05),
                             "timestamp": datetime.now().isoformat(timespec="seconds")}])
            return _log

        if len(Wlist) <= 8:
            return self.serves[0].evaluate_batch(Wlist, label=label, print_every=print_every,
                                                 on_result=make_logger())
        n = len(self.serves)
        chunks = [list(range(i, len(Wlist), n)) for i in range(n)]
        res = [None] * len(Wlist)

        def work(i):
            rs = self.serves[i].evaluate_batch(
                [Wlist[j] for j in chunks[i]], label="srv%d" % (i + 1),
                print_every=print_every, on_result=make_logger())
            for j, r in zip(chunks[i], rs):
                res[j] = r

        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(work, range(n)))
        return res

    def close(self):
        for s in self.serves:
            s.close()


# ============================ 日志 ============================
LOG_LOCK = threading.Lock()


def append_log(recs):
    with LOG_LOCK, open(LOG_PATH, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


EVAL_IDX = [SPENT_BEFORE]     # 占位；main() 启动时会用 load_max_eval_id() 校正


def next_eval_id():
    with LOG_LOCK:
        v = EVAL_IDX[0]
        EVAL_IDX[0] = v + 1
        return v


# ============================ 权重工具 ============================
def build_rows(rows):
    W = np.vstack([np.asarray(r, float) for r in rows])
    W = np.clip(W, EPS, None)
    W = W / W.sum(axis=1, keepdims=True)
    return W.ravel()


def random_row(n=N_F):
    x = np.random.rand(n)
    x = -np.log(np.clip(x, 1e-300, None))
    x = x / x.sum()
    x = np.clip(x, EPS, None)
    return x / x.sum()


def random_weight():
    rows = [random_row(N_F) for _ in range(N_B)]
    return build_rows(rows)


def dominant_of(rows):
    return tuple(int(np.argmax(r)) for r in rows)


def sig_count(rows, thr=0.05):
    return sum(1 for row in rows for v in row if v >= thr)


# ============================ 阶段1：撒网（均匀 + 稀疏） ============================
def random_sparse_weight():
    """结构化稀疏采样：每栋楼随机选 1-3 个特征占据 90%+ 权重（近顶点/棱区域）。"""
    rows = []
    for _ in range(N_B):
        k = int(np.random.choice([1, 2, 3], p=[0.20, 0.45, 0.35]))
        idx = np.random.choice(N_F, size=k, replace=False)
        mass = np.random.uniform(0.90, 0.99)
        share = np.random.dirichlet(np.ones(k))
        row = np.full(N_F, EPS)
        row[idx] = mass * share
        rest = 1.0 - row.sum()
        others = [j for j in range(N_F) if j not in set(idx.tolist())]
        if others:
            row[others] = max(rest, EPS) / len(others)
        rows.append(row)
    return build_rows(rows)


def load_done_uniform():
    """从 eval_log 读取阶段1 已落盘的所有点（P1_random 均匀 + P1_sparse 稀疏）。"""
    uniform_done, sparse_done = [], []
    if not os.path.exists(LOG_PATH):
        return uniform_done, sparse_done
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ph = rec.get("phase")
            w = np.asarray(rec["weights"], float)
            r = float(rec["R"])
            if ph == "P1_random":
                uniform_done.append((w, r))
            elif ph == "P1_sparse":
                sparse_done.append((w, r))
    return uniform_done, sparse_done


def load_max_eval_id():
    """返回 eval_log 中最大的 eval_id，用于续跑不重叠编号。"""
    mx = SPENT_BEFORE - 1
    if not os.path.exists(LOG_PATH):
        return mx
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                eid = int(json.loads(line).get("eval_id", -1))
                if eid > mx:
                    mx = eid
            except Exception:
                continue
    return mx


def phase1_random(ev):
    print("\n=== 阶段1：撒网（均匀随机 + 结构化稀疏，合计 %d 点）===" % RANDOM_SAMPLE, flush=True)
    t0 = time.time()

    done_uniform, done_sparse = load_done_uniform()
    n_uniform = len(done_uniform)
    n_sparse_total = max(0, RANDOM_SAMPLE - n_uniform)   # 计划稀疏点总数
    n_sparse_done = len(done_sparse)
    n_sparse_remaining = max(0, n_sparse_total - n_sparse_done)
    print("  续跑检测: 均匀 %d 已落盘；稀疏 计划 %d，已评估 %d，待评估 %d"
          % (n_uniform, n_sparse_total, n_sparse_done, n_sparse_remaining), flush=True)

    samples = [w for w, _ in done_uniform] + [w for w, _ in done_sparse]
    Rs = [r for _, r in done_uniform] + [r for _, r in done_sparse]

    # 跳过前 n_sparse_done 个稀疏点的随机数消耗，保证新生成的点与原始计划一致
    for _ in range(n_sparse_done):
        random_sparse_weight()

    if n_sparse_remaining > 0:
        sparse = [random_sparse_weight() for _ in range(n_sparse_remaining)]
        Rs_sparse = ev.evaluate(sparse, label="P1s", print_every=5, log_phase="P1_sparse")
        samples.extend(sparse)
        Rs.extend(Rs_sparse)
        a = np.array(Rs_sparse)
        print("  稀疏点 R 分布: min=%.4f p05=%.4f p25=%.4f median=%.4f max=%.4f"
              % (a.min(), np.percentile(a, 5), np.percentile(a, 25), np.median(a), a.max()), flush=True)

    dt = time.time() - t0
    print("  耗时 %.1fs" % dt, flush=True)

    Rs = list(Rs)
    Rarr = np.array(Rs)
    print("  合并 R 分布 (n=%d): min=%.4f p05=%.4f p10=%.4f p25=%.4f median=%.4f max=%.4f"
          % (len(Rs), Rarr.min(), np.percentile(Rarr, 5), np.percentile(Rarr, 10),
             np.percentile(Rarr, 25), np.median(Rarr), Rarr.max()), flush=True)

    # dominant 统计
    from collections import Counter
    dom_counter = Counter()
    for i in range(len(samples)):
        rows = np.asarray(samples[i]).reshape(N_B, N_F)
        d = dominant_of(rows)
        dom_counter[d] += 1
    print("  dominant top10:")
    for d, c in dom_counter.most_common(10):
        best_in_d = min(Rs[i] for i in range(len(samples)) if dominant_of(np.asarray(samples[i]).reshape(N_B, N_F)) == d)
        print("    %s  n=%5d  best_R=%.4f" % (d, c, best_in_d), flush=True)

    # sig_feats 统计
    sig_counter = Counter()
    for i in range(len(samples)):
        rows = np.asarray(samples[i]).reshape(N_B, N_F)
        sig_counter[sig_count(rows, 0.05)] += 1
    print("  sig_feats 分布:")
    for k in sorted(sig_counter):
        print("    sig_feats=%2d  n=%5d" % (k, sig_counter[k]), flush=True)

    # 阶段1 结果已在评估过程中逐点落盘，此处不再重复写入

    # 选精修起点（R 最低 + dominant 去重）
    top = sorted(range(len(samples)), key=lambda i: Rs[i])
    chosen = []
    seen_dom = set()
    for rank, i in enumerate(top):
        rows = np.asarray(samples[i]).reshape(N_B, N_F)
        d = dominant_of(rows)
        if len(chosen) < 10 or d not in seen_dom:
            chosen.append({"R": float(Rs[i]), "w": samples[i].copy(),
                           "rows": rows, "rank": rank + 1, "dominant": d,
                           "sig_count": sig_count(rows, 0.05)})
            seen_dom.add(d)
        if len(chosen) >= TOP_N_REFINE:
            break

    print("\n  精修起点 %d 个（覆盖 %d 种 dominant）:" % (len(chosen), len(seen_dom)), flush=True)
    for c in chosen:
        print("    rank=%3d  R=%.4f  dom=%s  sig_feats=%d" % (c["rank"], c["R"], c["dominant"], c["sig_count"]), flush=True)

    best = {"R": float(Rs[top[0]]), "w": samples[top[0]].copy()}
    return best, chosen


# ============================ 阶段2：贪心坐标下降 ============================
def coord_candidates(rows):
    out = []
    for b in range(N_B):
        row = np.asarray(rows[b], float).copy()
        order = np.argsort(-row)[:TOP_K_FEAT]
        for i_idx in range(len(order)):
            for j_idx in range(len(order)):
                if i_idx == j_idx:
                    continue
                i, j = int(order[i_idx]), int(order[j_idx])
                for sign in (+1, -1):
                    r2 = row.copy()
                    r2[i] += sign * STEP
                    r2[j] -= sign * STEP
                    if r2[i] > EPS * 0.5 and r2[j] > EPS * 0.5:
                        rr = [np.asarray(x, float).copy() for x in rows]
                        rr[b] = r2
                        out.append(rr)
    return out


def phase2_refine(ev, chosen, refine_budget):
    print("\n=== 阶段2：精修（预算 %d）===" % refine_budget, flush=True)
    best_ever = {"R": float("inf"), "w": None, "label": ""}
    used = 0

    for si, start in enumerate(chosen):
        if used >= refine_budget:
            print("  预算耗尽，提前停止", flush=True)
            break

        rows = [np.asarray(r, float).copy() for r in start["rows"]]
        curW = build_rows(rows)
        curR = float(start["R"])
        print("\n  起点%d dom=%s R=%.4f" % (si + 1, start["dominant"], curR), flush=True)

        patience = 0
        for rd in range(1, MAX_ROUNDS + 1):
            if used >= refine_budget:
                break

            cands = coord_candidates(rows)
            # 去重
            seen = set()
            unique_cands = []
            for rr in cands:
                k = tuple(np.round(build_rows(rr), 12))
                if k not in seen:
                    seen.add(k)
                    unique_cands.append(rr)
            cands = unique_cands

            if not cands:
                break

            # 截断到预算
            take = min(len(cands), refine_budget - used)
            chunk = [build_rows(cands[j]) for j in range(take)]
            Rs = ev.evaluate(chunk, label="P2s%dr%d" % (si + 1, rd), print_every=5)
            used += take

            best_idx = int(np.argmin(Rs))
            best_R_round = float(Rs[best_idx])

            # 落盘本轮
            recs = []
            for j, (cw, cr) in enumerate(zip(chunk, Rs)):
                rows_j = np.asarray(cw).reshape(N_B, N_F)
                recs.append({"eval_id": next_eval_id(),
                             "phase": "P2_refine_s%d_r%d" % (si + 1, rd),
                             "weights": cw.tolist(), "R": float(cr),
                             "dominant": dominant_of(rows_j),
                             "sig_count": sig_count(rows_j, 0.05)})
            append_log(recs)

            if best_R_round < curR - 1e-12:
                rows = [np.asarray(cands[best_idx][j], float).copy() for j in range(N_B)]
                curR = best_R_round
                patience = 0
                print("    轮%d → R=%.4f  剩余预算 %d" % (rd, curR, refine_budget - used), flush=True)
                if curR < best_ever["R"]:
                    best_ever = {"R": curR, "w": build_rows(rows).copy(),
                                 "label": "P2_s%d_r%d dom=%s" % (si + 1, rd, start["dominant"])}
            else:
                patience += 1
                print("    轮%d 无改善  best_round=%.4f cur=%.4f patience=%d/%d 剩余 %d"
                      % (rd, best_R_round, curR, patience, IMPROVE_PATIENCE, refine_budget - used), flush=True)
                if patience >= IMPROVE_PATIENCE:
                    break

    return best_ever, used


# ============================ 主入口 ============================
def main():
    t_total = time.time()
    print("=== random_simplex_search 开始 ===", flush=True)
    print("  SEED=%d  RANDOM_SAMPLE=%d  BUDGET=%d  SPENT_BEFORE=%d  REFINE_BUDGET=%d"
          % (SEED, RANDOM_SAMPLE, BUDGET, SPENT_BEFORE, REFINE_BUDGET), flush=True)
    print("  TOP_N_REFINE=%d  STEP=%s  TOP_K_FEAT=%d"
          % (TOP_N_REFINE, STEP, TOP_K_FEAT), flush=True)
    print("  工作目录: %s" % RUN_DIR, flush=True)

    cleanup_pyi()
    ensure_solver()
    EVAL_IDX[0] = load_max_eval_id() + 1   # 校正：接上现有 eval_id
    print("  续跑起点 eval_id = %d" % EVAL_IDX[0], flush=True)
    ev = ParallelEval()
    try:
        best_random, chosen = phase1_random(ev)
        best_ever, refine_used = phase2_refine(ev, chosen, REFINE_BUDGET)

        total_used = EVAL_IDX[0]
        final_best = best_ever if best_ever["w"] is not None else best_random

        print("\n" + "=" * 60, flush=True)
        print("=== 搜索完毕 ===", flush=True)
        print("  累计真实评估: %d / %d（其中上一轮 %d，本轮 %d）"
              % (total_used, BUDGET, SPENT_BEFORE, total_used - SPENT_BEFORE), flush=True)
        print("  阶段1撒网最优 R=%.6f" % best_random["R"], flush=True)
        if best_ever["w"] is not None:
            print("  阶段2精修最优 R=%.6f（%s）" % (best_ever["R"], best_ever["label"]), flush=True)
        print("  全程最优 R=%.6f" % final_best["R"], flush=True)
        if final_best["w"] is not None:
            W = np.asarray(final_best["w"]).reshape(N_B, N_F)
            print("  权重矩阵:", flush=True)
            for i, r in enumerate(W):
                print("    楼%d: %s  dom=F%d" % (i + 1, np.round(r, 4).tolist(), np.argmax(r) + 1), flush=True)

        result = {
            "params": {"SEED": SEED, "RANDOM_SAMPLE": RANDOM_SAMPLE, "BUDGET": BUDGET,
                        "SPENT_BEFORE": SPENT_BEFORE,
                        "REFINE_BUDGET": REFINE_BUDGET, "TOP_N_REFINE": TOP_N_REFINE,
                        "STEP": STEP, "TOP_K_FEAT": TOP_K_FEAT, "EPS": EPS},
            "phase1_best_R": float(best_random["R"]),
            "phase2_best_R": float(best_ever["R"]) if best_ever["w"] is not None else None,
            "overall_best_R": float(final_best["R"]),
            "overall_best_weights": final_best["w"].tolist() if final_best["w"] is not None else None,
            "overall_best_label": final_best.get("label", ""),
            "total_evals": total_used,
            "refine_evals": refine_used,
            "elapsed_seconds": time.time() - t_total,
            "run_dir": RUN_DIR, "eval_log": LOG_PATH,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("结果已落盘: %s" % RESULT_PATH, flush=True)

    finally:
        ev.close()


if __name__ == "__main__":
    main()
