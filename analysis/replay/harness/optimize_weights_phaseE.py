# -*- coding: utf-8 -*-
"""
阶段 E：地形测绘（允许真实 0 权重，粗粒度大步长，不做精扫）。

要点：
  * 每栋楼 7 个特征、非负、行和=1；允许精确 0.0（不是 EPS）。
  * E0  链路验证：均匀点对拍 + 纯特征点 + 含零格点，确认求解器接受 0.0。
  * E1  粗网格切片：步长 0.2 的非负单纯形格点（462 点/楼），单栋变化，
        其余两栋固定为均匀；三栋各扫一遍。
  * E2  稀疏随机撒点（Dirichlet alpha<1 / 随机置零）覆盖 1–7 非零谱；
        以 v5 最优为锚的 F5/F6 切片；每栋 top-4 结构的楼间笛卡尔抽样。
  * E3  仅对当前最优做一层 0.05 局部邻域（含 0↔正 的进出零面移动）。

所有数值均来自真实求解器调用，逐批追加到 optimization/eval_log.jsonl。
单进程串行；每 20 点一批，每批前后检查 C 盘；低于 1.5 GB 停批。
用法：python optimize_weights_phaseE.py E0|E1|E2|E3 [本次最多新增评估数]
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

# ============================ 集中配置 ============================
SEED = 20261107
N_B, N_F = 3, 7
STEP_COARSE = 0.2                 # E1 粗格点步长
STEP_FINE = 0.05                  # E3 局部步长
P_THIRD = [0.05, 0.10, 0.20]      # E2 第三特征占比
BATCH = 20
BUDGET = 1800                     # 本轮（vE）真实评估总预算
WORKERS = 4
NPROC = 1                         # 实测单进程最快
DISK_MIN_GB = 1.5
EPS = 0.00010000960092168848      # 仅用于与历史日志对照，不写入新点

ROOT = os.environ.get("SDM_WORKSPACE", os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "optimization", "eval_log.jsonl")
RES_DIR = os.path.join(ROOT, "optimization")
V5_PATH = os.path.join(ROOT, "best_weights_v5.json")
E1_PATH = os.path.join(RES_DIR, "vE1.json")
E2_PATH = os.path.join(RES_DIR, "vE2.json")
E3_PATH = os.path.join(RES_DIR, "vE3.json")
BEST_PATH = os.path.join(ROOT, "best_weights_phaseE.json")
BASE_EXE = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "RampingSolver.exe")
SOLVER_EXE = os.path.join(ROOT, "solver", "solver.exe")
RUN_LIMIT = 0

rng = np.random.default_rng(SEED)


# ============================ 基础设施 ============================
def disk_free_gb():
    return shutil.disk_usage("C:\\").free / (1024 ** 3)


def purge_mei():
    tmp = os.environ.get("TEMP", r"C:\Windows\Temp")
    n = 0
    for d in glob.glob(os.path.join(tmp, "_MEI*")):
        try:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
        except Exception:
            pass
    return n


def disk_guard(stage, ctx=None):
    free = disk_free_gb()
    print("  [磁盘] %s：C 盘可用 %.2f GB" % (stage, free), flush=True)
    if free < DISK_MIN_GB:
        if ctx is not None:
            try:
                ctx["ev"].close()
            except Exception:
                pass
        n = purge_mei()
        raise SystemExit("C 盘可用 %.2f GB < %.1f GB：已停止新批次，清理 _MEI %d 个"
                         % (free, DISK_MIN_GB, n))


def ensure_base_solver():
    if os.path.exists(BASE_EXE) and os.path.getsize(BASE_EXE) > 1e8:
        return BASE_EXE
    ps = ("$a=[System.Reflection.Assembly]::LoadFile('{src}');"
          "$s=$a.GetManifestResourceStream('RampingSolver.exe');"
          "$fs=[System.IO.File]::Create('{dst}');"
          "$s.CopyTo($fs);$fs.Close();$s.Close()").format(src=SOLVER_EXE, dst=BASE_EXE)
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return BASE_EXE


class Serve:
    def __init__(self):
        self.p = None

    def _start(self):
        exe = ensure_base_solver()
        self.p = subprocess.Popen([exe, "--serve", "--workers", str(WORKERS)],
                                  cwd=os.path.dirname(exe),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                  bufsize=1)

    def evaluate(self, Wlist):
        out = []
        for s in range(0, len(Wlist), BATCH):
            batch = Wlist[s:s + BATCH]
            for attempt in range(2):
                try:
                    if self.p is None or self.p.poll() is not None:
                        self._start()
                    self.p.stdin.write("".join(
                        json.dumps(np.asarray(w, float).ravel().tolist()) + "\n" for w in batch))
                    self.p.stdin.flush()
                    res = []
                    for _ in batch:
                        line = self.p.stdout.readline()
                        if not line:
                            raise RuntimeError("serve 退出: " + self.p.stderr.read()[:300])
                        obj = json.loads(line)
                        if "R" not in obj or obj["R"] is None:
                            raise RuntimeError("响应异常: " + line[:200])
                        res.append(float(obj["R"]))
                    out.extend(res)
                    break
                except Exception as e:
                    print("  [重试] %s" % e, flush=True)
                    try:
                        self.p.kill()
                    except Exception:
                        pass
                    self.p = None
                    if attempt == 1:
                        raise
        return out

    def close(self):
        if self.p is not None:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=15)
            except Exception:
                self.p.kill()


def key_of(w):
    return tuple(np.round(np.asarray(w, float).ravel(), 12))


def load_cache():
    cache, last = {}, 0
    if os.path.exists(LOG_PATH):
        for line in open(LOG_PATH, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            cache[key_of(r["weights"])] = (r["R"], r["eval_id"])
            last = max(last, int(r["eval_id"]))
    return cache, last


def append_log(recs):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_json_best(ctx):
    b = dict(ctx["best"])
    if isinstance(b.get("w"), np.ndarray):
        b["w"] = b["w"].tolist()
    return b


def make_ctx():
    cache, last = load_cache()
    used = sum(1 for line in open(LOG_PATH, encoding="utf-8")
               if '"vE' in line) if os.path.exists(LOG_PATH) else 0
    return {"cache": cache, "last_id": last, "used": used, "new": 0, "run_new": 0,
            "ev": Serve(), "best": {"R": float("inf"), "w": None, "eval_id": None, "label": ""}}


def eval_candidates(ctx, cands, labels, phase):
    """未评估点按 BATCH 分批真实评估并逐批刷盘；已评估点直接复用日志。"""
    keys, todo, tl = [], [], []
    for w, lb in zip(cands, labels):
        k = key_of(w)
        if k in ctx["cache"] or k in keys:
            continue
        keys.append(k)
        todo.append(np.asarray(w, float))
        tl.append(lb)
    room = BUDGET - ctx["used"]
    if RUN_LIMIT > 0:
        room = min(room, RUN_LIMIT - ctx["run_new"])
    skipped = 0
    n_new = 0
    if todo and room > 0:
        if len(todo) > room:
            skipped = len(todo) - room
            todo, tl = todo[:room], tl[:room]
        n_batches = (len(todo) + BATCH - 1) // BATCH
        for bi in range(n_batches):
            chunk = todo[bi * BATCH:(bi + 1) * BATCH]
            cl = tl[bi * BATCH:(bi + 1) * BATCH]
            disk_guard("第 %d/%d 批开始" % (bi + 1, n_batches), ctx)
            Rs = ctx["ev"].evaluate(chunk)
            eid = ctx["last_id"]
            recs = []
            for w, lb, r in zip(chunk, cl, Rs):
                eid += 1
                ctx["cache"][key_of(w)] = (r, eid)
                recs.append({
                    "eval_id": eid, "phase": phase, "weights": w.tolist(), "R": r,
                    "label": lb,
                    "nz": int((w > 0).sum()),
                    "nz_eff": int((w >= 1e-3).sum()),
                    "nz_rows": [int((w.reshape(N_B, N_F)[i] >= 1e-3).sum()) for i in range(N_B)],
                    "nz_rows_exact": [int((w.reshape(N_B, N_F)[i] > 0).sum()) for i in range(N_B)],
                    "timestamp": datetime.now().isoformat(timespec="seconds")})
                if r < ctx["best"]["R"]:
                    ctx["best"].update({"R": r, "w": w.copy(), "eval_id": eid, "label": lb})
            append_log(recs)
            ctx["last_id"] = eid
            ctx["used"] += len(chunk)
            ctx["new"] += len(chunk)
            ctx["run_new"] += len(chunk)
            n_new += len(chunk)
            disk_guard("第 %d/%d 批结束" % (bi + 1, n_batches), ctx)
    elif todo:
        skipped = len(todo)
    out = []
    for w, lb in zip(cands, labels):
        k = key_of(w)
        if k in ctx["cache"]:
            out.append((lb, ctx["cache"][k][0], k in keys))
        else:
            out.append((lb, float("inf"), False))
    for w, (lb, r, _n) in zip(cands, out):
        if r < ctx["best"]["R"]:
            k = key_of(w)
            ctx["best"].update({"R": r, "w": np.asarray(w, float).copy(),
                                "eval_id": ctx["cache"][k][1], "label": lb})
    print("  [%s] 候选 %d，本次新增 %d（vE 预算 %d/%d，本次上限 %s）%s"
          % (phase, len(cands), n_new, ctx["used"], BUDGET,
             RUN_LIMIT if RUN_LIMIT > 0 else "不限",
             "，跳过 %d（额度不足）" % skipped if skipped else ""), flush=True)
    return out


# ============================ 权重构造 ============================
def build_rows(rows):
    W = np.vstack([np.asarray(r, float) for r in rows])
    W = W / W.sum(axis=1, keepdims=True)
    return W.ravel()


def row_lattice(step):
    """非负、行和=1 的全部格点（含精确 0），返回列表。"""
    n = int(round(1.0 / step))
    out = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            out.append(prefix + [remaining])
            return
        for x in range(remaining + 1):
            rec(prefix + [x], remaining - x, slots - 1)

    rec([], n, N_F)
    rows = []
    for c in out:
        r = np.asarray(c, float) / float(n)
        r = r / r.sum()
        rows.append(r)
    return rows


def uniform_rows():
    return [np.full(N_F, 1.0 / N_F) for _ in range(N_B)]


def load_v5_rows():
    d = json.load(open(V5_PATH, encoding="utf-8"))
    rows = np.asarray(d["weights_by_building"], float)
    return [r / r.sum() for r in rows]


def row_pair(fi, fj, s):
    """fi 占 s，fj 占 1-s，其余精确 0。"""
    r = np.zeros(N_F)
    r[fi] = float(s)
    r[fj] = 1.0 - float(s)
    return r / r.sum()


def sparse_row(k, alpha=None):
    feats = rng.choice(N_F, size=int(k), replace=False)
    if alpha is None:
        alpha = float(rng.choice([0.3, 0.5, 0.8]))
    x = rng.dirichlet(np.full(int(k), alpha))
    r = np.zeros(N_F)
    r[feats] = x
    return r


def struct_sig(row, thr=0.01):
    return tuple(sorted(int(f) for f in np.where(np.asarray(row) >= thr)[0]))


# ============================ E0 链路验证 ============================
def phase_e0(ctx):
    cands, labels = [], []
    cands.append(build_rows(uniform_rows()))
    labels.append("E0 均匀")

    for b in range(N_B):
        for f in (4, 5):
            rows = uniform_rows()
            r = np.zeros(N_F)
            r[f] = 1.0
            rows[b] = r
            cands.append(build_rows(rows))
            labels.append("E0 楼%d 纯F%d" % (b + 1, f + 1))

    rows = [np.zeros(N_F) for _ in range(N_B)]
    for r in rows:
        r[5] = 1.0
    cands.append(build_rows(rows))
    labels.append("E0 三栋纯F6")

    samples = [(0, [0.2] * 5 + [0, 0]),
               (0, [0, 0, 0, 0, 0, 0.5, 0.5]),
               (1, [0.2, 0, 0, 0, 0.2, 0.6, 0]),
               (2, [0, 0, 0.4, 0, 0, 0.6, 0])]
    for b, raw in samples:
        rows = uniform_rows()
        rows[b] = np.asarray(raw, float) / float(np.sum(raw))
        cands.append(build_rows(rows))
        labels.append("E0 含零 楼%d %s" % (b + 1, raw))

    out = eval_candidates(ctx, cands, labels, "vE0")
    rs = [r for _, r, _ in out]
    print("\n=== E0 链路验证 ===", flush=True)
    for (lb, r, isnew), w in zip(out, cands):
        print("  %-28s R=%.9f  %s" % (lb, r, "新增" if isnew else "缓存"), flush=True)
    if not all(np.isfinite(rs)) and len(rs):
        raise SystemExit("存在非有限 R，链路异常，停止。")
    uni = rs[0]
    ok = abs(uni - 156.24560858344194) < 1e-6
    print("  均匀点对拍：%.9f  %s" % (uni, "一致" if ok else "不一致（注意！）"), flush=True)
    print("  含零点最小 R=%.6f（可接受 0.0 输入）" % min(rs[1:]) if len(rs) > 1 else "", flush=True)


# ============================ E1 粗网格切片 ============================
def phase_e1(ctx):
    lattice = row_lattice(STEP_COARSE)
    print("E1 格点数/楼 = %d（步长 %.2f，含零）" % (len(lattice), STEP_COARSE), flush=True)
    all_res = []
    for b in range(N_B):
        cands, labels, meta = [], [], []
        for row in lattice:
            rows = uniform_rows()
            rows[b] = row
            cands.append(build_rows(rows))
            labels.append("E1 楼%d 格点 %s" % (b + 1, np.round(row, 3).tolist()))
            meta.append(row)
        out = eval_candidates(ctx, cands, labels, "vE1_b%d" % (b + 1))
        for (lb, r, isnew), row in zip(out, meta):
            all_res.append({"building": b + 1, "row": [float(x) for x in row],
                            "R": r, "label": lb, "new": bool(isnew),
                            "nz": int((row > 0).sum())})
        vals = [rec["R"] for rec in all_res if rec["building"] == b + 1]
        if vals:
            print("  楼%d 切片：最优 %.6f（均匀锚点）" % (b + 1, min(vals)), flush=True)

    all_res.sort(key=lambda x: x["R"])
    with open(E1_PATH, "w", encoding="utf-8") as f:
        json.dump({"all": all_res, "top20": all_res[:20],
                   "best": to_json_best(ctx)}, f, ensure_ascii=False, indent=2)
    print("\n=== 里程碑 E1：粗格点切片最优 top10 ===", flush=True)
    for n, rec in enumerate(all_res[:10], 1):
        print("  %2d. 楼%d %s R=%.6f" % (n, rec["building"],
                                         np.round(rec["row"], 3).tolist(), rec["R"]), flush=True)
    print("  全局最优 R=%.9f (id=%s) %s" % (ctx["best"]["R"], ctx["best"]["eval_id"],
                                          ctx["best"]["label"]), flush=True)


# ============================ E2 稀疏随机 / 锚点切片 / 楼间组合 ============================
def per_building_top(d1, topn=4):
    out = {}
    for b in range(1, N_B + 1):
        rows = [r for r in d1["all"] if r["building"] == b]
        rows.sort(key=lambda x: x["R"])
        picked, seen = [], set()
        for rec in rows:
            sig = tuple(np.round(rec["row"], 3).tolist())
            if sig in seen:
                continue
            seen.add(sig)
            picked.append(rec)
            if len(picked) == topn:
                break
        out[b] = picked
    return out


def phase_e2(ctx):
    d1 = json.load(open(E1_PATH, encoding="utf-8"))
    per = per_building_top(d1, 4)
    cands, labels = [], []

    # (1) 稀疏随机：三栋各自稀疏
    for n in range(70):
        ks = rng.integers(1, N_F + 1, size=N_B)
        rows = [sparse_row(k) for k in ks]
        cands.append(build_rows(rows))
        labels.append("E2 稀疏随机 三栋 k=%s" % ks.tolist())
    # (2) 稀疏随机：单栋稀疏，其余均匀
    for n in range(60):
        b = int(rng.integers(0, N_B))
        k = int(rng.integers(1, N_F + 1))
        rows = uniform_rows()
        rows[b] = sparse_row(k)
        cands.append(build_rows(rows))
        labels.append("E2 稀疏随机 楼%d k=%d（其余均匀）" % (b + 1, k))

    # (3) v5 锚点：F5/F6 纯对切片 + 第三特征进入
    v5 = load_v5_rows()
    for b in range(N_B):
        for s in np.arange(0, 1.0001, STEP_FINE):
            rows = [r.copy() for r in v5]
            rows[b] = row_pair(4, 5, float(s))
            cands.append(build_rows(rows))
            labels.append("E2 v5锚 楼%d F5/F6 s=%.2f" % (b + 1, s))
    thirds = [1, 6, 1]   # 楼1/2/3 的第三特征：F2, F7, F2
    for b in range(N_B):
        base = v5[b]
        ratio = base[4] / (base[4] + base[5]) if base[4] + base[5] > 0 else 0.5
        for t in {thirds[b], 6}:
            if t in (4, 5):
                continue
            for u in P_THIRD:
                raw = np.zeros(N_F)
                raw[4] = ratio * (1.0 - u)
                raw[5] = (1.0 - ratio) * (1.0 - u)
                raw[t] = u
                rows = [r.copy() for r in v5]
                rows[b] = raw
                cands.append(build_rows(rows))
                labels.append("E2 v5锚 楼%d F5/F6+F%d u=%.2f" % (b + 1, t + 1, u))

    # (4) 楼间组合：每栋 top-4 结构的笛卡尔抽样
    for ra in per[1]:
        for rb in per[2]:
            for rc in per[3]:
                rows = [np.asarray(ra["row"]), np.asarray(rb["row"]), np.asarray(rc["row"])]
                cands.append(build_rows(rows))
                labels.append("E2 组合 楼1%s 楼2%s 楼3%s"
                              % (np.round(ra["row"], 2).tolist(),
                                 np.round(rb["row"], 2).tolist(),
                                 np.round(rc["row"], 2).tolist()))

    out = eval_candidates(ctx, cands, labels, "vE2")
    res = [{"R": r, "label": lb, "new": bool(isnew)} for (lb, r, isnew) in out]
    res.sort(key=lambda x: x["R"])
    with open(E2_PATH, "w", encoding="utf-8") as f:
        json.dump({"all": res, "top20": res[:20], "best": to_json_best(ctx),
                   "per_building_top": {str(k): v for k, v in per.items()}},
                  f, ensure_ascii=False, indent=2)
    print("\n=== 里程碑 E2：稀疏/组合最优 top10 ===", flush=True)
    for n, rec in enumerate(res[:10], 1):
        print("  %2d. R=%.6f  %s" % (n, rec["R"], rec["label"]), flush=True)
    print("  全局最优 R=%.9f (id=%s) %s" % (ctx["best"]["R"], ctx["best"]["eval_id"],
                                          ctx["best"]["label"]), flush=True)


# ============================ E3 局部 0.05 邻域 ============================
def local_moves(row):
    row = np.asarray(row, float)
    out = []
    pos = [i for i in range(N_F) if row[i] > 0]
    zero = [i for i in range(N_F) if row[i] == 0]
    if not pos:
        return out
    top2 = sorted(pos, key=lambda i: -row[i])[:2]
    for i in top2:
        if row[i] < STEP_FINE - 1e-12:
            continue
        for j in range(N_F):
            if j == i:
                continue
            r = row.copy()
            r[i] -= STEP_FINE
            r[j] += STEP_FINE
            out.append(r / r.sum())
    # 最小正分量直接归零（进入零面）
    s = min(pos, key=lambda i: row[i])
    l = max(pos, key=lambda i: row[i])
    if s != l:
        r = row.copy()
        r[l] += r[s]
        r[s] = 0.0
        out.append(r / r.sum())
    # 从最大分量挪出，进入零特征
    for j in sorted(zero)[:2]:
        for amt in (STEP_FINE, 2 * STEP_FINE):
            if row[l] >= amt:
                r = row.copy()
                r[l] -= amt
                r[j] += amt
                out.append(r / r.sum())
    return out


def phase_e3(ctx):
    if ctx["best"]["w"] is None:
        raise SystemExit("E3 需要已有最优解")
    rows = [np.asarray(r, float).copy() for r in np.asarray(ctx["best"]["w"]).reshape(N_B, N_F)]
    hist = [{"round": 0, "R": ctx["best"]["R"], "label": "起点"}]
    curR = ctx["best"]["R"]
    for rd in range(1, 3):
        cands, labels = [], []
        if rd == 1:
            for b in range(N_B):
                for mv in local_moves(rows[b]):
                    rr = [x.copy() for x in rows]
                    rr[b] = mv
                    cands.append(build_rows(rr))
                    labels.append("E3 楼%d 局部0.05 %s" % (b + 1, np.round(mv, 3).tolist()))
        else:
            for b in range(N_B):
                for mv in local_moves(rows[b]):
                    rr = [x.copy() for x in rows]
                    rr[b] = mv
                    cands.append(build_rows(rr))
                    labels.append("E3 楼%d 局部0.05 %s" % (b + 1, np.round(mv, 3).tolist()))
        out = eval_candidates(ctx, cands, labels, "vE3_r%d" % rd)
        Rs = np.asarray([r for _, r, _ in out])
        k = int(np.argmin(Rs)) if len(Rs) else 0
        if len(Rs) and Rs[k] < curR - 1e-12:
            rows = [np.asarray(x, float) for x in
                    np.asarray(cands[k]).reshape(N_B, N_F)]
            curR = float(Rs[k])
        hist.append({"round": rd, "R": curR, "n_cand": len(Rs),
                     "best_cand": float(Rs[k]) if len(Rs) else None})
        print("  E3 第%d轮 → R=%.9f" % (rd, curR), flush=True)

    # 最终重复评估 2 次确认
    bw = build_rows(rows)
    reps = []
    for i in range(2):
        if ctx["used"] >= BUDGET:
            break
        disk_guard("E3 复评开始", ctx)
        reps.append(ctx["ev"].evaluate([bw])[0])
        eid = ctx["last_id"] + 1
        ctx["last_id"] = eid
        ctx["used"] += 1
        ctx["new"] += 1
        append_log([{"eval_id": eid, "phase": "vE3_confirm", "weights": bw.tolist(),
                     "R": reps[-1], "label": "E3 最终重复 %d/2" % (i + 1),
                     "nz": int((bw > 0).sum()),
                     "timestamp": datetime.now().isoformat(timespec="seconds")}])
        disk_guard("E3 复评结束", ctx)
    if not reps:
        reps = [ctx["cache"][key_of(bw)][0]] if key_of(bw) in ctx["cache"] else [float("nan")]

    W = np.asarray(bw).reshape(N_B, N_F)
    out = {"best_R": float(min(reps)), "repeat_R": [float(x) for x in reps],
           "stable": bool(max(reps) - min(reps) < 1e-9),
           "eval_id": int(ctx["last_id"]),
           "weights_flat": [float(x) for x in W.ravel()],
           "weights_by_building": [[float(x) for x in r] for r in W],
           "row_sums": [float(x) for x in W.sum(axis=1)],
           "min_weight": float(W.min()),
           "nz_per_building": [int((r > 0).sum()) for r in W],
           "hist": hist, "best_from_scan": to_json_best(ctx),
           "total_evals_vE": int(ctx["used"]), "new_evals_vE": int(ctx["new"]),
           "log_path": LOG_PATH}
    with open(BEST_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(E3_PATH, "w", encoding="utf-8") as f:
        json.dump({"hist": hist, "repeat_R": reps, "best": to_json_best(ctx),
                   "weights_by_building": out["weights_by_building"]},
                  f, ensure_ascii=False, indent=2)

    print("\n=== 里程碑 E3：局部最优点 ===", flush=True)
    print("  R=%.9f，重复 %s" % (min(reps), ["%.9f" % x for x in reps]), flush=True)
    for i, row in enumerate(W):
        print("  楼%d: %s 行和=%.9f 非零=%d"
              % (i + 1, np.round(row, 8).tolist(), row.sum(), (row > 0).sum()), flush=True)


# ============================ 主入口 ============================
def main():
    global RUN_LIMIT
    ph = (sys.argv[1] if len(sys.argv) > 1 else "all").upper()
    if len(sys.argv) > 2:
        RUN_LIMIT = int(sys.argv[2])
    ctx = make_ctx()
    purge_mei()
    disk_guard("启动前", ctx)
    print("=== 阶段E %s 开始（vE 预算 %d，已用 %d，本次上限 %s，seed=%d）==="
          % (ph, BUDGET, ctx["used"], RUN_LIMIT if RUN_LIMIT > 0 else "不限", SEED), flush=True)
    try:
        if ph in ("E0", "ALL"):
            phase_e0(ctx)
        if ph in ("E1", "ALL"):
            phase_e1(ctx)
        if ph in ("E2", "ALL"):
            phase_e2(ctx)
        if ph in ("E3", "ALL"):
            phase_e3(ctx)
    finally:
        ctx["ev"].close()
        print("=== 阶段E %s 结束：本次新增 %d，累计 vE %d ===" % (ph, ctx["new"], ctx["used"]), flush=True)


if __name__ == "__main__":
    main()
