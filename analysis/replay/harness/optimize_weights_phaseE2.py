# -*- coding: utf-8 -*-
"""
阶段 E2：三栋联合稀疏（精确 0）+ 代理模型加速。

与 E1 的区别：不再用“单栋变、其余均匀”，而是三栋同时稀疏、同时含精确 0。
流程：
  A0  以 R=105.940297 的锚结构为基准，第三特征由 0.05/0.10/0.01 逐步改为精确 0；
      含单栋轴扫 + 三栋随机联合。
  A1  三栋同时只保留 2 个非零特征的精确 0 版本：F5/F6 占比三栋联合 + 其它特征对抽样。
  A2  三栋联合 3 非零特征（锚特征组）占比枚举 + 换第三特征抽样。
  B   代理模型（随机森林 + LCB 采集）：用全部真实记录拟合，在 2–3 非零精确 0
      候选池中选点，逐轮真实评估并重拟合，直到预算用完。

每批 20 点落盘 optimization/eval_log.jsonl；每批结束写 optimization/E2_progress.json。
单进程串行；每批前后查 C 盘，低于 1.5 GB 停批。支持中断后按缓存续跑。
用法：python optimize_weights_phaseE2.py all|A0|A1|A2|B [本次最多新增评估数]
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = os.environ.get("SDM_WORKSPACE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optimize_weights_phaseE import (Serve, key_of, append_log, disk_guard,
                                     purge_mei, ensure_base_solver)

# ============================ 配置 ============================
SEED = 20261108
BUDGET = 800
BATCH = 20
N_B, N_F = 3, 7
DISK_MIN_GB = 1.5
NZ_THR = 1e-3
LOG_PATH = os.path.join(ROOT, "optimization", "eval_log.jsonl")
PROGRESS_PATH = os.path.join(ROOT, "optimization", "E2_progress.json")
BEST_PATH = os.path.join(ROOT, "best_weights_phaseE2.json")
V5_PATH = os.path.join(ROOT, "best_weights_v5.json")
RUN_LIMIT = 0

rng = np.random.default_rng(SEED)
U_THIRD = [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
RATIO_OFF = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
PAIRS = [(4, 5), (4, 1), (4, 2), (4, 6), (5, 1), (5, 2), (5, 6), (1, 6)]


# ============================ 基础 ============================
def build_rows(rows):
    W = np.vstack([np.asarray(r, float) for r in rows])
    W = W / W.sum(axis=1, keepdims=True)
    return W.ravel()


def clean_row(row):
    r = np.asarray(row, float).copy()
    r[r < NZ_THR] = 0.0
    s = r.sum()
    return r / s if s > 0 else r


def anchor_rows():
    d = json.load(open(V5_PATH, encoding="utf-8"))
    return [clean_row(r) for r in np.asarray(d["weights_by_building"], float)]


def row_2(fi, fj, ratio):
    r = np.zeros(N_F)
    r[fi] = ratio
    r[fj] = 1.0 - ratio
    return r


def row_3(fi, fj, ratio, t, u):
    r = np.zeros(N_F)
    r[fi] = ratio * (1.0 - u)
    r[fj] = (1.0 - ratio) * (1.0 - u)
    r[t] = u
    return r


def third_of(row):
    pos = [i for i in range(N_F) if row[i] > 0 and i not in (4, 5)]
    return pos[0] if pos else None


def pair_ratio(row):
    s = row[4] + row[5]
    return float(row[4] / s) if s > 0 else 0.5


def struct_sig(w):
    W = np.asarray(w, float).reshape(N_B, N_F)
    return tuple(tuple(int(x) for x in np.where(W[b] >= NZ_THR)[0]) for b in range(N_B))


def load_cache():
    cache, last, recs = {}, 0, []
    for line in open(LOG_PATH, encoding="utf-8"):
        try:
            r = json.loads(line)
            w = np.asarray(r["weights"], float)
            cache[key_of(w)] = (float(r["R"]), int(r["eval_id"]))
            last = max(last, int(r["eval_id"]))
            recs.append({"eval_id": int(r["eval_id"]), "R": float(r["R"]), "w": w,
                         "phase": str(r.get("phase", "")), "label": str(r.get("label", ""))})
        except Exception:
            continue
    return cache, last, recs


def write_progress(ctx, stage, note=""):
    b = ctx["best"]
    out = {"updated": datetime.now().isoformat(timespec="seconds"),
           "stage": stage, "note": note,
           "used_e2": int(ctx["used"]), "budget_e2": BUDGET,
           "new_this_run": int(ctx["new"]),
           "log_eval_id_last": int(ctx["last_id"]),
           "log_lines": int(ctx["last_id"] + 1),
           "best_R": float(b["R"]), "best_eval_id": b["eval_id"], "best_label": b["label"],
           "best_weights": (b["w"].tolist() if isinstance(b["w"], np.ndarray) else b["w"]),
           "best_nz_exact": (int((np.asarray(b["w"]) > 0).sum())
                             if b["w"] is not None else None),
           "best_nz_eff": (int((np.asarray(b["w"]) >= NZ_THR).sum())
                           if b["w"] is not None else None),
           "stage_counts": ctx["stage_counts"]}
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PROGRESS_PATH)


def make_ctx():
    cache, last, recs = load_cache()
    used = sum(1 for r in recs if r["phase"].startswith("vE2"))
    gb = min(recs, key=lambda x: x["R"]) if recs else None
    return {"cache": cache, "last_id": last, "used": used, "new": 0, "run_new": 0,
            "recs": recs, "ev": Serve(),
            "best": {"R": gb["R"] if gb else float("inf"),
                     "w": gb["w"].copy() if gb else None,
                     "eval_id": gb["eval_id"] if gb else None,
                     "label": gb["label"] if gb else ""},
            "stage_counts": {}}


def eval_candidates(ctx, cands, labels, phase, note=""):
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
        nb = (len(todo) + BATCH - 1) // BATCH
        for bi in range(nb):
            chunk = todo[bi * BATCH:(bi + 1) * BATCH]
            cl = tl[bi * BATCH:(bi + 1) * BATCH]
            disk_guard("第 %d/%d 批开始" % (bi + 1, nb), ctx)
            Rs = ctx["ev"].evaluate(chunk)
            eid = ctx["last_id"]
            recs_out = []
            for w, lb, r in zip(chunk, cl, Rs):
                eid += 1
                ctx["cache"][key_of(w)] = (r, eid)
                ctx["recs"].append({"eval_id": eid, "R": r, "w": w.copy(),
                                    "phase": phase, "label": lb})
                recs_out.append({
                    "eval_id": eid, "phase": phase, "weights": w.tolist(), "R": r,
                    "label": lb, "nz": int((w > 0).sum()),
                    "nz_eff": int((w >= NZ_THR).sum()),
                    "nz_rows": [int((w.reshape(N_B, N_F)[i] >= NZ_THR).sum()) for i in range(N_B)],
                    "nz_rows_exact": [int((w.reshape(N_B, N_F)[i] > 0).sum()) for i in range(N_B)],
                    "timestamp": datetime.now().isoformat(timespec="seconds")})
                if r < ctx["best"]["R"]:
                    ctx["best"].update({"R": r, "w": w.copy(), "eval_id": eid, "label": lb})
            append_log(recs_out)
            ctx["last_id"] = eid
            ctx["used"] += len(chunk)
            ctx["new"] += len(chunk)
            ctx["run_new"] += len(chunk)
            n_new += len(chunk)
            ctx["stage_counts"][phase] = ctx["stage_counts"].get(phase, 0) + len(chunk)
            write_progress(ctx, phase, note)
            disk_guard("第 %d/%d 批结束" % (bi + 1, nb), ctx)
    elif todo:
        skipped = len(todo)
    out = []
    for w, lb in zip(cands, labels):
        k = key_of(w)
        if k in ctx["cache"]:
            out.append((lb, ctx["cache"][k][0], k in keys))
        else:
            out.append((lb, float("inf"), False))
    print("  [%s] 候选 %d，新增 %d（E2 预算 %d/%d）%s"
          % (phase, len(cands), n_new, ctx["used"], BUDGET,
             "，跳过 %d（额度不足）" % skipped if skipped else ""), flush=True)
    return out


# ============================ 阶段 A0：第三特征进出零面 ============================
def phase_a0(ctx):
    a = anchor_rows()
    ths = [third_of(a[b]) for b in range(N_B)]
    cands, labels = [], []

    cands.append(build_rows(a))
    labels.append("A0 锚点 eps->精确0")
    rows0 = [row_2(4, 5, pair_ratio(a[b])) for b in range(N_B)]
    cands.append(build_rows(rows0))
    labels.append("A0 三栋第三特征全归零")

    for b in range(N_B):
        for u in U_THIRD:
            rows = [r.copy() for r in a]
            rows[b] = row_3(4, 5, pair_ratio(a[b]), ths[b], u)
            cands.append(build_rows(rows))
            labels.append("A0 楼%d 第三F%d u=%.3f" % (b + 1, ths[b] + 1, u))

    for _ in range(60):
        rows = [row_3(4, 5, pair_ratio(a[b]), ths[b], float(rng.choice(U_THIRD)))
                for b in range(N_B)]
        cands.append(build_rows(rows))
        labels.append("A0 联合 u=%s" % [round(r[ths[b]], 3) for b, r in enumerate(rows)])

    out = eval_candidates(ctx, cands, labels, "vE2_A0", "第三特征进出零面")
    res = [{"R": r, "label": lb} for lb, r, _ in out]
    res.sort(key=lambda x: x["R"])
    print("  A0 最优：%.6f  %s" % (res[0]["R"], res[0]["label"]) if res else "  A0 无结果",
          flush=True)


# ============================ A1：三栋同时 2 个非零 ============================
def phase_a1(ctx):
    a = anchor_rows()
    r0 = [pair_ratio(a[b]) for b in range(N_B)]
    cands, labels = [], []

    # 参考：v2 的 107.673 三栋同构 F5/F6 占比（精确 0 版本）
    r_ref = 0.22989 / (0.22989 + 0.76962)
    cands.append(build_rows([row_2(4, 5, r_ref) for _ in range(N_B)]))
    labels.append("A1 三栋同构 F5/F6 r=%.3f（精确0）" % r_ref)

    # 轴扫：单栋占比偏移
    for b in range(N_B):
        for off in RATIO_OFF:
            rr = [x for x in r0]
            rr[b] = float(np.clip(r0[b] + off, 0.02, 0.98))
            cands.append(build_rows([row_2(4, 5, rr[k]) for k in range(N_B)]))
            labels.append("A1 三栋F5/F6 楼%d r=%.3f" % (b + 1, rr[b]))

    # 三栋联合随机占比
    for _ in range(60):
        rr = [float(np.clip(r0[b] + rng.normal(0, 0.06), 0.03, 0.97)) for b in range(N_B)]
        cands.append(build_rows([row_2(4, 5, rr[b]) for b in range(N_B)]))
        labels.append("A1 三栋F5/F6 联合 r=%s" % [round(x, 3) for x in rr])

    # 其它特征对的联合抽样
    for _ in range(60):
        rows = []
        for b in range(N_B):
            fi, fj = PAIRS[int(rng.integers(len(PAIRS)))]
            rows.append(row_2(fi, fj, float(rng.uniform(0.05, 0.95))))
        cands.append(build_rows(rows))
        labels.append("A1 其它对联合")

    # 全 F5/F6、finer 0.05 占比的随机组合（补充低洼带）
    for _ in range(40):
        rr = [float(np.clip(r_ref + rng.normal(0, 0.05), 0.03, 0.97)) for _ in range(N_B)]
        cands.append(build_rows([row_2(4, 5, x) for x in rr]))
        labels.append("A1 F5/F6 低洼带 r=%s" % [round(x, 3) for x in rr])

    out = eval_candidates(ctx, cands, labels, "vE2_A1", "三栋同时2非零")
    res = [{"R": r, "label": lb} for lb, r, _ in out]
    res.sort(key=lambda x: x["R"])
    print("  A1 最优：%.6f  %s" % (res[0]["R"], res[0]["label"]) if res else "  A1 无结果",
          flush=True)


# ============================ A2：三栋联合 3 非零 ============================
def phase_a2(ctx):
    a = anchor_rows()
    ths = [third_of(a[b]) for b in range(N_B)]
    cands, labels = [], []

    for _ in range(100):
        rows = []
        for b in range(N_B):
            ratio = float(np.clip(pair_ratio(a[b]) + rng.normal(0, 0.03), 0.03, 0.97))
            rows.append(row_3(4, 5, ratio, ths[b], float(rng.choice(U_THIRD))))
        cands.append(build_rows(rows))
        labels.append("A2 锚特征组联合 u=%s" % [round(r[ths[b]], 3)
                                               for b, r in enumerate(rows)])

    for _ in range(50):
        rows = []
        for b in range(N_B):
            t = int(rng.choice([1, 2, 6]))
            ratio = float(np.clip(pair_ratio(a[b]) + rng.normal(0, 0.05), 0.03, 0.97))
            rows.append(row_3(4, 5, ratio, t, float(rng.choice([0.01, 0.05, 0.10]))))
        cands.append(build_rows(rows))
        labels.append("A2 换第三特征联合")

    # 局部：锚点三栋 u 同时小步变化
    for d in (-0.01, 0.01):
        for b in range(N_B):
            rows = [r.copy() for r in a]
            u = float(np.clip(rows[b][ths[b]] + d, 0.0, 0.3))
            rows[b] = row_3(4, 5, pair_ratio(a[b]), ths[b], u)
            cands.append(build_rows(rows))
            labels.append("A2 楼%d u%+.2f" % (b + 1, d))

    out = eval_candidates(ctx, cands, labels, "vE2_A2", "三栋联合3非零")
    res = [{"R": r, "label": lb} for lb, r, _ in out]
    res.sort(key=lambda x: x["R"])
    print("  A2 最优：%.6f  %s" % (res[0]["R"], res[0]["label"]) if res else "  A2 无结果",
          flush=True)


# ============================ B：代理模型 + LCB ============================
def features(w):
    W = np.asarray(w, float).reshape(N_B, N_F)
    f = list(np.asarray(w, float))
    for b in range(N_B):
        f.extend([W[b].sum(), W[b, 4] + W[b, 5], W[b, 5],
                  int((W[b] >= NZ_THR).sum()), int((W[b] == 0).sum())])
    return f


def fit_model(recs):
    X = np.asarray([features(r["w"]) for r in recs], float)
    y = np.asarray([r["R"] for r in recs], float)
    m = RandomForestRegressor(n_estimators=250, min_samples_leaf=2,
                              max_features=1.0, n_jobs=1, random_state=SEED)
    m.fit(X, y)
    return m


def predict_ms(model, pts):
    X = np.asarray([features(w) for w in pts], float)
    pm = np.asarray([t.predict(X) for t in model.estimators_])
    return pm.mean(axis=0), pm.std(axis=0)


def gen_pool(ctx, n=22000):
    a = anchor_rows()
    ths = [third_of(a[b]) for b in range(N_B)]
    top = sorted(ctx["recs"], key=lambda x: x["R"])[:80]
    pts = []

    for _ in range(n // 2):
        rows = []
        for b in range(N_B):
            k = int(rng.choice([1, 2, 2, 2, 3, 3, 3]))
            feats = rng.choice(N_F, size=k, replace=False)
            if k >= 2 and 4 not in feats and rng.random() < 0.7:
                feats[0] = 4
            if k >= 2 and 5 not in feats and rng.random() < 0.7:
                feats[1] = 5
            feats = np.unique(feats)
            x = rng.dirichlet(np.full(len(feats), 0.6))
            row = np.zeros(N_F)
            row[feats] = x
            rows.append(row)
        pts.append(build_rows(rows))

    for _ in range(n // 4):
        base = np.asarray(top[int(rng.integers(len(top)))]["w"], float).reshape(N_B, N_F)
        rows = [r.copy() for r in base]
        for _ in range(int(rng.integers(1, 3))):
            b = int(rng.integers(N_B))
            r = rows[b]
            i, j = rng.choice(N_F, size=2, replace=False)
            amt = float(rng.choice([0.01, 0.02, 0.05, 0.08, 0.12]))
            if r[i] >= amt:
                r[i] -= amt
                r[j] += amt
            else:
                r[r < NZ_THR] = 0.0
                s = r.sum()
                if s > 0:
                    r = r / s
        pts.append(build_rows(rows))

    for _ in range(n // 4):
        rows = []
        for b in range(N_B):
            fi, fj = PAIRS[int(rng.integers(len(PAIRS)))]
            k = int(rng.integers(2, 4))
            if k == 2:
                rows.append(row_2(fi, fj, float(rng.uniform(0.05, 0.95))))
            else:
                t = int(rng.choice([x for x in range(N_F) if x not in (fi, fj)]))
                rows.append(row_3(fi, fj, float(rng.uniform(0.05, 0.95)), t,
                                  float(rng.choice([0.01, 0.02, 0.05, 0.10]))))
        pts.append(build_rows(rows))
    return pts


def select(model, pool, k, beta=0.6):
    mu, sd = predict_ms(model, pool)
    score = mu - beta * sd
    order = np.argsort(score)
    picks = []
    for idx in order:
        w = pool[idx]
        if all(np.abs(w - p).sum() > 0.02 for p in picks):
            picks.append(w)
        if len(picks) >= k:
            break
    return picks


def phase_b(ctx):
    rounds = 0
    while ctx["used"] < BUDGET and rounds < 10:
        rounds += 1
        model = fit_model(ctx["recs"])
        pool = gen_pool(ctx)
        # 去掉已评估点
        pool = [w for w in pool if key_of(w) not in ctx["cache"]]
        if not pool:
            break
        k = min(50, BUDGET - ctx["used"])
        picks = select(model, pool, k)
        cands = picks
        labels = ["B 代理选点 r%d#%d" % (rounds, i + 1) for i in range(len(cands))]
        out = eval_candidates(ctx, cands, labels, "vE2_B_r%d" % rounds,
                              "代理模型第%d轮，池%d" % (rounds, len(pool)))
        Rs = [r for _, r, _ in out]
        best_r = min(Rs) if Rs else float("nan")
        print("  B 第%d轮：候选 %d，本轮最优 %.6f，全局最优 %.6f"
              % (rounds, len(Rs), best_r, ctx["best"]["R"]), flush=True)
        if ctx["best"]["R"] < 103:
            print("  ！！！ 出现 R<103：%.9f  %s"
                  % (ctx["best"]["R"], ctx["best"]["label"]), flush=True)


# ============================ 主入口 ============================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for t in self.streams:
            try:
                t.write(s)
                t.flush()
            except Exception:
                pass

    def flush(self):
        for t in self.streams:
            try:
                t.flush()
            except Exception:
                pass


def main():
    global RUN_LIMIT
    logf = open(os.path.join(ROOT, "optimization", "E2_stdout.log"), "a",
                encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, logf)
    sys.stderr = Tee(sys.__stderr__, logf)
    stage = (sys.argv[1] if len(sys.argv) > 1 else "all").upper()
    if len(sys.argv) > 2:
        RUN_LIMIT = int(sys.argv[2])
    ctx = make_ctx()
    purge_mei()
    disk_guard("启动前", ctx)
    write_progress(ctx, stage, "启动")
    print("=== 阶段E2 %s 开始（预算 %d，已用 %d，本次上限 %s，seed=%d）==="
          % (stage, BUDGET, ctx["used"], RUN_LIMIT if RUN_LIMIT > 0 else "不限", SEED),
          flush=True)
    try:
        if stage in ("A0", "ALL"):
            phase_a0(ctx)
        if stage in ("A1", "ALL"):
            phase_a1(ctx)
        if stage in ("A2", "ALL"):
            phase_a2(ctx)
        if stage in ("B", "ALL"):
            phase_b(ctx)
    finally:
        ctx["ev"].close()
        write_progress(ctx, stage, "结束")
        print("=== 阶段E2 %s 结束：本次新增 %d，累计 E2 %d，全局最优 %.9f ==="
              % (stage, ctx["new"], ctx["used"], ctx["best"]["R"]), flush=True)


if __name__ == "__main__":
    main()
