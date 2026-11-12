# -*- coding: utf-8 -*-
"""阶段 E5：放开"每栋 4 非零"，进入 5–6 非零结构 + 量化阶梯验证。

背景（E4 结论）：最优 R=104.851733；凹口极窄（≤0.0025）但浅（每坐标 0.01–0.11）；
同一 R 值在不同权重上反复出现，疑似求解器内部对权重分箱/量化。

阶段：
  Q1  判定性采样（~43 点）：5 非零（10 个结构 × 2 档 m56）、6 非零（5 个结构）、
      楼间异构 5 非零、以及"从 E4 最优向新特征转移 0.05/0.10/0.15"。
      → 若最低点 < 锚点−0.05，说明支持集扩展有信号，进入 Q2。
  Q2  扩展（仅 Q1 有信号）：楼间异构 5 非零组合、次特征双取、两栋改用非 F5+F6 主对。
  Q3  量化验证（~49 点）：固定结构，单坐标 0.0001 级微扰（两段 21 点 + 粗档 11 点），
      看 R 是否在少数固定值之间跳，并据此估计量化步长。

全部真实评估；每批 20 点落 optimization/eval_log.jsonl；
每批写 optimization/E5_progress.json；R<103 立刻写 ALERT 文件。
用法：py -3 optimize_weights_phaseE5.py all|Q1|Q2|Q3 [本次最多新增评估数]
"""
import itertools
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.environ.get("SDM_WORKSPACE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optimize_weights_phaseE import (key_of, append_log, disk_guard, purge_mei,
                                     ensure_base_solver)
from optimize_weights_phaseE2 import build_rows, load_cache, NZ_THR
from solver_robust import RobustServe

# ============================ 配置 ============================
SEED = 20261109
BUDGET = 400
BATCH = 20
N_B, N_F = 3, 7
LOG_PATH = os.path.join(ROOT, "optimization", "eval_log.jsonl")
PROGRESS_PATH = os.path.join(ROOT, "optimization", "E5_progress.json")
E4BEST_PATH = os.path.join(ROOT, "best_weights_phaseE4.json")
E3BEST_PATH = os.path.join(ROOT, "best_weights_phaseE3.json")
BEST_PATH = os.path.join(ROOT, "best_weights_phaseE5.json")
SUMMARY_PATH = os.path.join(ROOT, "optimization", "E5_summary.json")
ALERT_PATH = os.path.join(ROOT, "optimization", "ALERT_R_under_103.json")
RUN_LIMIT = 0

Q1_SIGNAL = 0.05          # Q1 最低点需低于锚点 0.05 才进入 Q2
MIN3 = [tuple(c) for c in itertools.combinations((0, 1, 2, 3, 6), 3)]   # 10 个三特征组合
MIN4 = [tuple(c) for c in itertools.combinations((0, 1, 2, 3, 6), 4)]   # 5 个四特征组合

rng = np.random.default_rng(SEED)


# ============================ 结构构造 ============================
def row_m(minors, m56, r56, pat):
    """F5+F6 占 m56（按 r56 分给 F5），其余 m56 之外的权重按 pat 分给 minors。"""
    row = np.zeros(N_F)
    row[4] = m56 * r56
    row[5] = m56 * (1.0 - r56)
    rest = 1.0 - m56
    for f, p in zip(minors, pat):
        row[f] = rest * p
    row[row < 1e-12] = 0.0
    return row


def row_pair(pi, pj, mmain, ratio, minors, pat):
    """任意主对 (pi,pj) 占 mmain，其余按 pat 分给 minors。"""
    row = np.zeros(N_F)
    row[pi] = mmain * ratio
    row[pj] = mmain * (1.0 - ratio)
    rest = 1.0 - mmain
    for f, p in zip(minors, pat):
        row[f] = rest * p
    row[row < 1e-12] = 0.0
    return row


def load_anchor():
    p = E4BEST_PATH if os.path.exists(E4BEST_PATH) else E3BEST_PATH
    d = json.load(open(p, encoding="utf-8"))
    W = np.asarray(d["weights_by_building"], float)
    return W / W.sum(axis=1, keepdims=True), float(d["best_R"]), p


def add_from(row, f_new, delta):
    """从 F5/F6 按原比例拿出 delta 给新特征 f_new（保持行和=1）。"""
    x = np.asarray(row, float).copy()
    s = float(x[4] + x[5])
    if s <= delta:
        return None
    r = x[4] / s
    x[4] = (s - delta) * r
    x[5] = (s - delta) * (1.0 - r)
    x[f_new] += delta
    return x


# ============================ 进度 / 评估（沿用 E4 契约）============================
def write_progress(ctx, stage, note=""):
    b = ctx["best"]
    under = sum(1 for r in ctx["recs"] if r["phase"].startswith("vE5") and r["R"] < 103)
    out = {"updated": datetime.now().isoformat(timespec="seconds"),
           "stage": stage, "note": note,
           "used_e5": int(ctx["used"]), "budget_e5": BUDGET,
           "new_this_run": int(ctx["new"]),
           "log_eval_id_last": int(ctx["last_id"]), "log_lines": int(ctx["last_id"] + 1),
           "best_R": float(b["R"]), "best_eval_id": b["eval_id"], "best_label": b["label"],
           "best_weights": (b["w"].tolist() if isinstance(b["w"], np.ndarray) else b["w"]),
           "below_103_count": under,
           "anchor_R": float(ctx["anchor_R"]),
           "q1_best_R": ctx.get("q1_best", None),
           "q1_best_label": ctx.get("q1_label", None),
           "q1r_best_R": ctx.get("q1r_best", None),
           "q1r_best_label": ctx.get("q1r_label", None),
           "retry_events": len(getattr(ctx.get("ev"), "events", []) or []),
           "retry_last": (ctx["ev"].events[-1] if getattr(ctx.get("ev"), "events", None) else None),
           "failed_points": ctx.get("failed", [])[-10:],
           "stage_counts": ctx["stage_counts"]}
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PROGRESS_PATH)


def write_alert(ctx):
    b = ctx["best"]
    W = np.asarray(b["w"], float).reshape(N_B, N_F)
    rec = {"R": float(b["R"]), "eval_id": b["eval_id"], "label": b["label"],
           "weights": np.asarray(b["w"], float).ravel().tolist(),
           "weights_by_building": W.tolist(),
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(ALERT_PATH, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)


def make_ctx():
    cache, last, recs = load_cache()
    used = sum(1 for r in recs if r["phase"].startswith("vE5"))
    gb = min(recs, key=lambda x: x["R"]) if recs else None
    anchor, anchor_R, src = load_anchor()
    return {"cache": cache, "last_id": last, "used": used, "new": 0, "run_new": 0,
            "recs": recs, "ev": RobustServe(workers=4, batch=BATCH), "failed": [],
            "anchor": anchor, "anchor_R": anchor_R, "anchor_src": src,
            "q1_best": None, "q1_label": None, "q1r_best": None, "q1r_label": None,
            "best": {"R": gb["R"] if gb else float("inf"),
                     "w": gb["w"].copy() if gb else None,
                     "eval_id": gb["eval_id"] if gb else None,
                     "label": gb["label"] if gb else ""},
            "stage_counts": {}}


def eval_candidates(ctx, cands, labels, phase, note="", room_cap=None):
    keys, todo, tl = [], [], []
    for w, lb in zip(cands, labels):
        if w is None:
            continue
        k = key_of(w)
        if k in ctx["cache"] or k in keys:
            continue
        keys.append(k)
        todo.append(np.asarray(w, float))
        tl.append(lb)
    room = BUDGET - ctx["used"]
    if room_cap is not None:
        room = min(room, max(0, int(room_cap)))
    if RUN_LIMIT > 0:
        room = min(room, RUN_LIMIT - ctx["run_new"])
    skipped, n_new = 0, 0
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
            recs_out, n_ok = [], 0
            for w, lb, r in zip(chunk, cl, Rs):
                if r is None or not np.isfinite(r):
                    ctx["failed"].append({"label": lb, "weights": w.tolist(),
                                          "t": datetime.now().isoformat(timespec="seconds")})
                    print("  [失败点] 无有效 R（不计预算）：%s" % lb, flush=True)
                    continue
                n_ok += 1
                eid += 1
                ctx["cache"][key_of(w)] = (r, eid)
                ctx["recs"].append({"eval_id": eid, "R": r, "w": w.copy(),
                                    "phase": phase, "label": lb})
                recs_out.append({"eval_id": eid, "phase": phase, "weights": w.tolist(),
                                 "R": r, "label": lb,
                                 "nz": int((w > 0).sum()), "nz_eff": int((w >= NZ_THR).sum()),
                                 "nz_rows_exact": [int((w.reshape(N_B, N_F)[i] > 0).sum())
                                                   for i in range(N_B)],
                                 "timestamp": datetime.now().isoformat(timespec="seconds")})
                if r < ctx["best"]["R"]:
                    ctx["best"].update({"R": r, "w": w.copy(), "eval_id": eid, "label": lb})
                    if r < 103:
                        write_alert(ctx)
                        print("  ！！！出现 R<103：%.9f  eval_id=%d  %s" % (r, eid, lb),
                              flush=True)
            append_log(recs_out)
            ctx["last_id"] = eid
            ctx["used"] += n_ok
            ctx["new"] += n_ok
            ctx["run_new"] += n_ok
            n_new += n_ok
            ctx["stage_counts"][phase] = ctx["stage_counts"].get(phase, 0) + n_ok
            write_progress(ctx, phase, note)
            disk_guard("第 %d/%d 批结束" % (bi + 1, nb), ctx)
    elif todo:
        skipped = len(todo)
    out = []
    for w, lb in zip(cands, labels):
        if w is None:
            continue
        k = key_of(w)
        out.append((lb, ctx["cache"][k][0] if k in ctx["cache"] else float("inf")))
    print("  [%s] 候选 %d，新增 %d（E5 %d/%d）%s"
          % (phase, len(cands), n_new, ctx["used"], BUDGET,
             "，跳过 %d" % skipped if skipped else ""), flush=True)
    return out


# ============================ Q1：判定性采样 ============================
def phase_q1(ctx):
    cands, labels = [], []

    # A：10 个 5 非零结构，三栋同构，m56=0.65
    for tri in MIN3:
        rows = [row_m(tri, 0.65, 0.25, (0.40, 0.35, 0.25)) for _ in range(N_B)]
        cands.append(build_rows(rows))
        labels.append("Q1A 5nz %s m56=.65" % (tri,))

    # B：同 10 结构，m56=0.60 更小主对
    for tri in MIN3:
        rows = [row_m(tri, 0.60, 0.30, (0.50, 0.30, 0.20)) for _ in range(N_B)]
        cands.append(build_rows(rows))
        labels.append("Q1B 5nz %s m56=.60" % (tri,))

    # C：楼间异构 5 非零（三栋不同次特征组合/不同 m56）
    het = [((0, 1, 6), (1, 2, 6), (0, 2, 6), (0.65, 0.60, 0.62)),
           ((0, 1, 2), (0, 2, 6), (1, 2, 6), (0.60, 0.65, 0.58)),
           ((0, 3, 6), (1, 3, 6), (2, 3, 6), (0.62, 0.68, 0.60)),
           ((0, 1, 3), (1, 2, 3), (0, 2, 3), (0.66, 0.60, 0.64)),
           ((0, 6, 2), (1, 6, 3), (0, 1, 6), (0.58, 0.64, 0.70)),
           ((1, 2, 6), (0, 1, 2), (0, 3, 6), (0.60, 0.62, 0.66))]
    for t1, t2, t3, ms in het:
        rows = [row_m(t1, ms[0], 0.25, (0.40, 0.35, 0.25)),
                row_m(t2, ms[1], 0.30, (0.45, 0.30, 0.25)),
                row_m(t3, ms[2], 0.28, (0.35, 0.35, 0.30))]
        cands.append(build_rows(rows))
        labels.append("Q1C het5 %s|%s|%s" % (t1, t2, t3))

    # D：6 非零（4 个次特征），三栋同构
    for q in MIN4:
        rows = [row_m(q, 0.60, 0.27, (0.30, 0.30, 0.20, 0.20)) for _ in range(N_B)]
        cands.append(build_rows(rows))
        labels.append("Q1D 6nz %s" % (q,))

    # E：从 E4 最优向一个新特征转移 delta（三栋同时）
    for f in (0, 1, 2, 3):
        for d in (0.05, 0.10, 0.15):
            rows = [add_from(ctx["anchor"][b], f, d) for b in range(N_B)]
            if any(r is None for r in rows):
                continue
            cands.append(build_rows(rows))
            labels.append("Q1E trans F%d +%.2f" % (f + 1, d))

    out = eval_candidates(ctx, cands, labels, "vE5_Q1", "5–6 非零判定性采样")
    fin = [(lb, r) for lb, r in out if np.isfinite(r)]
    if fin:
        lb, R = min(fin, key=lambda x: x[1])
        ctx["q1_best"], ctx["q1_label"] = float(R), lb
        print("  Q1 最优 %.6f（%s）；锚点 %.6f；改善 %.4f"
              % (R, lb, ctx["anchor_R"], ctx["anchor_R"] - R), flush=True)


# ============================ Q2：有信号才扩展 ============================
def phase_q2(ctx):
    cands, labels = [], []

    # H：三栋各用不同 5 非零组合（三不同）
    tris = [(0, 1, 6), (0, 2, 6), (1, 2, 6), (0, 1, 2)]
    for perm in itertools.permutations(range(4), 3):
        t = [tris[i] for i in perm]
        rows = [row_m(t[0], 0.65, 0.25, (0.40, 0.35, 0.25)),
                row_m(t[1], 0.60, 0.30, (0.50, 0.30, 0.20)),
                row_m(t[2], 0.68, 0.28, (0.35, 0.35, 0.30))]
        cands.append(build_rows(rows))
        labels.append("Q2H het5 %s|%s|%s" % tuple(t))

    # M：两栋改用非 F5+F6 主对（{F4,F5} 或 {F6,F3}），第三栋保持 F5+F6
    nonpairs = [((3, 4), (1, 5, 6)), ((5, 2), (0, 1, 6))]
    for pi, pj in nonpairs:
        for which in ((0, 1), (0, 2), (1, 2), (0, 1, 2)):
            rows = []
            for b in range(N_B):
                if b in which:
                    rows.append(row_pair(pi, pj, 0.75, 0.50, (1, 5, 6) if pi == 3 else (0, 1, 6),
                                         (0.40, 0.35, 0.25)))
                else:
                    rows.append(row_m((0, 1, 6), 0.65, 0.25, (0.40, 0.35, 0.25)))
            cands.append(build_rows(rows))
            labels.append("Q2M pair%d-%d in%s" % (pi + 1, pj + 1, which))

    # T：次特征同时取两个（5 非零，两个次特征 ≥0.10，其余两小）
    for tri in [(0, 1, 6), (1, 2, 6), (0, 2, 6)]:
        for pat in [(0.35, 0.30, 0.35), (0.40, 0.20, 0.40)]:
            rows = [row_m(tri, 0.62, 0.26, pat) for _ in range(N_B)]
            cands.append(build_rows(rows))
            labels.append("Q2T two-minor %s %s" % (tri, pat))

    # S：6 非零异构（三栋不同 four-set）
    for t1, t2, t3 in [((0, 1, 2, 6), (0, 1, 3, 6), (0, 2, 3, 6)),
                       ((1, 2, 3, 6), (0, 1, 2, 3), (0, 1, 2, 6))]:
        rows = [row_m(t1, 0.62, 0.28, (0.30, 0.30, 0.20, 0.20)),
                row_m(t2, 0.58, 0.25, (0.25, 0.25, 0.25, 0.25)),
                row_m(t3, 0.66, 0.30, (0.35, 0.25, 0.20, 0.20))]
        cands.append(build_rows(rows))
        labels.append("Q2S het6 %s|%s|%s" % (t1, t2, t3))

    out = eval_candidates(ctx, cands, labels, "vE5_Q2", "5–6 非零扩展")
    fin = [(lb, r) for lb, r in out if np.isfinite(r)]
    if fin:
        lb, R = min(fin, key=lambda x: x[1])
        print("  Q2 最优 %.6f（%s）" % (R, lb), flush=True)


# ============================ Q1R：随机撒点（无信号时的强证伪）============================
def phase_q1r(ctx):
    """在 5–6 非零族内随机撒 120 点：随机结构、随机 m56(0.45–0.85)、随机 r56 与占比。"""
    cands, labels = [], []
    for i in range(120):
        rows = []
        for b in range(N_B):
            k = int(rng.integers(3, 5))                      # 3 或 4 个次特征 → 5/6 非零
            tri = tuple(sorted(rng.choice((0, 1, 2, 3, 6), size=k, replace=False).tolist()))
            m56 = float(rng.uniform(0.45, 0.85))
            r56 = float(rng.uniform(0.10, 0.50))
            pat = rng.dirichlet(np.ones(k))
            rows.append(row_m(tri, m56, r56, pat))
        cands.append(build_rows(rows))
        labels.append("Q1R rand #%d" % i)
    out = eval_candidates(ctx, cands, labels, "vE5_Q1R", "5–6 非零随机撒点")
    fin = [(lb, r) for lb, r in out if np.isfinite(r)]
    if fin:
        lb, R = min(fin, key=lambda x: x[1])
        ctx["q1r_best"], ctx["q1r_label"] = float(R), lb
        print("  Q1R 随机撒点最优 %.6f（%s）；锚点 %.6f"
              % (R, lb, ctx["anchor_R"]), flush=True)


def has_signal(ctx):
    vals = [v for v in (ctx.get("q1_best"), ctx.get("q1r_best")) if v is not None]
    return bool(vals) and min(vals) < ctx["anchor_R"] - Q1_SIGNAL


# ============================ Q3：量化阶梯验证 ============================
def phase_q3(ctx):
    base = ctx["anchor"]
    b = 0
    # A：楼1 F6→F5 转移 δ，0.0001 级（±0.001）
    cands, labels, grid = [], [], []
    for d in np.round(np.arange(-0.001, 0.001 + 1e-12, 0.0001), 7):
        row = np.asarray(base[b], float).copy()
        row[4] += d
        row[5] -= d
        rows = [row if k == b else base[k].copy() for k in range(N_B)]
        cands.append(build_rows(rows))
        labels.append("Q3A F6->F5 d=%+.4f" % d)
        grid.append(float(d))
    out = eval_candidates(ctx, cands, labels, "vE5_Q3A", "0.0001 微扰（F5/F6）")
    print("  Q3A 曲线：%s" % [(g, round(r, 4)) for g, (lb, r) in zip(grid, out)], flush=True)

    # B：楼1 F2 在锚点值附近的 0.0001 级扫描（差额由 F5/F6 按比例吸收）
    cands, labels, grid = [], [], []
    for d in np.round(np.arange(-0.001, 0.001 + 1e-12, 0.0001), 7):
        row = np.asarray(base[b], float).copy()
        rest = float(row[4] + row[5])
        if rest <= abs(d) + 1e-6:
            continue
        ratio = row[4] / rest
        row[4] = (rest - d) * ratio
        row[5] = (rest - d) * (1.0 - ratio)
        row[1] += d
        rows = [row if k == b else base[k].copy() for k in range(N_B)]
        cands.append(build_rows(rows))
        labels.append("Q3B F2 d=%+.4f" % d)
        grid.append(float(d))
    out = eval_candidates(ctx, cands, labels, "vE5_Q3B", "0.0001 微扰（F2）")
    print("  Q3B 曲线：%s" % [(g, round(r, 4)) for g, (lb, r) in zip(grid, out)], flush=True)

    # C：粗档 ±0.005，定位"开始变化"的阈值
    cands, labels, grid = [], [], []
    for d in (-0.005, -0.004, -0.003, -0.002, 0.002, 0.003, 0.004, 0.005):
        row = np.asarray(base[b], float).copy()
        row[4] += d
        row[5] -= d
        rows = [row if k == b else base[k].copy() for k in range(N_B)]
        cands.append(build_rows(rows))
        labels.append("Q3C F6->F5 d=%+.4f" % d)
        grid.append(float(d))
    out = eval_candidates(ctx, cands, labels, "vE5_Q3C", "粗档 ±0.005")
    print("  Q3C 曲线：%s" % [(g, round(r, 4)) for g, (lb, r) in zip(grid, out)], flush=True)


# ============================ Q4：量化边界精定位 ============================
def phase_q4(ctx):
    """在 Q3A 已定位的两个层边界附近用 2e-5 步长把边界夹到更窄，估计量化步长。"""
    base = ctx["anchor"]
    b = 0
    ds = [-0.00072, -0.00070, -0.00068, -0.00066, -0.00064, -0.00062,
          0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012]
    cands, labels, grid = [], [], []
    for d in ds:
        row = np.asarray(base[b], float).copy()
        row[4] += d
        row[5] -= d
        rows = [row if k == b else base[k].copy() for k in range(N_B)]
        cands.append(build_rows(rows))
        labels.append("Q4 F6->F5 d=%+.5f" % d)
        grid.append(float(d))
    out = eval_candidates(ctx, cands, labels, "vE5_Q4", "量化边界精定位 2e-5")
    print("  Q4 曲线：%s" % [(g, round(r, 4)) for g, (lb, r) in zip(grid, out)], flush=True)


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


def write_best(ctx):
    b = ctx["best"]
    W = np.asarray(b["w"], float).reshape(N_B, N_F) if b["w"] is not None else None
    out = {"best_R": float(b["R"]), "eval_id": int(b["eval_id"]), "label": b["label"],
           "weights_flat": np.asarray(b["w"], float).ravel().tolist() if W is not None else None,
           "weights_by_building": W.tolist() if W is not None else None,
           "row_sums": W.sum(axis=1).tolist() if W is not None else None,
           "nz_exact": int((W > 0).sum()) if W is not None else None,
           "nz_rows_exact": [int((W[i] > 0).sum()) for i in range(N_B)] if W is not None else None,
           "total_evals_log": int(ctx["last_id"] + 1),
           "total_evals_e5": int(ctx["used"]), "budget_e5": BUDGET,
           "anchor_R": float(ctx["anchor_R"]), "anchor_src": ctx.get("anchor_src"),
           "q1_best_R": ctx.get("q1_best"), "q1_best_label": ctx.get("q1_label"),
           "below_103_count": sum(1 for r in ctx["recs"]
                                  if r["phase"].startswith("vE5") and r["R"] < 103)}
    with open(BEST_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    global RUN_LIMIT
    logf = open(os.path.join(ROOT, "optimization", "E5_stdout.log"), "a",
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
    print("=== 阶段E5 %s 开始（预算 %d，已用 %d，本次上限 %s，日志 %d 行，锚点 %.6f 来自 %s）==="
          % (stage, BUDGET, ctx["used"], RUN_LIMIT if RUN_LIMIT > 0 else "不限",
             ctx["last_id"] + 1, ctx["anchor_R"], os.path.basename(ctx["anchor_src"])),
          flush=True)
    try:
        if stage in ("Q1", "ALL"):
            phase_q1(ctx)
        if stage in ("Q3", "ALL"):
            phase_q3(ctx)
        if stage in ("Q4", "ALL"):
            phase_q4(ctx)
        if stage in ("Q1R", "ALL", "Q2"):
            if ctx.get("q1r_best") is None and stage in ("Q1R", "ALL") and not has_signal(ctx):
                phase_q1r(ctx)
        if stage in ("Q2", "ALL"):
            if stage == "Q2" or has_signal(ctx):
                phase_q2(ctx)
            else:
                print("  Q2 跳过：Q1/Q1R 最优未低于锚点 %.6f − %.2f（支持集扩展无信号）"
                      % (ctx["anchor_R"], Q1_SIGNAL), flush=True)
    finally:
        ctx["ev"].close()
        write_best(ctx)
        write_progress(ctx, stage, "结束")
        print("=== 阶段E5 %s 结束：本次新增 %d，累计 E5 %d，全局最优 %.9f（eval %s）==="
              % (stage, ctx["new"], ctx["used"], ctx["best"]["R"], ctx["best"]["eval_id"]),
              flush=True)


if __name__ == "__main__":
    main()
