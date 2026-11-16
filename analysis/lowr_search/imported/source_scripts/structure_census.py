# -*- coding: utf-8 -*-
"""
structure_census.py — 343 个 dominant 结构格定向普查 + top 格精修

动机：随机撒点在 21 维单纯形里几乎不可能凑出"某栋楼 0.73/0.24/0.02/0.01"这类
精确配比，所以 343 个结构格里 211 个至今只有 8~12 个随机点。本脚本改为
"每个结构格都用该格自己的代表性配方生成 6 个点"，保证每个格子被公平试过。

预算（上限 8000）：
  已用            4904
  普查 343×6      2058
  top16 精修 16×60  960
  合计            7922 ≤ 8000
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_simplex_search as rss

LIMIT = 8000
USED_BEFORE = 4904
CENSUS_PER_CELL = 6
REFINE_TOP = 16
REFINE_PER_CELL = 60

ROOT = rss.ROOT
RUN_DIR = os.path.join(ROOT, "structure_census_run1")
os.makedirs(RUN_DIR, exist_ok=True)
rss.RUN_DIR = RUN_DIR
rss.LOG_PATH = os.path.join(RUN_DIR, "eval_log.jsonl")
rss.RESULT_PATH = os.path.join(RUN_DIR, "result.json")
LOG_PATH = rss.LOG_PATH

np.random.seed(20261114)

CELLS = [(a, b, c) for a in range(7) for b in range(7) for c in range(7)]
POOL = [4, 5, 6, 2, 3, 1, 0]   # 次特征优先池：degradation/ramping/resilience/peak

FEAT_NAMES = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]


# ============================ 结构配方 ============================
def build_row(assign):
    """assign: [(idx, value), ...]，重复索引累加。"""
    v = np.zeros(7)
    for i, x in assign:
        v[i] += x
    v = np.clip(v, rss.EPS, None)
    return v / v.sum()


def pick_pool(d, b, k):
    pool = [x for x in POOL if x != d]
    return pool[(k + b) % len(pool)]


def census_row(d, b, k):
    """给定主导特征 d、楼号 b、模板号 k，返回该楼的 7 维权重行。"""
    if k == 0:                       # 近顶点
        return build_row([(d, 0.97)])
    if k == 1:                       # 双特征，主 0.85
        s = pick_pool(d, b, 1)
        return build_row([(d, 0.85), (s, 0.14)])
    if k == 2:                       # 已发现最优的配方形状
        s = pick_pool(d, b, 2)
        return build_row([(d, 0.72), (s, 0.24), (2, 0.02), (6, 0.01)])
    if k == 3:                       # 双特征，次特征取索引相邻
        s = (d + 1) % 7
        return build_row([(d, 0.73), (s, 0.26)])
    if k == 4:                       # 三特征
        s1 = pick_pool(d, b, 4)
        s2 = (d + 1) % 7
        return build_row([(d, 0.60), (s1, 0.25), (s2, 0.12)])
    s1 = pick_pool(d, b, 5)          # 摊平分布
    return build_row([(d, 0.40), (s1, 0.22), ((d + 1) % 7, 0.15),
                      ((d + 2) % 7, 0.10), ((d + 3) % 7, 0.06)])


def census_point(cell, k):
    return rss.build_rows([census_row(cell[b], b, k) for b in range(3)])


# ============================ 去重 ============================
def key_of(w):
    return tuple(np.round(np.asarray(w, float).ravel(), 6))


def load_landscape_keys():
    import csv
    path = os.path.join(ROOT, "landscape_bundle", "landscape_data.csv")
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        idx = [header.index(c) for c in header if c.startswith("w_b")]
        for row in rd:
            try:
                keys.add(tuple(np.round(np.array([float(row[i]) for i in idx]), 6)))
            except Exception:
                continue
    return keys


def load_log():
    recs = []
    if not os.path.exists(LOG_PATH):
        return recs
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rec["_key"] = key_of(rec["weights"])
            recs.append(rec)
    return recs


# ============================ 精修 ============================
def refine_moves(rows, step, topk):
    out = []
    for b in range(3):
        order = np.argsort(-np.asarray(rows[b], float))[:topk]
        for ii in range(len(order)):
            for jj in range(len(order)):
                if ii == jj:
                    continue
                i, j = int(order[ii]), int(order[jj])
                for sign in (1.0, -1.0):
                    rr = [np.asarray(x, float).copy() for x in rows]
                    rr[b][i] += sign * step
                    rr[b][j] -= sign * step
                    if rr[b][i] > rss.EPS and rr[b][j] > rss.EPS:
                        out.append(rr)
    return out


def refine_cell(ev, rows, best_R, budget, label, used_counter):
    steps = [(0.05, 2), (0.03, 2), (0.015, 2), (0.006, 3)]
    cur = [np.asarray(r, float).copy() for r in rows]
    curR = float(best_R)
    bestW, bestR = rss.build_rows(cur), float(best_R)
    for step, topk in steps:
        left = budget - used_counter[0]
        if left <= 0:
            break
        cands = refine_moves(cur, step, topk)[:left]
        if not cands:
            continue
        Ws = [rss.build_rows(c) for c in cands]
        Rs = ev.evaluate(Ws, label=label, print_every=20, log_phase=label)
        used_counter[0] += len(Ws)
        k = int(np.argmin(Rs))
        if Rs[k] < bestR:
            bestR = float(Rs[k])
            bestW = Ws[k].copy()
        if Rs[k] <= curR:
            cur, curR = cands[k], float(Rs[k])
    return bestW, bestR


# ============================ 主流程 ============================
def main():
    t0 = time.time()
    print("=== 结构普查开始 ===", flush=True)
    print("  上限 %d  已用 %d  普查 %d×%d  精修 top%d×%d"
          % (LIMIT, USED_BEFORE, len(CELLS), CENSUS_PER_CELL, REFINE_TOP, REFINE_PER_CELL),
          flush=True)

    rss.cleanup_pyi()
    rss.ensure_solver()
    log = load_log()
    done = {r["_key"]: r["R"] for r in log}
    rss.EVAL_IDX[0] = max([int(r.get("eval_id", USED_BEFORE)) for r in log] + [USED_BEFORE]) + 1
    print("  已有日志 %d 条，续跑 eval_id=%d" % (len(log), rss.EVAL_IDX[0]), flush=True)

    existing = load_landscape_keys()
    ev = rss.ParallelEval()
    try:
        # ---------- 阶段1：结构格普查 ----------
        print("\n=== 阶段1：343 格 × 6 配方 ===", flush=True)
        cand_keys, cand_cells, cand_W, cand_known = [], [], [], []
        skipped = 0
        for cell in CELLS:
            for k in range(CENSUS_PER_CELL):
                w = census_point(cell, k)
                kk = key_of(w)
                if kk in existing and kk not in done:
                    skipped += 1
                    continue
                cand_keys.append(kk)
                cand_cells.append(cell)
                cand_W.append(w)
                cand_known.append(done.get(kk))

        pending = [i for i, r in enumerate(cand_known) if r is None]
        print("  候选 %d，其中已评估 %d，待评估 %d，跳过与旧数据重复 %d"
              % (len(cand_W), len(cand_W) - len(pending), len(pending), skipped), flush=True)

        R_census = [float(r) if r is not None else None for r in cand_known]
        if pending:
            Wp = [cand_W[i] for i in pending]
            Rp = ev.evaluate(Wp, label="CENSUS", print_every=60, log_phase="CENSUS")
            for i, r in zip(pending, Rp):
                R_census[i] = float(r)

        best_per_cell = {}
        for cell, r, w in zip(cand_cells, R_census, cand_W):
            if r is None:
                continue
            if cell not in best_per_cell or r < best_per_cell[cell][0]:
                best_per_cell[cell] = (r, w)

        ranked = sorted(best_per_cell.items(), key=lambda kv: kv[1][0])
        print("\n  结构格排名（前 25）:", flush=True)
        for cell, (r, w) in ranked[:25]:
            print("    cell=%s  best_R=%.4f  dom=%s" % (cell, r, tuple(np.argmax(w.reshape(3, 7), axis=1))), flush=True)
        print("\n  结构格排名（后 10）:", flush=True)
        for cell, (r, w) in ranked[-10:]:
            print("    cell=%s  best_R=%.4f" % (cell, r), flush=True)
        print("  覆盖结构格 %d / 343，全局普查最优 R=%.6f" % (len(ranked), ranked[0][1][0]), flush=True)

        # ---------- 阶段2：top 格精修 ----------
        print("\n=== 阶段2：top%d 结构格精修（每格 %d 次）===" % (REFINE_TOP, REFINE_PER_CELL), flush=True)
        done_refine = {}
        for r in log:
            ph = r.get("phase", "")
            if ph.startswith("REFINE_c"):
                done_refine[ph] = done_refine.get(ph, 0) + 1

        bestW, bestR, bestLabel = ranked[0][1][1], ranked[0][1][0], "CENSUS_%s" % (ranked[0][0],)
        for rank, (cell, (r, w)) in enumerate(ranked[:REFINE_TOP]):
            label = "REFINE_c%02d" % rank
            rows = np.asarray(w, float).reshape(3, 7)
            left = REFINE_PER_CELL - done_refine.get(label, 0)
            if left <= 0:
                print("  [%d] cell=%s 已完成" % (rank + 1, cell), flush=True)
                continue
            print("\n  精修 %d/%d cell=%s 起点 R=%.4f  dom=%s"
                  % (rank + 1, REFINE_TOP, cell, r, tuple(np.argmax(rows, axis=1))), flush=True)
            used_counter = [0]
            ww, rr = refine_cell(ev, rows, r, left, label, used_counter)
            print("    → R=%.6f  (用了 %d 次)" % (rr, used_counter[0]), flush=True)
            if rr < bestR:
                bestR, bestW, bestLabel = rr, ww, "%s_cell=%s" % (label, cell)

        # ---------- 汇总 ----------
        total = rss.EVAL_IDX[0] - 1
        print("\n" + "=" * 60, flush=True)
        print("=== 结构普查完毕 ===", flush=True)
        print("  累计真实评估: %d / %d" % (total, LIMIT), flush=True)
        print("  普查最优 R=%.6f" % ranked[0][1][0], flush=True)
        print("  全程最优 R=%.6f  (%s)" % (bestR, bestLabel), flush=True)
        W = np.asarray(bestW).reshape(3, 7)
        for i, row in enumerate(W):
            print("    楼%d: %s  dom=F%d" % (i + 1, np.round(row, 4).tolist(), np.argmax(row) + 1), flush=True)
        print("  历史最优对照: 本次 %.3f | anchor 106.522 | landscape 全局 104.852 | 目标 96"
              % bestR, flush=True)

        result = {
            "params": {"LIMIT": LIMIT, "USED_BEFORE": USED_BEFORE,
                       "CENSUS_PER_CELL": CENSUS_PER_CELL, "REFINE_TOP": REFINE_TOP,
                       "REFINE_PER_CELL": REFINE_PER_CELL, "SEED": 20261114},
            "census_best_R": float(ranked[0][1][0]),
            "overall_best_R": float(bestR),
            "overall_best_weights": np.asarray(bestW, float).ravel().tolist(),
            "overall_best_label": bestLabel,
            "cells_covered": len(ranked),
            "cell_ranking": [{"cell": list(c), "best_R": float(v[0])} for c, v in ranked],
            "total_evals": total,
            "elapsed_seconds": time.time() - t0,
            "eval_log": LOG_PATH,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(rss.RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("结果已落盘: %s" % rss.RESULT_PATH, flush=True)

    finally:
        ev.close()


if __name__ == "__main__":
    main()
