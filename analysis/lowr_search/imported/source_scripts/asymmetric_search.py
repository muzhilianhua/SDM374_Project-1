# -*- coding: utf-8 -*-
"""
asymmetric_search.py — 不对称探索

依据：固定另两楼为 F5/F6 主导时，楼2 对 cost(F1) 和 resilience(F7) 最宽容
  楼2: F1(cost)=108.1  F7(resilience)=106.2  均优于楼1/楼3 同特征
假设：让楼2 吸收更多 cost/resilience 权重，楼1/楼3 保持纯 F6 族，
      可能在系统层面得到比 104.85 更低的 R。

设计（预算 ~1000 点）：
  楼1: 4 个 F6 族模板
  楼3: 4 个 F6 族模板
  楼2: 8(cost) × 8(resilience) = 64 个组合，ramping 填满，degradation 固定 0.02
  合计 4×4×64 = 1024 点
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_simplex_search as rss

USED_BEFORE = 7922            # 结构普查预算上限（已跑 876 点，实际占用按日志）
POINT_BUDGET = 1024

ROOT = rss.ROOT
RUN_DIR = os.path.join(ROOT, "asymmetric_run1")
os.makedirs(RUN_DIR, exist_ok=True)
rss.RUN_DIR = RUN_DIR
rss.LOG_PATH = os.path.join(RUN_DIR, "eval_log.jsonl")
rss.RESULT_PATH = os.path.join(RUN_DIR, "result.json")
LOG_PATH = rss.LOG_PATH

np.random.seed(20261115)

F = {"cost": 0, "carbon": 1, "peak": 2, "discomfort": 3,
     "degradation": 4, "ramping": 5, "resilience": 6}


def row(assign):
    v = np.zeros(7)
    for i, x in assign:
        v[i] += x
    v = np.clip(v, rss.EPS, None)
    return v / v.sum()


# 楼1/楼3 的 F6 族模板：ramping 主导，配 degradation / resilience / peak
B13_TEMPLATES = [
    # (ramping, degradation, resilience, peak)
    row([(F["ramping"], 0.71), (F["degradation"], 0.25), (F["resilience"], 0.02), (F["peak"], 0.02)]),
    row([(F["ramping"], 0.68), (F["degradation"], 0.30), (F["resilience"], 0.02)]),
    row([(F["ramping"], 0.75), (F["degradation"], 0.20), (F["resilience"], 0.03), (F["peak"], 0.02)]),
    row([(F["ramping"], 0.80), (F["degradation"], 0.15), (F["resilience"], 0.05)]),
]

# 楼2 的 cost/resilience 扫描网格
COST_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
RES_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
DEG_FIXED = 0.02


def b2_row(cost, res):
    """楼2：cost + resilience 扫描，ramping 填剩余，degradation 固定。"""
    ramp = max(rss.EPS, 1.0 - cost - res - DEG_FIXED - 4 * rss.EPS)
    return row([(F["cost"], cost), (F["resilience"], res),
                (F["ramping"], ramp), (F["degradation"], DEG_FIXED)])


def build_points():
    pts = []
    for t1 in B13_TEMPLATES:
        for t3 in B13_TEMPLATES:
            for c in COST_GRID:
                for r in RES_GRID:
                    w = rss.build_rows([t1, b2_row(c, r), t3])
                    pts.append((w, c, r))
    return pts


def key_of(w):
    return tuple(np.round(np.asarray(w, float).ravel(), 6))


def load_all_keys():
    keys = set()
    import csv
    for path in [os.path.join(ROOT, "landscape_bundle", "landscape_data.csv")]:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            h = next(rd)
            idx = [h.index(c) for c in h if c.startswith("w_b")]
            for row_csv in rd:
                try:
                    keys.add(tuple(np.round(
                        np.array([float(row_csv[i]) for i in idx]), 6)))
                except Exception:
                    continue
    for lg in [r"\random_search_run1\eval_log.jsonl",
               r"\structure_census_run1\eval_log.jsonl",
               r"\asymmetric_run1\eval_log.jsonl"]:
        p = ROOT + lg
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
                keys.add(key_of(r["weights"]))
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
                rec["_key"] = key_of(rec["weights"])
                recs.append(rec)
            except Exception:
                continue
    return recs


def main():
    t0 = time.time()
    print("=== 不对称探索开始 ===", flush=True)
    print("  点预算 %d（楼1 4模 × 楼3 4模 × 楼2 8×8 网格）" % POINT_BUDGET, flush=True)

    rss.cleanup_pyi()
    rss.ensure_solver()
    log = load_log()
    done = {r["_key"]: r["R"] for r in log}
    rss.EVAL_IDX[0] = max([int(r.get("eval_id", USED_BEFORE)) for r in log] + [USED_BEFORE]) + 1
    existing = load_all_keys()

    all_pts = build_points()
    print("  候选点 %d，去重后评估中..." % len(all_pts), flush=True)

    pending_W = []
    skipped = 0
    for w, c, r in all_pts:
        kk = key_of(w)
        if kk in existing:
            skipped += 1
            continue
        if kk in done:
            continue
        pending_W.append(w)

    print("  待评估 %d，跳过已有 %d" % (len(pending_W), skipped), flush=True)

    ev = rss.ParallelEval() if pending_W else None
    try:
        if pending_W:
            Rs = ev.evaluate(pending_W, label="ASYMM", print_every=50, log_phase="ASYMM")
            for w, rv in zip(pending_W, Rs):
                done[key_of(w)] = float(rv)

        # 汇总
        results = []
        for w, c, r in all_pts:
            kk = key_of(w)
            if kk in done:
                results.append((done[kk], w, c, r))
        results.sort(key=lambda x: x[0])

        print("\n=== top 20 不对称配置 ===", flush=True)
        for rv, w, c, r in results[:20]:
            rows = w.reshape(3, 7)
            dom = tuple(int(np.argmax(x)) for x in rows)
            print("  R=%.4f  b2_cost=%.2f b2_res=%.2f  dom=%s"
                  % (rv, c, r, dom), flush=True)

        bestR, bestW = results[0][0], results[0][1]
        print("\n全程最优 R=%.6f" % bestR, flush=True)
        W = np.asarray(bestW).reshape(3, 7)
        for i, row in enumerate(W):
            print("  楼%d: %s  dom=F%d" % (i + 1, np.round(row, 4).tolist(), np.argmax(row) + 1),
                  flush=True)

        # 按楼2 的 (cost, res) 分组看最优
        best_by_grid = {}
        for rv, w, c, r in results:
            best_by_grid[(c, r)] = min(best_by_grid.get((c, r), 1e9), rv)
        print("\n=== 楼2 cost/res 网格最优 R ===", flush=True)
        header = "       " + " ".join("r=%.2f" % r for r in RES_GRID)
        print(header, flush=True)
        for c in COST_GRID:
            line = "c=%.2f " % c + " ".join(
                "%.1f" % best_by_grid.get((c, r), 0) for r in RES_GRID)
            print(line, flush=True)

        result = {
            "params": {"POINT_BUDGET": POINT_BUDGET, "B13_TEMPLATES": len(B13_TEMPLATES),
                       "COST_GRID": COST_GRID, "RES_GRID": RES_GRID, "DEG_FIXED": DEG_FIXED,
                       "SEED": 20261115},
            "best_R": float(bestR),
            "best_weights": np.asarray(bestW, float).ravel().tolist(),
            "total_evals": len(done),
            "grid_best": {str(k): float(v) for k, v in best_by_grid.items()},
            "elapsed_seconds": time.time() - t0,
            "eval_log": LOG_PATH,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(rss.RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n结果已落盘: %s" % rss.RESULT_PATH, flush=True)

    finally:
        if ev is not None:
            ev.close()


if __name__ == "__main__":
    main()
