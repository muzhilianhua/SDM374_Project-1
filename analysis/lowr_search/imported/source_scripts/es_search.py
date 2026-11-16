# -*- coding: utf-8 -*-
"""
es_search.py — 第二轮：结构化 (1+λ) 进化策略，预算 3000

为什么用它：
  前一阶段证明最优区是"窄脊"——单坐标移动 0.01 就让 R 恶化，
  坐标扫描会在脊上左右乱撞。这里改成在完整 21 维单纯形上做
  稀疏乘性扰动（每次只动约一半坐标），按 (1+λ) 精英策略接受，
  sigma 自适应：连续改进则放大、停滞则收缩，长期停滞则换起点重启。

  另一件坐标扫描做不到的事：直接把"哪栋楼做峰值专精"当结构变量，
  起点集里包含楼1/楼2/楼3 三种专精排列。

预算分配：
  ES 主体          2600
  收尾细扫         400（对当前最优逐坐标做 ±0.0005/±0.001/±0.002 试探）
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_simplex_search as rss

BUDGET = 3000
ES_BUDGET = 2600
SEED = 20261117
LAMBDA = 16      # 需 >8 才会被 ParallelEval 拆成 4 路并行

ROOT = rss.ROOT
RUN_DIR = os.path.join(ROOT, "es_run1")
os.makedirs(RUN_DIR, exist_ok=True)
rss.RUN_DIR = RUN_DIR
rss.LOG_PATH = os.path.join(RUN_DIR, "eval_log.jsonl")
rss.RESULT_PATH = os.path.join(RUN_DIR, "result.json")
LOG_PATH = rss.LOG_PATH

np.random.seed(SEED)
EPS = rss.EPS
N_B, N_F = 3, 7

UPLOADED_W = np.array([
    0.0008566065989881307, 0.05983697050911368, 0.002008012420456479,
    0.00011005730318200956, 0.2521365414897022, 0.6850517411646035, 7.051395392316168e-08,
    0.009807287938449276, 0.0016442747505591045, 0.9115182931639005,
    5.178291422368062e-08, 0.002296759075126124, 0.015295761139369193, 0.05943757214968143,
    3.125691451618951e-08, 2.6870414717718473e-08, 0.02410886295070521,
    0.0002742198820332293, 0.23894042155412554, 0.7141073154834251, 0.022569122002381734,
])


def key_of(w):
    return tuple(np.round(np.asarray(w, float).ravel(), 6))


def norm_rows(W):
    W = np.clip(np.asarray(W, float).reshape(N_B, N_F), EPS, None)
    return (W / W.sum(axis=1, keepdims=True)).ravel()


def mutate(w, sigma, sparse=0.5, jump=False):
    """稀疏乘性扰动：只动一部分坐标，保证仍在单纯形内部。"""
    W = np.asarray(w, float).reshape(N_B, N_F)
    Z = np.random.randn(N_B, N_F)
    mask = np.random.rand(N_B, N_F) < sparse
    if not mask.any():
        mask[np.random.randint(N_B), np.random.randint(N_F)] = True
    s = sigma * (3.0 if jump else 1.0)
    return norm_rows(W * np.exp(s * Z * mask))


def load_known():
    """汇总所有已评估点，用于起点选择与去重。"""
    pts, keys = [], set()

    def add(w, r):
        w = np.asarray(w, float).ravel()
        if w.size != N_B * N_F:
            return
        k = key_of(w)
        if k in keys:
            return
        keys.add(k)
        pts.append((float(r), w))

    import csv
    csv_path = os.path.join(ROOT, "landscape_bundle", "landscape_data.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            h = next(rd)
            idx = [h.index(c) for c in h if c.startswith("w_b")]
            ri = h.index("R")
            for row in rd:
                try:
                    add([float(row[i]) for i in idx], float(row[ri]))
                except Exception:
                    continue
    for sub in ("random_search_run1", "structure_census_run1",
                "asymmetric_run1", "ridge_run1", "es_run1"):
        p = os.path.join(ROOT, sub, "eval_log.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            try:
                rec = json.loads(line)
                add(rec["weights"], rec["R"])
            except Exception:
                continue
    return pts, keys


class Budget:
    def __init__(self, budget, known_keys, replay):
        self.left = budget
        self.spent = 0
        self.cache = {}
        self.rows = []
        self.ev = None
        for line in replay:
            try:
                rec = json.loads(line)
                self.cache[key_of(rec["weights"])] = float(rec["R"])
                self.rows.append((float(rec["R"]), np.asarray(rec["weights"], float)))
                self.spent += 1
            except Exception:
                continue
        self.left = max(0, budget - self.spent)
        self.known = known_keys

    def _ensure(self):
        if self.ev is None:
            self.ev = rss.ParallelEval()
        return self.ev

    def run(self, Wlist, phase):
        out = [None] * len(Wlist)
        newW, newI = [], []
        for i, w in enumerate(Wlist):
            k = key_of(w)
            if k in self.cache:
                out[i] = self.cache[k]
            elif k in self.known:
                self.known.add(k)      # 已知点不消耗预算，也不重复记录
                out[i] = None
            elif self.left > 0:
                newW.append(w)
                newI.append(i)
                self.left -= 1
        if not newW:
            return out
        Rs = self._ensure().evaluate(newW, label=phase, print_every=80, log_phase=phase)
        for w, i, r in zip(newW, newI, Rs):
            self.cache[key_of(w)] = float(r)
            self.rows.append((float(r), np.asarray(w, float)))
            out[i] = float(r)
        self.spent += len(newW)
        return out

    def best(self):
        return min(self.rows, key=lambda x: x[0]) if self.rows else None

    def close(self):
        if self.ev is not None:
            self.ev.close()


def build_seeds(known_pts):
    """起点：历史最优若干个 + 上传点的三种专精排列。"""
    seeds = []
    seen = set()

    def push(w):
        k = key_of(w)
        if k in seen:
            return
        seen.add(k)
        seeds.append(np.asarray(w, float).ravel())

    ranked = sorted(known_pts, key=lambda x: x[0])
    for r, w in ranked[:8]:
        push(w)

    W = UPLOADED_W.reshape(N_B, N_F)
    for perm in ([0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]):
        push(norm_rows(W[list(perm)]))
    return seeds


def main():
    t0 = time.time()
    dry = "--dry" in sys.argv
    print("=== 第二轮 ES 搜索开始%s ===" % ("（DRY）" if dry else ""), flush=True)

    known_pts, known_keys = load_known()
    print("  已知唯一点 %d" % len(known_pts), flush=True)
    if known_pts:
        br = min(known_pts, key=lambda x: x[0])
        print("  历史最优 R=%.4f" % br[0], flush=True)

    replay = []
    if not dry and os.path.exists(LOG_PATH):
        replay = [ln for ln in open(LOG_PATH, encoding="utf-8") if ln.strip()]
        print("  续跑：本轮已有日志 %d 条" % len(replay), flush=True)

    if dry:
        bu = Budget(BUDGET, set(), [])
        opt = np.array(UPLOADED_W)
        opt[:7] *= 1.03
        opt = norm_rows(opt.reshape(N_B, N_F))

        class FakeEv:
            def evaluate(self, Wlist, label="", print_every=0, log_phase=None):
                return [float(1e3 * np.abs(np.asarray(w) - opt).sum()) + 100.0 for w in Wlist]

            def close(self):
                pass

        bu._ensure = lambda: FakeEv()
    else:
        rss.cleanup_pyi()
        rss.ensure_solver()
        bu = Budget(BUDGET, known_keys, replay)

    seeds = build_seeds(known_pts)
    print("  起点数 %d" % len(seeds), flush=True)

    try:
        # 起点评估（只消耗少量预算）
        Rs = bu.run(seeds, "ES_seed")
        ranked = [(r, w) for r, w in zip(Rs, seeds) if r is not None]
        if not ranked:
            b = bu.best()
            ranked = [(b[0], b[1])]
        ranked.sort(key=lambda x: x[0])
        cur_R, cur_W = ranked[0]
        print("  起点最优 R=%.4f" % cur_R, flush=True)

        best_R, best_W = cur_R, cur_W.copy()
        sigma, stall, gen, seed_i = 0.020, 0, 0, 0

        # ---------- ES 主体 ----------
        while bu.left > 0 and bu.spent < ES_BUDGET:
            gen += 1
            jump = (gen % 17 == 0)
            cands = [mutate(cur_W, sigma, sparse=np.random.choice([0.3, 0.5, 0.7]), jump=jump)
                     for _ in range(LAMBDA)]
            Rs = bu.run(cands, "ES")
            vals = [(r, w) for r, w in zip(Rs, cands) if r is not None]
            if not vals:
                if bu.left <= 0:
                    break
                continue
            r1, w1 = min(vals, key=lambda x: x[0])
            if r1 < cur_R:
                cur_R, cur_W = r1, w1
                stall = 0
                sigma = min(0.06, sigma * 1.06)
                if r1 < best_R:
                    best_R, best_W = r1, w1.copy()
                    print("  [gen %d] 新最优 R=%.4f sigma=%.4f" % (gen, best_R, sigma), flush=True)
            else:
                stall += 1
                sigma = max(0.0015, sigma * 0.97)

            if stall >= 45:
                # 换起点重启：优先用还没深挖过的历史最优
                seed_i = (seed_i + 1) % max(1, len(ranked))
                if seed_i == 0:
                    cur_W, cur_R = best_W.copy(), best_R
                else:
                    cur_R, cur_W = ranked[seed_i][0], ranked[seed_i][1].copy()
                sigma, stall = 0.030, 0
                print("  [gen %d] 停滞重启 @ R=%.4f" % (gen, cur_R), flush=True)

            if gen % 20 == 0:
                print("  gen %d  spent %d  best %.4f  cur %.4f  sigma %.4f"
                      % (gen, bu.spent, best_R, cur_R, sigma), flush=True)

        # ---------- 收尾细扫 ----------
        print("\n=== 收尾细扫（剩余 %d）===" % bu.left, flush=True)
        wbest = best_W.copy()
        for step in (0.0005, 0.001, 0.002):
            base = wbest.reshape(N_B, N_F)
            cands = []
            for b in range(N_B):
                for f in range(N_F):
                    for s in (+step, -step):
                        Wc = base.copy()
                        Wc[b, f] = max(EPS, Wc[b, f] + s)
                        cands.append(norm_rows(Wc))
            Rs = bu.run(cands, "ES_polish_%g" % step)
            vals = [(r, w) for r, w in zip(Rs, cands) if r is not None]
            if vals:
                r1, w1 = min(vals, key=lambda x: x[0])
                if r1 < best_R:
                    best_R, best_W = r1, w1.copy()
                    print("  细扫 %.4f → R=%.4f" % (step, best_R), flush=True)

        # ---------- 汇总 ----------
        Wb = np.asarray(best_W).reshape(N_B, N_F)
        names = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]
        print("\n" + "=" * 58, flush=True)
        print("实际评估 %d / %d，用时 %.1f 分钟" % (bu.spent, BUDGET, (time.time() - t0) / 60), flush=True)
        print("第二轮最优 R=%.6f" % best_R, flush=True)
        for i, row in enumerate(Wb):
            print("  楼%d: %s  dom=%s" % (i + 1, np.round(row, 4).tolist(), names[int(np.argmax(row))]),
                  flush=True)
        print("对照: 上传点 103.461 | landscape 104.852 | 目标 96", flush=True)

        if not dry:
            with open(rss.RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "params": {"BUDGET": BUDGET, "ES_BUDGET": ES_BUDGET, "SEED": SEED,
                               "LAMBDA": LAMBDA, "seed_mode": "sparse_multiplicative"},
                    "best_R": float(best_R),
                    "best_weights": [float(x) for x in best_W],
                    "spent": bu.spent,
                    "elapsed_seconds": time.time() - t0,
                    "eval_log": LOG_PATH,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }, f, ensure_ascii=False, indent=2)
            print("结果已落盘: %s" % rss.RESULT_PATH, flush=True)
    finally:
        bu.close()


if __name__ == "__main__":
    main()
