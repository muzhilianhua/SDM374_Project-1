# -*- coding: utf-8 -*-
"""
ridge_search.py — 沿 uploaded best_weights.json 所在脊线定向搜索（预算 1000）

起点结构（R=103.4609，真实求解器已复核）：
  楼1  carbon 0.060 / degradation 0.252 / ramping 0.685
  楼2  peak   0.912 / resilience 0.059 / ramping 0.015
  楼3  peak   0.024 / resilience 0.023 / degradation 0.239 / ramping 0.714

关键线索：最近邻与该点只差楼1 carbon（0 vs 0.06），R 却差 4~14，
说明低 R 是一条窄脊，楼1 carbon 是敏感轴。

7 个自由参数：
  b1.carbon, b1.degradation, b2.peak, b2.resilience,
  b3.resilience, b3.peak, b3.degradation
（每栋楼剩余质量落在 ramping；其余特征保留种子点的小背景值）

搜索：
  A 三轮坐标扫描（粗→细）
  B 最敏感 3 参数的两两网格耦合
  C top-4 起点做细步多起点扫描
  D 剩余预算做随机扰动 + 贪心接受
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_simplex_search as rss

BUDGET = 1000
SEED = 20261116

ROOT = rss.ROOT
RUN_DIR = os.path.join(ROOT, "ridge_run1")
os.makedirs(RUN_DIR, exist_ok=True)
rss.RUN_DIR = RUN_DIR
rss.LOG_PATH = os.path.join(RUN_DIR, "eval_log.jsonl")
rss.RESULT_PATH = os.path.join(RUN_DIR, "result.json")
LOG_PATH = rss.LOG_PATH

np.random.seed(SEED)
EPS = rss.EPS

FEATS = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]

SEED_W = np.array([
    [0.0008566065989881307, 0.05983697050911368, 0.002008012420456479,
     0.00011005730318200956, 0.2521365414897022, 0.6850517411646035, 7.051395392316168e-08],
    [0.009807287938449276, 0.0016442747505591045, 0.9115182931639005,
     5.178291422368062e-08, 0.002296759075126124, 0.015295761139369193, 0.05943757214968143],
    [3.125691451618951e-08, 2.6870414717718473e-08, 0.02410886295070521,
     0.0002742198820332293, 0.23894042155412554, 0.7141073154834251, 0.022569122002381734],
])

PARAM_NAMES = ["b1.carbon", "b1.degradation", "b2.peak", "b2.resilience",
               "b3.resilience", "b3.peak", "b3.degradation"]
SEED_P = np.array([SEED_W[0, 1], SEED_W[0, 4], SEED_W[1, 2], SEED_W[1, 6],
                   SEED_W[2, 6], SEED_W[2, 2], SEED_W[2, 4]])
BOUNDS = [(0.001, 0.20), (0.05, 0.45), (0.75, 0.999), (0.001, 0.15),
          (0.001, 0.10), (0.001, 0.10), (0.05, 0.45)]

# 楼1/楼2/楼3 的"残余特征"下标（保持种子值，不参与参数化）
ROW_SPEC = [
    (0, {1: 0, 4: 1}, 5),   # 楼1: 分配 carbon, degradation；余量给 ramping
    (1, {2: 2, 6: 3}, 5),   # 楼2: peak, resilience
    (2, {6: 4, 2: 5, 4: 6}, 5),  # 楼3: resilience, peak, degradation
]


def make_point(p):
    p = np.asarray(p, float)
    rows = []
    for b, assign, resid in ROW_SPEC:
        row = SEED_W[b].copy()
        for fi, pi in assign.items():
            row[fi] = max(p[pi], EPS)
        others = sum(row[j] for j in range(7) if j not in assign and j != resid)
        row[resid] = 1.0 - others - sum(row[fi] for fi in assign)
        row = np.clip(row, EPS, None)
        rows.append(row / row.sum())
    return rss.build_rows(rows)


def key_of(w):
    return tuple(np.round(np.asarray(w, float).ravel(), 6))


def params_of(w):
    """从权重反推 7 个参数，用于续跑时重建已评估记录。"""
    W = np.asarray(w, float).reshape(3, 7)
    return np.array([W[0, 1], W[0, 4], W[1, 2], W[1, 6], W[2, 6], W[2, 2], W[2, 4]])


class Budget:
    """统一预算 + 去重 + 增量落盘。"""

    def __init__(self, budget, rediscover=None):
        self.cache = {}
        self.rows = []            # (p, R) 所有已评估
        self.ev = None
        self.spent = 0
        for line in (rediscover or []):
            try:
                rec = json.loads(line)
                w = np.asarray(rec["weights"], float)
                self.cache[key_of(w)] = float(rec["R"])
                self.rows.append((params_of(w), float(rec["R"])))
                self.spent += 1
            except Exception:
                continue
        # 续跑时预算要扣除已真实消耗的点，避免跨重启超支
        self.left = max(0, budget - self.spent)

    def best(self):
        if not self.rows:
            return None
        return min(self.rows, key=lambda x: x[1])

    def _ensure(self):
        if self.ev is None:
            self.ev = rss.ParallelEval()
        return self.ev

    def run(self, plist, phase):
        """评估参数点，返回与 plist 对齐的 R 列表（预算不足的为 None）。"""
        Ws = [make_point(p) for p in plist]
        out = [None] * len(plist)
        newP, newW, newI = [], [], []
        for i, (p, w) in enumerate(zip(plist, Ws)):
            k = key_of(w)
            if k in self.cache:
                out[i] = self.cache[k]
            elif self.left > 0:
                newP.append(p)
                newW.append(w)
                newI.append(i)
        if not newW:
            return out
        if len(newW) > self.left:
            keep = self.left
            newP, newW, newI = newP[:keep], newW[:keep], newI[:keep]
        Rs = self._ensure().evaluate(newW, label=phase, print_every=60, log_phase=phase)
        self.left -= len(newW)
        self.spent += len(newW)
        for p, w, i, r in zip(newP, newW, newI, Rs):
            self.cache[key_of(w)] = float(r)
            self.rows.append((np.asarray(p, float), float(r)))
            out[i] = float(r)
        return out

    def close(self):
        if self.ev is not None:
            self.ev.close()


def sweep_round(bu, p0, r0, scales, tag):
    """一轮坐标扫描：每个参数取 9 个值，逐个采纳改进。"""
    p, bestR = np.array(p0, float), float(r0)
    for j in range(7):
        lo, hi = BOUNDS[j]
        vals = np.linspace(max(lo, p[j] - scales[j]), min(hi, p[j] + scales[j]), 9)
        cands = []
        for v in vals:
            if abs(v - p[j]) < 1e-9:
                continue
            q = p.copy()
            q[j] = v
            cands.append(q)
        if not cands:
            continue
        Rs = bu.run(cands, "%s_%s" % (tag, PARAM_NAMES[j]))
        valid = [(r, q) for q, r in zip(cands, Rs) if r is not None]
        if not valid:
            continue
        r, q = min(valid, key=lambda x: x[0])
        if r < bestR:
            bestR, p = r, q
    return p, bestR


def bucket_mean(Rs, n):
    """按 n 个值分桶，返回每桶的最小 R，用于衡量参数敏感度。"""
    out = []
    step = max(1, len(Rs) // n)
    for i in range(0, len(Rs), step):
        chunk = [x for x in Rs[i:i + step] if x is not None]
        if chunk:
            out.append(min(chunk))
    return out


def main():
    t0 = time.time()
    dry = "--dry" in sys.argv
    print("=== 脊线定向搜索开始%s ===" % ("（DRY 模式）" if dry else ""), flush=True)
    print("  预算 %d，起点参数 %s" % (BUDGET, np.round(SEED_P, 4).tolist()), flush=True)

    seed_w = make_point(SEED_P)
    print("  起点 R=103.460860（已由真实求解器复核）", flush=True)

    replay = []
    if not dry and os.path.exists(LOG_PATH):
        replay = [ln for ln in open(LOG_PATH, encoding="utf-8") if ln.strip()]
        print("  续跑：已有日志 %d 条" % len(replay), flush=True)

    if dry:
        bu = Budget(BUDGET)
        state = {"calls": 0}

        class FakeEv:
            def evaluate(self, Wlist, label="", print_every=0, log_phase=None):
                state["calls"] += len(Wlist)
                out = []
                for w in Wlist:
                    d = np.abs(np.asarray(w) - make_point(SEED_P)).max()
                    out.append(103.46 + 40.0 * d + float(np.random.rand() * 0.02))
                return out

            def close(self):
                pass

        bu._ensure = lambda: FakeEv()
    else:
        rss.cleanup_pyi()
        rss.ensure_solver()
        bu = Budget(BUDGET, rediscover=replay)

    try:
        # ---------- A: 三轮坐标扫描 ----------
        print("\n=== A 三轮坐标扫描 ===", flush=True)
        base = [0.06, 0.08, 0.09, 0.06, 0.04, 0.03, 0.08]
        p, r = SEED_P.copy(), 103.46086008194834
        prev = bu.best()
        if prev is not None and prev[1] < r:
            p, r = prev[0].copy(), prev[1]
            print("  续跑：从上轮最优点 R=%.4f 继续" % r, flush=True)
        stageA = []
        for k, frac in enumerate([1.0, 0.5, 0.25]):
            sc = [b * frac for b in base]
            Rs = bu.run([p], "A_probe_%d" % k)
            if Rs[0] is not None:
                r = min(r, Rs[0])
            p, r = sweep_round(bu, p, r, sc, "A_r%d" % (k + 1))
            stageA.append((p.copy(), r))
            print("  轮%d → R=%.4f  p=%s" % (k + 1, r, np.round(p, 4).tolist()), flush=True)

        # ---------- B: 最敏感参数两两网格 ----------
        print("\n=== B 敏感参数两两耦合 ===", flush=True)
        sens = []
        for j in range(7):
            lo, hi = BOUNDS[j]
            vals = np.linspace(max(lo, p[j] - 2 * base[j]), min(hi, p[j] + 2 * base[j]), 7)
            cands = []
            for v in vals:
                q = p.copy()
                q[j] = v
                cands.append(q)
            Rs = bu.run(cands, "B_sens_%s" % PARAM_NAMES[j])
            valid = [x for x in Rs if x is not None]
            if valid:
                sens.append((max(valid) - min(valid), j))
        sens.sort(reverse=True)
        top = [j for _, j in sens[:3]]
        print("  最敏感 3 参数: %s" % [PARAM_NAMES[j] for j in top], flush=True)

        best_p, bestR = p.copy(), r
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                ja, jb = top[a], top[b]
                ga = np.linspace(max(BOUNDS[ja][0], best_p[ja] - 2 * base[ja]),
                                 min(BOUNDS[ja][1], best_p[ja] + 2 * base[ja]), 7)
                gb = np.linspace(max(BOUNDS[jb][0], best_p[jb] - 2 * base[jb]),
                                 min(BOUNDS[jb][1], best_p[jb] + 2 * base[jb]), 7)
                cands = []
                for va in ga:
                    for vb in gb:
                        q = best_p.copy()
                        q[ja], q[jb] = va, vb
                        cands.append(q)
                Rs = bu.run(cands, "B_grid_%s_x_%s" % (PARAM_NAMES[ja], PARAM_NAMES[jb]))
                valid = [(x, q) for q, x in zip(cands, Rs) if x is not None]
                if valid:
                    rr, qq = min(valid, key=lambda x: x[0])
                    print("    %s × %s → R=%.4f" % (PARAM_NAMES[ja], PARAM_NAMES[jb], rr), flush=True)
                    if rr < bestR:
                        bestR, best_p = rr, qq

        # ---------- C: top-4 多起点细步 ----------
        print("\n=== C top 起点细步扫描 ===", flush=True)
        ranked = sorted(bu.rows, key=lambda x: x[1])
        starts = []
        for q, rr in ranked:
            if all(np.abs(q - s).max() > 0.03 for s in starts):
                starts.append(q.copy())
            if len(starts) >= 4:
                break
        for si, q0 in enumerate(starts):
            r0 = min(rr for q, rr in ranked if np.abs(q - q0).max() < 1e-9)
            sc = [b * 0.12 for b in base]
            p1, r1 = sweep_round(bu, q0, r0, sc, "C_s%d" % si)
            print("  起点%d R=%.4f → %.4f" % (si + 1, r0, r1), flush=True)
            if r1 < bestR:
                bestR, best_p = r1, p1

        # ---------- D: 随机扰动 + 贪心 ----------
        print("\n=== D 随机扰动（剩余预算 %d）===" % bu.left, flush=True)
        sig = np.array([(hi - lo) * 0.06 for lo, hi in BOUNDS])
        while bu.left > 0:
            scale = float(np.random.choice([0.5, 1.0, 2.0], p=[0.6, 0.3, 0.1]))
            q = best_p + np.random.randn(7) * sig * scale
            for j in range(7):
                q[j] = min(max(q[j], BOUNDS[j][0]), BOUNDS[j][1])
            rs = bu.run([q], "D_pert")
            if rs[0] is not None and rs[0] < bestR:
                bestR, best_p = rs[0], q
                print("    改进 R=%.4f  p=%s" % (bestR, np.round(best_p, 4).tolist()), flush=True)

        # ---------- 汇总 ----------
        best_w = make_point(best_p)
        print("\n" + "=" * 60, flush=True)
        print("=== 脊线搜索完毕 ===", flush=True)
        print("  实际评估 %d / %d" % (bu.spent, BUDGET), flush=True)
        print("  最优 R=%.6f" % bestR, flush=True)
        print("  参数: %s" % dict(zip(PARAM_NAMES, np.round(best_p, 4).tolist())), flush=True)
        rows = best_w.reshape(3, 7)
        for i, row in enumerate(rows):
            print("    楼%d: %s  dom=F%d" % (i + 1, np.round(row, 4).tolist(), np.argmax(row) + 1),
                  flush=True)
        print("  对照: 起点 103.461 | landscape 全局 104.852 | 目标 96", flush=True)

        if not dry:
            result = {
                "params": {"BUDGET": BUDGET, "SEED": SEED, "SEED_P": SEED_P.tolist(),
                           "PARAM_NAMES": PARAM_NAMES, "BOUNDS": BOUNDS},
                "best_R": float(bestR),
                "best_params": dict(zip(PARAM_NAMES, [float(x) for x in best_p])),
                "best_weights": [float(x) for x in best_w],
                "spent": bu.spent,
                "elapsed_seconds": time.time() - t0,
                "eval_log": LOG_PATH,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with open(rss.RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print("结果已落盘: %s" % rss.RESULT_PATH, flush=True)

    finally:
        bu.close()


if __name__ == "__main__":
    main()
