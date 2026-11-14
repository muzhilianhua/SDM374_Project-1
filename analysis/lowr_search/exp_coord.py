"""E12: coordinate-descent rounds from the best point in the log.

Each round, for every building at once (the buildings are additive, so their scans are independent):
  - every minor weight m_k: a log grid 1e-6 .. 3e-3 (and 0), plus factors 0.6 .. 1.6 of its current value
  - r = degradation / (degradation + ramping): current r +- 0.01 in steps of 0.0005
Per building the two best single moves are kept; their combination is measured too. Then all
2 x 2 x 2 (best move or not, per building) combinations are measured and the best becomes the new start.
Stops after --rounds rounds or when a round brings nothing.

    python exp_coord.py [rounds]
"""
import itertools
import sys
import time

import numpy as np

from common import MINOR, NAMES, block, to_ratio
from pool import SolverPool, load_log
from common import write_json

MN = [NAMES[k] for k in MINOR]
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LOGG = np.concatenate([[0.0], 10 ** np.linspace(-6, np.log10(3e-3), 22)])
FACT = np.array([0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.45, 1.6])
DR = np.arange(-0.01, 0.01001, 0.0005)
W, R, _ = load_log()
i = int(np.argmin(R)); cur = W[i].reshape(3, 7).copy(); cur_R = float(R[i])
pool = SolverPool(n_workers=10)
t0 = time.time()
hist = {"start_R": cur_R, "rounds": []}
print(f"start R = {cur_R:.9f}", flush=True)

for rnd in range(ROUNDS):
    X, meta = [], []
    for b in range(3):
        r0, m0 = to_ratio(cur[b])
        for j in range(5):
            vals = set(LOGG.tolist())
            if m0[j] > 0:
                vals |= set((m0[j] * FACT).tolist())
            for v in sorted(vals):
                if abs(v - m0[j]) < 1e-15:
                    continue
                m = m0.copy(); m[j] = v
                Wn = cur.copy(); Wn[b] = block(r0, m); X.append(Wn.ravel()); meta.append((b, MN[j], v))
        for dr in DR:
            if abs(dr) < 1e-12:
                continue
            Wn = cur.copy(); Wn[b] = block(r0 + dr, m0); X.append(Wn.ravel()); meta.append((b, "r", r0 + dr))
    y = pool.evaluate(np.array(X), tag=f"E12_r{rnd}") - cur_R
    X = np.array(X).reshape(-1, 3, 7)
    print(f"round {rnd + 1}: {len(X)} scan points, {time.time() - t0:.0f}s; points below the start: {int((y < -1e-9).sum())}", flush=True)
    rec = {"scan_points": len(X), "moves": []}
    cand = [[cur[b]] for b in range(3)]
    for b in range(3):
        idx = [k for k, mt in enumerate(meta) if mt[0] == b]
        best_per_coord = {}
        for k in idx:
            c = meta[k][1]
            if c not in best_per_coord or y[k] < y[best_per_coord[c]]:
                best_per_coord[c] = k
        ranked = sorted(best_per_coord.values(), key=lambda k: y[k])
        print(f"  b{b + 1}: " + "  ".join(f"{meta[k][1]}={meta[k][2]:.3g}:{y[k]:+.4f}" for k in ranked), flush=True)
        good = [k for k in ranked if y[k] < -1e-9][:2]
        for k in good:
            cand[b].append(X[k][b]); rec["moves"].append({"building": b + 1, "coord": meta[k][1], "value": meta[k][2], "delta": float(y[k])})
        if len(good) == 2:                       # both moves together (they touch different coordinates)
            r_a, m_a = to_ratio(X[good[0]][b]); r_b, m_b = to_ratio(X[good[1]][b]); r00, m00 = to_ratio(cur[b])
            r_c = r_a if meta[good[0]][1] == "r" else (r_b if meta[good[1]][1] == "r" else r00)
            m_c = m00.copy()
            for k, mm in ((good[0], m_a), (good[1], m_b)):
                if meta[k][1] != "r":
                    j = MN.index(meta[k][1]); m_c[j] = mm[j]
            cand[b].append(block(r_c, m_c))
    if all(len(c) == 1 for c in cand):
        print("  no single move lowers R; stopping"); hist["rounds"].append(rec); break
    combos = list(itertools.product(*[range(len(c)) for c in cand]))
    Xc = np.array([np.concatenate([cand[b][c[b]] for b in range(3)]) for c in combos])
    yc = pool.evaluate(Xc, tag=f"E12_c{rnd}") - cur_R
    k = int(np.argmin(yc))
    print(f"  {len(combos)} combinations; best {yc[k]:+.4f} with choice {combos[k]}", flush=True)
    rec["combo_best"] = float(yc[k])
    if yc[k] < -1e-9:
        cur, cur_R = Xc[k].reshape(3, 7), cur_R + float(yc[k])
    rec["R_after"] = cur_R; hist["rounds"].append(rec)
    print(f"  -> R = {cur_R:.9f}", flush=True)
    write_json("E12_coord.json", hist | {"best_R": cur_R, "best_W": cur.tolist()}, indent=1)

np.set_printoptions(precision=6, suppress=True, linewidth=200)
print(f"\nfinal R = {cur_R:.9f}; {pool.n_solver_evals} new evaluations in {time.time() - t0:.0f}s"); print(cur)
write_json("E12_coord.json", hist | {"best_R": cur_R, "best_W": cur.tolist()}, indent=1)
