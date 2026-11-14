"""E8: what the minor weights do. One building varies, the other two stay at the incumbent.

A minor weight is written m_k = w_k / (degradation + ramping); r and the other m stay fixed while one varies.

  loo    start from the incumbent block and vary one minor weight from 0 to 1 (log grid, denser near its
         incumbent value): what does changing this weight cost, with the others in place?
  add    start from the block with no minor weights (same r) and add one minor weight alone:
         how much of the minor weights' value can a single objective recover?
  pair   10 x 10 grid over the building's two most valuable minor weights (0 and 1/8 .. 8 times the
         incumbent values): do they substitute for each other or do they need each other?
"""
import json
import time

import numpy as np

from common import MINOR, NAMES, block, to_ratio
from pool import SolverPool

state = json.load(open("fms_state_run4.json", encoding="utf-8"))
inc = np.array(state["inc"]); inc = inc / inc.sum(1, keepdims=True); inc_R = state["inc_R"]
LOG = 10 ** np.linspace(-4, 0, 25)
NEAR = np.array([0.7, 0.85, 0.95, 1.05, 1.2, 1.5])
FACT = np.array([0, 1 / 8, 1 / 4, 1 / 2, 2 ** -0.5, 1, 2 ** 0.5, 2, 4, 8])
PAIRS = {0: (1, 3), 1: (4, 1), 2: (2, 4)}            # positions in the m vector: cost, carbon, peak, discomfort, resilience
pool = SolverPool(n_workers=10)
out = {"incumbent_R": inc_R, "minor_names": [NAMES[k] for k in MINOR], "loo": [], "add": [], "pair": []}
t0 = time.time()

X, meta = [], []
for b in range(3):
    r0, m0 = to_ratio(inc[b])
    for j in range(5):
        grid = np.concatenate([[0.0], LOG, m0[j] * NEAR if m0[j] > 0 else []])
        for v in np.unique(grid):
            m = m0.copy(); m[j] = v
            W = inc.copy(); W[b] = block(r0, m); X.append(W.ravel()); meta.append((b, j, float(v)))
y = pool.evaluate(np.array(X), tag="E8_loo") - inc_R
print(f"[loo] {len(X)} points, {pool.n_solver_evals} new evaluations so far, {time.time() - t0:.0f}s")
for (b, j, v), d in zip(meta, y):
    out["loo"].append({"building": b + 1, "minor": j, "m": v, "delta": float(d)})
json.dump(out, open("E8_dose.json", "w"), indent=1)

X, meta = [], []
for b in range(3):
    r0, m0 = to_ratio(inc[b])
    for j in range(5):
        for v in LOG:
            m = np.zeros(5); m[j] = v
            W = inc.copy(); W[b] = block(r0, m); X.append(W.ravel()); meta.append((b, j, float(v)))
y = pool.evaluate(np.array(X), tag="E8_add") - inc_R
print(f"[add] {len(X)} points, {pool.n_solver_evals} new evaluations so far, {time.time() - t0:.0f}s")
for (b, j, v), d in zip(meta, y):
    out["add"].append({"building": b + 1, "minor": j, "m": v, "delta": float(d)})
json.dump(out, open("E8_dose.json", "w"), indent=1)

X, meta = [], []
for b in range(3):
    r0, m0 = to_ratio(inc[b])
    j1, j2 = PAIRS[b]
    for f1 in FACT:
        for f2 in FACT:
            m = m0.copy(); m[j1] = m0[j1] * f1; m[j2] = m0[j2] * f2
            W = inc.copy(); W[b] = block(r0, m); X.append(W.ravel()); meta.append((b, j1, j2, float(f1), float(f2)))
y = pool.evaluate(np.array(X), tag="E8_pair") - inc_R
print(f"[pair] {len(X)} points, {pool.n_solver_evals} new evaluations in total, {time.time() - t0:.0f}s")
for (b, j1, j2, f1, f2), d in zip(meta, y):
    out["pair"].append({"building": b + 1, "minor1": j1, "minor2": j2, "f1": f1, "f2": f2, "delta": float(d)})
json.dump(out, open("E8_dose.json", "w"), indent=1)
best = min(min(e["delta"] for e in out[k]) for k in ("loo", "add", "pair"))
print(f"lowest delta in E8: {best:+.4f}")
