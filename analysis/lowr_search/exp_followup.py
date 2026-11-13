"""E10: follow up the points below the incumbent that E8 turned up.

E8 found that a tiny weight on an objective the incumbent does not use can lower R: cost in building 1
(m about 3e-4 .. 2e-3), a slightly larger discomfort weight in building 2, discomfort in building 3.
At this size a weight only breaks near-ties, so R moves in a few wide steps along the weight.

  step 1  fine scans of those three weights around the improving range
  step 2  per building the best block (middle of its plateau); all 2 x 2 x 2 combinations, measured
  step 3  from the best combination: every still-unused objective of every building at a tiny weight
  step 4  combine again and report
"""
import itertools
import json
import time

import numpy as np

from common import MINOR, NAMES, block, to_ratio
from pool import SolverPool

MN = [NAMES[k] for k in MINOR]
state = json.load(open("fms_state_run4.json", encoding="utf-8"))
inc = np.array(state["inc"]); inc = inc / inc.sum(1, keepdims=True); inc_R = state["inc_R"]
pool = SolverPool(n_workers=10)
t0 = time.time()
np.set_printoptions(precision=6, suppress=True, linewidth=200)


def scan(base, b, j, grid, tag):
    """Vary minor weight j of building b over grid, everything else as in base. Returns deltas vs inc_R."""
    r0, m0 = to_ratio(base[b])
    X = []
    for v in grid:
        m = m0.copy(); m[j] = v
        W = base.copy(); W[b] = block(r0, m); X.append(W.ravel())
    return pool.evaluate(np.array(X), tag=tag) - inc_R, np.array(X).reshape(-1, 3, 7)


def best_of(d, X):
    """Lowest value; among ties the middle point of the plateau."""
    ties = np.where(d < d.min() + 1e-9)[0]
    i = int(ties[len(ties) // 2])
    return float(d[i]), X[i], len(ties)


def combine(base, cands, tag):
    """cands[b]: alternative block for building b or None. Measures every combination, returns the best."""
    opts = [[base[b]] + ([cands[b]] if cands[b] is not None else []) for b in range(3)]
    combos = list(itertools.product(*[range(len(o)) for o in opts]))
    X = np.array([np.concatenate([opts[b][c[b]] for b in range(3)]) for c in combos])
    y = pool.evaluate(X, tag=tag) - inc_R
    for c, v in zip(combos, y):
        print(f"      {''.join('new ' if k else 'old ' for k in c)} {v:+.4f}")
    i = int(np.argmin(y))
    return X[i].reshape(3, 7), float(y[i])


print(f"baseline {inc_R:.6f}\nstep 1: fine scans")
plan = [(0, 0, 10 ** np.linspace(np.log10(2e-4), np.log10(3.2e-3), 61)),
        (1, 3, 10 ** np.linspace(np.log10(2e-4), np.log10(5e-4), 25)),
        (2, 3, 10 ** np.linspace(np.log10(2.2e-4), np.log10(4.6e-4), 17))]
cands = [None, None, None]
out = {"baseline_R": inc_R, "scans": []}
for b, j, grid in plan:
    d, X = scan(inc, b, j, grid, "E10_scan")
    v, Xb, n = best_of(d, X)
    steps = [(float(grid[0]), float(d[0]))] + [(float(grid[i]), float(d[i])) for i in range(1, len(d)) if abs(d[i] - d[i - 1]) > 1e-9]
    print(f"  b{b + 1} {MN[j]}: best {v:+.4f} on a plateau of {n} grid points; steps (m: delta) " + "  ".join(f"{m:.2e}:{x:+.4f}" for m, x in steps))
    out["scans"].append({"building": b + 1, "minor": MN[j], "m": grid.tolist(), "delta": d.tolist()})
    if v < -1e-9:
        cands[b] = Xb[b]
print("step 2: combinations of the per-building best blocks (b1 b2 b3)")
cur, cur_d = combine(inc, cands, "E10_combo")
print(f"  best combination {cur_d:+.4f}  ->  R = {inc_R + cur_d:.6f}")

print("step 3: unused objectives at a tiny weight, from the best combination")
TINY = 10 ** np.linspace(-4, np.log10(3.2e-3), 13)
cands = [None, None, None]
for b in range(3):
    m0 = to_ratio(cur[b])[1]
    bestv = cur_d
    for j in range(5):
        if m0[j] > 0:
            continue
        d, X = scan(cur, b, j, TINY, "E10_tiny")
        v, Xb, n = best_of(d, X)
        print(f"  b{b + 1} {MN[j]}: best {v - cur_d:+.4f} vs the combination (plateau of {n}); curve " + " ".join(f"{x - cur_d:+.3f}" for x in d))
        if v < bestv - 1e-9:
            bestv, cands[b] = v, Xb[b]
if any(c is not None for c in cands):
    print("step 4: combinations again")
    cur, cur_d = combine(cur, cands, "E10_combo2")
print(f"\nbest after follow-up: R = {inc_R + cur_d:.9f}  ({cur_d:+.4f} vs {inc_R:.4f});  {pool.n_solver_evals} new evaluations in {time.time() - t0:.0f}s")
print(cur)
out["best_R"] = inc_R + cur_d; out["best_W"] = cur.tolist()
json.dump(out, open("E10_followup.json", "w"), indent=1)
