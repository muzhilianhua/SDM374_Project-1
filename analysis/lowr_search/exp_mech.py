"""E6: first mechanism probes (about 130 points). Four small sets, each a direct question to the evaluator:

  vertex   one objective only, (a) in all three buildings, (b) in one building with the others at the incumbent.
           Degradation-only keeps the battery idle, so it is the "no battery control" reference.
  knock    the incumbent with one minor weight removed; with all minor weights of one building removed;
           with all minor weights removed everywhere. What is each minor weight worth where it stands?
  face     one building on the pure degradation+ramping face at a grid of r, the others at the incumbent.
  stale    archived blocks whose *estimated* delta is negative but which were never measured against the
           current incumbent: one representative per distinct estimate.
"""
import json
import time

import numpy as np

from common import MINOR, NAMES
from pool import SolverPool

np.set_printoptions(precision=5, suppress=True, linewidth=200)
state = json.load(open("fms_state_run4.json", encoding="utf-8"))
inc = np.array(state["inc"]); inc = inc / inc.sum(1, keepdims=True); inc_R = state["inc_R"]
E = np.eye(7)
pool = SolverPool(n_workers=10)
t0 = time.time()
out = {"incumbent_R": inc_R}

# --- vertex -------------------------------------------------------------------------------------------
X = [np.tile(E[k], (3, 1)).ravel() for k in range(7)]
for b in range(3):
    for k in range(7):
        W = inc.copy(); W[b] = E[k]; X.append(W.ravel())
y = pool.evaluate(np.array(X), tag="E6_vertex")
print(f"incumbent {inc_R:.4f}\n\n[vertex] one objective only")
print("  objective      all three   | one building (others at the incumbent): delta vs incumbent  b1       b2       b3")
out["vertex_all"] = dict(zip(NAMES, y[:7].tolist()))
out["vertex_one"] = {NAMES[k]: (y[7 + np.arange(3) * 7 + k] - inc_R).tolist() for k in range(7)}
for k in range(7):
    d = y[7 + np.arange(3) * 7 + k] - inc_R
    print(f"  {NAMES[k]:12s}  {y[k]:9.3f}   |                                                      {d[0]:+8.3f} {d[1]:+8.3f} {d[2]:+8.3f}")

# --- knock-outs ---------------------------------------------------------------------------------------
X, lab = [], []
for b in range(3):
    for k in MINOR:
        if inc[b, k] > 0:
            W = inc.copy(); W[b, k] = 0; X.append(W.ravel()); lab.append((b, NAMES[k], float(inc[b, k])))
    W = inc.copy(); W[b, MINOR] = 0; X.append(W.ravel()); lab.append((b, "ALL minor", float(inc[b, MINOR].sum())))
W = inc.copy(); W[:, MINOR] = 0; X.append(W.ravel()); lab.append((-1, "ALL minor, all buildings", 0.0))
y = pool.evaluate(np.array(X), tag="E6_knock")
print("\n[knock] incumbent with a minor weight removed (the rest of the block keeps its ratios)")
out["knock"] = []
for (b, name, w), r in zip(lab, y):
    print(f"  {'b%d' % (b + 1) if b >= 0 else '  '} {name:26s} w = {w:.6f}   R = {r:.4f}  ({r - inc_R:+.4f})")
    out["knock"].append({"building": b + 1, "removed": name, "weight": w, "R": float(r), "delta": float(r - inc_R)})

# --- pure degradation + ramping face ------------------------------------------------------------------
GRID = [0.0, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.50, 0.70]
X = []
for b in range(3):
    for r in GRID:
        W = inc.copy(); W[b] = 0; W[b, 4], W[b, 5] = r, 1 - r; X.append(W.ravel())
y = pool.evaluate(np.array(X), tag="E6_face").reshape(3, len(GRID)) - inc_R
print("\n[face] one building with degradation+ramping only, r = degradation share; delta vs incumbent")
print("  r      " + "".join(f"{r:7.2f}" for r in GRID))
for b in range(3):
    print(f"  b{b + 1}     " + "".join(f"{v:+7.2f}" for v in y[b]))
out["face"] = {"r": GRID, "delta": y.tolist()}

# --- thresholds on the pure face ----------------------------------------------------------------------
# A two-objective reading of the controller (x_prev = 0, symmetric battery range) predicts that ramp
# cancelling switches off between r = 1/3 and r = 1/2 and that nothing depends on r above 1/2.
FINE = [0.30, 0.31, 0.32, 0.33, 0.3333, 0.3334, 0.335, 0.34, 0.36, 0.38, 0.42, 0.45, 0.48, 0.49, 0.4999, 0.5001, 0.51,
        0.55, 0.9, 0.99, 0.999]
X = []
for b in range(3):
    for r in FINE:
        W = inc.copy(); W[b] = 0; W[b, 4], W[b, 5] = r, 1 - r; X.append(W.ravel())
y = pool.evaluate(np.array(X), tag="E6_thresh").reshape(3, len(FINE)) - inc_R
print("\n[thresh] same face, fine grid around r = 1/3 and r = 1/2; delta vs incumbent")
for j, r in enumerate(FINE):
    print(f"  r = {r:<7g} " + "".join(f"{v:+9.3f}" for v in y[:, j]))
out["thresh"] = {"r": FINE, "delta": y.tolist()}

# --- stale negative estimates -------------------------------------------------------------------------
X, lab = [], []
for b in range(3):
    seen = set()
    for a in sorted(state["arch"][b], key=lambda a: a["delta"]):
        if a["delta"] >= -1e-9:
            break
        v = round(a["delta"], 5)
        if v in seen:
            continue
        seen.add(v)
        W = inc.copy(); W[b] = np.array(a["w"]) / np.sum(a["w"]); X.append(W.ravel()); lab.append((b, a["delta"]))
print(f"\n[stale] {len(X)} distinct negative estimates never measured against the current incumbent")
out["stale"] = []
if X:
    y = pool.evaluate(np.array(X), tag="E6_stale")
    d = y - inc_R
    for b in range(3):
        m = np.array([l[0] == b for l in lab])
        if m.any():
            est = np.array([l[1] for l in lab])[m]
            print(f"  b{b + 1}: n = {int(m.sum())}; estimated {est.min():+.3f} .. {est.max():+.3f}; "
                  f"measured min {d[m].min():+.4f}, median {np.median(d[m]):+.4f}, max {d[m].max():+.4f}; truly below incumbent: {int((d[m] < -1e-9).sum())}")
    out["stale"] = [{"building": l[0] + 1, "estimated": l[1], "measured": float(v)} for l, v in zip(lab, d)]
    if (d < -1e-9).any():
        i = int(np.argmin(d)); print("  NEW BEST", y[i], np.array(X[i]).reshape(3, 7))

print(f"\n{pool.n_solver_evals} new evaluations in {time.time() - t0:.0f}s")
json.dump(out, open("E6_mech.json", "w"), indent=1)
