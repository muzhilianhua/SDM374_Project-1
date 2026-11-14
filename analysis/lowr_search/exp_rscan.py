"""E9: fine scan of r = degradation / (degradation + ramping) for one building at a time.

Two variants per building, the other two buildings at the incumbent:
  minor   the incumbent's minor weights kept in the same ratio to (degradation + ramping)
  pure    no minor weights
r runs from 0.15 to 0.50 in steps of 0.002. The plateau of the incumbent is far narrower than the step
(about 0.0002 in r), so neighbouring samples are different cells: the scan shows the trend and the
scatter around it, not the exact piecewise-constant structure.
"""
import time

import numpy as np

from common import block, to_ratio
from pool import SolverPool
from common import read_json, write_json

state = read_json("fms_state_run4.json")
inc = np.array(state["inc"]); inc = inc / inc.sum(1, keepdims=True); inc_R = state["inc_R"]
GRID = np.round(np.arange(0.15, 0.5001, 0.002), 4)
X, meta = [], []
for b in range(3):
    r0, m0 = to_ratio(inc[b])
    for variant, m in (("minor", m0), ("pure", np.zeros(5))):
        for r in GRID:
            W = inc.copy(); W[b] = block(float(r), m); X.append(W.ravel()); meta.append((b, variant))
pool = SolverPool(n_workers=10)
t0 = time.time()
y = pool.evaluate(np.array(X), tag="E9_rscan") - inc_R
print(f"{pool.n_solver_evals} new evaluations in {time.time() - t0:.0f}s; incumbent {inc_R:.4f}")
out = {"r": GRID.tolist(), "incumbent_R": inc_R, "incumbent_r": [float(to_ratio(inc[b])[0]) for b in range(3)], "delta": {}}
for b in range(3):
    for variant in ("minor", "pure"):
        d = y[[m == (b, variant) for m in meta]]
        out["delta"][f"b{b + 1}_{variant}"] = d.tolist()
        i = int(np.argmin(d))
        print(f"b{b + 1} {variant:5s}: min {d.min():+.4f} at r = {GRID[i]:.3f}; below incumbent: {int((d < -1e-9).sum())}; "
              f"median for r < 0.30: {np.median(d[GRID < 0.30]):+.3f}")
write_json("E9_rscan.json", out, indent=1)
if y.min() < -1e-9:
    i = int(np.argmin(y)); print("NEW BEST", y[i] + inc_R, np.array(X[i]).reshape(3, 7))
