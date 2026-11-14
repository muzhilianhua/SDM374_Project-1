"""E11: does a vanishing weight differ from an exact zero?

E8 showed that some minor weights already change R at m = 1e-4, the smallest value scanned, so the
earlier statement "below about 1e-4 a weight is the same as zero" (E2) does not hold in general.
The controller breaks exact score ties (within 1e-12) by a fixed rule; a positive weight on another
objective, however small, would break them by that objective instead. This probe replaces one minor
weight of one building at a time by m in {0, 1e-13, 1e-10, 1e-8, 1e-6, 1e-5, 3e-5, 1e-4}
(m = weight / (degradation + ramping)), starting from the best point in the log.
"""
import itertools
import time

import numpy as np

from common import MINOR, NAMES, block, to_ratio
from pool import SolverPool, load_log
from common import write_json

MN = [NAMES[k] for k in MINOR]
LEVELS = [0.0, 1e-13, 1e-10, 1e-8, 1e-6, 1e-5, 3e-5, 1e-4]
W, R, _ = load_log()
i = int(np.argmin(R)); best = W[i].reshape(3, 7).copy(); best_R = float(R[i])
pool = SolverPool(n_workers=10)
t0 = time.time()
X, meta = [], []
for b in range(3):
    r0, m0 = to_ratio(best[b])
    for j in range(5):
        for v in LEVELS:
            m = m0.copy(); m[j] = v
            Wn = best.copy(); Wn[b] = block(r0, m); X.append(Wn.ravel()); meta.append((b, j, v))
y = pool.evaluate(np.array(X), tag="E11_tiny") - best_R
X = np.array(X).reshape(-1, 3, 7)
print(f"start {best_R:.9f}; {pool.n_solver_evals} new evaluations in {time.time() - t0:.0f}s")
print("delta vs the start when one weight is replaced by m\n  bld objective    current m  " + "".join(f"{v:>10.0e}" for v in LEVELS))
out, cands = {"start_R": best_R, "levels": LEVELS, "rows": []}, [None, None, None]
for b in range(3):
    m0 = to_ratio(best[b])[1]; bestv = -1e-9
    for j in range(5):
        sel = [k for k, mt in enumerate(meta) if mt[0] == b and mt[1] == j]
        d = y[sel]
        print(f"  b{b + 1}  {MN[j]:11s} {m0[j]:10.3e}  " + "".join(f"{v:+10.4f}" for v in d))
        out["rows"].append({"building": b + 1, "minor": MN[j], "current_m": float(m0[j]), "delta": d.tolist()})
        k = int(np.argmin(d))
        if d[k] < bestv:
            bestv, cands[b] = float(d[k]), X[sel[k]][b]
if any(c is not None for c in cands):
    opts = [[best[b]] + ([cands[b]] if cands[b] is not None else []) for b in range(3)]
    combos = list(itertools.product(*[range(len(o)) for o in opts]))
    Xc = np.array([np.concatenate([opts[b][c[b]] for b in range(3)]) for c in combos])
    yc = pool.evaluate(Xc, tag="E11_combo") - best_R
    k = int(np.argmin(yc))
    print("combinations of the per-building best changes: " + "  ".join(f"{''.join(str(v) for v in c)}:{v2:+.4f}" for c, v2 in zip(combos, yc)))
    print(f"best {yc[k]:+.4f} -> R = {best_R + yc[k]:.9f}"); print(Xc[k].reshape(3, 7))
    out["best_R"] = best_R + float(yc[k]); out["best_W"] = Xc[k].reshape(3, 7).tolist()
else:
    print("no change lowers R")
write_json("E11_tiny.json", out, indent=1)
