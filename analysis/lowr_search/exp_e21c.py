"""E21c: fine ramping x degradation scan of building 2 at F = best_point_E20.

E21b showed the low points of building 2 lie in a thin trough where ramping (relative to the minor weights) is nearly
constant and degradation varies. Here building 2's ramping is scaled by 0.98-1.02 (step 0.002) and degradation by
0.90-1.10 (step 0.025), unnormalised, then the block is renormalised. Research only.
"""
import json
import sys

import numpy as np

from exp_alt_descent import norm
from pool import SolverPool

FR = np.round(np.arange(0.98, 1.0201, 0.002), 4)
FD = np.round(np.arange(0.90, 1.1001, 0.025), 4)


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    f = json.load(open("best_point_E20.json", encoding="utf-8"))
    F, RF = np.array(f["weights_by_building"], float), f["R"]
    pool = SolverPool(n_workers=n_workers)
    X, keys = [], []
    for a in FR:
        for b in FD:
            w = F[1].copy(); w[5] *= a; w[4] *= b
            W = F.copy(); W[1] = norm(w); X.append(W.ravel()); keys.append((float(a), float(b)))
    y = pool.evaluate(np.array(X), tag="E21c_grid")
    d = (y - RF).reshape(len(FR), len(FD))
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("ramp \ degr", FD, flush=True)
    for i, a in enumerate(FR):
        print(f"{a:.3f}", d[i], flush=True)
    order = np.argsort(y)
    print("best:", ", ".join(f"ramp x{keys[j][0]} degr x{keys[j][1]} {y[j] - RF:+.4f}" for j in order[:8]), flush=True)
    j = int(order[0])
    out = {"R_F": RF, "ramp_f": FR.tolist(), "degr_f": FD.tolist(), "d": d.tolist(), "best_R": float(y[j]),
           "best_weights_by_building": X[j].reshape(3, 7).tolist()}
    json.dump(out, open("E21c_grid.json", "w", encoding="utf-8"), indent=1)
    print(f"best R {y[j]:.10f} (F {RF:.10f}, delta {y[j] - RF:+.4f}); below F: {(y < RF - 1e-9).sum()} of {len(y)}",
          flush=True)


if __name__ == "__main__":
    main()
