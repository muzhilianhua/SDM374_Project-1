"""E22b: extend the E22 transplant grid past its edge (best points were at q = 0.6, r 0.10-0.13): q 0.7-0.95, r 0.02-0.14."""
import json
import sys

import numpy as np

from exp_e21 import block
from pool import SolverPool

RS = np.round(np.arange(0.02, 0.1401, 0.01), 3)
QS = [0.7, 0.8, 0.9, 0.95]


def main():
    f = json.load(open("best_point_E20.json", encoding="utf-8"))
    F, RF = np.array(f["weights_by_building"], float), f["R"]
    pool = SolverPool(n_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
    cands, X = [], []
    for k in (0, 2):
        for q in QS:
            for r in RS:
                w = block(F[k], 0.88, r, q); W = F.copy(); W[k] = w
                cands.append((k, q, float(r), w)); X.append(W.ravel())
    y = pool.evaluate(np.array(X), tag="E22b_grid")
    d = y - RF
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for k in (0, 2):
        print(f"building {k + 1}: rows q, columns r {RS}")
        for q in QS:
            print(f"  q {q}:", np.array([d[i] for i in range(len(cands)) if cands[i][0] == k and cands[i][1] == q]))
    j = int(np.argmin(d))
    print(f"best: building {cands[j][0] + 1} q {cands[j][1]} r {cands[j][2]} {d[j]:+.4f}")
    json.dump({"R_F": RF, "points": [{"building": k + 1, "q": q, "r": r, "d": float(v), "block": w.tolist()}
                                      for (k, q, r, w), v in zip(cands, d)]},
              open("E22b_grid.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
