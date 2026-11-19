"""E21b: fine (r, q) map of building 2 around F = best_point_E20 (R 102.0959).

E21 (coarse grid, `E21_coarse_log.json`) showed that for peak share p in 0.78-0.97 R does not depend on p, and that F sits
on a needle in (r, q): r 0.28 / 0.36 cost +1.8 / +2.3. Here p and the minor proportions stay at F's; r = deg/(deg+ramp)
and q = (deg+ramp)/(1-p) move on a 17 x 17 grid (r 0.30-0.34, q 0.48-0.56). If some grid point beats F, its block is
descended (others at F, up to 4 rounds) and block-coordinate cycles follow. Research only.
"""
import json
import sys

import numpy as np

from exp_alt_descent import descend, norm
from exp_e20 import cycles
from exp_e21 import block
from pool import SolverPool

RS = np.round(np.linspace(0.30, 0.34, 17), 4)
QS = np.round(np.linspace(0.48, 0.56, 17), 4)


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    f = json.load(open("best_point_E20.json", encoding="utf-8"))
    F, RF = np.array(f["weights_by_building"], float), f["R"]
    w2 = F[1]
    p = w2[2]
    pool = SolverPool(n_workers=n_workers)
    cands, rq = [], []
    for r in RS:
        for q in QS:
            cands.append(block(w2, p, r, q)); rq.append((float(r), float(q)))
    X = []
    for c in cands:
        W = F.copy(); W[1] = c; X.append(W.ravel())
    y = pool.evaluate(np.array(X), tag="E21b_grid")
    d = (y - RF).reshape(len(RS), len(QS))
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("r \ q", QS, flush=True)
    for i, r in enumerate(RS):
        print(f"{r:.4f}", d[i], flush=True)
    order = np.argsort(y)
    print("best grid:", ", ".join(f"r{rq[j][0]} q{rq[j][1]} {y[j] - RF:+.4f}" for j in order[:10]), flush=True)
    print(f"below F: {(y < RF - 1e-9).sum()} of {len(y)}; within +0.1: {(y < RF + 0.1).sum()}", flush=True)
    log = [{"stage": "fine_grid", "R_F": RF, "r": RS.tolist(), "q": QS.tolist(), "d": d.tolist()}]
    json.dump(log, open("E21b_log.json", "w", encoding="utf-8"), indent=1)
    j = int(order[0])
    if y[j] >= RF - 1e-9:
        print("no grid point beats F: stop", flush=True)
        return
    picked = [j]
    for k in order[1:]:
        if y[k] < RF - 1e-9 and max(abs(rq[k][0] - rq[j][0]), abs(rq[k][1] - rq[j][1])) > 0.006:
            picked.append(int(k)); break
    sd = {f"r{rq[k][0]}_q{rq[k][1]}": (1, cands[k]) for k in picked}
    st = descend(F, RF, sd, "E21b_seed", "E21b_seed.json", max_rounds=4, n_workers=n_workers, rng_seed=215)
    best = min(st.values(), key=lambda s: s["d"])
    options = [[F[0]], [F[1], norm(best["w"])], [F[2]]]
    labels = [["o"], ["o", "n"], ["o"]]
    base, R = cycles(pool, F, RF, options, labels, n_workers, log, prefix="E21b", rng0=215)
    print(f"final R {R:.10f} (F {RF:.10f}, delta {R - RF:+.4f})", flush=True)
    log.append({"stage": "final", "R": R, "weights_by_building": np.asarray(base).tolist()})
    json.dump(log, open("E21b_log.json", "w", encoding="utf-8"), indent=1)
    np.set_printoptions(precision=6, suppress=True)
    print(base)


if __name__ == "__main__":
    main()
