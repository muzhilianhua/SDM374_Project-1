"""E21: explore building 2's peak-dominated valley found in E20, around F = best_point_E20 (R 102.0959).

Building 2 block family: peak share p, battery share q of the non-peak mass (degradation + ramping), r = deg/(deg+ramp);
the other minor weights (cost, carbon, discomfort, resilience) keep F's building-2 proportions.
  1. grid p x r x q (7 x 6 x 3 = 126 points) plus the 5 descended E20 seed blocks, buildings 1 and 3 at F;
  2. descend the 4 best distinct grid/seed blocks (building 2 only, others at F, up to 4 rounds);
  3. block-coordinate cycles (all buildings) from the best, as E20, max 3 cycles.
Research only.
"""
import itertools
import json
import sys

import numpy as np

from exp_alt_descent import descend, norm
from exp_e20 import cycles
from pool import SolverPool

PS = [0.70, 0.78, 0.84, 0.88, 0.91, 0.94, 0.97]
RS = [0.15, 0.22, 0.28, 0.32, 0.36, 0.42]
QS = [0.30, 0.515, 0.70]


def block(w, p, r, q):
    minor = w[[0, 1, 3, 6]] / w[[0, 1, 3, 6]].sum() * (1 - p) * (1 - q)
    v = np.zeros(7)
    v[[0, 1, 3, 6]] = minor
    v[2] = p
    v[4], v[5] = (1 - p) * q * r, (1 - p) * q * (1 - r)
    return norm(v)


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    f = json.load(open("best_point_E20.json", encoding="utf-8"))
    F, RF = np.array(f["weights_by_building"], float), f["R"]
    w2 = F[1]
    print(f"F {RF:.10f}; b2 p {w2[2]:.4f} q {(w2[4] + w2[5]) / (1 - w2[2]):.4f} r {w2[4] / (w2[4] + w2[5]):.4f}",
          flush=True)
    pool = SolverPool(n_workers=n_workers)
    cands, names = [], []
    for p, r, q in itertools.product(PS, RS, QS):
        cands.append(block(w2, p, r, q)); names.append(f"p{p}_r{r}_q{q}")
    seeds = json.load(open("E20_seed.json", encoding="utf-8"))
    for n, v in seeds.items():
        if v["building"] == 2 and n != "v2_s11_b2":
            cands.append(norm(v["block"])); names.append(n)
    X = []
    for c in cands:
        W = F.copy(); W[1] = c; X.append(W.ravel())
    y = pool.evaluate(np.array(X), tag="E21_grid")
    d = y - RF
    order = np.argsort(d)
    print("best grid:", ", ".join(f"{names[j]} {d[j]:+.4f}" for j in order[:15]), flush=True)
    log = [{"stage": "grid", "R_F": RF, "points": [{"name": names[i], "d": float(d[i]), "block": cands[i].tolist()}
                                                   for i in order]}]
    json.dump(log, open("E21_log.json", "w", encoding="utf-8"), indent=1)
    # per p: profile of the best (r, q)
    for p in PS:
        sel = [i for i in range(len(names)) if names[i].startswith(f"p{p}_")]
        j = min(sel, key=lambda i: d[i])
        print(f"  p {p}: best {names[j]} {d[j]:+.4f}", flush=True)
    # 2. descend 4 distinct good blocks (skip near-duplicates of F's own block and of each other)
    picked = []
    for j in order:
        if any(np.abs(cands[j] - cands[k]).max() < 0.02 for k in picked) or np.abs(cands[j] - w2).max() < 0.005:
            continue
        picked.append(j)
        if len(picked) == 4:
            break
    sd = {names[j]: (1, cands[j]) for j in picked}
    st = descend(F, RF, sd, "E21_seed", "E21_seed.json", max_rounds=4, n_workers=n_workers, rng_seed=210)
    best = min(st.values(), key=lambda s: s["d"])
    log.append({"stage": "seed_descent", "result": {n: {"d": s["d"], "w": np.asarray(s["w"]).tolist()}
                                                     for n, s in st.items()}})
    json.dump(log, open("E21_log.json", "w", encoding="utf-8"), indent=1)
    options = [[F[0]], [F[1]], [F[2]]]
    labels = [["o"], ["o"], ["o"]]
    if best["d"] < -1e-9:
        options[1].append(norm(best["w"])); labels[1].append("n")
    if len(options[1]) > 1:
        base, R = cycles(pool, F, RF, options, labels, n_workers, log, prefix="E21", rng0=210)
    else:
        print("no building-2 block beat F: no cycles", flush=True)
        base, R = F, RF
    print(f"final R {R:.10f} (F {RF:.10f}, delta {R - RF:+.4f})", flush=True)
    log.append({"stage": "final", "R": R, "weights_by_building": np.asarray(base).tolist()})
    json.dump(log, open("E21_log.json", "w", encoding="utf-8"), indent=1)
    np.set_printoptions(precision=6, suppress=True)
    print(base)


if __name__ == "__main__":
    main()
