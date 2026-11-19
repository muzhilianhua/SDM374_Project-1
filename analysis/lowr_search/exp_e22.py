"""E22: transplant building 2's peak-dominated form into buildings 1 and 3 at F = best_point_E20 (R 102.0959).

For building k in {1, 3}, the other two buildings stay at F:
  1. seeds: the 3 best distinct peak-dominated (peak > 0.5) blocks of building k in the 10 T37 rehearsal logs, and a
     direct copy of F's building-2 block;
  2. grid: peak share p = 0.88 (E21: R does not depend on p in 0.78-0.97), the building's own minor proportions,
     r = deg/(deg+ramp) 0.10-0.45 step 0.01, q = (deg+ramp)/(1-p) in {0.4, 0.5, 0.6};
  3. descend the 2 best distinct blocks per building (up to 4 rounds); if a block beats F, block-coordinate cycles.
Research only: rehearsal points are read, nothing flows back into the online policy.
"""
import json
import sys

import numpy as np

from exp_alt_descent import descend, norm
from exp_e19 import REHEARSAL
from exp_e20 import cycles
from exp_e21 import block
from pool import SolverPool

RS = np.round(np.arange(0.10, 0.4501, 0.01), 3)
QS = [0.4, 0.5, 0.6]


def log_seeds(k, n=3):
    rows = []
    for d in sorted(REHEARSAL.iterdir()):
        for line in open(d / "online_log.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("type") != "point" or r["R"] is None:
                continue
            w = norm(np.array(r["W"], float).reshape(3, 7)[k])
            if w[2] > 0.5:
                rows.append((r["R"], d.name, w))
    rows.sort(key=lambda t: t[0])
    out = []
    for R, name, w in rows:
        if all(np.abs(w - v).max() > 0.02 for _, v in out):
            out.append((f"log_{name}", w))
        if len(out) == n:
            break
    return out


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    f = json.load(open("best_point_E20.json", encoding="utf-8"))
    F, RF = np.array(f["weights_by_building"], float), f["R"]
    pool = SolverPool(n_workers=n_workers)
    cands = []                                   # (building, name, block)
    for k in (0, 2):
        for name, w in log_seeds(k):
            cands.append((k, name, w))
        cands.append((k, "copy_b2", norm(F[1])))
        for q in QS:
            for r in RS:
                cands.append((k, f"r{r}_q{q}", block(F[k], 0.88, r, q)))
    X = []
    for k, _, w in cands:
        W = F.copy(); W[k] = w; X.append(W.ravel())
    y = pool.evaluate(np.array(X), tag="E22_grid")
    d = y - RF
    log = [{"stage": "grid", "R_F": RF,
            "points": [{"building": k + 1, "name": n, "d": float(v), "block": w.tolist()} for (k, n, w), v in zip(cands, d)]}]
    json.dump(log, open("E22_log.json", "w", encoding="utf-8"), indent=1)
    seeds = {}
    for k in (0, 2):
        idx = sorted((i for i in range(len(cands)) if cands[i][0] == k), key=lambda i: d[i])
        print(f"building {k + 1} best:", ", ".join(f"{cands[i][1]} {d[i]:+.3f}" for i in idx[:10]), flush=True)
        for q in QS:
            sel = [i for i in idx if cands[i][1].endswith(f"_q{q}")]
            print(f"  q {q}: median {np.median(d[sel]):+.3f}, best {cands[sel[0]][1]} {d[sel[0]]:+.3f}", flush=True)
        picked = []
        for i in idx:
            if d[i] > 1.5:
                break
            if all(np.abs(cands[i][2] - cands[j][2]).max() > 0.02 for j in picked):
                picked.append(i)
            if len(picked) == 2:
                break
        for i in picked:
            seeds[f"b{k + 1}_{cands[i][1]}"] = (k, cands[i][2])
    if not seeds:
        print("no transplant within +1.5 of F: stop", flush=True)
        return
    st = descend(F, RF, seeds, "E22_seed", "E22_seed.json", max_rounds=4, n_workers=n_workers, rng_seed=220)
    log.append({"stage": "seed_descent", "result": {n: {"d": s["d"], "w": np.asarray(s["w"]).tolist()}
                                                     for n, s in st.items()}})
    json.dump(log, open("E22_log.json", "w", encoding="utf-8"), indent=1)
    options = [[F[0]], [F[1]], [F[2]]]
    labels = [["o"], ["o"], ["o"]]
    for k in (0, 2):
        win = [s for s in st.values() if s["b"] == k and s["d"] < -1e-9]
        if win:
            options[k].append(norm(min(win, key=lambda s: s["d"])["w"])); labels[k].append("n")
    if all(len(o) == 1 for o in options):
        print("no transplanted block beat F: stop", flush=True)
        return
    base, R = cycles(pool, F, RF, options, labels, n_workers, log, prefix="E22", rng0=220)
    print(f"final R {R:.10f} (F {RF:.10f}, delta {R - RF:+.4f})", flush=True)
    log.append({"stage": "final", "R": R, "weights_by_building": np.asarray(base).tolist()})
    json.dump(log, open("E22_log.json", "w", encoding="utf-8"), indent=1)
    np.set_printoptions(precision=6, suppress=True)
    print(base)


if __name__ == "__main__":
    main()
