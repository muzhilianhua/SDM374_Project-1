"""E20: cross the best points of the other 9 T37 rehearsal runs with the E19 best point E (R 102.6609).

Research only: the online points are read from the rehearsal logs, nothing flows back into the online policy.

  1. swaps: for each run X, replace every non-empty subset of E's buildings by X's blocks (9 x 7 points);
  2. seed descent: the (up to 6) single-building swaps within +0.6 of E are descended per building with the other
     two buildings at E (as E15), up to 3 rounds;
  3. if a swap or a descended seed beats E: E19-style block-coordinate cycles from the best point (combinations of
     old/new blocks, then per-building descent; stop after a cycle without improvement, max 3 cycles).
"""
import itertools
import json
import sys

import numpy as np

from exp_alt_descent import descend, norm
from exp_e19 import online_best
from pool import SolverPool

RUNS = ["v1_s2026", "v1_s7", "v1_s11", "v2_s2026", "v2_s7", "v2_s11", "v3_s2026", "v4_s2026", "v4_s11"]


def cycles(pool, base, R, options, labels, n_workers, log, prefix="E20", rng0=200):
    for cycle in range(1, 4):
        idx = list(itertools.product(*[range(len(o)) for o in options]))
        if len(idx) > 1:
            cands = [np.array([options[k][i[k]] for k in range(3)]) for i in idx]
            y = pool.evaluate(np.array([c.ravel() for c in cands]), tag=f"{prefix}_combo{cycle}")
            names = ["".join(labels[k][i[k]] for k in range(3)) for i in idx]
            order = np.argsort(y)
            print(f"cycle {cycle} combinations:", ", ".join(f"{names[j]} {y[j]:.4f}" for j in order), flush=True)
            j = int(order[0])
            improved = y[j] < R - 1e-9
            if improved:
                base, R = cands[j], float(y[j])
            log.append({"stage": f"cycle{cycle}", "R": R, "combos": {names[i]: float(y[i]) for i in range(len(y))},
                        "weights_by_building": base.tolist()})
            json.dump(log, open(f"{prefix}_log.json", "w", encoding="utf-8"), indent=1)
            if cycle > 1 and not improved:
                break
        seeds = {f"b{k + 1}": (k, norm(base[k])) for k in range(3)}
        st = descend(base, R, seeds, f"{prefix}_c{cycle}", f"{prefix}_c{cycle}.json", max_rounds=3,
                     n_workers=n_workers, rng_seed=rng0 + cycle)
        options, labels = [], []
        for k in range(3):
            win = st[f"b{k + 1}"]["d"] < -1e-9
            options.append([base[k]] + ([st[f"b{k + 1}"]["w"]] if win else []))
            labels.append(["o", "n"] if win else ["o"])
        if all(len(o) == 1 for o in options):
            print(f"cycle {cycle}: no building improved, stop", flush=True)
            break
    return base, R


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    e = json.load(open("best_point_E19.json", encoding="utf-8"))
    E, RE = np.array(e["weights_by_building"], float), e["R"]
    pool = SolverPool(n_workers=n_workers)
    log = []
    # 1. subset swaps
    X, meta = [], []
    for run in RUNS:
        S, RS = online_best(run)
        print(f"{run}: online best {RS:.4f}", flush=True)
        for mask in itertools.product([0, 1], repeat=3):
            if not any(mask):
                continue
            W = np.array([norm(S[k]) if mask[k] else E[k] for k in range(3)])
            X.append(W); meta.append((run, mask))
    y = pool.evaluate(np.array([w.ravel() for w in X]), tag="E20_swap")
    swaps = sorted(({"run": r, "mask": "".join("X" if m else "E" for m in mk), "R": float(v), "d": float(v - RE)}
                    for (r, mk), v in zip(meta, y)), key=lambda t: t["R"])
    print("best swaps:", ", ".join(f"{s['run']}:{s['mask']} {s['d']:+.4f}" for s in swaps[:12]), flush=True)
    log.append({"stage": "swaps", "R_E": RE, "swaps": swaps})
    json.dump(log, open("E20_log.json", "w", encoding="utf-8"), indent=1)
    j = int(np.argmin(y))
    base, R = (X[j], float(y[j])) if y[j] < RE - 1e-9 else (E, RE)
    # 2. seed descent of the promising single-building swaps
    single = [s for s in swaps if s["mask"].count("X") == 1 and s["d"] < 0.6][:6]
    seeds = {}
    for s in single:
        k = s["mask"].index("X")
        seeds[f"{s['run']}_b{k + 1}"] = (k, norm(online_best(s["run"])[0][k]))
    options = [[base[k]] for k in range(3)]
    labels = [["o"] for _ in range(3)]
    if seeds:
        st = descend(E, RE, seeds, "E20_seed", "E20_seed.json", max_rounds=3, n_workers=n_workers, rng_seed=200)
        log.append({"stage": "seed_descent", "result": {n: {"d": s["d"], "w": np.asarray(s["w"]).tolist()}
                                                         for n, s in st.items()}})
        json.dump(log, open("E20_log.json", "w", encoding="utf-8"), indent=1)
        best_by_k = {}
        for n, s in st.items():
            if s["d"] < -1e-9 and (s["b"] not in best_by_k or s["d"] < best_by_k[s["b"]][0]):
                best_by_k[s["b"]] = (s["d"], s["w"])
        for k, (d, w) in best_by_k.items():
            options[k].append(norm(w)); labels[k].append("n")
    # 3. cycles, only if something beat E
    if R < RE - 1e-9 or any(len(o) > 1 for o in options):
        base, R = cycles(pool, base, R, options, labels, n_workers, log)
    else:
        print("nothing beat E: no cycles", flush=True)
    print(f"final R {R:.10f} (E {RE:.10f}, delta {R - RE:+.4f})", flush=True)
    log.append({"stage": "final", "R": R, "weights_by_building": np.asarray(base).tolist()})
    json.dump(log, open("E20_log.json", "w", encoding="utf-8"), indent=1)
    np.set_printoptions(precision=6, suppress=True)
    print(base)


if __name__ == "__main__":
    main()
