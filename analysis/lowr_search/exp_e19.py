"""E19: refine the v4 seed 7 online point S and cross it with the best point B.

S (R 103.2753, last call of the T37 rehearsal v4_s7) sits close to B (R 103.2473, E17) but its building 2 uses a
different mix (cost, carbon, peak and resilience 0.06-0.13 each, ramping 0.51) and the run was still descending.
Research only: S is read from the rehearsal log, nothing here flows back into the online policy.

  1. measure the 2x2x2 combinations of B's and S's blocks per building (B and S included);
  2. block-coordinate cycles from the best measured point, as in E17: descend each building's block with the
     other two fixed (up to 3 rounds), then measure the combinations of the winners; stop after a cycle without
     a measured improvement (max 3 descent cycles).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from exp_alt_descent import descend, norm
from pool import SolverPool

REHEARSAL = Path(__file__).resolve().parent.parent / "online_rehearsal" / "rehearsals"


def online_best(run):
    recs = [json.loads(l) for l in open(REHEARSAL / run / "online_log.jsonl", encoding="utf-8")]
    p = min((r for r in recs if r.get("type") == "point" and r["R"] is not None), key=lambda r: r["R"])
    return np.array(p["W"], float).reshape(3, 7), float(p["R"])


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    b = json.load(open("best_point_E17.json", encoding="utf-8"))
    B, RB = np.array(b["weights_by_building"], float), b["R"]
    S, RS = online_best("v4_s7")
    print(f"B {RB:.10f}  S {RS:.10f} (online log)", flush=True)
    options = [[norm(B[k]), norm(S[k])] for k in range(3)]
    labels = [["B", "S"]] * 3
    pool = SolverPool(n_workers=n_workers)
    base, R, history = None, np.inf, []
    for cycle in range(1, 5):
        idx = list(itertools.product(*[range(len(o)) for o in options]))
        cands = [np.array([options[k][i[k]] for k in range(3)]) for i in idx]
        y = pool.evaluate(np.array([c.ravel() for c in cands]), tag=f"E19_combo{cycle}")
        names = ["".join(labels[k][i[k]] for k in range(3)) for i in idx]
        order = np.argsort(y)
        print(f"cycle {cycle} combinations:", ", ".join(f"{names[j]} {y[j]:.4f}" for j in order), flush=True)
        j = int(order[0])
        improved = y[j] < R - 1e-9
        if improved:
            base, R = cands[j], float(y[j])
        history.append({"cycle": cycle, "R": R, "combos": {names[i]: float(y[i]) for i in range(len(y))},
                        "weights_by_building": base.tolist()})
        json.dump(history, open("E19_cycles.json", "w", encoding="utf-8"), indent=1)
        if (cycle > 1 and not improved) or cycle == 4:
            break
        seeds = {f"b{k + 1}": (k, norm(base[k])) for k in range(3)}
        st = descend(base, R, seeds, f"E19_c{cycle}", f"E19_c{cycle}.json", max_rounds=3,
                     n_workers=n_workers, rng_seed=190 + cycle)
        options, labels = [], []
        for k in range(3):
            win = st[f"b{k + 1}"]["d"] < -1e-9
            options.append([base[k]] + ([st[f"b{k + 1}"]["w"]] if win else []))
            labels.append(["o", "n"] if win else ["o"])
        if all(len(o) == 1 for o in options):
            print(f"cycle {cycle}: no building improved, stop", flush=True)
            break
    print(f"final R {R:.10f} (B {RB:.10f}, delta {R - RB:+.4f})", flush=True)
    np.set_printoptions(precision=6, suppress=True)
    print(base)


if __name__ == "__main__":
    main()
