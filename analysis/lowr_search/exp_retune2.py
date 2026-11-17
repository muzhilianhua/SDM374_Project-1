"""E17: combine the E16 winners and run block-coordinate cycles at the new point.

E16 showed that building 1 interacts with the new building-3 block (a block that was +0.28 at the old point is
-0.58 at the new one), so one-building deltas no longer simply add. Each cycle therefore:
  1. measures the combinations of the current best blocks per building and takes the best measured point;
  2. descends every building's block at that point (other two fixed), up to 3 rounds;
  3. stops when a full cycle brings no measured improvement (max 3 descent cycles).
"""
import itertools
import json
import sys

import numpy as np

from exp_alt_descent import descend, norm
from pool import SolverPool


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    best = json.load(open("best_point_E15.json", encoding="utf-8"))
    base, R = np.array(best["weights_by_building"], float), best["R"]
    e16 = json.load(open("E16_retune.json", encoding="utf-8"))
    options = [[base[0], np.array(e16["b1_incumbent"]["block"]), np.array(e16["b1_ramp_cost"]["block"])],
               [base[1], np.array(e16["b2_incumbent"]["block"])],
               [base[2]]]
    pool = SolverPool(n_workers=n_workers)
    history = []
    for cycle in range(1, 5):
        cands = [np.array(c) for c in itertools.product(*options)]
        y = pool.evaluate(np.array([c.ravel() for c in cands]), tag=f"E17_combo{cycle}")
        i = int(np.argmin(y))
        print(f"cycle {cycle}: {len(cands)} combinations, best {y[i]:.6f} (was {R:.6f})", flush=True)
        improved = y[i] < R - 1e-9
        if improved:
            base, R = cands[i], float(y[i])
        history.append({"cycle": cycle, "R": R, "weights_by_building": base.tolist()})
        json.dump(history, open("E17_cycles.json", "w", encoding="utf-8"), indent=1)
        if (cycle > 1 and not improved) or cycle == 4:     # cycle 4 only measures cycle 3's combinations
            break
        seeds = {f"b{b + 1}": (b, norm(base[b])) for b in range(3)}
        st = descend(base, R, seeds, f"E17_c{cycle}", f"E17_c{cycle}.json", max_rounds=3,
                     n_workers=n_workers, rng_seed=170 + cycle)
        options = [[base[b]] + ([st[f"b{b + 1}"]["w"]] if st[f"b{b + 1}"]["d"] < -1e-9 else []) for b in range(3)]
        if all(len(o) == 1 for o in options):
            print(f"cycle {cycle}: no building improved, stop", flush=True)
            break
    print(f"final R {R:.10f}", flush=True)
    np.set_printoptions(precision=6, suppress=True)
    print(base)


if __name__ == "__main__":
    main()
