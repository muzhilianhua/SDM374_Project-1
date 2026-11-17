"""E16: retune buildings 1 and 2 at the new best point (E15: W* with building 3 in the second valley).

W*'s blocks for buildings 1 and 2 were tuned with the old building-3 block. Additivity near the optimum says
they should still be good, but the minor-weight windows are narrow, so each is descended again with
building 3 at its new block. The two buildings-1/2 candidates of E15 that were still improving after 6 rounds
(+0.26, +0.28 relative to the old point) continue here in the new context.
"""
import json
import sys

import numpy as np

from exp_alt_descent import descend, norm


def main():
    best = json.load(open("best_point_E15.json", encoding="utf-8"))
    W0, R0 = np.array(best["weights_by_building"], float), best["R"]
    e15 = json.load(open("E15_descent.json", encoding="utf-8"))
    seeds = {
        "b1_incumbent": (0, norm(W0[0])),
        "b2_incumbent": (1, norm(W0[1])),
        "b1_ramp_cost": (0, norm(e15["b1_ramp_cost_x0.85"]["block"])),
        "b2_ramp_carb": (1, norm(e15["b2_ramp_carb"]["block"])),
    }
    descend(W0, R0, seeds, "E16", "E16_retune.json", max_rounds=8,
            n_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10, rng_seed=16)


if __name__ == "__main__":
    main()
