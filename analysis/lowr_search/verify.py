"""Re-measure the best point outside the cache and describe how fragile it is.

  1. two fresh solver.exe calls (alone, and inside a batch) must return the logged R exactly;
  2. the same weights rounded to 8 / 6 / 5 / 4 / 3 decimals (what a copy-paste would send);
  3. random multiplicative perturbations of every positive weight at 1e-6 .. 1e-2.
"""
import json

import numpy as np

from common import team_best
from pool import SolverPool, load_log, normalize

W, R, tags = load_log()
i = int(np.argmin(R))
best, best_R = W[i], float(R[i])
pool = SolverPool(n_workers=10)
alone = pool._call(0, best[None])[0]
other = normalize([team_best().ravel(), np.full(21, 1 / 7)])
batch = pool._call(1, np.vstack([other[0], best, other[1]]))
print(f"logged {best_R!r}\nalone  {alone!r}\nbatch  {batch[1]!r}  (team best in the same call: {batch[0]:.6f}, uniform weights: {batch[2]:.6f})")
print("reproduced exactly:", alone == best_R and batch[1] == best_R)

rounded = [np.round(best, d) for d in (8, 6, 5, 4, 3)]
y = pool.evaluate(np.array(rounded), tag="verify_round")
for d, r in zip((8, 6, 5, 4, 3), y):
    print(f"rounded to {d} decimals: R = {r:.6f}  ({r - best_R:+.4f})")

rng = np.random.default_rng(7)
X, scale = [], []
for s in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    for _ in range(12):
        X.append(best * 10 ** rng.normal(0, s, 21)); scale.append(s)
y = pool.evaluate(np.array(X), tag="verify_perturb")
scale = np.array(scale)
print("\nrelative noise on every weight -> share of perturbed points with the same R, median and worst change")
for s in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    d = y[scale == s] - best_R
    print(f"  10^N(0,{s:g}): same R {np.mean(np.abs(d) < 1e-9):.2f}   median {np.median(d):+.4f}   worst {d.max():+.4f}   best {d.min():+.4f}")
json.dump({"R": best_R, "alone": alone, "batch": batch[1], "weights": best.reshape(3, 7).tolist()},
          open("verification.json", "w"), indent=1)
