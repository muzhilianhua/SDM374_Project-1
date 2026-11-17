"""Verify the E19 best point: two fresh solver calls (bypassing the cache) and rounding sensitivity."""
import json

import numpy as np

from pool import SolverPool, normalize

hist = json.load(open("E19_cycles.json", encoding="utf-8"))
W = np.array(hist[-1]["weights_by_building"], float)
R = hist[-1]["R"]
pool = SolverPool(n_workers=3)
single = pool._call(0, W.reshape(1, 21))[0]
nb = json.load(open("best_point_E17.json", encoding="utf-8"))
B = np.array(nb["weights_by_building"], float)
trio = pool._call(1, np.array([B.ravel(), W.ravel(), B.ravel()]))
rounding = {}
for d in (6, 5, 4, 3):
    Wr = np.round(W, d)
    Wr = Wr / Wr.sum(axis=1, keepdims=True)
    rounding[str(d)] = float(pool.evaluate(Wr.reshape(1, 21), tag=f"E19_round{d}")[0])
out = {"R": R, "weights_by_building": W.tolist(),
       "source": "E19: B (E17, 103.2473) buildings 1-2 crossed with building 3 of the T37 v4_s7 online best point "
                 "(103.2753), then block-coordinate descent (3 cycles)",
       "fresh_calls": {"single": single, "middle_of_three": trio[1]}, "rounding_decimals": rounding}
json.dump(out, open("best_point_E19.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "weights_by_building"}, indent=1))
