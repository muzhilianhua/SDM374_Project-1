"""Verify the E20 best point: two fresh solver calls (bypassing the cache) and rounding sensitivity."""
import json

import numpy as np

from pool import SolverPool, normalize

hist = json.load(open("E20_log.json", encoding="utf-8"))
W = np.array(hist[-1]["weights_by_building"], float)
R = hist[-1]["R"]
pool = SolverPool(n_workers=3)
single = pool._call(0, W.reshape(1, 21))[0]
nb = json.load(open("best_point_E19.json", encoding="utf-8"))
B = np.array(nb["weights_by_building"], float)
trio = pool._call(1, np.array([B.ravel(), W.ravel(), B.ravel()]))
rounding = {}
for d in (6, 5, 4, 3):
    Wr = np.round(W, d)
    Wr = Wr / Wr.sum(axis=1, keepdims=True)
    rounding[str(d)] = float(pool.evaluate(Wr.reshape(1, 21), tag=f"E20_round{d}")[0])
out = {"R": R, "weights_by_building": W.tolist(),
       "source": "E20: E19 best point (102.6609) crossed with the best points of the other 9 T37 rehearsal runs; building 2 from v2_s11 (online best 107.49) descended 3 rounds, then block-coordinate cycles (2)",
       "fresh_calls": {"single": single, "middle_of_three": trio[1]}, "rounding_decimals": rounding}
json.dump(out, open("best_point_E20.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "weights_by_building"}, indent=1))
