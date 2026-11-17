"""E13: are there other valleys? Transplant blocks the online rehearsals settled on into the best point.

The rehearsals (Project-1/online_rehearsal, T37) ended near 105-107 with blocks that look unlike the known
basin (e.g. building 2 almost all peak or all discomfort). With R additive over buildings near the optimum,
putting one such block into the best point W* (other two buildings unchanged) measures that block's value
relative to W*'s block for the same building. A block from another regime within ~1 of W*'s block, without
any local refinement, is a candidate second valley (in-basin random blocks are +0.4..+0.9, E3).
"""
import json
import sys
from pathlib import Path

import numpy as np

from pool import SolverPool

REH = Path(__file__).resolve().parent.parent / "online_rehearsal" / "rehearsals"
# E13/E14 ran on 2026-11-15 on the outside copies of the v1-v3 runs (names real_*); same bytes as these folders
RUNS = {"real_s2026": "v1_s2026", "real_v1_s7": "v1_s7", "real_v1_s11": "v1_s11", "real_v2_s2026": "v2_s2026",
        "real_v2_s7": "v2_s7", "real_v2_s11": "v2_s11", "real_v3_s2026": "v3_s2026"}
NAMES = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]


def regime(w):
    """Coarse descriptor: objectives holding >= 10 % of the block, largest first."""
    order = np.argsort(-w)
    return "+".join(NAMES[k][:4] for k in order if w[k] >= 0.10)


def main():
    best = json.load(open("best_point.json", encoding="utf-8"))
    Wstar = np.array(best["weights_by_building"], float)
    Rstar = best["R"]
    cand = {}                      # (building, regime) -> (point R, block, source)
    for name, folder in sorted(RUNS.items()):
        pts = [json.loads(l) for l in open(REH / folder / "online_log.jsonl", encoding="utf-8")]
        pts = sorted((p for p in pts if p.get("type") == "point" and p["R"] is not None), key=lambda p: p["R"])[:60]
        for p in pts:
            W = np.array(p["W"]).reshape(3, 7)
            for b in range(3):
                key = (b, regime(W[b]))
                if key not in cand or p["R"] < cand[key][0]:
                    cand[key] = (p["R"], W[b], f"{name}#c{p['call']}")
    X, meta = [], []
    for (b, reg), (r, blk, src) in sorted(cand.items()):
        W = Wstar.copy()
        W[b] = blk
        X.append(W.ravel())
        meta.append((b, reg, r, src, blk))
    pool = SolverPool(n_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
    y = pool.evaluate(np.array(X), tag="E13_alt")
    res = []
    for (b, reg, r, src, blk), yy in zip(meta, y):
        res.append({"building": b + 1, "regime": reg, "delta": float(yy - Rstar), "R_in_rehearsal": r,
                    "source": src, "block": blk.tolist()})
    res.sort(key=lambda d: (d["building"], d["delta"]))
    json.dump(res, open("E13_alt.json", "w", encoding="utf-8"), indent=1)
    np.set_printoptions(precision=3, suppress=True)
    print(f"W* R = {Rstar:.4f}; W* regimes: " + ", ".join(regime(w) for w in Wstar))
    for d in res:
        print(f"b{d['building']} {d['regime']:28s} delta {d['delta']:+7.3f}  (rehearsal point {d['R_in_rehearsal']:.2f}, "
              f"{d['source']})  {np.array(d['block'])}")


if __name__ == "__main__":
    main()
