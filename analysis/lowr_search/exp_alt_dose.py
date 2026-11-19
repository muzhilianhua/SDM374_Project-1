"""E14 (phase 0): are the E13 candidate blocks genuinely different control regimes?

For each candidate block (one building, transplanted into the best point W*):
  dose  - scale its dominant weight by 0 ... 100, other weights fixed. A flat curve means the dominant
          objective is inert (the block is the known valley in another parametrisation); a curve with a
          minimum near x1 and large changes elsewhere means the objective shapes the controller.
  ctx   - additivity recheck in the block's own rehearsal context: R(rehearsal point) minus R(same point with
          this building's block replaced by W*'s block) should equal the E13 delta.
"""
import json
import sys
from pathlib import Path

import numpy as np

from pool import SolverPool

from exp_alt import REH, RUNS
NAMES = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]
PICK = [(3, "disc"), (3, "ramp+peak+degr+cost"), (2, "ramp+carb"), (2, "disc"), (2, "peak"), (1, "ramp+cost")]
FACTORS = [0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2, 3, 5, 10, 30, 100]


def rehearsal_point(source):
    run, call = source.split("#c")
    best = None
    for line in open(REH / RUNS[run] / "online_log.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if p.get("type") == "point" and p["call"] == int(call) and p["R"] is not None:
            if best is None or p["R"] < best["R"]:
                best = p
    return np.array(best["W"]).reshape(3, 7), best["R"]


def main():
    best = json.load(open("best_point.json", encoding="utf-8"))
    Wstar, Rstar = np.array(best["weights_by_building"], float), best["R"]
    e13 = json.load(open("E13_alt.json", encoding="utf-8"))
    cands = [next(d for d in e13 if d["building"] == b and d["regime"] == reg) for b, reg in PICK]
    X, meta = [], []
    for ci, d in enumerate(cands):
        b = d["building"] - 1
        blk = np.array(d["block"])
        k = int(np.argmax(blk))
        for f in FACTORS:
            w = blk.copy()
            w[k] *= f
            if w.sum() <= 0:
                continue
            W = Wstar.copy()
            W[b] = w / w.sum()
            X.append(W.ravel()); meta.append(("dose", ci, f))
        P, RP = rehearsal_point(d["source"])
        # the rehearsal point must contain this block; use the block's own point from the log
        if not np.allclose(P[b], blk, atol=1e-9):
            P = None
        if P is not None:
            Q = P.copy(); Q[b] = Wstar[b]
            X.append(Q.ravel()); meta.append(("ctx", ci, RP))
    pool = SolverPool(n_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
    y = pool.evaluate(np.array(X), tag="E14_dose")
    out = []
    for ci, d in enumerate(cands):
        b = d["building"] - 1
        k = int(np.argmax(d["block"]))
        dose = [(f, float(r - Rstar)) for (kind, c, f), r in zip(meta, y) if kind == "dose" and c == ci]
        ctx = [(float(RP - r)) for (kind, c, RP), r in zip(meta, y) if kind == "ctx" and c == ci]
        out.append({"building": d["building"], "regime": d["regime"], "dominant": NAMES[k],
                    "e13_delta": d["delta"], "dose": dose, "ctx_delta": ctx[0] if ctx else None,
                    "block": d["block"]})
        print(f"b{d['building']} {d['regime']:22s} dominant {NAMES[k]:11s} E13 {d['delta']:+.3f} "
              f"ctx {ctx[0] if ctx else float('nan'):+.3f}")
        print("   x" + "  ".join(f"{f:g}:{v:+.2f}" for f, v in dose))
    json.dump(out, open("E14_dose.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
