"""E15 (phase 1): per-building local descent from the alternative-valley seeds of E13/E14.

Each seed is one building's block; the other two buildings stay at the best point W*, so R - R* is the block's
value relative to W*'s block (additivity near W*). Per round and seed:
  * a one-coordinate scan of all 7 weights (positive: x0, x0.3 ... x3; zero: tiny values 1e-5 .. 3e-2 of
    degradation + ramping), the move type that found E10's gains;
  * combinations of the best improving coordinates of the previous round;
  * 12 small random moves (all positive weights x 10^N(0, sigma), sigma 0.002 / 0.005 / 0.01).
A seed moves to its best point after every round and stops after a round without improvement (max 6 rounds).
All seeds of a round are evaluated together, so the pool stays busy. Reruns cost nothing (pool cache).
"""
import json
import sys

import numpy as np

from pool import SolverPool

NAMES = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]
POS_F = [0, 0.3, 0.5, 0.7, 0.85, 0.93, 0.97, 1.03, 1.07, 1.15, 1.4, 2, 3]
TINY = np.geomspace(1e-5, 3e-2, 10)
MAX_ROUNDS = 6


def norm(w):
    w = np.asarray(w, float)
    w = np.where(w < 1e-12, 0.0, w)
    return w / w.sum()


def seeds():
    e13 = {(d["building"], d["regime"]): d for d in json.load(open("E13_alt.json", encoding="utf-8"))}
    def blk(b, reg, k=None, f=1.0):
        w = np.array(e13[(b, reg)]["block"], float)
        if k is not None:
            w[k] *= f
        return b - 1, norm(w)
    return {
        "b3_disc_x0.01": blk(3, "disc", 3, 0.01),
        "b3_disc": blk(3, "disc"),
        "b3_ramp_peak_degr_cost": blk(3, "ramp+peak+degr+cost"),
        "b2_ramp_carb": blk(2, "ramp+carb"),
        "b2_disc": blk(2, "disc"),
        "b2_peak_x0.1": blk(2, "peak", 2, 0.1),
        "b1_ramp_cost_x0.85": blk(1, "ramp+cost", 5, 0.85),
    }


def apply(w, ops):
    v = w.copy()
    for k, val in ops:
        v[k] = val
    return norm(v) if v.sum() > 0 else None


def neighbours(w, rng, combos):
    """(label, ops, block); ops = [(coordinate, new unnormalised value)] applied to w."""
    out = []
    s = w[4] + w[5]
    for k in range(7):
        if w[k] > 0:
            for f in POS_F:
                ops = [(k, w[k] * f)]
                v = apply(w, ops)
                if v is not None:
                    out.append((f"{NAMES[k]}x{f:g}", ops, v))
        else:
            for t in TINY:
                ops = [(k, t * max(s, 1e-3))]
                out.append((f"{NAMES[k]}={t:.1e}", ops, apply(w, ops)))
    for name, ops in combos:
        v = apply(w, ops)
        if v is not None:
            out.append((name, ops, v))
    pos = w > 0
    for sig in (0.002, 0.005, 0.01):
        for _ in range(4):
            v = w.copy(); v[pos] *= 10 ** rng.normal(0, sig, pos.sum())
            out.append((f"jitter{sig}", None, norm(v)))
    return out


def descend(Wstar, Rstar, seed_blocks, tag, out, max_rounds=MAX_ROUNDS, n_workers=10, rng_seed=15):
    """Per-building descent of each seed (building index, block) with the other buildings at Wstar."""
    pool = SolverPool(n_workers=n_workers)
    rng = np.random.default_rng(rng_seed)
    st = {}
    for name, (b, w) in seed_blocks.items():
        st[name] = {"b": b, "w": w, "d": None, "active": True, "combos": [], "hist": []}
    # round 0: measure the seeds themselves
    X = []
    for s in st.values():
        W = Wstar.copy(); W[s["b"]] = s["w"]; X.append(W.ravel())
    y = pool.evaluate(np.array(X), tag=f"{tag}_seed")
    for s, r in zip(st.values(), y):
        s["d"] = float(r - Rstar); s["hist"].append(s["d"])
    print("seeds:", {n: round(s["d"], 4) for n, s in st.items()}, flush=True)
    for rnd in range(1, max_rounds + 1):
        batch, owner = [], []
        for n, s in st.items():
            if not s["active"]:
                continue
            for lab, ops, v in neighbours(s["w"], rng, s["combos"]):
                W = Wstar.copy(); W[s["b"]] = v
                batch.append(W.ravel()); owner.append((n, lab, ops, v))
        if not batch:
            break
        y = pool.evaluate(np.array(batch), tag=f"{tag}_r{rnd}")
        for n, s in st.items():
            if not s["active"]:
                continue
            res = [(float(r - Rstar), lab, ops, v) for (o, lab, ops, v), r in zip(owner, y) if o == n and np.isfinite(r)]
            res.sort(key=lambda t: t[0])
            imp = [t for t in res if t[0] < s["d"] - 1e-9]
            if imp:
                old_w = s["w"]
                s["d"], lab, _, s["w"] = imp[0]
                # next round: the best improving single-coordinate moves of this round, applied together
                # (ops are unnormalised values on old_w; rescale them to the new block's scale)
                scale = s["w"][4] + s["w"][5]
                old_scale = old_w[4] + old_w[5]
                parts, seen = [], set()
                for d, l, ops, v in imp:
                    if ops is None or ops[0][0] in seen or len(ops) != 1:
                        continue
                    seen.add(ops[0][0]); parts.append(ops[0])
                    if len(parts) == 4:
                        break
                s["combos"] = []
                ratio = (scale / old_scale) if old_scale > 0 else 1.0
                rel = [(k, val / old_w.sum()) for k, val in parts]      # old_w is normalised: sum 1
                for i in range(len(rel)):
                    for j in range(i + 1, len(rel)):
                        s["combos"].append((f"combo:{NAMES[rel[i][0]]}+{NAMES[rel[j][0]]}",
                                            [(rel[i][0], rel[i][1] * ratio), (rel[j][0], rel[j][1] * ratio)]))
                if len(rel) >= 3:
                    s["combos"].append(("combo:top3", [(k, v * ratio) for k, v in rel[:3]]))
                print(f"round {rnd} {n}: {s['d']:+.4f} via {lab} ({len(imp)} improving of {len(res)})", flush=True)
            else:
                s["active"] = False
                print(f"round {rnd} {n}: no improvement, stop at {s['d']:+.4f}", flush=True)
            s["hist"].append(s["d"])
        json.dump({n: {"building": s["b"] + 1, "delta": s["d"], "R": Rstar + s["d"], "hist": s["hist"],
                       "block": s["w"].tolist()} for n, s in st.items()},
                  open(out, "w", encoding="utf-8"), indent=1)
    print("final:", {n: round(s["d"], 4) for n, s in st.items()}, flush=True)
    return st


def main():
    best = json.load(open("best_point.json", encoding="utf-8"))
    Wstar, Rstar = np.array(best["weights_by_building"], float), best["R"]
    descend(Wstar, Rstar, seeds(), "E15", "E15_descent.json",
            n_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)


if __name__ == "__main__":
    main()
