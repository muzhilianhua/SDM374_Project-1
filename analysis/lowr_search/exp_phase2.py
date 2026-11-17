"""E18 (phase 2): systematic search for other valleys, judged after local descent, with mode combinations.

Context: the current best point B (best_point_E17.json, 103.2473). E15-E17 showed that (a) a region's value is
only visible after local refinement (raw samples understate it by 0.4-0.9), and (b) buildings interact across
valleys (a building-1 block moved by 0.85 when building 3 changed valley), so modes must be combined and
measured, not added.

  A  discovery   per building, 500 blocks with the other two at B: 70 % sparse blocks (1-4 objectives, log-uniform
                 ratios over 3 decades), 30 % "dominant + mixture" blocks (one objective 0.6-0.95, the rest a
                 sparse log-uniform block). Niche = objectives holding >= 10 % of the block (as in E13).
  B  refinement  per building, the best raw sample of the 5 best niches other than B's own niche, descended for
                 2 rounds (exp_alt_descent.descend) at B.
  C  combination per building {B's block} + the 2 best refined blocks; all 27 combinations measured; if one
                 beats B, one block-coordinate cycle (descend all three buildings, 2 rounds) and a final
                 combination measurement.
State is saved after each stage (E18_*.json); a rerun reuses the pool cache, so finished work costs nothing.
"""
import itertools
import json
import sys

import numpy as np

from exp_alt_descent import descend, norm
from pool import SolverPool

NAMES = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]
N_DISCOVER = 500
N_NICHES = 5


def niche(w):
    order = np.argsort(-w)
    return "+".join(NAMES[k][:4] for k in order if w[k] >= 0.10)


def sample_block(rng):
    if rng.random() < 0.7:
        s = rng.choice([1, 2, 3, 4], p=[0.1, 0.35, 0.35, 0.2])
        sup = rng.choice(7, size=s, replace=False)
        w = np.zeros(7)
        w[sup] = 10 ** rng.uniform(-3, 0, s)
        return norm(w)
    k = rng.integers(7)
    rest = np.zeros(7)
    s = rng.choice([1, 2, 3])
    sup = rng.choice([j for j in range(7) if j != k], size=s, replace=False)
    rest[sup] = 10 ** rng.uniform(-3, 0, s)
    rest = rest / rest.sum()
    a = rng.uniform(0.6, 0.95)
    w = (1 - a) * rest
    w[k] += a
    return norm(w)


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    best = json.load(open("best_point_E17.json", encoding="utf-8"))
    B, RB = np.array(best["weights_by_building"], float), best["R"]
    B = B / B.sum(axis=1, keepdims=True)
    pool = SolverPool(n_workers=n_workers)
    rng = np.random.default_rng(18)

    # ---------------------------------------------------------------- A: discovery
    X, meta = [], []
    for b in range(3):
        for _ in range(N_DISCOVER):
            w = sample_block(rng)
            W = B.copy(); W[b] = w
            X.append(W.ravel()); meta.append((b, w))
    y = pool.evaluate(np.array(X), tag="E18_discover")
    disc = [{"building": b + 1, "niche": niche(w), "delta": float(r - RB), "block": w.tolist()}
            for (b, w), r in zip(meta, y) if np.isfinite(r)]
    json.dump(disc, open("E18_discover.json", "w", encoding="utf-8"), indent=0)
    seeds = {}
    for b in range(3):
        own = niche(B[b])
        best_per = {}
        for d in disc:
            if d["building"] == b + 1 and d["niche"] != own:
                if d["niche"] not in best_per or d["delta"] < best_per[d["niche"]]["delta"]:
                    best_per[d["niche"]] = d
        ranked = sorted(best_per.values(), key=lambda d: d["delta"])
        ds = np.array([d["delta"] for d in disc if d["building"] == b + 1])
        print(f"A b{b + 1}: own niche {own}; {len(best_per)} other niches; raw delta quantiles "
              f"min/5%/50% {np.round(np.percentile(ds, [0, 5, 50]), 2).tolist()}", flush=True)
        for d in ranked[:N_NICHES]:
            print(f"     {d['niche']:28s} raw {d['delta']:+.3f}", flush=True)
            seeds[f"b{b + 1}:{d['niche']}"] = (b, np.array(d["block"]))

    # ---------------------------------------------------------------- B: refinement
    st = descend(B, RB, seeds, "E18_refine", "E18_refine.json", max_rounds=2, n_workers=n_workers, rng_seed=181)

    # ---------------------------------------------------------------- C: combination
    options = []
    for b in range(3):
        mine = sorted(((s["d"], s["w"]) for n, s in st.items() if s["b"] == b), key=lambda t: t[0])[:2]
        options.append([B[b]] + [w for _, w in mine])
    cands = [np.array(c) for c in itertools.product(*options)]
    y = pool.evaluate(np.array([c.ravel() for c in cands]), tag="E18_combo")
    order = np.argsort(y)
    print("C combinations (best 5):", [round(float(y[i]), 4) for i in order[:5]], f"B = {RB:.4f}", flush=True)
    best_W, best_R = (cands[order[0]], float(y[order[0]])) if y[order[0]] < RB - 1e-9 else (B, RB)
    if best_R < RB - 1e-9:
        seeds = {f"b{b + 1}": (b, norm(best_W[b])) for b in range(3)}
        st2 = descend(best_W, best_R, seeds, "E18_cycle", "E18_cycle.json", max_rounds=2,
                      n_workers=n_workers, rng_seed=182)
        opts = [[best_W[b]] + ([st2[f"b{b + 1}"]["w"]] if st2[f"b{b + 1}"]["d"] < -1e-9 else []) for b in range(3)]
        cands = [np.array(c) for c in itertools.product(*opts)]
        if len(cands) > 1:
            y = pool.evaluate(np.array([c.ravel() for c in cands]), tag="E18_combo2")
            i = int(np.argmin(y))
            if y[i] < best_R - 1e-9:
                best_W, best_R = cands[i], float(y[i])
    json.dump({"R": best_R, "improved": best_R < RB - 1e-9, "base_R": RB, "weights_by_building": best_W.tolist()},
              open("E18_result.json", "w", encoding="utf-8"), indent=1)
    np.set_printoptions(precision=6, suppress=True)
    print(f"final R {best_R:.10f} (base {RB:.10f})", flush=True)
    print(best_W)


if __name__ == "__main__":
    main()
