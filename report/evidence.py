"""Numbers quoted in the report's experiment section, computed from committed real-simulator rehearsal logs.

    uv run python report/evidence.py            # all runs under analysis/online_rehearsal/rehearsals/
    uv run python report/evidence.py --json report/evidence.json   # also write every number as JSON

Definitions (all on measured R only):
- final: best R after 20 calls; curve: best R after each call.
- hit: a point below the R of its call's base (block policies) or below the incumbent (v4j).
- additivity error: measured minus predicted R of base and combination points, whose prediction is the
  incumbent plus the sum of the measured block effects.
- repeat: a point whose R equals, within 1e-9, an R already measured earlier in the same run.
- credit: each lowering of the incumbent is credited to the move class of the point that achieved it.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REH = ROOT / "analysis" / "online_rehearsal" / "rehearsals"
BEST_TRAIN = 115.76481925917324
CLASS = {"model": "model", "random": "random", "fill": "random", "nav": "nav",
         "fine": "fine", "zero": "fine", "tiny": "fine", "grid": "scan", "scan": "scan", "icombo": "scan"}


def load(run: str):
    points = [json.loads(l) for l in (REH / run / "online_log.jsonl").open(encoding="utf-8")]
    return [p for p in points if p["type"] == "point" and p["R"] is not None]


def kind_of(p) -> str:
    m = p["meta"]
    if m["kind"] in ("base", "combo"):
        return "combination"
    if m["kind"] == "jump":
        return "jump"
    return CLASS.get(m.get("move"), "random")


def analyse(run: str) -> dict:
    pts = load(run)
    calls = defaultdict(list)
    for p in pts:
        calls[p["call"]].append(p)
    inc, curve, credit, seen = BEST_TRAIN, [], defaultdict(float), set()
    gain = {"combination_step": 0.0, "moves": 0.0}
    multi_improve = []
    hits = defaultdict(lambda: [0, 0])
    repeats = {"early": [0, 0], "late": [0, 0]}
    add_err = []
    for c in sorted(calls):
        batch = calls[c]
        base = next((p for p in batch if p["meta"]["kind"] == "base"), None)
        ref = base["R"] if base is not None else inc
        if base is not None and base["R"] < inc:
            gain["combination_step"] += inc - base["R"]
        lowest = min(p["R"] for p in batch)
        if lowest < min(inc, ref):
            gain["moves"] += min(inc, ref) - lowest
        multi_improve.append(len({p["meta"]["building"] for p in batch
                                  if p["meta"]["kind"] == "single" and p["R"] < ref - 1e-9}))
        for p in batch:
            k = kind_of(p)
            if k != "combination":
                hits[k][0] += p["R"] < ref - 1e-9
                hits[k][1] += 1
            else:
                hits[k][0] += p["R"] < inc - 1e-9
                hits[k][1] += 1
                if "pred" in p["meta"]:
                    add_err.append(p["R"] - p["meta"]["pred"])
            phase = "late" if c >= 11 else "early"
            r = round(p["R"], 9)
            repeats[phase][0] += r in seen
            repeats[phase][1] += 1
        seen |= {round(p["R"], 9) for p in batch}
        best = min(batch, key=lambda p: p["R"])
        if best["R"] < inc - 1e-9:
            credit[kind_of(best)] += inc - best["R"]
            inc = best["R"]
        curve.append(inc)
    return {"final": inc, "curve": curve, "gain": gain, "multi_improve": multi_improve, "credit": dict(credit), "hits": {k: tuple(v) for k, v in hits.items()},
            "repeat": {k: v[0] / v[1] for k, v in repeats.items() if v[1]}, "add_err": add_err,
            "model_err": [p["R"] - p["meta"]["pred"] for p in pts
                          if p["meta"].get("move") == "model" and "pred" in p["meta"]]}


def permutation_p(a, b, rounds=None) -> float:
    """Two-sided exact permutation p-value for the difference of means (unpaired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, n = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool); mask[list(idx)] = True
        total += 1
        count += abs(pooled[mask].mean() - pooled[~mask].mean()) >= obs - 1e-12
    return count / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {"variants": {}, "tests": {}, "groups": {}}
    runs = sorted(p.name for p in REH.iterdir() if (p / "online_log.jsonl").exists())
    by_variant = defaultdict(dict)
    for run in runs:
        variant, seed = run.rsplit("_s", 1)
        by_variant[variant][seed] = analyse(run)

    print("== finals by variant")
    for v, seeds in sorted(by_variant.items()):
        f = [seeds[s]["final"] for s in sorted(seeds)]
        c = np.array([seeds[s]["curve"] for s in sorted(seeds) if len(seeds[s]["curve"]) == 20])
        at = " / ".join(f"{c[:, i].mean():.2f}" for i in (4, 9, 14)) if len(c) else "-"
        out["variants"][v] = {"n": len(f), "mean": float(np.mean(f)),
                              "sd": float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
                              "min": float(min(f)), "max": float(max(f)),
                              "finals": {s: seeds[s]["final"] for s in sorted(seeds)},
                              "mean_best_after": {str(i + 1): float(c[:, i].mean()) for i in range(20)} if len(c) else {}}
        print(f"{v:4s} n={len(f)} mean {np.mean(f):.3f} sd {np.std(f, ddof=1) if len(f) > 1 else 0:.3f} "
              f"min {min(f):.3f} max {max(f):.3f} | mean best after call 5/10/15: {at} | "
              + " ".join(f"s{s}={seeds[s]['final']:.3f}" for s in sorted(seeds)))
    # "v1+v2+v4" pools every block-descent variant against the joint-move ablation.
    for a, b in (("v4", "v2"), ("v4", "v1"), ("v2", "v1"), ("v4", "v4j"), ("v2", "v4j"), ("v1", "v4j"),
                 ("v1+v2+v4", "v4j")):
        fa = [r["final"] for v in a.split("+") for r in by_variant.get(v, {}).values()]
        fb = [r["final"] for v in b.split("+") for r in by_variant.get(v, {}).values()]
        if len(fa) > 1 and len(fb) > 1:
            out["tests"][f"{a}-{b}"] = {"diff": float(np.mean(fa) - np.mean(fb)), "p": permutation_p(fa, fb),
                                        "n": [len(fa), len(fb)]}
            print(f"{a} - {b}: mean diff {np.mean(fa) - np.mean(fb):+.3f}, permutation p = {permutation_p(fa, fb):.3f} "
                  f"(n = {len(fa)}, {len(fb)})")

    for group in (("v1", "v2", "v4"), ("v4",), ("v2",), ("v4j",)):
        rs = [r for v in group for r in by_variant.get(v, {}).values()]
        if not rs:
            continue
        print(f"\n== within-run evidence, {'+'.join(group)} ({len(rs)} runs)")
        hits = defaultdict(lambda: [0, 0])
        credit = defaultdict(float)
        for r in rs:
            for k, (h, n) in r["hits"].items():
                hits[k][0] += h; hits[k][1] += n
            for k, v in r["credit"].items():
                credit[k] += v
        total = sum(credit.values())
        g = {"hits": {k: list(v) for k, v in hits.items()}, "credit": {k: v / total for k, v in credit.items()}}
        for k in sorted(hits, key=lambda k: -hits[k][1]):
            h, n = hits[k]
            print(f"  {k:12s} points {n:5d} ({n / sum(x[1] for x in hits.values()):5.1%})  hit rate {h / n:6.1%}  "
                  f"credit {credit.get(k, 0) / total:6.1%} of {total / len(rs):.2f} per run")
        cs = sum(r["gain"]["combination_step"] for r in rs); mv = sum(r["gain"]["moves"] for r in rs)
        mi = np.array([x for r in rs for x in r["multi_improve"]])
        g["combination_step_share"] = cs / (cs + mv) if cs + mv else 0.0
        g["calls_two_plus_buildings_improve"] = float(np.mean(mi >= 2))
        print(f"  descent from the combination step {g['combination_step_share']:.1%}; calls with >=2 buildings "
              f"improving {g['calls_two_plus_buildings_improve']:.1%}")
        e = np.array([x for r in rs for x in r["add_err"]])
        m = np.array([x for r in rs for x in r["model_err"]])
        if len(e):
            print(f"  combinations: n={len(e)} median err {np.median(e):+.3f} MAE {np.abs(e).mean():.3f} "
                  f"|err|<0.1: {np.mean(np.abs(e) < 0.1):.1%}")
        if len(m):
            print(f"  surrogate picks: n={len(m)} mean err {m.mean():+.2f} MAE {np.abs(m).mean():.2f}")
        rep = {ph: np.mean([r["repeat"][ph] for r in rs]) for ph in ("early", "late")}
        g.update(repeat={k: float(v) for k, v in rep.items()},
                 additive={"n": int(len(e)), "median": float(np.median(e)) if len(e) else None,
                           "mae": float(np.abs(e).mean()) if len(e) else None,
                           "within_0_1": float(np.mean(np.abs(e) < 0.1)) if len(e) else None},
                 surrogate={"n": int(len(m)), "mean": float(m.mean()) if len(m) else None,
                            "mae": float(np.abs(m).mean()) if len(m) else None})
        out["groups"]["+".join(group)] = g
        print(f"  repeated-R share: calls 1-10 {rep['early']:.1%}, calls 11-20 {rep['late']:.1%}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
