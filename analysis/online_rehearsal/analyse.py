"""Summarise rehearsal runs.

    python analyse.py                 # table over variants and seeds
    python analyse.py --plot          # same, and redraw convergence.png
    python analyse.py v2_s7 v1_s2026  # per run: best-so-far per call, move types, model prediction error
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(run):
    recs = [json.loads(l) for l in (HERE / "rehearsals" / run / "online_log.jsonl").open(encoding="utf-8")]
    return [r for r in recs if r["type"] == "point"], [r for r in recs if r["type"] == "call_done"]


def details(run):
    pts, calls = load(run)
    best, curve = 115.76481925917324, []
    wins = defaultdict(lambda: [0, 0])
    for c in calls:
        cp = [p for p in pts if p["call"] == c["call"] and p["R"] is not None]
        for p in cp:
            m = p["meta"]
            k = m["kind"] if m["kind"] != "single" else m["move"]
            wins[k][1] += 1
            if p["R"] < best - 1e-9:
                wins[k][0] += 1
        best = min(best, min(p["R"] for p in cp))
        curve.append(best)
    print(f"== {run}: final {best:.4f}")
    print("best after call:", " ".join(f"{v:.2f}" for v in curve))
    print("calls to reach  <115.68: %s  <110: %s  <108: %s  <107.5: %s" % tuple(
        next((i + 1 for i, v in enumerate(curve) if v < t), "-") for t in (115.68, 110, 108, 107.5)))
    print("move: new-best hits / points:", {k: f"{a}/{n}" for k, (a, n) in sorted(wins.items())})
    for kind in ("base", "combo", "jump"):
        e = [p["R"] - p["meta"]["pred"] for p in pts if p["meta"]["kind"] == kind and p["R"] is not None
             and "pred" in p["meta"]]
        if e:
            print(f"{kind}: n={len(e)} real-pred mean {np.mean(e):+.3f} median {np.median(e):+.3f}")
    e = [(c, p["R"] - p["meta"]["pred"]) for p in pts for c in [p["call"]]
         if p["meta"].get("move") == "model" and p["R"] is not None]
    for lo, hi in ((1, 5), (6, 10), (11, 20)):
        v = [x for c, x in e if lo <= c <= hi]
        print(f"model picks calls {lo}-{hi}: real-pred mean {np.mean(v):+.2f}  MAE {np.mean(np.abs(v)):.2f}")
    b = min((p for p in pts if p["R"] is not None), key=lambda p: p["R"])
    W = np.array(b["W"]).reshape(3, 7)
    np.set_printoptions(precision=4, suppress=True)
    print("best W (cost carbon peak discomfort degradation ramping resilience):")
    print(W)


# ---------------------------------------------------------------- summary table and figure
RUNS = {"v1": ["v1_s2026", "v1_s7", "v1_s11"], "v2": ["v2_s2026", "v2_s7", "v2_s11"], "v3": ["v3_s2026"]}


def best_curve(run):
    _, calls = load(run)
    best, out = 115.76481925917324, []       # best training row (p01969)
    for c in calls:
        best = min(best, c["batch_min"])
        out.append(best)
    return np.array(out)


def summary(plot: bool):
    for v, runs in RUNS.items():
        C = np.array([best_curve(r) for r in runs])
        print(f"{v}: final {np.round(C[:, -1], 3).tolist()} mean {C[:, -1].mean():.3f} | "
              f"mean best after call 5/10/15: {np.round(C[:, [4, 9, 14]].mean(0), 2).tolist()}")
    if not plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for v, runs in RUNS.items():
        for i, run in enumerate(runs):
            c = best_curve(run)
            ax.plot(range(1, len(c) + 1), c, color={"v1": "C0", "v2": "C1", "v3": "C2"}[v],
                    ls=["-", "--", ":"][i], marker="o", ms=2, label=f"{v} seed {run.split('_s')[1]}: {c[-1]:.2f}")
    ax.axhline(115.68, ls="--", c="gray", lw=1)
    ax.text(20, 115.9, "best offline result 115.68", ha="right", fontsize=8)
    ax.axhline(104.37, ls=":", c="gray", lw=1)
    ax.text(20, 104.55, "lowest point found by analysis/lowr_search (about 15,000 evaluations)", ha="right", fontsize=8)
    ax.set_xlabel("call (50 points each)")
    ax.set_ylabel("best R so far")
    ax.set_xticks(range(1, 21, 2))
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(HERE / "convergence.png", dpi=130)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plot"]
    for run in args:
        details(run)
    if not args:
        summary(plot="--plot" in sys.argv)
