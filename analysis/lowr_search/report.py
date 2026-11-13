"""Summarise the search: best points, convergence, operator statistics, figure.

    python report.py            # prints a summary, writes best_point.json and convergence.png
"""
import collections
import json
from pathlib import Path

import numpy as np

from common import NAMES
from pool import load_log, normalize

HERE = Path(__file__).resolve().parent
np.set_printoptions(precision=5, suppress=True, linewidth=220)

team = [json.loads(l) for l in (HERE.parent / "replay" / "evidence" / "eval_log.jsonl").open(encoding="utf-8")]
Rt = np.array([r["R"] for r in team])
W, R, tags = load_log()
print(f"team log: {len(Rt)} evaluations, best {Rt.min():.6f}")
print(f"this harness: {len(R)} evaluations, best {R.min():.6f}  (improvement {Rt.min() - R.min():+.4f})")
for thr in (104.8517, 104.5, 104.0, 103.5, 103.0):
    print(f"  points below {thr}: {(R < thr - 1e-9).sum()}")

order = np.argsort(R)
print("\ntop distinct values:")
seen, shown = set(), 0
for i in order:
    v = round(R[i], 6)
    if v in seen:
        continue
    seen.add(v); shown += 1
    print(f"  {R[i]:.6f}  [{tags[i]}]")
    for b in range(3):
        print("      b%d " % (b + 1), W[i].reshape(3, 7)[b])
    if shown == 5:
        break

i = int(np.argmin(R))            # first evaluation that reached the minimum (ties share a cell)
best = {"R": float(R[i]), "tag": tags[i], "objectives": NAMES,
        "weights_by_building": W[i].reshape(3, 7).tolist(), "row_sums": W[i].reshape(3, 7).sum(1).tolist(),
        "evaluations_this_harness": int(len(R)), "evaluations_team_log": int(len(Rt)),
        "note": "exploration result from local solver.exe runs; not an input to the graded online policy (decisions 19, 29)"}
(HERE / "best_point.json").write_text(json.dumps(best, indent=1), encoding="utf-8")

prog = HERE / "fms_progress.jsonl"
if prog.exists():
    gens = [json.loads(l) for l in prog.open(encoding="utf-8")]
    ops = collections.OrderedDict()
    imp = np.zeros(3, int)
    for g in gens:
        imp += np.array(g["improving"])
        for op, (n, k, dm) in g["ops"].items():
            s = ops.setdefault(op, [0, 0, 0.0]); s[0] += n; s[1] += k; s[2] = min(s[2], dm)
    print(f"\nFMS: {len(gens)} generations, {sum(g['new_evals'] for g in gens)} evaluations, "
          f"{sum(g['gain'] > 0 for g in gens)} generations improved; improving candidates per building {imp.tolist()}")
    print("operator            trials  improved   rate    best gain")
    for op, (n, k, dm) in sorted(ops.items(), key=lambda t: -t[1][1] / max(t[1][0], 1)):
        print(f"  {op:16s} {n:6d} {k:8d}  {k / max(n, 1):6.3f}   {-dm:9.4f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    allR = np.concatenate([Rt, R])
    cm = np.minimum.accumulate(allR)
    ax[0].plot(np.arange(1, len(allR) + 1), cm, lw=1.5)
    ax[0].axvline(len(Rt), color="gray", ls="--", lw=0.8)
    ax[0].set_ylim(cm.min() - 0.5, 112); ax[0].set_xlabel("cumulative evaluations (team log, then this harness)")
    ax[0].set_ylabel("best R so far"); ax[0].set_title("Convergence")
    ax[1].hist(R[R < Rt.min() + 3], bins=80)
    ax[1].axvline(Rt.min(), color="gray", ls="--", lw=0.8)
    ax[1].set_xlabel("R of points evaluated by this harness (within 3 of the team best)"); ax[1].set_title("Where the samples land")
    fig.tight_layout(); fig.savefig(HERE / "convergence.png", dpi=130)
    print("\nwrote convergence.png, best_point.json")
except Exception as exc:                                    # plotting is optional
    print("plot skipped:", exc)
