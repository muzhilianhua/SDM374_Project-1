"""Summarise preds.csv from compare.py: accuracy, calibration and within-call selection quality per model.

    uv run python analysis/online_rehearsal/surrogate_compare/summarise.py > analysis/online_rehearsal/surrogate_compare/summary.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "preds.csv")
d["err"] = d.R - d.pred                      # > 0: model optimistic
d["z"] = d.err / d["std"].clip(lower=1e-9)
d["hit"] = d.R < d.R_inc
d["phase"] = pd.cut(d.call, [1, 6, 13, 20], labels=["calls 2-6", "calls 7-13", "calls 14-20"])
d["unscreened"] = d.move != "model"
ORDER = ["t37", "t37_frozen", "azq_fixed", "hp_global", "trend_hp"]
K = 5


def per_call(g):
    out = {"spearman": g.R.corr(g.pred, method="spearman") if len(g) > 4 else np.nan}
    for f in (0.0, 1.0, 2.0):
        top = g.nsmallest(K, "pred") if f == 0 else g.assign(l=g.pred - f * g["std"]).nsmallest(K, "l")
        out[f"lift{f:g}"] = top.R.mean() - g.R.mean()          # < 0: picks better than the call's average
        out[f"hits{f:g}"] = top.hit.sum()
        out[f"regret{f:g}"] = top.R.min() - g.R.min()
    out["hits_all"] = g.hit.sum()
    out["n"] = len(g)
    return pd.Series(out)


def table(sub, label):
    rows = []
    for m in ORDER:
        g = sub[sub.model == m]
        pc = g.groupby(["run", "call"]).apply(per_call)
        rows.append(dict(
            model=m, n=len(g),
            bias=g.err.mean(), mae=g.err.abs().mean(), med_abs=g.err.abs().median(),
            z_mean=g.z.mean(), z_sd=g.z.std(), cover95=(g.z.abs() < 1.96).mean(),
            spearman=pc.spearman.mean(),
            lift_mean=pc.lift0.mean(), lift_lcb1=pc.lift1.mean(), lift_lcb2=pc.lift2.mean(),
            hit5_mean=pc.hits0.sum() / (K * len(pc)), hit5_lcb1=pc.hits1.sum() / (K * len(pc)),
            hit5_lcb2=pc.hits2.sum() / (K * len(pc)), hit_base=pc.hits_all.sum() / pc.n.sum(),
            regret_lcb1=pc.regret1.mean()))
    t = pd.DataFrame(rows).set_index("model")
    print(f"\n### {label}  ({sub[['run', 'call']].drop_duplicates().shape[0]} run-calls)")
    print(t.round(3).to_string())


pd.set_option("display.width", 250)
print("runs:", d.run.nunique(), " points per model:", len(d) // d.model.nunique())
print("err = R - pred (> 0 optimistic); z = err/std; lift = mean R of top-5 picks - mean R of the call's points;"
      " hit = R below best-so-far; lcbF = rank by pred - F*std")
table(d, "all measured points")
table(d[d.unscreened], "unscreened points (move != model; free of T37's selection bias)")
for ph in d.phase.cat.categories:
    table(d[d.unscreened & (d.phase == ph)], f"unscreened, {ph}")
v4 = d[d.run.str.startswith("v4") & d.unscreened]
table(v4, "unscreened, v4 runs only")
