"""Render the recorded method comparison with standard scientific plotting tools."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sdm.paths import REPO_ROOT

os.environ.setdefault("MPLCONFIGDIR",str(REPO_ROOT/"analysis/method_expansion/results/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    args=parser.parse_args()
    table=pd.read_csv(args.summary/"model_scores.csv").sort_values("v32_score")
    selection=json.loads((args.summary/"selection.json").read_text())
    ref=selection["reference_metrics"]
    fig,(left,right)=plt.subplots(1,2,figsize=(12,7),gridspec_kw={"width_ratios":[1.2,1]},layout="constrained")
    colors=["#27856b" if x else "#7293b6" for x in table.eligible]
    positions=list(range(len(table)))
    left.scatter(table.v32_score,positions,c=colors,s=45,zorder=3)
    left.set_yticks(positions,labels=table.label)
    left.grid(axis="x",alpha=.15)
    left.axvline(ref["v32_score"],color="#af5139",linestyle="--",linewidth=1.5,label="Existing weighted XGBoost")
    for i,score in enumerate(table.v32_score): left.text(score+.12,i,f"{score:.2f}",va="center",fontsize=8)
    left.set_xlim(max(0,table.v32_score.min()-5),table.v32_score.max()+3)
    left.set_xlabel("Frozen-v3.2 score (higher is better)")
    left.set_title("15 methods: discovery seed 2026",loc="left",fontweight="bold")
    left.legend(loc="lower right",fontsize=8)
    right.scatter(table.mae,table.v32_top1_regret,c=colors,s=55,alpha=.9)
    offsets={"cat-rmse":(4,-12),"cat-weighted":(4,6)}
    for _,row in table.iterrows():
        right.annotate(row.label,(row.mae,row.v32_top1_regret),
            xytext=offsets.get(row.label,(4,4)),textcoords="offset points",fontsize=7)
    right.scatter([ref["mae"]],[ref["v32_top1_regret"]],marker="*",s=180,color="#af5139",zorder=5)
    right.set_xlabel("Global MAE (lower is better)")
    right.set_ylabel("First-choice regret (lower is better)")
    right.set_title("Accuracy and candidate selection",loc="left",fontweight="bold")
    right.set_xlim(table.mae.min()-.08,table.mae.max()+.23)
    right.grid(alpha=.15)
    fig.suptitle("Alternative surrogate methods",fontsize=16,fontweight="bold")
    fig.supxlabel("Green: meets prespecified discovery guard. Reused benchmark; no simulator measurements.",fontsize=9)
    for suffix in ("png","svg"):
        fig.savefig(args.summary/f"method_comparison.{suffix}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
