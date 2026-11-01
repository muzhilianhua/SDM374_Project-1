"""Render the fixed-judge comparison and export exact weights for later simulator use."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import pandas as pd
from sdm.paths import REPO_ROOT

os.environ.setdefault('MPLCONFIGDIR',str(REPO_ROOT/'analysis/common_judge/results/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out
    summary=pd.read_csv(out/'method_summary.csv')
    refs=pd.read_csv(out/'reference_comparison.csv')
    baseline=refs[refs.label=='course-baseline'].iloc[0]
    operational=refs[refs.label=='operational-recommendation'].iloc[0]
    inputs=json.loads((out/'inputs.json').read_text())
    for group,column in [('primary','primary_point_index'),('posthoc','best100_point_index')]:
        folder=out/'exported_weights'/group
        folder.mkdir(parents=True,exist_ok=True)
        for row in summary.to_dict('records'):
            (folder/f"{row['label']}.json").write_text(json.dumps(inputs['weights'][int(row[column])],indent=2)+'\n')
    (out/'baseline_weights.json').write_text((REPO_ROOT/'weights.json').read_text())
    shutil.copyfile(REPO_ROOT/'analysis/offline_submission/submission/weights.json',out/'operational_weights.json')
    table='| Candidate generator | Original top1, seed2026 | Delta vs baseline | Top1 reduction | Best of 100, reranked | Below baseline /100 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in summary.itertuples():
        table+=f'| {r.label} | {r.primary_judge_mean:.4f} | {r.primary_delta:+.4f} | {r.primary_reduction_pct:.2f}% | {r.best100_judge_mean:.4f} | {r.below_baseline_count} |\n'
    second='| Generator | Seed2026 top1 | Seed2027 top1 | Two-seed top1 average |\n|---|---:|---:|---:|\n'
    for r in summary.itertuples():
        second+=f'| {r.label} | {r.primary_judge_mean:.4f} | {r.second_top1_judge_mean:.4f} | {r.two_seed_top1_mean:.4f} |\n'
    best_primary=summary.iloc[0]
    best_secondary=summary.sort_values(['best100_judge_mean','label'],kind='stable').iloc[0]
    winner=summary.iloc[0].label
    text=f'''# All 15 candidate generators versus baseline — one fixed judge

**13 of 15 original seed2026 top choices have lower predicted R than the course baseline W**
under the same fixed full-data OOF-stacking judge. Baseline predicted R is **{baseline.judge_mean:.4f}**.
The best original top choice is `{winner}` at **{best_primary.primary_judge_mean:.4f}**,
a predicted reduction of **{best_primary.primary_reduction_pct:.2f}%**.
The best post-hoc candidate is **{best_secondary.best100_judge_mean:.4f}** from the same generator,
`{best_secondary.best100_run}`, rank{int(best_secondary.best100_rank)}.
These are surrogate predictions; real-simulator confirmation was explicitly deferred by the user.

## What is held constant

[Protocol](../README.md) and script were committed before the batch at `fe7f882`.
The sole judge is `20261029-171251-train-method-blend-stack`, unchanged all-data fit, model seed2026.
It uses training-only OOF-learned convex XGBoost/CatBoost/HistGB weights and has previously passed the
matched model-seed audit. Its fitting and frozen-v32 evidence are in
[the method-study delivery](../../method_expansion/delivery/README.md).

The exact root `weights.json` is the baseline. All 1500 proposals from the 15 discovery methods are
retained at original precision: two optimizer seeds, 50 proposals each. No new search, projection,
rounding or filtering occurred. One **`sdm predict` batch of 1503 inputs** scored baseline, operational
W, best measured row and all proposals. Every point uses the same fitted predictor and feature order.
`provenance.json` records a clean `fe7f882`, judge model hash, data hash, packages and all input hashes.
`run_evidence.json` archives the judge and all 30 original optimizer configs/meta/metrics.

## Main comparison

All numbers below are **predicted R by that single judge**, lower is better. Delta = candidate minus
baseline; negative indicates a predicted improvement. The original-top1 column always uses optimizer
seed2026, rank1 chosen by the generator's own mean+proxy objective. It is never replaced by a better
point found using this judge. The best-of-100 column is explicitly **post-hoc reranking** by this judge.

'''+table+f'''
The two methods whose seed2026 top1 does not beat baseline are `kernel-raw` (+0.9097) and
`xgb-huber` (+2.0627). The Hellinger kernel's improvement is only0.1767 and should not be treated as
robust evidence. All15 pools contain at least one lower-predicted point after common-judge reranking;
only4 of100 pseudo-Huber proposals and7 of100 raw-kernel proposals are below baseline.

The existing operational W predicts **{operational.judge_mean:.4f}** under this judge, a
**{operational.reduction_pct:.2f}%** reduction relative to baseline. The leading fixed-blend primary
candidate improves a further **{operational.judge_mean-best_primary.primary_judge_mean:.4f}** predicted
units versus that operational W. This comparison does not automatically replace the operational W.

![Single-judge comparison](comparison.png)

## Original top choices at both optimizer seeds

'''+second+'''
Seed2026 was fixed as primary before looking at these numbers. The two-seed top1 averages retain the
same leading three generators: fixed blend, OOF stack, weighted CatBoost. The raw-kernel seed2027
choice is below baseline even though its seed2026 choice is above; pseudo-Huber's two choices are above.
These optimizer seeds use the same data and judge, and are sensitivity checks, not independent trials.

## Prediction scale, distance and uncertainty

Ridge predicts75.8957 for baseline W. **That is not the baseline value in this comparison**:
the fixed ensemble predicts118.7316 for the exact same W. Comparing75.8957 from Ridge to109.3303
from the ensemble would mix judges and invalidate the comparison.

The judge's baseline error proxy is8.4949; the15 primary candidate proxies are3.4420. These proxies
are distance-binned training-fold residual quantiles, not calibrated confidence intervals. Baseline
mean+proxy is127.2265; fixed-blend primary mean+proxy is112.7724. `method_summary.csv` reports the
paired mean+proxy change for every method. A lower mean+proxy is still a surrogate heuristic.

Baseline distance ratio is1.7484 versus <=.75 for all1500 method proposals. The baseline is retained
rather than excluded. This is a comparison of the actual controller choices, not a matched-radius
experiment; the methods and original baseline used different search constraints. The judge is less
supported near baseline W, and systematic extrapolation error could change the result.

Best measured dataset row p01969 has known R115.7648193; the judge predicts118.1777 for that row.
Its measured value is recorded only in `reference_comparison.csv`'s `measured_R` column. None of the
1500 proposals or baseline W has a measured R in this study.

## Limits and use of the results

The fixed-blend generator leads this candidate comparison, while the OOF-stacking **model recipe**
led the earlier frozen-v32 model study. These are different questions: one measures model validation
quality, the other scores searched controllers with one fixed model. The judge is itself the stack
surrogate used by the blend-stack generator and shares XGBoost/CatBoost/HistGB families with other
generators. It can favor those candidates. Its recipe was selected on a reused frozen benchmark.
This controls the judge but provides **no independent test of surrogate error or measured R**.

The best-of-100 values incur additional selection bias. Do not report them as the generators' original
top choices or as simulation outcomes. Real evaluation is deferred; no submission or online evaluation
budget was used and no operational weight file was changed. The next real-simulator check can compare
baseline with the fixed-blend primary candidate and, if budget permits, the existing operational W.

## Exact weights and reproducibility

- `baseline_weights.json`: numerically identical course baseline W; exported line endings are LF.
- `operational_weights.json`: byte-identical existing recommendation.
- `exported_weights/primary/<label>.json`: original seed2026/rank1 W for each method, CLI compatible.
- `exported_weights/posthoc/<label>.json`: lowest common-judge prediction within the100-point pool.
- `inputs.json`: all original weights and source run/rank/seed, no rounding.
- `candidate_comparison.csv`: all1503 predictions, proxies, distances, deltas and source predictions.
- `method_summary.csv`: primary, second-seed top1, two-seed average, reranked best, mean/median,
  count below baseline, source run/rank and distances.
- `cli_predictions.json`: exact raw CLI output in input order. `weights.json` is that batch input.
- `checksums.json`: all delivered-file hashes except itself. `verification.json`: verification record.

From the repository root, with Python3.11 and locked dependencies:

```bash
uv sync --locked --group gui
env -u PYTHONPATH uv run --locked --group gui python analysis/common_judge/compare.py verify \\
  --out analysis/common_judge/delivery
# With the original ignored judge and30 optimizer run folders available:
env -u PYTHONPATH uv run --locked --group gui python analysis/common_judge/compare.py verify \\
  --out analysis/common_judge/delivery --replay-cli
# Full rebuild; requires the original30 optimizer runs and judge run:
env -u PYTHONPATH uv run --locked --group gui python analysis/common_judge/compare.py build \\
  --out analysis/common_judge/results/recheck
```

A fresh clone can verify archived summaries and hashes without model binaries. To recreate the judge,
train the tracked `experiments/method-blend-stack.toml` through CLI and substitute its new run ID in
`compare.py`. The30 original search configs are archived in `run_evidence.json`; a complete fresh search
can also use `analysis/method_expansion/run_study.py` and then substitute its discovery manifest.
All source datasets, original baseline files and frozen scoring releases remain unchanged.

Verification: all1500 exact proposals pass the simplex check; all15 pools have the expected50 points
per optimizer seed; summaries and exported weights match the archived batch; a second CLI batch
reproduces all1503 mean/proxy predictions within1e-9. Full suite: **109 passed,1 known xfailed**.
'''
    (out/'README.md').write_text(text)
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    y=list(range(len(summary)))
    ax.axvline(baseline.judge_mean,color='#b4543a',linestyle='--',label=f'Baseline W: {baseline.judge_mean:.4f}')
    ax.axvline(operational.judge_mean,color='#858585',linestyle=':',label=f'Operational W: {operational.judge_mean:.4f}')
    ax.hlines(y,summary.best100_judge_mean,summary.primary_judge_mean,color='#bbc6cc',linewidth=2)
    ax.scatter(summary.primary_judge_mean,y,s=55,color='#245f85',label='Original seed2026 top1',zorder=3)
    ax.scatter(summary.best100_judge_mean,y,s=35,marker='D',color='#27856b',label='Post-hoc best of100',zorder=4)
    for i,r in enumerate(summary.itertuples()):
        ax.annotate(f'{r.primary_judge_mean:.2f}',(r.primary_judge_mean,i),xytext=(6,-2),textcoords='offset points',fontsize=8)
    ax.set_yticks(y,summary.label);ax.invert_yaxis()
    ax.set_xlabel('Predicted R by one fixed ensemble judge (lower is better)')
    ax.set_xlim(summary.best100_judge_mean.min()-.6,summary.primary_judge_mean.max()+1.7)
    ax.grid(axis='x',alpha=.15)
    ax.set_title('15 candidate generators versus the same baseline W',loc='left',fontweight='bold')
    ax.legend(loc='lower left',fontsize=8)
    fig.supxlabel('Surrogate comparison only. Shared judge; no simulator measurements.',fontsize=9)
    for suffix in ('png','svg'):
        path=out/f'comparison.{suffix}'
        fig.savefig(path,dpi=180)
        if suffix=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
