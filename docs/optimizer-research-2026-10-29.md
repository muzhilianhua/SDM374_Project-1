# Optimizer research log — 2026-10-29

Role C investigated offline search on branch `azq`, starting from main commit `3f1a6d7`.
The current default remains the first recommendation. A second, exploratory candidate has better
boosted-model predictions but does not pass the prespecified replacement gate. Neither has been
measured by the real evaluator. The best measured training row remains `p01969`, R = 115.7648.

## Deliverable and selection decision

| Quantity (lower is better) | Default | Exploratory challenger |
|---|---:|---:|
| HistGB prediction | 108.9778 | 107.8045 |
| XGBoost prediction | 108.8717 | 107.2767 |
| Full-data max(member mean + error proxy) | 112.3377 | 111.1645 |
| Distance / median training spacing | 0.5410 | 0.6382 |
| Screening risk q90 | 119.4843 | 118.0428 |
| Audit mean risk | 117.8677 | 116.7766 |
| ExtraTrees diagnostic prediction | 125.2473 | 126.0904 |
| Real-evaluator R | Unknown | Unknown |

- Default: proposal 16 of `20261029-125052-optimize-histgb-v32`;
  [recommended weights](../analysis/robust_search/delivery/recommended_weights.json).
- Challenger: rank 1 of `20261029-132648-optimize-consensus-r075-2026`;
  [challenger weights](../analysis/robust_search/delivery/challenger_weights.json).
- Each file is a 3 × 7 matrix: buildings in rows; cost, carbon, peak, discomfort, degradation,
  ramping, resilience in columns. Rows are nonnegative and sum to one.

The challenger improves average audit risk by 1.0911, but only three of five paired comparisons
improve. Differences (challenger minus default) are -3.2984, +0.0875, -2.4704, -0.0169, +0.2427.
The predefined rule requires better full-data and screening scores, improvement on all five audit
pairs, and mean improvement at least 1.0. Therefore the default is retained; this does not establish
that it is truly better. See the [decision record](../analysis/robust_search/delivery/decision.json)
and [complete second-round report](../analysis/robust_search/delivery/README.md).

## Round 1: compare search methods

Added projected differential evolution (`simplex-de`) with trust-region constraints, distinct
proposals and deterministic seeds. Compared it with `trust-dirichlet` on HistGB and XGBoost at
optimizer seeds 2026/2027, holding model settings fixed. Both use risk coefficient 1 and distance
limit 0.75. DE generates 11,904 trials and Dirichlet 12,000 samples; rejected infeasible proposals
mean actual prediction-query counts differ. This is approximately budget-matched evidence.

Select the lowest worst-member mean-plus-error-proxy score among eligible novel candidates.
DE did not improve that criterion in these eight runs, so the recommendation comes from Dirichlet.
The low XGBoost-only prediction 99.3992 and ridge-only prediction 75.8957 are not measured results;
cross-model disagreement and extrapolation respectively make them poor selection evidence.
The distance-filtered held-out comparison gave identical mean regret 5.1299 for all three selection
rules, so it provides no evidence that consensus improves held-out ranking.

Evidence and exact settings:

- [Methodology](../analysis/offline_submission/README.md) and
  [first-round report](../analysis/offline_submission/submission/README.md).
- Evaluations: `20261029-125034-evaluate-{baseline-v32,extratrees-v32,histgb-v32,xgboost-v32}`.
- Reference searches: `20261029-125052-optimize-{baseline,extratrees-default,histgb-v32,xgboost-v32}`.
- Controlled searches: `20261029-125621-optimize-offline-<model>-<optimizer>-<seed>`;
  tracked configurations in `experiments/offline-*.toml`.
- Source commits: `3f1a6d7` (references), `ef4e060` (optimizer comparison),
  `de001b9` (selection script); delivery commit `d792137`. Evidence runs record `dirty=false`.

## Round 2: directly search consensus and audit sensitivity

Added `boosted-consensus` to expose unchanged HistGB/XGBoost predictors through the existing
Surrogate interface. At optimizer risk=1, it minimizes the worse member's mean plus error proxy.
Its `std` field is a risk envelope, not a calibrated standard deviation. The model wrapper is the
task-required change outside role C's usual paths; no shared interface was changed.

Searched radii 0.35/0.75/1.5 with seeds 2026/2027/2028, returning 50 proposals per run.
The larger two radii returned identical top points for each seed. Deduplication and references
yielded 304 unique points, of which 301 were eligible. Screening used five 90% training subsets
(seeds 27101–27105), selecting by q90 risk and including the incumbent. Only the selected challenger
and fixed references entered the separately seeded audit (28101–28105); no reselection followed.

Evidence and exact settings:

- [Methodology](../analysis/robust_search/README.md),
  [complete report](../analysis/robust_search/delivery/README.md), and
  [source run evidence](../analysis/robust_search/delivery/run_evidence.json).
- Model runs: `20261029-132622-{train,evaluate}-consensus-r075-2026`.
  Frozen-v32 score 73.4123 does not beat XGBoost's 73.4746 from the first round.
- Searches: `20261029-132648-optimize-consensus-r<035|075|150>-<seed>`, except
  r150/2028 at `20261029-132649-optimize-consensus-r150-2028`.
- Analyses: `20261029-consensus-screen-v1` and `20261029-consensus-audit-v1`.
  Compressed provenance in the delivery includes exact sampled row IDs, hashes, seeds and packages.
- Source code commit `60c76e2`; all source runs are clean. Delivery commit `80acec1`.

These overlapping subsets are sensitivity checks on the same dataset, not independent validation.
ExtraTrees still disagrees materially with both boosted models. No real-evaluator calls or
official submissions were made, and no online evaluation budget was used.

## Environment, verification and reproduction

Local environment: uv 0.13.6, Python 3.11.9, locked core/dev/GUI dependencies and macOS `libomp`.
Unset the ROS-provided `PYTHONPATH` for project commands to avoid importing conflicting packages.
From this repository root:

```bash
uv sync --locked --group gui
env -u PYTHONPATH uv run --locked --group gui pytest -q
env -u PYTHONPATH uv run --locked --group gui sdm reproduce \
  20261029-132648-optimize-consensus-r075-2026
sh analysis/robust_search/delivery/reproduce.sh \
  analysis/robust_search/results/recheck-screen \
  analysis/robust_search/results/recheck-audit
```

The last two commands require the original local `runs/` folders. A fresh clone includes configs,
weights, prediction tables and provenance, but not ignored model binaries, raw run directories,
the virtual environment or result ZIPs. To regenerate on another machine, run `sdm train`,
`sdm evaluate` and `sdm optimize` with the tracked configs, then substitute the new run IDs in
the analysis commands; the delivered reproduction script refers to the original IDs.

Verification: 88 tests passed and 1 known expected failure (unchanged baseline PGD duplicate-batch
behavior). The selected consensus search reproduced within rtol=1e-9; exported weights pass the
simplex checks, and predictions recomputed from those weights match the stored values.
Frozen data, evaluation protocols and course-provided files were preserved.

## Round 3: full-support soft weighting

The subsequent full-support weighting comparison is complete in
[the third-round handoff](../analysis/soft_weighting/delivery/README.md): eight frozen-v32 workflows,
32 CLI runs and guarded training-only sigmoid tuning. Sigmoid XGBoost scores 72.4696 versus uniform
70.6201, but does not beat full-support step XGBoost at 73.4746; sigmoid HistGB also fails to improve
on uniform. The original controller weights remain recommended. The updated suite has 97 passing
tests and 1 known expected failure. Full parameters, source predictions and reproduction evidence
are included in that handoff; no simulator measurements were added.

## Round 4: broader model and objective exploration

The [alternative-method handoff](../analysis/method_expansion/delivery/README.md) records 15 fixed
recipes: CatBoost, robust/quantile GBDT, pseudo-Huber XGBoost, explicit interactions, pairwise losses,
raw/Hellinger kernel ridge and fixed/OOF-learned convex ensembles. Sixty discovery CLI runs have
clean source commit `076b75f`; four matched seed-audit runs have clean `8c39898` (model code unchanged).
This task-required model work extends outside role C's usual paths without changing shared interfaces.

The discovery-selected OOF stack scores 75.1137 at seed2026; weighted CatBoost is close at75.0766.
Freeze the stack before auditing it and the step-XGBoost reference at seeds2027/2028. Its three-seed
mean is75.3591 versus73.1322; MAE improves at all three seeds and average first-choice regret falls
to1.1078 from1.7181. The prespecified model-replacement guard passes. Recommend the stacking recipe
in `experiments/method-blend-stack.toml`, not a retrospectively selected seed. Full-data blend weights
are .2920246/.4437868/.2641886 for XGBoost/CatBoost/HistGB; each outer fit relearns them.

The expanded candidate panel prefers CatBoost-search proposal6 from
`20261029-171333-optimize-method-cat-weighted-search2026`, distance ratio .6450, risk116.9736
versus incumbent119.9333 within that same panel. This W is exploratory: no separate candidate audit
or real-evaluator call exists, and ExtraTrees remains pessimistic. Preserve the old operational W.
Kernel models and tested pairwise losses underperform; retain their negative results in the report.
The suite has109 passing tests and1 known xfail; selected model evaluation and source search reproduce.
Reused frozen scores and sensitivity seeds do not establish independent SOTA or measured improvement.

## Next work

Use real-evaluator results to compare the default and challenger if the course submission budget
permits. Record exact weights, evaluator version and measured R before changing the recommendation.
Resolve model disagreement with measured observations, then design diverse online batches and
adapt the trust radius using observed successes and failures. Further offline optimization alone
cannot establish improvement over the measured best row.
