# Full-support weighting study: results and parameter handoff

## Decision

Use **full-data XGBoost with the lowest 20% of training labels weighted 3x** as the primary
surrogate, with uniform HistGB as a complementary judge. Keep the original controller submission
in `recommended_weights.json`. The sigmoid experiment is complete, but does not establish an
improvement over that existing full-support XGBoost workflow. No new real ramping score was measured.

Sigmoid-weighted XGBoost improves on uniform XGBoost by 1.8495 v3.2 points, but trails step-weighted
XGBoost by 1.0050. Sigmoid HistGB trails uniform HistGB by .3650. The highest descriptive score in
this controlled study is 73.6639 for trimmed XGBoost, but its first-choice regret is substantially
worse than the full-data step model. The historical trimmed/tuned result 74.7167 is a different
workflow and is not reproduced or exceeded here.

## Evaluation

All rows use the unchanged frozen 2x5 outer splits and approved v3.2 scorer. Higher v3.2 is better;
lower MAE and first-choice regret are better. Global MAE below is the CLI `mae` metric, averaged
over complete out-of-fold repeats; low MAE and regret are averages over the ten outer folds.

| Model | Weighting workflow | v3.2 | Global MAE | Low MAE | First-choice regret | Proxy coverage |
|---|---|---:|---:|---:|---:|---:|
| HistGB | Uniform, all data | 72.3465 | 4.7920 | 6.6728 | 1.8768 | .6958 |
| HistGB | Lowest 20% x3, all data | 72.3427 | 4.8551 | 6.5521 | 2.0084 | .6941 |
| HistGB | Lowest 90%; lowest 20% x3 | 72.4516 | 5.0870 | 6.3544 | 2.1074 | .6863 |
| HistGB | Nested sigmoid, all data | 71.9815 | 4.8183 | 6.6115 | 2.0084 | .6982 |
| XGBoost | Uniform, all data | 70.6201 | 4.9405 | 7.0593 | 3.6221 | .6860 |
| **XGBoost** | **Lowest 20% x3, all data** | **73.4746** | **4.9765** | **6.6331** | **1.0653** | **.6880** |
| XGBoost | Lowest 90%; lowest 20% x3 | 73.6639 | 5.1581 | 6.5507 | 2.8742 | .6921 |
| XGBoost | Nested sigmoid, all data | 72.4696 | 4.9974 | 6.7467 | 2.5778 | .6855 |

Exact source IDs are in `run_manifest.json` and `model_scores.csv`. Key runs:

- Primary model: `20261029-160333-evaluate-soft-xgboost-step-2026`.
- Sigmoid XGBoost: `20261029-160420-evaluate-soft-xgboost-soft-2026`.
- Uniform HistGB: `20261029-160156-evaluate-soft-histgb-unweighted-2026`.
- Sigmoid HistGB: `20261029-160232-evaluate-soft-histgb-soft-2026`.
- Trimmed XGBoost: `20261029-160357-evaluate-soft-xgboost-trimmed-step-2026`.

Relative to step XGBoost, sigmoid increases first-choice regret from 1.0653 to 2.5778.
The trimmed variant gains only .1893 score points while regret rises to 2.8742 and global MAE
increases by about 3.65%. These tradeoffs favor the full-data step workflow for choosing one W.
The mean score gaps do not establish statistical significance or a guaranteed deployment order.

## How the sigmoid parameters were selected

Training weights are

`1 + alpha * sigmoid((q20(training_y) - y) / (temperature * std(training_y)))`.

All observations have positive weights; sigmoid training drops zero rows. In each outer training
set, compare uniform alpha=0 with alpha={1,3} x temperature={.1,.3} on three training-only inner
folds. The inner folds stratify on the approved cutoff 132.8 and use seed 2127. Prefer the highest
inner v3.2 subject to global MAE <=1.05x and low MAE <=1.10x uniform. Within .25 score points of
the best eligible candidate, prefer smaller alpha and then wider temperature. Recompute quantiles,
scales and training weights separately inside every inner training fold.

No outer-test label chooses alpha or temperature within a fit. All-data training repeats this
rule, yielding the following final fitted settings:

| Family | Alpha | Temperature / training SD | Width in ramping units | q20 cutoff | Effective sample size |
|---|---:|---:|---:|---:|---:|
| XGBoost sigmoid | 3 | .1 | 1.5767 | 148.0848 | 1389.70 |
| HistGB sigmoid | 1 | .3 | 4.7302 | 148.0848 | 1917.06 |

The training q20 cutoff is distinct from the scorer's fixed low-region cutoff 132.8.
**72.4696 evaluates the entire nested selection workflow**, whose selected settings can differ
by outer fold. It must not be attached to a fixed alpha=3, temperature=.1 workflow without
evaluating that fixed workflow separately. `full_data_parameters.json` contains all final
inner trials, estimator settings and training diagnostics; XGBoost's NaN missing-value sentinel
is represented as the string `NaN` so the file remains valid JSON.

## Generalization diagnostics

The XGBoost sigmoid workflow improves over uniform in five folds, ties in two, and loses in three.
The mean score gains are +1.3036 and +2.3955 in the two repeats. Against step XGBoost, its mean
differences are -1.7583 and -.2516: neither repeat supports replacing the step model.
HistGB's sigmoid-vs-uniform differences change sign between repeats (+.3669 / -1.0970).
Exact paired differences are in `paired_comparisons.json`. These overlapping folds do not support
independent-fold confidence intervals or significance claims.

For each outer test fold, sort points by distance to the outer training rows and divide into
near/middle/far thirds without using labels. Mean global MAE over the far thirds is:

| Family | Uniform | Step | Trimmed-step | Sigmoid |
|---|---:|---:|---:|---:|
| HistGB | 7.0077 | 7.0245 | 7.8577 | 7.0288 |
| XGBoost | 7.0671 | 7.0988 | 7.7807 | 7.1852 |

Trimming harms error in the far thirds despite its slightly higher aggregate score. Sigmoid does
not improve global far-third MAE either. It lowers far-third low-region MAE for XGBoost (8.2335 vs
8.6036 step), which is a narrower tradeoff worth retaining as a diagnostic, not evidence of a
general extrapolation improvement. These are distance-stratified views of the same random splits,
not a spatial-block holdout test. No new week, weather, load or simulator scenario was tested.

Error proxies use the existing distance-binned residual convention, with weighting recalculated
inside calibration training folds. Tuned settings are held fixed during residual calibration,
so tuning and calibration reuse training labels. The .68-style proxies have no formal post-selection
coverage guarantee, especially after optimization. Outer empirical coverage is reported above.

## Recommended operational parameters

Use [`soft-xgboost-step-2026.toml`](../../../experiments/soft-xgboost-step-2026.toml):

```text
model = lowramp-boosted
family = xgboost
strategy = step; tune = false
fraction = .20; alpha = 2.0       # lowest floor(.20*n) receive weight 1+alpha = 3
max_depth = 3; learning_rate = .1; iterations = 600
subsample = .9; seed = 2026; n_jobs = 1
```

Train run: `20261029-160351-train-soft-xgboost-step-2026`.
Complementary model: `soft-histgb-unweighted-2026.toml`, 15 leaves, learning rate .05,
600 iterations, L2=0, all rows uniformly weighted, seed 2026.

For candidate generation: `trust-dirichlet`, risk=1, maximum distance ratio=.75, 16 centers,
30 iterations x 25 samples per center (12,000 generated points), concentration=200, growth=1.15,
minimum separation ratio=.25; 50 returned points, optimizer seeds 2026 and 2027.

## Controller weights and candidate search

Eight workflows x two optimizer seeds generated 800 proposal entries through the CLI. Adding
the incumbent, previous challenger and measured best reference yields 802 unique rounded points.
The expanded four-judge criterion takes the maximum mean-plus-error-proxy over original and
sigmoid HistGB/XGBoost, within distance ratio .75. The original incumbent remains selectable.
ExtraTrees and the other workflows provide dissenting diagnostics.

| Point | Expanded four-judge risk | Original two-judge risk | Distance ratio | ExtraTrees mean |
|---|---:|---:|---:|---:|
| Default incumbent | 114.6746 | 112.3377 | .5410 | 125.2473 |
| Previous challenger | 114.4615 | 111.1645 | .6382 | 126.0904 |
| New study point | 114.3701 | 114.1971 | .6130 | 124.8298 |

The new point is rank 6 of `20261029-160431-optimize-soft-histgb-soft-2027`, saved separately in
`study_weights.json`. Its original HistGB/XGBoost means are 110.8371 / 106.9467; sigmoid means
are 110.9376 / 108.0207. The expanded risk improves by only .3045 relative to the incumbent,
while original two-judge risk worsens by 1.8593. ExtraTrees remains pessimistic. Nearby sigmoid
predictions also rise: under 128 Dirichlet perturbations, HistGB median is 113.3590 versus 110.9376
at the exact point (`local_sensitivity.json`). No separate candidate audit was performed this round.

Keep `recommended_weights.json`, copied byte-for-byte from the first delivery, as the operational
default. The new point remains exploratory. Its generation by the rejected sigmoid HistGB does
not itself invalidate it, but cross-scoring supplies insufficient evidence for replacing an
already audited recommendation. Actual R is unknown for every proposed new point; the best
measured record remains p01969 at 115.76481925917324.

## Verification and reproduction

- Full test suite: **97 passed, 1 known expected failure**, the unchanged baseline PGD duplicate batch.
- `lowramp-boosted` is listed by `sdm list` and passes the model contract.
- Reproduced sigmoid XGBoost evaluation through `sdm reproduce`: source
  `20261029-160420-evaluate-soft-xgboost-soft-2026`, rerun
  `20261029-160610-evaluate-soft-xgboost-soft-2026`, rtol=1e-9.
- Reproduced the selected source search: source `20261029-160431-optimize-soft-histgb-soft-2027`,
  rerun `20261029-160656-optimize-soft-histgb-soft-2027`, same tolerance.
- Exported weights pass the simplex constraints; CLI predictions on exported JSON match the table.
- CLI source runs are clean at code commit `092efff`. Analysis commit `957b707` only adjusts the
  summary utility and reference-run handling; the evaluated model code is unchanged.

`run_evidence.json` preserves metadata, configs and metrics for all 32 new runs plus three
original model references. `outer_predictions-*.csv.gz` preserves all eight workflows' predictions
for independent rescoring; `fold_scores.csv` and `model_scores.csv` give the complete evaluation.
Candidate weights, cross-model predictions and source ranks are included. `provenance.json`
records input hashes, dataset checksum, package versions and the exact generation run manifest.

Run `reproduce.sh NEW_STUDY_DIR NEW_SUMMARY_DIR` from the repository root to regenerate the study.
Original reference model runs must exist locally, or supply three freshly trained reference run IDs
as arguments 3–5 in HistGB / XGBoost / ExtraTrees order. On a fresh clone, create those with
`sdm train -c experiments/histgb-v32.toml`, `xgboost-v32.toml` and `extratrees-v32.toml`.
Use `env -u PYTHONPATH uv run --locked --group gui` before project commands in the ROS environment.
The script launches training, evaluation and search through the CLI, then reruns the summary with
the newly recorded run IDs. New run IDs will differ from this handoff's IDs.

Run folders and model binaries remain ignored under `runs/`; the tracked delivery is 1.6 MB of
weights, predictions, evidence and documentation. No course-provided file, frozen split,
evaluation policy or regression tolerance was changed. No real evaluator or official submission
budget was used. The reused frozen benchmark establishes neither SOTA nor cross-scenario performance.
