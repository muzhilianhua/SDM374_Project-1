# Continued offline investigation: direct consensus search

## Decision

Keep the previous submission as the default (`recommended_weights.json`). A new exploratory
candidate is provided separately in `challenger_weights.json`. It has lower full-data predictions
under both boosted models and a lower average resampling risk score, but fails the predefined
replacement gate because two of five audit pairs show small regressions. The original submission
files have not been overwritten. Neither candidate has a real-evaluator score.

| Candidate | Full-data HistGB | Full-data XGBoost | Max(mean + error proxy) | Distance ratio |
|---|---:|---:|---:|---:|
| Previous recommendation | 108.9778 | 108.8717 | 112.3377 | 0.5410 |
| New challenger | 107.8045 | 107.2767 | 111.1645 | 0.6382 |
| Previous XGBoost-only minimum | 112.0136 | 99.3992 | 115.3736 | 0.7201 |
| Baseline PGD weights | 117.6234 | 114.2081 | 126.1526 | 1.7484 |

These are predictions at the named weights. In particular, the baseline PGD point's own quadratic
prediction is 75.8957, not the boosted-model values in this table. The best measured training row
remains p01969 with actual ramping 115.7648; that value does not belong to either new candidate.

New challenger source: **`20261029-132648-optimize-consensus-r075-2026`, rank 1**.
Previous recommendation: `20261029-125052-optimize-histgb-v32`, rank 16.
Previous XGBoost-only minimum: `20261029-125052-optimize-xgboost-v32`, rank 1.
Baseline source: `20261029-125052-optimize-baseline`, rank 1.

## What changed in the search

Previously, each model generated candidates separately and the results were cross-scored. The
new `boosted-consensus` wrapper directly minimizes the worse member's mean plus error proxy
through the existing Dirichlet optimizer. Its mean is the average of the two model predictions;
the interface's `std` field is an explicitly documented risk envelope. It is not a Gaussian
standard deviation or a confidence interval. Risk must be 1 to recover the minimax objective.

The nine CLI search runs use radius ratios 0.35, 0.75 and 1.5 with optimizer seeds 2026, 2027 and
2028. All share one fitted committee (`20261029-132622-train-consensus-r075-2026`) with unchanged
member hyperparameters and model seed 2026. Each generates 12,000 samples and returns 50 proposals.

| Radius limit | Seed | Best full-data minimax score | Distance ratio |
|---|---:|---:|---:|
| 0.35 | 2026 | 112.5287 | 0.3464 |
| 0.35 | 2027 | 113.4654 | 0.3217 |
| 0.35 | 2028 | 113.6126 | 0.3495 |
| 0.75 | 2026 | 111.1645 | 0.6382 |
| 0.75 | 2027 | 111.3689 | 0.6660 |
| 0.75 | 2028 | 111.2526 | 0.6016 |
| 1.5 | 2026 | 111.1645 | 0.6382 |
| 1.5 | 2027 | 111.3689 | 0.6660 |
| 1.5 | 2028 | 111.2526 | 0.6016 |

Run IDs are `20261029-132648-optimize-consensus-r<035|075|150>-<seed>`, except radius 1.5,
seed 2028, which is `20261029-132649-optimize-consensus-r150-2028`. Exact configs, metrics and
provenance for all nine searches plus the train/evaluate runs are in `run_evidence.json`.
Increasing the radius from 0.75 to 1.5 did not improve the first proposal for any of these seeds.
This is evidence about this search budget and initialization, not a global optimality claim.

## Screening and audit were separated before inspecting results

The 450 proposal entries, previous recommendation and three fixed references reduce to 304
unique eight-decimal weight vectors. Of these, 301 are eligible for selection (novel, selectable,
distance ratio <=0.75). All weights satisfy the product-of-simplices constraint.

Screening analysis **`20261029-consensus-screen-v1`** retrains both models on five seeded 90%
subsamples (27101..27105). Hyperparameters and model seeds remain fixed. Selection minimizes the
90th percentile of five worst-member mean-plus-error-proxy scores. The previous recommendation
is included as a selectable incumbent, so selection does not force a change.

Screening q90: previous **119.4843**, challenger **118.0428**. This chooses the new challenger.
Only this challenger and fixed references proceed to the audit; the candidate pool is not
reselected after observing audit results.

Audit analysis **`20261029-consensus-audit-v1`** uses five separately seeded 90% subsamples
(28101..28105). Its paired risk-score differences (challenger minus previous; negative is better) are:

| Audit seed | Difference |
|---|---:|
| 28101 | -3.2984 |
| 28102 | +0.0875 |
| 28103 | -2.4704 |
| 28104 | -0.0169 |
| 28105 | +0.2427 |

| Audit statistic | Previous | Challenger |
|---|---:|---:|
| Mean worst-member risk score | 117.8677 | 116.7766 |
| 90th percentile risk score | 121.2742 | 120.3697 |
| Mean HistGB point prediction | 112.0597 | 111.3368 |
| Mean XGBoost point prediction | 112.4327 | 110.7180 |

The average paired improvement is **1.0911**, but only three of five pairs improve. The replacement
gate was fixed before fitting either panel: better full-data and screening scores, improvements
on all five audit pairs, and average improvement >=1.0. The challenger fails the all-five condition.
This conservative decision does not prove that the old point is truly better: the two regressions
are small and the new point remains a reasonable exploratory candidate for real evaluation.

The panels reuse the original dataset and their training rows overlap heavily. Different random
seeds do not make them independent test sets. These checks measure sensitivity to omitted training
observations; they cannot establish simulator performance or statistical significance.

## Model evaluation and dissenting models

Under unchanged frozen-v32, the committee's arithmetic-mean prediction obtains v3.2 **73.4123**,
global MAE **4.7241**, low-region MAE **6.5510**, and first-choice regret **1.8768**
(`20261029-132622-evaluate-consensus-r075-2026`). XGBoost's existing v3.2 score is 73.4746 and
first-choice regret is 1.0653 (`20261029-125034-evaluate-xgboost-v32`). Thus the committee does
not establish a new overall model-scoring record; its role here is joint candidate search.

ExtraTrees predicts **126.0904** for the challenger versus **125.2473** for the previous recommendation.
Quadratic ridge predicts **95.1292** versus **95.5531**. These diagnostic predictions come from
`20261029-125052-optimize-extratrees-default` and `20261029-125052-optimize-baseline`, respectively,
with hashes and per-reference values in `additional_diagnostics.json`.
ExtraTrees still disagrees materially with both boosted models; this concern has not been resolved.

## Delivery and verification

- `recommended_weights.json`: the previous recommendation, numerically unchanged.
- `challenger_weights.json`: the new exploratory point; shape 3 x 7, nonnegative, row sums one.
- Column order: cost, carbon, peak, discomfort, degradation, ramping, resilience.
- `candidate_summary.csv` and `screen_predictions.csv`: complete screening evidence.
- `audit_predictions.csv`, `audit_summary.csv`, `decision.json`: frozen audit and replacement decision.
- `screen_provenance.json.gz`, `audit_provenance.json.gz`: full input hashes, package versions,
  original dataset checksum, source commit, all seeds and exact training-row IDs for each fit.
- **88 tests passed, 1 known expected failure** (unchanged baseline PGD duplicate-batch behavior).
- The selected source search was reproduced through `sdm reproduce` within rtol=1e-9.
- Both exported weight files were checked, and the challenger's stored model predictions match
  predictions recomputed on the exported JSON weights.

All source runs are clean at commit `60c76e2` on local branch `azq`. No real-evaluator calls,
submissions or online budget were used. Existing course files and prior submission are unchanged.
The model wrapper is the task-required addition outside role C's optimizer directory; shared
interfaces, frozen splits and the scorer are unchanged.

Reproduction commands are in `reproduce.sh`; choose new output directories to preserve this
delivery. The original CLI run folders/models remain under git-ignored `runs/`. This lightweight
package contains evidence and submitted weights, not model binaries or the entire dataset.
