# Alternative methods: prespecified exploration

Continue on `azq` for role C. This task requires surrogate extensions under `models/` and a pinned
CatBoost dependency; shared interfaces, prior plugins, frozen releases and scoring are unchanged.

## Discovery scope fixed before inspecting scores

Fifteen configurations cover:

- CatBoost RMSE uniform, RMSE low20 x3, and MAE low20 x3 (depth 5, 600 iterations, rate .05).
- Classic GBDT squared error, Huber and median quantile (depth 3, 600 iterations, rate .05,
  minimum leaf 5, subsample .9, low20 x3).
- XGBoost pseudo-Huber (slope 5) and explicit 105-dimensional interaction features (depth 3,
  600 trees, rate .1, subsample .9, low20 x3).
- Standardized XGBoost regression plus low30 pairwise logistic loss, strengths 0/.1/.3,
  8192 sampled pairs. The zero-strength control uses identical target standardization.
- Centered RBF kernel ridge in raw and Hellinger coordinates. Alpha=.1; bandwidth=.5 times
  the median of 2048 seeded within-training pair distances; uniform sample weights.
- Fixed convex XGBoost/CatBoost/HistGB blend (.5/.3/.2) and a training-only OOF convex stack.
  Stacking minimizes low20-x3 weighted squared error plus .02 shrinkage toward the fixed blend.

All data remain in training; low20 thresholds are recomputed inside each fit. All models use
seed 2026 in discovery. The existing ten frozen outer splits and v3.2 scorer are unchanged.
New residual proxies use three training folds and four distance bins at quantile .68, reported
as heuristic error scales. Learned stacking weights reuse OOF calibration labels, so their
residual proxies are selection-aware; empirical outer coverage must be reported.
The stack's OOF training and weight fitting occur entirely within each outer training fold.
They never use the outer-test observations. Stacking members are XGBoost step-weighted,
CatBoost RMSE step-weighted and uniform HistGB; no member is chosen on outer labels within a fit.

Every discovery case runs CLI evaluate/train and two trust-dirichlet searches (seeds 2026/2027,
50 returned points, risk=1, maximum distance ratio=.75). Raw trials and failures are retained.
CatBoost is pinned to 1.2.8 in `pyproject.toml`/`uv.lock`; additional package provenance includes it.

## Model selection and seed audit fixed before discovery

Reference: full-data step-weighted XGBoost,
`20261029-160333-evaluate-soft-xgboost-step-2026` (v3.2 73.4746).
For a replacement contender require v3.2 >= reference + .5, global MAE <=1.05x reference,
low MAE <=1.05x reference, first-choice regret <=2.0, and coverage between .55 and .80.
Select the highest v3.2 among eligible new methods; if none qualify, audit the descriptive
highest-score method as an exploratory alternative without declaring a replacement.

Audit the selected new method and the matched reference with new model seeds 2027 and 2028.
Do not reselect from the discovery pool after the audit. Replacement requires discovery eligibility,
higher mean score over seeds 2026/2027/2028, global MAE <=1.05x the reference at each seed, and
average first-choice regret <= reference average +.5. These are practical comparison guards,
not a significance test. New model seeds reuse the same outer observations and are not fresh data.

After a broad search, a best frozen score is selection-aware and cannot establish independent
SOTA or simulator performance. Original controller weights remain the operational default;
new candidate points are exploratory. Cross-score candidates with original HistGB/XGBoost and
the two highest-scoring new full-data fits. Include the incumbent as selectable and report
ExtraTrees dissent. Changing judges also changes the risk criterion; compare points only within
one fixed panel, never compare risk numbers across panels as if they measured actual R.

## Reproduction

Use `env -u PYTHONPATH uv run --locked --group gui` for commands in the ROS environment.
CLI run folders are ignored; a tracked delivery will include metadata, configs, predictions,
per-fold metrics, exact source IDs and provenance. All real-evaluator scores remain unknown.
