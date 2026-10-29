# Full-support soft target weighting

Completed results, recommended parameters and weights: [delivery/README.md](delivery/README.md).

Role C's task-required surrogate extension on `azq`. No shared interface, original plugin,
frozen release or evaluation protocol is changed. All model/search evidence uses the `sdm` CLI.

## Prespecified comparison

Both XGBoost (depth 3, rate .1, 600 trees, subsample .9) and HistGB (15 leaves, rate .05,
600 iterations, L2=0) use model seed 2026 and four workflows:

1. All observations, uniform weights.
2. All observations, lowest floor(.2*n) labels weighted 3 (step).
3. Lowest floor(.9*n) observations, lowest floor(.2*n) of original training labels weighted 3.
4. Full-support sigmoid weights, with training-only inner selection and uniform fallback.

The trimmed comparator is a controlled local implementation of the historical strategy, not a
reproduction of its 74.7167: that historical workflow used different environments and tuning.

Soft weight = 1 + alpha * sigmoid((q20(y) - y) / (temperature * population_std(y))).
The fixed grid is alpha in {1,3}, temperature in {.1,.3}, plus alpha=0 (uniform).
Three inner folds stratified on the approved low-region cutoff use seed 2127. Maximize mean inner
v3.2 subject to global MAE <=1.05x and low MAE <=1.10x uniform. Within .25 score points of the
best eligible setting, prefer smaller alpha, then wider temperature. Cutoffs/scales/weights are
recomputed from each inner training fold. The entire procedure is evaluated inside each of the
unchanged ten frozen outer splits; outer labels never select weighting parameters within a fit.
Final all-data training repeats the same inner rule, rather than choosing settings on outer scores.

All strategies recompute target weights inside each calibration training fold. Distance-binned
residual quantiles follow the existing plugin convention. For tuned models, the selected setting
is held fixed during residual calibration: these error proxies are selection-aware and do not
provide formal conformal coverage. Outer-test empirical coverage is reported.

## Candidate policy fixed before evaluation

Train and search each workflow (50 points, trust-dirichlet, risk=1, radius=.75, seeds 2026/2027).
Report each model's optimum and cross-score the pool with the two tuned full-support workflows,
the existing HistGB/XGBoost references and ExtraTrees. The old incumbent and previous challenger
remain fixed references. Round weights to eight decimals and recheck the simplex before scoring.

For a full-support model recommendation, require outer global MAE <=1.05x the matched uniform
workflow, low MAE <=1.10x and mean v3.2 not below uniform. If neither tuned workflow passes,
retain uniform as the model fallback and describe soft weighting as unproven.
Outer-score model selection is itself reuse of the benchmark, not an independent final test.

Within distance ratio <=.75, select the smallest maximum mean-plus-error-proxy across the original
and tuned HistGB/XGBoost judges. Keep the incumbent selectable. This conservative rule can retain
the old recommendation. Report ExtraTrees dissent and sensitivity, rather than declaring new
simulator performance. No real evaluator calls are made.

## Evidence and reproducibility

Tracked configurations: `experiments/soft-*.toml`. Analysis utilities write source hashes,
dataset checksum, git state, package versions and exact input run IDs to `provenance.json`.
Compare outer scores per fold and repeat, global/low error, first-choice regret, empirical proxy
coverage and error by distance to outer-training data. Fold spreads and paired differences are
descriptive because records and training sets overlap.
