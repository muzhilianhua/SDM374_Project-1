# Direct consensus search and resampling stress test

Continue role C's offline investigation on branch `azq`. The previous submission is preserved.
The `boosted-consensus` wrapper makes a public Surrogate predictor out of unchanged HistGB and
XGBoost plugins. Its mean is the arithmetic average; its `std` field is explicitly a risk envelope:
`mean + envelope = max(HistGB mean + std, XGBoost mean + std)`. This field is not a calibrated sigma.
At optimizer risk=1, the existing Dirichlet optimizer directly minimizes the two-model objective.
The wrapper lives under models only to reuse the unchanged CLI/optimizer interfaces; this is the
task-required extension outside role C's usual optimizer paths. No shared interface is changed.

Search grid fixed before execution: radii 0.35, 0.75, 1.5 times median training spacing; optimizer
seeds 2026, 2027, 2028; 50 returned proposals per run, model seed 2026 and default Dirichlet budget.
The existing frozen-v32 protocol evaluates the wrapper's mean prediction without changing scoring.
Radius 1.5 is exploratory; replacement candidates must still satisfy distance ratio <=0.75.

After generation, freeze the candidate pool. Fit HistGB and XGBoost on five seeded 90% subsamples
for screening (seeds 27101..27105), keeping model parameters/seed fixed. Rank candidates by the
90th percentile across the five per-subsample worst-member `mean + std` scores. Include the old
submission as a selectable incumbent. Deduplicate candidates after eight-decimal simplex rounding.

Select one challenger, then audit only that challenger plus fixed references on another five
90% subsamples (seeds 28101..28105). Do not reselect from the pool after inspecting audit results.
Replace the incumbent only if the challenger improves the full-data minimax score, all five audit
pair scores improve, and their average score improvement is at least 1.0. These are prespecified
practical guards, not a hypothesis test or a guarantee of real improvement.

Both panels reuse the same original data and substantially overlapping training rows. They test
sensitivity to removing observations, not new ground truth or independent generalization. Quantile
and fold spread are descriptive. ExtraTrees and quadratic ridge remain disagreement diagnostics.
The simulator is not called. All real-evaluator scores for new weights remain unknown.

All search/model evidence is generated through `sdm` CLI. The resampling analysis uses the registry
under the documented analysis exception and writes dataset/package/git/input provenance beside
its outputs, including the exact sampled candidate IDs and the fixed seeds.
