# Selected-model retraining and consensus search comparison

> Historical study completed before real feedback. P1 subsequently measured 119.71206678869203
> (failed); baseline W* 119.0916 is TA-confirmed. Earlier first-test recommendations are superseded.
> Current scores/rules: [team decisions](../../../docs/decisions.md).
> Large evidence and model files: [release restoration](../../submission_handoff/README.md).


Repeated six CLI experiments on unchanged 2, 048 observations and frozen-v32. No new measured
labels, retuning, scorer changes or simulator queries. New run IDs are in run_manifest.json;
run_evidence.json preserves every config and metadata. The original published model is unchanged.

## Fresh model evaluation

| model | mae | v32_local_mae | v32_score | v32_top1_regret |
|---|---|---|---|---|
| ridge_eval | 6.9151 | 11.9657 | 66.4508 | 4.5095 |
| stack_eval | 4.6933 | 6.3988 | 75.1137 | 1.0632 |
| xgb_eval | 4.9765 | 6.6331 | 73.4746 | 1.0653 |

The maximum old/new metric difference is 0;
the maximum aligned outer-prediction difference is
0. Global MAE is the CLI `mae` metric.
These results reproduce earlier evidence; they are not an additional improvement or independent test.

## Fresh training and weight search

The stack all-data fit relearns the same three convex coefficients (see retrained_stack/metrics.json).
The consensus model is separately retrained and searched at risk1, radius0.75, seed 2026, 50 proposals.
Maximum old/new difference across all 50 source proposals is
0. New rank 1 versus the old
eight-decimal exported challenger differs by at most
4.87426805168e-09; distinguish export rounding
from a new strategy. The fresh stack and fresh consensus are compared with their old fitted
counterparts on 1, 554 inputs, maximum prediction/proxy difference
0.

| candidate | old_stack_mean | new_stack_mean | new_consensus_risk |
|---|---|---|---|
| baseline | 118.7316 | 118.7316 | 126.1526 |
| incumbent | 112.0811 | 112.0811 | 112.3377 |
| p01969 | 118.1777 | 118.1777 | 120.3946 |
| blend-fixed:s2026:r1 | 109.3303 | 109.3303 | 114.0330 |
| blend-fixed:s2027:r1 | 109.1711 | 109.1711 | 112.8243 |
| blend-fixed:s2027:r2 | 110.0416 | 110.0416 | 113.6119 |
| blend-stack:s2027:r2 | 110.3082 | 110.3082 | 113.5379 |
| challenger | 111.4190 | 111.4190 | 111.1645 |
| new-consensus-rank 1 | 111.4190 | 111.4190 | 111.1645 |

All R values are surrogate predictions. Consensus risk is max(member mean+proxy), not actual R.
Because the data/settings are unchanged, retraining is expected to reproduce the old result.
This does not establish any new true-R gain or settle the remaining model disagreement.

## Deliverables

- retrained_stack/model.pkl: transferable fresh model bundle, CLI-load verification passed.
- new_consensus_rank1.json: raw-precision new first proposal; the prior operational W is unchanged.
- metric_comparison.csv, outer_prediction_comparison.csv: old/new validation evidence.
- candidate_predictions.csv: all 1, 554 inputs, old/new stack and consensus means and proxies.
- comparison-weights.json.gz, prediction JSONs: exact CLI batch inputs/outputs.
- verification.json, provenance.json, run_evidence.json: numerical checks and source traceability.

Source model hashes can differ even when predictions match, because serialized estimator metadata
may vary. Prediction agreement, rather than byte equality, is the relevant reproducibility check.
No new model is declared superior solely because it was fitted later.

Full-suite verification: `env -u PYTHONPATH uv run --locked --group gui pytest -q` reports
115 passed and 1 known expected failure in52.91 seconds. The fresh portable bundle also passes
an independent CLI load/prediction check on the challenger.
