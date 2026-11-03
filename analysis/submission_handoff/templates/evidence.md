# EVIDENCE INDEX — HISTORICAL EXPERIMENTS

**Current status:** P1 was submitted and measured 119.71206678869203 (failed). Baseline W* is
TA-confirmed at 119.0916. See [known observations](../KNOWN_OBSERVATIONS.json). Historical source
recommendations below predate these measurements and must not override current team decisions.

- model_evaluation.csv and source_runs.json contain exact CLI run IDs and frozen-v32 metrics.
- candidate_judges.csv contains historical surrogate predictions, not measured labels.
- fresh_neighborhood_audit.csv and training_subset_summary.csv test two different surrogate stability
  properties. Neither establishes real controller robustness; P1 subsequently failed the real bar.
- precision_*_summary.csv records serialization and near-zero cleanup checks. Exported weights were
  rescored because rounding may change tree predictions.
- source_model_decision.json records the rejected selection-correction experiment.
- historical_source_recommendation.json preserves the old recommendation as historical evidence.
- source_training_test_result.json and source_verification.json are the historical source checks;
  the current handoff checks are in ../VERIFICATION.json.

Source paths are relative to their original analysis/selection_training/delivery context. Current
package paths and aliases are in ../MANIFEST.json. Large raw evidence and model binaries are stored
as [release assets](../../analysis/submission_handoff/README.md), with tracked checksums/provenance.
Packaging imported the team's reported real results and made no new evaluator calls.
