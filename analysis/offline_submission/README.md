# Offline submission search

Role C study: compare the existing trust-region Dirichlet search with a new projected
DE/rand/1/bin optimizer (`simplex-de`). Both call only the surrogate's public `predict` method.
Tree models are piecewise constant, so direct input gradient descent is not applicable.

The eight `experiments/offline-*.toml` configurations compare HistGB/XGBoost, the two optimizers,
and optimizer seeds 2026/2027. Model settings remain fixed at the existing defaults (seed 2026).
Each configuration uses a distance limit of 0.75 times median training nearest-neighbor spacing
and risk coefficient 1. Existing default searches at radius 1.5 provide additional references.
DE generates 11,904 trial points; Dirichlet generates 12,000. Infeasible points are rejected before
surrogate scoring, so the number of actual prediction queries differs. These are surrogate
queries, not real-evaluator calls. This is a small, approximately budget-matched diagnostic study.

Generate every candidate through `sdm optimize` so it has a run ID. `--model-run` may reuse
the corresponding original optimize run's fitted model, avoiding any refitting differences.

`select.py` consumes recorded proposals and judges them with saved models. Its rule is fixed:
minimize the worse of HistGB/XGBoost `mean + std`, among novel candidates within distance ratio
0.75. Ridge and ExtraTrees provide additional diagnostics, not votes in this selection rule,
because their frozen-v32 first-choice errors are larger. Ranking uses the eight-decimal exported
weights, with each building's sum corrected to one. The measured best row is retained as a reference,
not silently submitted as a new candidate.

Outputs include `weights.json`, the complete cross-model candidate table, source run IDs and hashes,
nearby-point sensitivity, and selection-rule diagnostics on saved frozen held-out predictions.
The `std` risk penalty is a heuristic error scale, not a formal confidence bound after optimization.
Repeated use of the frozen folds and agreement between models do not establish out-of-distribution
accuracy. Only the real evaluator can establish the submitted weights' score.

Example (substitute the recorded run IDs):

```bash
env -u PYTHONPATH uv run --group gui python analysis/offline_submission/select.py \
  --proposals <optimize-run-1> <optimize-run-2> \
  --judges histgb=<histgb-model-run> xgboost=<xgboost-model-run> extratrees=<extratrees-model-run> ridge2=<ridge-run> \
  --evaluations histgb=<histgb-v32-evaluate-run> xgboost=<xgboost-v32-evaluate-run> \
  --out analysis/offline_submission/results/<unique-name>
```
