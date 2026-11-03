# Retrain the selected recipe and rerun consensus optimization

Protocol fixed before execution, 2026-11-01, role C on `azq`.

Use the unchanged 2, 048-row dataset. No fresh measured online labels are present in the pending
worksheet. The baseline R119.0916 mentioned in the supplied HTML is not added without a verified
evaluator record. Do not edit the source dataset, fixed splits, scorer, or existing model bundle.

## Fixed runs

1. Evaluate `experiments/baseline-v32.toml` again under frozen-v32.
2. Evaluate `experiments/soft-xgboost-step-2026.toml` again as the matched strong reference.
3. Evaluate and all-data train `experiments/method-blend-stack.toml` again, with seed 2026 and
   the same training-only OOF convex ensemble procedure. No new parameter or seed selection.
4. All-data train `experiments/consensus-r075-2026.toml` again and use that model run for
   `sdm optimize -n 50`, unchanged risk1, distance ratio0.75, optimizer seed 2026.

Every execution uses `sdm` CLI and new run folders. Compare evaluation metrics and aligned outer
predictions with the archived Oct29 runs; compare old/new stack and old/new consensus predictions
on all 1, 504 archived candidates plus the new 50 proposals. Compare new 50 proposals with the
original consensus search. Archive source IDs, metadata, configs, dataset/model hashes, metrics,
candidate predictions, and a loadable copy of the new fitted stack.

The original stack and controller files remain unchanged. Expect equivalent results if the same
data, algorithm, parameters and random seeds remain deterministic. Repetition supplies
reproducibility evidence, not independent generalization or an additional real-R improvement.

Reference runs:

- Ridge evaluation: `20261029-125034-evaluate-baseline-v32`.
- Step-XGBoost evaluation: `20261029-160333-evaluate-soft-xgboost-step-2026`.
- Stack evaluation: `20261029-171154-evaluate-method-blend-stack`.
- Stack full fit: `20261029-171251-train-method-blend-stack` / `artifacts/best_model`.
- Consensus full fit: `20261029-132622-train-consensus-r075-2026`.
- Consensus search: `20261029-132648-optimize-consensus-r075-2026`.

The completed delivery will be stored in `analysis/retrain_comparison/delivery/`.
