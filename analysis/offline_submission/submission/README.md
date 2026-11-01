# Offline submission recommendation

Submit `weights.json`: one 3 x 7 matrix, with one building per row. Columns are **cost, carbon,
peak, discomfort, degradation, ramping, resilience**. No surrounding object or metadata is required.
This is a new, unmeasured candidate. The real ramping score is unknown.

## Recommended weights

```json
[
  [
    0.06955046,
    0.0,
    0.03035122,
    0.31109209,
    0.01771437,
    0.5623603600000001,
    0.0089315
  ],
  [
    0.17958762,
    0.00077106,
    0.17117251,
    9.928e-05,
    0.0,
    0.58431322,
    0.06405631
  ],
  [
    0.0,
    0.03226244,
    0.0730665,
    0.16515604,
    0.0,
    0.35434136,
    0.37517365999999996
  ]
]
```

Source: `20261029-125052-optimize-histgb-v32`, proposal **16** (one-based). The same point is also
proposal 15 of `20261029-125621-optimize-offline-histgb-trust-dirichlet-2026`.
Eight-decimal export changes the raw weights by at most 6.33e-9. All 21 weights are finite and
nonnegative, and each building sums to one within 1e-9. The exported file itself was scored again
through `sdm predict` with both boosted models.

## Why this candidate

1. Frozen-v32 model evidence favors the two boosted-tree models over the quadratic baseline for
   choosing good held-out observations. The frozen splits and scoring policy were not changed.
2. Trees have no useful input gradient. Search uses gradient-free Dirichlet sampling or projected
   differential evolution, initialized around good measured rows and constrained near the data.
3. Candidate selection uses the worse of HistGB/XGBoost **mean + std**, within distance ratio 0.75.
   The selected point scores well under both models, rather than exploiting one model's lowest number.
4. The selected point is closer to measured data than the baseline PGD point. This is a useful
   heuristic guardrail, not proof that the response between observations is known.

| Predictor | Prediction at submitted weights | Error-scale proxy |
|---|---:|---:|
| HistGB | 108.9778 | 3.3600 |
| XGBoost | 108.8717 | 3.4436 |
| ExtraTrees (diagnostic) | 125.2473 | 7.5703 |
| Quadratic ridge (diagnostic) | 95.5531 | 9.3596 |

The selection score is **112.3377**. The boosted-model disagreement is **0.1060**.
Nearest-training distance ratio is **0.5410** (median training spacing 0.377113).
The best measured row is `p01969`, actual R = **115.7648**. It is not a measurement of this submission.
The corresponding boosted predictions at p01969 are HistGB 116.8217 and XGBoost 116.9510.
No predicted improvement is presented as a verified real improvement or a guarantee of beating
course baseline performance.

## Model evaluation evidence

All rows below use frozen-v32, the same ten outer splits (2 repeats x 5 folds). Higher v3.2 is
better; lower MAE and first-choice regret are better. These are diagnostics on reused data,
not an independent final test or validation of a newly optimized W.

| Model | v3.2 | Global MAE | First-choice regret | Evaluate run |
|---|---:|---:|---:|---|
| Quadratic ridge | 66.4508 | 6.9151 | 4.5095 | `20261029-125034-evaluate-baseline-v32` |
| ExtraTrees | 68.0005 | 5.7045 | 5.8488 | `20261029-125034-evaluate-extratrees-v32` |
| HistGB | 72.3465 | 4.7919 | 1.8768 | `20261029-125034-evaluate-histgb-v32` |
| XGBoost | 73.4746 | 4.9764 | 1.0653 | `20261029-125034-evaluate-xgboost-v32` |

## Optimizer comparison

Model parameters and model seed remain fixed; optimizer seeds are 2026/2027. Every run requests
50 proposals. Dirichlet generates 12,000 candidates; DE generates 11,904 trial points, rejecting
infeasible points before scoring. Their prediction query counts therefore differ. Both methods use
risk = 1 and maximum distance ratio = 0.75 for this controlled comparison.

Each row reports the best *cross-model selection score among eligible returned candidates*, not
necessarily that optimizer's own first-ranked point. Lower score is better only under this heuristic.

| Search model | Optimizer | Seed | HistGB prediction | XGBoost prediction | Selection score | Distance ratio | Rank in source run |
|---|---|---:|---:|---:|---:|---:|---:|
| histgb | trust-dirichlet | 2026 | 108.9778 | 108.8717 | 112.3377 | 0.5410 | 15 |
| histgb | trust-dirichlet | 2027 | 108.6692 | 109.3996 | 112.8432 | 0.6507 | 3 |
| histgb | simplex-de | 2026 | 112.7253 | 111.7891 | 116.0853 | 0.5921 | 1 |
| histgb | simplex-de | 2027 | 113.6749 | 113.0111 | 117.0349 | 0.6574 | 12 |
| xgboost | trust-dirichlet | 2026 | 112.0136 | 99.3992 | 115.3736 | 0.7201 | 1 |
| xgboost | trust-dirichlet | 2027 | 112.2459 | 106.1412 | 115.6059 | 0.6920 | 9 |
| xgboost | simplex-de | 2026 | 114.0458 | 105.5885 | 118.1154 | 0.7433 | 36 |
| xgboost | simplex-de | 2027 | 115.4363 | 103.3958 | 118.7963 | 0.6400 | 5 |

Run IDs for this table are `20261029-125621-optimize-offline-<model>-<optimizer>-<seed>`.
DE did not improve on Dirichlet in this small study. This is evidence for today's choice, not a
claim that differential evolution is generally inferior. Initial radius-1.5 default runs and the
quadratic-PGD/ExtraTrees references were also included in the candidate pool. There were 551
proposal entries (some repeated across runs), of which 546 passed the selection constraints,
plus one separately marked measured reference.

The raw XGBoost minimum (source `20261029-125052-optimize-xgboost-v32`, rank 1) predicts 99.3992
under XGBoost but 112.0136 under HistGB; its selection score is 115.3736, worse than the submitted
point. The baseline PGD proposal (`20261029-125052-optimize-baseline`) predicts 75.8957 under ridge,
but its distance ratio is 1.7484 and it fails the submission distance guardrail.

## Checks and limitations

- Full GUI-inclusive suite: **82 passed, 1 known xfailed**. The expected failure is the unchanged
  baseline PGD duplicate-batch test. The selected source optimize run and a DE optimize run both
  reproduce within rtol=1e-9. Re-running the selection script produces byte-identical weights,
  selection metadata, candidate table and frozen-pool diagnostic table.
- Local sensitivity: 128 Dirichlet perturbations, concentration 2000, seed 2026; median perturbation
  distance 0.03049. HistGB prediction median / 10th / 90th percentiles: 110.9133 / 109.3645 / 111.9472;
  XGBoost: 109.2088 / 108.0010 / 111.1127. These summarize nearby model behavior, not real uncertainty.
- ExtraTrees disagrees materially: 125.2473, about 4.08 worse than its prediction at p01969. Its
  weaker frozen-v32 first-choice metric is why it is diagnostic rather than a selection judge;
  it still provides a reason to be cautious about the submission.
- `mean + std` is a heuristic penalty. The stored std proxies are not guaranteed confidence bounds
  at optimizer-selected points, and model agreement is not independent evidence.
- Applying the same distance-filtered selection rules to the frozen held-out record pools yields
  **5.1299 mean regret for all three rules** (HistGB, XGBoost, consensus), with identical choices on
  each split. Thus this check provides no evidence that consensus improves held-out selection.
  The strict distance filter excludes some good records; do not compare its regret as if it were
  the unfiltered frozen-v32 first-choice metric above.
- The real evaluator was not available in this repository. No submissions, online queries, or real
  evaluation budget were spent. `best_observed_reference.json` is a reference only; use of an existing
  training row as an official submission would require checking the course rules.

## Traceability and reproduction

All optimization/evaluation evidence runs were made from committed code with dirty=false.
Original evaluate/default optimize runs: commit `3f1a6d7`; controlled optimizer study: `ef4e060`;
selection script: `de001b9`. `provenance.json` records input file hashes, run IDs, package versions,
dataset checksum, the selection seed and the exact submitted weights hash.

The package includes complete candidate predictions and selection-rule diagnostic CSVs.
`run_evidence.json` file preserves the source configs, metadata and metrics for all 16 source runs.
The original models and full run folders remain under the repository's git-ignored `runs/` directory;
they are required for direct `sdm reproduce <run-id>`. They are not embedded in this lightweight
submission package. The study configurations are tracked in `experiments/offline-*.toml`.

From the project root, using the recorded local runs:

```bash
env -u PYTHONPATH uv run --group gui sdm predict \
  --model-run 20261029-125052-optimize-histgb-v32 \
  --weights analysis/offline_submission/submission/weights.json
env -u PYTHONPATH uv run --group gui sdm reproduce 20261029-125052-optimize-histgb-v32
```

The full selection command is in `reproduce_selection.sh`; pass a fresh output directory to
preserve the delivered artifacts. It recomputes all predictions and chooses the same weight file.
