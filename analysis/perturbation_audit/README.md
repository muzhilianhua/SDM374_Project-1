# Independent audit of the offline-candidate HTML

This protocol is fixed before scoring perturbations. Role C uses unchanged saved models and
existing candidate weights. There is no retraining, candidate optimization, real-evaluator call,
or change to the frozen-v32 protocol. All prediction batches use `sdm predict`.

## Reproduction boundary

The supplied `../离线候选表.html` describes seed 2026, forty random directions and radii0.02/0.05,
but its referenced scripts and sample CSVs are unavailable in this checkout. Its Nov1 trained
model folders are also absent. We use the archived Oct29 HistGB/XGBoost/ExtraTrees fitted models
and published stack. First compare the five shortlist single-point differences and p01969
predictions with the HTML at its displayed precision. A match is numerical corroboration, not
proof of byte-identical models or an exact reproduction of the missing perturbation code.

## Candidate panel and endpoints

Rescore the existing1503 common-judge inputs plus the consensus challenger. Retain every existing
proposal with worst(stack,HistGB,XGBoost) predicted difference versus p01969 below-6. Add all five
HTML shortlist candidates and the original seed 2026 fixed-blend rank 1 regardless of screening.
The challenger is included. This panel is fixed from single-point predictions before perturbation
outcomes; it is not all possible feasible W and excludes weaker pool points from perturbation
ranking. Candidate selection and all models reuse the original data.

All candidates share identical tangent Gaussian directions within each seed; the same directions
are reused at both radii. Subtract each building's direction mean and normalize the entire21-vector
to unit Euclidean norm. Report actual distance after enforcing feasibility.

1. Descriptive HTML-like check: 40 directions, seed 2026, both radii. Compare (a) Euclidean simplex
   projection at the nominal step, and (b) nonnegative clipping followed by per-building
   normalization. These are explicitly defined alternatives to an unspecified original method.
2. Main audit: 512 directions per seed 2027/2028/2029, both radii. Follow the projected ray and solve
   its step by bisection so every final feasible point has actual Euclidean distance equal to the
   requested radius. This prevents apparent stability from smaller boundary-projected moves.
   If a ray is stationary under projection (possible at a simplex vertex), reverse that ray
   before solving its length; record this deterministic geometric convention in the source.

For every neighbor, subtract each judge's own p01969 prediction, then take the largest difference
across the three judges. Primary endpoint: fraction with this worst difference strictly below-3.
Secondary: median/q90 difference, per-seed fractions and achieved distances. Paired differences
against the challenger use shared(seed,direction) samples. Conditional normal-approximation
intervals describe finite direction sampling with fixed judges; they do not quantify real-R
uncertainty, independent generalization, or familywise superiority over the candidate panel.

Decision: report whether challenger leads the prespecified panel at both radii and whether that
ordering is consistent across the three audit seeds and alternative nominal-step definitions.
Do not reinterpret input perturbations as training-subsample stability, true improvement, or a
replacement of the prior operational W under its separate gate. No actual-R estimate is inferred.

Run from repository root after committing source:

```sh
env -u PYTHONPATH uv run --locked --group gui python analysis/perturbation_audit/audit.py \
  --html ../离线候选表.html --out analysis/perturbation_audit/delivery
```

The delivery archives source hashes, exact candidate weights, prediction commands, compressed
neighbor weights and predictions, per-seed/pooled summaries, paired comparisons, single-point
HTML checks and a generated report. The original HTML is read-only and outside the repository.
