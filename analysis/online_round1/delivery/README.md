# Online round1: 50 feasible weights, prepared for later real testing

This packet contains **exactly50 distinct controller weight sets**, with stable IDs r01-001..r01-050.
Each W is3x7: buildings1/2/3, then cost/carbon/peak/discomfort/degradation/ramping/resilience.
**No real evaluator has been called and no official submission has been made.** Every result row is
pending, including the intentional calibration repeat. The existing operational W is preserved.

## Course slide requirements and project representation

The supplied `SDM374_Project1_v3.pdf` was text-extracted and visually checked on pages3 and7.
Page3 requires each building's7 weights to be nonnegative and at least one positive; page7 permits
at most20 online iterations and at most50 evaluated points per iteration, minimizing R. The exact
source PDF checksum and page references are in `constraints.json` and `provenance.json`.
The source PDF remains in its original local location, not copied into this packet.

We keep each building's weights summing to1, following the repository data/model convention. This is
stronger than the slides' stated feasibility requirement, and satisfies it. Slides do not specify
JSON/CSV schema, a trust radius, exploration ratios, or a decimal-precision requirement.
File-schema compatibility with the actual evaluator must be checked when it is released.

## Files to use

- **`weights.json`**: a plain numeric50x3x7 JSON array, in candidate-ID order. This is the main batch.
- **`weights_w0_w20.csv`**: candidate_id plus21 columns w0..w20, matching the slide's building-major order.
- **`candidates.csv`**: identical values using repository names b1_cost..b3_resilience.
- **`weights/r01-001.json` ... `weights/r01-050.json`**:50 individual3x7 JSON files for per-point APIs.
- **`plan.csv`**: strategy, provenance, predicted means/proxies for five models, distance to training,
  selection scores, intentional-retest flag and the known offline R where available.
- **`pending_results.csv`**: round, ID,21 weights, blank measured_R/evaluator_version, pending status, notes.
  This is a handoff worksheet, not a change to the project's future shared online-log schema.
- **`round1_package.zip`**: ready-to-transfer weights, plan, worksheet, constraints and validation.
- `source_pool.csv.gz`, `pool_weights.json.gz`, `predictions-*.json.gz`: the entire candidate pool and
  original CLI predictions for tracing how the final50 were selected.
- `model_evidence.json`, `provenance.json`, `validation.json`, `budget.json`, `checksums.json`: provenance,
  reproducibility, constraint checks, and planned budget. Actual team budget usage is not assumed.

Do not submit prediction columns or the whole ZIP to an evaluator unless its documented schema accepts
that format. The numeric JSON payload and individual JSON files contain only the controller weights.
CSV uses17 significant digits to retain original anchor values; new samples were rounded to8 decimals
with per-building sum repair. Reference/controller anchors retain their original full precision.

## Portfolio composition

| IDs | Count | Role |
|---|---:|---|
| r01-001..r01-030 |30| Exploit promising regions while preserving spatial diversity |
| r01-031..r01-045 |15| Learn where the stronger models disagree |
| r01-046..r01-050 |5| Baseline, calibration and physical-objective controls |

This30/15/5 split is a practical first-round heuristic; it has not been chosen by an offline replay
and is not required by the slides. All50 planned evaluations, including controls and the deliberate
known-row repeat, count toward the50-point round limit. The packet does not reserve server budget.

Exploitation preserves four previously discussed anchors before filling the remaining26 slots by the
maximum mean+error-proxy across the stack, weighted CatBoost, HistGB and XGBoost. Each exploitation
point is novel and within distance ratio .75. Exploration accepts distance ratio<=1 and stack
prediction<=baseline prediction+15, then balances committee disagreement, heuristic error proxy,
and distance from already selected points. These points intentionally trade some predicted performance
for information; the batch is not50 copies of the lowest predicted point.

The candidate pool includes1500 prior search proposals plus four anchors, five controls,3600 seeded
local Dirichlet draws around64 low-R observations and the anchors, and600 Dirichlet(.6) coverage draws.
Selection uses minimum Euclidean separation .25 times the dataset's median nearest-neighbor spacing.
ExtraTrees predictions are diagnostic; that model is not a selection judge.

## Important anchors and controls

| ID | Purpose | Stack prediction | Distance ratio |
|---|---|---:|---:|
| r01-001 | fixed-blend-primary | 109.3303 | 0.7112 |
| r01-002 | fixed-blend-posthoc | 109.1711 | 0.5542 |
| r01-003 | operational | 112.0811 | 0.5410 |
| r01-004 | consensus-challenger | 111.4190 | 0.6382 |
| r01-046 | course-baseline | 118.7316 | 1.7484 |
| r01-047 | p01969-calibration-repeat | 118.1777 | 0.0000 |
| r01-048 | uniform | 156.4005 | 0.4604 |
| r01-049 | pure-ramping | 127.1988 | 2.4160 |
| r01-050 | ramping-resilience-half | 123.8776 | 1.9854 |

- r01-001 is the fixed-blend generator's original primary controller; r01-002 is its explicitly post-hoc
  alternative. r01-003 retains the existing operational recommendation, and r01-004 is the prior
  separately explored consensus challenger. All four are included rather than silently replaced.
- r01-046 is the exact course baseline W, so it establishes a common real-evaluator reference later.
- r01-047 is **p01969**, known offline R115.76481925917324. It intentionally repeats one offline
  observation to check evaluator version, scenario definition and numerical agreement. Its new measured_R
  remains blank; do not treat the old label as a fresh simulator result or append duplicate training rows
  automatically. If the evaluator/scenarios differ, its old R is contextual rather than an expected match.
- r01-048 is uniform over objectives. r01-049 uses only ramping; r01-050 assigns .5 to ramping and .5 to
  resilience in every building. These test simple physical-objective priors. Baseline and the last two
  controls are outside the .75 exploitation radius. They remain course-feasible, but their surrogate
  predictions have greater extrapolation uncertainty. The uniform control is not selected for low R.

The unchanged stack model in `artifacts/best_model/` provides the main prediction column. All five
models are reused without retraining and score the candidate pool via `sdm predict`. Error proxies are
training-fold residual quantiles, not calibrated confidence intervals. Predictions on searched points
can be optimistic; especially small differences do not establish a real improvement.

## Validation and preparation budget

`validation.json` records the actual checks:50 finite nonnegative W, each building has positive mass
and sum1 (maximum floating-point sum residual about2.22e-16),50 distinct within-batch points,49 novel
points relative to the training table, and one deliberate calibration repeat. Minimum pair distance
is about .09482. JSON, both CSV representations, the50 individual files and the pending worksheet agree.
A fresh `sdm predict` on the final50-point JSON matches the stored stack predictions within1e-9.
Full suite:112 passed and1 known expected failure (unchanged baseline PGD duplicates).

`budget.json` records planned round1/50 points and **zero real calls performed by this task**.
No offline submission opportunity or online round was consumed. Only preparation files were written;
course baseline files, original dataset, frozen releases and evaluation protocols are unchanged.

## Use for later online testing

1. Check the actual evaluator's documented input format, scenario/week definition, version and budget.
   Use the plain batch JSON or matching individual files; retain the stable IDs when recording results.
2. Record each fresh result in pending_results.csv, including evaluator version and status. Keep unknown
   R blank; do not replace it with a model prediction. Keep baseline/control measurements identifiable.
3. Compare the known-row control with the old offline label only if the evaluator definition matches.
   Review baseline and model prediction errors, then retrain on valid new observations for round2.
4. Keep this round's weights and provenance immutable. Create subsequent round packets separately;
   never edit the protected offline dataset or frozen validation splits to add new observations.

From the repository root:

```bash
env -u PYTHONPATH uv run --locked --group gui python analysis/online_round1/prepare.py verify \
  --out analysis/online_round1/delivery --replay-cli
env -u PYTHONPATH uv run --locked --group gui sdm predict \
  --model-run artifacts/best_model --weights analysis/online_round1/delivery/weights.json
# Full rebuild into a new ignored directory; requires the four original local auxiliary model runs:
env -u PYTHONPATH uv run --locked --group gui python analysis/online_round1/prepare.py build \
  --out analysis/online_round1/results/recheck
```

Source generation commit is recorded in provenance.json; the source scoring pass used clean1309a05.
The original candidate-method labels and exact source run/rank are retained in plan.csv/source_pool.csv.gz.
The supplied PDF is needed for a full rebuild's source checksum; archived feasibility verification only
needs the repository dataset and packet. Replay uses the published stack binary and locked dependencies.
