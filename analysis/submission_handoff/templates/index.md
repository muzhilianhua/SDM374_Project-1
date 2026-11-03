# MEMBER C — REVIEWED HANDOFF

**P1 was offline submission 1: real R = 119.71206678869203, above the bar 115.7648. Do not resubmit it.**
Current course rules and team decisions are in [docs/decisions.md](../docs/decisions.md), decisions 17–23.

| Label | Current use | Payload | Alias |
|---|---|---|---|
| **P1 — MEASURED, FAILED** | Historical submission; exclude from further graded calls | [weights](01_FIRST_REAL_TEST/weights.json) | r01-004 |
| **P2 — HISTORICAL DEFAULT** | Preserved original recommendation; not selected for submission 2 | [weights](02_ORIGINAL_RECOMMENDATION/weights.json) | r01-003 |
| **P3 — EXPLORATORY ONLY** | Unmeasured research candidate | [weights](03_EXPLORATORY_ONLY/weights.json) | direction probe p01 |
| **OFFLINE REFERENCE MODEL** | Honest-stack; predictive performance does not establish real controller improvement | [model and restore instructions](04_RECOMMENDED_MODEL/README.md) | [parameters](04_RECOMMENDED_MODEL/parameters.json) |
| **50-POINT PACKET — SUPERSEDED** | Historical idea pool only; do not submit as a batch | [weights](05_ONLINE_ROUND1_50/weights.json) | [current status](05_ONLINE_ROUND1_50/README.md) |
| **20 DIRECTION PROBES — EXPLORATORY** | Research pool, not a frozen 20-call policy | [weights](06_DIRECTION_PROBES_20/weights.json) | [manifest](06_DIRECTION_PROBES_20/manifest.csv) |
| **EVIDENCE** | Historical evaluations and audits, with current status recorded separately | [index](07_EVIDENCE/README.md) | [manifest](MANIFEST.json) |

Legacy directory names are preserved for existing links. `01_FIRST_REAL_TEST` no longer indicates priority.

## Real results and remaining uncertainty

| Controller | Same reference-model predicted R | Real R |
|---|---:|---:|
| P1 consensus challenger | 111.418998 | **119.71206678869203 — failed** |
| P2 original recommendation | 112.081146 | UNKNOWN |
| P3 later exploration | 110.613116 | UNKNOWN |

P1's prediction was optimistic by about 8.29 R units. P2 and P3 carry the same model-selection risk;
their lower predictions do not establish improvement. P2's former training-subsample gate and P1's
input-neighborhood audit are historical evidence, not current submission endorsements. Offline
submission 2 must be chosen under decision 23; this package does not nominate it.

Baseline W* = 119.0916 is TA-confirmed. Training row p01969 = 115.76481925917324 is already known.
The real-score sources and payload hashes are in [KNOWN_OBSERVATIONS.json](KNOWN_OBSERVATIONS.json).
These records were imported from the team's decisions, without a new evaluator call or a change to
the frozen training data. P2, P3 and the direction probes remain unmeasured.

## Current online rules

The submitted policy may make **at most 20 sequential evaluator calls, one W per call**, and starts
with only the 2048 training rows. It must not repeat known points or training rows. The historical
50-point packet is an idea pool; r01-004 (P1), r01-046 (baseline W*) and r01-047 (p01969) are excluded
from graded calls. The separate 20 probes are also not a ready-to-run sequential policy.

## Payload format

Each single weights.json is a plain 3 × 7 numeric matrix: buildings 1/2/3, with columns
**cost, carbon, peak, discomfort, degradation, ramping, resilience**. Entries are finite and
nonnegative; each row sums to 1. Pass one individual file to the evaluator, not an N × 3 × 7 pool.
Weight bytes and aliases are preserved. P3/probes use 8 decimals; historical weights retain their
original precision, including tiny positive values.

## Model, evidence and verification

The offline reference is an OOF XGBoost/CatBoost/HistGB blend with label-excluded error calibration.
The selection-corrected recipe failed its replacement gate. Exact run IDs, metrics and limitations
are in [the model card](04_RECOMMENDED_MODEL/README.md) and [source_runs.json](07_EVIDENCE/source_runs.json).
Scores on reused frozen folds do not establish independent SOTA or real-R improvement.

Large model/evidence files are release assets. Restore them from the repository root before loading
the model or verifying all package files:

```sh
uv sync --locked --group gui
python3 analysis/submission_handoff/assets.py restore
env -u PYTHONPATH uv run --locked --group gui python analysis/submission_handoff/package.py verify
```

[Download the complete updated handoff ZIP](https://github.com/muzhilianhua/SDM374_Project-1/releases/download/c-pr7-evidence-2026-11-03/SUBMISSION_READY_2026-11-03.zip).
[Asset hashes and restoration details](../analysis/submission_handoff/README.md).
The repository supplies source code and uv.lock; [CHECKSUMS.json](CHECKSUMS.json) and
[VERIFICATION.json](VERIFICATION.json) record the package checks.
