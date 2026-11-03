# P1 — MEASURED, FAILED; do not resubmit

**Payload: [weights.json](weights.json)** — one numeric 3x7 matrix, buildings 1/2/3.
Column order: cost,carbon,peak,discomfort,degradation,ramping,resilience.

Source: `analysis/selection_training/delivery/first_test_weights.json`. Source candidate: `challenger`. Exact source bytes are preserved.
Recommended-model prediction: **111.418998**; heuristic error scale:
3.563397; distance ratio:0.638171.
**Measured R: 119.71206678869203 (FAILED; do not resubmit).**
Predictions are not simulator results. Error scales are not confidence bounds.
Also present as `r01-004`; these are aliases of the same controller, not separate discoveries.

P1 failed the real bar. P2 preserves the historical default; P2/P3 are not selected for submission 2.
P1 was optimistic by about 8.29 R; the other surrogate predictions carry the same risk.
Use [the package index](../README.md) for the complete decision and evaluation evidence.
