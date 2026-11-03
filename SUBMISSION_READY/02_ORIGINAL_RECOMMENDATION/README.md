# P2 — HISTORICAL ORIGINAL DEFAULT; not selected for submission 2

**Payload: [weights.json](weights.json)** — one numeric 3x7 matrix, buildings 1/2/3.
Column order: cost,carbon,peak,discomfort,degradation,ramping,resilience.

Source: `analysis/offline_submission/submission/weights.json`. Source candidate: `operational`. Exact source bytes are preserved.
Recommended-model prediction: **112.081146**; heuristic error scale:
3.563397; distance ratio:0.541002.
**Measured R: UNKNOWN.**
Predictions are not simulator results. Error scales are not confidence bounds.
Also present as `r01-003`; these are aliases of the same controller, not separate discoveries.

P1 failed the real bar. P2 preserves the historical default; P2/P3 are not selected for submission 2.
P1 was optimistic by about 8.29 R; the other surrogate predictions carry the same risk.
Use [the package index](../README.md) for the complete decision and evaluation evidence.
