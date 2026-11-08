# T20 — Mechanism notes (offline)

Owner: member D. Backlog item: T20 in `docs/roles/D.md`.

This folder analyses how the 21 weights affect R **using only the course-provided offline table**.
No real evaluator calls were made for these results.

## Data

`dataset_same8/train.csv` — 2048 rows, 3 buildings x 7 objectives, each building's weights sum to 1,
target column `ramping`. Offline R range: 115.765–242.521 (median 161.914).
These offline R values are not comparable with online evaluator scores.

## Artifacts

- `T20_机理笔记.ipynb` — analysis notebook (Chinese); figures are pre-rendered and the code cells rerun the
  model comparison, grouped permutation importance and share-octile analysis.
- `figures/` — five figures used by the notebook.

## Findings

1. **Within-building ratio effect.** The share of the `ramping` objective is the dominant lever:
   Spearman with R is **-0.6434**; octile median R falls monotonically from **173.48 to 145.90** as the
   three-building average `ramping` share increases. `degradation` (+0.220), `carbon` (+0.192) and
   `cost` (+0.185) shares are positively associated with R; `peak`, `discomfort` and `resilience`
   are close to zero.
2. **Cross-building additivity does not hold.** A joint 21-feature GBM reaches CV R² = **0.7726**,
   whereas three per-building GBMs summed reach only **0.2630**; adding pairwise cross-building terms
   lifts a linear model from 0.5143 to 0.6247. R depends on the joint composition of the three buildings.
3. **Building 2 has low sensitivity.** Grouped permutation of building 2's seven weights reduces the
   joint model's R² by only **0.0728** (building 1: 0.9087, building 3: 0.4356); a model trained on
   building 2 alone gives CV R² = **-0.1375**.

## Modelling priors for B/C

1. Represent each building by its 7-dimensional share vector, not by raw magnitudes.
2. Use one joint 21-dimensional model; do not sum three independently fitted per-building models.
   If a decomposed model is required for interpretability, add explicit cross-building interactions.
3. Spend proposal/model budget on buildings 1 and 3; keep building 2 conservative.

## Limitations

- The 2048 rows are a fixed offline sample; they cannot resolve narrow plateaus or single-point notches
  in the online evaluator response surface.
- Permutation importance is model-dependent and is not a causal effect.
- "Building 2 is insensitive" holds within this offline sample and still needs online confirmation.

## Reproduce

Open `T20_机理笔记.ipynb` and run all cells. Random seed 0, 5-fold CV, no external packages beyond
numpy/pandas/scikit-learn/matplotlib.
