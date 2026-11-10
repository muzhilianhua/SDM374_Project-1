# T20 — Mechanism notes (offline)

Owner: member D (`docs/team.md`). This folder answers the three T20 questions in
`docs/online-tasks.md` using **only** the course-provided `dataset_same8` table (2048 rows).
No evaluator calls are made here.

| Question | Answer |
|---|---|
| Within-building ratio effect | The share of the `ramping` objective dominates: Spearman -0.6434, octile median R 173.48 -> 145.90 |
| Cross-building additivity | Holds well enough to model additively: additive-by-building GBM 0.8090 vs joint 0.7726 |
| Building-2 sensitivity | Confirmed low: permutation R² drop 0.0783 vs 0.9429 / 0.4416; single-building CV R² -0.1375 |

## Data and scale

`dataset_same8/train.csv`, 2048 rows, 3 buildings x 7 objectives, each building's weights sum to 1.
Offline R spans 115.765–242.521 (median 161.914).

The offline table and the evaluator are on the same scale: the solver smoke test reproduces offline
values exactly (`analysis/solver_smoke/README.md`) — training row `p01969` -> 115.76481925917324,
offline submission 1 -> 119.71206678869203, baseline W* -> 119.0916. What the 2048 rows cannot cover
is the *shape* of the response surface away from the sampled points, not the scale.

## Findings

### 1. Within-building ratio effect

Each building's seven weights are shares. The three-building mean share of `ramping` has
Spearman **-0.6434** with R; splitting on that share and taking octile medians gives a monotone
decline from **173.48 to 145.90**. Objective shares that co-move with higher R: `degradation`
(+0.2203), `carbon` (+0.1921), `cost` (+0.1853). `peak` (-0.0809), `discomfort` (-0.0427) and
`resilience` (-0.0333) carry almost no signal.

### 2. Cross-building additivity (revised)

A GBM whose interactions are constrained to stay inside each building scores **0.8090** 5-fold CV R²,
above the unconstrained joint GBM's **0.7726**. Linear models say the same thing more sharply:

| model | CV R² |
|---|---|
| ridge, additive | 0.5143 |
| ridge + within-building squares and products | 0.6703 |
| ridge + cross-building products only | 0.4466 |
| GBM, joint 21 features | 0.7726 |
| GBM, interactions constrained per building | **0.8090** |

So the interaction structure that matters is **within** a building; adding cross-building products
makes the linear model worse than plain additive. Treating the three buildings as separate additive
blocks is a sound inductive bias, and it is what B's additive ARD GP (offline score 0.8315) already
does.

> Correction: the first version of this note reported "three independent GBMs summed" at 0.2630 and
> concluded that additivity fails. That implementation fitted three models to the full target and
> averaged their centred predictions instead of fitting one additive model. It has been replaced by the
> constrained GBM and the two linear variants above. The earlier "ridge + all pairwise terms 0.6247"
> result mixed within- and cross-building terms and is superseded by the separated rows.

### 3. Building-2 sensitivity

Grouped permutation importance: one shared row permutation shuffles all seven weights of a
building together, so the composition (weights sum to 1) stays intact.

| building | permutation R² drop | single-building CV R² |
|---|---|---|
| 1 | 0.9429 | 0.4585 |
| 2 | **0.0783** | **-0.1375** |
| 3 | 0.4416 | 0.1506 |

Building 2 alone does not beat the mean, and shuffling its weights barely moves the joint model.
It still contributes inside a full composition, but it is the last place to spend proposal budget.

> Correction: the first version of this check permuted each of the seven columns independently,
> which broke each building's sum-to-1 composition (deviations up to 2.06) and reported
> 0.9087 / 0.0728 / 0.4356. The block permutation above is the documented grouped-building
> importance; the ranking is unchanged.

## Priors handed to B/C

1. Use the seven-dimensional share vector per building rather than raw magnitudes; `ramping` share is
   the first lever, then `degradation` / `cost` / `carbon`.
2. An additive-by-building model is appropriate. Do not drop the additive ARD GP for being additive,
   and do not add cross-building interaction features — they do not help.
3. Allocate proposal and feature budget to buildings 1 and 3; keep building 2 conservative.

## Artifacts

| path | contents |
|---|---|
| `study.py` | the whole study: share effect, additivity comparison, sensitivity |
| `results/20261109/results.json` | all numbers above |
| `results/20261109/provenance.json` | git state, dataset sha256, package versions, seed |
| `results/20261109/figures/` | `share_vs_R.png`, `additivity.png`, `building_sensitivity.png` |
| `t20_mechanism_notes.ipynb` | narrative notebook with saved outputs |

`analysis/**/results/` is gitignored (line 8 of `.gitignore`); `study.py` regenerates that folder, and
the notebook embeds the numbers and figures so the PR carries the outcome.

## Reproduce

```bash
uv run python analysis/mechanism/study.py
```

Seed 0, 5 folds, `HistGradientBoostingRegressor(max_iter=200)` with `interaction_cst` grouped per
building. The run is single-machine and takes a few minutes.

## Limits

- 2048 rows cannot resolve narrow plateaus or single-point notches in the evaluator response surface.
- Permutation importance is model-dependent and is not a causal effect.
- "Building 2 is insensitive" holds over this sample; it is a prior, not a guarantee, for online runs.
