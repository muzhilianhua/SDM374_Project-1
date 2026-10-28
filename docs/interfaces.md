# Interfaces

Reference for everything a plugin, script or agent touches. Source of truth is the code in `src/sdm/`;
if this file and the code disagree, the code wins and this file needs a fix.

## Weight layout (`sdm.simplex`)

| Name | Value |
|---|---|
| `METRICS` | `("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")` |
| `FEATURES` | `b1_cost … b1_resilience, b2_cost …, b3_cost … b3_resilience` (train.csv column order) |
| `DIM` | 21 (`N_BUILDINGS = 3` × `N_METRICS = 7`) |
| `as_batch(X)` | accepts `(21,)`, `(3, 7)`, `(n, 21)` or `(n, 3, 7)`; returns `(n, 21)` float |
| `check_weights(X)` | raises `ValueError` unless finite, non-negative, each building sums to 1 (tol 1e-9) |
| `project(X)` | Euclidean projection of each building's 7 weights onto the simplex; keeps shape |
| `sample(n, rng)` | `n` uniform samples on the product of three simplices |

## Surrogate (`sdm.interfaces.Surrogate`)

```python
@register_model("name")                    # sdm.registry
class MyModel(Surrogate):
    """One-line summary shown by `sdm list` and the GUI."""

    @dataclass(frozen=True)
    class Params:                          # every field needs a default; lists in TOML become tuples
        seed: int = 0

    def fit(self, X, y) -> "MyModel": ...  # X (n, 21), y (n,); return self; optional self.metrics = {str: float}
    def predict(self, X): ...              # return (mean, std), both (n,), std >= 0 and finite
    def gradient(self, X): ...             # optional: d mean / d W, (n, 21); needed by gradient optimizers
    def smoothness(self) -> float: ...     # optional: Lipschitz bound of gradient; pgd derives its step from it
```

- `std` is mandatory. If the model has no per-point uncertainty, use a constant such as the CV RMSE
  (that is what `ridge2` does).
- Do not override `__init__`; the base class sets `self.params` and `self.metrics`.
- Models are pickled into `runs/<id>/model.pkl`; keep attributes picklable (no open files, lambdas).

## Optimizer (`sdm.interfaces.Optimizer`)

```python
@register_optimizer("name")
class MyOptimizer(Optimizer):
    """One-line summary."""

    @dataclass(frozen=True)
    class Params:
        seed: int = 0

    def propose(self, model, X, y, n=1):   # -> (n, 21), best first by its own objective; rows pass check_weights
        ...                                # optional: self.diagnostics = {str: float | int | bool}
```

- `X, y` are all observations so far: the 2048 training rows offline, plus evaluated points online.
- `model` is a fitted `Surrogate`; use only its public methods.
- Online rounds ask for `n = 50`; the rows must be distinct.

## Registered plugins

| Name | Kind | Notes |
|---|---|---|
| `ridge2` | model | course baseline (frozen); has `gradient` / `smoothness`; constant std |
| `extratrees` | model | extra-trees, std = spread across trees; no gradient (piecewise constant) |
| `pgd` | optimizer | course baseline (frozen); needs `gradient` + `smoothness`; returns duplicates for n > 1 |
| `trust-dirichlet` | optimizer | gradient-free Dirichlet local search from the best rows, inside `max_distance_ratio · median_spacing` of the data, ranks by mean + risk · std, distinct batches |

## Registry (`sdm.registry`)

`models()`, `optimizers()` → `{name: class}`; `build_model(name, params_dict)`,
`build_optimizer(name, params_dict)`; `describe(cls)` → `[{"name", "type", "default"}]`.
Modules whose file name starts with `_` are not auto-imported (templates, shared helpers).

## Config (`sdm.config`)

```toml
[data]
dataset = "dataset_same8"      # relative to the repo root

[model]
name = "ridge2"
[model.params]                 # only fields of the plugin's Params
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

[optimizer]                    # required for `sdm optimize`
name = "pgd"
[optimizer.params]
starts = 16

[evaluate]                     # team-wide protocols; do not change the numbers
protocol = "kfold"             # or "frozen-v32", which must omit folds and seed
folds = 5
seed = 2026
top_k = 50
```

`load_config(path)` validates sections; unknown plugin parameters fail when the plugin is built.
`dump_config(dict)` writes TOML; `write_new_config(name, dict)` validates and creates
`experiments/<name>.toml` without overwriting (used by the GUI form).

## Data (`sdm.data`)

`load_dataset(REPO_ROOT / "dataset_same8")` → `Dataset(ids, X (2048, 21), y (2048,), version, sha256,
path)`. It verifies the manifest sha256 and sorts by `candidate_id` (the CV folds depend on this order).

## Evaluation (`sdm.evaluate`)

- `cross_validate(build, X, y, folds=5, seed=2026, top_k=50)` → `mae, rmse, r2, spearman,
  top50_recall, mean_predicted_std, std_coverage` (share of rows with |error| ≤ std; ≈ 0.68 if calibrated). `build` is a zero-argument factory returning an unfitted model.
- `out_of_fold(...)` → `(mean, std)` per row; `score(y, mean, std)` → the metrics above.
- `nearest_distances(points, X)` → distance from each point to its nearest training row.
- `median_spacing(X)` → median nearest-neighbour distance inside `X` (0.377 for the training table); the
  unit for trust regions and the `distance_ratio` extrapolation flag.
- Baseline (ridge2) on this protocol: MAE 6.828, R² 0.646, Spearman 0.838, top-50 recall 0.46,
  std_coverage 0.76 (constant std = out-of-fold RMSE).

## Frozen v3.2 protocol (`sdm.frozen_v32`, `[evaluate] protocol = "frozen-v32"`)

- Inputs are member B's frozen releases, read in place and checked by sha256: the outer splits from
  `frozen_data/SDM374_data_splits_frozen_v1.zip` (2 repeats × 5 folds, every row tested once per repeat) and
  the approved scorer `evaluation_v3_2/surrogate_scoring_v3_2.py` with its policy JSON. Rules of the score:
  `evaluation_v3_2/README.md`.
- `load_outer_splits(dataset)` → 10 `OuterSplit(split_id, repeat, train, test)` with row positions.
- `out_of_fold(build, dataset, splits)` → DataFrame with one prediction per (split, test row).
- `score_predictions(predictions, dataset, splits, top_k=50)` → `(metrics, fold_scores)`. It raises
  `ValueError` unless the rows are exactly the 10 frozen splits, one row per test id of each split, with the
  split's `repeat` and the dataset's `ramping` as `actual` (saved prediction files are checked, not trusted). Metrics:
  `v32_score` (mean over the 10 splits, 0–100, higher is better), `v32_score_std` (population SD across
  splits; descriptive only, the splits reuse rows), `v32_points_<component>` for `local_order`, `local_mae`,
  `global_order`, `global_mae`, `top1_regret`, the raw `v32_local_pair_accuracy`, `v32_global_pair_accuracy`,
  `v32_local_mae`, `v32_global_mae`, `v32_top1_regret`, and the k-fold metrics above computed on each
  repeat's out-of-fold predictions and averaged over the two repeats.
- The scale `s` of each split is the population SD of that split's outer-train labels. The inner 3-fold
  lists in the release are not used by `sdm evaluate`; a plugin that tunes on v3.2 may read them.
- Baseline (ridge2) on this protocol: v32_score 66.45 (SD 9.19), top-1 regret 4.51, low-region MAE 11.97,
  MAE 6.915 — run `20261027-174545-evaluate-baseline-v32`.

## Runs (`runs/<run-id>/`, `sdm.runs`, `sdm.catalog`)

Run id: `<YYYYmmdd-HHMMSS>-<kind>-<config file name or --name>`. `runs/` is git-ignored, so each machine
starts empty; run `sdm evaluate -c experiments/baseline.toml` to get a local reference.

| File | Written by | Content |
|---|---|---|
| `config.toml` | all | copy of the config used |
| `meta.json` | all | kind, options, git commit + dirty flag, dataset sha256, package versions |
| `metrics.json` | all | metrics; missing means the run failed |
| `predictions.csv` | evaluate | `candidate_id, actual, predicted_mean, predicted_std`; `frozen-v32` adds `split_id, repeat` (each row appears twice) |
| `fold_scores.csv` | evaluate (`frozen-v32`) | per outer split: total, the five v3.2 points, pair accuracies, MAEs, top-1 regret and candidate |
| `model.pkl` | train, optimize | fitted model (pickle: load only runs made by this team) |
| `weights.json` | optimize | one 3×7 matrix (`-n 1`, same format as the course `weights.json`) or a list |
| `proposals.json` | optimize | per proposal: weights, predicted mean/std, nearest training distance |

`sdm.catalog.run_records(root=None)` → DataFrame of all runs (default root `runs/`); `leaderboard(records)` → evaluate runs, `frozen-v32` first by `v32_score` (descending), then `kfold` by MAE;
`candidates(records)` → optimize runs with `distance_ratio` and an `extrapolating` flag (ratio > 1.5).
`runs/.jobs/` holds GUI background jobs and is not a run.

## CLI (`sdm`)

All subcommands accept the global `--runs-dir`. `list`, `runs` and `check-data` accept `--json`.
`train`, `evaluate`, `optimize` print `run: <path>` first; `reproduce` prints `rerun: <path>` and exits
1 when outputs differ beyond rtol 1e-9. `predict --weights -` reads JSON from stdin. Errors print
`error: ...` to stderr with exit code 2.
