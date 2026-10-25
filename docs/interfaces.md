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

    def propose(self, model, X, y, n=1):   # -> (n, 21), best first, every row passes check_weights
        ...                                # optional: self.diagnostics = {str: float | int | bool}
```

- `X, y` are all observations so far: the 2048 training rows offline, plus evaluated points online.
- `model` is a fitted `Surrogate`; use only its public methods.
- Online rounds ask for `n = 50`; the rows must be distinct.

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

[evaluate]                     # team-wide protocol; do not change
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
- Baseline (ridge2) on this protocol: MAE 6.828, R² 0.646, Spearman 0.838, top-50 recall 0.46,
  std_coverage 0.76 (constant std = out-of-fold RMSE).

## Runs (`runs/<run-id>/`, `sdm.runs`, `sdm.catalog`)

Run id: `<YYYYmmdd-HHMMSS>-<kind>-<config file name or --name>`. `runs/` is git-ignored, so each machine
starts empty; run `sdm evaluate -c experiments/baseline.toml` to get a local reference.

| File | Written by | Content |
|---|---|---|
| `config.toml` | all | copy of the config used |
| `meta.json` | all | kind, options, git commit + dirty flag, dataset sha256, package versions |
| `metrics.json` | all | metrics; missing means the run failed |
| `predictions.csv` | evaluate | `candidate_id, actual, predicted_mean, predicted_std` |
| `model.pkl` | train, optimize | fitted model (pickle: load only runs made by this team) |
| `weights.json` | optimize | one 3×7 matrix (`-n 1`, same format as the course `weights.json`) or a list |
| `proposals.json` | optimize | per proposal: weights, predicted mean/std, nearest training distance |

`sdm.catalog.run_records(root=None)` → DataFrame of all runs (default root `runs/`); `leaderboard(records)` → evaluate runs by MAE;
`candidates(records)` → optimize runs with `distance_ratio` and an `extrapolating` flag (ratio > 1.5).
`runs/.jobs/` holds GUI background jobs and is not a run.

## CLI (`sdm`)

All subcommands accept the global `--runs-dir`. `list`, `runs` and `check-data` accept `--json`.
`train`, `evaluate`, `optimize` print `run: <path>` first; `reproduce` prints `rerun: <path>` and exits
1 when outputs differ beyond rtol 1e-9. `predict --weights -` reads JSON from stdin. Errors print
`error: ...` to stderr with exit code 2.
