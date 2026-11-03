# RECOMMENDED MODEL — honest-stack, seed 2026

**Restore model.pkl with `python3 analysis/submission_handoff/assets.py restore` from the repository root, then load it from this directory.** This is the freshly fitted recommended full-data model;
the selection-corrected experiment did not pass its replacement gate. The older published model
in artifacts/best_model is historical. The release asset provides the original fitted bytes; original config/meta/metrics, resolved
parameters and environment versions remain tracked here. Code and uv.lock are in the repository root.

| Member | Iterations | Learning rate | Main setting | Fitted blend weight |
|---|---:|---:|---|---:|
| XGBoost |600|.10|max_depth 3,subsample.9|.2920246172436632|
| CatBoost |600|.05|depth 5,l2_leaf_reg 5,border_count 128|.4437867721586836|
| HistGB |600|.05|max_leaf_nodes 15,min_samples_leaf 20|.2641886105976533|

Inputs are raw 21 weights in parameters.json order; 2048 measured rows, seed 2026. XGBoost/CatBoost
weight the lowest 20% training-target rows3x; HistGB uses uniform weights. Convex blend weights
use 3-fold training OOF predictions. Independent nested3-fold residual calibration uses 4 distance
bins and the .68 absolute-error quantile. The returned std is a heuristic error scale.

| Frozen-v32 comparison | Score (higher is better) | MAE | Top1 regret |
|---|---:|---:|---:|
| Recommended recipe, seed 2026 |75.1136584760|4.6932867464|1.0632429468|
| Recommended recipe, three-seed mean |75.3590513811|4.6980812519|1.1078203701|
| Selection-corrected experiment, three-seed mean |75.2920482885|4.6934024481|1.1078203701|

The binary is seed 2026, not an ensemble of three training seeds.
Source fit: `20261101-224249-train-selection-study-reference-fit`.
Reference evaluations: `20261101-223009-evaluate-selection-study-reference-2026`,
`20261101-223250-evaluate-selection-study-reference-2027`,
`20261101-223538-evaluate-selection-study-reference-2028`.
P1 was predicted at 111.418998 but measured 119.71206678869203; the model
remains an offline reference, with no demonstrated real improvement at its searched optimum. See
../07_EVIDENCE/model_evaluation.csv and source_runs.json for exact run IDs and metrics. The
frozen-v32 splits/scorer are unchanged. Reused offline folds do not establish independent SOTA
or a measured improvement for a new W. Three-region generalization remains unresolved.

From the repository root:

```sh
uv sync --locked --group gui
env -u PYTHONPATH uv run --locked --group gui sdm predict \
  --model-run SUBMISSION_READY/04_RECOMMENDED_MODEL \
  --weights SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json
```

Run evaluate/train with experiments/selection-reference-2026.toml to refit from the unchanged
dataset. Training/optimization implementations live in src/sdm; this handoff adds no new model.
