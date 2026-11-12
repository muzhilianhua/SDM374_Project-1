# Evaluation and ablation tables

This folder contains the report-facing table builder. It reads compact, archived `sdm evaluate` run records and writes a CSV, Markdown table, and provenance record. Every row retains its run id.

From the repository root:

```bash
uv run python analysis/ablation/build_table.py
```

The default input is `analysis/ablation/evidence/` and the default output is the ignored `analysis/ablation/results/`. Each evidence subdirectory contains only `config.toml`, `meta.json`, and `metrics.json` copied from the original run. Do not rank `frozen-v32` and `kfold` rows against one another: the protocols use different splits and metrics.

This delivery covers T23 and the evaluation-only portion of T24. Optimizer linkage and feature ablations are intentionally left pending; `--include-unverified` is available for exploratory historical rows and must not be used as report evidence.

By default the generator requires a recorded commit, explicit `dirty=false`, and an archived config
matching the current `experiments/<run-name>.toml`. It never silently incorporates the old offline
studies, whose configs remain at `offline-final`. To inspect your own current runs use
`uv run python analysis/ablation/build_table.py --runs-dir runs`.
`evidence/CHECKSUMS.json` records the exact archived bytes; tests verify all six source files.
These compact records rebuild the table, not the training predictions. To refit the baselines run
`uv run sdm evaluate -c experiments/baseline-v32.toml` and
`uv run sdm evaluate -c experiments/baseline.toml`; each produces a new run ID. The archived run
commit, data hash and package versions are recorded in its unmodified `meta.json`.

The baseline evidence used in the current report is:

- `20261109-222524-evaluate-baseline-v32`: frozen-v3.2 score 66.4508, top-1 regret 4.5095.
- `20261109-222557-evaluate-baseline`: five-fold baseline, MAE 6.8275 and top-50 recall 0.46.
