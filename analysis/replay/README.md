# T18 — Strategy parameter comparison

Owner: member D. Backlog item: T18 in `docs/roles/D.md`.

This folder contains the strategy-parameter comparison for the online phase and the recommended
settings. R values come from **local development runs of `solver.exe`**, recorded in
`evidence/eval_log.jsonl` (6,568 single-call evaluations). They are not graded or official online
calls; the official 20-call log remains the source of truth for scoring.

## Artifacts

- `T18_策略参数比较与推荐.ipynb` — comparison notebook (Chinese), figures pre-rendered.
- `figures/` — six figures generated from `eval_log.jsonl`.
- `evidence/eval_log.jsonl` — one record per evaluated weight vector (21 weights, R, phase, timestamp).
- `evidence/best_weights_phaseE5.json` — best configuration found (eval_id 6335).

## Results

- **Best configuration:** 4 non-zero slots per building (12 non-zero in total; F5/F6 dominated,
  one small slot each), R = **104.851733** (eval_id 6335).
- **Strategy families** (best R per family): random 21-D 124.595; single-feature vertices 120.299;
  blind random over all sparsity levels 109.443; 5-non-zero supports 108.909; single-building
  heterogeneous supports 104.852.
- **Step sizes:** 0.20 coarse grid reaches 109.456; 0.00125 one-dimensional scan reaches 104.852;
  refining below 0.0025 yields only 0.1–0.9 improvement per notch.
- **Recommendation to T15:** exploit the support set first; one-dimensional `r = F5/(F5+F6)` scan
  afterwards; keep exact zeros (never `1e-9`); batch sizes and steps in the notebook table.

## Caveats

- This notebook compares strategies on locally measured points; it is not a substitute for T17's
  reproducible replay (table hold-out pool or semi-synthetic truth).
- The evidence log contains development runs, including quota resets; it must not be presented as the
  graded online log.
- The recommendation still has to be copied into `docs/decisions.md` and `experiments/online-*.toml`
  once T15/T12 interfaces are final; that step is pending.

## Reproduce

Open `T18_策略参数比较与推荐.ipynb`; code cells read `evidence/eval_log.jsonl` and
`evidence/best_weights_phaseE5.json` relative to this folder.

## Full evidence bundle

- `evidence/eval_log.jsonl` — 6,568 records from local evaluator runs (weights, R, phase, timestamp).
- `evidence/optimization/` — phase progress files, stdout/stderr logs, diagnostics and semantic probes
  from `vE1` and `v5_phaseA`–`v5_phaseD`.
- `evidence/weights/` — every `best_weights*.json` snapshot kept during the search.
- `evidence/E3_summary.json`, `E4_progress.json`, `E5_summary.json` — aggregated terrain, line-scan
  and quantisation results.
- `evidence/dedupe_report.csv` — audit of overlap between the offline table and probe candidates.
- `figures/` — all figures generated from these runs.

Excluded on purpose: `solver.exe`, `usage.count`, local runner scripts and `__pycache__`.
