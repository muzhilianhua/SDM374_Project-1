# Online result figures

Generate the running-best curve from the append-only evaluator log:

```bash
uv run python analysis/online_results/plot.py path/to/online_log.csv --out analysis/online_results/results
```

To add the prediction-versus-measurement figure, provide a CSV with exactly
`call,point,predicted_R`:

```bash
uv run python analysis/online_results/plot.py path/to/online_log.csv \
  --predictions path/to/predictions.csv --out analysis/online_results/results \
  --offline-baseline 115.68 --training-best 115.76
```

The script rejects missing, extra, duplicate, non-finite, or invalid keys and reports MAE and
prediction bias. It validates inputs before creating output. Actual `R` values always come from
the evaluator log, whose `n` column must certify complete batches (T07). Both figures print all source
run IDs; the scatter colors points by call. `summary.json` saves the metrics, per-call minima and
prediction errors, all run IDs, input SHA256 hashes, the plotting git state, and the reference values.
The offline target is met only for a strictly lower R. Thresholds are configurable; the defaults are
the rounded values recorded in `docs/decisions.md`. The default `results/` directory is git-ignored.
Git provenance is null if git is unavailable. Its dirty flag follows the shared `sdm.runs.git_state`
contract (core code/configuration paths); commit plotting code before generating final report figures.
