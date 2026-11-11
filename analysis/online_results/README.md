# Online result figures

Generate the running-best curve from the append-only evaluator log:

```bash
python analysis/online_results/plot.py path/to/online_log.csv --out analysis/online_results/delivery
```

To add the prediction-versus-measurement figure, provide a CSV with exactly
`call,point,predicted_R`:

```bash
python analysis/online_results/plot.py path/to/online_log.csv \
  --predictions path/to/predictions.csv --out analysis/online_results/delivery
```

The script rejects missing or duplicate keys and reports MAE and prediction bias. The actual `R`
values always come from the evaluator log. The generated figures are report artifacts and should be
labelled with the run id or online log provenance.
