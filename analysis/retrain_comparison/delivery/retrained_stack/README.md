# Retrained selected stack

This is the new all-data fit, not an overwrite of artifacts/best_model. Validation belongs to
separately fitted outer-fold counterparts. Load through the existing CLI from repository root:

```sh
env -u PYTHONPATH uv run --locked --group gui sdm predict \
  --model-run analysis/retrain_comparison/delivery/retrained_stack \
  --weights analysis/robust_search/delivery/challenger_weights.json
```

The parent report provides exact run IDs, evaluation, comparisons and limitations.
