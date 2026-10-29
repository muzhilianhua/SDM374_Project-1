# Recommended trained surrogate

`model.pkl` is the **existing full-data OOF convex ensemble**, copied byte-for-byte from
`20261029-171251-train-method-blend-stack`. It combines XGBoost, CatBoost and HistGradientBoosting.
This upload packages the trained artifact; it does not retrain or change it.

The recipe scored75.1137 at model seed2026 under the unchanged frozen-v3.2 protocol. Its matched
three-seed audit averaged75.3591 versus73.1322 for the step-weighted XGBoost reference. The full-data
fit uses2048 training rows and seed2026. The validation scores describe the training procedure,
which refits inside each fold, rather than a held-out test of this all-data binary.
See [model selection evidence](../../analysis/method_expansion/delivery/README.md).

## Files

- `model.pkl`: serialized fitted `sdm.models.method_study.MethodStudy` object,2448929 bytes.
- `config.toml`: exact original training configuration.
- `meta.json`: exact original CLI training metadata, including dataset checksum and clean source commit
  `076b75f8f9a4335e3d5202af0117feba22e77264`. Historical absolute paths are provenance only.
- `metrics.json`: original full-data fit diagnostics and learned ensemble coefficients.
- `verification.json`: package load and prediction agreement against all1503 archived comparison inputs.
- `manifest.json`: package versions and SHA256 checksums for every other file in this bundle.

The fitted ensemble coefficients are XGBoost .2920246172, CatBoost .4437867722 and HistGB .2641886106.
They are internal model coefficients, distinct from the21 controller objective weights.

## Use after cloning the azq branch

The pickle references repository Python classes and pinned libraries. Run from the repository root
with Python3.11 and the locked environment; the original ignored `runs/` directory is not required.

```bash
uv sync --locked --group gui
env -u PYTHONPATH uv run --locked --group gui sdm predict \
  --model-run artifacts/best_model \
  --weights analysis/common_judge/delivery/exported_weights/primary/blend-fixed.json
```

This uses the fixed-blend generator's original seed2026/rank1 controller candidate, whose predicted
R under this model is109.3303 (lower is better). Its weights are
[already tracked here](../../analysis/common_judge/delivery/exported_weights/primary/blend-fixed.json).
For the course baseline W, use `--weights weights.json`; this model predicts118.7316 for that W.
No simulator measurement is claimed for either point.

The package was verified through the existing `sdm predict` CLI against the complete1503-input
comparison batch; both predicted mean and error proxy match the archived outputs within1e-9.
Use the repository CLI with this run directory so no custom loading code is necessary.

The controller comparison and its limitations are in
[the fixed-judge report](../../analysis/common_judge/delivery/README.md).
