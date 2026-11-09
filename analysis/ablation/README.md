# Evaluation and ablation tables

This folder contains the report-facing table builder. It reads completed `sdm evaluate` run folders and writes a CSV, Markdown table, and provenance record. Every row retains its run id.

From the repository root:

```bash
python analysis/ablation/build_table.py
```

The default output is `analysis/ablation/delivery/`, which is a generated artifact. Do not rank `frozen-v32` and `kfold` rows against one another: the protocols use different splits and metrics.

The baseline evidence used in the current report is:

- `20261109-222524-evaluate-baseline-v32`: frozen-v3.2 score 66.4508, top-1 regret 4.5095.
- `20261109-222557-evaluate-baseline`: five-fold baseline, MAE 6.8275 and top-50 recall 0.46.
