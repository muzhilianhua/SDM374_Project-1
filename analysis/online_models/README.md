# Online surrogate model: hp_noise001

Candidate surrogate model for the online phase (T12, online model selection). `hp_noise001` is a
hyper-parameter variant of the offline `gp-ard` baseline: a **block-wise ARD Matern-5/2 Gaussian process
with a global interaction term and white noise**, where the noise level is fixed at `0.01` and the length
scale upper bound is relaxed to `6`.

## 1. What the model is

| Item | Value |
| --- | --- |
| Type | Gaussian process regression (fitted with sklearn GPR; inference needs only numpy + scipy) |
| Kernel | `BlockARD(Matern-5/2, ARD, 3 blocks) + global interaction + WhiteKernel`, nu = 2.5 |
| Length scale | 7 ARD dimensions (`cost / carbon / peak / discomfort / degradation / ramping / resilience`), bounds [0.03, 6] |
| Interaction weight | `global_weight = 0.1` (fixed) |
| Noise | `noise = 0.01` (fixed, not optimised) |
| Training data | `dataset_same8/train.csv`, 2048 rows; frozen split `sdm374.same8.splits.2x5x3.frozen.v1` |
| Target | `ramping`, raw scale |
| Seed | 2026 |

Fitted parameters: `length_scale = [0.289, 0.248, 1.202, 6.0, 0.187, 0.191, 6.0]`, `loc = 160.45`,
`scale = 15.77`. The length scales of `discomfort` and `resilience` sit on the upper bound, i.e. those two
metrics enter R very smoothly across the simplex.

## 2. Metrics

| Dataset | n | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| OOF (frozen 2x5x3 splits, `metrics.json`) | 4096 | 5.230 | 3.340 | 0.890 |
| in-sample (2048 fitted rows, optimistic) | 2048 | 0.578 | 0.403 | 0.999 |
| Local solver log `eval_log`, subset R < 180 | 6297 | 11.33 | 9.20 | 0.804 |
| All real solver rows minus train, subset R < 180 | 1257 | 2.88 | 2.08 | 0.980 |

Within the R < 180 search region `hp_noise001` has the lowest real RMSE of the six offline candidates,
and two independent real datasets rank the six models identically (Spearman rho = 0.94). The full
comparison lives in the evaluation workspace (`rank_Rlt180/`).

## 3. Environment

- Inference matches the repository online environment (`pyproject.toml`): **Python 3.11, numpy 1.24.4,
  scipy 1.10.1**. The repository `.venv` works as is; `requirements.txt` only pins those versions.
- Training ran under Python 3.13.5 / numpy 2.1.3 (see `hp_noise001/meta.json`), so the original pickle
  referenced `numpy._core` and cannot be unpickled by numpy 1.x. The committed `model.pkl` was therefore
  **re-serialised equivalently under numpy 1.24.4**: array values are unchanged and the 2048-row
  predictions differ from the original file by at most `2.8e-12`. The original file's sha256 is
  `486d97396a355ad3e19fddbf0c7d48cfc16ea5bae1edf9a69ea7a2b5f6ece49a`; the committed hash is in
  `hp_noise001/CHECKSUMS.json`.

## 4. Usage

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("analysis/online_models")))
from modellib import load_model  # numpy-only inference kernel

bundle = load_model("analysis/online_models/hp_noise001")
mean, std = bundle.predict(W)    # W: (n, 21), buildings 1->3, metrics cost->...->resilience
```

`W` is the raw 21-dim weight vector. The solver normalises each building internally; callers must ensure
each building has at least one positive weight.

## 5. Reproduction

```bash
.venv/Scripts/python analysis/online_models/verify_package.py
```

The script (1) checks every sha256 in `hp_noise001/CHECKSUMS.json`, and (2) recomputes the in-sample
metrics on the 2048 `dataset_same8/train.csv` rows, comparing against `metrics.json`
(RMSE `0.578354`, MAE `0.403200`, tolerance 1e-6).

## 6. Known limitations

- Training labels lie in R in [110, 243]. For far points with R >= 180 (about 4% of `eval_log`) all six
  candidates have RMSE above 25 and R2 < 0: that is an out-of-distribution region, do not use this model
  to rank points there.
- The GP predictive std is optimistic: in the largest-std bin `mean|resid| / mean_std` is about 1.4-1.9,
  so inflate it when std drives the exploration budget.

## 7. Provenance

Offline workspace `外推方案/模型库/hp_noise001` (an untracked local directory that was never committed).
Training script: the `hp_noise001` config in `外推方案/gp提分/run_variant.py`; per-fold summary:
`外推方案/gp提分/results/summary_hp_noise001.json`.
