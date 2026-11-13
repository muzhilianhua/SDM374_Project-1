# Online surrogate model bundle: hp_noise001

Candidate surrogate model for the online phase (T12, online model selection). `hp_noise001` is a
hyper-parameter variant of the offline `gp-ard` baseline: a **block-wise ARD Matérn-5/2 Gaussian
process with a global interaction term and white noise**, where the noise level is fixed at
`0.01` and the length-scale upper bound is relaxed to `6`.

This package is one candidate input to T12 (刘润泽). The model plugin registration (T13 refit on the
online rows) is B's task (`src/sdm/models/`, AGENTS.md §3). See `offline_search_provenance.json` for
how this variant was selected among six offline candidates.

## 1. What the model is

| Item | Value |
| --- | --- |
| Type | Gaussian process regression (fitted with sklearn GPR; inference needs only numpy + scipy) |
| Kernel | `BlockARD(Matern-5/2, ARD, 3 blocks) + global interaction + WhiteKernel`, ν = 2.5 |
| Length scale | 7 ARD dimensions (`cost / carbon / peak / discomfort / degradation / ramping / resilience`), bounds [0.03, 6] |
| Interaction weight | `global_weight = 0.1` (fixed) |
| Noise | `noise = 0.01` (fixed, not optimised) |
| Training data | `dataset_same8/train.csv`, 2048 rows; frozen split `sdm374.same8.splits.2x5x3.frozen.v1` |
| Target | `ramping`, raw scale |
| Seed | 2026 |

Fitted parameters: `length_scale = [0.289, 0.248, 1.202, 6.0, 0.187, 0.191, 6.0]`, `loc = 160.45`,
`scale = 15.77`. The `discomfort` and `resilience` length scales sit on the upper bound, i.e. those
two metrics affect R very smoothly across the simplex.

## 2. Metrics

| Dataset | n | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| OOF (frozen 2x5x3 splits, `metrics.json`) | 4096 | 5.230 | 3.340 | 0.890 |
| **In-sample** (2048 fitted rows, optimistic — not OOF) | 2048 | 0.578 | 0.403 | 0.999 |
| Local solver log `eval_log`, subset R < 180 | 6297 | 11.33 | 9.20 | 0.804 |
| All real solver rows minus train, subset R < 180 | 1257 | 2.88 | 2.08 | 0.980 |

Two independent real datasets (the local solver `eval_log.jsonl` and the merged "all real data" CSV,
both R < 180) agree on 4 of 6 rank positions for RMSE; `blend_xgb_w20x3` and `blend_histgb_low80` are
swapped. Cross-dataset Spearman rank correlation = **ρ = 0.94**. The detailed rankings with source
hashes are in `offline_search_provenance.json` § Real-data ranking; the per-dataset CSVs are in
`rank_Rlt180/`.

## 3. Input contract — normalised weights required

Every building's 7 weights **must be non-negative and sum to 1** (Decision 21 of `docs/decisions.md`).
`ModelBundle.predict` enforces this by calling `_check_simplex` on every query; any input that fails
raises a `ContractError` describing the offending rows and buildings.

Rationale: the GP kernel is **not scale-invariant**. Multiplying building 1's weights by 3 while
leaving the other two untouched represents the same normalised point but produces a different
prediction — rescaling without renormalising is silently wrong. If you have unnormalised weights,
normalise each building's block before calling `predict`.

## 4. Environment

- Inference matches the repository online environment (`pyproject.toml`): **Python 3.11, numpy 1.24.4,
  scipy 1.10.1**. Use the repository `.venv` or `uv run python`.
- The original offline fit ran under Python 3.13.5 / numpy 2.1.3 / scipy 1.15.3 / scikit-learn 1.6.1
  (see `hp_noise001/meta.json`). We commit `.npz` (plain dict-of-arrays, `allow_pickle=False`)
  specifically to avoid the `numpy._core` versus `numpy.core` pickle incompatibility that would
  otherwise break loading under numpy 1.x.

## 5. Usage

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("analysis/online_models")))
from modellib import load_model  # numpy-only inference kernel

bundle = load_model("analysis/online_models/hp_noise001")
mean, std = bundle.predict(W)        # W: (n, 21), normalised (each building sums to 1, >= 0)
```

## 6. Reproduction — verify the bundle

```bash
uv run python analysis/online_models/verify_package.py
```

Checks every sha256 in `hp_noise001/CHECKSUMS.json` and recomputes the in-sample metrics against
`metrics.json` (RMSE 0.578354, MAE 0.403200). Expected: `PASS`.

## 7. Reproduction — rebuild the model from scratch (untrained spec)

`hp_noise001_untrained/train_config.json` freezes every ingredient of the model and contains **no
weights**: the 7 length scales, the global interaction weight, noise `0.01`, jitter `1e-8`, unit
sample weights, the target scaler rule (`loc = mean(y)`, `scale = std(y, ddof=0)`), the dataset pin
(`dataset_same8/train.csv`, 2048 rows, sha256 `877b5f8e...`, frozen split
`sdm374.same8.splits.2x5x3.frozen.v1`) and the environment pins.

With all parameters fixed there is nothing to optimise, so training is a single Cholesky solve:

```bash
uv run python analysis/online_models/train_hp_noise001.py
uv run python analysis/online_models/train_hp_noise001.py --write out/model.npz  # optional dump
```

The script verifies the dataset sha256 and row count, re-derives `loc`/`scale`, rebuilds `alpha` from
the fixed kernel, and compares against the committed bundle. Expected reproduction diffs: `alpha`
max abs diff `6.9e-12`, training-prediction max abs diff `4.1e-12`. No RNG is used; `seed = 2026` only
records the offline multi-start search.

## 8. Known limitations

- Training labels lie in R ∈ [110, 243]. For far points with R ≥ 180 (about 4% of `eval_log`) all six
  offline candidates have RMSE above 25 and R² < 0 — this is an out-of-distribution region, do not use
  this model to rank points there.
- The GP predictive std is optimistic: in the largest-std bin `mean |resid| / mean_std ≈ 1.4–1.9`, so
  inflate it when std drives the exploration budget.
- This bundle fits the frozen 2048-row training set only. Refitting on online rows after every solver
  call (T13) requires a `src/sdm/models/` plugin; see the discussion at PR #24.

## 9. Provenance

- Offline workspace: `外推方案/模型库/hp_noise001` (an untracked local directory that was never
  committed).
- Training script: the `hp_noise001` config in `外推方案/gp提分/run_variant.py`; per-fold summary:
  `外推方案/gp提分/results/summary_hp_noise001.json`.
- Full selection evidence (offline OOF scores, v3.2 scores, the six-candidate shortlist and the
  real-data ranking with source hashes): `offline_search_provenance.json`.
- Per-dataset rank tables: `rank_Rlt180/`.
