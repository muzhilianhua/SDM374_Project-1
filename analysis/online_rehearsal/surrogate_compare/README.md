# Offline comparison of online surrogates (T37, 2026-11-17)

Owner: member A (张子木). Developed outside the repository (`Project-1/surrogate_compare/`) and copied here on
2026-11-18. Only the paths in `compare.py` and the default run list changed, and `summarise.py` reproduces
`summary.txt` byte for byte from the copied `preds.csv`.

**Question.** Is there a better replacement for the surrogate that T37 v4 uses: a ridge trend plus a local
600-row ARD-RBF GP whose hyperparameters are fitted at run time? Decision 31 requires constant
hyperparameters, so the replacement should preferably meet that rule too.

**Method.** `compare.py` replays the runs call by call. It uses the 16 v1–v6 rehearsal runs in `../rehearsals/`.
For every call k from 2 to 20, each model is fitted on exactly the data the policy had: the 2048 training
rows plus the run's points from calls before k, centred on the best point so far. The model then predicts
the points the run measured in call k. In total this covers 304 run-calls and 15,060 points per model. No
solver call is made.

The main table uses only the **points that the model did not select** (9,768 of them). T37 chose its model
picks by its own lower confidence bound, which would flatter T37. `summarise.py` writes `summary.txt`.
`paired.txt` holds the paired bootstrap and the kappa sensitivity; it was computed interactively from
`preds.csv`, and its script was not kept.

The compared models use only the training rows and the run's own data. The `hp_noise001` constants were
chosen by PR #24 from the training table.

## Candidates

| Name | What it is |
|---|---|
| `t37` | T37 as used in v1–v6; hyperparameters fitted at every call |
| `t37_frozen` | T37 structure, kernel hyperparameters fitted once on the training rows and then frozen (`frozen_kernel.txt`) |
| `azq_fixed` | `submission/surrogate.py` of `origin/azq`: RBF length 1.0, noise 0.1, GPR alpha 1.0 |
| `hp_global` | PR #24 `hp_noise001`: block-additive ARD Matérn-5/2 plus a 0.1 global term, all constants fixed, constant mean, fitted on all data (training rows plus online points) |
| `trend_hp` | T37's ridge trend with the `hp_noise001` kernel on the residuals of the local 600 rows |

## Result (points not selected by the model, mean over 304 run-calls)

Column meanings:
- **Bias**: R − prediction. A positive value means the model is optimistic.
- **z SD**: the SD of (R − prediction) / std. A value of 1 means the std is calibrated; below 1 means the std is too large.
- **Spearman**: the rank correlation between predicted and measured R within one call.
- **Top-5 lift**: the mean measured R of the 5 points with the lowest predicted mean, minus the call's mean R.
- **Top-5 hits**: the share of those 5 points that fall below the best R so far. Over all points the rate is 0.227.
- **Top-5 regret**: the lowest R among those 5 points, minus the call's lowest R.

| Model | Bias | MAE | z SD | Spearman | Top-5 lift | Top-5 hits | Top-5 regret | Fit time per call |
|---|---|---|---|---|---|---|---|---|
| t37 | −0.92 | 2.41 | 1.81 | 0.379 | −0.65 | 0.241 | 0.27 | about 50 s |
| t37_frozen | −1.63 | 3.05 | 0.94 | 0.213 | −0.44 | 0.182 | 0.34 | < 0.1 s |
| azq_fixed | −4.03 | 5.50 | 0.74 | 0.123 | −0.22 | 0.186 | 0.40 | < 0.1 s |
| **hp_global** | −0.81 | **1.41** | 0.65 | **0.517** | **−1.07** | **0.256** | **0.20** | about 3 s |
| trend_hp | −1.05 | 3.15 | 2.91 | 0.249 | −0.57 | 0.211 | 0.31 | < 0.1 s |

Paired difference `hp_global` − `t37` per run-call, bootstrap over the 16 runs:

| Metric | Difference | 95 % CI | Runs where `hp_global` is better |
|---|---|---|---|
| Spearman | +0.138 | [+0.056, +0.211] | 13/16 |
| Top-5 lift | −0.419 | [−0.638, −0.190] | 14/16 |
| Top-5 regret | −0.071 | [−0.135, −0.008] | 10/16 |
| Top-5 hit rate | +0.015 | [−0.015, +0.049] | 7/16, not significant |

By phase, Spearman for `hp_global` against `t37`:
- calls 2–6: 0.57 against 0.34;
- calls 7–13: 0.54 against 0.41;
- calls 14–20: 0.46 against 0.38. In this phase `t37` has a slightly higher top-5 hit rate (0.120 against 0.095), and both are close to the base rate (0.102).

## Kappa of the lower confidence bound (rank by prediction − kappa · std)

| Model | kappa = 0 | 0.25 | 0.5 | 1 (v4) |
|---|---|---|---|---|
| t37 top-5 lift / hits | −0.65 / 0.241 | −0.42 / 0.214 | −0.24 / 0.188 | +0.09 / 0.153 |
| hp_global top-5 lift / hits | −1.07 / 0.256 | −1.03 / 0.238 | −0.97 / 0.227 | −0.80 / 0.179 |

Offline, kappa = 0 (rank by the mean only) is best for every model. With v4's kappa = 1, `t37`'s picks are worse
than the call's average, which may be one reason why v4's model picks hit only about 3 % of the time.

## Offline conclusions

1. `hp_global` ranks measured points better, and it already meets decision 31: its hyperparameters are
   constants, so a fit is one Cholesky factorisation. It is significantly better than T37 as is, and better
   still than the two frozen variants. A likely reason is that the block-additive kernel matches the
   one-building candidates: the other two buildings did not change, so the kernel can borrow the base's
   measured value.
2. Freezing T37's hyperparameters loses much of its ranking ability (Spearman 0.38 → 0.21, or 0.12 with
   `azq`'s constants). A frozen policy should change the kernel rather than freeze T37's.
3. A ridge trend plus Matérn residuals (`trend_hp`) is poorly calibrated (z SD 2.9) and ranks worse than
   `hp_global`.

## Real-solver check (2026-11-18): the offline gain does not carry over

`../policy_v7.py` (v4 + `hp_global` + kappa 0) and `../policy_v8.py` (v4 + `hp_global` + kappa 1) were run
on the real solver, three seeds each. See `../README.md`, section v7.

| Variant | Seed 2026 | Seed 7 | Seed 11 | Mean |
|---|---|---|---|---|
| v4 (T37 model, kappa 1) | 108.049 | 103.275 | 105.440 | 105.59 |
| v7 (`hp_global`, kappa 0) | 107.687 | 106.574 | 107.428 | 107.23 |
| v8 (`hp_global`, kappa 1) | 106.433 | 105.076 | 107.153 | 106.22 |

The better offline ranking did not lower the final R. Offline conclusion 3 on kappa does not hold in real
runs: most of v7's loss came from kappa 0. Ranking points that were already measured cannot show the value
of the exploration term. What this study measures, ranking of measured points, is not what limits the
final R.

## Limits

- Only measured points are evaluated, and the v1–v6 moves generated them. The policy ranks about 1,500
  candidates per building and call, and the unmeasured ones cannot be scored here.
- The difference in top-5 hit rate is not significant.
- `hp_global` factorises 2,000–3,000 rows per fit, about 3 s. Its cross-machine reproducibility should be
  better than an L-BFGS fit's, but it still needs a `verify` check.

## Files

| File | Content |
|---|---|
| `compare.py` | call-by-call replay; writes `preds.csv` (one row per run, call, point and model) and `frozen_kernel.txt` |
| `summarise.py`, `summary.txt` | accuracy, calibration and within-call selection metrics by model and phase |
| `paired.txt` | kappa table and paired bootstrap (computed interactively from `preds.csv`) |
| `run.log` | console output of the `compare.py` run (2026-11-17) |
