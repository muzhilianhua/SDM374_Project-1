# Offline submission 2: candidate selection

Recommendation for the second offline submission, chosen without the solver (decision 23). Known real
results: training row `p01969` 115.7648 (the bar, decision 17), baseline W\* 119.0916, offline
submission 1 119.71206678869203. Nothing here is a measured score; every number below is a surrogate
prediction.

**Recommended W: [`delivery/sign/max-p.json`](delivery/sign/max-p.json)**, a step of 0.08 from `p01969`
along consensus direction 14840, chosen by the sign study below to maximize the calibrated probability of
beating `p01969` (0.89; 0.87 at the worst point of a radius-0.01 ball). It passes `check_weights`
(minimum weight 0.038). It supersedes the first recommendation, `delivery/consensus-d0.05.json` (same
direction, calibrated probability 0.88). The submission itself and its real R are recorded in
`docs/decisions.md` when made.

Reproduce from the repository root (about 4 minutes; all evidence below comes from commit `e4fa38f`,
clean):

```bash
uv run python analysis/offline_submission2/study.py
uv run sdm evaluate -c experiments/local-gp-v32.toml
```

## Judges

Every judge's "gain" is its prediction minus its own prediction for `p01969`; negative is better. This
cancels part of each model's level bias.

| Judge | Plugin and settings | Why it is trusted, and how far |
|---|---|---|
| Additive ARD GP (`ard`) | `bayes-search`, `additive-ard`, as `experiments/bayes-study-gp-ard.toml` | Best v3.2, 83.15 (`20261102-195112-evaluate-bayes-study-gp-ard`). Right sign at both real points: baseline +5.23 (real +3.33), submission 1 +5.70 (real +3.95). Where it predicts below 120, the real value is 1.49 higher on average (`delivery/ard_oof_calibration.csv`). |
| Local GP (`lgp`) | `local-gp`, new in this study: Matern 5/2 ARD over all 21 weights, fitted on the 800 rows nearest the best row | Different structure from `ard` (no additive-by-building assumption). Predicted submission 1 at 120.12 (real 119.71). Poor global ranker: v3.2 64.50 (`20261105-145418-evaluate-local-gp-v32`) because rows far from the anchor fall back to its local mean; std_coverage 0.70. Where it predicts below 120, the real value is 1.14 higher on average. |
| ExtraTrees (`et`) | `extratrees` defaults | Veto only. Right sign at both real points but overstates harm by 1–2.7, and cannot represent the `p01969` dip (predicts 121.17 there). A candidate is ineligible when its gain exceeds +4.9, its gain at submission 1. |
| Variants | `*_real`: fitted with the two real points added; `*_loo`: fitted without `p01969`; `lgp400` / `lgp1200`: 400 / 1200 neighbours | Not used to search, so they check the search's choices out of sample. |

Boosted trees (stack, histgb, xgboost) are not judges: they predicted 107–111 for submission 1, which
measured 119.71 (decisions.md Findings).

## Check 1: the PR #9 line searches (`delivery/lines.csv`)

The research lead in `analysis/bayesian_search/delivery/current/` is the r06 line at distance 0.08
(ARD gain −2.63). The local GP disagrees: +1.73 there, and positive along the whole r06 line. The r10
line is flat under the local GP (−0.16 to +0.66). The r05 line is the only one both GPs like (local GP
−0.84 at 0.07), but its direction has cosine 0.79 with the step from `p01969` to submission 1, which
measured 3.95 worse. Adding the two real points as labels changes no line gain by more than 0.06: they lie
0.58 and 0.91 away.

## Check 2: consensus directions (`delivery/consensus_top.csv`)

20,000 random directions that keep each building's weights summing to 1; 19,483 remain after removing
directions with cosine ≥ 0.5 to either measured failure. Steps of 0.03, 0.05 and 0.07 are scored by
`ard`, `lgp` and `et`, ranked by the worse of the two GP gains, with the ExtraTrees veto. Only 18–23 % of
eligible points have both GP gains negative. Direction 14840 ranks first at every step:

| Step | ARD gain | Local GP gain | ExtraTrees gain | Cosine to submission 1 / baseline |
|---:|---:|---:|---:|---|
| 0.03 | −1.37 | −1.12 | +1.06 | 0.37 / 0.15 |
| 0.05 | −2.15 | −1.70 | +1.91 | 0.37 / 0.15 |
| 0.07 | −2.83 | −2.16 | +3.44 | 0.37 / 0.15 |

## Check 3: robustness of the finalists (`delivery/robustness.csv`)

`smooth_worst` is the worst gain over six smooth judges (`ard`, `ard_loo`, `lgp400`, `lgp`, `lgp1200`,
`lgp_loo`) and over 64 points at radius 0.01 around each finalist.

| Finalist | ARD | Local GP 400 / 800 / 1200 | Smooth worst | ExtraTrees |
|---|---:|---|---:|---:|
| consensus d0.03 | −1.37 | −1.31 / −1.12 / −1.11 | −0.83 | +1.06 |
| **consensus d0.05** | **−2.15** | **−1.98 / −1.70 / −1.67** | **−1.42** | **+1.91** |
| consensus d0.07 | −2.83 | −2.48 / −2.16 / −2.11 | −1.86 | +3.44 |
| r05 d0.07 | −2.03 | −0.46 / −0.84 / −1.11 | +0.14 | +3.41 |
| r06 d0.08 (PR #9 lead) | −2.63 | +2.56 / +1.73 / +2.11 | +3.20 | +4.51 |
| r10 d0.05 | −1.57 | +0.16 / −0.13 / −0.02 | +0.59 | +2.45 |

## Recommendation and its limits

First-pass recommendation, superseded by the sign study below.

Consensus d0.05 is the only kind of finalist that every smooth judge and variant, including the
worst point of its 0.01 ball, predicts better than `p01969`, with the smallest ExtraTrees penalty
among those. d0.07 promises more but its ExtraTrees penalty (+3.44) approaches the veto; d0.03 is the
safest step but its smooth-worst gain (−0.83) is small next to the judges' error.

Limits, in order of importance:

- All judges are fitted on the same 2048 rows, and no row lies within 0.30 of `p01969`. Agreement among
  them is not independent evidence.
- Direction 14840 is the best of 19,483 under `ard` and `lgp`, so its gains are biased toward optimism.
  The held-out variants reduce but do not remove this.
- Neither GP has been validated below 117: where they predict low, the real value was 1.1–2.3 higher.
  With that offset the ARD prediction for consensus d0.05, 113.64 ± 1.44, becomes about 115.1–115.9
  against the bar of 115.76. A rough probability of passing is 45–65 %.

## Sign study: maximizing the probability of passing (`delivery/sign/`)

Passing needs only R(w) < R(`p01969`), so the target is P(gain < 0), not the size of the gain.
`sign_study.py` estimates it and picks the step that maximizes it.

```bash
uv run python analysis/offline_submission2/sign_study.py                    # about 17 minutes
uv run python analysis/offline_submission2/sign_study.py --reuse-backtest   # scan only, about 1 minute
```

The backtest pairs come from the full run at commit `6e3954d`; the scan and finalists from
`--reuse-backtest` at `1f005ff`. The backtest code did not change in between. Both commits are clean.

**Joint-posterior z-score.** For each judge, z = −gain / √(SD(gain)² + noise²). SD(gain) is the latent
posterior SD of f(w) − f(anchor), including their covariance; noise is the judge's fitted white noise,
standing for roughness the data cannot resolve. Positive z means the judge expects w to be better.

**Backtest (`backtest_pairs.csv.gz`).**
- Setup: in each of the 10 frozen outer splits, the 12 lowest training rows that have test rows within
  0.45 act as anchors. The judges are fitted on that split's training rows: ARD on all of them, the local
  GP on the 800 nearest each anchor. This gives 4,256 anchor/test pairs at distances 0.21–0.45; the data
  has no pairs closer than 0.165.
- Base rate: only 2.2 % of pairs improve on their anchor.
- Calibration: out-of-split logistic calibration of improvement on (z_ard, z_lgp) is reliable. Predicted
  0.25 / 0.39 / 0.60 / 0.83 against observed 0.29 / 0.35 / 0.64 / 0.73 (`reliability.csv`). Raw Φ(z_ard)
  is almost as well calibrated: among the 23 pairs with z_ard > 1, 91 % improved against 92 % predicted.
- Agreement (`agreement.csv`):

  | Both judges say better (z > 0.3) | One says better, the other worse |
  |---|---|
  | 38 pairs, 82 % improved | 41 pairs, about 36 % improved |

- Adding ExtraTrees gain or distance to the calibration does not improve its out-of-split log loss
  (0.0345 → 0.0344 / 0.0345). The +4.9 ExtraTrees veto is kept only as the guard learned from
  submission 1.
- Selection check (`selection_backtest.csv`): when each anchor submits its neighbour with the highest
  calibrated probability, the observed rate matches the prediction (observed 0.45 vs predicted 0.45;
  observed 0.77 vs predicted 0.75 for picks above 0.5).

**Scan (`scan_top.csv`, `scan_by_step.csv`, `finalists.csv`).**
- Scope: the 19,483 directions of check 2, at steps 0.01 to 0.10.
- Calibrated probability by step: it rises from 0.76 at 0.01 to about 0.90 at 0.04, then stays flat
  through 0.10.
- Finalist rule: both judges must say better (z > 0.3), and the whole radius-0.01 ball must stay inside
  the ExtraTrees veto. Finalists are then ranked by their worst calibrated probability on that ball.

| Finalist | Calibrated P | Ball worst | z ARD / local GP | ExtraTrees gain (ball max) |
|---|---:|---:|---|---:|
| **direction 14840, step 0.08** | **0.890** | **0.873** | 1.57 / 0.88 | 3.93 (4.25) |
| direction 15713, step 0.05 | 0.896 | 0.870 | 1.90 / 0.34 | 2.39 (2.91) |
| direction 14840, step 0.07 | 0.889 | 0.868 | 1.58 / 0.84 | 3.44 (4.14) |
| direction 14840, step 0.05 | 0.881 | 0.849 | 1.58 / 0.71 | 1.91 (2.78) |
| direction 14840, step 0.10 | 0.889 | 0.875 | 1.54 / 0.92 | 4.47 (5.10, vetoed) |

**What the probability means.** About 0.89 is the most the current information supports. The finalists
differ by less than the calibration's own uncertainty: its top bin holds 41 pairs and was overconfident
(0.83 predicted, 0.73 observed). A realistic reading is 0.75–0.89.

Remaining risks:
- The backtest measures sign skill at 0.21–0.45 and applies it to a step of 0.08.
- `p01969` is the global best row, so it is a stricter anchor than most backtest anchors.
- The direction is the best of many.
