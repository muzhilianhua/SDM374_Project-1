# Offline submission 2: candidate selection

Recommendation for the second offline submission, chosen without the solver (decision 23). Known real
results: training row `p01969` 115.7648 (the bar, decision 17), baseline W\* 119.0916, offline
submission 1 119.71206678869203. Nothing here is a measured score; every number below is a surrogate
prediction.

**Recommended W: [`delivery/sign/max-p.json`](delivery/sign/max-p.json)**, a step of 0.05 from `p01969`
along consensus direction 14840. It is chosen by the sign study's final rule: steps of at most 0.05, both
GP judges clearly saying "better" (z > 0.5), ranked by the worst calibrated probability on a radius-0.01
ball. The file is identical to `delivery/consensus-d0.05.json` and passes `check_weights` (minimum weight
0.035). Its probability is not independently validated (see "What the probability means"); a defensible
reading is 0.6–0.8. The submission itself and its real R are recorded in `docs/decisions.md` when made.

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
| ExtraTrees (`et`) | `extratrees` defaults | Reported only. It cannot represent the `p01969` dip (predicts 121.17 there) and predicts every step from it as worse (0 % of 19,483 directions at any step, median gain +0.64 at 0.01 rising to +4.96 at 0.08; `delivery/sign/scan_by_step.csv`), so its gain measures step length, not improvement. Its "right sign" at both real points carries no information because both were worse than `p01969`. Checks 1–3 below used its +4.9 gain at submission 1 as a veto; the final rule replaces that with an explicit step cap. |
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

The backtest pairs come from the full run at commit `6e3954d`; calibration, scan and finalists from
`--reuse-backtest` at `4e06606`. The backtest code did not change in between. Both commits are clean.

**Joint-posterior z-score.** For each judge, z = −gain / √(SD(gain)² + noise²). SD(gain) is the latent
posterior SD of f(w) − f(anchor), including their covariance; noise is the judge's fitted white noise,
standing for roughness the data cannot resolve. Positive z means the judge expects w to be better.

**Backtest (`backtest_pairs.csv.gz`).**
- Setup: in each of the 10 frozen outer splits, the 12 lowest training rows that have test rows within
  0.45 act as anchors. The judges are fitted on that split's training rows: ARD on all of them, the local
  GP on the 800 nearest each anchor. This gives 4,256 anchor/test pairs at distances 0.21–0.45; the data
  has no pairs closer than 0.165.
- Base rate: only 2.2 % of pairs improve on their anchor.
- Holdout: the two CV repeats re-create most pairs (3,148 of 4,256 records have the same pair in another
  split), so a plain leave-one-split-out would reuse labels (PR #11 review). Each calibration fold therefore
  holds out one outer split and drops every training record that involves one of its candidates, as
  candidate or anchor. Anchor labels stay shared: anchors are known reference rows, as `p01969` is for the
  real step, and purging them too leaves no improving examples. The z features are not nested (each split's
  judges saw the other splits' test rows), so leakage is reduced, not removed.
- Calibration (`reliability.csv`), purged: predicted 0.25 / 0.38 / 0.61 / 0.83 against observed
  0.35 / 0.40 / 0.61 / 0.63. Before the purge the top bin read 0.73 observed. The top bin is overconfident.
- Agreement (`agreement.csv`, unique pairs):

  | Both judges say better (z > 0.3) | One says better, the other worse |
  |---|---|
  | 28 pairs, 79 % improved | 24 pairs, about 29 % improved |

  The pairs closest to the recommended point (z_ard > 1.3 and z_lgp > 0.6, unique) improved 9 of 11 times.

- Adding ExtraTrees gain or distance to the calibration did not improve its split-holdout log loss
  (0.0345 → 0.0344 / 0.0345, before the purge). ExtraTrees predicted an improvement for only 2 of 2,682
  unique pairs, both wrong (`extratrees_backtest.csv`). Of the 28 agreeing pairs, the 19 its +4.9 veto would have rejected improved
  74 % of the time.
- Selection check (`selection_backtest.csv`, purged): when each anchor submits its neighbour with the
  highest calibrated probability, 0.45 is observed against 0.44 predicted overall, and 0.69 against 0.74
  for picks above 0.5.

**Scan (`scan_top.csv`, `scan_by_step.csv`, `finalists.csv`).**
- Scope: the 19,483 directions of check 2, at steps 0.01 to 0.10.
- Calibrated probability by step (fitted on unique pairs): it rises from 0.72 at 0.01 to about 0.86 at
  0.04, then stays flat through 0.10.
- Finalist rule (commit `ab6cbf0`, outputs regenerated at `4e06606`):
  - **Step at most 0.05.** The calibrated probability stops rising at 0.04, so a longer step buys no
    supported probability but adds risks the models cannot see: unresolved roughness below 0.165, the
    closest pair spacing in the data, and selection among more directions. The GPs' measured failures
    (submission 1 at 0.58; V3.3.2 region-holdout picks 0.31–0.39 from any training row) lie far beyond 0.05,
    where the prior correlation with `p01969` is still 0.99.
  - **Both judges clearly agree: z_ard > 0.5 and z_lgp > 0.5.** Among unique backtest pairs, those where
    only one judge was clearly confident improved about 67 % of the time, against 79 % when both were.
  - **Ranking:** eligible finalists are ranked by their worst calibrated probability on the radius-0.01 ball.
  - **ExtraTrees** gain is reported, not used.

| Finalist | Eligible | Calibrated P | Ball worst | z ARD / local GP | ExtraTrees gain |
|---|---|---:|---:|---|---:|
| **direction 14840, step 0.05** | **yes** | **0.844** | **0.808** | 1.58 / 0.71 | 1.91 |
| direction 7986, step 0.05 | yes | 0.837 | 0.800 | 1.63 / 0.51 | 2.70 |
| direction 14840, step 0.04 | yes | 0.833 | 0.786 | 1.55 / 0.62 | 1.49 |
| direction 15713, step 0.05 | no (z_lgp 0.34) | 0.861 | 0.831 | 1.90 / 0.34 | 2.39 |
| direction 14840, step 0.08 | no (step) | 0.856 | 0.835 | 1.57 / 0.88 | 3.93 |

  The previous rule (z > 0.3 and the ExtraTrees veto) chose direction 14840 at step 0.08. The two choices
  differ by 0.03 in ball-worst probability, well inside the calibration's uncertainty.

**What the probability means.** The calibrated values above are model outputs, not validated success
probabilities. All finalists fall in the top calibration bin, where the purged backtest observed 0.63
(38 pairs, standard error about 0.08) against 0.83 predicted. The backtest pairs most like the recommended
point, where both judges are confident, did better: 79 % of the 28 unique agreeing pairs and 9 of 11 of
the closest ones improved. A defensible reading for the recommended W is 0.6–0.8. It is not a validated
number: the features are not nested, and the best-row case below is barely tested. The finalists differ
by less than this uncertainty, so the choice among them rests on agreement and the ball check, not on
the probability gaps.

Remaining risks:
- The backtest measures sign skill at 0.21–0.45 and applies it to a step of 0.05.
- `p01969` is the global best row, so it is a stricter anchor than most backtest anchors. Each split's best
  anchor contributed 316 pairs but only one pair where both judges were confident.
- The direction is the best of many.

## V3.3.2 check of the judges (`delivery/v332/`)

`v332_judges.py` runs the ARD GP and the local GP on the frozen V3.3.2 protocol
(`analysis/v33_protocol/release_v2/`). It uses the same 14 tasks, P1 holdout, feedback stage and scorer
that PR #12 used for the three boosted-tree recipes, and does not modify the release. It ran at clean
commit `890bf5f`.

```bash
uv run python analysis/offline_submission2/v332_judges.py   # about 17 minutes, 60 fits
```

| Model (primary stage) | Score /100 | Hidden elite | Region holdout | Supported low | True-low MAE | Hidden-elite MAE | Full-fit P1 (real 119.71) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARD GP | 59.96 | **73.06** | 38.82 | **63.98** | **3.59** | **5.17** | **121.49** |
| Local GP | 54.36 | 68.12 | 33.41 | 57.95 | 6.13 | 9.80 | 120.12 |
| low60 HistGB (PR #12) | **60.51** | 68.19 | **51.54** | 61.16 | 6.56 | 7.04 | 115.23 |
| low80 HistGB (PR #12) | 55.28 | 56.56 | 51.67 | 56.45 | 6.48 | 6.85 | 115.15 |
| low90 XGBoost (PR #12) | 52.28 | 54.71 | 39.71 | 57.34 | 7.15 | 8.73 | 113.04 |

What it says about the submission-2 judges:

- **Overall.** ARD is level with the best boosted recipe overall and is by far the most accurate on low R:
  - true-low MAE 3.59 against 6.5–7.2;
  - its full-data P1 prediction misses by +1.78, where the boosted recipes miss by −4.5 to −6.7.
- **Hidden-elite tasks** (top 5, 10 or 20 rows moved to the test set). Every model ranks `p01969` first,
  so these tasks do not separate the judges on the question that matters here.
- **Region-holdout tasks** (`p01969`'s neighbourhood removed from training). Both GPs fail all 3 with
  false breakthroughs:
  - their picks were predicted 111.3–118.1, against a training best of about 120.5, and measured 120.6–124.6;
  - the boosted recipes broke through in all 3, with conservative predictions.

  Where no training row is near, the GPs' low predictions can be several points too optimistic. This is
  the main new warning.
- **Supported tasks with `p01969` in training** (f02–f05). Neither GP predicted any test row below the
  known best 115.76 (ARD picks 118.8–119.2). Where the anchor is known, they did not make false claims of
  beating it. These test rows are at least 0.3 from `p01969`, so this is weaker evidence than a short
  step would need.

For the recommended step this is mixed evidence.
- **Against:** the region-holdout failures show the GPs extrapolating too optimistically into data-sparse
  space, and the step leaves `p01969` into space with no row within 0.3.
- **For:** in the real case the anchor `p01969` is known; the GPs measure the gain from it; and the
  supported tasks show no false claims of beating a known best.
- **Distance check:** the failing picks lay 0.31–0.39 from any training row, while the recommended step
  is 0.05 from `p01969`. The warning therefore rules out long steps, not the 0.05 step.

The PR #12 boosted recipes, refitted on all rows, give the recommended direction mixed gains:

| Recipe | d0.05 | d0.08 | Gain on P1 (real +3.95) |
|---|---:|---:|---:|
| low60 | −0.51 | −0.96 | −0.34 |
| low80 | +0.87 | +0.80 | −1.41 |
| low90 | +1.20 | +1.96 | −4.94 |

All three judged P1 wrongly, so their sign is not used.
