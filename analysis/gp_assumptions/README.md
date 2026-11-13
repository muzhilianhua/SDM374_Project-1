# T35 — Do the GP-ARD surrogate's assumptions hold?

Owner: member A (张子木). Task: T35 in `docs/online-tasks.md`. No evaluator calls are made here.

The team's GP-ARD is `bayes-search` with `kernel = "additive-ard"`, `lengths = [0.5]`, `noises = [0.03]`
(the offline configs `bayes-search-gp-ard-*.toml` at tag `offline-final`; best v3.2 score offline,
`20261102-195112-evaluate-bayes-study-gp-ard`). It assumes:

1. the function is deterministic (only a small white-noise term);
2. R is additive over the three buildings, with the same seven length scales for every building, plus a
   weak (weight 0.1) 21-dimensional interaction;
3. Matern 5/2 smoothness;
4. stationarity: one set of length scales and one amplitude for the whole simplex;
5. a constant prior mean (the target is standardised), so far from data the prediction returns to the mean;
6. Gaussian residuals with constant variance, hence a calibrated predictive `std`.

This study tests each assumption on two data sources and then tests three changes to the model.

| Source | Use |
|---|---|
| `dataset_same8` (2048 rows) | fit on 1400 random rows, score on the other 648; 3 random splits (seeds 0–2) |
| local solver log, `analysis/replay/evidence/eval_log.jsonl` (6,568 `solver.exe` calls, 6,548 distinct points) | score only (never fitted, except in part C). Points are binned by distance ratio to the fit rows: near < 1, mid 1–1.5, far > 1.5 spacings |

The solver log was recorded during the offline exploration (`analysis/replay/README.md`, search history). Decision 27
allows local evaluator results to choose the model and its settings; nothing from the log is embedded in
any code or config the online run uses. The log's 200 lowest points (R 104.85–105.32, mean 105.13) are
called the **basin** below; they sit 1.71–1.89 spacings (median 1.87; spacing 0.377 = `median_spacing` of
the 2048 rows) from the nearest training row, on the simplex boundary
(each building ≈ 0.7 ramping, 0.25 degradation, cost/peak/discomfort ≈ 0).

## Verdict

| Assumption | Verdict | Evidence (mean ± sd over 3 splits unless noted) |
|---|---|---|
| 1. deterministic | **holds** | 7 repeated points in the log, R spread 0. The fitted noise (team 0.50 R, generic ARD 2.0 R) is therefore not noise but structure finer than the training spacing (part C) |
| 2. additive over buildings, shared length scales | **holds overall; sharing is approximate** | team kernel vs generic 21-dim ARD: held-out MAE 3.40 vs 4.81, Spearman 0.948 vs 0.896, log marginal likelihood / row −0.35 vs −0.56. Free per-building scales (part D) show building 2 about twice as smooth in cost, carbon, degradation and ramping, and resilience mattering only in building 3, in line with T20 (`analysis/mechanism/`) |
| 3. Matern 5/2 smoothness | **acceptable; RBF is too smooth** | Matern 3/2 and 5/2 tie in held-out terms; RBF is worst everywhere and far from data covers only 36 % of points with its 95 % interval |
| 4. stationarity | **holds at large scale, fails at small scale** | refitted per region (part C): training rows and far-from-basin log points give similar amplitude (20.7 / 19.5 R) and length scales (geo-mean 2.0 / 3.6). Near the basin five length scales are 0.010–0.025, 10–30× shorter than the shortest global scale (0.30); two sit at the lower bound. A single kernel absorbs these ripples (amplitude ≈ 1 R) as noise |
| 5. constant prior mean | **fails** | far from data the real R is systematically below the prediction (team: z mean −1.48, MAE 15.0). Basin: team predicts 124.4 ± 8.8 against a real 105.1; P(R < 115.765) = 0.17 |
| 6. Gaussian, homoscedastic, calibrated | **calibrated only in distribution** | held-out 95 % coverage 0.948, but residuals are heavy-tailed (LOO skew +0.90, excess kurtosis 4.9, \|z\| > 3 five times the normal rate) and shrink toward the mean (lowest-R quartile over-predicted, z mean −0.35; highest under-predicted, +0.43). On optimizer-chosen log points near the data z sd is 1.90: the `std` is about half the size of the actual error (winner's curse) |

## Part A — kernel family and structure (3 splits)

| metric | team | Matern 5/2 ARD-21 | Matern 3/2 ARD-21 | RBF ARD-21 | Matern 5/2 isotropic | Matern 5/2 ARD-21, √W |
|---|---|---|---|---|---|---|
| held-out MAE | **3.40** | 4.81 | 4.74 | 5.16 | 5.98 | 4.74 |
| held-out Spearman | **0.948** | 0.896 | 0.898 | 0.882 | 0.841 | 0.908 |
| held-out NLPD | **2.94** | 3.25 | 3.24 | 3.31 | 3.43 | 3.17 |
| log near MAE | **3.74** | 5.23 | 5.17 | 5.47 | 6.63 | 5.62 |
| log near z sd | 1.90 | 1.99 | 1.97 | 1.95 | 1.86 | **1.35** |
| log far MAE | **15.0** | 24.1 | 20.3 | 31.7 | 20.8 | 23.5 |
| log far 95 % coverage | 0.54 | 0.60 | **0.92** | 0.36 | 0.68 | 0.42 |
| log Spearman (all points) | **0.920** | 0.830 | 0.864 | 0.680 | 0.875 | 0.849 |
| basin predicted R (real 105.1) | **124.4** | 140.0 | 134.4 | 150.7 | 133.8 | 139.1 |
| basin P(R < 115.765) | **0.17** | 0.07 | 0.13 | 0.02 | 0.10 | 0.05 |
| real R of the top-ranked log point | 116.1 ± 3.4 | **111.3 ± 1.0** | **111.3 ± 1.0** | 112.1 | 121.1 | 113.9 |
| mean real R of the top 10 | 113.8 | **112.7** | 113.7 | 114.3 | 121.0 | 115.1 |
| fitted noise (R) | 0.50 | 1.99 | 0.84 | 3.03 | 3.64 | 2.00 |

The team kernel is best on every average measure. At the very top of the ranking the generic ARD kernels pick
slightly better log points (top-1 111.3 vs 116.1, but the team's top-1 swings 113.6–121.0 across splits).
The square-root (Hellinger) geometry gives the best calibrated `std` near data and worse extrapolation.

## Part D — three changes to the team model (3 splits)

| metric | team | + trend | + log target | + per-building scales | + trend + log | all three |
|---|---|---|---|---|---|---|
| held-out MAE | 3.40 | 3.99 | 3.44 | **3.36** | 4.01 | 3.98 |
| held-out Spearman | 0.948 | 0.924 | 0.947 | **0.950** | 0.924 | 0.924 |
| held-out z kurtosis | 9.1 | 6.6 | 7.3 | 10.7 | **5.5** | 6.2 |
| log near MAE | 3.74 | 4.18 | 3.74 | **3.63** | 4.13 | 4.01 |
| log near z mean | 0.32 | 0.45 | 0.34 | **0.22** | 0.46 | 0.34 |
| log mid MAE | 6.01 | 7.69 | 6.05 | **5.83** | 7.44 | 7.24 |
| log far MAE | 15.0 | 16.5 | 16.0 | 15.0 | 14.3 | **13.6** |
| log far 95 % coverage | **0.54** | 0.58 | 0.35 | 0.48 | 0.44 | 0.49 |
| basin predicted R (real 105.1) | 124.4 | 123.8 | 126.7 | 125.2 | 121.0 | **120.6** |
| basin P(R < 115.765) | 0.17 | 0.19 | 0.06 | 0.15 | 0.22 | **0.24** |
| mean real R of the top 10 | 113.8 | 118.6 | 114.0 | **113.1** | 118.6 | 119.6 |
| optimizer converged (of 3) | 2 | 0 | 3 | 0 | 3 | 0 |

- **Quadratic trend** (`trend = true`, BayesianRidge on degree-2 features): worse on every local measure (top-10
  118.6 vs 113.8) and it does not pull predictions toward the basin (123.8). Not recommended.
- **Log target** (`target = "log"`): lighter residual tails and a slightly better `std` near data, but it compresses
  the low end, so far from data it is more pessimistic (basin P 0.06, far coverage 0.35). Not for search.
- **Per-building length scales** (study-local kernel `PerBuildingARDKernel`, not a plugin): small, consistent gains
  near the data (near MAE 3.63, near bias 0.22, top-10 113.1), but the held-out gain (0.04) is within the split
  spread (0.05), and its 21 scales never converged in the team's 18 L-BFGS iterations. Worth a plugin variant
  with a larger budget (owner B).
- **Trend + log, all three**: best far MAE and basin probability, paid for with local ranking (top-10 118.6–119.6).

The team model itself did not converge in 1 of 3 splits within 18 iterations; whether a larger budget helps is
untested.

## Parts B and C

Leave-one-out residuals of the generic Matern 5/2 ARD fit (split 0, 1400 rows): z sd 1.057, skew +0.90,
excess kurtosis 4.89, share \|z\| > 3 1.36 % (normal 0.27 %).

| region (Matern 5/2 ARD-21 refitted on 800 points) | median radius (spacings) | R range | amplitude (R) | noise (R) | geo-mean length | shortest scales (dimension) |
|---|---|---|---|---|---|---|
| training rows | 1.51 | 119.3–242.5 | 20.71 | 2.08 | 2.03 | 0.30 (b1 ramping), 0.31 (b1 degradation), 0.34 (b3 ramping) |
| solver log near its lowest R | 0.10 | 104.9–108.4 | 1.18 | 0.12 | 0.13 | 0.010 (b1 cost), 0.010 (b1 degradation), 0.014 (b1 ramping) |
| solver log > 2 spacings from its lowest R | 2.17 | 108.2–292.7 | 19.45 | 2.64 | 3.56 | 0.17 (b1 cost), 0.18 (b1 ramping), 0.20 (b1 carbon) |

A region's size bounds what its fit can see: in a region of radius 0.10 spacings (0.04 in weight units) the
geo-mean length and the amplitude cannot be compared with the global fit directly. What the near-basin fit does
show is variation along five dimensions at scales (0.010–0.025) shorter than the region itself and far shorter
than any global scale.

## What this means for the online phase

- **Ranking near the data:** the team GP-ARD is the right default (best held-out and log-wide ranking).
  Per-building length scales are the only change worth carrying forward.
- **Uncertainty:** on points an optimizer picked, the `std` is about half the size of the real error near the
  data. An acquisition rule should inflate it (T14) rather than trust the in-distribution coverage.
- **Exploration:** none of the kernels or changes predicts the basin. The constant-mean assumption makes every
  stationary GP return to the mean away from the data, so a policy that trusts the surrogate will not walk
  about 1.9 spacings to the boundary (the team's limit is `distance_ratio` ≤ 1.5). Reaching that region needs exploration that does not depend on the model
  (simplex vertices, edges, faces), or a non-stationary or local model. This is a design question for T15 /
  T18, not something a kernel change fixes.
- **Replay (T17):** a stationary-GP posterior sample conditioned on the training rows would reproduce this
  pessimism, so it is not a faithful synthetic truth.

## Limits

- The log points come from search paths, not a uniform design: they are dense near what earlier searches
  favoured, which is also why no model recovers any of the 50 lowest points in its top 50.
- Three random splits of one table; held-out differences below about 0.05 MAE are within split noise.
- The team model and its changes use the team's 18-iteration L-BFGS budget; several fits stop before converging.
- `PerBuildingARDKernel` lives in this study only; the numbers are not plugin evidence. Leaderboard claims still
  need `sdm evaluate`.

## Reproduce

```bash
uv run python analysis/gp_assumptions/study.py --splits 3 --out analysis/gp_assumptions/results/20261112
```

About 20 minutes on one laptop. `results/` is git-ignored; the committed copies are in `delivery/20261112/`
(`summary.md`, `results.json`, `provenance.json`). Provenance records the git commit (`08b77c8`, clean), the
dataset and solver-log sha256, package versions, split seeds and the sha256 of `study.py`
(`a1e8bbad…e0c5e88f`), because the git dirty flag does not cover `analysis/`.

After the PR #23 review (2026-11-13), `search_scores` always returns the near / mid / far bands, with `n = 0`
and empty scores for a band without points, so `summarize` cannot fail on a missing key. The delivered run
had at least 1,017 points in every band of every record, so this change does not alter any delivered number;
the recorded sha256 is that of the version that produced them.

## Files

| path | what it is |
|---|---|
| `study.py` | the study: parts A–D, summary writer, provenance |
| `delivery/20261112/summary.md` | all tables, generated by `study.py` |
| `delivery/20261112/results.json` | every split × variant record, LOO and region fits |
| `delivery/20261112/provenance.json` | git state, checksums, packages, seeds |
