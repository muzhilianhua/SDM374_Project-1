# Model × optimizer coupling study

Question: does the quality of the W* an optimizer finds depend on which surrogate it searches, and how much
does the choice of optimizer matter compared with the choice of surrogate?

```bash
uv run python analysis/coupling/study.py --seeds 5          # ~20 s; results/<stamp>/ (git-ignored)
```

## Method

1. **Compatibility.** Try every registered model × optimizer pair. A `NotImplementedError` (e.g. `pgd`
   asking a tree model for `smoothness`) marks the pair incompatible.
2. **Cross-judging on the real table.** Fit each model on all 2048 rows, let each compatible pair propose W*,
   and score W* with every model. Record `distance_ratio` = distance to the nearest training row /
   `median_spacing` (0.377). Seeds change both the model's and the optimizer's randomness.
3. **Semi-synthetic benchmark.** The real value of a new W* is unknown before the evaluator exists, so let
   a fitted model play the true function f_T. Labels are f_T(X) plus residuals resampled from T's
   out-of-fold errors, which gives realistic noise. Each pair is refit on those labels, proposes W*, and
   is scored exactly:
   - `regret = f_T(W*) − min_i f_T(X_i)`, where negative means better than every training row;
   - `optimism = f_T(W*) − model's own prediction`, where positive means the model over-promised.

   Two truths with opposite behaviour outside the data are used: `ridge2` is quadratic and rewards
   simplex corners, while `extratrees` is flat beyond the data.

## Results (5 seeds, 2026-10-25, commit 49aa0ea + uncommitted plugin code; rerun after merging)

| Pair | real data: judged by ridge2 / extratrees | distance ratio | regret, truth = extratrees | regret, truth = ridge2 | optimism (extratrees / ridge2 truth) |
|---|---|---|---|---|---|
| ridge2 × pgd | 86.8 / 126.9 | 1.80 | 7.12 ± 1.12 | −23.35 ± 4.03 | +31.6 / +7.1 |
| ridge2 × trust-dirichlet | 89.8 / 126.2 | 1.19 | 6.87 ± 1.81 | −23.41 ± 3.29 | +29.5 / +4.2 |
| extratrees × trust-dirichlet | 107.4 / 121.4 | 0.10 | **1.59 ± 1.23** | +1.51 ± 3.28 | +3.6 / −2.4 |
| extratrees × pgd | incompatible: no gradient / smoothness bound | — | — | — | — |

Best measured training row: p01969, R = 115.76.

## What it shows

1. **Interface dependency.** A gradient optimizer requires a differentiable surrogate. pgd cannot search a
   tree model at all; trust-dirichlet only calls `predict` and runs on any model.
2. **The surrogate decides, the optimizer barely matters.**
   - On ridge2, pgd and trust-dirichlet reach the same region; their regret differs by ≤ 0.25, well inside
     the seed spread.
   - Switching the surrogate moves regret by about 5.5 under the extratrees truth and about 25 under the
     ridge2 truth.
   - The surrogate's shape decides where the optimum lies. The quadratic puts it outside the data
     (ratio 1.2–1.8, pressed against the trust-region edge). The trees put it on the data (ratio 0.1),
     and a search there collapses onto the neighbourhood of the best rows.
3. **Which pair is "better" depends on an unknown fact about the real response.**
   - If R rises outside the data the way a tree-like truth implies, ridge2's W* over-promises by about 30
     and lands about 7 worse than the best row.
   - If R keeps falling toward the corners the way ridge2 implies, that same W* beats every row by about
     23, while extratrees × trust-dirichlet stays near the best row and misses the gain.
   - Neither surrogate can settle this offline.
4. **A risk-aware objective needs a heteroscedastic std.** ridge2's std is constant, so `mean + risk · std`
   ranks exactly like the mean; only the trust region restrains it. That restraint cut ridge2's
   over-promise slightly (31.6 → 29.5, 7.1 → 4.2) without changing regret.

## Implications

- Spend effort on the surrogate's behaviour outside the data (calibrated uncertainty, sensible
  extrapolation) before more optimizer sophistication.
- Use the two week-8 offline submissions as an experiment that tells the two truths apart:
  - one conservative W* (extratrees × trust-dirichlet, near the data);
  - one aggressive W* (ridge2, outside the data).

  Whichever scores better indicates how the real response extrapolates and should steer the online phase.

## Caveats

- The truths are models fitted to this same table, so the benchmark measures behaviour under two
  plausible worlds, not the real one.
- On the real data, seed changes ridge2's internal CV folds and therefore alpha. That is why ridge2's
  W* varies across seeds (seed spread 0.15–0.44).
- extratrees' std is too wide (`std_coverage` 0.857 vs 0.68 ideal); calibrating it would sharpen the
  risk term.
