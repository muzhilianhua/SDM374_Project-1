# Member B — surrogate models

## Mission
Predict R from the 21 weights as accurately as possible **at the low-R end**, with an honest per-point
uncertainty, so that optimizers can trust the surrogate where it matters.

## You own
`src/sdm/models/*.py` (except `ridge2.py`, which is frozen as the baseline), `experiments/<model>-*.toml`,
tests for your models under `tests/`.

## Ask A first
Any change to `src/sdm/interfaces.py`, `evaluate.py` or the `[evaluate]` protocol.

## Backlog — offline (until the week 8 lab)
- [x] Reference tree model `extratrees` (A, 2026-10-25): mae 5.723, top50_recall 0.56, spearman 0.882,
      std_coverage 0.857 (std too wide) — run `20261025-145809-evaluate-extratrees-default`
- [ ] Model leaderboard via `sdm evaluate`: gradient-boosted trees, random forest, Gaussian process
      (a read-only probe gave GP-ARD mae 5.02 / top50_recall 0.60 but ~4 min per evaluate), MLP ensemble
- [ ] Smooth or extrapolation-aware models matter for optimization: see `analysis/coupling/README.md`
- [ ] Feature engineering inside `fit` / `predict`: within-building ratios (e.g. ramping vs.
      cost / carbon / degradation), per-building additive structure
- [ ] Per-point `std` for every model (ensemble spread, GP posterior, quantile models). Calibration
      target: `std_coverage` (share of rows whose out-of-fold error is within one std) near 0.68,
      and `mean_predicted_std` of the same order as `rmse`. Bootstrap spread alone is usually too narrow
- [ ] Provide `gradient` / `smoothness` where cheap so `pgd` can be reused

## Backlog — online (weeks 8–10)
- [ ] Retrain each round on offline + online rows; report the previous round's predicted vs. measured error
- [ ] Recalibrate `std`; optionally up-weight rows near the current trust region

## Acceptance criteria
- Beat ridge2 on the shared protocol: **`mae` < 6.828** and **`top50_recall` > 0.46**
  (`spearman` > 0.838 is a plus), each claimed with an `evaluate` run id.
- `std_coverage` between 0.55 and 0.80, or the PR explains why not.
- `tests/test_plugin_contract.py` passes for every model you add.

## Interfaces
Implements `Surrogate` (`docs/interfaces.md`). Consumed by C's optimizers, D's replay and the CLI.

## Handoff
PR description: model name, key params, evaluate run id(s), metric table vs. ridge2, known weaknesses.
