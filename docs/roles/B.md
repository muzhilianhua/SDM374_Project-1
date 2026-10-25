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

- [x] Frozen v3.2 and final-artifact handoff (2026-10-25): `analysis/surrogate_release/README.md`.
      Original split/scorer/model ZIPs and teammate Word report preserved; 11-group historical scores
      separated from full-data configurations. Saved full-config predictions score 72.47 / 72.35 /
      72.34 under v3.2 (analysis `20261025-152147-release-verification`, dirty source provenance).
      All three full-data artifacts load and match supplied in-sample predictions; this is a smoke
      check, not a new generalization test. No training, tuning, plugin integration or W optimization.
- [x] User-requested local artifact review (2026-10-25): screenshot top-five peer groups plus six earlier
      finalists, 110 frozen outer-fold results, no fitting or retuning. Analysis id
      `20261025-204551-artifact-review-v32`; source `analysis/artifact_review/`. Peer held-out errors
      reproduced from 50 saved artifacts; 60 original prediction sets rescored exactly. This uses the
      preserved local v3.2 protocol, not the team's `sdm evaluate` leaderboard. Source was uncommitted
      at execution because Git identity was unset, and was later committed as `c00e955`; original
      provenance retains source/input hashes and the dirty flag.
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
