# Member C — optimizers and batch proposal

## Mission
Turn a surrogate into good candidate weights: one robust W* offline, and diverse, budget-aware batches
of 50 per online round.

## You own
`src/sdm/optimizers/*.py` (except `pgd.py`, frozen as the baseline; add a new optimizer instead of
changing it), `experiments/<optimizer>-*.toml`, tests for your optimizers.

## Ask A first
Any change to `Optimizer.propose`'s signature or to `sdm.simplex`.

## Backlog — offline (until the week 8 lab)
- [x] Reference optimizer `trust-dirichlet` (A, 2026-10-25): gradient-free, distinct batches, trust region,
      pessimistic objective. Improve on it (CMA-ES, better batch diversity) rather than on frozen `pgd`
- [x] Add `simplex-de`: gradient-free differential evolution with simplex projection, a trust region,
      observed-row exclusion, distinct batches and deterministic seeded search (2026-10-29).
- [x] Finish the two-seed HistGB/XGBoost optimizer comparison and export the offline submission with
      cross-model selection evidence (`analysis/offline_submission/submission/`, 2026-10-29).
      Recommended: proposal 16 from `20261029-125052-optimize-histgb-v32`; HistGB/XGBoost predictions
      108.9778/108.8717, distance ratio 0.5410. Real score unknown; ExtraTrees disagrees (125.2473).
      DE did not improve the cross-model selection score in this small fixed-budget comparison.
      82 tests passed, 1 known xfail; source optimizer run and final selection reproduce.
- [ ] Direct HistGB/XGBoost consensus search at three radii and three seeds, followed by separate
      five-subsample screening and five-subsample audit (`analysis/robust_search/`). Preserve the
      original submission unless the predefined replacement gate passes; no real score is available.
- [ ] Read `analysis/coupling/README.md` first: on ridge2 the optimizer choice barely changes the result
- [ ] Optimizers on the 3 × 7 simplex: CMA-ES with softmax parametrisation, Dirichlet local perturbation
      around the best observed rows
- [ ] Trust region: only search near the best measured rows; avoid simplex corners (see the extrapolation
      flag, `nearest_train_distance` in optimize runs)
- [ ] Pessimistic objective `mean + k · std` for choosing the offline W*

## Backlog — online (weeks 8–10)
- [ ] Each round, 50 points split into exploit / explore / probe (ratios from D's replay)
- [ ] TuRBO-style trust-region radius: grow after successes, shrink after failures

## Acceptance criteria
- `propose(n=50)` returns 50 distinct rows that pass `check_weights`; the uniqueness check in
  `tests/test_plugin_contract.py` passes for your optimizer.
- Offline candidates come with `optimize` run ids and a `distance_ratio` ≤ 1.5 unless D signs off.

## Interfaces
Implements `Optimizer` (`docs/interfaces.md`). Uses B's `Surrogate.predict` (mean, std).

## Handoff
PR description: algorithm, params, optimize run id(s), predicted mean/std and distance ratio of the top
proposals, batch diversity statistic.
