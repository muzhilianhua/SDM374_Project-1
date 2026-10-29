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
- [x] Direct HistGB/XGBoost consensus search at three radii and three seeds, followed by separate
      five-subsample screening and five-subsample audit (`analysis/robust_search/delivery/`). New
      exploratory candidate: `20261029-132648-optimize-consensus-r075-2026`, rank 1, full-data
      HistGB/XGBoost 107.8045/107.2767, distance ratio 0.6382. Audit mean risk improves by 1.0911,
      but two of five pairs regress slightly; retain the original default under the predefined gate.
      88 tests pass, 1 known xfail; source search reproduced. No real score is available.
- [x] Read `analysis/coupling/README.md`: on ridge2 the optimizer choice barely changes the result.
- [x] Study Dirichlet local perturbations around good observed rows and projected DE on the 3 × 7 simplex.
- [x] Compare distance-limited searches; require distance ratio <=0.75 for the current submission.
      This limits extrapolation but does not exclude every simplex boundary coordinate.
- [x] Apply pessimistic `mean + k · std` and cross-model risk selection; document that the error
      proxies are not calibrated confidence bounds.
- [x] Consolidate methods, decisions, provenance and handoff in
      [the research log](../optimizer-research-2026-10-29.md).
- [ ] Explore CMA-ES with softmax parametrisation if further optimizer comparisons are justified.
- [x] Complete the full-support sigmoid-weighting study: training-only inner selection, eight unchanged
      frozen-v32 evaluations and sixteen CLI candidate searches (`analysis/soft_weighting/delivery/`).
      Sigmoid XGBoost 72.4696 beats uniform 70.6201 but trails full-data step weighting 73.4746.
      Sigmoid HistGB 71.9815 trails uniform 72.3465. Retain step XGBoost and the original W default;
      the new point only improves expanded risk by .3045. 97 tests pass, 1 known xfail; evaluation
      and selected search reproduce. No measured improvement or new SOTA is established.
- [x] Broaden the search to CatBoost, robust/quantile losses, pairwise regression, explicit
      interactions, kernel geometry and convex ensembles; freeze discovery selection before a
      matched model-seed audit (`analysis/method_expansion/delivery/`). Fifteen recipes and 64
      CLI evidence runs; OOF stack passes the gate: mean score 75.3591 vs reference 73.1322,
      global MAE improves at every seed, average regret 1.1078 vs 1.7181. Recommend the stack
      surrogate recipe; preserve operational W. New CatBoost-search rank6 W is exploratory,
      with no real evaluation. 109 tests pass, 1 known xfail; evaluation and search reproduce.
- [x] Compare all1500 proposals from the15 methods with root baseline W using one unchanged
      audited ensemble judge (`analysis/common_judge/delivery/`). At original seed2026/rank1,
      13 of15 predict lower R; fixed-blend top1 is109.3303 versus baseline118.7316. Keep post-hoc
      best-of100 separate (fixed blend109.1711); archive source ranks, all predictions and exact
      weights. CLI replay matches all1503 inputs; 109 tests pass,1 known xfail. Real simulation
      is deferred by the user; operational W is unchanged and no measured improvement is claimed.

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
