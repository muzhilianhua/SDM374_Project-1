# Member C — optimizers and sequential proposal

## Mission
Turn a surrogate into good candidate weights offline and a reproducible policy that proposes one W
per call, using feedback within a total budget of 20 calls (decisions 19–22).

## You own
`src/sdm/optimizers/*.py` (except `pgd.py`, frozen as the baseline; add a new optimizer instead of
changing it), `experiments/<optimizer>-*.toml`, tests for your optimizers.

## Ask A first
Any change to `Optimizer.propose`'s signature or to `sdm.simplex`.

## Current status — PR #7 review

- P1 (`20261029-132648-optimize-consensus-r075-2026`, rank 1) was offline submission 1:
  real R 119.71206678869203, above the bar (team decisions, Findings). Do not resubmit it.
- Baseline W* 119.0916 is TA-confirmed. P2 and P3 remain unmeasured historical/research candidates;
  the old operational-default designation is not a submission-2 endorsement.
- The 50-point packet is superseded and is only an idea pool. The separate 20 direction probes are
  also a research pool, not an implemented sequential policy.
- [x] Synchronize decisions 17–23; revise generated handoff status, known-result aliases and the root link.
- [x] Handle SLSQP non-success with feasibility/objective checks, recorded diagnostics and prior fallback;
      refine long-ray bisection and verify its remaining-interval error; warn on nonzero residual correction.
- [x] Move large regenerable artifacts to checksummed release assets while preserving source history.
- [ ] Choose offline submission 2 using decision 23; no selection is made by this PR.

## Bayesian study — PR #9 revision

- Preserve main6dd38eb and PR7's reviewed fixes; replay only the five Bayesian commits.
- Historical additive ARD GP frozen-v3.2 score83.1496 exceeds the same-machine stack74.3862;
  top1 regret2.0014 versus1.8202 does not improve. The original r06 submission recommendation is withdrawn.
- EI now requires a native GP posterior; the blend retains a heuristic spread for mean/pessimistic
  acquisition. Read committed acquisition configs and validate modes before executing.
- Honor and deduplicate ARD starts; record learned kernel values and optimizer termination diagnostics.
  GP seed is a no-op; the old seed comparison varies only the tree half.
- [x] Repair all eleven unsupported EI TOMLs; exercise actual proposal compatibility for all26
  shipped Bayesian configs, including the two trust-dirichlet audits.
- [x] Replace wheel-layout replay instructions with the supported editable workflow in the generator,
  generated guides and artifact instructions. Add an isolated ASCII-path worktree replay check with
  PYTHONPATH unset; verification records are in delivery/current/editable_replay/.
- Current records use submission1=119.71206678869203 and baseline W*=119.0916. The bar remains
  R<115.76. Preserve old evidence as historical; decision23 forbids solver screening of submission2.
- Use ExtraTrees only as a separate eligibility veto, not a generator or ranking model. Its two-point
  motivation is limited evidence, not a guarantee. Ray and candidate audits remain offline research.
- Role B's model-owner review remains pending before merge. No solver call, official submission or merge.

## Backlog — offline (historical completed work)

- [x] Publish the labelled handoff and reproducible builder in `analysis/submission_handoff/`.
      [Current package](../../SUBMISSION_READY/README.md) records real feedback and superseded pool rules.
- [x] Complete selection-conditioned retraining with disjoint internal roles and frozen outer evaluation.
      The correction failed its replacement gate; the full-data fit selected zero correction.
      [Study and exact run IDs](../../analysis/selection_training/delivery/README.md).
      Precision, fresh-direction and matched training-subset audits remain archived. The separately
      declared multi-anchor probes improve direction coverage without claiming measured improvement.
- [x] Complete stable subsample training and label-excluded calibration. Regularized alternatives failed
      the replacement gate; calibration-only changes preserved predictive means. Regional generalization
      remains unresolved. [Study and exact run IDs](../../analysis/stable_training/delivery/README.md).
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
      88 tests pass, 1 known xfail; source search reproduced. The challenger later measured 119.71206678869203 and failed; see Current status above.
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
      surrogate recipe; preserve operational W. New CatBoost-search rank 6 W is exploratory,
      with no real evaluation. 109 tests pass, 1 known xfail; evaluation and search reproduce.
- [x] Compare all 1500 proposals from the 15 methods with root baseline W using one unchanged
      audited ensemble judge (`analysis/common_judge/delivery/`). At original seed 2026/rank 1,
      13 of15 predict lower R; fixed-blend top1 is109.3303 versus baseline 118.7316. Keep post-hoc
      best-of100 separate (fixed blend109.1711); archive source ranks, all predictions and exact
      weights. CLI replay matches all 1503 inputs; 109 tests pass, 1 known xfail. Real simulation
      is deferred by the user; operational W is unchanged and no measured improvement is claimed.
- [x] Publish the user-requested selected full-data stack binary in `artifacts/best_model/`, with
      original config/meta/metrics, dependency versions, checksums and CLI loading instructions.
      Exact source-model bytes are preserved; all 1503 comparison predictions match on CLI replay.
- [x] Independently audit the supplied candidate HTML's input-neighborhood stability
      (`analysis/perturbation_audit/delivery/`, 2026-11-01). Archived judges reproduce all 15
      displayed single-point differences. Shared directions compare29 candidates with 93, 728
      feasible neighbors. Challenger ranks first by success fraction in all three audit seeds:
      pooled81.90% at actual radius.02 and 46.55% at.05. The HTML's62% at.05 is not reproduced;
      nearest runner-up is45.05%, so no clear large-radius superiority over it is established.
      This historically supported a real test; that candidate subsequently failed. The audit is not evidence of real improvement.
      No model retraining or simulator call; 115 tests pass and 1 known xfail.
- [x] Retrain the selected seed 2026 stack and consensus strategy on the unchanged 2, 048 rows
      (`analysis/retrain_comparison/delivery/`, 2026-11-01). Six fresh CLI runs reevaluate Ridge,
      step-XGBoost and stack, refit stack/consensus, and rerun consensus search. Stack remains
      v3.2=75.1137,MAE=4.6933; all three model evaluations reproduce their old outer predictions
      exactly. Old/new stack and consensus predictions agree on 1, 554 inputs; all 50 raw search
      proposals reproduce exactly. Publish a separate loadable fresh stack bundle without
      replacing the old artifact or operational W. No new measured rows or real-R improvement;
     115 tests pass, 1 known xfail. Exact run IDs and comparisons are archived in the delivery.
- [x] Revise the user-facing offline-candidate decision page inside the designated repository
      (`analysis/candidate_table_revision/delivery/`, 2026-11-01). Preserve the external original.
      Historically proposed challenger priority and alternate rankings; these are now superseded. Required
      require verified feedback before choosing submission2. Export7 candidate files and 2 controls,
      verify feasibility and old/new stack prediction agreement, and record a proposal in
      `docs/decisions.md` without replacing the operational W or the settled evaluation protocol.

## Backlog — online (weeks 8–10)
- [x] Prepare the first 50-point online packet under the supplied course slides' page3/page7 rules
      (`analysis/online_round1/delivery/`). Thirty exploitation, 15 exploration, 5 controls; this initial
      ratio is a heuristic pending D's replay. All50 are mutually distinct and simplex feasible;
      one p01969 repeat was deliberate under the old rules; graded calls now exclude known points. Include baseline, existing recommendations,
      mechanism probes, JSON/CSV/individual files and a pending worksheet. No evaluator calls.
      Full suite: 112 pass, 1 known xfail; selected stack predictions replay and all payloads agree.
- [ ] Superseded by revised decision 19 (2026-11-01): there are no 50-point rounds. The online deliverable is
      a policy making 20 sequential single-point calls; the packet above is a pool of starting ideas only
- [ ] One point per call: exploit / explore choice per call (settings from D's replay)
- [ ] TuRBO-style trust-region radius: grow after successes, shrink after failures
- [ ] Note for the policy: offline submission 1 (the consensus challenger) measured 119.71 against predicted
      107–111 (decisions, Findings). Boosted-tree consensus is over-optimistic where it was searched

## Acceptance criteria
- The online policy requests `propose(n=1)` and excludes training rows and all earlier/known points.
  The general optimizer API still supports distinct multi-point research pools and its contract tests.
- Offline candidates come with `optimize` run ids and a `distance_ratio` ≤ 1.5 unless D signs off.

## Interfaces
Implements `Optimizer` (`docs/interfaces.md`). Uses B's `Surrogate.predict` (mean, std).

## Handoff
PR description: algorithm, params, optimize run id(s), predicted mean/std and distance ratio of the top
proposals, batch diversity statistic.
