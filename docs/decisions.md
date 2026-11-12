# Decisions and open questions

Settled choices, so nobody (human or agent) re-litigates them. To change one, raise it with member A
and record the new decision here with a date.

## Settled

| # | Decision | Why |
|---|---|---|
| 1 | Python 3.11 + uv; versions pinned to the course baseline (numpy 1.24.4, scikit-learn 1.2.2) | Same numbers on every machine; numpy 1.24 has no 3.12 wheels |
| 2 | Plugins register by decorator and are auto-discovered from `sdm.models` / `sdm.optimizers` | Adding a model or optimizer never edits shared files |
| 3 | `Surrogate.predict` returns `(mean, std)`; `std` is mandatory | Pessimistic offline selection and online exploration both need uncertainty |
| 4 | `Optimizer.propose(model, X, y, n)` takes NumPy arrays, not a DataFrame | Simpler, and online data is appended as rows of `X, y` |
| 5 | Experiments are TOML files in `experiments/`; every result comes from an `sdm` command and lives in a run folder | Reproducible numbers with provenance (`sdm reproduce`) |
| 6 | Models are compared only with `sdm evaluate`. Primary protocol (2026-10-27): `frozen-v32` — the frozen 2×5 outer splits in `frozen_data/`, ranked by the mean v3.2 score from `evaluation_v3_2/`. The original `kfold` protocol (5 folds, seed 2026, top-50 recall) stays available, and its metrics are also reported under `frozen-v32` | Internal CV used for hyper-parameter choice is optimistic; v3.2 rewards what optimization needs (low-region order, first-choice regret), and member B's earlier results use the same splits |
| 7 | `runs/` is not in git; results are shared by run id or by copying the folder | Run folders hold pickles and are large |
| 8 | The GUI only launches `sdm` CLI commands (background jobs in `runs/.jobs/`) and never computes results in-process | One execution path for humans, agents and graders |
| 9 | Core code and docs in English; GUI text in Chinese | Team convention |
| 10 | `ridge2` + `pgd` must stay equivalent to `baseline.py` (regression tests) | It is the reference every improvement is measured against |
| 11 | `Optimizer.propose` orders rows by the optimizer's own objective (e.g. mean + risk · std), not necessarily by mean | Risk-aware optimizers rank by a pessimistic score (2026-10-25) |
| 12 | Analysis scripts under `analysis/` may call plugins directly if they write `provenance.json` | Factorial studies need many fits that are not individual experiments (2026-10-25) |
| 13 | `frozen_data/` and `evaluation_v3_2/` are read-only releases; `sdm.frozen_v32` reads them in place and refuses to run if their sha256 changes. The v3.2 scale `s` is the population SD of each split's outer-train labels | One copy of the splits and scorer for everyone; reproduces member B's published scores exactly (2026-10-27) |
| 14 | `kfold` and `frozen-v32` results are never ranked against each other; the leaderboard lists `frozen-v32` runs first (by `v32_score`, higher is better), then `kfold` runs (by MAE) | The two protocols use different splits (2026-10-27) |
| 15 | `xgboost==3.2.0` is a dependency (works with numpy 1.24.4); macOS needs `brew install libomp` once | B's best artifacts are XGBoost; macOS wheels link the system OpenMP runtime (2026-10-29) |
| 16 | Boosted-tree plugins share `models/_gbdt.py`: std is the 0.68 quantile of \|out-of-fold residual\| in bins of distance to the nearest training row, non-decreasing with distance | Boosted trees have no native uncertainty; distance bins widen std where the optimizer extrapolates (2026-10-29) |
| 17 | Offline bar is **R < 115.76**: the TA sets it to min(baseline W\*'s real score 119.0916, best training row `p01969` 115.7648). Submitting a training row only ties the bar | TA, 2026-10-31 |
| 18 | Offline: two submissions, made one after the other with the first result visible before the second; any gap between them; both due before the end of class on **2026-11-05**. The window is open, so the first can go in early | TA, 2026-10-31 and 2026-11-01. Plan the two as a decision tree: the second depends on the first result |
| 19 | Online (revised 2026-11-06, from the released `solver/README.md`): the submitted, reproducible code may call the evaluator **at most 20 times, sequentially**; each call sends **up to 50 weight sets** and gets one R per set, and may use the results of earlier calls. Before its first call the code may use **only the 2048 training rows**. A call scores the **minimum** R of its batch and the group grade is the **best** call, i.e. the lowest R over all ≤ 1000 points. During development the evaluator runs locally without limit: deleting `usage.count` resets the 20-call quota | TA 2026-10-31 (minimum per batch, best of 20); user 2026-11-01 (only training rows before the first call); solver README 2026-11-06 (≤ 50 sets per call), adopted by the user 2026-11-06. Supersedes the 2026-11-01 "one weight set per call" reading. Batches should maximise the chance that one point is very low, not the batch mean |
| 20 | The evaluator is deterministic (same weights → same R). The policy must never re-evaluate a training row or a point it already sent, within a batch or across calls; local development runs are free | TA, 2026-10-31. With deterministic code and data, a local run of the frozen policy reproduces the graded result exactly |
| 21 | Evaluator constraint: each building's 7 weights non-negative with at least one > 0; they need not sum to 1. We still submit exactly normalised weights that pass `sdm.simplex.check_weights` | TA, 2026-10-31. Whether the evaluator renormalises is unknown, so normalised inputs are the only ones whose meaning matches the training table |
| 22 | The online code must be frozen and fully reproducible on the course computer: re-running it there sends the same batches in the same order | TA, 2026-11-01 |
| 23 | Offline submission 2 is chosen offline: from the 2048 training rows, the two known real scores (baseline W\*, submission 1) and surrogates. The course `RampingSolver.exe` is not used to screen candidates; it currently runs only on the course machines | User, 2026-11-02 |
| 24 | The offline phase is frozen at annotated tag `offline-final` (PR #11 merge, `62a2274`). Online work happens on `feat/online`, which drops the offline studies under `analysis/`, `SUBMISSION_READY/`, `artifacts/`, the offline experiment configs and their tests. It keeps `src/sdm` (every plugin), `app/`, `analysis/coupling/`, `analysis/v33_protocol/` and the two `SUBMISSION_READY` files that V3.3 reads. Rows and findings in this file that cite removed paths resolve at the tag | User, 2026-11-08: concentrate on online script development and testing |
| 25 | `solver/` is in the repository with its course files unchanged; `solver.exe` (182 MB, above GitHub's 100 MB limit) and `usage.count` are git-ignored. Everyone copies the exe from the course `solver.zip` and checks it against `solver/CHECKSUMS.sha256` (sha256 `293867b8…a077247`) | User, 2026-11-08 |
| 26 | Online framework design (T06, T08). Every source of R implements one `Evaluator` interface (`sdm.online`): the real `solver.exe`, a table of measured rows and a synthetic truth, with the solver's input rules and a per-object 20-call quota; the graded run never touches `usage.count`. The online loop (T08) will: drive proposals through a new `BatchPolicy` interface that receives the whole online state (call number, quota left, every sent point and result) and wraps existing optimizers; keep its state in a run folder `runs/<id>-online-<config>/` and resume with `sdm online --resume <run-id>` after checking the config and code version; when the pre-submit check drops points (duplicates of training rows or sent points, simplex violations), ask the policy for more candidates and send the first 50 that pass in the policy's order, logging the reasons, and stop without calling only when none pass | User, 2026-11-10 |

## Findings

- **Offline submission 2 met the bar; the online target is 115.68** (submitted 2026-11-05, recorded
  2026-11-09). Real R = **115.68**, below the bar 115.7648 (`p01969`, decision 17) by about 0.08. The
  exact value returned by the evaluator is still to be added. The best offline result is therefore
  min(119.7121, 115.68) = **115.68**: the online phase earns its 8 % only with a call whose batch minimum
  is below it.
  The submitted W is `p01969` + 0.05 · u[15713]: direction 15713 of the seed-2026 directions in
  `analysis/offline_submission2/study.py` (tag `offline-final`), one of the 19,483 that avoid both measured
  failures. It is the top pick when only the ARD GP is required. PR #11's final rule recommended direction
  14840 instead, because the local GP was not clearly confident about 15713 (z 0.34 < 0.5). Judges at
  `4030012`:
  - ARD GP: gain −2.85, z 1.90;
  - local GP: gain −0.86, z 0.34;
  - calibrated P 0.861, worst on the radius-0.01 ball 0.831;
  - ExtraTrees: +2.39.

  The real gain is about −0.08. Both GPs had the right sign but overstated the improvement, ARD by about
  2.8 and the local GP by about 0.8. One point does not show whether PR #11's agreement rule was wrong.
  The file submitted was `02_A_dir15713_step0.05_ARD_only.json` (sha256 `0ec11414…c53b4d7`) from a
  candidate bundle built from `4030012`. It is not in the repository, so the weights are kept here.
  Minimum weight 0.0245; it passes `check_weights` and matches `p01969` + 0.05 · u[15713] to 1e-16.
  Rows are buildings 1–3 and columns follow `METRICS`:
  ```json
  [[0.15769301241644582, 0.05264139579999994, 0.20854081714531605, 0.22837332588192233, 0.033200075292113704, 0.23276077156978842, 0.08679060189441362],
   [0.2503591199835666, 0.13676967852502162, 0.08313656511837968, 0.0604233371724592, 0.12222408788989679, 0.28509901488585504, 0.06198819642482098],
   [0.15385821627102042, 0.024476314274646253, 0.04676748001670422, 0.07253577354819109, 0.10757230502504825, 0.36784761044033554, 0.22694230042405422]]
  ```
  This W and its R count as known points: graded online calls must not resend it (decision 19).
- **Offline submission 2 recommendation** (2026-11-05, [study](../analysis/offline_submission2/README.md),
  commit `e4fa38f`, no solver calls). The PR #9 research lead (r06 line, distance 0.08, ARD gain −2.63) is
  rejected by a second smooth judge, the new `local-gp` (`20261105-145418-evaluate-local-gp-v32`):
  +1.73, worse than `p01969`. The r05 line, the only one both GPs like, points within cosine 0.79 of the
  step to submission 1. A search over 19,483 directions that avoid both measured failures picks
  direction 14840; at step 0.05 every smooth judge variant (ARD with and without `p01969`, local GP with
  400 / 800 / 1200 neighbours and without `p01969`) predicts a gain between −1.42 and −2.27, also at the
  worst point of a radius-0.01 ball, and extratrees +1.91 is inside its +4.9 veto. Recommended W:
  `analysis/offline_submission2/delivery/consensus-d0.05.json`. Where either GP predicts below 120, the
  real value was 1.1–2.3 higher, and the direction is the best of many, so the rough chance of R < 115.76
  is 45–65 %.
  *Superseded the same day by the sign study* (`analysis/offline_submission2/sign_study.py`, commits
  `6e3954d` backtest and `228dcae` scan, both clean). The study back-tests the judges on 4,256 anchor/test
  pairs of the frozen outer splits and calibrates P(improvement) per held-out split. On unique pairs,
  agreeing GPs were right 79 % of the time against about 29 % when they disagreed; ExtraTrees adds nothing.
  The calibrated P(R < R(`p01969`)) is flat for steps 0.04–0.10 along direction 14840.
  Recommended W at the time: `analysis/offline_submission2/delivery/sign/max-p.json` (then step 0.08).
  *Corrected after the PR #11 review*: the two CV repeats re-created most pairs, so the split holdout
  reused labels. With held-out candidates purged (commit `228dcae`), the top calibration bin is 0.83
  predicted against 0.63 observed. Unique agreeing pairs improved 79 % of the time. A defensible but
  unvalidated reading for the recommended W is 0.6–0.8.
  *V3.3.2 check of the judges* (`analysis/offline_submission2/v332_judges.py`, clean `890bf5f`, frozen
  release unchanged). ARD GP scores 59.96, level with the best boosted recipe (60.51), with the best low-R
  accuracy (true-low MAE 3.59 vs 6.5–7.2) and the closest full-fit P1 (121.49 vs 113.0–115.2). Both GPs fail
  all three region-holdout tasks with false breakthroughs (predicted 111–118, measured 120.6–124.6), so
  their low predictions far from data can be several points optimistic. With `p01969` in training they
  claimed no test row below it. Their failing picks lay 0.31–0.39 from any training row.
  *Final rule* (`ab6cbf0`): ExtraTrees predicts every step from `p01969` as worse, so its veto only capped
  step length and is replaced by an explicit cap: steps ≤ 0.05, where the calibrated probability has
  already plateaued (from 0.04), and both GPs at z > 0.5. Recommended W: `delivery/sign/max-p.json`,
  direction 14840 at step 0.05, identical to `consensus-d0.05.json`. Defensible, unvalidated reading 0.6–0.8.
- **Offline submission 1 failed the bar** (2026-11-02). Submitted C's consensus challenger
  (`20261029-132648-optimize-consensus-r075-2026` rank 1, identical to `SUBMISSION_READY/01_FIRST_REAL_TEST`);
  real R = **119.7121**, 3.95 above `p01969` and 0.62 above baseline W\*. Predictions for the same W: stack
  111.42 (error −8.29), histgb 107.80 (−11.91), xgboost 107.28 (−12.44), extratrees 126.09 (+6.38), ridge2
  95.13 (−24.58). Predicted gain over each model's own `p01969` prediction: stack −6.76, histgb −9.02,
  xgboost −9.67, extratrees **+4.93** (real +3.95). Consequences:
  - The boosted judges' shared basin (all good candidates within cosine ≥ 0.88 of this direction) is an
    artefact of correlated tree models searched on their own error; its alternatives (C's retained
    recommendation, fixed-blend rank 1) are not worth a submission.
  - Extratrees is the only model whose predicted gain had the right sign at both real points (baseline W\*
    +6.05 vs real +3.33; challenger +4.93 vs real +3.95). Two points; treat as a veto, not as a ranking.
  - `p01969` is isolated: its nearest training row is 0.30 away (R 127.5) and the next-best row (117.58) is
    0.78 away. A local Gaussian process (Matern 5/2, ARD, 800 nearest rows, scratch analysis, no run id)
    predicted the challenger *blind* at 120.08 ± 5.03, where the boosted judges said 107–111.
- **Baseline W\* settles the extrapolation question** (TA real score 119.0916, 2026-10-31). Predictions for the
  same `weights.json`: ridge2 75.8957 (error −43.20), stack 118.7316 (−0.36, `artifacts/best_model`),
  histgb 117.6234 (−1.47, `20261101-140115-train-histgb-v32`), xgboost 114.2081 (−4.88,
  `20261101-140119-train-xgboost-v32`). The quadratic surrogate's corner-of-simplex optimum is an artefact,
  so the "aggressive ridge2 candidate" below is dropped; tree ensembles, the stack above all, are the
  surrogates to trust near this region. Rank offline candidates by predicted improvement over each
  model's own prediction for `p01969`, not by raw predicted R.
  *Qualified 2026-11-02: accurate at baseline W\* is not accurate where the trees were searched; submission 1
  (above) missed by 8–12 on stack and the boosted trees.*
- **Member C stable training and honest calibration (historical study, reviewed 2026-11-03)**:
  [completed study](../analysis/stable_training/delivery/README.md). The mild regularized recipe
  averaged v3.2 73.7974 across `20261101-182343-evaluate-stable-study-mild-2026`,
  `20261101-183234-evaluate-stable-study-mild-2027` and
  `20261101-183234-evaluate-stable-study-mild-2028`. The reference averaged 75.3591 across
  `20261101-182343-evaluate-stable-study-reference-2026`,
  `20261101-183141-evaluate-stable-study-reference-2027` and
  `20261101-183141-evaluate-stable-study-reference-2028`, so the replacement gate failed.
  The calibration-only follow-up preserved outer mean predictions in
  `20261101-183914-evaluate-stable-study-honest-2026`,
  `20261101-183914-evaluate-stable-study-honest-2027` and
  `20261101-184157-evaluate-stable-study-honest-2028`, while excluding calibration labels from
  predictive fitting. This establishes no predictive gain or guaranteed interval coverage.
  Auxiliary regional and candidate diagnostics are recorded with provenance in the study archive;
  regional generalization remains unresolved. The challenger was subsequently submitted as offline
  submission 1 and failed (see the measured finding above). The former first-test recommendation
  is superseded; preserved historical weights are not selected for submission 2.
- **Member C candidate-table proposal (historical, superseded after real feedback)**:
  [archived decision page](../analysis/candidate_table_revision/delivery/candidate_table.html).
  It proposed `20261029-132648-optimize-consensus-r075-2026`, rank 1, for the first real test.
  That exact controller became offline submission 1 and measured 119.71206678869203 (failed).
  Baseline W* 119.0916 is TA-confirmed (decision 17). Input-perturbation stability and
  training-subsample stability are different surrogate diagnostics; neither guaranteed real gain.
  Same-data retraining reproduced the old outputs. The historical alternate rankings are not a
  new submission decision. Offline submission 2 follows decision 23 and the updated real feedback.
- **Model–optimizer coupling** (`analysis/coupling/README.md`, 5 seeds): which W* an optimizer finds is
  decided mostly by the surrogate's shape, not by the optimizer. On ridge2, pgd and trust-dirichlet land in
  the same extrapolated region (semi-synthetic regret differs by ≤ 0.25); switching the surrogate changes
  regret by 5–25. Whether ridge2's extrapolated W* is good depends on how the real response behaves outside
  the data, which only the evaluator can tell. Recommended offline submissions: one conservative pair
  (extratrees + trust-dirichlet, near the data) and one aggressive pair (ridge2, outside the data).
  *Superseded 2026-11-01 for the aggressive pair: the baseline W\* finding above refutes ridge2's extrapolation.*
- Gradient optimizers are incompatible with piecewise-constant surrogates (pgd × extratrees).
- **Boosted trees in the framework** (commit `d1504d6`): histgb 72.35 (`20261029-073439-evaluate-histgb-v32`,
  identical to B's saved `histgb_low80_full` predictions to 1e-14), xgboost 73.47
  (`20261029-073455-evaluate-xgboost-v32`, top-1 regret 1.07). Conformal std is calibrated: std_coverage
  0.696 / 0.687. With trust-dirichlet, both give 50 distinct proposals inside the trust region (distance
  ratio ≈ 0.7), but their best predictions (106.9 / 99.4, runs `…-073519-optimize-histgb-v32`,
  `…-073522-optimize-xgboost-v32`) sit 9–16 below the best measured row, which is likely optimism from
  searching on model error. Treat them as candidates, not expected scores.
- **First `frozen-v32` results** (commit `28cab43`): ridge2 66.45 (`20261027-174545-evaluate-baseline-v32`),
  extratrees 68.00 (`20261027-174547-evaluate-extratrees-v32`). Extratrees wins on ordering and global MAE
  but loses on first-choice regret (5.85 vs 4.51); the 1.55-point gap is well inside the fold SD (≈ 8–9).
  For reference, member B's tree models score 72.3–74.7 on the same splits (`analysis/surrogate_release` on
  branch `runze-liu`), so gradient-boosted plugins are the obvious next step.

- **T18 replay defaults; earlier strategy ranking withdrawn** (2026-11-09, commit `300193b`,
  [replay](../analysis/replay/README.md), `analysis/replay/study.py`, no evaluator calls). The ranking of
  strategy families from 6,568 local `solver.exe` calls, and the structural prior handed to T15 (old
  notebook §6) and the note that T12 could adopt that ranking (old §9), are **withdrawn as policy input**:
  decision 19 admits only the 2048 training rows before the first call and T03 is unanswered, so no value
  measured with the real evaluator may enter the first batch. The log stays in `analysis/replay/evidence/`
  as a search history only. Re-run on the replay (20 calls × up to 50 points, 1600-row holdout pool, 448
  initial rows, 5 seeds, `table` and `semi-synthetic` truth), with curves scoring submitted points only
  and the training incumbent kept as a separate reference: under `semi-synthetic` every setting reaches
  the truth minimum (128.175) in one call, so that arm cannot rank settings. Under `table` only the first
  batch separates: exploit ≥ 0.5 puts each seed's pool minimum in the first batch (117.350 / 117.507),
  while pure exploration starts at 126.937 ± 2.833 and needs a median of 5 calls; every exploit ≥ 0.5 run
  then exhausts its eligible pool, so later calls and the final value (117.350 ± 1.957) cannot rank those
  settings. Those pool rows are training rows and the online pre-submit drops duplicates (decisions 20
  and 26), so 117.350 / 117.507 are selection tests on the table, not achievable online scores. The
  radius is **not** separated by this replay: under `table` it only filters a fixed pool
  (seed 0 / `histgb` / exploit 0.5 submits 10 points at radius 0.5 against 501 at 1.0), and under
  `semi-synthetic`, which uses the real `bayes-pool` proposer, both radii give the same result
  (128.263 after call 1, 128.175 final), so radius 1.0 is a judgment call.
  (Revised 2026-11-10: the first version scored known rows as if submitted and silently filled short
  batches out of radius, which made all table-truth settings appear to tie at 115.765. Revised
  2026-11-11: dropped the "radius 0.5 starves the batch" reasoning — it measured how many points a fixed
  pool could supply, not the effect of the radius.) Defaults:
  `experiments/online-policy-wide.toml` (radius 1.0) with `-narrow.toml` kept as the comparison arm;
  exploit share 0.5 is recorded for T15 (not a parameter of any proposer on `main` yet); the surrogate is
  left to T12. These are defaults, not measured gains.

## Known issues

- `xgboost` results can differ slightly across operating systems and CPUs (floating-point order in the
  library), so `sdm reproduce` may report small mismatches on another machine. Reproduce on the machine
  that made the run; compare across machines with a tolerance. Run provenance records the xgboost version.

- `pgd` returns duplicate rows when `n > 1` (all starts converge to the same point). Each call needs up to
  50 distinct points that also differ from every earlier call. Owner: C. Tracked by the `xfail` in
  `tests/test_plugin_contract.py`.
- The round-1 packet (`analysis/online_round1/delivery/`) has the right 50-point shape again, but call 1 must be
  generated by the frozen policy from the 2048 rows (decision 19), and the packet spends points on known values
  (`r01-046` baseline W\*, `r01-047` = `p01969`). Use it as a design reference, not as a submission. Owner: C.
- The baseline W* is an extrapolation: nearest training row 0.66 away vs. typical spacing 0.38; ridge2
  predicted 75.9 but the real score is 119.09. Treat corner-of-simplex predictions with suspicion.

## Open questions

- Course computer environment (OS, Python, whether uv and `libomp` are available): decision 22 requires
  reproducing there. Seen on 2026-11-02 when submission 1 was scored: Windows, PowerShell, a conda
  environment. The TA ran `solver/run.py` with Python 3.14 (its `__pycache__` is `cpython-314`), while this
  repository pins 3.11 (numpy 1.24.4 has no newer wheels). Whether uv can be installed is unknown. Owner: A.
  2026-11-08: the stdlib-only smoke test ran with Python 3.14.7 on A's Windows PC (see Resolved), so the
  evaluator side works with the course computer's Python; the `sdm` package side is still unverified there.
- How much the policy may be tuned on local evaluator runs. A deterministic policy's local result equals its
  graded result, so picking among policies or seeds by their local 20-call score selects on the test function.
  Ask the TA whether that is allowed (owner: A).
- Schema of the policy's call log and how each call's result is merged into `X, y` (owner: A, after the
  evaluator format is known).

## Resolved

- Evaluator determinism: deterministic (decision 20, 2026-10-31).
- Submitting a training row offline: allowed but only ties the bar (decision 17, 2026-10-31).
- Whether weights must sum to 1: no, non-negative with one positive per building (decision 21, 2026-10-31).
- Evaluator input/output (2026-11-02, from submission 1): `RampingSolver.exe --weights <file>.json` reads a
  3 × 7 nested JSON list (one row per building, `FEATURES` order) and prints R as one float on stdout.
- Online evaluator input/output (2026-11-06, `solver/README.md`): `solver.exe` reads JSON on stdin,
  `{"weights": [...]}` with 1–50 sets, each a 3 × 7 matrix or 21 numbers in building order (a single set may
  also be passed bare). It prints `{"R": [...]}` in input order; on error it writes the reason to stderr and
  exits with code 2. Weights must be finite and non-negative with at least one positive per building.
- Windows smoke test of `solver.exe` (2026-11-08, `analysis/solver_smoke/reports/PC-20261108-115634.json`
  (`--quick`) and `PC-20261108-120254.json` (`--quota`); Windows 11 Pro, Intel 20 threads, Python 3.14.7,
  exe sha256 matches). Every check passes except `readme_example`:
  - Which machine runs development evaluations: A's Windows PC (host `PC`), where the exe runs unmodified.
  - The `solver/README.md` all-ones example is not real: all-ones weights return R 156.24560858344194, not
    114.43. Known points reproduce exactly (`p01969` 115.76481925917324, submission 1 119.71206678869204,
    baseline W\* 119.09161796723492), deterministically and independently of batch order and composition.
  - The evaluator renormalises each building's weights: scaling a building's weights leaves R unchanged
    (consistent with decision 21).
  - Run time per call: 1 set 11.5 s, 10 sets 61.7 s, 50 sets 288.9 s (about 5.8 s per set in a full batch),
    so a full 20 × 50 rehearsal costs about 1.6 h of evaluator time.
  - Quota: counted per call, not per set, in `usage.count` next to the exe; calls 1–20 succeed and call 21
    exits with code 2 and `Usage limit reached (20 batch invocations).` Deleting `usage.count` resets it.
