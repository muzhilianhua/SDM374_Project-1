# T37 — Real-evaluator rehearsal of a compliant online policy

Owner: member A (张子木). Task: T37 in `docs/online-tasks.md`. Written 2026-11-14.

> **Condition.** Choosing a policy on local `solver.exe` runs relies on decision 27 (PR #20, not merged yet; the user's judgment of
> 2026-11-11, **not yet confirmed by the TA**). Quote every number here with that condition.
>
> **Data boundary.** The policy only ever sees the 2048 training rows and the results of the calls it made in
> the same run (`State` in `policy.py`). It reads nothing from `analysis/lowr_search/` or any other local
> evaluator log: the solver pool runs with this folder's own worker copies and a fresh per-run log, so not even
> its cache holds an earlier measurement. Every structural constant is either generic (step sizes, batch
> split) or computed at run time from the training rows (`TrainPriors`, `train_checks.log`). The
> `lowr_search` findings are used below only to *explain* results, never inside the policy.
>
> **Not the submitted code.** This is an analysis prototype outside `src/sdm`. Using it in the graded run means
> porting it to a `BatchPolicy` (T15, 艾子齐).

## Question

`analysis/lowr_search` reached R = 104.37 with about 15,000 unrestricted evaluations. What does a policy that
obeys the online rules (≤ 20 sequential calls × 50 points, only training rows before call 1) reach on the
real evaluator, and can it get near 104?

## Answer

| Variant | Change | Seed 2026 | Seed 7 | Seed 11 | Mean | Mean best after call 5 / 10 / 15 |
|---|---|---|---|---|---|---|
| v1 | base policy | 107.026 | 107.211 | 106.425 | 106.89 | 109.18 / 107.88 / 107.28 |
| v2 | v1 + shift bias derived from the training rows | 106.951 | 105.107 | 107.490 | 106.52 | 109.51 / 107.50 / 106.69 |
| v3 | v2 + fewer surrogate picks after call 6 | 107.777 | — | — | — | 110.03 / 108.90 / 108.61 |
| **v4** | v2 + null-step filter, late coordinate scans, adaptive shares, no repeats (2026-11-16, `policy_v4.py`) | 108.049 | **103.275** | 105.440 | **105.59** | 108.51 / 107.00 / 105.98 |

Each run is 20 policy batches of at most 50 points, about 40 minutes with 10 parallel solver instances. v1–v3 sent
963–995 points (the screen dropped repeats that the prototype did not refill); v4 sends all 1,000. The local pool
splits each policy batch into smaller worker invocations and resets quota state only inside those private worker
folders; `solver_invocations` in each new `call_done` record reports this development-only detail (the committed
historical logs predate this field). Curves: `convergence.png`.

- All 10 runs beat the best offline result 115.68 in **call 1** (113.2–115.1).
- **v4 is the recommended policy.** Its mean is 0.93 below v2's and it is ahead at calls 5, 10 and 15. Per seed it
  is better for seeds 7 (−1.83) and 11 (−2.05) and worse for seed 2026 (+1.10); with three seeds this is a
  likely, not a proven, improvement.
- v4 seed 7 reached **103.275** within 1,000 compliant points, close to the lowest point `analysis/lowr_search`
  found with about 20,000 unrestricted evaluations (103.247). Its late coordinate scans and same-building
  combinations kept descending through calls 11–20 (105.11 → 103.28), where v1–v3 runs stalled.
- Seed spread is large (v4: 103.3–108.0). Expect about **105.5**, roughly 103–108 depending on where a run lands.
- v3 (one seed) is clearly slower: surrogate picks still matter in the middle calls, which is why v4 keeps a
  model floor of 25 % in its adaptive shares.
- v1/v2 runs ended in different regions at similar R (e.g. building 2 almost all peak or all discomfort); several
  near-equivalent valleys exist, which `analysis/lowr_search` later confirmed (§1.8–§1.11 there).

### Picking the seed is a decision for the user / TA

Policy and evaluator are deterministic, so freezing v4 with seed 7 would reproduce **103.275** bit for bit in
the graded run (v2 seed 7: 105.107). That selects a seed on the test function. Decision 27 allows choosing
hyperparameters on local runs but is not TA-confirmed, and the earlier open question in `docs/decisions.md` named
exactly this case. It also needs the course computer to emit the same batches (T09) and the same `solver.exe`
(decision 30, also in PR #20). The case is recorded under the open question on tuning in `docs/decisions.md`.
Without seed picking, plan for about 105.5 with v4.

## Policy (`policy.py`, `surrogate.py`)

| Part | What it does | Basis |
|---|---|---|
| Features | per building: r = deg/(deg+ramp), log10(deg+ramp share), log10(minor / (deg+ramp)) floored at 1e-6 | training CV R² 0.844 vs 0.775 for raw shares (`train_checks.log`) |
| Navigation surrogate | ridge trend on all data (carries the search out of the training cloud) + ARD-RBF GP on the residuals of the 600 rows nearest the incumbent | — |
| Additivity over buildings | every candidate changes one building, the other two stay at the call's *base*; R − R(base) is that block's delta. The next call combines the improving blocks of different buildings into a new base and a few combinations | training CV R² 0.846 for a building-additive model vs 0.844 unconstrained (`train_checks.log`) |
| Batch of 50 | 1 base + up to 6 combinations; the rest split over the 3 buildings: 50 % surrogate picks by lower confidence bound (mean − std), 25 % unscreened moves at the trust scale, 25 % fine moves (small multiplicative steps, set-to-zero, tiny weights); from call 13 half of the fine slots are one-coordinate log grids | the score is the batch minimum |
| Moves | one weight × 10^N(0, σ); all weights jittered; shift mass toward one objective; zero a minor weight; set a tiny minor/deg+ramp ratio (the 0.1 %–10 % training-weight quantiles, 1.6e-6 … 7.2e-3); zero-coordinate revival uses the same quantiles as an absolute share | `TrainPriors` |
| Trust scale | per building σ × 1.5 on success, × 0.6 on failure, within 0.01–1 | generic |
| Call 1 | start from the best training row (not resent): per-building surrogate candidates + 8 multi-building jumps | — |
| v2 shift bias | the shift target is drawn with probability ∝ max(0, −Spearman(share, R)) + 0.02 on the training rows (ramping 0.43–0.63 in every building); calls 1–6 add 3 unscreened biased shifts per building | training rows only |

## v4 changes (`policy_v4.py`)

All four use only the run's own results.

| Change | What it does | Why |
|---|---|---|
| Null-step filter | a one-coordinate move that returns exactly the base's R marks a plateau; per building, base block and coordinate the widest such interval is kept and no move inside it is proposed | in v1–v3, 30 % of the points in calls 11–20 repeated an R already seen in the run |
| Late coordinate scans (from call 9) | one-coordinate steps just outside the known plateau (smallest effective step first, coordinates rotating), plus the previous call's improving coordinates of the same building applied to the current base (E15-style combinations) | v1–v3 late moves were too fine; one-coordinate scans and their combinations found the late gains in `analysis/lowr_search` (E10, E15) |
| Adaptive shares | per building, slots go to model / random / fine / scan in proportion to their hit rate over the last 4 calls, posterior (hits + 1) / (n + 5), with floors (model 0.25, others 0.10) | model picks hit 9 % of the time but held 42 % of the slots |
| No repeats, no call-1 jumps | candidates that repeat a training row or a sent point are never proposed, so every call sends 50 points; the call-1 multi-building jumps are gone | 5–37 points per run were dropped; the jumps hit 0 times in 56 points |

In v4 the late repeat rate fell from 39 % to 28 % (seed 2026; the rest are repeats of other plateaus than the
base's), and scans were the most productive late move (hits / points: 35/152, 54/178, 26/138).

## What the logs show (the 7 v1–v3 runs, `analyse.py`)

"Hit" = a point below the best R known when its call was sent.

| Kind | Hits / points |
|---|---|
| base, combination | 44/60, 113/154 (73 %) |
| v2 navigation shifts | 64/216 (30 %) |
| fine multiplicative | 198/709 (28 %) |
| set to zero | 66/292 (23 %) |
| tiny weight | 76/466 (16 %) |
| unscreened trust-scale move | 210/1648 (13 %) |
| surrogate pick | 270/2900 (9 %) |
| coordinate grid | 21/360 (6 %) |
| call-1 multi-building jump | 0/56 |

- **Additivity holds online.** Bases and combinations: measured − predicted median +0.013, MAE 0.22 (214 points).
  Advancing three 7-dimensional problems per call works.
- **The surrogate is optimistic.** Call-1 jumps were 9.3–12+ worse than predicted (median +11.8), the same
  failure as the offline GPs. Surrogate picks average +3.6 above their prediction. They still help early and in
  the middle (v3 removed some and was slower).
- **Coordinate grids did little here** (6 %): the tiny-weight windows that paid off in lowr_search sit near
  an incumbent much closer to the optimum than these runs reach.

## Reproducibility

`check_replay.py` re-runs the committed policy on the logged history and compares each proposed batch with the
logged one, point by point (no solver calls). Result in `check_replay.log`: all 20 calls of `v1_s2026`, calls
1, 7, 14, 20 of the other v1–v3 runs and calls 1, 9, 14, 20 of the three v4 runs are identical. `v1_s2026` was run before v2/v3 were added, so this also shows
that adding them left v1 unchanged.

### Verification package for v4 seed 7 (`package_v4s7/`, 2026-11-16)

A standalone folder for other machines (instructions in Chinese in its `README.md`): `online_run.py check`
(versions, logical CPUs, BLAS, sha256 of data, package and `solver.exe`), `verify` (re-computes all 20
batches from the reference history without the solver, about 10 minutes) and `run` (a full sequential
20 × 50 run compared call by call with the reference; resumable; never touches `usage.count`). The policy
files are byte-identical to this folder's at commit b705164 apart from line endings (`provenance.json`);
`.gitattributes` keeps the committed bytes so `CHECKSUMS.sha256` holds after checkout.

On this machine: a fresh `setup.ps1` environment reproduced 20/20 batches, and calls 1–2 of `run` matched
weights and R. **Cross-machine determinism is fragile:** the GP hyperparameter fit amplifies BLAS rounding
into about 0.01 of predicted R, enough to swap a selected candidate. With `OPENBLAS_NUM_THREADS=8` call 3
differs, with `=1` call 20 differs. The package forces `OPENBLAS_CORETYPE=Prescott` (the kernel OpenBLAS
uses here), but the thread count defaults to the logical CPU count (20 here) and cannot be raised above it,
so a machine with another count will probably diverge at some call and end elsewhere in the 103–108 range.
Making the run machine-independent would need a deterministic surrogate fit (fixed or rounded GP
hyperparameters), which changes the trajectory and needs new seed runs.

## Files

| File | Content |
|---|---|
| `policy.py`, `surrogate.py` | the policy (v1–v3) and its surrogate |
| `policy_v4.py` | v4 as a subclass of the v1–v3 policy |
| `rehearse.py` | driver: 20 calls, pre-submit screen (training rows, sent points, 1e-9), log, resume; `--evaluator synthetic` exercises the code without the solver |
| `analyse.py` | summary table, per-run details, `--plot` redraws `convergence.png` |
| `check_replay.py`, `check_replay.log` | batch-by-batch reproduction check |
| `train_checks.py`, `train_checks.log` | additivity and feature checks, `TrainPriors` constants; training rows only |
| `rehearsals/<variant>_s<seed>/` | `online_log.jsonl` (every point: call, weights, R, move, prediction; one summary per call), `config.json`, `console.log` |
| `provenance.json` | git base, dataset and solver sha256, packages, file hashes; `scripts_sha256_lf` is the historical run input and `review_fix_scripts_sha256_lf` records the current bookkeeping fixes |
| `package_v4s7/` | standalone verification package for v4 seed 7 (`check` / `verify` / `run`) |

```powershell
$env:PYTHONUTF8 = 1
uv run python analysis/online_rehearsal/rehearse.py --evaluator synthetic --variant v2 --seed 1 --run dry --calls 2
uv run python analysis/online_rehearsal/rehearse.py --evaluator solver --variant v2 --seed 2026 --run v2_s2026_again
uv run python analysis/online_rehearsal/analyse.py --plot
uv run python analysis/online_rehearsal/check_replay.py v2_s7 1 7 14 20
```

A solver run needs `solver/solver.exe` (checked against the pinned sha256 by `analysis/lowr_search/pool.py`) and
about 40 minutes; it creates an isolated `analysis/online_rehearsal/workers/<run>/` tree (git-ignored) and never touches `solver/usage.count`.

## Limits

- Three seeds for v1, v2 and v4, one for v3; seed spread (up to ±2.4 for v4) is as large as the variant differences.
- Up to 37 of 1,000 points per run were lost to dropped repeats; a port to `BatchPolicy` should rank extra
  candidates so the loop can refill (decision 26).
- One machine, one evaluator build (sha256 `293867b8…a077247`).
- Each variant was chosen after looking at earlier rehearsal results on the same evaluator; that is the
  decision-27 condition above.
