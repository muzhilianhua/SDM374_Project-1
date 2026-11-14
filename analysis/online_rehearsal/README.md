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
| v2 | v1 + shift bias derived from the training rows | 106.951 | **105.107** | 107.490 | **106.52** | 109.51 / 107.50 / 106.69 |
| v3 | v2 + fewer surrogate picks after call 6 | 107.777 | — | — | — | 110.03 / 108.90 / 108.61 |

Each run is 20 calls of 50 points (963–995 points were sent: the screen drops proposals that repeat a sent point and the prototype does not refill them), about 40 minutes
with 10 parallel solver instances. Curves: `convergence.png`.

- All 7 runs beat the best offline result 115.68 in **call 1** (114.5–115.1).
- Expect **106.5 ± 1**; about 105 in a lucky run. **104 is not reachable** within 1,000 compliant points.
  Most runs settle at 106.4–107.5 after call 13–15, which is the level of the degradation + ramping face
  (lowr_search E6: 107–108). Going lower needs narrow minor-weight windows that lowr_search found with about
  8,000 targeted evaluations.
- Seed spread (about ±1) is larger than the v1/v2 difference (0.37 on the mean); three seeds cannot rank them.
  v2 is not worse and has the better best case, so it is the recommended starting point. v3 (one seed) is
  clearly slower: surrogate picks still matter in the middle calls.
- v2 seed 7 is the outlier: it reached a better region in calls 7–9 (107.0 → 105.75). Its final point uses
  ramping as the main weight in every building; v1/v2 seed 2026 ended in regions where building 2 is almost
  all peak (0.98) or all discomfort (0.90) at nearly the same R, so several equivalent plateaus exist near 107.

### Picking the seed is a decision for the user / TA

Policy and evaluator are deterministic, so freezing v2 with seed 7 would reproduce **105.107** bit for bit in
the graded run. That selects a seed on the test function. Decision 27 allows choosing hyperparameters on local
runs but is not TA-confirmed, and the earlier open question in `docs/decisions.md` named exactly this case. It
also needs the course computer to emit the same batches (T09) and the same `solver.exe` (decision 30, also in PR #20). The case is
recorded under the open question on tuning in `docs/decisions.md`. Without seed picking, plan for about 106.5.

## Policy (`policy.py`, `surrogate.py`)

| Part | What it does | Basis |
|---|---|---|
| Features | per building: r = deg/(deg+ramp), log10(deg+ramp share), log10(minor / (deg+ramp)) floored at 1e-6 | training CV R² 0.844 vs 0.775 for raw shares (`train_checks.log`) |
| Navigation surrogate | ridge trend on all data (carries the search out of the training cloud) + ARD-RBF GP on the residuals of the 600 rows nearest the incumbent | — |
| Additivity over buildings | every candidate changes one building, the other two stay at the call's *base*; R − R(base) is that block's delta. The next call combines the improving blocks of different buildings into a new base and a few combinations | training CV R² 0.846 for a building-additive model vs 0.844 unconstrained (`train_checks.log`) |
| Batch of 50 | 1 base + up to 6 combinations; the rest split over the 3 buildings: 50 % surrogate picks by lower confidence bound (mean − std), 25 % unscreened moves at the trust scale, 25 % fine moves (small multiplicative steps, set-to-zero, tiny weights); from call 13 half of the fine slots are one-coordinate log grids | the score is the batch minimum |
| Moves | one weight × 10^N(0, σ); all weights jittered; shift mass toward one objective; zero a minor weight; set a tiny weight (range = 0.1 %–10 % quantile of the training weights, 1.6e-6 … 7.2e-3) | `TrainPriors` |
| Trust scale | per building σ × 1.5 on success, × 0.6 on failure, within 0.01–1 | generic |
| Call 1 | start from the best training row (not resent): per-building surrogate candidates + 8 multi-building jumps | — |
| v2 shift bias | the shift target is drawn with probability ∝ max(0, −Spearman(share, R)) + 0.02 on the training rows (ramping 0.43–0.63 in every building); calls 1–6 add 3 unscreened biased shifts per building | training rows only |

## What the logs show (all 7 runs, `analyse.py`)

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
logged one, point by point (no solver calls). Result in `check_replay.log`: all 20 calls of `v1_s2026` and calls
1, 7, 14, 20 of the other six runs are identical. `v1_s2026` was run before v2/v3 were added, so this also shows
that adding them left v1 unchanged.

## Files

| File | Content |
|---|---|
| `policy.py`, `surrogate.py` | the policy and its surrogate |
| `rehearse.py` | driver: 20 calls, pre-submit screen (training rows, sent points, 1e-9), log, resume; `--evaluator synthetic` exercises the code without the solver |
| `analyse.py` | summary table, per-run details, `--plot` redraws `convergence.png` |
| `check_replay.py`, `check_replay.log` | batch-by-batch reproduction check |
| `train_checks.py`, `train_checks.log` | additivity and feature checks, `TrainPriors` constants; training rows only |
| `rehearsals/<variant>_s<seed>/` | `online_log.jsonl` (every point: call, weights, R, move, prediction; one summary per call), `config.json`, `console.log` |
| `provenance.json` | git base, dataset and solver sha256, packages, file hashes |

```powershell
$env:PYTHONUTF8 = 1
uv run python analysis/online_rehearsal/rehearse.py --evaluator synthetic --variant v2 --seed 1 --run dry --calls 2
uv run python analysis/online_rehearsal/rehearse.py --evaluator solver --variant v2 --seed 2026 --run v2_s2026_again
uv run python analysis/online_rehearsal/analyse.py --plot
uv run python analysis/online_rehearsal/check_replay.py v2_s7 1 7 14 20
```

A solver run needs `solver/solver.exe` (checked against the pinned sha256 by `analysis/lowr_search/pool.py`) and
about 40 minutes; it creates `analysis/online_rehearsal/workers/` (git-ignored) and never touches `solver/usage.count`.

## Limits

- Three seeds for v1 and v2, one for v3; the variant ranking is not significant.
- Up to 37 of 1,000 points per run were lost to dropped repeats; a port to `BatchPolicy` should rank extra
  candidates so the loop can refill (decision 26).
- One machine, one evaluator build (sha256 `293867b8…a077247`).
- Each variant was chosen after looking at earlier rehearsal results on the same evaluator; that is the
  decision-27 condition above.
