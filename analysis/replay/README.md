# T18 — Strategy parameters on the replay (exploratory)

> **Exploratory only. Not an input to the online policy.** The previous version of this folder
> (`online` branch, commits `8b4d67f` / `f47c28e`) ranked strategy families on 6,568 local
> `solver.exe` calls and handed T15 a structural prior (old §6) plus a note that T12 could adopt the
> same ranking directly (old §9). **Both are withdrawn.** Decision 19 allows only the 2048 training
> rows before the first evaluator call, and T03 (whether local evaluator runs may be used to tune
> parameters) is still unanswered, so no number measured with the real evaluator may enter the first
> batch. Everything below comes from the replayer and uses the table only.
>
> The 6,568-call log is kept in `evidence/` as a **search history** (what was tried, in what order).
> It is not evidence about parameter values and is not used in any recommendation.

Owner: member D (胡馨澜). Task: T18 in `docs/online-tasks.md`; the T17 replayer is D's as well.

## Replayer status

The replayer this study runs on is `analysis/replay/study.py`, a prototype of T17. `src/sdm/replay.py`
is not on `main` yet (T06/T08 land first), so this harness re-implements only the loop it needs:
fit → propose a batch of ≤ 50 → score the batch → append → repeat, 20 times. When T17 lands the same
settings will be re-run through `sdm replay` so the numbers quoted in the report come from the shared
tool; the protocol below is what that run has to reproduce.

## Protocol

| item | value |
|---|---|
| calls × batch | 20 × up to 50 (the online budget: 1000 points) |
| truth sources | `table` (held-out rows scored by lookup) and `semi-synthetic` (`histgb` fitted on all 2048 rows, following `analysis/coupling/`) |
| candidate pool | 1600 rows; the other 448 rows are the policy's known data |
| seeds | 0–4 (policy RNG and model seed); the pool split is per seed |
| policies | exploit share 0.0 / 0.5 / 1.0, trust radius 1.0 (and 0.5 for the strongest surrogate) |
| surrogates | `ridge2`, `histgb` (replay-cheap settings) |
| evaluator calls | none |

**Scoring.** Curves record the best R among *submitted points only*. The best of the 448 known rows
(the training incumbent) is reported as a separate reference line and never enters the curve. The
first version of this study took the minimum over known + submitted, so first-call and final scores
quoted a row the policy never submitted; that bug made every table-truth setting appear to tie at
115.765.

**Trust-radius rule (table truth).** A pool row is eligible only within `median spacing × radius` of
the known data. When fewer rows qualify than the batch needs, the batch is simply smaller — the
radius is never widened silently. Every call logs the eligible count, the submitted count and the
actual distance ratios (`runs.jsonl` → `call_log`); the results table carries mean points per call
and the worst observed distance ratio so the constraint is auditable. (The first version filled
short batches with out-of-radius rows — up to 45 of 50 — so the old radius arm never implemented the
stated constraint.)

The pool is larger than the budget on purpose: with a smaller pool every policy eventually sees every
row. Even so, most policies stop improving inside a few calls, so **the final value alone is not the
discriminating quantity**. The table therefore also reports the best after the first call (the graded
batch minimum) and the call at which a run stops improving.

## Results

70 seeded runs — `results/20261109/results.json`, one line per run in `runs.jsonl` (including the
per-call radius audit), curves in `figures/`, numbers below from `summary.md`.

Training incumbent (reference, mean over seeds; never submitted): **table 117.437**, semi-synthetic
128.371.

| truth | surrogate | exploit | radius | best after call 1 | median calls to the run's best | mean final R | std | mean pts/call | max dist ratio |
|---|---|---|---|---|---|---|---|---|---|
| semi-synthetic | ridge2 | 0.5 | 1.0 | 128.183 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | histgb | 0.5 | 1.0 | 128.263 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | histgb | 1.0 | 1.0 | 128.263 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | histgb | 0.5 | 0.5 | 128.263 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | ridge2 | 1.0 | 1.0 | 128.540 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | ridge2 | 0.0 | 1.0 | 128.563 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | histgb | 0.0 | 1.0 | 128.563 | 1 | 128.175 | 0.000 | — | — |
| table | ridge2 | 1.0 | 1.0 | 117.350 | 1 | 117.350 | 1.957 | 27.8 | 1.00 |
| table | histgb | 0.5 | 1.0 | 117.350 | 1 | 117.350 | 1.957 | 27.3 | 1.00 |
| table | histgb | 1.0 | 1.0 | 117.350 | 1 | 117.350 | 1.957 | 26.7 | 1.00 |
| table | ridge2 | 0.5 | 1.0 | 117.507 | 1 | 117.350 | 1.957 | 28.4 | 1.00 |
| table | histgb | 0.5 | 0.5 | 133.697 | 1 | 133.697 | 7.677 | 0.9 | 0.50 |
| table | ridge2 | 0.0 | 1.0 | 126.937 | 5 | 117.350 | 1.957 | 27.6 | 1.00 |
| table | histgb | 0.0 | 1.0 | 126.937 | 5 | 117.350 | 1.957 | 27.6 | 1.00 |

Reading:

- **The two truths tell different stories now.** Under `semi-synthetic` every setting reaches the
  truth minimum (128.175) in one call (median); first-call values span only 128.183–128.563. That
  arm cannot rank settings — it is a sanity check, not evidence.
- **Under `table` the settings separate.** Exploit ≥ 0.5 with radius 1.0 puts each seed's pool
  minimum in the first batch (mean final 117.350 ± 1.957: 115.765 for seeds 0/3/4; 120.119 and
  119.334 for seeds 1/2, where `p01969` sits in the known rows). Pure exploration starts worse
  (126.937 ± 2.833) and needs a median of 5 calls to catch up.
- **Radius 0.5 starves the batch.** With the radius strictly enforced the narrow arm averages 0.9
  submitted points per call (7–34 over the whole run): the in-radius supply of unused pool rows is
  exhausted after the first calls. It ends at 133.697 ± 7.677, 16.3 points worse than radius 1.0.
  The earlier "0.5 reaches the same final, only slower" was an artifact of silently filling batches
  out of radius.
- **Surrogate.** ridge2 and histgb are indistinguishable under both truths. T12 owns model
  selection; this replay does not.
- The training incumbent is reference only. For seeds 1 and 2 it equals 115.765 (`p01969` already
  known), which no submission can beat; the submitted curve then converges to the pool minimum
  instead.

## Recommendation (defaults for T15; the online policy input stays D's)

| parameter | value | source |
|---|---|---|
| exploit share | 0.5 | tied with 1.0 under both truths (final 117.350 already at call 1 under table); 0.5 keeps half the batch exploring. Judgment call, not a measured gain |
| trust radius | 1.0 (max distance ratio) | the narrow radius starves the batch (< 1 point/call) and ends 16 points worse under table truth |
| batch composition | 25 acquisition + 25 exploration points, no repeats | decision 19 / T15 |
| surrogate | whatever T12 selects | this replay cannot rank models |
| first batch | same settings as later batches | the winning settings already put the run's best in batch 1 |

Delivered as `experiments/online-policy-wide.toml` (default) and `experiments/online-policy-narrow.toml`
(comparison arm); the T15 policy itself lands in decision 19.

These are **defaults, not measured gains**: no setting here has been compared against another on the
real objective. If T03 later allows local evaluator runs for tuning, this table must be rebuilt from
that evidence.

## Search history (kept, not used as evidence)

Before the withdrawal above, this folder tuned strategies on 6,568 local `solver.exe` calls and
recommended proposals from that ranking. The log is kept in `evidence/` purely so the team can see
what was tried and in what order. Per decision 19 / T03 it is not used for any parameter value, and
it is not an input to the online policy. Details and checksums: `evidence/README.md`.

## Files

| path | what it is |
|---|---|
| `study.py` | the replayer: policy harness, both truths, seeded runs; `--from-runs` re-aggregates an existing `runs.jsonl` |
| `t18_strategy_comparison.ipynb` | the narrative notebook with the saved figures |
| `results/20261109/` | `results.json` / `provenance.json` / `runs.jsonl` (includes the per-call radius audit) / figures / `summary.md` |
| `evidence/` | the withdrawn 6,568-call search history |

## Reproduce

```bash
uv run python analysis/replay/study.py --seeds 5 --out analysis/replay/results/20261109
```

A full pass takes about fifteen minutes on one laptop. `results/` is gitignored; the notebook embeds
the table and figures so the PR carries the outcome. Raw data, provenance and the aggregation code
are in this folder, and the markdown table above is `results/20261109/summary.md`.

## Limits

- Two truth sources, both offline. Neither is the evaluator, and the semi-synthetic arm cannot
  separate settings at all (everything reaches the truth minimum in one call). The separating
  evidence comes from the table truth, whose optimum is a single row. A truth that keeps improving
  over 20 calls would discriminate further; that is a T17 design question.
- The table pool holds few rows near the known data: even radius 1.0 exhausts its in-radius supply
  after ~550 submissions, so mean points/call is ~27 rather than 50. The real policy proposes points
  from a continuous space and has no such supply limit; the radius comparison therefore overstates
  the narrow setting's failure cost but not its direction.
- When `sdm replay` lands, this run has to be repeated on the shared tool before the numbers go into
  the report; the T06–T09 merge order decides when that happens.
