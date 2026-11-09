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
| calls × batch | 20 × 50 (the online budget: 1000 points) |
| truth sources | `table` (held-out rows scored by lookup) and `semi-synthetic` (`histgb` fitted on all 2048 rows, following `analysis/coupling/`) |
| candidate pool | 1600 rows; the other 448 rows are the policy's known data |
| seeds | 0–4 (policy RNG and model seed); the pool split is per seed |
| policies | exploit share 0.0 / 0.5 / 1.0, trust radius 1.0 (and 0.5 for the strongest surrogate) |
| surrogates | `ridge2`, `histgb` (replay-cheap settings) |
| evaluator calls | none |

The pool is larger than the budget on purpose: with a smaller pool every policy eventually sees every
row. Even so, most policies stop improving inside a few calls, so **the final value is not the
discriminating quantity**. The table therefore also reports the best after the first call (the graded
batch minimum) and the call at which a run stops improving.

## Results

| truth | surrogate | exploit | radius | best after call 1 | median calls to the run's best | mean final R | std |
|---|---|---|---|---|---|---|---|
| semi-synthetic | ridge2 | 0.5 | 1.0 | 128.175 | 0 | 128.175 | 0.000 |
| semi-synthetic | ridge2 | 1.0 | 1.0 | 128.175 | 0 | 128.175 | 0.000 |
| semi-synthetic | histgb | 0.5 | 1.0 | 128.263 | 0 | 128.175 | 0.000 |
| semi-synthetic | histgb | 1.0 | 1.0 | 128.263 | 0 | 128.175 | 0.000 |
| semi-synthetic | histgb | 0.5 | 0.5 | 128.263 | 0 | 128.175 | 0.000 |
| semi-synthetic | ridge2 | 0.0 | 1.0 | 128.371 | 0 | 128.175 | 0.000 |
| semi-synthetic | histgb | 0.0 | 1.0 | 128.371 | 0 | 128.175 | 0.000 |
| table | ridge2 | 0.5 | 1.0 | 115.765 | 1 | 115.765 | 0.000 |
| table | ridge2 | 1.0 | 1.0 | 115.765 | 1 | 115.765 | 0.000 |
| table | histgb | 0.5 | 1.0 | 115.765 | 1 | 115.765 | 0.000 |
| table | histgb | 1.0 | 1.0 | 115.765 | 1 | 115.765 | 0.000 |
| table | ridge2 | 0.0 | 1.0 | 117.437 | 3 | 115.765 | 0.000 |
| table | histgb | 0.0 | 1.0 | 117.437 | 3 | 115.765 | 0.000 |
| table | histgb | 0.5 | 0.5 | 117.437 | 10 | 115.765 | 0.000 |

Mean over 5 seeds; the per-seed curves are in `results/20261109/runs.jsonl`.

Reading:

- **This is a near-null result, and it has to be reported as one.** Under both truth sources every
  setting ends at the same value (table 115.765, semi-synthetic 128.175) with zero spread. Under
  `semi-synthetic` four of five seeds already hold a point at the truth minimum among the 448 known
  rows, so no call can improve on them; only seed 0 needs two calls. Under `table` the optimum is a
  single held-out row, and every setting still finds it inside 20 calls.
- **Exploit share.** 0.5 and 1.0 tie: both have the best point by call 1 in the table truth. Pure
  exploration needs 3 calls (median) and its first batch is worse (117.437 against 115.765). So there
  is evidence against 0.0 and none separating 0.5 from 1.0.
- **Trust radius.** The narrow radius (0.5) needs 10 calls against 1, with the same final value.
  This is the only setting the replay separates cleanly.
- **Surrogate.** `ridge2` and `histgb` are indistinguishable. This replay is **not** a model-selection
  experiment; T12 owns that decision and keeps it on `sdm evaluate`.

## Recommendation

| parameter | value | source |
|---|---|---|
| exploit share | 0.5 | tied with 1.0 in both truths; 0.5 keeps half the batch exploring. Judgment call, not a measured gain |
| trust radius | 1.0 (max distance ratio) | narrow radius is 9 calls slower, final value unchanged |
| batch composition | 25 acquisition + 25 exploration points, no repeats | decision 19 / T15 |
| surrogate | whatever T12 selects | this replay cannot rank models |
| first batch | same settings as later batches | no replay evidence for a special first batch |

Delivered as [`experiments/online-policy-wide.toml`](../../experiments/online-policy-wide.toml) with the
narrow arm kept in `experiments/online-policy-narrow.toml` for re-running the radius comparison. The
explore share is not a parameter of any proposer on `main` yet (T15 is in progress), so the configs
carry the model and optimizer parameters that exist today; the share is recorded in `decisions.md`
and must be added to the proposer when T15 lands.

These are **defaults, not measured gains**. They are consistent with the replay and with the protocol
(no repeats, ≥ 20-point diversity per batch); they are not derived from any real-evaluator number.

## Search history (real evaluator runs, not policy evidence)

Grouped by stage in the order they were run. `evidence/eval_log.jsonl` holds all 6,568 single-call
records; every group is a different search strategy, not a controlled comparison.

| stage (phase prefix) | calls | best R |
|---|---|---|
| screening (`screen`, `vertex`) | 387 | 120.299 |
| `v2`–`v4` grid and line scans | 181 | 106.522 |
| `v5` semi-blind supports | 1,056 | 105.940 |
| `vE0` probes | 11 | 117.802 |
| `vE1` three-building axes | 1,377 | 131.370 |
| `vE2` structured supports | 806 | 105.118 |
| `vE3` terrain and grid | 1,994 | 105.022 |
| `vE4` one-dimensional scans | 535 | 104.852 |
| `vE5` quantisation probes | 221 | 104.852 |

Why this is history and not evidence:

- Sample sizes are very uneven (11–1,994) and the stages are sequential, so each stage starts from
  what the previous one found; a lower best R in a later stage does not mean that strategy is better.
- The runs are not seeded or frozen: `optimize_weights_phaseE*.py` used Python's `random`/NumPy
  defaults in places and the batch splits were chosen by hand. They cannot be replayed exactly.
- The best point (R = 104.851733, eval_id 6335, 4 non-zero slots per building) is a **local-run
  measurement**. It says the response surface has low points of that shape; it does not license a
  first-batch prior, and under decision 19 the policy may not start from it.

## Files

| path | contents |
|---|---|
| `study.py` | the replay: policies, both truth sources, curves, convergence metrics, figures |
| `results/20261109/results.json` | per-setting aggregates and mean curves |
| `results/20261109/runs.jsonl` | one record per run (20 curves, first-call best, call it stops improving) |
| `results/20261109/provenance.json` | git state, dataset sha256, package versions, protocol |
| `results/20261109/figures/` | `replay_table.png`, `replay_semi_synthetic.png`, `calls_to_final_best.png` |
| `t18_strategy_comparison.ipynb` | narrative notebook with saved outputs |
| `evidence/` | the 6,568-call search history (log, stage summaries, best weights, dedupe audit) |
| `figures/` | figures drawn from the search history |
| `harness/` | the four scripts that produced `evidence/` (`SDM_WORKSPACE` overrides the data root) |
| `evidence/provenance.json` | solver sha256, host, seeds, exclusions, timestamp normalisation |

`analysis/**/results/` is gitignored (line 8 of `.gitignore`); the run folder is regenerated by the
command below, and the notebook embeds the table and figures so the PR carries the outcome.

## Reproduce

```bash
uv run python analysis/replay/study.py --seeds 5 --out analysis/replay/results/20261109
```

No evaluator call is made. The run takes about half an hour on one laptop; the pool split and every
policy are seeded, so the numbers reproduce exactly.

## Limits

- The table truth can only propose rows that exist in the table, so its optimum is a single known row;
  the real response surface is continuous. Treat convergence differences as ordinal, not as an
  expected online gain.
- `semi-synthetic` truth is a `histgb` fitted on the same 2048 rows the policy learns from, which makes
  it easier than the real evaluator. The offline findings already show predictions can be several
  points optimistic far from the data.
- 5 seeds is enough to rank settings that differ by several calls, not to resolve ties.
- The near-null result is structural: with 448 known rows the truth is already close to the policy's
  reach, and the table truth has a single optimum. A discriminating replay would need a truth that can
  improve on the known rows over 20 calls (for example a cold start with far fewer known rows, or a
  continuous truth with distance drift). That is a T17 design question, not a T18 claim.
- Nothing here tests the runner, the log, deduplication or the quota (T06–T09), and no cross-machine
  determinism is checked (T09).
