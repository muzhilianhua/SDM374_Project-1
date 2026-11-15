# online_validation_105 — agent instructions

Self-contained reproduction of an independent 20 × 50 online surrogate-optimisation run.
Human-readable write-up: `REPORT.md`. This file is the machine-facing contract: what to run,
what must be true afterwards, and what is forbidden.

## 1. What this is

A closed loop that starts from `data/train.csv` (2048 rows, 21 weights) only, fits an additive
ARD Matérn-2.5 Gaussian process, proposes 50 new weight sets per round, evaluates them with the
course solver, and refits on all data. 20 rounds → 1000 measured points.

The complete run is committed under `runs/independent_gp_2026/`. Reproduction means producing
an equivalent run, not necessarily bit-identical R values (the solver is deterministic, the
pipeline is seeded, so values should in fact match on the same data).

## 2. Hard constraints

1. **Do not delete, reset, or edit `usage.count` next to `solver.exe`.** The course solver
   enforces 20 calls total; this run already consumed all 20 on the machine that produced it.
   A fresh reproduction consumes a fresh quota, not the committed one.
2. **Do not use any prior search artefact as input**: `best_weights*.json`, `landscape_bundle`,
   earlier ridge / evolution-strategy / random-search points and their R values. They must not be
   seeds, templates, structural priors, tuning input, or reference lines. The only reference value
   is `115.76481925917324`, computed from `data/train.csv`.
3. **Never resend a sent-but-unmeasured batch.** A started solver call is spent even if it fails.
4. **Do not relax the legality checks** (simplex, non-negativity, dedupe at `1e-9`) to fill quotas.
5. Do not claim a target R. The committed outcome is 105.9905; it did not reach 103.46.

## 3. Environment

Recorded in `runs/independent_gp_2026/environment.json`:

- Python 3.13.5 (Anaconda), `numpy 2.1.3`, `scipy 1.15.3`, `scikit-learn 1.6.1`,
  `pandas 2.2.3`, `matplotlib 3.10.0`
- `requirements.txt` pins the same versions
- BLAS/OMP/NumExpr threads are forced to 1 inside `run_online.py` (import side effect);
  keep this, otherwise GP fitting is slower and may differ slightly

## 4. Layout

```
config.json                 all numeric settings (model, candidates, selection, solver)
run_online.py               data loading, GP, candidate generation, selection, solver client, logging
plot_results.py             validation + figures + summary.json; never affects selection
verify_run.py               structural + numeric checks on a finished run; no solver, no quota
test_validation.py          offline unit tests; never starts the real solver
data/train.csv              frozen 2048-row training table (sha256 pinned in config.json)
runs/independent_gp_2026/
  evaluations.csv           1000 rows: one per measured point
  metrics_by_call.csv       20 rows: per-round minima, MAE/RMSE/bias/Spearman/coverage
  calls/call_01..20.json    full per-round record (seeds, model diagnostics, 50 points, R)
  attempts.jsonl            append-only: `started` then `ok`/`failed`, fsynced before the solver start
  summary.json              authoritative numeric result of the run
  best_weights.json         best weights from this run only
  environment.json          interpreter, deps, thread env, script/train/solver hashes, timestamps
  output_manifest.json      sha256 of every generated artefact
  r_descent.png, learning_curve.png, prediction_vs_actual.png
```

Note: the repository `.gitignore` ignores `runs/`, so this run directory is force-added. If you
copy the project elsewhere, copy it verbatim; do not move `runs/` to another name, because
`plot_results.py` resolves `runs/<run_id>`.

## 5. Reproduce

Edit `config.json` → `solver.path`, `solver.inner_path`, `solver.sha256`, `solver.inner_sha256`
for the target machine. Everything else is portable. `inner_path` must point at the
`RampingSolver.exe` embedded in the outer executable (extract it, then verify the hash matches
`e7afe2c2…` below).

Then, in this directory:

```bash
python -m unittest -v test_validation.py        # offline, no solver quota
python run_online.py --synthetic                # 2-round synthetic loop in runs/synthetic_smoke; interface smoke test only
python run_online.py --dry-run --run-id check    # builds round-1 batch, starts no process
python run_online.py --run-id <new_run_id>       # 20 real calls, resumes automatically
python plot_results.py <new_run_id>              # writes summaries + 3 figures
```

`--dry-run` must be deterministic: run it twice and the 50 weights must be identical.

## 6. Verify the committed run (no quota, no solver process)

```bash
python verify_run.py independent_gp_2026
```

It checks, and exits non-zero on the first violation:

- 1000 rows, 20 calls, exactly 50 points per call, no missing/non-finite R
- non-negative weights, every building summing to 1 (`<1e-12`), 1000 unique rows at `1e-9`
- running best non-increasing across calls
- `attempts.jsonl` has exactly 20 `started` and 20 `ok`, no `failed`, no unresolved call
- `calls/call_NN.json`, `attempts.jsonl` and `evaluations.csv` agree on weights, R and predictions
- every file listed in `output_manifest.json` matches its hash, and the manifest covers
  `attempts.jsonl` and every `calls/call_NN.json`
- `summary.json` agrees with `evaluations.csv`
- for the committed run: best R `105.9905449313228` at call 20, point 9

On a reproduction with a different run id, `verify_run.py <run_id>` skips the pinned best value
and checks only the structural invariants.


## 7. Numeric contract (what "equivalent" means)

- best R = `105.9905449313228`, call 20, point 9
- 1000 unique rows at `atol=1e-9`, zero duplicates against `data/train.csv`
- per-round running best must be non-increasing; `metrics_by_call.csv` is authoritative
- calibration coefficient reaches 1.0 from round 5 onward; 95% coverage stays in `[0.92, 1.00]`
- first-round quotas 15/25/10; later rounds 30/10/10 when a new global best appeared within the
  last 3 calls, otherwise 20/20/10

If a reproduction differs, first confirm: same `data/train.csv` hash, same seeds/config, same
solver binary hash, and that no round was resumed twice.

## 8. Pinned hashes

| artefact | sha256 |
|---|---|
| `data/train.csv` | `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023` |
| `config.json` | `12fc246f814c7b736b8440e99295e81c075d5e82382dea28cf4993ec19f71f9e` |
| `run_online.py` | `680ed0f0e2b30eca435227340e222100c484361dcda0052b9653766d448af459` |
| `plot_results.py` | `a4629cf8d2d210b9d4ee365ba712b680377264c3b366cd8b0c8855e4225c4891` |
| outer `solver.exe` | `293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247` |
| inner `RampingSolver.exe` | `e7afe2c252a8c6caed4645ae0893664d134146fac4b2db19a05c7b36828d1112` |

`config.json` is not edited by a reproduction that only changes `solver.*` paths; if you must
change other keys, expect a different run and do not overwrite the committed one.

## 9. Solver transport (the non-obvious part)

The outer `solver.exe` batch entry point delivers empty stdin to its inner solver on some
machines: exit code 2 with `Expecting value: line 1 column 1 (char 0)`. This project therefore
talks to the embedded solver directly:

- process: `RampingSolver.exe --serve --workers 4`
- stdin: one flat 21-element JSON array per line
- stdout: one `{"R": <number>}` line per input, in input order, interleaved with non-JSON log lines
- `parse_solver_output` skips non-JSON lines, rejects non-finite R, and rejects `{"error": ...}`

A failed batch records `stdout`/`stderr`/`returncode` into `attempts.jsonl` and stops the run. To
continue past an unresolved `started` record you must pass `--accept-lost-attempt`; the batch is
then marked lost and never resent. Do not use that flag to paper over a real failure.

## 10. Do not

- Do not regenerate `summary.json`, `best_weights.json`, or the figures from a different run than
  the one in `runs/independent_gp_2026/`.
- Do not add smoothing, post-hoc correction, or a second model to improve the reported curve.
- Do not modify `data/train.csv`; it is the independence guarantee.
- Do not commit `usage.count`, solver workers, or `__pycache__`.
