# SDM374 Project 1 — online submission (policy v4, seed 7)

This folder is the frozen online run (decision 31). `run.py` is the course `solver/run.py` with its two todos
filled in: it calls `solver.exe` at most 20 times, one after the other, piping `weight.json` into it through
PowerShell exactly as the template does; after each call it reads the R values, refits the surrogate on the
training rows plus this run's results, and writes the next `weight.json` (at most 50 weight sets).

| File | Content |
|---|---|
| `run.py` | the 20-call loop, pre-submit check, checkpoints, resume; `check`, `verify`, `compare` |
| `policy.py` | batch policy v4: additive block descent per building, plateau filter, late coordinate scans |
| `surrogate.py` | navigation surrogate: ridge trend + local Gaussian process on the residuals |
| `config.json` | constants only (policy parameters, seed 7, 20 calls × 50 sets, solver timeout), each with its source |
| `train.csv` | the 2048 course training rows, the only data file |
| `pyproject.toml`, `uv.lock` | Python 3.11, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.2.2 (exact versions) |

Pinned values, checked by `run.py`:

- `train.csv` sha256 `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023` (a mismatch stops the run).
- course `solver.exe` sha256 `293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247` (`solver/CHECKSUMS.sha256`;
  a mismatch prints a warning and is recorded in every checkpoint, but the run continues: the course computer's
  exe is the one that is graded).

## Data boundary

Before call 1 the policy reads only `train.csv`; afterwards only the results of this run. No point, R value or
weight measured on a local evaluator is in this folder; `config.json` holds constants chosen on local
rehearsals (decisions 27 and 32), and they are never optimised at run time. Every call refits from scratch.

## Graded run

1. Put the course `solver.exe` next to `run.py` (or name it with `--solver`). **Never delete or edit the
   `usage.count` next to the graded copy**; `run.py` never touches it.
2. Install [uv](https://docs.astral.sh/uv/) if needed, then in this folder:

   ```powershell
   uv run run.py
   ```

   uv creates a Python 3.11 environment from `uv.lock` on first use. The run prints one line per call
   (`[k/20] batch min … best …`), takes about 2–4 hours (a 50-set call takes 5–11 minutes of solver time plus
   10–25 s of refitting), and leaves everything under `checkpoints/`.
3. If it is interrupted, run the same command again: it resumes after the last complete call. A call that had
   started but has no result counts as used and is never resent (decision 28).

Before call 1, `run.py` sends one deliberately invalid set (a negative weight). The solver rejects it without
spending a call; the run starts only if the rejection message shows that the input reached `solver.exe`
(issue #28), and stops if the copy's quota is already used up.

Exit codes of `solver.exe`: 0 with R values is a completed call. 2 means the solver refused the batch before
evaluating it (invalid input, empty stdin or `Usage limit reached`): the batch is moved to
`checkpoints/call_<kk>.rejected.json`, the call is not counted and the run stops; a person decides whether to
send again with `uv run run.py --after-rejection`. Anything else after the solver started (another exit code,
the timeout, a crash, output without R) is a batch sent without R: it is recorded, counted and never resent,
and the run goes on with the next call.

## Development rehearsals (not the graded copy)

Keep a development copy of `solver.exe` in its own folder outside this folder and outside the repository. On
that copy only, deleting its `usage.count` between rehearsals resets the 20-call quota.

```powershell
uv run run.py check --solver D:\sdm374-dev-solver\solver.exe
uv run run.py --solver D:\sdm374-dev-solver\solver.exe --out runs\rehearsal-1
uv run run.py --evaluator synthetic --out runs\synthetic --max-calls 2     # no solver, closed-form stand-in
uv run run.py --evaluator mymodule:MyTruth --out runs\replay              # development hook (T17, T18)
```

## Checking that a machine reproduces the batches

The policy is deterministic, but linear-algebra rounding depends on the CPU and the OpenBLAS thread count, and
a difference in one call changes every later batch (decision 31, Determinism). `run.py` forces the generic
OpenBLAS kernels (`OPENBLAS_CORETYPE=Prescott`) and reports the BLAS backend at start.

- `uv run run.py verify --reference <online_log.jsonl>` recomputes each call's batch from a recorded run's
  earlier results and compares it point by point, without calling the solver. The reference rehearsal is
  `analysis/online_rehearsal/package_v4s7/reference/v4_s7_online_log.jsonl` in the repository (outside this
  folder; it is never read by the run).
- `uv run run.py compare runs\a runs\b` compares two runs' canonical records (weights, R, policy and RNG state,
  floats written with `float.hex`) and names the first call and field that differ.

If batches differ, try in order: `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to 1.

## Checkpoints

- `checkpoints/run.json`, written before call 1: sha256 of `config.json`, `train.csv`, `run.py`, `policy.py`,
  `surrogate.py` and `solver.exe`. A resume refuses to continue if any of the first five changed.
- `checkpoints/call_<kk>.sent.json`, written before `solver.exe` starts: from then on the batch counts as sent.
- `checkpoints/call_<kk>/`: `weight.json` as sent, the solver's `stdout.txt` and `stderr.txt`, `result.json`
  (weights, R, move metadata), `policy_state.json` (seed, RNG key, parameters, diagnostics), `inputs.json`
  (train.csv and solver.exe hashes), `canonical.json`, and `manifest.json` with the sha256 of every file and of
  the previous manifest (the first link is `run.json`). It is written as `call_<kk>.tmp/` and renamed when
  complete. A resume stops and names the folder if an earlier checkpoint fails its manifest or its link.

## Expected result

The reference rehearsal of this exact policy and seed on the local `solver.exe` (Windows 11, 20 logical CPUs)
reached **R = 103.275** after 20 calls; other seeds of the same policy ended between 103 and 108, so the
expected score is that spread (decision 32). The graded run reproduces 103.275 only if the course computer
emits the same batches.
