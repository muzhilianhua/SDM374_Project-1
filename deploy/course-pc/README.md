# Course-computer kit (T10, temporary version)

An offline kit for running the online loop on the course computer (Windows 10, unknown Python, maybe no
internet). It brings its own Python 3.11 (python-build-standalone) and every dependency as Windows wheels,
so nothing is downloaded on the course computer and the system Python is not used. Built by
`deploy/course-pc/build_kit.sh`; `repo/VERSION` holds the commit it was built from.

This is the first, temporary version for the class-9 test on 2026-11-12. Its policy is a stand-in until
T12 (model), T15 (batch policy) and T18 (parameters) land: `online-gp-wide` copies the T18 model and
optimizer (additive ARD GP, pure-exploit pool, wide trust region) into the T08 `surrogate` policy;
`online-baseline` (ridge2 + trust-dirichlet) is the fast fallback.

## Layout

```
sdm374-course-pc-<commit>/
  00_probe.bat … 05_resume.bat, settings.bat, _env.bat
  python/    bundled Python 3.11 (python.exe)
  wheels/    dependency wheels for win_amd64, cp311
  repo/      the repository at <commit> (no .git), with VERSION
  results/   everything the steps write
```

Unzip to any folder (Chinese characters and spaces in the path are fine: the code runs with
`PYTHONPATH=repo\src`, nothing is installed in editable mode). Double-click the steps in order.

| Step | What it does | Calls solver.exe |
|---|---|---|
| `00_probe.bat` | Records OS, CPU, memory, Python/uv/conda/git, internet access, execution policy, the kit path, and every `solver.exe` it finds (sha256 against `CHECKSUMS.sha256`, whether `usage.count` exists; the file is never opened) → `results/probe-*.json` | no |
| `01_setup.bat` | Installs the wheels into the bundled Python offline, then `sdm check-data` and `sdm list` | no |
| `02_selftest.bat` | `pytest`; 3 online calls against the semi-synthetic truth; first-batch `--dry-run` of `online-baseline` and `online-gp-wide`; surrogate fit timing and prediction fingerprints (`bench_models.py`) | no |
| `settings.bat` | Edit: `SOLVER_EXE` (path to the course `solver.exe`; empty = `repo\solver\solver.exe`), `ONLINE_CONFIG`, `MAX_CALLS` | — |
| `03_run_online.bat` | The real online run: `sdm online -c <config>`, asks for `YES` first (`set SDM_NO_CONFIRM=1` skips the prompt) | **yes, every call uses quota** |
| `05_resume.bat [run-id]` | Continues the latest (or given) online run after an interruption, never resending a sent batch | **yes** |
| `04_collect.bat` | Zips `results/` into `results-<computer>-<time>.zip` to bring back | no |

The run refuses a `solver.exe` whose sha256 differs from `repo/solver/CHECKSUMS.sha256`; the probe shows
whether the course copy matches before anything runs.

## Script form (`run.py`)

`build_kit.sh ... script` builds `sdm374-course-pc-script-<commit>/` with a single `run.py` in place of the
.bat files, shaped like the course's `solver/run.py` (20 calls to `solver.exe`) with its todo filled by
`sdm online`. It needs only the standard library, so any Python on the course computer (3.14 included)
can start it; the online program runs in the bundled Python 3.11. Its default config is `online-histgb`
(histgb surrogate + the same pure-exploit pool, about 2 s to refit per call against the GP's 2.5 min).

| Command | Same as | Calls solver.exe |
|---|---|---|
| `python run.py check` | `00_probe` + `01_setup` + `02_selftest` | no |
| `python run.py probe` / `setup` / `selftest` | one of the three | no |
| `python run.py [--solver PATH] [--max-calls N]` | `03_run_online` (asks for `YES`) | **yes** |
| `python run.py --yes` | unattended graded-run form | **yes** |
| `python run.py resume [RUN]` | `05_resume` | **yes** |
| `python run.py collect` | `04_collect` | no |

`solver.exe` is looked up in this order: `--solver`, `SOLVER_EXE` at the top of `run.py`, next to `run.py`,
`repo/solver/`. Every step's output is also written to `results/*.log`.

## What to bring back

The zip from `04_collect.bat`. Compare on a development machine:
- `probe-*.json`: environment facts for T03/T10 and `docs/decisions.md`.
- `bench-*.json`: fit times (the per-call cost of each surrogate on that CPU) and prediction sha256s;
  equal sha256s on two machines mean the model is bitwise deterministic across them (T09).
- `runs/*online-*-dry-run*` or the dry-run folders' `calls/call_01_dry_run.json`: the first batch of
  each config, to compare point by point with the same dry run on our machines (T09, T21).
- `runs/*-online-<config>/`: the real run (`online_log.csv`, `attempts.jsonl`, `predictions.csv`).
