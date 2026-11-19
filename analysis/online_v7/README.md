# online_v7: keeping the v4 batches, replacing zero-yield slots with multiscale candidates

This directory is one complete, reproducible online search: 20 solver calls x 50 points = 1000 real
evaluations of the course evaluator, best R = **102.8058786104**.

The result, the comparison with the earlier policies and the limitations are in
[REPORT.md](REPORT.md). This file covers what is in the package, how to verify it without the solver
and how to rerun it.

## Headline result (seed 7 only)

| Item | Value |
|---|---|
| Best R | 102.8058786104 (call 20, point 25, `icombo`, building 1) |
| Frozen phase (calls 1-11) | 105.107716483, identical to PR 25 / v4 point by point |
| v4 at 20 calls, same seed | 103.275294878 |
| h6 at 20 calls, same seed | 103.222939235 |
| v4 at 35 calls, same seed | 102.1433224203 |

The experiment was designed after looking at the v4 seed-7 log, and it was then evaluated on the same
seed. **Only seed 7 has been run.** The difference of -0.469416 against v4 must therefore be read as
"on seed 7, the seed used during design", not as an expected gain. Rehearsals on seeds 2026 and 11
(the two other seeds that have v4 results) are still outstanding; until they exist, no average over
seeds can be quoted (decision 32 in PR 20).

## Naming: h5 / h6 / h7 versus the file names

PR 25 already merged a `policy_v5.py` (vertex probing with per-building channels) and a
`policy_v6.py` (channels only for stagnant buildings). The files in this package use the same names
for different policies. The file names are kept because `runs/explore_v7_s7/run_config.json` pins
their sha256, so renaming them would break the record of what produced the run. Documents and
`docs/team.md` use the `h` labels below instead.

| Label | File | Policy |
|---|---|---|
| v4 (= PR 25) | `policy/policy_v4.py` | reference policy, frozen phase of this run |
| h5 | `policy/policy_v5.py` | 15-call accelerated schedule (negative result, best 107.133834) |
| h6 | `policy/policy_v6.py` | multiscale pool from call 12 (negative result, best 103.222939) |
| h7 | `policy/policy_v7.py` | this run: keep every v4 point, replace zero-yield slots |

## Layout

```
analysis/online_v7/
├── README.md                this file
├── REPORT.md                policy, results, comparison
├── QUICK_VERIFY.md          the three zero-solver commands and what each one proves
├── requirements-lock.txt    full environment lock (adds matplotlib for the plots)
├── requirements.txt         runtime dependencies (sha256 pinned in run_config.json)
├── online_run.py            shared runner: data/solver checks, state, screening, audit hook
├── explore_v7.py            the runner that produced the result (sha256 pinned, do not edit)
├── verify_v7.py             zero-solver verification (freeze / replay / audit)
├── plot_v7.py               plots from the archived log
├── policy/
│   ├── policy.py            v1 base policy (shared helpers, features, ZERO/MINOR sets)
│   ├── policy_v4.py         v4 = PR 25 policy, used verbatim for calls 1-11
│   ├── policy_v5.py         h5, 15-call accelerated schedule (not used in this run)
│   ├── policy_v6.py         h6 parent class of the multiscale candidates
│   ├── policy_v7.py         h7, this run
│   └── surrogate.py         GP / Ridge / feature interface
├── data/
│   ├── train.csv            frozen 2048 x 23 training table
│   └── manifest.json        dataset metadata and sha256
├── runs/
│   ├── PROVENANCE.md        how each log was produced, what it supports, what was redacted
│   ├── explore_v7_s7/       this run: config, point log, freeze check, audit, summary
│   ├── pr25_repro/          v4 seed-7 rerun, 20 calls (frozen-phase reference)
│   ├── from_call20/         v4 seed-7 continuation, calls 21-35
│   ├── accelerated15_s7/    h5 log, h5 summary and the solver chain check
│   └── explore_v6_s7/       h6 log, used by the plots only
└── viz_explore_v7/          four figures, per-call table and text summary
```

`runs/` and `analysis/**/results/` are ignored by the repository `.gitignore`. The files above were
force-added because they are finished-run snapshots, not regenerable working directories.

## Fixed conditions

| Item | Value |
|---|---|
| Seed | `7` (all randomness derives from `(seed, call_no)`; `rng_uses_clock = false`) |
| Points per call | `50` |
| Calls | `20` (the solver allows 20) |
| Frozen calls | `1-11`, identical to PR 25 (`max abs dW = 0`) |
| Training table | `data/train.csv`, 2048 x 23, sha256 `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023` |
| Solver | course `solver.exe`, sha256 `293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247` |
| Python | `3.11.18` |
| Runtime packages | numpy 1.24.4 / scipy 1.10.1 / scikit-learn 1.2.2 / joblib 1.6.0 / threadpoolctl 3.7.0 |
| Policy sha256 | see `policy_sha256` in `runs/explore_v7_s7/run_config.json` |
| Runner sha256 | `c0229c16f9421c4912894ee47a44c14f035195dd5e1bd5f346af8838632d4ae7` (`explore_v7.py`) |
| Thread environment | not forced, same default as the reference run (`MKL` / `OMP` / `OPENBLAS` unset) |
| Machine | `Windows-10-10.0.26100-SP0`, 12 logical CPUs, one solver process at a time |

`run_config.json` is written once at the start of the run and refuses to resume if the environment
changed; `audit` re-checks the recorded environment against the current one.

Three files are CRLF and the rest are LF. The root `.gitattributes` sets
`analysis/online_v7/** -text` so that a macOS or Linux checkout keeps the exact bytes, and the sha256
values above match on every platform.

## Verification without the solver

The three verification commands take a few minutes each (the policy refits its surrogate per call);
none of them starts `solver.exe`:

```bash
uv venv --python 3.11
uv pip install -r analysis/online_v7/requirements-lock.txt

uv run python analysis/online_v7/verify_v7.py freeze   # calls 1-11 against the v4 reference log
uv run python analysis/online_v7/verify_v7.py replay   # calls 12-20 against this run's own log
uv run python analysis/online_v7/verify_v7.py audit    # rebuilds the ten audit checks
uv run python analysis/online_v7/plot_v7.py            # regenerates the four figures
```

Results go to `analysis/online_v7/verify_out/` (git-ignored); the committed evidence under `runs/`
is not touched. See [QUICK_VERIFY.md](QUICK_VERIFY.md) for the expected output of each command.

## Rerunning the real search

This consumes 20 calls of a fresh solver quota and takes hours.

1. Put the course `solver.exe` at `analysis/online_v7/solver/solver.exe` and check its sha256
   against the table above. The runner copies it into `explore_v7_solver/` so the main copy's
   `usage.count` is untouched; both paths are git-ignored.
2. `uv run python analysis/online_v7/explore_v7.py verify-freeze` and `dry-run` first (no solver).
3. `uv run python analysis/online_v7/explore_v7.py run`. The log is append-only and the same command
   resumes after an interruption.

`explore_v7.py` is byte-identical to the runner that produced the committed log, which is why
`run_config.json` still matches it. It writes into `runs/explore_v7_s7/`; use `verify_v7.py` if you
only want to check the committed evidence.

## Data boundary

Before call 1 the policy reads only `data/train.csv`; afterwards it uses this run's own results.
Audit checks 2 and 3 scan the policy source and the runtime file accesses for any read of `runs/`,
`reference/`, `.csv` or `.jsonl`. `runs/pr25_repro` is read only by the frozen-phase comparison and
is never passed to the policy (`history_used_for_policy = false`).

## Relationship to the other packages

`data/train.csv`, `policy/policy.py`, `policy/policy_v4.py`, `policy/surrogate.py` and `online_run.py`
are the same source as `analysis/online_rehearsal/package_v4s7` (PR 25) after normalising line
endings; see [runs/PROVENANCE.md](runs/PROVENANCE.md) for the hashes and for the origin of every log.

## Known limitations

- One seed only, and it is the seed the policy was designed on; see the headline section.
- The surrogate model re-optimises its GP hyperparameters on every call. Decision 31 asks the
  submitted policy to use fixed constants, so this has to be addressed when the policy is ported to
  `submission/`; it is not changed here.
- The multiscale channels produced one new best in nine calls; the block and restart candidates
  produced none.
