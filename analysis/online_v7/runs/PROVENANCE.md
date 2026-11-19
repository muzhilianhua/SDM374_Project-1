# Provenance of the files under `runs/`

`runs/` is listed in the repository `.gitignore`; every file below was force-added because it is
an immutable snapshot of a finished run, not a regenerable temporary directory. No file here was
produced by a later analysis step, and the verification commands (`verify_v7.py`, `plot_v7.py`)
never write into this directory.

## Summary

| Path | Produced by | Run | Size | What it supports |
|---|---|---|---:|---|
| `explore_v7_s7/` | `explore_v7.py run` (this package, seed 7) | 20 calls / 1000 points, 2026-11-18, Windows 10.0.26100, 12 logical CPUs | 683 KB | The v7 result: best R = 102.8058786104, plus `run_config.json`, `freeze_check.json`, `summary.json`, `audit_report.json` |
| `pr25_repro/` | `online_run.py run` with the files of `analysis/online_rehearsal/package_v4s7` | 20 calls / 1000 points, v4 seed 7, 2026-11-17 | 519 KB | The frozen phase 1-11 of v7 equals v4 point by point; the v4 comparison at 20 calls (103.275295) |
| `from_call20/` | `followup_from_call20.py`, continuing the 20-call v4 state | calls 21-35 / 749 points, v4 seed 7, 2026-11-17 | 418 KB | The long-budget v4 numbers quoted in REPORT section 5.2 (calls 21, 22 and 35) |
| `accelerated15_s7/` | `accelerated_15.py` (h5 accelerated policy) | 15 calls / 750 points, seed 7 | 527 KB | The h5 negative result (best 107.133834) and `chain_check.jsonl`, the two identical solver calls that pin the solver transport |
| `explore_v6_s7/` | `explore_v6.py` (h6 multiscale policy) | 20 calls / 1000 points, seed 7 | 656 KB | The h6 comparison at 20 calls (103.222939); read by `plot_v7.py` for the reference curve only |

All five logs are append-only `online_log.jsonl` files written by the runner that produced them;
`explore_v7_s7` additionally carries the configuration and audit files emitted at the end of the run.

## Redactions applied when committing

Two committed logs contained absolute paths from the machine that produced them. The path strings
were replaced with package-relative paths; no weight, R value, timestamp or hash of measured data was
altered.

| File | Field | Before | After |
|---|---|---|---|
| `runs/pr25_repro/online_log.jsonl` (line 1) | `solver` | `C:\\Users\\...\\pr25_repro_run\\solver\\solver.exe` | `solver/solver.exe` |
| `runs/from_call20/online_log.jsonl` (line 1) | `solver` | `C:\\Users\\...\\followup_solver\\solver.exe` | `solver/solver.exe` |
| `runs/from_call20/online_log.jsonl` (line 1) | `parent_log` | `C:\\Users\\...\\runs\\pr25_repro\\online_log.jsonl` | `runs/pr25_repro/online_log.jsonl` |
| `runs/from_call20/online_log.jsonl` (line 1) | `parent_sha256` | sha256 of the pre-redaction parent log | sha256 of the committed parent log (`027c37b2...`), flagged by `parent_sha256_note` |
| `runs/explore_v7_s7/audit_report.json` | `checks.9_freeze_and_solver.freeze_check` | absolute path | `runs/explore_v7_s7/freeze_check.json` |
| `runs/explore_v7_s7/summary.json` | `policy` | `v6-multiscale` (label written by a string constant in the runner) | `v7-hybrid` (matches `run_config.json`; no measured value changed) |

Because `parent_sha256` changed, the `from_call20` log no longer byte-matches its pre-redaction
original. Everything it records about the search itself is unchanged, and the original file is still
available on the machine that ran it.

## Relationship to `analysis/online_rehearsal/package_v4s7` (PR 25)

Five files here are the same source as the merged PR 25 package, only with different line endings
(this package preserves the CRLF bytes that the runner hashed). After converting CRLF to LF they are
byte-identical to the PR 25 copies:

| File | LF-normalised sha256 | PR 25 copy |
|---|---|---|
| `data/train.csv` | `ec3e13a167502592...` | `analysis/online_rehearsal/package_v4s7/data/train.csv` |
| `policy/policy.py` | `ef0f2d8df376691b...` | `analysis/online_rehearsal/package_v4s7/policy/policy.py` |
| `policy/policy_v4.py` | `14782f21a7c2456e...` | `analysis/online_rehearsal/package_v4s7/policy/policy_v4.py` |
| `policy/surrogate.py` | `4c02dace00279d60...` | `analysis/online_rehearsal/package_v4s7/policy/surrogate.py` |
| `online_run.py` | `b926c5c009010e3a...` | `analysis/online_rehearsal/package_v4s7/online_run.py` |

They are duplicated here instead of imported so that the whole v7 package stays self-contained and
the sha256 values pinned in `run_config.json` keep pointing at files inside this directory.
`policy_v5.py`, `policy_v6.py`, `policy_v7.py`, `explore_v7.py`, `plot_v7.py` and `verify_v7.py`
exist only in this package. Note that `policy_v5.py` and `policy_v6.py` are unrelated to the
same-named files of PR 25; see the naming table in the README.
