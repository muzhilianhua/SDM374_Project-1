# Solver smoke test (Windows)

`smoke.py` checks the course evaluator `solver/solver.exe` before any online-policy work relies on it.
It uses only the Python standard library, so it runs on a bare Windows Python 3.8+ (the course computer has
3.14); no `uv` and no project dependencies are needed. It never blocks on the 20-call quota: it deletes
`solver/usage.count` before each call and once more at the end.

| Check | Pass condition |
|---|---|
| `exe_sha256` | `solver.exe` matches `solver/CHECKSUMS.sha256` |
| `readme_example` | two all-ones sets in one call → 114.4317428462673 each (the README example) |
| `known_points` | baseline W\* → 119.0916 (±6e-5), training row `p01969` → 115.76481925917324, offline submission 1 → 119.71206678869203 (±1e-6) |
| `deterministic_repeat`, `batch_order_independent`, `batch_composition_independent` | the same point gives bit-identical R when called again, in reverse order, or alone |
| `single_set_formats` | bare 21 numbers, bare 3×7 and `{"weights": 3×7}` give the same R |
| `renormalises_per_building` | info only: whether scaling each building's weights changes R (decision 21) |
| `course_weight_json` | the TA's unnormalised `solver/weight.json` is accepted |
| `rejects_*` | 51 sets, a negative weight, an all-zero building, 20 numbers and non-JSON input → exit code 2, no R |
| `usage_counter` | info only: how `usage.count` changes after valid and invalid calls |
| `timing` | wall time of 1, 10 and 50 distinct points per call (`--quick` runs only 1) |
| `quota_after_20_calls` | info only, with `--quota`: what call 21 returns without a reset |

The report is written to `analysis/solver_smoke/reports/<host>-<time>.json`. It is rewritten after
every check, so a crash still leaves partial evidence, and it records host, CPU, Python and git commit.
`tests/test_solver_smoke.py` runs the script against a fake evaluator on macOS/Linux.

## Run it on Windows

```powershell
git clone https://github.com/muzhilianhua/SDM374_Project-1.git
cd SDM374_Project-1
git switch feat/online
# copy solver.exe from the course solver.zip into .\solver\
python analysis\solver_smoke\smoke.py --quick            # minutes: correctness checks, one 1-set timing
python analysis\solver_smoke\smoke.py --quota            # full: adds 10/50-set timing and the quota probe
```

Then commit the report(s) on a branch from `feat/online` and record the outcome in `docs/decisions.md`
(Resolved, and the open questions on Python version, renormalisation, the 114.43 example and run time).

## Brief for a Claude Code session on the Windows machine

> Work in the SDM374 Project 1 repo on branch `feat/online` (read `AGENTS.md`; you work for 张子木, member A).
> Copy `solver.exe` from the course `solver.zip` into `solver/`, then run
> `python analysis\solver_smoke\smoke.py --quick` and, if it passes, `python analysis\solver_smoke\smoke.py --quota`
> (run in the background; a 50-set call may take long). Do not edit `solver/` or the script's expected
> values. Report every FAIL with its details. Then, on branch `test/solver-smoke-windows` from `feat/online`,
> commit the JSON reports under `analysis/solver_smoke/reports/` and update `docs/decisions.md`: move the
> answered open questions (README 114.43 example, renormalisation, run time per call, which machine runs the
> exe, Python version used) to Resolved with the report file name as evidence. Push the branch and open a PR
> against `feat/online`.
