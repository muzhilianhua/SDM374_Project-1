# Quick verification (no solver, no quota)

These commands check the committed evidence in a couple of minutes. None of them starts
`solver.exe`, and none of them writes into `runs/`; output goes to `verify_out/`, which is ignored by
git.

## Setup

```bash
uv venv --python 3.11
uv pip install -r analysis/online_v7/requirements-lock.txt
```

## 1. Frozen phase versus PR 25

```bash
uv run python analysis/online_v7/verify_v7.py freeze
```

Expected: eleven lines ending in `PASS`, each with `max|dW|=0.00e+00`, then `freeze: PASS`.
It rebuilds calls 1-11 from the reference history and compares each batch with
`runs/pr25_repro/online_log.jsonl` as a multiset (call 3 is stored in a different order).

## 2. The nine hybrid calls

```bash
uv run python analysis/online_v7/verify_v7.py replay
```

Expected: nine lines (`call 12` to `call 20`) ending in `PASS` with `max|dW|=0.00e+00`, then
`replay: PASS`. Batches are compared as multisets with tolerance `1e-14`; on the machine that
produced the log the match is exact. This recomputes each batch from the history recorded in
`runs/explore_v7_s7/online_log.jsonl` and compares it with what was submitted. Unlike audit check 4
(which reads the flag stored during the run), this is an independent recomputation.

## 3. Audit

```bash
uv run python analysis/online_v7/verify_v7.py audit
```

Expected:

```json
{"complete":true,"overall_pass":true,"report":"verify_out/audit_report.json"}
```

The report contains the same ten checks as `runs/explore_v7_s7/audit_report.json`, rebuilt from the
current tree, with repository-relative paths.

## 4. Figures

```bash
uv run python analysis/online_v7/plot_v7.py
```

Writes `verify_out/viz/01_R_curve.png`, `02_learning_curve.png`, `03_step_evolution.png`,
`04_structure.png`, `per_call.csv` and `summary.txt`. Add `--write` to overwrite the committed copies
in `viz_explore_v7/` instead.

## What is not verified here

The real run itself: 20 calls of the course solver. That needs `analysis/online_v7/solver/solver.exe`
with sha256 `293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247`, a fresh copy for the
`usage.count` counter and several hours; see [README.md](README.md). The two verification commands
above are the part that can be checked on any machine.
