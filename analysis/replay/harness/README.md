# Evaluator harness (development runs)

These are the scripts that produced `../evidence/eval_log.jsonl` (6,568 single-point evaluations).

They drive the course evaluator (`solver/solver.exe`) directly and write to a local `optimization/`
folder. They are kept for provenance, not for scoring:

- `optimize_weights_phaseE.py` — base loop: batch construction, append-only log, disk guard, process
  cleanup, uniform-weight sanity check.
- `optimize_weights_phaseE2.py` — support-set scan and one-dimensional `r = F5/(F5+F6)` line scan.
- `optimize_weights_phaseE5.py` — sawtooth search around the current best; the run that produced the
  final log entries.
- `solver_robust.py` — process wrapper: timeout, kill process tree, rebuild the pool on failure.

Run them from a workspace that contains `solver/solver.exe`:

```bash
export SDM_WORKSPACE=/path/to/workspace   # defaults to this folder's parent
python analysis/replay/harness/optimize_weights_phaseE5.py
```

Excluded from git on purpose: `solver.exe`, `usage.count`, `_MEI*` temp folders and any file that
would let someone spend the graded calls by accident.

**These runs are exploration, not scoring.** `usage.count` was reset between development batches, so
the log does not correspond to the official 20-call record. See `../README.md` for how the log may and
may not be used.
