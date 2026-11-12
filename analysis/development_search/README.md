# Bounded development search

This is a local, warm-started investigation of the basin in D's historical
`analysis/replay/evidence/eval_log.jsonl`. It is not a graded online policy:
those historical observations are unavailable at the start of a graded run.
No historical weights or scores are embedded in policy code or configuration.

The candidate pool mixes simplex-preserving transfers on the current support
with transfers that activate new coordinates, at seven scales from 0.0001 to
0.04. Some transfers change several buildings together. An ExtraTrees model
fits the 1,800 nearest historical points; each batch combines low predicted
scores, a tree-disagreement heuristic, and random probes. Tree disagreement
is not calibrated predictive uncertainty. Only returned solver scores count
as evidence of improvement.

Run from the repository root after activating the project environment:

```powershell
python -u analysis/development_search/study.py `
  --exe .tools/development-solver-20261112/solver.exe `
  --out analysis/development_search/results/20261112-local-search `
  --calls 4 --batch-size 50 --seed 20261112
```

Use an independent development copy of the pinned course evaluator. The script
never resets its counter, never automatically retries a failed/uncertain call,
and refuses to overwrite an existing output directory. Each pending attempt
is flushed before launching the evaluator. Results contain weights, returned
scores, timing, source/data/executable hashes, packages, seed and git state.
An interrupted attempt requires manual reconciliation before further work.

The development quota is separate from the official run, whose proposals must
start with only the 2,048 training rows and obey the team's graded-call rules.
