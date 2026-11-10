# Online batch optimizer replay

This study runs the actual `online-batch` optimizer through `sdm.replay.run_replay`.
It complements member D's table-pool study by proposing new weights in the
continuous simplex. It compares the configured policy with two one-change arms:
100% exploitation and a fixed radius. Each arm sees the same initial weights
and labels and uses the same optimizer seeds. The default budget is 20 batches
of 50 weights for each trajectory.

Extra Trees and histogram gradient boosting play the role of the evaluator.
Each is fitted on the 2,048 training rows. The policy starts with those same
2,048 weights, relabeled by the oracle so that initial and replay scores have
the same meaning. The policy never sees the oracle or its original labels.
The study does not call the course solver or read earlier solver runs.
The results measure behavior on these two models and cannot establish the
best setting for the official run.

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
conda run --no-capture-output -n sdm374 python analysis/optimizer_replay/study.py
```

Use `--calls 2 --batch-size 8 --seeds 2026` for a smoke run. For each trajectory,
the output records the proposed weights, predicted and oracle values, and
trust region state after each batch. `summary.json` separates the first batch
minimum and best new evaluation from the incumbent, which also counts the
initial labels. `provenance.json` records the commit, dataset hash, dependencies,
seeds and settings.

The replay API can also accept a table lookup callback paired with a policy
that proposes from the holdout pool. T08 can reuse the replay's in-memory
steps once its evaluator, logging and resume contract are ready.
