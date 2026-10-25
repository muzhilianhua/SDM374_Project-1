# Decisions and open questions

Settled choices, so nobody (human or agent) re-litigates them. To change one, raise it with member A
and record the new decision here with a date.

## Settled

| # | Decision | Why |
|---|---|---|
| 1 | Python 3.11 + uv; versions pinned to the course baseline (numpy 1.24.4, scikit-learn 1.2.2) | Same numbers on every machine; numpy 1.24 has no 3.12 wheels |
| 2 | Plugins register by decorator and are auto-discovered from `sdm.models` / `sdm.optimizers` | Adding a model or optimizer never edits shared files |
| 3 | `Surrogate.predict` returns `(mean, std)`; `std` is mandatory | Pessimistic offline selection and online exploration both need uncertainty |
| 4 | `Optimizer.propose(model, X, y, n)` takes NumPy arrays, not a DataFrame | Simpler, and online data is appended as rows of `X, y` |
| 5 | Experiments are TOML files in `experiments/`; every result comes from an `sdm` command and lives in a run folder | Reproducible numbers with provenance (`sdm reproduce`) |
| 6 | Models are compared only with `sdm evaluate` (outer 5-fold CV, seed 2026, top-50 recall) | Internal CV used for hyper-parameter choice is optimistic |
| 7 | `runs/` is not in git; results are shared by run id or by copying the folder | Run folders hold pickles and are large |
| 8 | The GUI only launches `sdm` CLI commands (background jobs in `runs/.jobs/`) and never computes results in-process | One execution path for humans, agents and graders |
| 9 | Core code and docs in English; GUI text in Chinese | Team convention |
| 10 | `ridge2` + `pgd` must stay equivalent to `baseline.py` (regression tests) | It is the reference every improvement is measured against |

## Known issues

- `pgd` returns duplicate rows when `n > 1` (all starts converge to the same point). Online rounds need
  distinct points. Owner: C. Tracked by the `xfail` in `tests/test_plugin_contract.py`.
- The baseline W* is an extrapolation: nearest training row 0.66 away vs. typical spacing 0.38; predicted
  75.9 vs. best measured 115.76. Treat corner-of-simplex predictions with suspicion.

## Open questions (resolve in the week 8 lab)

- Evaluator input/output format, run time per point, and whether results are deterministic.
- Whether the evaluator renormalises each building's weights (the slides only require "non-negative,
  at least one positive"; the data package requires sum = 1).
- Whether submitting a row from the training table (e.g. `p01969`, R = 115.76) is allowed offline.
- Schema of `online_log.csv` and how online results are merged into `X, y` (owner: A, after the above).
