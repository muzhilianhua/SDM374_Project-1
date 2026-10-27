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
| 6 | Models are compared only with `sdm evaluate`. Primary protocol (2026-10-27): `frozen-v32` — the frozen 2×5 outer splits in `frozen_data/`, ranked by the mean v3.2 score from `evaluation_v3_2/`. The original `kfold` protocol (5 folds, seed 2026, top-50 recall) stays available, and its metrics are also reported under `frozen-v32` | Internal CV used for hyper-parameter choice is optimistic; v3.2 rewards what optimization needs (low-region order, first-choice regret), and member B's earlier results use the same splits |
| 7 | `runs/` is not in git; results are shared by run id or by copying the folder | Run folders hold pickles and are large |
| 8 | The GUI only launches `sdm` CLI commands (background jobs in `runs/.jobs/`) and never computes results in-process | One execution path for humans, agents and graders |
| 9 | Core code and docs in English; GUI text in Chinese | Team convention |
| 10 | `ridge2` + `pgd` must stay equivalent to `baseline.py` (regression tests) | It is the reference every improvement is measured against |
| 11 | `Optimizer.propose` orders rows by the optimizer's own objective (e.g. mean + risk · std), not necessarily by mean | Risk-aware optimizers rank by a pessimistic score (2026-10-25) |
| 12 | Analysis scripts under `analysis/` may call plugins directly if they write `provenance.json` | Factorial studies need many fits that are not individual experiments (2026-10-25) |
| 13 | `frozen_data/` and `evaluation_v3_2/` are read-only releases; `sdm.frozen_v32` reads them in place and refuses to run if their sha256 changes. The v3.2 scale `s` is the population SD of each split's outer-train labels | One copy of the splits and scorer for everyone; reproduces member B's published scores exactly (2026-10-27) |
| 14 | `kfold` and `frozen-v32` results are never ranked against each other; the leaderboard lists `frozen-v32` runs first (by `v32_score`, higher is better), then `kfold` runs (by MAE) | The two protocols use different splits (2026-10-27) |

## Findings

- **Model–optimizer coupling** (`analysis/coupling/README.md`, 5 seeds): which W* an optimizer finds is
  decided mostly by the surrogate's shape, not by the optimizer. On ridge2, pgd and trust-dirichlet land in
  the same extrapolated region (semi-synthetic regret differs by ≤ 0.25); switching the surrogate changes
  regret by 5–25. Whether ridge2's extrapolated W* is good depends on how the real response behaves outside
  the data, which only the evaluator can tell. Recommended offline submissions: one conservative pair
  (extratrees + trust-dirichlet, near the data) and one aggressive pair (ridge2, outside the data).
- Gradient optimizers are incompatible with piecewise-constant surrogates (pgd × extratrees).

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
