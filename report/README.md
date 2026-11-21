# SDM374 Project 1 report

IEEE conference paper, at most four pages. Build with a TeX Live installation:

```bash
cd report
latexmk -pdf main.tex
```

## Sections

| File | Content | Main sources |
|---|---|---|
| `sections/introduction.tex` | problem, rules, contributions | `docs/decisions.md` (decisions 17–22) |
| `sections/offline.tex` | frozen-v3.2 model comparison, the two real offline points, submission 2 | `docs/decisions.md` Findings (offline), `analysis/coupling/` |
| `sections/landscape.tex` | why the surrogate misses the low region; structure of R | `analysis/gp_assumptions/` (T35), `analysis/lowr_search/` (T36) |
| `sections/strategy.tex` | additive block descent, v4 slot rules, h7 hybrid | `analysis/online_rehearsal/` (T37), `analysis/online_v7/REPORT.md` |
| `sections/online_results.tex` | rehearsal table and figures | rehearsal logs (see below) |
| `sections/runner.tex` | graded package: determinism, checkpoints, budget | `submission/README.md`, decision 31 |

Red `\todo{...}` marks are numbers still to come (h7 seeds 2026 and 11); none may remain at submission. The
experiment that produces them, and how to run it on Windows, is in [`pending-experiments.md`](pending-experiments.md).

## Figures

`figures/make_figures.py` rebuilds `figures/convergence.pdf` and `figures/h7_batches.pdf` from the committed
real-simulator logs, without the solver:

```bash
uv run python report/figures/make_figures.py
```

The submitted run's log lives in `analysis/online_v7/runs/explore_v7_s7/online_log.jsonl` (PR #29 / #32). Until that
folder is on this branch, extract it and pass `--h7-log`:

```bash
git show origin/feat/submission-v7s7:analysis/online_v7/runs/explore_v7_s7/online_log.jsonl > /tmp/h7_s7.jsonl
uv run python report/figures/make_figures.py --h7-log /tmp/h7_s7.jsonl
```

The figure PDFs are kept in git (see `.gitignore`); other build products are not.
