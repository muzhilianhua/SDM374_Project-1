# SDM374 Project 1 report

IEEE conference paper, at most four pages. Build with a TeX Live installation:

```bash
cd report
latexmk -pdf main.tex
```

## Sections

| File | Content | Main sources |
|---|---|---|
| `sections/introduction.tex` | setting, difficulty, the two contributions | `docs/decisions.md` (decisions 17–22) |
| `sections/problem.tex` | problem, why surrogates fail away from the data (Table I), structure of R | `analysis/coupling/`, `analysis/gp_assumptions/` (T35), `analysis/lowr_search/` (T36) |
| `sections/method.tex` | additive block descent, evidence-driven allocation, Algorithm 1, deployment | `analysis/online_rehearsal/policy*.py` (T37), `submission/README.md` |
| `sections/experiments.tex` | v1/v2/v4/v4j comparison (Table II), mechanism evidence, h7, reproducibility | `report/evidence.py`, `analysis/online_rehearsal/rehearsals/`, `analysis/online_v7_seeds/` |

Every number in Section IV comes from `uv run python report/evidence.py` (add `--json FILE` to dump them), which
reads the committed rehearsal logs. Red `\todo{...}` marks text still to be written; none may remain at
submission. The pending runs (v4j ablation, extra v2/v4 seeds) are described in
[`pending-experiments-2.md`](pending-experiments-2.md).

## Figures

`figures/make_figures.py` rebuilds `figures/convergence.pdf` (mean best-so-far per variant, range over seeds) and
`figures/allocation.pdf` (slot share and hit rate per move class, v2 vs v4) from the same logs, without the solver:

```bash
uv run python report/figures/make_figures.py
```

The figure PDFs are kept in git (see `.gitignore`); other build products are not.

## Chinese backup

`zh/` holds a Chinese translation with the same content, numbers, tables and figures (the figures are shared and keep
English labels). Build it with XeLaTeX on macOS (it uses the system Songti and Times New Roman fonts):

```bash
cd report/zh
latexmk -xelatex main.tex
```

Update it whenever the English sources change.
