# AGENTS.md — SDM374 Project 1

Guide for coding agents working in this repository. Read it fully before changing anything, then read
the brief for the team member you work for: `docs/roles/<A|B|C|D|E>.md`. If nobody told you which
member you work for, ask before writing code.

## 1. The project in one paragraph

Course project for SDM374 (Machine Learning System Design, SUSTech). Three buildings share a microgrid;
each building's energy controller has 7 objective weights (cost, carbon, peak, discomfort, degradation,
ramping, resilience), so a candidate is 21 weights `W`, and **each building's 7 weights must be
non-negative and sum to 1**. A black-box evaluator simulates weeks 1–8 and returns **R**, the mean over
the 8 weeks of `sum_t |E[t] - E[t-1]|` (E = district net load per hour, 167 differences per week).
**Lower R is better.** We train surrogate models `R ≈ M(W)` on `dataset_same8/train.csv` (2048 rows;
best measured R = 115.76, row `p01969`) and optimize `W` over the surrogate.

| Phase | When | Budget | Grade |
|---|---|---|---|
| Offline: submit W* from the 2048-row table | Week 8 lab | 2 submissions, best counts | 7%: beat the course baseline |
| Online: propose → real evaluator → retrain loop | Weeks 8–10 | ≤ 20 rounds × ≤ 50 points | 8%: beat our offline result; 5%: rank among teams |
| Presentation + code + IEEE report (≤ 4 pages) | Week 10 lab | — | 5% innovation, 5% report |

The real evaluator is released in week 8; until then everything is offline and only the surrogate's
predictions exist. A prediction is never a measured score.

## 2. Setup and everyday commands

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required (numpy 1.24.4 has no 3.12+ wheels).

```bash
uv sync                                   # core + dev dependencies
uv run pytest -q                          # must pass before you finish any task
uv run sdm list --json                    # registered models/optimizers and their parameters
uv run sdm check-data --json              # dataset checksum and summary
uv run sdm evaluate -c experiments/<name>.toml    # shared offline protocol (outer 5-fold CV)
uv run sdm train    -c experiments/<name>.toml    # fit on all rows, save model.pkl
uv run sdm optimize -c experiments/<name>.toml -n 50 [--model-run <train-run-id>]
uv run sdm predict  --model-run <run-id> --weights weights.json   # or --weights - (stdin)
uv run sdm runs --json                    # all recorded runs with metrics
uv run sdm reproduce <run-id>             # re-execute and compare; exit 1 if outputs differ
uv sync --group gui && uv run --group gui streamlit run app/main.py   # workbench (humans)
```

## 3. Repository map

| Path | What it is | Owner |
|---|---|---|
| `src/sdm/models/` | Surrogate plugins, one file per model | B |
| `src/sdm/optimizers/` | Optimizer plugins, one file per algorithm | C |
| `src/sdm/replay.py`, `analysis/mechanism/` | Offline replay of the online loop; mechanism studies | D |
| `analysis/ablation/`, `report/` | Ablation tables and figures; IEEE report | E |
| `experiments/*.toml` | One config per experiment (`_template.toml` is a template) | author of the experiment |
| `src/sdm/{interfaces,registry,config,data,simplex,evaluate,runs,catalog,cli}.py` | Core framework | A (others: propose changes in an issue/PR) |
| `app/` | Streamlit workbench; every action calls the `sdm` CLI | A |
| `tests/` | Regression, contract, CLI and GUI tests | A; everyone adds tests for their code |
| `docs/` | Interfaces, decisions, role briefs | A; each member keeps their own role brief's backlog current |
| `runs/` | Generated run folders (git-ignored) | nobody edits by hand |
| `baseline.py`, `model.npz`, `weights.json`, `requirements.txt`, `dataset_same8/` | Course-provided files | **read-only** |

Details of every interface: `docs/interfaces.md`. Settled design decisions and open questions:
`docs/decisions.md`. Do not re-open a settled decision without asking the user.

## 4. Hard rules

- MUST NOT edit the course-provided read-only files listed above.
- MUST NOT loosen tolerances or frozen numbers in `tests/test_baseline_regression.py`.
- MUST NOT import `streamlit` (or anything in `app/`) from `src/sdm`. Dependencies point from `app/` to `src/sdm` only.
- MUST NOT change the `[evaluate]` protocol (5 folds, seed 2026, top_k 50) used to compare models.
- MUST NOT add a second way to execute experiments: the GUI and scripts go through the `sdm` CLI so every result has a run folder.
- Model inputs stay the raw 21 weights in `FEATURES` order (`sdm.simplex.FEATURES`). Feature engineering happens inside a plugin's `fit` / `predict`.
- Every proposed `W` must pass `sdm.simplex.check_weights`.
- New dependencies: `uv add <package>` (updates `pyproject.toml` and `uv.lock`; commit both). Keep versions compatible with the pinned numpy 1.24.4 / scikit-learn 1.2.2.
- Edit files outside your role's paths only when the task requires it, and say so in the PR description.

## 5. How to do a typical task

1. Branch from `main` (`git switch -c feat/<short-topic>`). If `AGENTS.md` is not on `main` yet, branch
   from the branch you were told to use and say so in the PR.
2. New model or optimizer: copy `src/sdm/models/_template.py` or `src/sdm/optimizers/_template.py` to a new
   file without the leading underscore, rename the class and the registry name. The registry imports
   every non-underscore module automatically; nothing else needs registering.
3. Copy `experiments/_template.toml` to `experiments/<plugin>-<variant>.toml` (e.g. `gbdt-default.toml`)
   and point it at your plugin. The file name becomes part of the run id.
4. `uv run pytest -q`. `tests/test_plugin_contract.py` checks every registered plugin automatically
   (shapes, finiteness, `std >= 0`, determinism, simplex constraint, ordering, distinct batches). Add
   plugin-specific tests in `tests/test_<plugin>.py` for behaviour the contract cannot know about.
5. `runs/` is git-ignored and starts empty on every machine, so create your own reference first:
   `uv run sdm evaluate -c experiments/baseline.toml`.
6. Produce evidence with the CLI, not ad-hoc scripts: `uv run sdm evaluate -c experiments/<name>.toml`
   (models) or `uv run sdm optimize ...` (optimizers). Keep one `evaluate` under about two minutes on a
   laptop (it fits the model five times).
7. Commit before the evidence runs. A run records the commit and a `dirty` flag that is true when
   `src/`, `experiments/`, `pyproject.toml` or `uv.lock` have uncommitted changes. If you were told not to
   commit, the runs will be dirty: say so next to the run ids.
8. Compare: `uv run sdm runs --json`, or in Python
   `from sdm import catalog; catalog.leaderboard(catalog.run_records())`.
9. Update the backlog in your role brief, then open a PR using `.github/pull_request_template.md`.

## 6. Conventions

- Code, comments, docstrings and docs in English; GUI text in Chinese.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `build:`).
- One branch per task; merge to `main` through a reviewed PR.
- Run ids are `<YYYYmmdd-HHMMSS>-<kind>-<config file name or --name>`, e.g.
  `20261025-134915-evaluate-gbdt-default`. Quote results by run id; never paste numbers without one.
- Metric names are the JSON keys in `metrics.json`: `mae`, `rmse`, `r2`, `spearman`, `top50_recall`,
  `mean_predicted_std`, `std_coverage` (evaluate); `predicted_mean`, `nearest_train_distance`, ... (optimize).
- Plugin docstrings: the first line is shown in `sdm list` and the GUI, so make it a one-line summary.
- Configs created from the GUI form are named `[a-z0-9-]+` and are never overwritten.

## 7. Definition of done

- [ ] `uv run pytest -q` passes (the GUI tests need `uv sync --group gui`; run them if you touched `app/`).
- [ ] `uv run sdm list` shows any new plugin with sensible defaults.
- [ ] Evidence runs exist, their ids are in the PR description, and any `dirty` run is flagged as such.
- [ ] No read-only file changed; no regression tolerance changed.
- [ ] Your role brief's backlog reflects what is done and what is next.
