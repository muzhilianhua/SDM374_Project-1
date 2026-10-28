# Member A — coordination, framework and evaluation operations

## Mission
Keep everyone's work pluggable into one pipeline, keep results reproducible, and (online phase) be the
only person who sends points to the real evaluator so the 20 × 50 budget is never wasted.

## You own
`src/sdm/{interfaces,registry,config,data,simplex,evaluate,runs,catalog,cli}.py`, `app/`, `tests/`
(infrastructure), `docs/`, `AGENTS.md`, `pyproject.toml`, `uv.lock`, `.streamlit/`, `.github/`.

## Backlog — offline (until the week 8 lab)
- [x] Repository skeleton, plugin interfaces, registry, `sdm` CLI, run records, `reproduce`
- [x] Baseline regression tests; shared evaluation protocol (`sdm evaluate`)
- [x] Workbench GUI that launches every CLI command as a background job
- [x] `frozen-v32` evaluate protocol: B's frozen 2×5 splits + approved v3.2 scorer, hash-checked;
      leaderboard ranks it first (2026-10-27, see decisions 6, 13, 14)
- [ ] Review and merge B/C/D/E pull requests; keep `docs/interfaces.md` in sync with the code
- [ ] Before the lab: run the offline-submission decision meeting; pick two candidates from
      `sdm.catalog.candidates` (first conservative and non-extrapolating, second more aggressive)
- [ ] Prepare an evaluator smoke-test checklist (input format, time per point, determinism, renormalisation)

## Backlog — online (weeks 8–10)
- [ ] Define `online_log.csv` (round, point id, 21 weights, R, source strategy, run id) and an append-only writer in `src/sdm/`
- [ ] Make online rows flow into `X, y` for training and proposing (CLI option or dataset overlay)
- [ ] GUI online console: budget ledger (used / 1000), pre-submit checks (constraint, duplicates, budget), best-so-far curve
- [ ] Each round: submit C's reviewed 50 points, log results, host the 15-minute sync

## Acceptance criteria
- `uv run pytest -q` green on `main` at all times.
- Every number in the report traces to a run id that `sdm reproduce` confirms.
- Budget ledger never exceeds 20 rounds × 50 points; no point is evaluated twice by accident.

## Interfaces
Provides every interface in `docs/interfaces.md`. Consumes plugins from B and C, replay results from D,
figures and tables from E.

## Handoff
PRs that change a shared interface must update `docs/interfaces.md` and `docs/decisions.md` in the same PR.
