# Member A — coordination, framework and evaluation operations

## Mission
Keep everyone's work pluggable into one pipeline, keep results reproducible, and (online phase) be the
only person who sends points to the real evaluator so the 20-call budget is never wasted (decision 19).

## You own
`src/sdm/{interfaces,registry,config,data,simplex,evaluate,runs,catalog,cli}.py`, `app/`, `tests/`
(infrastructure), `docs/`, `AGENTS.md`, `pyproject.toml`, `uv.lock`, `.streamlit/`, `.github/`.

## Backlog — offline (until the week 8 lab)

> Paths to offline studies, `SUBMISSION_READY/` and `artifacts/` in this section resolve at git tag
> `offline-final`; they are no longer on the online branch (decision 24).

- [x] Repository skeleton, plugin interfaces, registry, `sdm` CLI, run records, `reproduce`
- [x] Baseline regression tests; shared evaluation protocol (`sdm evaluate`)
- [x] Workbench GUI that launches every CLI command as a background job
- [x] `frozen-v32` evaluate protocol: B's frozen 2×5 splits + approved v3.2 scorer, hash-checked;
      leaderboard ranks it first (2026-10-27, see decisions 6, 13, 14)
- [ ] Review and merge B/C/D/E pull requests; keep `docs/interfaces.md` in sync with the code
- [x] Offline submission 1 (2026-11-02): consensus challenger, real R 119.7121, bar not met (Findings)
- [ ] Before 2026-11-05 end of class: offline submission 2. The bar is R < 115.76 (decision 17) and it is
      chosen without the solver (decision 23). Recommended W: `analysis/offline_submission2/delivery/sign/max-p.json`,
      direction 14840 at step 0.05 from `p01969` (step ≤ 0.05, both GPs z > 0.5, ball-robust); a
      defensible, unvalidated P(R < R(p01969)) reading is 0.6–0.8 (README there). Record the real R in decisions.md Findings after submitting
- [x] `local-gp` judge plugin (2026-11-05, B's path, added for submission 2): v32_score 64.50,
      std_coverage 0.70 — run `20261105-145418-evaluate-local-gp-v32`
- [ ] Prepare an evaluator smoke-test checklist (input format, time per point, renormalisation; determinism
      is settled, decision 20)
- [ ] Confirm the course computer environment (OS, Python, uv, `libomp`) for decision 22

## Backlog — online (weeks 8–10)
- [ ] Define the call log `online_log.csv` (call number, 21 weights, R, source strategy, run id) and an append-only writer in `src/sdm/`
- [ ] Make each call's result flow into `X, y` for training and proposing (CLI option or dataset overlay)
- [ ] GUI online console: budget ledger (used / 20), pre-submit checks (constraint, duplicates, budget), best-so-far curve
- [ ] Freeze the 20-call policy (code, configs, seeds) and rehearse it end to end on the course computer (decision 22)

## Acceptance criteria
- `uv run pytest -q` green on `main` at all times.
- Every number in the report traces to a run id that `sdm reproduce` confirms.
- Budget ledger never exceeds 20 calls; no point is evaluated twice by accident.

## Interfaces
Provides every interface in `docs/interfaces.md`. Consumes plugins from B and C, replay results from D,
figures and tables from E.

## Handoff
PRs that change a shared interface must update `docs/interfaces.md` and `docs/decisions.md` in the same PR.
