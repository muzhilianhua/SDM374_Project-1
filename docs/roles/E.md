# Member E — baseline, ablations and the report

## Mission
Make every claim in the report backed by a reproducible run, and write the IEEE report (≤ 4 pages) and
the slides without a last-minute rush.

## You own
`report/` (IEEE LaTeX), `analysis/ablation/` (scripts that build tables and figures from runs),
`experiments/ablation-*.toml`.

## Ask A first
Changes to `sdm.catalog` or run-folder formats (add helpers in `analysis/ablation/` instead).

## Backlog — offline (until the week 8 lab)
- [ ] Reproduce the baseline through the CLI (`sdm evaluate` / `sdm optimize` with `experiments/baseline.toml`) and record the run ids
- [ ] Ablation framework: build "model × features × optimizer" tables from runs with `sdm.catalog.leaderboard` / `candidates`
- [ ] IEEE LaTeX template and one plotting style (shared module in `analysis/ablation/`)
- [ ] Draft Introduction and Method from `docs/interfaces.md` and `docs/decisions.md`

## Backlog — online (weeks 8–10)
- [ ] After each round: best-so-far curve and per-round score distribution
- [ ] Collect sections from A–D, keep the report within 4 pages; build the slides and the innovation story
- [ ] Final code package: README commands that reproduce every reported number

## Acceptance criteria
- Every number and figure in `report/` lists the run id(s) it came from.
- Figures regenerate from runs with one command.

## Interfaces
Reads runs through `sdm.catalog`; does not modify run folders.

## Handoff
PR with the table/figure scripts, the run ids used, and the rendered PDF.
