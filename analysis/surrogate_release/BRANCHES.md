# Repository branches and collaboration boundaries

Reviewed remote state on 2026-10-25:

| Branch | Current role |
| --- | --- |
| main (`835c50d`) | Course baseline plus the teammate's supplied three-model directory |
| feat/repo-skeleton (`818b762`) | Proposed shared framework, collaboration rules, CLI/GUI, plugins and coupling study |
| azq (`2e4aca1` before this handoff) | Existing personal branch; publication destination requires owner confirmation |

`feat/repo-skeleton` turns the small baseline repository into a shared project structure. The core
loads data with checksum validation, checks the 21-weight constraints, discovers model/optimizer
plugins, runs TOML experiments through the `sdm` CLI, and records results with source/data provenance.
`AGENTS.md`, the five role briefs, interfaces and decisions specify who changes which files.
The Streamlit workbench launches those same CLI commands, avoiding a separate GUI implementation
of training, evaluation or optimization.

The branch adds Extra Trees as a reference surrogate and trust-dirichlet as a gradient-free optimizer.
The latter samples near good observed candidates, restricts distance relative to data spacing,
ranks predictions with an uncertainty penalty, and returns distinct proposals. The coupling study
explores how the surrogate's shape affects optimizer behavior using semi-synthetic responses; it
does not measure new W against the unreleased real evaluator.

This is a proposed integration foundation, not the personal model-selection branch or a replacement
for the delivered model weights. At review time it is not merged into main. In particular, its
team evaluator uses 5 folds, seed 2026 and top-50 recall, while the user's independent frozen v3.2
research uses 2x5x3 membership and the five-component score. The two protocols remain separate.

The local handoff branch contains this framework ancestry and merges the current main so both
sets of upstream work remain available together. Pushing a personal branch publishes that snapshot;
it does not update main or feat/repo-skeleton. A later reviewed merge would be a separate action.
The final three model binaries already published on main are retained without modification.

Git commit records changes on the currently checked-out local branch. Git push updates only the
explicit remote destination. A local branch named main is not automatically updated by fetch;
`origin/main` records the latest fetched remote state. Check branch and destination before publishing.
