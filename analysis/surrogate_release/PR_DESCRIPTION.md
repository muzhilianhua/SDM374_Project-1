## What and why

Member B handoff: preserve the immutable 2048-record partition, approved v3.2 scorer, specified
11-group saved-artifact comparison, and the three supplied full-data model artifacts. Historical
trimmed-model scores had to be separated from the full-data configurations because their training
support and fixed parameter choices changed. The release provides exact score formulas, file
hashes, original source packages, raw prediction evidence and instructions for the next agent.

This branch starts from `origin/feat/repo-skeleton` (`818b762`), which supplied AGENTS and the
framework, and merges `origin/main` (`835c50d`), which supplied the final-model directory. Only
research handoff files, member B's backlog, and a root README navigation link are added beyond that
integration. Shared core code, protocol, course originals and dependency pins are unchanged.
Documentation outside B's plugin paths is necessary for the user-requested cross-member handoff.

## Evidence

| Analysis id | Action | Result |
| --- | --- | --- |
| 20261025-204551-artifact-review-v32 | Predict 50 saved peer fold artifacts and rescore 60 earlier held-out sets | Specified 11-group ranking reproduced; no fitting/tuning; original run dirty |
| 20261025-152147-release-verification | `python analysis/surrogate_release/verify_release.py` | 14 package checksums, 12288 predictions, 30 exact frozen test memberships; full-config totals 72.47 / 72.35 / 72.34; run dirty |

These are provenance-backed local v3.2 analyses, not evidence runs for the team's `sdm evaluate`
leaderboard. Source hashes are retained. Full-data configuration training execution is declared by
the provider; only saved test membership, labels, metrics and checksums were verified here.

## Validation

See `VALIDATION.json` for release integrity, model smoke checks and the final test result. No new
model plugin was registered, no W optimization ran, and no fresh training was authorized/performed.
The models require a separate newer inference environment and do not yet implement the mandatory
uncertainty interface. Small differences on reused overlapping folds do not establish superiority.

## Publication

The user explicitly requested publication on the new personal branch `runze-liu`. Main and
feat/repo-skeleton are not publication destinations; merging main remains a separate action.
Original teammate Word content is unchanged and
retained as an attributed source; only the specified 11-group scope was independently reproduced.
