# Frozen data and final surrogate model handoff

Release date: 2026-10-25. Owner: member B, Runze Liu.
Personal publication branch: `runze-liu`; main and feat/repo-skeleton remain unchanged.

This handoff preserves the immutable data partition, the approved v3.2 scoring definition, the
11-group selection results, and the three supplied full-data deployment models for minimizing
`ramping` over 21 controller weights. The supplied final models use all 2048 records. Their names
retain their historical low60/low80 lineage, but **none of the delivered models trims training data**.

The earlier trimmed models scored 74.72 / 74.50 / 74.21. Those numbers do not describe the new
full-data configurations. Rescoring the supplied full-data configurations' saved outer-test
predictions gives 72.47 / 72.35 / 72.34. No model was trained, tuned, or optimized during this handoff.

## Read and use this release

| Path | Purpose |
| --- | --- |
| [DATA_FREEZE.md](DATA_FREEZE.md) | Immutable sample identities, outer/inner membership and leakage rules |
| [SCORING_V3_2.md](SCORING_V3_2.md) | Exact five-part score and plain interpretation |
| [MODEL_SELECTION.md](MODEL_SELECTION.md) | Historical ranking, final configurations and performance boundaries |
| `packages/SDM374_data_splits_frozen_v1.zip` | Original, unchanged data-only partition package |
| `packages/SDM374_scoring_v3_2.zip` | Original, unchanged standalone scorer distribution |
| `packages/SDM374_final_models_v1.zip` | Original, unchanged three-model distribution with its checksums |
| `scoring/` | Byte-preserved policy, scorer and detailed protocol for direct inspection |
| `evidence/selection_11_groups/` | Previously completed 110-fold review, workbook, plots and compressed held-out predictions |
| `evidence/full_data_configs/` | Derived v3.2 scores for the supplied 30 full-data-config outer-fold prediction sets |
| `source_reports/` | Original teammate Word report, retained unchanged as an attributed source |
| [AI_HANDOFF.md](AI_HANDOFF.md) | Instructions for the next agent |
| [PR_DESCRIPTION.md](PR_DESCRIPTION.md) | Review-ready change description and protocol boundaries |
| `SHA256SUMS.json` | File-level release inventory; excludes itself and generated `results/` |

The existing repository directory [`../../SDM374_final_models_v1/`](../../SDM374_final_models_v1/)
comes from GitHub main commit `835c50d`. The ZIP is the authoritative byte-preserved upstream
distribution: GitHub's unpacked text files underwent newline conversion in seven files, so the
original package checksums should be verified against a fresh ZIP extraction. All three model
binaries match exactly. Neither copy has been modified to repair upstream documentation.

For inference, extract the final-model ZIP into a separate directory and use its Python 3.13.5
environment with pinned dependencies. Then run its supplied `predict.py` with a CSV of valid raw
21-weight vectors. Call `sdm.simplex.check_weights` before proposing any vector. Models return
point predictions, without calibrated uncertainty or gradients. They are not yet registered
plugins in the team's CLI/GUI. Keep their newer environment separate from the team's Python 3.11
environment; no shared dependency pins were changed.

For a read-only reproduction of the full-config v3.2 scores, use an environment with NumPy installed:

```bash
python analysis/surrogate_release/verify_release.py
```

The script verifies ZIP contents, frozen data and exact held-out sample identities, then scores the
provider's saved predictions. It writes provenance and results under ignored `results/`; it never
loads, fits or retunes a model. It verifies prediction membership, not the provider's undisclosed
training execution. The recorded 30-fold result id is `20261025-152147-release-verification`
(UTC timestamp in this analysis id); its source was uncommitted when run, as its provenance states.

For model-loading smoke checks, run `python analysis/surrogate_release/smoke_models.py` in the
separate pinned peer environment. This compares predictions on the original rows against the
package's in-sample reference and is strictly a loading check, not a test of generalization.

## Research protocol and team protocol

This is the user's frozen **2 repeats × 5 outer folds × 3 inner folds**, v3.2 research handoff.
It does not replace the team's `sdm evaluate` protocol (5 folds, seed 2026, top_k 50), change the GUI
leaderboard, or claim that these unregistered artifacts pass the plugin interface. Existing framework,
optimizers, course data and regression tolerances are unchanged.

Old experiments and environments remain available locally under ignored `outputs/`, `incoming/`
and `local_research/`. They are not part of the Git submission. The complete pre-cleanup backup and
original upload backup remain outside the repository. No historical model weights were deleted.
