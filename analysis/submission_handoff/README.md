# Member C handoff and archived evidence

Current controller status is in [SUBMISSION_READY](../../SUBMISSION_READY/README.md).
P1 was measured at 119.71206678869203 and failed the offline bar. The 50-point packet is superseded
by the 20-call sequential format. No controller is nominated for offline submission 2 here.

## Release assets

Large generated files and fitted pickles are kept in the
[PR #7 evidence release](https://github.com/muzhilianhua/SDM374_Project-1/releases/tag/c-pr7-evidence-2026-11-03).
Configs, summary metrics, provenance, weights and checksum manifests stay in Git.
[release_assets.json](release_assets.json) lists each external file's path, size and SHA256.

- `C_PR7_LARGE_EVIDENCE_2026-11-03.zip`: original raw predictions, neighborhoods and fitted binaries.
- `SUBMISSION_READY_2026-11-03.zip`: complete updated handoff, including the loadable reference model.
- `C_PR7_ORIGINAL_HISTORY.bundle`: original experiment commits, including source commit IDs cited by
  run provenance. The PR is consolidated to keep generated binaries out of merged Git history.

From the repository root, with Python 3.11 and GitHub CLI:

```sh
python3 analysis/submission_handoff/assets.py restore
python3 analysis/submission_handoff/assets.py verify
env -u PYTHONPATH uv run --locked --group gui python analysis/submission_handoff/package.py verify
```

Alternatively download the evidence ZIP from the release and pass `restore --archive /path/to/zip`.
Restoration validates the archive and every file before writing and refuses to overwrite a differing
local file. It never loads a pickle. Model loading requires the repository's trusted code and locked
environment. Downloaded source bytes are preserved exactly, including historical recommendations
inside archived JSON; the current package status and team decisions take precedence.

To inspect an original experiment commit, download the history bundle and verify its SHA256 against
`release_assets.json`, then run `git bundle verify <bundle>` and `git fetch <bundle>` from this repo.
The bundle uses the already merged `fc3223f` as its prerequisite. Original run IDs and original Git
provenance are historical records and are not rewritten to the review-fix commit.

## Regeneration

After restoration, build a new directory (the builder refuses to overwrite an existing one):

```sh
env -u PYTHONPATH uv run --locked --group gui python analysis/submission_handoff/package.py build \
  --out analysis/submission_handoff/results/rebuilt
```

The builder reads current wording from `templates/`, imports the team's real-result records, derives
the pool status sheet, replays the three controller predictions and regenerates all manifests and ZIP
contents. It makes no evaluator calls. Refit the reference with
`sdm train -c experiments/selection-reference-2026.toml`; evaluate with `sdm evaluate` using the same
config and frozen protocol. Refitting generates a new run; historical bytes remain in the release.
