# Provenance and third-party components

Owner: member C, 艾子齐 (`azq`). The release is added at the user's request to the
existing `azq` branch. It is separate from the repository's `sdm` model/optimizer
experiments and has its own dependency lock to preserve the main environment.

## Original course assets

The scenario JSON, CSV inputs and three `.pth` checkpoints in `environment/` were
extracted from the course executable with SHA-256:

```text
293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247
```

This is the executable pinned by the repository's `solver/CHECKSUMS.sha256`.
Original assets remain course-provided material; they are not newly trained models
or newly collected building measurements. The reconstruction does not establish
the complete upstream training-data provenance or assign a new license to these
assets. Their exact original bytes are recorded in `assets_manifest.json`.

## Reconstructed and new code

`source_view/` is a readable semantic reconstruction of the Python 3.11 task
bytecode, with added explanations/formatting. It is not a verbatim recovery of all
original source. The public bounded branch is implemented; the unused base-agent
prediction branch is explicitly marked as omitted. The release runner, path
bootstrap, validation and tests are newly written support code.

All public-path helper imports now use reconstructed source. This differs from
the earlier workspace prototype, which still imported original `citylearn_task`
bytecode. The saved source-only validation report tests this change explicitly.

## CityLearn

`vendor/` contains unmodified files from the upstream
`citylearn-2.5.0-py3-none-any.whl`, SHA-256:

```text
e6f0ed39d0ce438bfc7ff9caa1b9c3e9d76594d1b6758a926696a7663ce3af5b
```

Upstream: <https://github.com/intelligent-environments-lab/CityLearn>.
The original [MIT license](../vendor/citylearn-2.5.0.dist-info/LICENSE), metadata and
wheel RECORD are retained. CityLearn is third-party code, not reconstructed task
code. Large unused dependencies for autosizing are not installed; the validated
fixed-equipment task path disables those functions.

Other dependencies, including CPU PyTorch, are installed from their package
indexes according to `uv.lock`, retaining their upstream licenses in the installed
distributions. No executable, DLL, compiled Python bytecode, virtual environment,
usage counter, search cache or access token is committed.

`provenance.json` records the release's base commit, dataset hash (for repository
context; the training CSV is not an evaluator input), seed, package versions and
the historical evidence sources. Hashes establish file identity, not correctness
or optimality by themselves.
