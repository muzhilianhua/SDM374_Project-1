# Reconstructed eight-week solver

This directory runs the complete fixed eight-week evaluation using readable Python
control code, the original scenario/model assets, and CityLearn 2.5.0. It is a
self-contained release inside the `azq` branch: **no course EXE, extracted directory,
frozen bytecode, Conda environment, or files outside the clone are required.**

Supported and tested environment: **Windows x64, Python 3.11**. The separate uv
project here locks NumPy 1.26.4 and CPU PyTorch 2.5.1; it does not change the main
repository's NumPy 1.24.4 environment. Initial installation needs internet access
and downloads approximately 300 MB, mostly the CPU PyTorch wheel. Evaluation after
installation is local.

## Clone and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once, then use
PowerShell:

```powershell
git clone --branch azq --single-branch https://github.com/muzhilianhua/SDM374_Project-1.git
cd SDM374_Project-1
uv sync --project reverse_analysis --python 3.11 --locked
uv run --project reverse_analysis --frozen python reverse_analysis/run.py --check
uv run --project reverse_analysis --frozen python reverse_analysis/run.py
```

The last command evaluates the committed `best_weights.json` and prints:

```json
{"R": [103.46086008194834]}
```

All paths are resolved relative to the script, not the current directory. Keep the
entire `reverse_analysis/` folder together. The installation creates only its own
ignored `.venv/`; Python also creates ignored `__pycache__/` directories.

## Evaluate your own weights

A candidate is 21 finite nonnegative numbers, or a 3 by 7 matrix. Rows correspond
to Buildings 1, 2 and 3. Each row needs a positive value; the evaluator normalizes
each row internally, so an input sum of one is not required.

```powershell
uv run --project reverse_analysis --frozen python reverse_analysis/run.py --weights weights.json
Get-Content weights.json -Raw | uv run --project reverse_analysis --frozen python reverse_analysis/run.py --stdin
```

The original batch protocol is also accepted: `{"weights": [candidate1, candidate2]}`,
with 1 to 50 candidates. Outputs are in input order, as `{"R": [...]}`. The saved
three-candidate [example input](evidence/continued_real_payload.json) can be used directly:

```powershell
uv run --project reverse_analysis --frozen python reverse_analysis/run.py --weights reverse_analysis/evidence/continued_real_payload.json
```

`--workers 1` runs sequentially; the default is 8 worker processes, capped at eight
weeks per candidate. To retain per-week scores and environment details, create a
new report:

```powershell
New-Item -ItemType Directory -Force reverse_analysis/artifacts
uv run --project reverse_analysis --frozen python reverse_analysis/run.py --report reverse_analysis/artifacts/my-run.json
```

Reports are never overwritten. Malformed weights or missing/modified assets return
exit code 2. Reference verification returns exit code 1 on a numerical mismatch.

## Verify the release

```powershell
uv run --project reverse_analysis --frozen python -m unittest discover -s reverse_analysis/tests -v
uv run --project reverse_analysis --frozen python reverse_analysis/run.py --verify-reference
```

The first command checks input handling, all 124 asset hashes and a full hourly
three-building trace for week 1. The second independently simulates all three saved
original-evaluator candidates over eight weeks each. The recorded maximum absolute
R difference is `1.4210854715202004e-14`; the best candidate matches exactly.
See [validation](docs/VALIDATION.md) and [the new source-only report](evidence/source_release_validation.json).

## Contents and scope

| Path | Purpose |
| --- | --- |
| `run.py`, `runtime.py` | Single/batch CLI, validation, worker pool, scoring |
| `source_view/` | Reconstructed controller, evaluation, accounting and task helpers; these `.py` files actually run |
| `environment/` | Original eight weekly scenarios, CSV inputs and three LSTM checkpoints (67 files, about 1.4 MB) |
| `vendor/` | Unmodified upstream CityLearn 2.5.0 Python files and distribution metadata/license |
| `pyproject.toml`, `uv.lock` | Independent, reproducible CPU runtime dependencies |
| `best_weights.json`, `final_result.py` | Current best weights, preserving full floating-point precision |
| `assets_manifest.json`, `provenance.json` | Source executable/wheel identities, asset hashes, release provenance |
| `evidence/` | Saved original inputs/outputs, local search record and hourly reference trace |
| `docs/` | [Model](docs/MODEL.md), [validation](docs/VALIDATION.md), [provenance and licensing](docs/PROVENANCE.md) |

The current best is **103.46086008194834**, above 100; it is a discovered candidate,
not a proof of global optimality. Agreement on saved cases does not prove equivalence
for every weight, another dataset or another library version. This is a separate
reconstruction study, not a surrogate leaderboard result or a change to the frozen
online policy. Historical search accelerators/caches are not required to run it.
