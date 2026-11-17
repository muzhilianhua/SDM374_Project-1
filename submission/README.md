# SDM374 standalone submission package

This folder is the frozen online runner. It sends at most 20 sequential calls, with at most 50 legal and
previously unseen weight sets per call. Before the first call, the policy reads only `data/train.csv`;
afterwards it adds only measured results from this run. It does not read repository research logs or embed a
historical solver result.

## Setup

Use Python 3.11. With `uv`:

```powershell
uv sync --frozen
```

Without `uv`, create a Python 3.11 virtual environment and install `requirements.txt`.

## Checks

Run the package checksum and training-data checks without using the solver:

```powershell
uv run python run.py check
uv run python run.py run --evaluator synthetic --max-calls 2 --out runs\synthetic-check
```

The synthetic run is a development check and does not consume course calls. It writes append-only logs and
hash-chained checkpoints under the chosen output directory. A second invocation resumes only when the
previous call has a complete checkpoint; a started call without a completed result stops for manual review.

## Formal run

Copy the course `solver.exe` beside `run.py`, or provide a separate development copy explicitly:

```powershell
uv run python run.py check --solver C:\path\solver.exe
uv run python run.py run --evaluator solver --solver C:\path\solver.exe --out runs\formal
```

The solver SHA-256 is checked against the course build. The preflight sends one invalid batch and requires
the expected input-validation error before the first billable call; this detects the known empty-stdin
failure. The runner never deletes or modifies `usage.count`. Use a separate development copy when testing.

`config.json` contains the frozen seed and policy constants. `run.py` loads those values rather than
optimising them at runtime. `CHECKSUMS.sha256` is checked by `run.py check`.
