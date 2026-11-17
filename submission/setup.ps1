$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
uv sync --frozen
uv run python run.py check
