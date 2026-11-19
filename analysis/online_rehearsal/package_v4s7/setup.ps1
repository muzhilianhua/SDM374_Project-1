# Creates .venv with Python 3.11 and the pinned packages (needs uv: https://docs.astral.sh/uv/).
# Without uv: install Python 3.11, then  py -3.11 -m venv .venv ; .venv\Scripts\pip install -r requirements.txt
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe online_run.py check
