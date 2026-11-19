# Windows development checks (T10 / T34)

Run from any PowerShell location; checkout paths containing Chinese characters are supported:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_windows.ps1 -Task Setup
powershell -ExecutionPolicy Bypass -File scripts/check_windows.ps1 -Task CheckData
powershell -ExecutionPolicy Bypass -File scripts/check_windows.ps1 -Task Test
powershell -ExecutionPolicy Bypass -File scripts/check_windows.ps1 -Task DryRun
```

Install `uv` first. Setup uses the frozen lockfile, Python 3.11 selected by `.python-version`,
and a **non-editable** project installation. With Python 3.11 on a GBK Windows installation,
an editable `.pth` containing a UTF-8 Chinese path can prevent Python from starting, even
with `PYTHONUTF8=1`. A non-editable wheel removes that `.pth`; subsequent checks use
`--no-sync` so uv does not put it back. `PYTHONPATH=src` makes tests use the current sources.

If an existing editable environment already fails before Setup can start, rebuild just the
development environment with `uv venv --clear`, then rerun Setup. This discards installed
packages in `.venv`; it does not affect any run records.

These commands do not call the real evaluator. DryRun only proposes and screens a batch;
it may require `solver.exe` to be present for its pinned-hash preflight. This helper is a
development check, not the final course-computer bundle. The official execution path and
code freeze remain the responsibility of T21/T22; use the team's course kit for that run.
