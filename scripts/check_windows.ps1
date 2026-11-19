<#
Development checks only. No command here starts the real course evaluator.
Use -Task Setup once, then Test, CheckData or DryRun.
#>
[CmdletBinding()]
param(
    [ValidateSet('Setup', 'Test', 'CheckData', 'DryRun')]
    [string]$Task = 'CheckData',
    [string]$Config = 'experiments/online-baseline.toml',
    [string]$Name = 'windows-check'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if ($Task -eq 'Setup') {
        & uv sync --frozen --no-editable --all-groups
        exit $LASTEXITCODE
    }
    # Python 3.11's Windows site.py reads editable .pth paths with the system
    # encoding. A wheel avoids this problem even when the checkout has CJK paths.
    # --no-sync prevents uv from reinstalling the editable project before launch.
    # Tests still import current sources rather than a stale installed wheel.
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    $env:PYTHONUTF8 = '1'
    switch ($Task) {
        'Test' { & uv run --no-sync python -m pytest -q }
        'CheckData' { & uv run --no-sync sdm check-data --json }
        'DryRun' {
            & uv run --no-sync sdm online -c $Config --dry-run --name $Name
        }
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
