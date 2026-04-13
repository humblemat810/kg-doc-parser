param(
    [switch]$SkipPoetryInstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$localKogwistar = Join-Path $repoRoot 'kogwistar'
$hasLocalKogwistar = Test-Path (Join-Path $localKogwistar 'pyproject.toml')

Push-Location $repoRoot
try {
    if (-not $SkipPoetryInstall) {
        poetry install
    }

    if ($hasLocalKogwistar) {
        Write-Host "Local kogwistar checkout detected; installing editable override from $localKogwistar"
        poetry run python -m pip install -e $localKogwistar
    }
    else {
        Write-Host "No local kogwistar checkout found; keeping GitHub dependency from pyproject.toml"
    }
}
finally {
    Pop-Location
}
