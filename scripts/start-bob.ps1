# Launch Bob Shell from this repository with its Python environment.
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoPath '.venv'
$pythonPath = Join-Path $venvPath 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The project .venv is missing. Follow docs/SETUP.md, step 5.'
}
if (-not (Get-Command bob -ErrorAction SilentlyContinue)) {
    throw 'Bob Shell is missing from PATH. See https://bob.ibm.com/download?bob=shell'
}
Set-Location -LiteralPath $repoPath
$env:VIRTUAL_ENV = $venvPath
$env:PATH = (Join-Path $venvPath 'Scripts') + [IO.Path]::PathSeparator + $env:PATH
Write-Host "Project: $repoPath"
& $pythonPath --version
# Matches the lab's requested Auto-approve setting for this Bob session.
& bob chat --workspace $repoPath --auto-approve
