param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [string]$RuleRun = 'rules-v1'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path '.cache/tmp', '.cache/torch' | Out-Null
$env:TEMP = Join-Path $projectRoot '.cache/tmp'
$env:TMP = $env:TEMP
$env:TORCH_HOME = Join-Path $projectRoot '.cache/torch'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.venv/Scripts/python.exe' -B -m graphshield.cli train-rings --rule-run $RuleRun --run-name $RunName
if ($LASTEXITCODE -ne 0) { throw 'Phase 5 training failed. Inspect the run manifest.' }
& '.venv/Scripts/python.exe' -B scripts/verify_phase5.py --run-name $RunName
if ($LASTEXITCODE -ne 0) { throw 'Phase 5 verification failed.' }
