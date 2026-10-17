param([int]$Port=8765)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path '.cache/tmp','.cache/torch' | Out-Null
$env:TEMP=Join-Path $projectRoot '.cache/tmp'
$env:TMP=$env:TEMP
$env:TORCH_HOME=Join-Path $projectRoot '.cache/torch'
$env:PYTHONDONTWRITEBYTECODE='1'
if (-not (Test-Path 'artifacts/phase6/bundles/serving-v1/bundle.json')) {
    & '.venv/Scripts/python.exe' -B -m graphshield.cli prepare-serving
    if ($LASTEXITCODE -ne 0) { throw 'Unable to prepare the serving bundle.' }
}
& '.venv/Scripts/python.exe' -B -m graphshield.cli replay
if ($LASTEXITCODE -ne 0) { throw 'Unable to prepare the demonstration replay.' }
& '.venv/Scripts/python.exe' -B scripts/serve.py --port $Port
