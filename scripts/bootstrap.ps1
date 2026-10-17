$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path '.cache/tmp', '.cache/pip' | Out-Null
$env:TEMP = Join-Path $projectRoot '.cache/tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache/pip'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TORCH_HOME = Join-Path $projectRoot '.cache/torch'
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    throw 'Create .venv using a directly installed Python 3.11-3.13 interpreter. Do not use the Windows install-manager alias.'
}
if (Test-Path -LiteralPath 'requirements-lock.txt') {
    & '.venv/Scripts/python.exe' -B -m pip install -r requirements-lock.txt
} else {
    & '.venv/Scripts/python.exe' -B -m pip install 'setuptools>=68'
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (Test-Path -LiteralPath 'requirements-lock.txt') {
    & '.venv/Scripts/python.exe' -B -m pip install --no-deps --no-build-isolation -e .
} else {
    & '.venv/Scripts/python.exe' -B -m pip install --no-build-isolation -e '.[dev]'
}
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
