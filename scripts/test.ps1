$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path '.cache/tmp', '.cache/tests' | Out-Null
$env:TEMP = Join-Path $projectRoot '.cache/tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TORCH_HOME = Join-Path $projectRoot '.cache/torch'
# Each invocation uses a new contained directory; pytest does not delete earlier runs.
$testRun = '.cache/tests/' + [guid]::NewGuid().ToString('N')
& '.venv/Scripts/python.exe' -B -m pytest --basetemp $testRun -q @args
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
