$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$WorkerDir = Join-Path $RootDir "worker"

Set-Location $WorkerDir
python main.py
