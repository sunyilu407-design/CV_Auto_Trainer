$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$FrontendDistDefault = Join-Path $RootDir "frontend\dist"

if (-not $env:CV_AUTO_TRAINER_DB_URL) { $env:CV_AUTO_TRAINER_DB_URL = "postgresql://postgres:postgres@127.0.0.1:5432/cv_auto_trainer" }
if (-not $env:CV_AUTO_TRAINER_SECRET_KEY) { $env:CV_AUTO_TRAINER_SECRET_KEY = "change-me" }
if (-not $env:CV_AUTO_TRAINER_ADMIN_USERNAME) { $env:CV_AUTO_TRAINER_ADMIN_USERNAME = "admin" }
if (-not $env:CV_AUTO_TRAINER_ADMIN_PASSWORD) { $env:CV_AUTO_TRAINER_ADMIN_PASSWORD = "change-me" }
if (-not $env:CV_AUTO_TRAINER_FRONTEND_DIST) { $env:CV_AUTO_TRAINER_FRONTEND_DIST = $FrontendDistDefault }
if (-not $env:CV_AUTO_TRAINER_CORS_ORIGINS) { $env:CV_AUTO_TRAINER_CORS_ORIGINS = "http://127.0.0.1:8000,http://localhost:8000" }

Set-Location $BackendDir
python -m uvicorn main:app --host 127.0.0.1 --port 8000
