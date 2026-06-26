$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$FrontendDistDefault = Join-Path $RootDir "frontend\dist"

# ── Windows UTF-8 locale ─────────────────────────────────────────────────────
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# Windows 上 Python 默认按 GBK (CP_ACP) 编码文件名，导致中文文件名找不到。
# 设为 0 后 os.fsencode/fsdecode 使用 UTF-8，与文件系统实际字节对齐。
# 解决：上传/下载中文文件名图片 404 的问题。
$env:PYTHONLEGACYWINDOWSFSENCODING = "0"

$env:CV_AUTO_TRAINER_DB_URL = "sqlite:///cv_auto_trainer.db"
if (-not $env:CV_AUTO_TRAINER_SECRET_KEY) { $env:CV_AUTO_TRAINER_SECRET_KEY = "change-me" }
if (-not $env:CV_AUTO_TRAINER_ADMIN_USERNAME) { $env:CV_AUTO_TRAINER_ADMIN_USERNAME = "admin" }
if (-not $env:CV_AUTO_TRAINER_ADMIN_PASSWORD) { $env:CV_AUTO_TRAINER_ADMIN_PASSWORD = "change-me" }
if (-not $env:CV_AUTO_TRAINER_FRONTEND_DIST) { $env:CV_AUTO_TRAINER_FRONTEND_DIST = $FrontendDistDefault }
if (-not $env:CV_AUTO_TRAINER_CORS_ORIGINS) { $env:CV_AUTO_TRAINER_CORS_ORIGINS = "http://127.0.0.1:5173,http://localhost:5173" }

Set-Location $BackendDir
python -X utf8 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
