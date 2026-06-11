$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$WorkerDir = Join-Path $RootDir "worker"

# ── Windows UTF-8 locale（解决 pip install 时的 GBK 编码问题）───────────────
# Python 3.11+ 在 Windows 上需要这些环境变量才能正确处理非 ASCII 输出
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# pip 本身也要用 UTF-8
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

# Worker 启动时自动预装模型（YOLO-World + CLIP），已安装则跳过
# 设为 "off" 可禁用，改用手动 POST /prepare-models
if (-not $env:CV_AUTO_TRAINER_WORKER_AUTO_PREPARE) { $env:CV_AUTO_TRAINER_WORKER_AUTO_PREPARE = "on" }
# 是否在启动时同时预装 Moondream2 VQA 模型（会增加下载时间和显存占用）
if (-not $env:CV_AUTO_TRAINER_WORKER_PREPARE_MOONDREAM) { $env:CV_AUTO_TRAINER_WORKER_PREPARE_MOONDREAM = "off" }

Set-Location $WorkerDir
python -X utf8 main.py
