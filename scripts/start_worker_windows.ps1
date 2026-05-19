$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$WorkerDir = Join-Path $RootDir "worker"

# Worker 启动时自动预装模型（YOLO-World + CLIP），已安装则跳过
# 设为 "off" 可禁用，改用手动 POST /prepare-models
if (-not $env:CV_AUTO_TRAINER_WORKER_AUTO_PREPARE) { $env:CV_AUTO_TRAINER_WORKER_AUTO_PREPARE = "on" }
# 是否在启动时同时预装 Moondream2 VQA 模型（会增加下载时间和显存占用）
if (-not $env:CV_AUTO_TRAINER_WORKER_PREPARE_MOONDREAM) { $env:CV_AUTO_TRAINER_WORKER_PREPARE_MOONDREAM = "off" }

Set-Location $WorkerDir
python main.py
