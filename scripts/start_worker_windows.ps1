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

# ── 预装 ultralytics 8.4+ 所需的 clip 包 ───────────────────────────────
# 背景：ultralytics 8.4 把 set_classes() 改成硬要求 `import clip`（来自 ultralytics/CLIP.git），
# 首次调用会触发 auto-install。但 ultralytics 内部调 `uv pip install --python sys.executable`
# 写系统 C:\Python314\Lib\site-packages\ 会撞 os error 5（无管理员权限）。
# 所以这里用 uv 预装到 per-user 目录（与 sys.executable 对应的用户 site-packages 对齐）。
$PythonExe = (python -X utf8 -c "import sys; print(sys.executable)").Trim()
$ClipModule = python -c "import clip" 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host "[启动] 首次运行：预装 ultralytics/CLIP 仓库包..." -ForegroundColor Yellow
  # 用 uv 安装到用户 site-packages，--python 指向当前 Python
  uv pip install --user --python "$PythonExe" --quiet "git+https://github.com/ultralytics/CLIP.git" 2>&1 | Select-Object -Last 5
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[启动] 警告：clip 预装失败，Worker 启动后第一次打标可能会自动重试" -ForegroundColor Yellow
  } else {
    Write-Host "[启动] clip 包已就绪" -ForegroundColor Green
  }
}

Set-Location $WorkerDir
python -X utf8 main.py
