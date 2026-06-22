#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_DIR="$ROOT_DIR/worker"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Python interpreter not found. Expected $VENV_PYTHON, python3, or python." >&2
  exit 1
fi

# Worker 启动时自动预装模型（YOLO-World + CLIP），已安装则跳过
# 设为 "off" 可禁用，改用手动 POST /prepare-models
export CV_AUTO_TRAINER_WORKER_AUTO_PREPARE="${CV_AUTO_TRAINER_WORKER_AUTO_PREPARE:-on}"
# 是否在启动时同时预装 Moondream2 VQA 模型（会增加下载时间和显存占用）
export CV_AUTO_TRAINER_WORKER_PREPARE_MOONDREAM="${CV_AUTO_TRAINER_WORKER_PREPARE_MOONDREAM:-off}"

cd "$WORKER_DIR"
exec "$PYTHON_BIN" main.py
