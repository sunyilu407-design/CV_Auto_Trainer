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

# ── 预装 ultralytics 8.4+ 所需的 clip 包 ───────────────────────────────
# 背景：ultralytics 8.4 把 set_classes() 改成硬要求 `import clip`（来自 ultralytics/CLIP.git），
# 首次调用会触发 auto-install。预先用 `pip install --user` 装到 per-user 目录，
# 避免运行时 auto-install 撞 macOS 系统目录写权限（PEP 668 / SIP）。
if ! "$PYTHON_BIN" -c "import clip" 2>/dev/null; then
  echo "[启动] 首次运行：预装 ultralytics/CLIP 仓库包..." >&2
  if "$PYTHON_BIN" -m pip install --user --quiet "git+https://github.com/ultralytics/CLIP.git" 2>&1 | tail -3; then
    echo "[启动] clip 包已就绪" >&2
  else
    echo "[启动] 警告：clip 预装失败，Worker 启动后第一次打标可能会自动重试" >&2
  fi
fi

cd "$WORKER_DIR"
exec "$PYTHON_BIN" main.py
