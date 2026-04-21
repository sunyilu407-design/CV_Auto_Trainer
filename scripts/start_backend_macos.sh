#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIST_DEFAULT="$ROOT_DIR/frontend/dist"
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

export CV_AUTO_TRAINER_DB_URL="${CV_AUTO_TRAINER_DB_URL:-sqlite:///cv_auto_trainer.db}"
export CV_AUTO_TRAINER_SECRET_KEY="${CV_AUTO_TRAINER_SECRET_KEY:-change-me}"
export CV_AUTO_TRAINER_ADMIN_USERNAME="${CV_AUTO_TRAINER_ADMIN_USERNAME:-admin}"
export CV_AUTO_TRAINER_ADMIN_PASSWORD="${CV_AUTO_TRAINER_ADMIN_PASSWORD:-change-me}"
export CV_AUTO_TRAINER_FRONTEND_DIST="${CV_AUTO_TRAINER_FRONTEND_DIST:-$FRONTEND_DIST_DEFAULT}"
export CV_AUTO_TRAINER_CORS_ORIGINS="${CV_AUTO_TRAINER_CORS_ORIGINS:-http://127.0.0.1:8000,http://localhost:8000}"

cd "$BACKEND_DIR"
exec "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 8000
