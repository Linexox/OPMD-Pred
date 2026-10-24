#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
umask 077
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export XDG_CACHE_HOME="$PROJECT_ROOT/runtime/cache"
export HF_HOME="$PROJECT_ROOT/runtime/cache/huggingface"
export PIP_CACHE_DIR="$PROJECT_ROOT/runtime/cache/pip"
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/apps/server/linux_service.py" "$@"
