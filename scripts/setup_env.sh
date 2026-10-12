#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BACKEND="${1:-cpu}"
case "$BACKEND" in cpu|cu126|cu128) ;; *) echo 'Usage: bash scripts/setup_env.sh [cpu|cu126|cu128]' >&2; exit 2;; esac
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export UV_CACHE_DIR="$ROOT/.cache/uv"
if [[ ! -x .venv/bin/python ]]; then
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    "$PYTHON_BIN" -m venv .venv
  elif command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 --seed .venv
  else
    python3.11 -m venv .venv
  fi
fi
.venv/bin/python -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Use Python 3.10–3.12"'
if [[ "$BACKEND" != cpu && "$(uname -s)" != Linux ]]; then
  echo 'CUDA setup requires Linux and an NVIDIA GPU.' >&2; exit 2
fi
if [[ "$(uname -s)" == Linux ]]; then
  .venv/bin/python -m pip install torch==2.8.0 --index-url "https://download.pytorch.org/whl/$BACKEND"
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-index --only-binary=:all: torch-scatter==2.1.2 \
  -f "https://data.pyg.org/whl/torch-2.8.0+$BACKEND.html"
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch, torch_scatter, torch_geometric; print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
mkdir -p outputs
.venv/bin/python -m pip freeze > outputs/environment-freeze.txt
echo 'Environment ready. See README.md for asset and experiment commands.'
