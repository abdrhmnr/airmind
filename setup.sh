#!/bin/bash
# AirMind one-time setup: its own Python environment in .venv with everything it needs.
#   ./setup.sh            normal
#   ./setup.sh instant    also install INSTANT mode (PyTorch + YOLO-World, ~1 GB)
#   PIP_MIRROR=1 ./setup.sh   use the Tsinghua PyPI mirror (much faster from China)
set -e
cd "$(dirname "$0")"
PY=""
# same Python as the camera project that already works on this Mac (3.13 from miniconda)
for p in /opt/homebrew/Caskroom/miniconda/base/bin/python3.13 python3.13 python3.12 python3; do
  if command -v "$p" >/dev/null 2>&1; then PY="$p"; break; fi
done
echo "Using $($PY --version) at $(command -v $PY)"
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
EXTRA=()
[ -n "$PIP_MIRROR" ] && EXTRA=(-i https://pypi.tuna.tsinghua.edu.cn/simple)
python -m pip install -q --upgrade pip "${EXTRA[@]}"
python -m pip install -r requirements.txt "${EXTRA[@]}"
[ "$1" = "instant" ] && python -m pip install -r requirements-instant.txt "${EXTRA[@]}"
chmod +x start.sh
python selftest.py
echo
echo "Done. Start with:   FAST=1 ./start.sh --lang ar --learn zh"
