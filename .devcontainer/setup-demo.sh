#!/usr/bin/env bash
# One-time setup when the codespace is created: install PGW (CPU torch) and build the model pack.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -c pgw/deploy/hf-space/constraints.txt -e "./pgw[gateway,service,factory,pack]"
python pgw/scripts/client_demo/run.py --pack-only --pack-dir "$HOME/pgw-pack" --work "$HOME/pgw-demo"
rm -rf "$HOME/.cache/huggingface"
