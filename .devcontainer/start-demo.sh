#!/usr/bin/env bash
# Runs on every codespace start: the real PGW stack plus the demo UI on port 8080.
# Masks and demo keys live in ~/pgw-demo and stay in this codespace.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python pgw/scripts/client_demo/run.py --host 127.0.0.1 --port 8080 \
    --pack-dir "$HOME/pgw-pack" --work "$HOME/pgw-demo" \
    --slots 300 --refill-low 250 --refill-target 600
