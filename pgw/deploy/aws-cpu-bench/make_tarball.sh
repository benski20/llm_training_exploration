#!/bin/bash
# Builds pgw-cpu-bench.tar.gz (source only: no weights, keys or build output) for launch.sh.
set -euo pipefail
PGW=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$PWD/pgw-cpu-bench.tar.gz}
STAGE=$(mktemp -d)
cp -r "$PGW/src" "$PGW/rust" "$PGW/pyproject.toml" "$PGW/README.md" "$STAGE/"
[ -f "$PGW/LICENSE" ] && cp "$PGW/LICENSE" "$STAGE/"
mkdir -p "$STAGE/benchmarks" && cp "$PGW"/benchmarks/*.py "$STAGE/benchmarks/"
cp "$PGW/deploy/hf-space/constraints.txt" "$STAGE/constraints.txt"
find "$STAGE" \( -name __pycache__ -o -name target -o -name '*.so' -o -name '*.egg-info' \) -prune -exec rm -rf {} +
tar czf "$OUT" -C "$STAGE" .
rm -rf "$STAGE"
echo "$OUT ($(du -h "$OUT" | cut -f1))"
