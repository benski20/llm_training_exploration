#!/usr/bin/env bash
# Watch a running Kaggle kernel (no push) and download its outputs once it finishes.
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... bash kaggle/distill/watch.sh SLUG LOG OUTDIR
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
slug=$1; log=$2; out=$3
while true; do
  st=$(kaggle kernels status "$KAGGLE_USERNAME/$slug" 2>&1 | tail -1)
  echo "$(date -u +%H:%M:%S) $st" >> "$log"
  case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
  sleep 180
done
mkdir -p "$out"
kaggle kernels output "$KAGGLE_USERNAME/$slug" -p "$out" --force --file-pattern '(\.json|\.log|_head\.pt|_detector\.pt|_heads\.pt)$' >> "$log" 2>&1
echo "$(date -u +%H:%M:%S) outputs in $out" >> "$log"
