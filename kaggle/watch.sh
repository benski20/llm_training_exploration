#!/usr/bin/env bash
# Wait for an already-pushed Kaggle kernel to finish, then download its outputs.
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... bash kaggle/watch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
SLUG=${KAGGLE_SLUG:-verifiable-chat-retrofit}
OUT=${OUT:-results/kaggle/$SLUG}
while true; do
  st=$(kaggle kernels status "$KAGGLE_USERNAME/$SLUG" 2>&1 || true)
  echo "$(date -u +%H:%M:%S) $st"
  case "$st" in
    *COMPLETE*|*complete*) break ;;
    *ERROR*|*error*|*CANCEL*|*cancel*) echo "kernel did not complete"; break ;;
  esac
  sleep 120
done
mkdir -p "$OUT"
if [ "${WITH_MODEL:-0}" = 1 ]; then
  kaggle kernels output "$KAGGLE_USERNAME/$SLUG" -p "$OUT" --force
else
  kaggle kernels output "$KAGGLE_USERNAME/$SLUG" -p "$OUT" --force --file-pattern '(results\.json|\.log)$'
fi
echo "outputs in $OUT"
