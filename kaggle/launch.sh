#!/usr/bin/env bash
# Run the verifiable-chat retrofit (kaggle/verifiable_chat/train.py) as a private Kaggle
# kernel on a free T4 GPU, wait for it, and download its outputs.
#
#   KAGGLE_API_TOKEN=... [KAGGLE_USERNAME=...] bash kaggle/launch.sh [extra train.py args]
#
# The token is read from the environment only; nothing here writes it to disk.
# Needs network access to www.kaggle.com, api.kaggle.com and storage.googleapis.com.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN to a Kaggle API token}"
SLUG=${KAGGLE_SLUG:-verifiable-chat-retrofit}
OUT=${OUT:-results/kaggle/$SLUG}

if [ -z "${KAGGLE_USERNAME:-}" ]; then
  KAGGLE_USERNAME=$(python3 - <<'EOF'
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print(api.config_values.get("username") or "")
EOF
)
fi
: "${KAGGLE_USERNAME:?could not determine the Kaggle username; set KAGGLE_USERNAME}"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp kaggle/verifiable_chat/train.py "$STAGE/train.py"
if [ "$#" -gt 0 ]; then                       # bake extra arguments into the kernel
  python3 - "$STAGE/train.py" "$@" <<'EOF'
import sys
p, extra = sys.argv[1], sys.argv[2:]
src = open(p).read().replace('    a = ap.parse_args()', f'    a = ap.parse_args(sys.argv[1:] or {extra!r})', 1)
open(p, "w").write(src)
EOF
fi
cat > "$STAGE/kernel-metadata.json" <<EOF
{"id": "$KAGGLE_USERNAME/$SLUG", "title": "Verifiable chat retrofit", "code_file": "train.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true,
 "machine_shape": "NvidiaTeslaT4", "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
EOF
kaggle kernels push -p "$STAGE"
echo "pushed $KAGGLE_USERNAME/$SLUG; waiting for it to finish"
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
if [ "${WITH_MODEL:-0}" = 1 ]; then              # the merged model is about 3 GB
  kaggle kernels output "$KAGGLE_USERNAME/$SLUG" -p "$OUT" --force
else
  kaggle kernels output "$KAGGLE_USERNAME/$SLUG" -p "$OUT" --force --file-pattern '(results\.json|\.log)$'
fi
echo "outputs in $OUT"
