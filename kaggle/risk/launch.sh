#!/usr/bin/env bash
# Train the learned risk head (and claim probe) for a chat model on Kaggle GPUs.
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... bash kaggle/risk/launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
U=$KAGGLE_USERNAME; SLUG=${SLUG:-risk}; OUT=${OUT:-results/kaggle/risk}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
python3 kaggle/distill/bundle.py "$S/run.py" kaggle/risk/run.py --out /kaggle/working/risk ${EXTRA_ARGS:-}
cat > "$S/kernel-metadata.json" <<META
{"id": "$U/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "machine_shape": "NvidiaTeslaT4", "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
kaggle kernels push -p "$S"
bash kaggle/distill/watch.sh "$SLUG" "results/kaggle/${SLUG}.log" "$OUT"
