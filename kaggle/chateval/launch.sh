#!/usr/bin/env bash
# Chat-level evaluation of VeriChat with a risk head on Kaggle GPUs (false alarms on everyday chat,
# detection on factual questions, speed).
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... HEAD=results/.../X_risk_head.pt MODEL=... bash kaggle/chateval/launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"; : "${HEAD:?}"; : "${MODEL:?}"
U=$KAGGLE_USERNAME; SLUG=${SLUG:-chateval}; OUT=${OUT:-results/kaggle/chateval}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
SERVING="${HEAD%.pt}.serving.json"; [ -f "$SERVING" ] || SERVING=""          # the head's measured serving settings
BUNDLE_FILES="$HEAD $SERVING" python3 kaggle/distill/bundle.py "$S/run.py" kaggle/chateval/run.py \
    --model "$MODEL" --head "$(basename "$HEAD")" --out /kaggle/working/chateval ${EXTRA_ARGS:-}
cat > "$S/kernel-metadata.json" <<META
{"id": "$U/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "machine_shape": "NvidiaTeslaT4", "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
kaggle kernels push -p "$S"
bash kaggle/distill/watch.sh "$SLUG" "results/kaggle/${SLUG}.log" "$OUT"
