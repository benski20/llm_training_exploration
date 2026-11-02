#!/usr/bin/env bash
# The RAGTruth state-of-the-art attempt on a free Kaggle T4; waits for it and downloads results.json.
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... bash kaggle/ragtruth_sota/launch.sh
#   EXTRA_ARGS="--reader unsloth/Qwen3-32B-bnb-4bit --train 3000" ...      # the 32B (slower)
# The token is read from the environment only; nothing here writes it to disk.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
U=$KAGGLE_USERNAME; SLUG=${SLUG:-ragtruth-sota}; OUT=${OUT:-results/kaggle/ragtruth-sota}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
BUNDLE_FILES="experiments/ragtruth_sota.py experiments/evidence_shift.py" \
    python3 kaggle/distill/bundle.py "$S/run.py" kaggle/ragtruth_sota/run.py ${EXTRA_ARGS:-}
cat > "$S/kernel-metadata.json" <<META
{"id": "$U/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "machine_shape": "NvidiaTeslaT4", "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
kaggle kernels push -p "$S"
bash kaggle/distill/watch.sh "$SLUG" "results/kaggle/${SLUG}.log" "$OUT"
