#!/usr/bin/env bash
# Convert Qwen2.5-0.5B-Instruct into a glass-box model on a Kaggle T4 (uses kernel 1's teacher data).
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... bash kaggle/qwen_glassbox/launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
U=$KAGGLE_USERNAME; SLUG=${SLUG:-qwen-glassbox}; OUT=${OUT:-results/kaggle/qwen-glassbox}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
python3 kaggle/distill/bundle.py "$S/run.py" kaggle/qwen_glassbox/run.py --data /kaggle/input --out /kaggle/working/qwen_glassbox ${EXTRA_ARGS:-}
cat > "$S/kernel-metadata.json" <<META
{"id": "$U/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "machine_shape": "NvidiaTeslaT4", "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": ["$U/glassbox-distill-data"], "model_sources": []}
META
kaggle kernels push -p "$S"
bash kaggle/distill/watch.sh "$SLUG" "results/kaggle/${SLUG}.log" "$OUT"
