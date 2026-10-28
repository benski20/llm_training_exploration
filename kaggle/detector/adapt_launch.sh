#!/usr/bin/env bash
# Few-shot adaptation experiment on Kaggle GPUs: adapt a trained detector head to unseen HaluBench domains.
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... HEAD=results/.../X_detector.pt MODEL=Qwen/... bash kaggle/detector/adapt_launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"; : "${HEAD:?}"; : "${MODEL:?}"
U=$KAGGLE_USERNAME; SLUG=${SLUG:-detector-adapt}; OUT=${OUT:-results/kaggle/detector-adapt}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
BUNDLE_FILES="$HEAD" python3 kaggle/distill/bundle.py "$S/run.py" kaggle/detector/adapt_eval.py \
    --model "$MODEL" --head "$(basename "$HEAD")" --out /kaggle/working/adapt ${EXTRA_ARGS:-}
cat > "$S/kernel-metadata.json" <<META
{"id": "$U/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "machine_shape": "NvidiaTeslaT4", "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
kaggle kernels push -p "$S"
bash kaggle/distill/watch.sh "$SLUG" "results/kaggle/${SLUG}.log" "$OUT"
