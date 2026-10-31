#!/usr/bin/env bash
# The grounding probe on the served 32B, as a private Kaggle kernel on two free T4s; waits for it and downloads
# the probe and its results (the test-feature cache stays on Kaggle).
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... bash kaggle/grounding/launch.sh
# The token is read from the environment only; nothing here writes it to disk.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
U=$KAGGLE_USERNAME; SLUG=${SLUG:-grounding-32b}; OUT=${OUT:-results/kaggle/grounding-32b}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
BUNDLE_FILES="experiments/grounding_probe.py experiments/evidence_shift.py experiments/recall_neurons.py" \
    python3 kaggle/distill/bundle.py "$S/run.py" kaggle/grounding/run.py ${EXTRA_ARGS:-}
cat > "$S/kernel-metadata.json" <<META
{"id": "$U/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "machine_shape": "NvidiaTeslaT4", "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
kaggle kernels push -p "$S"
bash kaggle/distill/watch.sh "$SLUG" "results/kaggle/${SLUG}.log" "$OUT"
kaggle kernels output "$U/$SLUG" -p "$OUT" --force --file-pattern 'grounding_probe_.*\.npz$' >> "results/kaggle/${SLUG}.log" 2>&1
echo "probe: $(ls "$OUT"/grounding/grounding_probe_*.npz "$OUT"/grounding_probe_*.npz 2>/dev/null)"
