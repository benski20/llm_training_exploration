#!/usr/bin/env bash
# Glassbox-2 distillation on Kaggle GPUs: kernel 1 (teacher data), then kernel 2 (students).
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... bash kaggle/distill/launch.sh
# DATA_HOURS / STUDENT_HOURS set the time budgets; SKIP_DATA=1 reuses an existing kernel 1 output.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
U=$KAGGLE_USERNAME
DATA_HOURS=${DATA_HOURS:-2.0}; STUDENT_HOURS=${STUDENT_HOURS:-9.0}
OUT=results/kaggle/glassbox-distill

wait_for() {
  while true; do
    st=$(kaggle kernels status "$1" 2>&1 || true)
    echo "$(date -u +%H:%M:%S) $st"
    case "$st" in
      *COMPLETE*|*complete*) return 0 ;;
      *ERROR*|*error*|*CANCEL*|*cancel*) echo "kernel $1 did not complete"; return 1 ;;
    esac
    sleep 180
  done
}

push() {  # push DIR SLUG TITLE CODE SOURCES
  cat > "$1/kernel-metadata.json" <<META
{"id": "$U/$2", "title": "$3", "code_file": "$4", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "machine_shape": "NvidiaTeslaT4", "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": $5, "model_sources": []}
META
  kaggle kernels push -p "$1"
}

S1=$(mktemp -d); S2=$(mktemp -d); trap 'rm -rf "$S1" "$S2"' EXIT
if [ "${SKIP_DATA:-0}" != 1 ]; then
  python3 kaggle/distill/bundle.py "$S1/gen_data.py" kaggle/distill/gen_data.py --hours "$DATA_HOURS"
  push "$S1" glassbox-distill-data "Glassbox distill data" gen_data.py "[]"
  wait_for "$U/glassbox-distill-data"
fi
python3 kaggle/distill/bundle.py "$S2/train_student.py" kaggle/distill/train_student.py --data /kaggle/input \
    --hours "$STUDENT_HOURS"
push "$S2" glassbox-distill-train "Glassbox distill train" train_student.py "[\"$U/glassbox-distill-data\"]"
wait_for "$U/glassbox-distill-train" || true
mkdir -p "$OUT"
kaggle kernels output "$U/glassbox-distill-train" -p "$OUT" --force --file-pattern '(\.json|\.log)$'
echo "outputs in $OUT"
