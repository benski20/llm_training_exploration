#!/usr/bin/env bash
# Download the trained glass-box student package from the finished Kaggle kernel 2.
#   KAGGLE_API_TOKEN=... KAGGLE_USERNAME=... [SLUG=glassbox-distill-g24 OUT=results/kaggle/glassbox-distill-g24] \
#       bash kaggle/distill/fetch_package.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"; : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
SLUG=${SLUG:-glassbox-distill-train}; OUT=${OUT:-results/kaggle/glassbox-distill}
mkdir -p "$OUT/pkg_download"
kaggle kernels output "$KAGGLE_USERNAME/$SLUG" -p "$OUT/pkg_download" --force \
    --file-pattern 'glassbox_package/'
src=$(find "$OUT/pkg_download" -type d -name glassbox_package | head -1)
[ -n "$src" ] || { echo "no glassbox_package in the kernel output"; exit 1; }
rm -rf "$OUT/glassbox_package"; mv "$src" "$OUT/glassbox_package"; rm -rf "$OUT/pkg_download"
python3 -c "from vt.chat_serve import load_responder; r = load_responder('$OUT/glassbox_package'); print('package ok:', r.manifest['format'], r.signals)"
