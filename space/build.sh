#!/usr/bin/env bash
# Assemble the Space: the vt modules VeriChat needs, the trained 7B claim head and the detector head.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p vt heads
for m in __init__ correctness claims guided verichat qwen_glassbox detector retrieval; do
  if [ -f ../vt/$m.py ]; then cp ../vt/$m.py vt/; else : > vt/$m.py; fi
done
cp ../results/kaggle/claims-7b/claims/Qwen2.5-7B-Instruct_claims_head.pt heads/
cp ../results/kaggle/detector-3b/detector/Qwen2.5-3B-Instruct_detector.pt heads/
echo "space ready: $(pwd)"
