#!/usr/bin/env bash
# Assemble the Space: the vt modules VeriChat needs and the trained 7B claim head.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p vt heads
for m in __init__ correctness claims guided verichat qwen_glassbox; do
  if [ -f ../vt/$m.py ]; then cp ../vt/$m.py vt/; else : > vt/$m.py; fi
done
cp ../results/kaggle/claims-7b/claims/Qwen2.5-7B-Instruct_claims_head.pt heads/
echo "space ready: $(pwd)"
