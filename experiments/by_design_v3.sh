#!/usr/bin/env bash
# v3: ALiBi positions (length extrapolation) + lower intervention rate (less over-abstention)
cd "$(dirname "$0")/.."
python3 experiments/grounding_by_design.py --steps 6000 --names single --pos alibi --flip 0.15,0.5 \
    --out results/grounding_by_design_v3_alibi.json
python3 experiments/provenance_by_design.py --steps 6000 --names single --pos alibi \
    --out results/provenance_by_design_v3_alibi.json
