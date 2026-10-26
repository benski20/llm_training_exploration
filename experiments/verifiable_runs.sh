#!/usr/bin/env bash
# Typed + bottlenecked + pointer answerer: channel after block 1 vs. after the last block.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/verifiable_lm.py --steps 6000 --answer-only --layer 1 --out results/verifiable_lm_l1.json
python3 experiments/verifiable_lm.py --steps 6000 --answer-only --layer 3 --out results/verifiable_lm_l3.json
