#!/usr/bin/env bash
# Typed (copy-only facts) vs. unconstrained baseline on WikiText-2, 2 threads each, in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
bash experiments/fetch_wikitext2.sh
mkdir -p results/typed_wikitext
for M in typed baseline; do
  OMP_NUM_THREADS=2 python3 experiments/typed_wikitext.py --model $M --threads 2 \
      --out results/typed_wikitext/$M.json > results/typed_wikitext/$M.log 2>&1 &
done
wait
