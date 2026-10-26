#!/usr/bin/env bash
# Stanford Alpaca instruction data (CC BY-NC 4.0: research use only), pinned by checksum.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/alpaca
[ -f data/alpaca/alpaca_data.json ] || curl -fsS -o data/alpaca/alpaca_data.json \
  https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
echo "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419  data/alpaca/alpaca_data.json" | sha256sum -c
