#!/usr/bin/env bash
# WikiText-2 (Merity et al., 2016; CC BY-SA), from the PyTorch examples repository.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/wikitext-2
base=https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2
for f in train valid test; do
  [ -f data/wikitext-2/$f.txt ] || curl -fsS -o data/wikitext-2/$f.txt "$base/$f.txt"
done
sha256sum -c <<'SUMS'
9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f  data/wikitext-2/train.txt
f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8  data/wikitext-2/valid.txt
d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0  data/wikitext-2/test.txt
SUMS
