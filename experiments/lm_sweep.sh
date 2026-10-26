#!/usr/bin/env bash
# CPU LM sweep: {float, ternary} x {no margin, margin}. Resumable.
set -e
cd "$(dirname "$0")/.."
DATA=${DATA:-data/tinyshakespeare.txt}
[ -f "$DATA" ] || curl -sS -o "$DATA" https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
COMMON="--data $DATA --steps ${STEPS:-3000} --margin-eps 0.05"
for v in "base:" "margin:--margin-weight 0.5" "ternary:--ternary" "ternary_margin:--ternary --margin-weight 0.5"; do
  name=${v%%:*}; flags=${v#*:}
  [ -f runs/lm_$name/metrics.json ] && { echo "skip $name"; continue; }
  echo "== $name"; python3 experiments/train_lm.py $COMMON --out runs/lm_$name $flags
done
