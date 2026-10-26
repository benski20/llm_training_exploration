#!/usr/bin/env bash
# Weight-tied (position-shared) MixerNet: Toeplitz causal mixing + pooled readout.
cd "$(dirname "$0")/.."
run() { name=$1; shift
  [ -f results/cegis/$name/history.json ] && { echo "skip $name"; return; }
  echo "== $name"; python3 -m vt prove --out results/cegis/$name "$@" 2>&1 | grep --line-buffered -v Highs
}
run L16_shared --length 16 --float --shared --lr-decay 0.92 --epochs-first 200 --epochs-round 30 --rounds 15 --verify-time 300
run L32_shared --length 32 --float --shared --lr-decay 0.92 --epochs-first 200 --epochs-round 30 --rounds 30 --verify-time 600
run L64_shared --length 64 --float --shared --lr-decay 0.92 --epochs-first 200 --epochs-round 30 --rounds 30 --verify-time 900
