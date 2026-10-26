#!/usr/bin/env bash
# Counterexample-guided training to a proof, across lengths and weight formats.
# Each run writes model.npz, history.json and (if proved) certificate.json.
cd "$(dirname "$0")/.."
run() { name=$1; shift
  [ -f results/cegis/$name/history.json ] && { echo "skip $name"; return; }
  echo "== $name"; python3 -m vt prove --out results/cegis/$name "$@" 2>&1 | grep --line-buffered -v Highs
}
run L16_float        --length 16 --float --rounds 10 --verify-time 300
run L24_float        --length 24 --float --rounds 25 --verify-time 600
run L32_float        --length 32 --float --rounds 25 --verify-time 900
run L16_ternary_qat  --length 16 --qat-from-round 1 --rounds 12 --verify-time 300
run L32_causal_float --length 32 --float --causal --rounds 10 --verify-time 900
