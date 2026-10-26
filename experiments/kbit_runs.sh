#!/usr/bin/env bash
# k-bit QAT runs: does snapping weights to a grid make long-length proofs reachable?
cd "$(dirname "$0")/.."
run() { name=$1; shift
  [ -f results/cegis/$name/history.json ] && { echo "skip $name"; return; }
  echo "== $name"; python3 -m vt prove --out results/cegis/$name "$@" --threads 1 --workers 2 2>&1 | grep --line-buffered -v Highs
}
run L16_causal_4bit --length 16 --float --causal --bits 4 --qat-from-round 1 --rounds 15 --verify-time 300
run L32_causal_6bit --length 32 --float --causal --bits 6 --qat-from-round 3 --rounds 30 --verify-time 600
