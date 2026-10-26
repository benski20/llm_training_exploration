#!/usr/bin/env bash
# Sortedness (V=4) with window-2 uniform mixing, proved exactly at every listed length.
cd "$(dirname "$0")/.."
run() { name=$1; shift
  [ -f results/cegis/$name/history.json ] && { echo "skip $name"; return; }
  echo "== $name"; python3 -m vt prove --out results/cegis/$name "$@" --threads 1 --workers 1 2>&1 | grep --line-buffered -v Highs
}
evens() { seq -s, 2 2 $(( $1 - 2 )); }
run exact_sorted_w2_L2-16 --spec sorted --length 16 --also-lengths $(evens 16) --float --uniform --window 2 --rounds 30
run exact_sorted_w2_L2-32 --spec sorted --length 32 --also-lengths $(evens 32) --float --uniform --window 2 --rounds 30
