#!/usr/bin/env bash
# Uniform-mixing models, proved in exact rational arithmetic at every listed length.
cd "$(dirname "$0")/.."
run() { name=$1; shift
  [ -f results/cegis/$name/history.json ] && { echo "skip $name"; return; }
  echo "== $name"; python3 -m vt prove --out results/cegis/$name "$@" 2>&1 | grep --line-buffered -v Highs
}
evens() { seq -s, 2 2 $(( $1 - 2 )); }
run exact_dyck_L2-64       --spec dyck   --length 64  --also-lengths $(evens 64)  --float --uniform --rounds 30
run exact_sorted_L2-16     --spec sorted --length 16  --also-lengths $(evens 16)  --float --uniform --rounds 30
run exact_dyck_L2-128      --spec dyck   --length 128 --also-lengths $(evens 128) --float --uniform --rounds 30
run exact_sorted_L2-32     --spec sorted --length 32  --also-lengths $(evens 32)  --float --uniform --rounds 30
run exact_dyck_L2-256      --spec dyck   --length 256 --also-lengths $(evens 256) --float --uniform --rounds 30
