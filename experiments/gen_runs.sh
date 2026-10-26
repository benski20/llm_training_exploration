#!/usr/bin/env bash
# Certified generators: support proved equal to the spec language at every listed length.
cd "$(dirname "$0")/.."
run() { name=$1; shift
  [ -f results/gen/$name/certificate.json ] && { echo "skip $name"; return; }
  echo "== $name"; python3 -m vt gen-train --out results/gen/$name "$@" 2>&1 | grep --line-buffered -v Highs
}
evens() { seq -s, 2 2 $(( $1 - 2 )); }
run dyck_L2-64     --spec dyck --length 64 --also-lengths $(evens 64) --steps 4000
run sorted_w2_L2-16 --spec sorted --length 16 --also-lengths $(evens 16) --window 2 --steps 4000
run no_bb_L2-32    --spec-file specs/no_bb.json --length 32 --also-lengths $(evens 32) --window 2 --steps 3000
