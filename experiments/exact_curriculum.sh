#!/usr/bin/env bash
# Length curriculum with exact proofs: warm-start each stage from the previous
# stage's proved artifact and extend the length range.
cd "$(dirname "$0")/.."
run() { name=$1; shift
  [ -f results/cegis/$name/history.json ] && { echo "skip $name"; return; }
  echo "== $name"; python3 -m vt prove --out results/cegis/$name "$@" 2>&1 | grep --line-buffered -v Highs
}
evens() { seq -s, 2 2 $(( $1 - 2 )); }
run exact_dyck_L2-128_curriculum --spec dyck --length 128 --also-lengths $(evens 128) --float --uniform \
    --init results/cegis/exact_dyck_L2-64/model.npz --lr 3e-3 --epochs-first 30 --rounds 30
run exact_dyck_L2-256_curriculum --spec dyck --length 256 --also-lengths $(evens 256) --float --uniform \
    --init results/cegis/exact_dyck_L2-128_curriculum/model.npz --lr 1e-3 --epochs-first 20 --epochs-round 10 --rounds 30
