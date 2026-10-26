#!/usr/bin/env bash
# Stop the CEGIS sweep and any vt prove runs (by exact process name match).
for p in $(pgrep -f "^bash experiments/cegis_sweep.sh"); do kill "$p"; done
for p in $(pgrep -f "^python3 -m vt prove"); do kill "$p"; done
