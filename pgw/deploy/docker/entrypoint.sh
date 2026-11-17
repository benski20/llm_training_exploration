#!/bin/sh
# PGW container entrypoint.
#
# 1. Renders config templates that reference secrets (PGW_RENDER="src:dst[,src:dst...]") with
#    `pgw config render` (expands ${env:NAME} and ${file:PATH}; output mode 0600). This keeps
#    secrets such as the compute-service token out of ConfigMaps, images and command lines.
# 2. Execs the command (default: the image's CMD), so signals reach pgw directly (graceful drain).
set -eu

if [ -n "${PGW_RENDER:-}" ]; then
    old_ifs=$IFS
    set -f    # split on commas only: never glob-expand paths in PGW_RENDER
    IFS=,
    for pair in $PGW_RENDER; do
        src=${pair%%:*}
        dst=${pair#*:}
        if [ -z "$src" ] || [ -z "$dst" ] || [ "$src" = "$pair" ]; then
            echo "pgw-entrypoint: bad PGW_RENDER entry '$pair' (expected src:dst)" >&2
            exit 1
        fi
        pgw config render "$src" "$dst"
    done
    IFS=$old_ifs
    set +f
fi

exec "$@"
