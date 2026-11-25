#!/bin/sh
# PGW provider entrypoint: never run PGW code as root.
#
# * `pgw-runpod-boot ...` (RunPod): the boot supervisor starts as root to prepare the volume and the PKI, and runs
#   the pack tool and the service as uid 10001 itself (deploy/runpod/pod_boot.py).
# * anything else as root: create and hand over PGW_DATA_DIR, then drop to uid/gid 10001 with no capabilities and
#   no_new_privs, and continue with the generic entrypoint (config rendering, exec).
# * not root (`docker run --user 10001`, Kubernetes runAsUser): straight to the generic entrypoint.
set -eu
if [ "$(id -u)" = "0" ]; then
    case "${1:-}" in
        pgw-runpod-boot) exec "$@" ;;
    esac
    data=${PGW_DATA_DIR:-/var/lib/pgw}
    mkdir -p "$data"
    chown 10001:10001 "$data"
    exec setpriv --reuid=10001 --regid=10001 --clear-groups --inh-caps=-all --no-new-privs -- pgw-entrypoint "$@"
fi
exec pgw-entrypoint "$@"
