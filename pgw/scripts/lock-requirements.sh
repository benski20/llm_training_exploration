#!/bin/sh
# Regenerate the hashed dependency locks in requirements/ (one per image role) with uv.
#
# Every Docker image installs exactly these files with `pip install --require-hashes --no-deps`,
# so a new upstream release (compromised or broken) never enters an image by accident. Review the
# diff of requirements/*.txt like code; CI runs pip-audit on these locks.
#
# Usage (from the pgw/ directory): scripts/lock-requirements.sh [--upgrade]
# Base image digests are pinned in deploy/docker/*.Dockerfile (ARG PYTHON_IMAGE=...@sha256:...);
# refresh them deliberately, e.g. `docker buildx imagetools inspect python:3.12-slim-bookworm`.
set -eu
cd "$(dirname "$0")/.."
UPGRADE=""
[ "${1:-}" = "--upgrade" ] && UPGRADE="--upgrade"
CPU=https://download.pytorch.org/whl/cpu
CU=https://download.pytorch.org/whl/cu124

lock() {
    out=$1; index=$2; torch=$3; shift 3
    uv pip compile pyproject.toml "$@" $UPGRADE --generate-hashes \
        --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
        --extra-index-url "$index" --index-strategy unsafe-best-match -c "$torch" \
        --custom-compile-command "scripts/lock-requirements.sh" -o "requirements/$out.txt"
}

lock gateway      "$CPU" requirements/torch-cpu.in   --extra gateway --extra pack
lock service-cpu  "$CPU" requirements/torch-cpu.in   --extra service
lock factory-cpu  "$CPU" requirements/torch-cpu.in   --extra factory
lock service-cuda "$CU"  requirements/torch-cu124.in --extra service --extra gpu
lock factory-cuda "$CU"  requirements/torch-cu124.in --extra factory --extra gpu
# provider images (deploy/docker/provider*.Dockerfile): the service plus the pack tool, which builds packs into the
# provider's volume on first boot; the same versions as the service and gateway locks
lock provider-cpu  "$CPU" requirements/torch-cpu.in   --extra service --extra pack -c requirements/service-cpu.txt \
    -c requirements/pack-pins.in
lock provider-cuda "$CU"  requirements/torch-cu124.in --extra service --extra gpu --extra pack \
    -c requirements/service-cuda.txt -c requirements/pack-pins.in
