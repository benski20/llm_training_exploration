#!/usr/bin/env bash
# Inside a measurement job pod (pod_job.py runs it from the extracted bundle): the provider image's build steps on a
# public base image. Python 3.12 venv /opt/pgw with every dependency hash-checked from the lock
# (requirements/provider-cuda.txt or provider-cpu.txt), pgw itself, and the three native extensions (pgw_fast,
# pgw_ntt, pgw_pks) built with the pinned Rust toolchain (rustup-init checked by sha256, as in the Dockerfiles).
#
#   bash deploy/runpod/job_setup.sh gpu|cpu        (from pgw/; idempotent: a finished setup is reused)
#   JOB_SETUP_DRY_RUN=1 bash deploy/runpod/job_setup.sh cpu    # print the plan only
set -euo pipefail
role="${1:?usage: job_setup.sh gpu|cpu}"
cd "$(dirname "$0")/../.."
RUST_TOOLCHAIN=1.89.0
RUSTUP_VERSION=1.28.2
RUSTUP_INIT_SHA256_X86_64=20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c
case "$role" in
    gpu) req=requirements/provider-cuda.txt; idx=https://download.pytorch.org/whl/cu132 ;;
    cpu) req=requirements/provider-cpu.txt; idx=https://download.pytorch.org/whl/cpu ;;
    *) echo "role must be gpu or cpu" >&2; exit 2 ;;
esac
[ -f "$req" ] || { echo "missing $req (run from pgw/)" >&2; exit 2; }
if [ "${JOB_SETUP_DRY_RUN:-0}" = 1 ]; then
    echo "plan: apt build tools + python3.12; venv /opt/pgw; pip --require-hashes -r $req (index $idx);"
    echo "      pip install --no-build-isolation . ; rustup-init $RUSTUP_VERSION (sha256 $RUSTUP_INIT_SHA256_X86_64)"
    echo "      toolchain $RUST_TOOLCHAIN; cargo build --release --locked -p pgw-fast -p pgw-ntt -p pgw-pks"
    exit 0
fi
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH="/opt/pgw/bin:/opt/cargo/bin:$PATH"
if [ -x /opt/pgw/bin/python ] && /opt/pgw/bin/python -c "import pgw, pgw_fast, pgw_ntt, pgw_pks" 2>/dev/null; then
    echo "setup: already done"; exit 0
fi
t0=$(date +%s)
[ "$(uname -m)" = x86_64 ] || { echo "only x86_64 pods are pinned here" >&2; exit 2; }
pkgs="build-essential curl ca-certificates"
command -v python3.12 >/dev/null || pkgs="$pkgs python3.12 python3.12-venv python3.12-dev"
apt-get update -qq
apt-get install -y -qq --no-install-recommends $pkgs >/dev/null
python3.12 -m venv /opt/pgw
pip install -q --require-hashes --no-deps --extra-index-url "$idx" -r "$req"
pip install -q --no-deps --no-build-isolation .
echo "setup: python dependencies and pgw in $(( $(date +%s) - t0 )) s"
if [ ! -x /opt/cargo/bin/cargo ]; then
    curl --proto '=https' --tlsv1.2 -sSf -o /tmp/rustup-init \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init"
    echo "${RUSTUP_INIT_SHA256_X86_64}  /tmp/rustup-init" | sha256sum -c -
    chmod 0755 /tmp/rustup-init
    /tmp/rustup-init -y -q --no-modify-path --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
    rm -f /tmp/rustup-init
fi
(cd rust && PYO3_PYTHON=/opt/pgw/bin/python cargo build -q --release --locked -p pgw-fast -p pgw-ntt -p pgw-pks)
site="$(/opt/pgw/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for m in pgw_fast pgw_ntt pgw_pks; do
    cp "rust/target/release/lib$m.so" "src/$m.abi3.so"      # benchmarks run with PYTHONPATH=src
    cp "rust/target/release/lib$m.so" "$site/$m.abi3.so"    # the installed package (pgw service, pod_boot)
done
(cd / && /opt/pgw/bin/python -c "import pgw_fast, pgw_ntt, pgw_pks, torch; print('native ok; torch', torch.__version__, \
'cuda', torch.cuda.is_available())")
echo "setup: done in $(( $(date +%s) - t0 )) s"
