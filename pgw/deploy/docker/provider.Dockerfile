# syntax=docker/dockerfile:1.7
# PGW provider image (production, NVIDIA GPU): the masked compute service with accounts (SDK keys), the native
# trip lane and the pack tool, plus the RunPod boot (deploy/runpod/pod_boot.py). Sees only uniform ring noise.
#
#   docker build -f deploy/docker/provider.Dockerfile -t pgw-provider:dev .
#   docker run --gpus all -p 8443:8443 -v pgw:/var/lib/pgw -e PGW_SERVICE_PACKS=... pgw-provider:dev   # any host
#   (RunPod: deploy/runpod/launch.py starts it with `pgw-runpod-boot`)
#
# Reproducible: the base image is pinned by digest, every Python dependency (torch 2.5.1+cu124 and triton included)
# by version and sha256 in requirements/provider-cuda.txt (scripts/lock-requirements.sh), the Rust toolchain of the
# native extension by version. Behind a TLS-inspecting proxy add `--secret id=build_ca,src=<CA bundle>` (optional).
#
# Stages: `build` (compilers, Rust, headers: builds pgw and the native extension pgw_fast into /opt/pgw) and
# `runtime` (the venv, Python, gcc for Triton's kernel launcher stubs, no Rust, no build caches). The service runs
# as uid 10001: the entrypoint drops root after preparing the data directory (RunPod starts containers as root).

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-base-ubuntu24.04@sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e

# ---------------------------------------------------------------------------------------------------- build
FROM ${CUDA_IMAGE} AS build
ARG RUST_TOOLCHAIN=1.89.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG REQUIREMENTS=requirements/provider-cuda.txt
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev build-essential curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/pgw
ENV PATH=/opt/pgw/bin:/opt/cargo/bin:$PATH
COPY ${REQUIREMENTS} /tmp/requirements.txt
# all dependencies hash-checked (no resolver, no "latest"); the optional build CA only for a TLS-inspecting proxy
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    pip install --require-hashes --no-deps --extra-index-url "${TORCH_INDEX_URL}" -r /tmp/requirements.txt
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation ".[service,gpu,pack]" \
    && pip check \
    && pgw service --help > /dev/null
# native extension pgw_fast (lane, decode helpers): pinned toolchain, Cargo.lock honoured
COPY rust ./rust
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export CURL_CA_BUNDLE=/run/secrets/build_ca \
       SSL_CERT_FILE=/run/secrets/build_ca CARGO_HTTP_CAINFO=/run/secrets/build_ca; fi; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && cd /src/rust && cargo fetch --locked && cd /src \
    && python rust/tools/build_fast.py --site \
    && rm -rf /src/rust/target /opt/cargo /opt/rustup /tmp/rustup-init.sh \
    && cd / && python -c "from pgw.core import native; assert native.fast is not None, 'pgw_fast missing'; \
print('pgw_fast', native.fast.version())"

# -------------------------------------------------------------------------------------------------- runtime
FROM ${CUDA_IMAGE} AS runtime
LABEL org.opencontainers.image.title="pgw-provider" \
      org.opencontainers.image.description="PGW provider: masked compute service with SDK-key accounts (NVIDIA GPU)" \
      org.opencontainers.image.licenses="Proprietary"
ENV DEBIAN_FRONTEND=noninteractive
# python3.12 for the venv; gcc + libc6-dev + python3.12-dev: Triton compiles a small launcher stub per kernel at
# first use (the fused ring GEMV); tini as PID 1 (signals reach the service, zombies reaped)
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev gcc libc6-dev tini ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 pgw \
    && useradd --system --uid 10001 --gid pgw --home-dir /var/lib/pgw --shell /usr/sbin/nologin pgw \
    && mkdir -p /var/lib/pgw/packs /etc/pgw/tenants /etc/pgw/tls /workspace \
    && chown -R pgw:pgw /var/lib/pgw
COPY --from=build /opt/pgw /opt/pgw
COPY --chmod=0755 deploy/docker/entrypoint.sh /usr/local/bin/pgw-entrypoint
COPY --chmod=0755 deploy/docker/provider-entrypoint.sh /usr/local/bin/pgw-provider-entrypoint
COPY --chmod=0755 deploy/docker/healthcheck.py /usr/local/bin/pgw-healthcheck
COPY --chmod=0755 deploy/runpod/pod_boot.py /usr/local/bin/pgw-runpod-boot
COPY --chmod=0644 deploy/runpod/pod_boot.py deploy/runpod/rtt_probe.py /opt/pgw-deploy/runpod/
ENV PATH=/opt/pgw/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_TRIM_THRESHOLD_=268435456 \
    MALLOC_TOP_PAD_=67108864 \
    MALLOC_MMAP_THRESHOLD_=33554432 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PGW_LOG_JSON=true \
    PGW_SERVICE_HOST=0.0.0.0 \
    PGW_SERVICE_PORT=8443 \
    PGW_SERVICE_DEVICE=cuda \
    PGW_SERVICE_METRICS_PORT=9090 \
    PGW_SERVICE_METRICS_HOST=127.0.0.1 \
    PGW_HEALTHCHECK_SCHEME=https \
    PGW_DATA_DIR=/var/lib/pgw
WORKDIR /var/lib/pgw
EXPOSE 8443 8444
HEALTHCHECK --interval=15s --timeout=6s --start-period=1800s --retries=3 CMD ["pgw-healthcheck", "--live", "service"]
# Starts as root only to prepare the data directory (and, under `pgw-runpod-boot`, the volume and PKI); every
# PGW process runs as uid 10001. `docker run --user 10001` works as well.
ENTRYPOINT ["/usr/bin/tini", "--", "pgw-provider-entrypoint"]
CMD ["pgw", "service", "serve"]
