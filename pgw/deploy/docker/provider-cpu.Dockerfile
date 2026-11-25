# syntax=docker/dockerfile:1.7
# PGW provider image, CPU variant: the same service, accounts, native extension, pack tool and RunPod boot as
# provider.Dockerfile on the exact CPU backend. For tests and CI, small models, and the client-side test pod on
# RunPod (`pgw-runpod-boot client`: pgw and deploy/runpod/rtt_probe.py in the same data center as the GPU).
#
#   docker build -f deploy/docker/provider-cpu.Dockerfile -t pgw-provider-cpu:dev .
#
# Reproducible: base image by digest, dependencies hash-locked in requirements/provider-cpu.txt (torch 2.5.1+cpu),
# Rust toolchain by version. Optional `--secret id=build_ca,src=<CA bundle>` behind a TLS-inspecting proxy.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

# ---------------------------------------------------------------------------------------------------- build
FROM ${PYTHON_IMAGE} AS build
ARG RUST_TOOLCHAIN=1.89.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG REQUIREMENTS=requirements/provider-cpu.txt
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/pgw
ENV PATH=/opt/pgw/bin:/opt/cargo/bin:$PATH
COPY ${REQUIREMENTS} /tmp/requirements.txt
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    pip install --require-hashes --no-deps --extra-index-url "${TORCH_INDEX_URL}" -r /tmp/requirements.txt
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation ".[service,pack]" \
    && pip check \
    && pgw service --help > /dev/null
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
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="pgw-provider-cpu" \
      org.opencontainers.image.description="PGW provider: masked compute service with SDK-key accounts (CPU; tests)" \
      org.opencontainers.image.licenses="Proprietary"
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
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
    PGW_LOG_JSON=true \
    PGW_SERVICE_HOST=0.0.0.0 \
    PGW_SERVICE_PORT=8443 \
    PGW_SERVICE_DEVICE=cpu \
    PGW_HEALTHCHECK_SCHEME=https \
    PGW_DATA_DIR=/var/lib/pgw
WORKDIR /var/lib/pgw
EXPOSE 8443 8444
HEALTHCHECK --interval=10s --timeout=6s --start-period=600s --retries=3 CMD ["pgw-healthcheck", "--live", "service"]
ENTRYPOINT ["/usr/bin/tini", "--", "pgw-provider-entrypoint"]
CMD ["pgw", "service", "serve"]
