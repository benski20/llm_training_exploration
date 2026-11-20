# syntax=docker/dockerfile:1.7
# PGW gateway (customer side), CPU image.
#
# Runs the OpenAI-compatible API, the plaintext engine and the mask store client. Also contains
# the pack tool (`pgw pack`) and the mask store admin commands.
#
#   docker build -f deploy/docker/gateway.Dockerfile -t pgw-gateway:dev .      # from the pgw/ directory
#
# Build context: the pgw/ project root (pyproject.toml, src/, deploy/).

# Reproducible builds: the base image is pinned by digest and every Python dependency (torch included)
# by version and sha256 in requirements/gateway.txt (refresh both with scripts/lock-requirements.sh).
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

FROM ${PYTHON_IMAGE} AS build
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/pgw
ENV PATH=/opt/pgw/bin:$PATH
COPY requirements/gateway.txt /tmp/requirements.txt
# torch==${TORCH_VERSION}+cpu and all other dependencies, hash-checked (no resolver, no "latest")
RUN pip install --require-hashes --no-deps --extra-index-url "${TORCH_INDEX_URL}" -r /tmp/requirements.txt
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation ".[gateway,pack]" \
    && pip check \
    && pgw --help > /dev/null

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="pgw-gateway" \
      org.opencontainers.image.description="PGW private gateway: OpenAI-compatible API with one-time-mask inference" \
      org.opencontainers.image.licenses="Proprietary"
RUN groupadd --system --gid 10001 pgw \
    && useradd --system --uid 10001 --gid pgw --home-dir /var/lib/pgw --shell /usr/sbin/nologin pgw \
    && mkdir -p /var/lib/pgw/packs /var/lib/pgw/state /var/lib/pgw/masks /var/lib/pgw/counter /var/lib/pgw/secrets /etc/pgw /run/pgw \
    && chown -R pgw:pgw /var/lib/pgw /run/pgw \
    && chmod 0700 /var/lib/pgw/masks /var/lib/pgw/counter /var/lib/pgw/secrets
COPY --from=build /opt/pgw /opt/pgw
COPY --chmod=0755 deploy/docker/entrypoint.sh /usr/local/bin/pgw-entrypoint
COPY --chmod=0755 deploy/docker/healthcheck.py /usr/local/bin/pgw-healthcheck
ENV PATH=/opt/pgw/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MALLOC_TRIM_THRESHOLD_=268435456 \
    MALLOC_TOP_PAD_=67108864 \
    MALLOC_MMAP_THRESHOLD_=33554432 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/var/lib/pgw/state/hf \
    PGW_LOG_JSON=true \
    PGW_GATEWAY_HOST=0.0.0.0 \
    PGW_GATEWAY_PORT=8080 \
    PGW_GATEWAY_MASKSTORE_ROOT=/var/lib/pgw/masks \
    PGW_MASKSTORE_ROOT=/var/lib/pgw/masks
USER 10001:10001
WORKDIR /var/lib/pgw
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=6s --start-period=120s --retries=3 CMD ["pgw-healthcheck", "gateway"]
ENTRYPOINT ["pgw-entrypoint"]
CMD ["pgw", "gateway", "serve"]
