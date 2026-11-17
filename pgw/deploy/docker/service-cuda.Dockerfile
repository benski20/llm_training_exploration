# syntax=docker/dockerfile:1.7
# PGW compute service, CUDA image (operated by us; sees only uniform ring noise).
#
# Stateless: holds public int8 weights of the served packs and answers POST /pgw/v1/compute with
# W8 x_hat mod 2^K (K/8 int8 limb GEMMs). Never contains mask store or factory code paths in use,
# never receives masks, seeds or keys.
#
#   docker build -f deploy/docker/service-cuda.Dockerfile -t pgw-service-cuda:dev .
#   docker run --gpus all -p 8443:8443 -v packs:/var/lib/pgw/packs:ro \
#     -e PGW_SERVICE_PACKS='["/var/lib/pgw/packs/llama3-8b"]' -e PGW_SERVICE_TOKENS_FILE=/etc/pgw/tenants/tokens ...

# Reproducible builds: the base image is pinned by digest and every Python dependency (torch included)
# by version and sha256 in requirements/service-cuda.txt (refresh both with scripts/lock-requirements.sh).
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

FROM ${PYTHON_IMAGE} AS build
ARG TORCH_VERSION=2.5.1
ARG TORCH_CUDA=cu124
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/pgw
ENV PATH=/opt/pgw/bin:$PATH
COPY requirements/service-cuda.txt /tmp/requirements.txt
# torch==${TORCH_VERSION}+${TORCH_CUDA} and all other dependencies, hash-checked (no resolver, no "latest")
RUN pip install --require-hashes --no-deps --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
        -r /tmp/requirements.txt
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation ".[service,gpu]" \
    && pip check \
    && pgw service --help > /dev/null

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="pgw-service-cuda" \
      org.opencontainers.image.description="PGW stateless masked compute service (NVIDIA GPU)" \
      org.opencontainers.image.licenses="Proprietary"
RUN groupadd --system --gid 10001 pgw \
    && useradd --system --uid 10001 --gid pgw --home-dir /var/lib/pgw --shell /usr/sbin/nologin pgw \
    && mkdir -p /var/lib/pgw/packs /etc/pgw/tenants /etc/pgw/tls \
    && chown -R pgw:pgw /var/lib/pgw
COPY --from=build /opt/pgw /opt/pgw
COPY --chmod=0755 deploy/docker/entrypoint.sh /usr/local/bin/pgw-entrypoint
COPY --chmod=0755 deploy/docker/healthcheck.py /usr/local/bin/pgw-healthcheck
ENV PATH=/opt/pgw/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PGW_LOG_JSON=true \
    PGW_SERVICE_HOST=0.0.0.0 \
    PGW_SERVICE_PORT=8443 \
    PGW_SERVICE_DEVICE=cuda \
    PGW_SERVICE_METRICS_PORT=9090 \
    PGW_SERVICE_METRICS_HOST=0.0.0.0 \
    PGW_SERVICE_METRICS_ON_MAIN_PORT=false
USER 10001:10001
WORKDIR /var/lib/pgw
EXPOSE 8443 9090
HEALTHCHECK --interval=15s --timeout=6s --start-period=300s --retries=3 CMD ["pgw-healthcheck", "service"]
ENTRYPOINT ["pgw-entrypoint"]
CMD ["pgw", "service", "serve"]
