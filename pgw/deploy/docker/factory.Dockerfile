# syntax=docker/dockerfile:1.7
# PGW mask factory, CPU image (customer-run, or an independent partner).
#
# Generates one-time masks (W8 r mod 2^K + Freivalds vectors) into an encrypted mask store, or
# exports HPKE-sealed segments for a customer (partner mode). Needs no network access.
#
#   docker build -f deploy/docker/factory.Dockerfile -t pgw-factory:dev .      # from the pgw/ directory
#
# GPU hosts: use factory-cuda.Dockerfile (about 100x faster mask generation).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/pgw
ENV PATH=/opt/pgw/bin:$PATH
RUN pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[factory]" \
    && pgw factory --help > /dev/null

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="pgw-factory" \
      org.opencontainers.image.description="PGW mask factory (CPU)" \
      org.opencontainers.image.licenses="Proprietary"
RUN groupadd --system --gid 10001 pgw \
    && useradd --system --uid 10001 --gid pgw --home-dir /var/lib/pgw --shell /usr/sbin/nologin pgw \
    && mkdir -p /var/lib/pgw/packs /var/lib/pgw/masks /var/lib/pgw/export /var/lib/pgw/secrets /etc/pgw \
    && chown -R pgw:pgw /var/lib/pgw \
    && chmod 0700 /var/lib/pgw/masks /var/lib/pgw/secrets
COPY --from=build /opt/pgw /opt/pgw
COPY --chmod=0755 deploy/docker/entrypoint.sh /usr/local/bin/pgw-entrypoint
ENV PATH=/opt/pgw/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PGW_LOG_JSON=true \
    PGW_FACTORY_DEVICE=cpu \
    PGW_MASKSTORE_ROOT=/var/lib/pgw/masks
USER 10001:10001
WORKDIR /var/lib/pgw
HEALTHCHECK NONE
ENTRYPOINT ["pgw-entrypoint"]
CMD ["pgw", "factory", "run"]
