# syntax=docker/dockerfile:1.7
# PGW compute service, CPU image: for local demos, CI and small models only.
# Production serving uses service-cuda.Dockerfile. The CPU path is exact (cpu_int8 / cpu_f64 limbs).
#
#   docker build -f deploy/docker/service-cpu.Dockerfile -t pgw-service-cpu:dev .

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
RUN pip install ".[service]" \
    && pgw service --help > /dev/null

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="pgw-service-cpu" \
      org.opencontainers.image.description="PGW stateless masked compute service (CPU; demo/dev)" \
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
    PGW_LOG_JSON=true \
    PGW_SERVICE_HOST=0.0.0.0 \
    PGW_SERVICE_PORT=8443 \
    PGW_SERVICE_DEVICE=cpu
USER 10001:10001
WORKDIR /var/lib/pgw
EXPOSE 8443
HEALTHCHECK --interval=10s --timeout=6s --start-period=120s --retries=3 CMD ["pgw-healthcheck", "service"]
ENTRYPOINT ["pgw-entrypoint"]
CMD ["pgw", "service", "serve"]
