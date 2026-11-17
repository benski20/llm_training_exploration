# syntax=docker/dockerfile:1.7
# PGW mask factory, CUDA image (exact int8 limb GEMMs via torch._int_mm on NVIDIA GPUs).
#
# Measured: Llama-3-8B, K=48, T4: 0.0066 s per token slot (~550k slots/h).
#
#   docker build -f deploy/docker/factory-cuda.Dockerfile -t pgw-factory-cuda:dev .
#   docker run --gpus all -v masks:/var/lib/pgw/masks -v packs:/var/lib/pgw/packs:ro pgw-factory-cuda:dev
#
# The torch CUDA wheels bundle the CUDA runtime; the host needs only the NVIDIA driver and the
# NVIDIA container toolkit. TORCH_CUDA selects the wheel index (cu121, cu124).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
ARG TORCH_VERSION=2.5.1
ARG TORCH_CUDA=cu124
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/pgw
ENV PATH=/opt/pgw/bin:$PATH
RUN pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" "torch==${TORCH_VERSION}"
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[factory,gpu]" \
    && pgw factory --help > /dev/null

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="pgw-factory-cuda" \
      org.opencontainers.image.description="PGW mask factory (NVIDIA GPU)" \
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
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PGW_LOG_JSON=true \
    PGW_FACTORY_DEVICE=cuda \
    PGW_FACTORY_BACKEND=cuda_int8 \
    PGW_MASKSTORE_ROOT=/var/lib/pgw/masks
USER 10001:10001
WORKDIR /var/lib/pgw
HEALTHCHECK NONE
ENTRYPOINT ["pgw-entrypoint"]
CMD ["pgw", "factory", "run"]
