# syntax=docker/dockerfile:1
# Serving image: verification-gated inference, no torch needed.
#   docker build -t vt-serve .
#   docker run -p 8000:8000 -v $PWD/results/cegis/exact_dyck_L2-64:/models:ro vt-serve \
#       /models/model.npz /models/certificate.json
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
# Behind a TLS-inspecting proxy, pass its CA as a build secret (never stored in
# the image):  docker build --secret id=pipca,src=/path/ca.crt -t vt-serve .
RUN --mount=type=secret,id=pipca,required=false \
    if [ -f /run/secrets/pipca ]; then export PIP_CERT=/run/secrets/pipca; fi; \
    pip install --no-cache-dir numpy==2.5.3 scipy==1.18.1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY vt ./vt
RUN --mount=type=secret,id=pipca,required=false \
    if [ -f /run/secrets/pipca ]; then export PIP_CERT=/run/secrets/pipca; fi; \
    pip install --no-cache-dir --no-deps . && useradd --create-home --uid 10001 vt
USER vt
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
# The server re-proves the mounted artifact on startup; pin the spec with --spec/--spec-file.
ENTRYPOINT ["vt", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
