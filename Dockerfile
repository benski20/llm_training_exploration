# Serving image: verification-gated inference, no torch needed.
#   docker build -t vt-serve .
#   docker run -p 8000:8000 -v $PWD/results/cegis/exact_dyck_L2-64:/models:ro vt-serve \
#       /models/model.npz /models/certificate.json
# syntax=docker/dockerfile:1
FROM python:3.12-slim
# Behind a TLS-inspecting proxy, pass its CA as a build secret (never stored in
# the image):  docker build --secret id=pipca,src=/path/ca.crt -t vt-serve .
RUN --mount=type=secret,id=pipca,required=false \
    if [ -f /run/secrets/pipca ]; then export PIP_CERT=/run/secrets/pipca; fi; \
    pip install --no-cache-dir "numpy>=1.24" "scipy>=1.11"
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
ENTRYPOINT ["vt", "serve", "--host", "0.0.0.0", "--port", "8000"]
