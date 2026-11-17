#!/bin/sh
# Run the CI checks locally (same commands as .github/workflows/ci.yml, minus image builds).
#
#   ci/check.sh            lint + type-check + tests
#   ci/check.sh --fast     lint + type-check + tests without the slow end-to-end tests
set -eu

cd "$(dirname "$0")/.."
FAST=0
if [ "${1:-}" = "--fast" ]; then FAST=1; fi

echo "== ruff" >&2
ruff check src tests deploy

echo "== mypy" >&2
python -m mypy src

echo "== deploy artifacts" >&2
for f in deploy/docker/entrypoint.sh deploy/compose/demo-init.sh deploy/systemd/install.sh deploy/vmware/pgw-ovf-config.sh; do
    sh -n "$f"
done
if command -v helm >/dev/null 2>&1; then
    for chart in pgw-gateway pgw-factory pgw-service; do
        helm lint --strict "deploy/helm/$chart" -f "deploy/helm/$chart/ci/ci-values.yaml"
    done
fi

echo "== pytest" >&2
if [ "$FAST" = 1 ]; then
    PGW_OFFLINE=${PGW_OFFLINE:-1} python -m pytest -q -m "not gpu and not slow"
else
    PGW_OFFLINE=${PGW_OFFLINE:-1} python -m pytest -q -m "not gpu"
fi
