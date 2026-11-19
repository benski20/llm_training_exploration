#!/bin/sh
# Rebuild the operator console (pgw/console) into src/pgw/console/static/ and check the result.
#
#   scripts/build-console.sh            # npm ci, lint, unit tests, build, static checks
#   scripts/build-console.sh --quick    # npm ci, build, static checks (no lint or unit tests)
#
# Needs Node.js 20+ and npm. Run from anywhere; commit the changed files under
# src/pgw/console/static/ together with the console sources.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
QUICK=0
case "${1:-}" in
    --quick) QUICK=1 ;;
    "") ;;
    *) echo "usage: $0 [--quick]" >&2; exit 2 ;;
esac

command -v npm > /dev/null || { echo "build-console: npm (Node.js 20+) is required" >&2; exit 1; }

cd "$ROOT/console"
npm ci --no-audit --no-fund
if [ "$QUICK" = 0 ]; then
    npm run lint
    npm test
fi
npm run build    # tsc -b && vite build -> ../src/pgw/console/static/

cd "$ROOT"
PY=${PYTHON:-python3}
if "$PY" -c 'import pytest' 2> /dev/null; then
    PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pytest -q -p no:cacheprovider \
        tests/console/test_console_build.py -m "not slow"
else
    test -f src/pgw/console/static/index.html
    echo "build-console: pytest not found; only checked that index.html exists" >&2
fi
echo "build-console: done; commit src/pgw/console/static/"
