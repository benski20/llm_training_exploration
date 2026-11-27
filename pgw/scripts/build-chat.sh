#!/bin/sh
# Rebuild the chat web app (pgw/chat) into src/pgw/chat/static/ and check the result.
#
#   scripts/build-chat.sh            # npm ci, lint, unit tests, build, static checks
#   scripts/build-chat.sh --quick    # npm ci, build, static checks (no lint or unit tests)
#
# Needs Node.js 20+ and npm. Commit the changed files under src/pgw/chat/static/ with the sources.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
QUICK=0
case "${1:-}" in
    --quick) QUICK=1 ;;
    "") ;;
    *) echo "usage: $0 [--quick]" >&2; exit 2 ;;
esac

command -v npm > /dev/null || { echo "build-chat: npm (Node.js 20+) is required" >&2; exit 1; }

cd "$ROOT/chat"
npm ci --no-audit --no-fund
if [ "$QUICK" = 0 ]; then
    npm run lint
    npm test
fi
npm run build    # tsc -b && vite build -> ../src/pgw/chat/static/

cd "$ROOT"
PY=${PYTHON:-python3}
if "$PY" -c 'import pytest' 2> /dev/null; then
    PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pytest -q -p no:cacheprovider \
        tests/chat/test_chat_build.py
else
    test -f src/pgw/chat/static/index.html
    echo "build-chat: pytest not found; only checked that index.html exists" >&2
fi
echo "build-chat: done; commit src/pgw/chat/static/"
