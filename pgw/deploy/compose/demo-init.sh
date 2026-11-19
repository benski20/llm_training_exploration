#!/bin/sh
# One-shot, idempotent setup for the docker-compose demo (runs in the gateway image as root, then
# hands every volume to uid 10001). Never use the generated material in production.
set -eu

ROOT=${PGW_DEMO_ROOT:-/var/lib/pgw}   # overridable for tests
HERE=$(cd "$(dirname "$0")" && pwd)
PACKS=$ROOT/packs
PACK=$PACKS/demo
MASKS=$ROOT/masks
KEK=$ROOT/kek/kek
GW=$ROOT/secrets          # gateway-only secrets (service token, CA, models template)
STATE=$ROOT/state         # gateway state (API keys, quotas, audit log)
SVC=$ROOT/svc-secrets     # compute-service-only material (token hashes, TLS key)
RING=${PGW_DEMO_RING:-K32A12C27}
MODEL=${PGW_DEMO_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
SERVICE_URL=${PGW_DEMO_SERVICE_URL:-https://service:8443}
SERVICE_HOST=${PGW_DEMO_SERVICE_HOST:-service}
mkdir -p "$PACKS" "$MASKS" "$(dirname "$KEK")" "$GW" "$STATE" "$SVC"

log() { echo "demo-init: $*" >&2; }

if [ ! -f "$PACK/manifest.json" ]; then
    log "packing $MODEL (downloads the checkpoint once)"
    pgw pack "$MODEL" "$PACK" --name demo
fi
FP=$(pgw inspect "$PACK" --json | python -c 'import json,sys; print(json.load(sys.stdin)["fingerprint"])')
log "model pack fingerprint $FP"

if [ ! -f "$MASKS/state.sqlite" ]; then
    log "creating mask store and KEK"
    pgw maskstore init --root "$MASKS" --key-file "$KEK"
fi

python "$HERE/demo_certs.py" "$SVC" "$SERVICE_HOST" localhost 127.0.0.1
cp "$SVC/ca.crt" "$GW/ca.crt"

if [ ! -f "$GW/service_token" ]; then
    log "creating the tenant token (service stores only its SHA-256)"
    pgw service token create --tenant demo --tokens-file "$SVC/tokens" --out "$GW/service_token" > /dev/null
fi

if [ ! -f "$STATE/api_keys.json" ]; then
    pgw gateway keys create --name demo --keys-file "$STATE/api_keys.json" > "$GW/demo_api_key.json"
    log "demo API key (shown once, demo only): $(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["key"])' "$GW/demo_api_key.json")"
fi

# Admin token: /admin/* API and console sign-in (http://localhost:8080/console/). The gateway reads it
# from the file (PGW_GATEWAY_ADMIN_TOKEN_FILE), so it never appears in the compose file or `docker inspect`.
if [ ! -f "$GW/admin_token" ]; then
    (umask 077 && python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$GW/admin_token")
    log "demo admin token for the console (shown once, demo only): $(cat "$GW/admin_token")"
fi

# Models template: the service token is referenced, not inlined; the gateway entrypoint renders it
# into /run/pgw/models.json (tmpfs, 0600) at start-up.
cat > "$GW/models.tmpl.json" <<EOF
[
  {
    "name": "demo",
    "pack_path": "$PACK",
    "fingerprint": "$FP",
    "ring": "$RING",
    "service_url": "$SERVICE_URL",
    "service_token": "\${file:$GW/service_token}",
    "service_ca_file": "$GW/ca.crt",
    "max_concurrent_sessions": 8
  }
]
EOF

if [ "$(id -u)" = 0 ] && [ "${PGW_DEMO_CHOWN:-1}" = 1 ]; then
    chown -R 10001:10001 "$PACKS" "$MASKS" "$(dirname "$KEK")" "$GW" "$STATE" "$SVC"
fi
chmod 0700 "$MASKS" "$(dirname "$KEK")" "$GW" "$SVC"
chmod 0600 "$KEK"
log "done"
