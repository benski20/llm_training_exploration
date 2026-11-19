#!/usr/bin/env bash
# PGW end-to-end demo on one machine (CPU is fine), without containers:
#
#   pack -> mask store + factory -> compute service (127.0.0.1) -> gateway (127.0.0.1) -> chat
#
# Usage:
#   scripts/e2e_demo.sh                          # Qwen/Qwen2.5-0.5B-Instruct (downloaded once from the HF hub)
#   PGW_E2E_MODEL=tiny scripts/e2e_demo.sh       # tiny random Qwen2 model (no download, gibberish answers)
#   PGW_E2E_MODEL=/path/to/hf/checkpoint scripts/e2e_demo.sh
#
# Environment (all optional):
#   PGW_DEMO_DIR      working directory (default: a new temp dir; kept afterwards for inspection)
#   PGW_DEMO_RING     ring preset (default K32A12C27)
#   PGW_DEMO_SLOTS    mask slots to generate (default 256; one slot per prompt or generated token)
#   PGW_DEMO_PROMPT   user message (default: a short question)
#   PGW_DEMO_KEEP=1   keep service and gateway running after the demo (Ctrl-C stops them)
#
# Everything generated here (KEK, tokens, API key) is for the demo only. The service is reached over
# plain HTTP on loopback; production uses TLS/mTLS (see docs/operations and deploy/).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PGW_ROOT=$(cd "$HERE/.." && pwd)
MODEL=${PGW_E2E_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
RING=${PGW_DEMO_RING:-K32A12C27}
SLOTS=${PGW_DEMO_SLOTS:-256}
PROMPT=${PGW_DEMO_PROMPT:-What is the capital of France? Answer in one word.}
WORK=${PGW_DEMO_DIR:-$(mktemp -d -t pgw-demo.XXXXXX)}
PY=${PYTHON:-python3}
ADMIN_TOKEN="demo-admin-$($PY -c 'import secrets; print(secrets.token_hex(16))')"

log() { printf '\033[1;34m[e2e-demo]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[e2e-demo] %s\033[0m\n' "$*" >&2; exit 1; }

if command -v pgw > /dev/null 2>&1; then PGW=(pgw); else PGW=("$PY" -m pgw.cli); fi
"${PGW[@]}" --help > /dev/null 2>&1 || die "pgw is not installed: pip install -e '$PGW_ROOT[dev]'"
command -v curl > /dev/null || die "curl is required"

free_port() { "$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
wait_ready() {  # url pid name
    for _ in $(seq 1 600); do
        kill -0 "$2" 2> /dev/null || { tail -n 40 "$WORK/logs/$3.log" >&2; die "$3 exited"; }
        if curl -fsS "$1" > /dev/null 2>&1; then return 0; fi
        sleep 0.5
    done
    die "$3 not ready at $1"
}

PIDS=()
cleanup() {
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2> /dev/null || true; done
    wait 2> /dev/null || true
    log "work directory kept: $WORK (logs in $WORK/logs)"
}
trap cleanup EXIT

mkdir -p "$WORK"/{packs,secrets,state,logs}
chmod 0700 "$WORK/secrets"
log "work directory $WORK"

# ---- 1. model pack (int8 weights + manifest with SHA-256 fingerprint) ------------------------------
SRC=$MODEL
if [ "$MODEL" = tiny ]; then
    SRC="$WORK/tiny-hf"
    if [ ! -f "$SRC/config.json" ]; then
        log "creating a tiny random Qwen2 checkpoint"
        (cd "$PGW_ROOT" && "$PY" -c "import sys; from pathlib import Path; from tests.conftest import _save_tiny; \
_save_tiny(Path(sys.argv[1]), 'qwen2', seed=0)" "$SRC")
    fi
fi
PACK="$WORK/packs/demo"
if [ ! -f "$PACK/manifest.json" ]; then
    log "packing $MODEL"
    "${PGW[@]}" pack "$SRC" "$PACK" --name demo
fi
FP=$("${PGW[@]}" inspect "$PACK" --json | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["fingerprint"])')
log "pack fingerprint $FP"

# ---- 2. encrypted single-use mask store + factory --------------------------------------------------
MASKS="$WORK/masks"
KEK="$WORK/secrets/kek"
# Rollback protection: an external monotonic counter kept outside the mask directory (production puts
# it on a separate volume, SPEC 7.6). Without one the store refuses to hand out masks.
mkdir -p "$WORK/counter"
export PGW_MASKSTORE_COUNTER=file PGW_MASKSTORE_COUNTER_FILE="$WORK/counter/generation.counter"
if [ ! -f "$MASKS/state.sqlite" ]; then
    "${PGW[@]}" maskstore init --root "$MASKS" --key-file "$KEK" > /dev/null
fi
log "factory: generating $SLOTS mask slots ($RING)"
"${PGW[@]}" factory generate --pack "$PACK" --store "$MASKS" --key-file "$KEK" --ring "$RING" --slots "$SLOTS" \
    > "$WORK/logs/factory.json" 2> "$WORK/logs/factory.log"

# ---- 3. credentials ------------------------------------------------------------------------------------
if [ ! -f "$WORK/secrets/service_token" ]; then
    "${PGW[@]}" service token create --tenant demo --tokens-file "$WORK/secrets/tokens" \
        --out "$WORK/secrets/service_token" > /dev/null
fi
if [ ! -f "$WORK/state/api_keys.json" ]; then
    "${PGW[@]}" gateway keys create --name demo --keys-file "$WORK/state/api_keys.json" > "$WORK/secrets/api_key.json"
fi
API_KEY=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["key"])' "$WORK/secrets/api_key.json")

# ---- 4. compute service (our side: sees only uniform ring noise) ----------------------------------------
SVC_PORT=$(free_port)
log "starting compute service on 127.0.0.1:$SVC_PORT"
PGW_SERVICE_HOST=127.0.0.1 PGW_SERVICE_PORT=$SVC_PORT PGW_SERVICE_DEVICE=cpu \
PGW_SERVICE_PACKS="[\"$PACK\"]" PGW_SERVICE_RINGS="[\"$RING\"]" PGW_SERVICE_TOKENS_FILE="$WORK/secrets/tokens" \
    "${PGW[@]}" service serve > "$WORK/logs/service.log" 2>&1 &
PIDS+=($!)
wait_ready "http://127.0.0.1:$SVC_PORT/readyz" "${PIDS[-1]}" service

# ---- 5. gateway (your side: plaintext engine, masks, Freivalds checks) ------------------------------------
GW_PORT=$(free_port)
cat > "$WORK/secrets/models.json" << EOF
[{"name": "demo", "pack_path": "$PACK", "fingerprint": "$FP", "ring": "$RING",
  "service_url": "http://127.0.0.1:$SVC_PORT", "service_token_file": "$WORK/secrets/service_token"}]
EOF
chmod 0600 "$WORK/secrets/models.json"
log "starting gateway on 127.0.0.1:$GW_PORT"
PGW_GATEWAY_HOST=127.0.0.1 PGW_GATEWAY_PORT=$GW_PORT PGW_GATEWAY_MODELS_FILE="$WORK/secrets/models.json" \
PGW_GATEWAY_API_KEYS_FILE="$WORK/state/api_keys.json" PGW_GATEWAY_ADMIN_TOKEN="$ADMIN_TOKEN" \
PGW_GATEWAY_MASKSTORE_ROOT="$MASKS" PGW_GATEWAY_AUDIT_LOG_FILE="$WORK/state/audit.jsonl" \
PGW_MASKSTORE_ROOT="$MASKS" PGW_MASKSTORE_KEY_FILE="$KEK" PGW_MASKSTORE_LOW_WATER_SLOTS=16 \
    "${PGW[@]}" gateway serve > "$WORK/logs/gateway.log" 2>&1 &
PIDS+=($!)
wait_ready "http://127.0.0.1:$GW_PORT/readyz?models=all" "${PIDS[-1]}" gateway

# ---- 6. OpenAI-compatible requests ------------------------------------------------------------------------
GW="http://127.0.0.1:$GW_PORT"
BODY=$("$PY" -c 'import json,sys; print(json.dumps({"model": "demo", "temperature": 0, "max_tokens": 16,
"messages": [{"role": "user", "content": sys.argv[1]}]}))' "$PROMPT")
log "chat completion (non-streaming)"
curl -fsS "$GW/v1/chat/completions" -H "Authorization: Bearer $API_KEY" -H 'content-type: application/json' \
    -d "$BODY" | "$PY" -c 'import json,sys; r=json.load(sys.stdin); print("answer:", repr(r["choices"][0]["message"]["content"])); print("usage:", r["usage"])'
log "chat completion (streaming)"
STREAM_BODY=$("$PY" -c 'import json,sys; b=json.loads(sys.argv[1]); b["stream"]=True; print(json.dumps(b))' "$BODY")
curl -fsSN "$GW/v1/chat/completions" -H "Authorization: Bearer $API_KEY" -H 'content-type: application/json' \
    -d "$STREAM_BODY" | "$PY" -c '
import json, sys
out = []
for line in sys.stdin:
    if line.startswith("data: ") and line.strip() != "data: [DONE]":
        ev = json.loads(line[6:])
        if ev.get("choices"):
            out.append(ev["choices"][0]["delta"].get("content") or "")
print("streamed:", repr("".join(out)))'

# ---- 7. what happened -----------------------------------------------------------------------------------------
log "integrity and mask accounting"
curl -fsS "$GW/admin/status" -H "Authorization: Bearer $ADMIN_TOKEN" | "$PY" -c '
import json, sys
m = json.load(sys.stdin)["models"][0]
b = m["backend"]
print("round trips %d, Freivalds-checked items %d, integrity failures %d, bytes up %.1f MB, down %.1f MB" % (
    b["round_trips"], b["freivalds_checked"], b["integrity_failures"], b["bytes_up"] / 1e6, b["bytes_down"] / 1e6))
print("mask slots left:", m.get("masks_available"), "| noise monitor:", m.get("noise_monitor", {}).get("state"))'

if [ "${PGW_DEMO_KEEP:-0}" = 1 ]; then
    log "gateway: $GW  (API key: $API_KEY, admin token: $ADMIN_TOKEN). Ctrl-C to stop."
    wait
fi
log "done"
