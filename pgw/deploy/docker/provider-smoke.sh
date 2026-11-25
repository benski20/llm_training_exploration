#!/usr/bin/env bash
# Smoke test of a provider image (no GPU, no network, no model download):
#   deploy/docker/provider-smoke.sh pgw-provider-cpu:dev
#
# Starts the service with accounts on and no packs (loopback http), creates an organisation and an SDK key with
# the admin CLI, and checks POST /pgw/v1/sdk/enroll: a valid key gets past authentication (503: no model served
# yet), a forged key and the revoked key get 401. The service and the CLI both run as uid 10001.
set -euo pipefail
IMAGE=${1:?usage: provider-smoke.sh IMAGE}
NAME=pgw-provider-smoke-$$
cleanup() { docker rm -f "$NAME" > /dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" --network none \
    -e PGW_SERVICE_HOST=127.0.0.1 -e PGW_SERVICE_PORT=8443 -e PGW_SERVICE_PUBLIC_URL=http://127.0.0.1:8443 \
    -e PGW_SERVICE_ACCOUNTS_DB=/var/lib/pgw/accounts.db -e PGW_SERVICE_TOKENS_FILE=/var/lib/pgw/tenants.txt \
    -e PGW_SERVICE_PACKS='[]' -e PGW_HEALTHCHECK_SCHEME=http \
    "$IMAGE" sh -c 'touch /var/lib/pgw/tenants.txt && exec pgw service serve' > /dev/null

as_pgw() { docker exec -i -u 10001:10001 "$NAME" "$@"; }
for _ in $(seq 60); do
    if as_pgw pgw-healthcheck --live service > /dev/null 2>&1; then break; fi
    sleep 1
done
as_pgw pgw-healthcheck --live service
test "$(as_pgw id -u)" = 10001

KEY_JSON=$(as_pgw pgw service keys create --org smoke --create-org --name ci --service-url http://127.0.0.1:8443 \
    --accounts-db /var/lib/pgw/accounts.db --json)
KEY_ID=$(printf '%s' "$KEY_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["key_id"])')

# the key itself never leaves the container: it is read back from the JSON inside it
as_pgw python - "$KEY_JSON" <<'EOF'
import json, sys, urllib.request, urllib.error

key = json.loads(sys.argv[1])["key"]
forged = key[:-6] + ("aaaaaa" if not key.endswith("aaaaaa") else "bbbbbb")

def enroll(k):
    req = urllib.request.Request("http://127.0.0.1:8443/pgw/v1/sdk/enroll", method="POST",
                                 data=json.dumps({"installation_name": "smoke"}).encode(),
                                 headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code

assert enroll(key) == 503, "a valid key must pass authentication (no model served -> 503)"
assert enroll(forged) == 401, "a damaged key must be refused"
print("sdk enroll: valid key 503 (no packs), forged key 401")
EOF

as_pgw pgw service keys revoke "$KEY_ID" --accounts-db /var/lib/pgw/accounts.db --tokens-file /var/lib/pgw/tenants.txt
as_pgw python - "$KEY_JSON" <<'EOF'
import json, sys, time, urllib.request, urllib.error

key = json.loads(sys.argv[1])["key"]
code = 0
for _ in range(20):
    req = urllib.request.Request("http://127.0.0.1:8443/pgw/v1/sdk/enroll", method="POST",
                                 data=b'{"installation_name": "smoke"}',
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        code = e.code
    if code == 401:
        break
    time.sleep(0.5)
assert code == 401, f"revoked key not refused ({code})"
print("revoked key refused: 401")
EOF
echo "provider image smoke test passed: $IMAGE"
