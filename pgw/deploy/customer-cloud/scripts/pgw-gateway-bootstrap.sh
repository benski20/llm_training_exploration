#!/bin/bash
# PGW gateway VM first boot (customer cloud account; rendered into cloud-init by deploy/customer-cloud/aws).
#
# Reads /etc/pgw/deploy.env (written by cloud-init, no secrets) and then:
#   1. mounts the data volume (/var/lib/pgw) and the separate counter volume (/var/lib/pgw-counter);
#   2. installs Docker (and the NVIDIA toolkit in same-host mode on a GPU image), logs in to the registry;
#   3. enrolls with the one-time code from SSM (SecureString), then overwrites the parameter ("consumed");
#   4. builds the public model pack(s) from Hugging Face, initialises the mask store and its KEK;
#   5. mask supply, by PGW_MASK_SUPPLY:
#        cpu_encrypted - CPU-only encrypted factory on this VM (<= 7B; pgw.hemask, provider sees only ciphertexts)
#        same_host     - GPU mask service container on this VM writing into this VM's store
#        separate_vm   - publishes the store's recipient public key and store id to SSM, waits for the mask
#                        service's signing key and TLS certificate, then pulls signed, HPKE-sealed segments
#   6. starts the gateway (client API on 8443 with TLS; self-signed unless a certificate is provided).
# Idempotent: every step checks for its result first, so a re-run after a failure continues.
set -euo pipefail
exec > >(tee -a /var/log/pgw-bootstrap.log) 2>&1
# shellcheck disable=SC1091
. /etc/pgw/deploy.env

log() { echo "[pgw-bootstrap $(date -u +%H:%M:%S)] $*"; }
ssm_get() { aws ssm get-parameter --region "$AWS_REGION" --name "$1" ${2:+--with-decryption} \
  --query Parameter.Value --output text 2>/dev/null; }
ssm_put() {  # String parameters are Advanced tier (certificates exceed 4 KB); Terraform created them so
  local tier=()
  [ "${3:-String}" = String ] && tier=(--tier Advanced)
  aws ssm put-parameter --region "$AWS_REGION" --name "$1" --value "$2" --type "${3:-String}" "${tier[@]}" \
    --overwrite >/dev/null
}
ssm_wait() {  # ssm_wait NAME [decrypt]: poll until the parameter exists and is not a placeholder
  local v
  for _ in $(seq 1 720); do
    v=$(ssm_get "$1" "${2:-}" || true)
    if [ -n "$v" ] && [ "$v" != "pending" ] && [ "$v" != "None" ]; then printf '%s' "$v"; return 0; fi
    sleep 10
  done
  log "timed out waiting for $1"; return 1
}

mount_volume() {  # mount_volume DEVICE_HINT MOUNTPOINT: first unformatted NVMe/EBS disk of that size class
  local dev=$1 mnt=$2
  mkdir -p "$mnt"
  mountpoint -q "$mnt" && return 0
  [ -b "$dev" ] || { log "volume $dev not found; using the root disk for $mnt"; return 0; }
  blkid "$dev" >/dev/null 2>&1 || mkfs.ext4 -q -L "$(basename "$mnt")" "$dev"
  echo "$dev $mnt ext4 defaults,nofail,noatime 0 2" >> /etc/fstab
  mount "$mnt"
}

ebs_device() {  # ebs_device /dev/sdX -> the NVMe device whose serial names that attachment
  local want=${1#/dev/} d
  for d in /dev/nvme*n1; do
    if nvme id-ctrl -v "$d" 2>/dev/null | grep -q "$want"; then echo "$d"; return; fi
  done
  echo "$1"
}

log "starting (supply=$PGW_MASK_SUPPLY model_hint=$PGW_MODEL_HINT)"
command -v nvme >/dev/null || apt-get install -y -q nvme-cli >/dev/null 2>&1 || true
mount_volume "$(ebs_device /dev/sdf)" /var/lib/pgw
mount_volume "$(ebs_device /dev/sdg)" /var/lib/pgw-counter

if ! command -v docker >/dev/null; then
  apt-get update -q && apt-get install -y -q docker.io jq openssl >/dev/null
  systemctl enable --now docker
fi
command -v jq >/dev/null || apt-get install -y -q jq >/dev/null
command -v aws >/dev/null || snap install aws-cli --classic >/dev/null
if [[ "$PGW_IMAGE_REGISTRY" == *.dkr.ecr.*.amazonaws.com* ]]; then
  aws ecr get-login-password --region "$AWS_REGION" | docker login -u AWS --password-stdin "${PGW_IMAGE_REGISTRY%%/*}"
fi
GW_IMAGE="$PGW_IMAGE_REGISTRY/pgw-gateway:$PGW_IMAGE_TAG"
docker pull -q "$GW_IMAGE"

install -d -m 0750 -o 10001 -g 10001 /etc/pgw /var/lib/pgw/packs /var/lib/pgw/masks /var/lib/pgw/inbox \
  /var/lib/pgw/state /var/lib/pgw-counter
install -d -m 0700 -o 10001 -g 10001 /etc/pgw/secrets /etc/pgw/tls
pgw() {  # run the pgw CLI in the gateway image with the same paths as on the host
  docker run --rm -u 10001:10001 --network host -v /etc/pgw:/etc/pgw -v /var/lib/pgw:/var/lib/pgw \
    -v /var/lib/pgw-counter:/var/lib/pgw-counter --env-file /etc/pgw/gateway.env "$GW_IMAGE" pgw "$@"
}
touch /etc/pgw/gateway.env

# ---- 3. enrollment (one-time code; never written to disk outside the container's run) ---------------------
if [ ! -s /etc/pgw/enrollment.json ]; then
  CODE=$(ssm_get "$PGW_SSM_PREFIX/enrollment-code" decrypt)
  [ -n "$CODE" ] && [ "$CODE" != "consumed" ] || { log "no enrollment code in SSM"; exit 1; }
  pgw gateway enroll "$CODE" --config-dir /etc/pgw --packs-dir /var/lib/pgw/packs --name "$PGW_GATEWAY_NAME"
  unset CODE
  ssm_put "$PGW_SSM_PREFIX/enrollment-code" consumed SecureString
fi

# ---- 4. public model pack(s) and the mask store --------------------------------------------------------------
jq -c '.models[]' /etc/pgw/enrollment.json | while read -r m; do
  name=$(jq -r .name <<<"$m"); repo=$(jq -r .hf_repo <<<"$m"); rev=$(jq -r '.revision // empty' <<<"$m")
  pname=$(jq -r '.pack_name // .name' <<<"$m")
  if [ ! -s "/var/lib/pgw/packs/$name/manifest.json" ]; then
    log "building the public model pack $repo -> /var/lib/pgw/packs/$name"
    pgw pack "$repo" "/var/lib/pgw/packs/$name" --name "$pname" ${rev:+--revision "$rev"}
  fi
done
MODEL_JSON=$(jq -c '.models[0]' /etc/pgw/enrollment.json)
MODEL_NAME=$(jq -r .name <<<"$MODEL_JSON")
SERVICE_URL=$(jq -r .service_url /etc/pgw/enrollment.json)
cat > /etc/pgw/gateway.env <<EOF
PGW_LOG_JSON=true
PGW_GATEWAY_HOST=0.0.0.0
PGW_GATEWAY_PORT=8443
PGW_GATEWAY_MODELS_FILE=/etc/pgw/models.json
PGW_GATEWAY_MASKSTORE_ROOT=/var/lib/pgw/masks
PGW_GATEWAY_TLS__CERT_FILE=/etc/pgw/tls/tls.crt
PGW_GATEWAY_TLS__KEY_FILE=/etc/pgw/tls/tls.key
PGW_GATEWAY_REQUIRE_SERVICE_TLS=true
PGW_GATEWAY_API_KEYS_FILE=/var/lib/pgw/state/api_keys.json
PGW_GATEWAY_QUOTA_STATE_FILE=/var/lib/pgw/state/quota.json
PGW_GATEWAY_AUDIT_LOG_FILE=/var/lib/pgw/state/audit.jsonl
PGW_MASKSTORE_ROOT=/var/lib/pgw/masks
PGW_MASKSTORE_KEY_FILE=/etc/pgw/secrets/maskstore.kek
PGW_MASKSTORE_COUNTER=file
PGW_MASKSTORE_COUNTER_FILE=/var/lib/pgw-counter/generation.counter
EOF
[ -s /etc/pgw/secrets/maskstore.kek ] || pgw maskstore init --root /var/lib/pgw/masks \
  --key-file /etc/pgw/secrets/maskstore.kek

# ---- client API TLS ------------------------------------------------------------------------------------------
if [ ! -s /etc/pgw/tls/tls.crt ]; then
  CERT=$(ssm_get "$PGW_SSM_PREFIX/client-tls/cert" || true); KEY=$(ssm_get "$PGW_SSM_PREFIX/client-tls/key" decrypt || true)
  if [ -n "$CERT" ] && [ "$CERT" != "None" ] && [ -n "$KEY" ] && [ "$KEY" != "None" ]; then
    printf '%s\n' "$CERT" > /etc/pgw/tls/tls.crt; printf '%s\n' "$KEY" > /etc/pgw/tls/tls.key
  else
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 825 -subj "/CN=$PGW_GATEWAY_NAME" \
      -addext "subjectAltName=DNS:$PGW_GATEWAY_NAME,IP:$(hostname -I | awk '{print $1}')" \
      -keyout /etc/pgw/tls/tls.key -out /etc/pgw/tls/tls.crt 2>/dev/null
    ssm_put "$PGW_SSM_PREFIX/gateway/client-tls-cert" "$(cat /etc/pgw/tls/tls.crt)"  # for the thin client to pin
  fi
  chown 10001:10001 /etc/pgw/tls/tls.*; chmod 0600 /etc/pgw/tls/tls.key
fi

# ---- 5. mask supply ------------------------------------------------------------------------------------------
# check_field auto: p31 for the CPU encrypted factory (masks from the provider), ring for the customer's own GPU
# mask service. The gateway runs the p31 check (prime_check, on the native TripEngine) whenever p31 is in use.
if [ "$PGW_CHECK_FIELD" = p31 ] || { [ "$PGW_CHECK_FIELD" = auto ] && [ "$PGW_MASK_SUPPLY" = cpu_encrypted ]; }; then
  grep -q '^PGW_GATEWAY_PRIME_CHECK=' /etc/pgw/gateway.env || echo "PGW_GATEWAY_PRIME_CHECK=true" >> /etc/pgw/gateway.env
fi
docker rm -f pgw-factory pgw-mask-service pgw-mask-pull >/dev/null 2>&1 || true
case "$PGW_MASK_SUPPLY" in
  cpu_encrypted)
    HE_ENV=()
    if [ "${PGW_HE_SCHEME:-mlwe}" = "pks-v3" ]; then
      # pks-v3 (compact hint) is PENDING EXTERNAL REVIEW (lattice-estimator run and cryptographer sign-off); the
      # library refuses it without its acknowledgement, which this preset passes explicitly. The hint key
      # authenticates the hint store's metadata and is kept outside the hint directory.
      if [ ! -s /etc/pgw/secrets/pks-hint.key ]; then
        head -c 32 /dev/urandom > /etc/pgw/secrets/pks-hint.key
        chown 10001:10001 /etc/pgw/secrets/pks-hint.key; chmod 0600 /etc/pgw/secrets/pks-hint.key
      fi
      HE_ENV=(-e PGW_FACTORY_HE_SCHEME=pks-v3 -e PGW_FACTORY_HE_PKS_ACK=pks-v3-unreviewed-n8192
              -e PGW_FACTORY_HE_HINT_KEY_FILE=/etc/pgw/secrets/pks-hint.key)
    fi
    docker run -d --name pgw-factory --restart unless-stopped -u 10001:10001 --network host "${HE_ENV[@]}" \
      --cpus "$PGW_FACTORY_CPUS" -v /etc/pgw:/etc/pgw -v /var/lib/pgw:/var/lib/pgw \
      -v /var/lib/pgw-counter:/var/lib/pgw-counter --env-file /etc/pgw/gateway.env \
      -e PGW_FACTORY_PACK_PATH="/var/lib/pgw/packs/$MODEL_NAME" -e PGW_FACTORY_HE_SERVICE_URL="$SERVICE_URL" \
      -e PGW_FACTORY_HE_TOKEN_FILE=/etc/pgw/secrets/service_token -e PGW_FACTORY_HE_CA_FILE=/etc/pgw/service-ca.pem \
      -e PGW_FACTORY_HE_CLIENT_CERT_FILE=/etc/pgw/client.crt -e PGW_FACTORY_HE_CLIENT_KEY_FILE=/etc/pgw/secrets/client.key \
      -e PGW_FACTORY_HE_HINT_DIR=/var/lib/pgw/state/hemask -e PGW_FACTORY_CHECK_FIELD="$PGW_CHECK_FIELD" \
      -e PGW_FACTORY_NICE=10 "$PGW_IMAGE_REGISTRY/pgw-factory:$PGW_IMAGE_TAG" pgw factory run
    ;;
  same_host)
    MS_IMAGE="$PGW_IMAGE_REGISTRY/pgw-factory-cuda:$PGW_IMAGE_TAG"
    docker pull -q "$MS_IMAGE"
    docker run -d --name pgw-mask-service --restart unless-stopped --gpus all -u 10001:10001 --network none \
      -v /etc/pgw/secrets:/etc/pgw/secrets:ro -v /var/lib/pgw:/var/lib/pgw -v /var/lib/pgw-counter:/var/lib/pgw-counter \
      --env-file /etc/pgw/gateway.env -e PGW_FACTORY_PACK_PATH="/var/lib/pgw/packs/$MODEL_NAME" \
      -e PGW_FACTORY_CHECK_FIELD="$PGW_CHECK_FIELD" -e PGW_FACTORY_LOW_WATER_SLOTS="$PGW_LOW_WATER_SLOTS" \
      -e PGW_FACTORY_TARGET_SLOTS="$PGW_TARGET_SLOTS" -e PGW_FACTORY_SEGMENT_SLOTS="$PGW_SEGMENT_SLOTS" \
      "$MS_IMAGE" pgw factory mask-service
    ;;
  separate_vm)
    [ -s /etc/pgw/secrets/recipient.key ] || pgw maskstore recipient-key create /etc/pgw/secrets/recipient.key \
      --public-out /etc/pgw/recipient.pub >/dev/null
    STORE_ID=$(pgw maskstore status --json | jq -r .store_id)
    ssm_put "$PGW_SSM_PREFIX/gateway/recipient-public-key-hex" "$(od -An -tx1 -v /etc/pgw/recipient.pub | tr -d ' \n')"
    ssm_put "$PGW_SSM_PREFIX/gateway/store-id" "$STORE_ID"
    ssm_put "$PGW_SSM_PREFIX/gateway/model" "$MODEL_JSON"
    log "waiting for the mask service's signing key and outbox certificate"
    SIGN_PUB=$(ssm_wait "$PGW_SSM_PREFIX/mask-service/signing-public-key-hex")
    ssm_wait "$PGW_SSM_PREFIX/mask-service/outbox-cert" > /etc/pgw/mask-service-ca.pem
    ssm_get "$PGW_SSM_PREFIX/courier-token" decrypt > /etc/pgw/secrets/courier.token
    chown 10001:10001 /etc/pgw/secrets/courier.token; chmod 0600 /etc/pgw/secrets/courier.token
    cat >> /etc/pgw/gateway.env <<EOF
PGW_MASKSTORE_RECIPIENT_KEY_FILE=/etc/pgw/secrets/recipient.key
PGW_MASKSTORE_TRUSTED_PARTNER_KEYS=["$SIGN_PUB"]
EOF
    docker run -d --name pgw-mask-pull --restart unless-stopped -u 10001:10001 --network host \
      -v /etc/pgw:/etc/pgw -v /var/lib/pgw:/var/lib/pgw -v /var/lib/pgw-counter:/var/lib/pgw-counter \
      --env-file /etc/pgw/gateway.env "$GW_IMAGE" pgw maskstore pull --from "https://$PGW_MASK_SERVICE_IP:8470" \
      --token-file /etc/pgw/secrets/courier.token --ca-file /etc/pgw/mask-service-ca.pem \
      --inbox /var/lib/pgw/inbox --max-available-slots "$PGW_TARGET_SLOTS" --loop 5
    ;;
  *) log "unknown PGW_MASK_SUPPLY=$PGW_MASK_SUPPLY"; exit 1 ;;
esac

# ---- 6. gateway ----------------------------------------------------------------------------------------------
docker rm -f pgw-gateway >/dev/null 2>&1 || true
docker run -d --name pgw-gateway --restart unless-stopped -u 10001:10001 --network host \
  -v /etc/pgw:/etc/pgw -v /var/lib/pgw:/var/lib/pgw \
  -v /var/lib/pgw-counter:/var/lib/pgw-counter --env-file /etc/pgw/gateway.env "$GW_IMAGE" pgw gateway serve
log "gateway started on :8443 (supply $PGW_MASK_SUPPLY); create an API key with: docker exec pgw-gateway pgw gateway keys create"
