#!/bin/bash
# PGW GPU mask service VM first boot (customer cloud account; separate_vm mode of deploy/customer-cloud/aws).
#
# Runs on a GPU image with the NVIDIA driver, Docker and the NVIDIA container toolkit (AWS Deep Learning Base
# GPU AMI). Reads /etc/pgw/deploy.env (no secrets), then:
#   1. mounts the data volume (/var/lib/pgw: public model pack, export staging and outbox);
#   2. waits for the gateway's store recipient public key, store id and model (SSM, written by the gateway VM);
#   3. builds the same public model pack from Hugging Face (the fingerprint must equal the enrolled one);
#   4. creates this service's Ed25519 signing key and the outbox TLS certificate, publishes their public halves;
#   5. starts `pgw factory mask-service` (GPU; refuses to start unless its device output is bit-identical to the
#      CPU reference) writing HPKE-sealed, signed, store-bound segments, and `pgw factory outbox-serve` on 8470
#      (reachable from the gateway's security group only).
# Nothing secret leaves this VM: seeds and r exist only inside the mask-service process and its GPU.
set -euo pipefail
exec > >(tee -a /var/log/pgw-bootstrap.log) 2>&1
# shellcheck disable=SC1091
. /etc/pgw/deploy.env

log() { echo "[pgw-mask-service $(date -u +%H:%M:%S)] $*"; }
ssm_get() { aws ssm get-parameter --region "$AWS_REGION" --name "$1" ${2:+--with-decryption} \
  --query Parameter.Value --output text 2>/dev/null; }
ssm_put() {  # String parameters are Advanced tier (certificates exceed 4 KB); Terraform created them so
  local tier=()
  [ "${3:-String}" = String ] && tier=(--tier Advanced)
  aws ssm put-parameter --region "$AWS_REGION" --name "$1" --value "$2" --type "${3:-String}" "${tier[@]}" \
    --overwrite >/dev/null
}
ssm_wait() {
  local v
  for _ in $(seq 1 1080); do
    v=$(ssm_get "$1" "${2:-}" || true)
    if [ -n "$v" ] && [ "$v" != "pending" ] && [ "$v" != "None" ]; then printf '%s' "$v"; return 0; fi
    sleep 10
  done
  log "timed out waiting for $1"; return 1
}

command -v nvme >/dev/null || apt-get install -y -q nvme-cli >/dev/null 2>&1 || true
DATA_DEV=/dev/sdf
for d in /dev/nvme*n1; do
  if nvme id-ctrl -v "$d" 2>/dev/null | grep -q sdf; then DATA_DEV=$d; fi
done
mkdir -p /var/lib/pgw
if ! mountpoint -q /var/lib/pgw && [ -b "$DATA_DEV" ]; then
  blkid "$DATA_DEV" >/dev/null 2>&1 || mkfs.ext4 -q -L pgw "$DATA_DEV"
  echo "$DATA_DEV /var/lib/pgw ext4 defaults,nofail,noatime 0 2" >> /etc/fstab
  mount /var/lib/pgw
fi
command -v jq >/dev/null || { apt-get update -q && apt-get install -y -q jq >/dev/null; }
command -v aws >/dev/null || snap install aws-cli --classic >/dev/null
nvidia-smi -L || { log "no GPU visible; check the instance type and driver"; exit 1; }
if [[ "$PGW_IMAGE_REGISTRY" == *.dkr.ecr.*.amazonaws.com* ]]; then
  aws ecr get-login-password --region "$AWS_REGION" | docker login -u AWS --password-stdin "${PGW_IMAGE_REGISTRY%%/*}"
fi
IMAGE="$PGW_IMAGE_REGISTRY/pgw-factory-cuda:$PGW_IMAGE_TAG"
TOOLS_IMAGE="$PGW_IMAGE_REGISTRY/pgw-gateway:$PGW_IMAGE_TAG"  # has the pack builder; used for one-off commands
docker pull -q "$IMAGE"
docker pull -q "$TOOLS_IMAGE"
install -d -m 0750 -o 10001 -g 10001 /etc/pgw /var/lib/pgw/packs /var/lib/pgw/export
install -d -m 0700 -o 10001 -g 10001 /etc/pgw/secrets
pgw() { docker run --rm -u 10001:10001 --network host -v /etc/pgw:/etc/pgw -v /var/lib/pgw:/var/lib/pgw \
  "$TOOLS_IMAGE" pgw "$@"; }

log "waiting for the gateway (recipient key, store id, model)"
RECIPIENT_HEX=$(ssm_wait "$PGW_SSM_PREFIX/gateway/recipient-public-key-hex")
STORE_ID=$(ssm_wait "$PGW_SSM_PREFIX/gateway/store-id")
MODEL_JSON=$(ssm_wait "$PGW_SSM_PREFIX/gateway/model")
printf '%s\n' "$RECIPIENT_HEX" > /etc/pgw/recipient.pub
name=$(jq -r .name <<<"$MODEL_JSON"); repo=$(jq -r .hf_repo <<<"$MODEL_JSON")
rev=$(jq -r '.revision // empty' <<<"$MODEL_JSON"); pname=$(jq -r '.pack_name // .name' <<<"$MODEL_JSON")
want_fp=$(jq -r .fingerprint <<<"$MODEL_JSON")
PACK="/var/lib/pgw/packs/$name"
if [ ! -s "$PACK/manifest.json" ]; then
  log "building the public model pack $repo -> $PACK"
  pgw pack "$repo" "$PACK" --name "$pname" ${rev:+--revision "$rev"}
fi
have_fp=$(jq -r .fingerprint "$PACK/manifest.json")
[ "$have_fp" = "$want_fp" ] || { log "pack fingerprint $have_fp != enrolled $want_fp; refusing"; exit 1; }

[ -s /etc/pgw/secrets/sign.key ] || pgw maskstore partner-key create /etc/pgw/secrets/sign.key >/dev/null
SIGN_HEX=$(pgw maskstore partner-key show /etc/pgw/secrets/sign.key | jq -r .public_key_hex)
if [ ! -s /etc/pgw/outbox.crt ]; then
  IP=$(hostname -I | awk '{print $1}')
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 825 -subj "/CN=$IP" \
    -addext "subjectAltName=IP:$IP" -keyout /etc/pgw/secrets/outbox.key -out /etc/pgw/outbox.crt 2>/dev/null
  chown 10001:10001 /etc/pgw/secrets/outbox.key /etc/pgw/outbox.crt; chmod 0600 /etc/pgw/secrets/outbox.key
fi
ssm_get "$PGW_SSM_PREFIX/courier-token" decrypt > /etc/pgw/secrets/courier.token
chown 10001:10001 /etc/pgw/secrets/courier.token; chmod 0600 /etc/pgw/secrets/courier.token
ssm_put "$PGW_SSM_PREFIX/mask-service/signing-public-key-hex" "$SIGN_HEX"
ssm_put "$PGW_SSM_PREFIX/mask-service/outbox-cert" "$(cat /etc/pgw/outbox.crt)"

docker rm -f pgw-mask-service pgw-outbox >/dev/null 2>&1 || true
# The mask service needs no network at all: the pack is local and segments go to the outbox directory.
docker run -d --name pgw-mask-service --restart unless-stopped --gpus all -u 10001:10001 --network none \
  -v /etc/pgw:/etc/pgw:ro -v /var/lib/pgw:/var/lib/pgw \
  -e PGW_FACTORY_PACK_PATH="$PACK" -e PGW_FACTORY_CHECK_FIELD="$PGW_CHECK_FIELD" \
  -e PGW_FACTORY_SEGMENT_SLOTS="$PGW_SEGMENT_SLOTS" -e PGW_FACTORY_LOW_WATER_SLOTS="$PGW_LOW_WATER_SLOTS" \
  -e PGW_FACTORY_TARGET_SLOTS="$PGW_OUTBOX_TARGET_SLOTS" -e PGW_FACTORY_MAX_STORE_BYTES="$PGW_OUTBOX_MAX_BYTES" \
  "$IMAGE" pgw factory mask-service --export-dir /var/lib/pgw/export --recipient-key /etc/pgw/recipient.pub \
  --partner-key /etc/pgw/secrets/sign.key --destination-store "$STORE_ID"
docker run -d --name pgw-outbox --restart unless-stopped -u 10001:10001 -p 8470:8470 \
  -v /etc/pgw:/etc/pgw:ro -v /var/lib/pgw/export:/var/lib/pgw/export "$IMAGE" \
  pgw factory outbox-serve --export-dir /var/lib/pgw/export --token-file /etc/pgw/secrets/courier.token \
  --host 0.0.0.0 --port 8470 --tls-cert /etc/pgw/outbox.crt --tls-key /etc/pgw/secrets/outbox.key
log "mask service started; 'docker logs pgw-mask-service' shows the start-up check and the segment rate"
