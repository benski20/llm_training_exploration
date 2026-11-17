#!/bin/sh
# First-boot configuration of the PGW gateway appliance from OVF properties (vApp options).
#
# Reads the OVF environment through VMware Tools (guestinfo.ovfEnv) and writes:
#   /etc/pgw/gateway.env               non-secret settings
#   /etc/pgw/models.tmpl.yaml          the served model (token referenced, not inlined)
#   /etc/pgw/secrets/service-token     compute-service tenant token   (0600 root)
#   /etc/pgw/secrets/admin-token       gateway admin token            (0600 root)
#   /etc/pgw/secrets/kek               mask store KEK, generated here (0600 root)
#   /etc/nftables.d/pgw-egress.nft     egress allowlist: DNS, NTP and the compute service only
# OVF property values never reach logs. Runs once (marker /etc/pgw/.ovf-configured).
set -eu

umask 077
OVF=$(vmtoolsd --cmd "info-get guestinfo.ovfEnv" 2>/dev/null || true)
if [ -z "$OVF" ]; then
    echo "pgw-ovf-config: no OVF environment; configure /etc/pgw manually" >&2
    exit 0
fi

prop() {
    # Extract <Property oe:key="KEY" oe:value="VALUE"/> without echoing the document.
    printf '%s' "$OVF" | python3 -c '
import sys, xml.etree.ElementTree as ET
ns = {"oe": "http://schemas.dmtf.org/ovf/environment/1"}
root = ET.fromstring(sys.stdin.read())
for p in root.iter("{http://schemas.dmtf.org/ovf/environment/1}Property"):
    if p.get("{http://schemas.dmtf.org/ovf/environment/1}key") == sys.argv[1]:
        sys.stdout.write(p.get("{http://schemas.dmtf.org/ovf/environment/1}value") or "")
' "$1"
}

SERVICE_URL=$(prop pgw.service_url)
SERVICE_CIDR=$(prop pgw.service_cidr)
MODEL=$(prop pgw.model_name)
PACK_DIR=$(prop pgw.pack_dir)
FINGERPRINT=$(prop pgw.fingerprint)
RING=$(prop pgw.ring)

# Validate every value that is written into YAML or an nftables rule: a stray quote, newline or
# brace would otherwise change the generated configuration (or the firewall) silently.
fail() { echo "pgw-ovf-config: $*" >&2; exit 1; }
NL='
'
# matches REGEX VALUE: VALUE is one line that fully matches REGEX (grep -x alone checks per line).
matches() {
    case "$2" in *"$NL"*) return 1 ;; esac
    printf '%s' "$2" | grep -Eqx "$1"
}
[ -n "$SERVICE_URL" ] || fail "pgw.service_url is required"
matches 'https://[A-Za-z0-9.-]+(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?' "$SERVICE_URL" \
    || fail "pgw.service_url must be an https:// URL (host[:port][/path])"
[ -z "$MODEL" ] || matches '[A-Za-z0-9][A-Za-z0-9._:-]{0,127}' "$MODEL" || fail "pgw.model_name is invalid"
matches '[A-Za-z0-9][A-Za-z0-9._-]{0,127}' "$PACK_DIR" || fail "pgw.pack_dir must be a directory name"
matches '[0-9a-f]{64}' "$FINGERPRINT" || fail "pgw.fingerprint must be 64 lowercase hex digits"
[ -z "$RING" ] || matches 'K[0-9]{2}A[0-9]{1,2}' "$RING" || fail "pgw.ring must look like K40A12"
if [ -n "$SERVICE_CIDR" ]; then
    python3 -c 'import ipaddress, sys; n = ipaddress.ip_network(sys.argv[1], strict=False); sys.exit(n.version != 4)' \
        "$SERVICE_CIDR" 2>/dev/null || fail "pgw.service_cidr must be an IPv4 address or CIDR"
else
    echo "pgw-ovf-config: WARNING pgw.service_cidr not set: no egress firewall is installed" >&2
fi

install -d -m 0750 -o root -g root /etc/pgw/secrets
prop pgw.service_token > /etc/pgw/secrets/service-token
prop pgw.admin_token > /etc/pgw/secrets/admin-token
[ -s /etc/pgw/secrets/service-token ] || fail "pgw.service_token is required"
if [ ! -s /etc/pgw/secrets/kek ]; then
    head -c 32 /dev/urandom > /etc/pgw/secrets/kek
fi
chmod 0600 /etc/pgw/secrets/*

if [ ! -f /var/lib/pgw/masks/state.sqlite ]; then
    /opt/pgw/bin/pgw maskstore init --root /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek
    chown -R pgw:pgw /var/lib/pgw/masks
fi

cat > /etc/pgw/models.tmpl.yaml <<EOF
- name: ${MODEL:-default}
  pack_path: /var/lib/pgw/packs/${PACK_DIR}
  fingerprint: "${FINGERPRINT}"
  ring: ${RING:-K40A12}
  service_url: ${SERVICE_URL}
  service_token: "\${file:/run/credentials/pgw-gateway.service/service-token}"
EOF
chmod 0640 /etc/pgw/models.tmpl.yaml
chgrp pgw /etc/pgw/models.tmpl.yaml

if [ -n "$SERVICE_CIDR" ]; then
    install -d -m 0755 /etc/nftables.d
    cat > /etc/nftables.d/pgw-egress.nft <<EOF
table inet pgw_egress {
  chain output {
    type filter hook output priority 0; policy drop;
    oif "lo" accept
    ct state established,related accept
    udp dport { 53, 123 } accept
    tcp dport 53 accept
    ip daddr ${SERVICE_CIDR} tcp dport { 443, 8443 } accept
  }
}
EOF
    if ! grep -q pgw-egress /etc/nftables.conf 2>/dev/null; then
        echo 'include "/etc/nftables.d/pgw-egress.nft"' >> /etc/nftables.conf
    fi
    systemctl enable --now nftables
fi

touch /etc/pgw/.ovf-configured
echo "pgw-ovf-config: configured" >&2
