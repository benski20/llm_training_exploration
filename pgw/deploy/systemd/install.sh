#!/bin/sh
# Install PGW as systemd services on a Linux VM (used by the OVA / AMI / Azure image builds too).
#
#   sudo deploy/systemd/install.sh --role gateway --source /path/to/pgw [--torch-index URL]
#   sudo deploy/systemd/install.sh --role factory --wheel pgw-0.1.0-py3-none-any.whl --cuda cu132
#
# Creates the `pgw` system user, a virtualenv in /opt/pgw, /etc/pgw (config, 0750) with example
# files, /var/lib/pgw (state, 0700) and the systemd unit. It never generates or prints secrets:
# create /etc/pgw/secrets/* yourself (see deploy/vmware/README.md). Idempotent.
set -eu

ROLE=""
SOURCE=""
WHEEL=""
CUDA=""
TORCH_VERSION=${TORCH_VERSION:-2.15.0}
PYTHON=${PYTHON:-python3}
PREFIX=/opt/pgw
HERE=$(cd "$(dirname "$0")" && pwd)

usage() { echo "usage: $0 --role gateway|factory [--source DIR | --wheel FILE] [--cuda cu132]" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --role) ROLE=$2; shift 2 ;;
        --source) SOURCE=$2; shift 2 ;;
        --wheel) WHEEL=$2; shift 2 ;;
        --cuda) CUDA=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown option $1" >&2; usage ;;
    esac
done
case "$ROLE" in gateway|factory) ;; *) usage ;; esac
if [ -z "$SOURCE" ] && [ -z "$WHEEL" ]; then usage; fi
if [ "$(id -u)" != 0 ]; then echo "run as root" >&2; exit 1; fi

if ! id pgw >/dev/null 2>&1; then
    useradd --system --uid 10001 --user-group --home-dir /var/lib/pgw --shell /usr/sbin/nologin pgw
fi
install -d -m 0750 -o root -g pgw /etc/pgw
install -d -m 0750 -o root -g root /etc/pgw/secrets
install -d -m 0700 -o pgw -g pgw /var/lib/pgw /var/lib/pgw/masks
install -d -m 0755 -o pgw -g pgw /var/lib/pgw/packs
# Rollback-counter directory: mount a separate volume here (not snapshotted with /var/lib/pgw).
install -d -m 0700 -o pgw -g pgw /var/lib/pgw-counter
if command -v mountpoint >/dev/null 2>&1 && ! mountpoint -q /var/lib/pgw-counter; then
    echo "install.sh: WARNING /var/lib/pgw-counter is not a separate mount; a restore of the disk holding"
    echo "            /var/lib/pgw would also roll back the counter (see gateway.env.example)" >&2
fi

if [ ! -x "$PREFIX/bin/python" ]; then
    "$PYTHON" -m venv "$PREFIX"
fi
"$PREFIX/bin/pip" install --upgrade pip
if [ -n "$CUDA" ]; then
    "$PREFIX/bin/pip" install --index-url "https://download.pytorch.org/whl/$CUDA" "torch==$TORCH_VERSION"
else
    "$PREFIX/bin/pip" install --index-url https://download.pytorch.org/whl/cpu "torch==$TORCH_VERSION"
fi
EXTRAS=$ROLE
if [ "$ROLE" = gateway ]; then EXTRAS="gateway,pack"; fi
if [ -n "$CUDA" ]; then EXTRAS="$EXTRAS,gpu"; fi
if [ -n "$WHEEL" ]; then
    "$PREFIX/bin/pip" install "$WHEEL[$EXTRAS]" "pyyaml>=6"
else
    "$PREFIX/bin/pip" install "$SOURCE[$EXTRAS]" "pyyaml>=6"
    install -d -m 0755 "$PREFIX/share"
    if [ -f "$SOURCE/SPEC.md" ]; then install -m 0644 "$SOURCE/SPEC.md" "$PREFIX/share/SPEC.md"; fi
fi

FILES="$HERE/$ROLE.env.example"
if [ "$ROLE" = gateway ]; then FILES="$FILES $HERE/models.tmpl.yaml.example"; fi
for f in $FILES; do
    if [ ! -f "$f" ]; then continue; fi
    dest=/etc/pgw/$(basename "$f" .example)
    if [ ! -f "$dest" ]; then install -m 0640 -o root -g pgw "$f" "$dest"; fi
done
install -m 0644 "$HERE/pgw-$ROLE.service" "/etc/systemd/system/pgw-$ROLE.service"
systemctl daemon-reload
"$PREFIX/bin/pgw" doctor --role "$ROLE" --no-gpu || true
echo "installed pgw-$ROLE; configure /etc/pgw and /etc/pgw/secrets, then: systemctl enable --now pgw-$ROLE" >&2
