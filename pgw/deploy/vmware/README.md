# PGW gateway appliance for VMware (OVA): build notes

Topology A in SPEC section 13: the customer runs the gateway (and optionally a CPU or GPU mask factory)
on premises as a VM; the compute service runs in our cloud. This directory builds that VM as an OVA.

| file | purpose |
|---|---|
| `pgw-gateway.pkr.hcl` | Packer template (`vsphere-iso`): Ubuntu 24.04, PGW gateway + factory as systemd services, OVA export |
| `autoinstall/user-data.template`, `autoinstall/meta-data` | Ubuntu autoinstall seed for the build (build user is locked afterwards) |
| `ovf-properties.xml` | vApp properties the customer fills in at deploy time |
| `pgw-ovf-config.sh` | first-boot unit: turns the OVF properties into `/etc/pgw`, secrets and the egress firewall |
| `../systemd/` | hardened units (`pgw-gateway.service`, `pgw-factory.service`), env examples, `install.sh` |

## Build

1. Build the wheel in CI: `python -m build --wheel` (pgw root). Note its SHA-256 in the release notes.
2. Render the autoinstall seed with a throw-away build password (see the template header).
3. `packer init deploy/vmware && packer build -var-file=site.pkrvars.hcl deploy/vmware`
   (`vcenter_*`, `datacenter`, `cluster`, `datastore`, `network`, `pgw_wheel`, `build_password`).
4. Insert `ovf-properties.xml` into the exported `.ovf` `VirtualSystem`, update the manifest
   (`.mf`) hashes, and sign the OVA (`ovftool --privateKey=...`). Publish the OVA with its SHA-256.

The image contains **no secrets**: no KEK, no tokens, no SSH host keys (regenerated at first boot),
an empty machine-id, and a locked build account.

## Disks

* OS disk 40 GB.
* Data disk (`/var/lib/pgw`, mounted `nodev,nosuid,noexec`): model packs and the mask store. Size it
  for the mask inventory: 6.9 MB per token slot for an 8B model at K40 (`pgw inspect PACK` prints the
  exact figure); 1 M slots = 6.9 TB. Masks are encrypted by PGW (AES-256-GCM per segment), but place
  the disk on an encrypted datastore as well.
* **Never revert the data disk to a snapshot or clone the VM with its data disk.** Restoring an old
  mask store would allow mask reuse; the store detects this (generation journal) and refuses to
  serve. Back up packs and API keys, not masks. Exclude the data disk from VM snapshots
  (independent-persistent disk mode).

## First boot

`pgw-ovf-config.service` runs once before the PGW services:

* writes `/etc/pgw/secrets/{service-token,admin-token}` from the password-typed OVF properties,
  generates a fresh KEK (`/etc/pgw/secrets/kek`, 0600 root) and initialises the mask store;
* writes `/etc/pgw/models.tmpl.yaml` (the token is a `${file:...}` reference to the systemd credential);
* installs an nftables egress policy: only DNS, NTP and `pgw.service_cidr` on 443/8443. Together with
  the gateway's in-process egress guard this keeps prompts from leaving except as masked payloads to
  the compute service.

Secrets reach the services as systemd credentials (`LoadCredential=`), never via the environment of
other processes or the command line.

## Model packs

Copy a pack built with `pgw pack` (or downloaded from the release bucket) to `/var/lib/pgw/packs/<dir>`
and verify it: `pgw doctor --role gateway --pack /var/lib/pgw/packs/<dir>`. The fingerprint must match
the OVF property `pgw.fingerprint` and the compute service (`pgw doctor --service-url ...`).

## Masks

* With a local GPU or spare CPU: `pgw-factory.service` keeps the store between low water and target
  (`/etc/pgw/factory.env`). A CPU factory produces about 87 slots/s for Qwen2.5-0.5B on 4 cores (measured with
  `pgw bench factory --model qwen2.5-0.5b --device cpu`; run it on your host); 8B models need a GPU
  (T4: ~550k slots/h) or a partner factory.
* With a partner factory: give the partner your X25519 public key, then import delivered segments with
  `pgw maskstore import --root /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek FILES...`.

## Operations

* `systemctl status pgw-gateway pgw-factory`, `journalctl -u pgw-gateway` (JSON logs, no prompt data).
* `pgw doctor --role gateway --maskstore /var/lib/pgw/masks --service-url <url> --service-token-file ...`
* Updates: deploy a new OVA version and move the data disk, or `install.sh --wheel` on the running VM
  followed by `systemctl restart pgw-gateway pgw-factory`.
* Sizing: start with 8 vCPU / 32 GB RAM for an 8B model (the plaintext work is under 1% of FLOPs;
  the gateway mostly regenerates masks and waits on round trips) and size from measurements: the
  gateway exports `pgw_gateway_ttft_seconds` / `pgw_gateway_tpot_seconds`.
