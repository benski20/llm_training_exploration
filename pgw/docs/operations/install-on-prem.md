# Operator guide: on-prem gateway (topology A)

This guide is for customer platform teams installing the PGW gateway, mask store and (optionally)
mask factory in their own data center. The compute service runs in our cloud. For a gateway inside
your own cloud account, see [install-customer-cloud.md](install-customer-cloud.md). For a factory
with no network path to us, see [air-gapped-factory.md](air-gapped-factory.md). Capacity numbers
are in [sizing.md](sizing.md).

```
 your data center                                                   our cloud
 apps ─▶ pgw gateway ─(mTLS, HTTP/2, x̂ only)──────────────────────▶ pgw compute service
            ▲ reserve/take
         mask store (encrypted volume)  ◀── pgw factory (your GPU or CPU)
                                        ◀── or segments from an independent partner (HPKE)
```

## 1. Prerequisites

| item | requirement |
|---|---|
| Host | Linux x86-64, 4+ vCPU per gateway; RAM 8 GB for models up to about 3B, 16 GB for 8B (plaintext work is under 1% of FLOPs, but the embedding and LM head of Llama-3-8B take about 4.2 GB in fp32 plus the KV cache, see [sizing.md](sizing.md#6-gateway-host)) |
| Python | 3.12 or 3.13 (bare-metal install), or Docker / Podman 24+ for the `pgw-gateway` image |
| Disk | Mask store volume sized from [sizing.md](sizing.md): about 6.9 MB per token for 8B at K40A12, for at least 2 days of traffic plus the factory refill time. Local SSD/NVMe, encrypted at rest |
| Network | Outbound HTTPS (TCP 443 or the agreed port) to the service endpoint **only**; RTT matters: 128 round trips per 8B token (*analytic*) |
| Credentials from us | service URL, our CA bundle, a tenant token (and optionally an mTLS client certificate issued by our CA) |
| Model pack | the pack for your model, or build it yourself from the public checkpoint (section 3) and compare fingerprints |
| Optional GPU | any CUDA GPU for an in-house factory (a T4 produces 550k slots/h for 8B at K48, *measured*) |

## 2. Install the software

Bare metal (virtualenv):

```bash
python3.12 -m venv /opt/pgw && . /opt/pgw/bin/activate
pip install "pgw[gateway]"          # add ,factory for a factory on the same host
pgw version
pgw doctor --role gateway           # exact ring arithmetic, CSPRNG, dependencies
```

Container: use the `pgw-gateway` image (`deploy/docker/gateway.Dockerfile`). It runs as UID 10001
with a read-only root filesystem. Mount `/etc/pgw` (read-only config and secrets),
`/var/lib/pgw/masks` (mask store) and `/var/lib/pgw/packs` (model packs, read-only). Verify the image
signature and the SBOM before deployment (see the [audit checklist](../compliance/audit-readiness.md), S-03).

Create a dedicated service user with no login shell, and disable core dumps for it
(`LimitCORE=0` in systemd, or `ulimit -c 0`). A core dump of the gateway can contain plaintext.

## 3. Model pack

Either copy the pack we publish, or build it from the public checkpoint:

```bash
pgw pack Qwen/Qwen2.5-7B-Instruct /var/lib/pgw/packs/qwen7b --license apache-2.0
pgw inspect /var/lib/pgw/packs/qwen7b --verify full      # prints the SHA-256 fingerprint
```

The fingerprint must match the one the compute service reports. The gateway refuses to start
otherwise (`fingerprint_mismatch`). Pin it in the model configuration (section 6).

## 4. Mask store

```bash
install -d -m 0700 -o pgw -g pgw /var/lib/pgw/masks
pgw maskstore init --root /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek
pgw maskstore status --root /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek
```

* The KEK file is created with mode 0600. Keep it **off** the mask volume and out of backups of that
  volume. For production, prefer `PGW_MASKSTORE_KEY_PROVIDER=pkcs11` (HSM) or `kms`.
* Configure a monotonic counter for rollback protection. The default is `PGW_MASKSTORE_COUNTER=none`.
  Use `provider` with an HSM, or `file` with `PGW_MASKSTORE_COUNTER_FILE` on storage that is never
  backed up or snapshotted together with the mask volume. `tpm` (`PGW_MASKSTORE_TPM_NV_INDEX=...`)
  uses `tpm2-tools` and is preliminary in v1; test it with `pgw maskstore status` before you rely on
  it. Without a counter, only the HMAC journal protects the store, and the store logs that at info
  level. The journal detects a restored `state.sqlite` alone, but not a restore of the database and
  the journal from the same backup or snapshot (threat model T-22).
* **Never restore the mask store from a backup or snapshot.** A restored state could hand out slots a
  second time. The rollback check refuses to serve a restored database, and with a counter off the
  mask volume, a restored volume as well. Without such a counter, a whole-volume restore is **not**
  detected. Recovery is to initialise a new, empty store and refill it.
* Set `PGW_MASKSTORE_LOW_WATER_SLOTS` and `PGW_MASKSTORE_TARGET_SLOTS` from [sizing.md](sizing.md).

## 5. Masks: in-house factory or partner

In-house factory on a GPU host with access to the store volume:

```bash
pgw doctor --role factory
pgw factory benchmark --pack /var/lib/pgw/packs/qwen7b --ring K40A12   # measure your rate
pgw factory generate --pack /var/lib/pgw/packs/qwen7b --store /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek --slots 4096 --segments 4
pgw factory run --pack /var/lib/pgw/packs/qwen7b --store /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek
```

`pgw factory run` keeps every `(fingerprint, ring)` between low water and target. Run it as a systemd
service or a container with **no network access**: the factory needs only the pack and the store.

Partner masks: follow the partner section of [air-gapped-factory.md](air-gapped-factory.md#4-partner-delivery-hpke),
then import:

```bash
pgw maskstore import --root /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek /srv/incoming/*.pgwm
```

## 6. Gateway configuration

`/etc/pgw/models.json` (the YAML equivalent needs `pyyaml`):

```json
[
  {
    "name": "qwen7b",
    "pack_path": "/var/lib/pgw/packs/qwen7b",
    "fingerprint": "<64 hex characters from pgw inspect>",
    "mode": "masked",
    "ring": "K40A12",
    "service_url": "https://compute.example.net:8443",
    "service_ca_file": "/etc/pgw/secrets/service-ca.pem",
    "client_cert_file": "/etc/pgw/secrets/client.crt",
    "client_key_file": "/etc/pgw/secrets/client.key",
    "max_concurrent_sessions": 32
  }
]
```

The service tenant token is a secret. Do not put it in `models.json` in clear text in shared
repositories. Inject it from your secret manager when you render the file (`pgw config render
TEMPLATE OUT`) and keep the rendered file mode 0600.

**Faster decoding (optional).** Add `"draft_pack_path": "/packs/qwen2.5-0.5b"` (a small pack with the
same tokenizer) to turn on speculative decoding. The draft runs in plaintext on the gateway and
proposes `spec_tokens` tokens (default 4). The masked target checks them all in one pass. Every token
is still chosen by the target with the request's own sampler, so responses are identical to
decoding without a draft, and nothing about the draft leaves the gateway. Speculation only uses
mask slots that a request has already reserved beyond what it needs; it never makes a request run
out of slots. `draft_device` puts the draft on a gateway GPU if there is one; on CPU the draft
uses int8 weights.

`"spec_lookup": 3` adds prompt-lookup drafting: when the last few tokens already occurred earlier in
the conversation, the gateway proposes what followed them, at no compute cost. It works with or
without a draft model and helps most when answers quote their input (RAG, code edits, summaries,
structured output). Output is unchanged. Rejected proposals use mask rows, so it raises mask use
somewhat on text that does not repeat.

Two per-model throughput settings. `"inline_max_rows"` (default 8) lets decode round trips do their
gateway-side work on the event loop instead of handing it to the model's worker thread, which saves
two thread switches per round trip; it never applies to steps that could block (a mask window
extension, a reservation's first pass) or to the LM head. Set 0 to turn it off. `"batch_lm_head"`
computes the float32 LM head of all rows of a step (every concurrently decoding request, and all rows
of speculative verify passes) in one pass, so the head weights are read once per step instead of once
per row (with 4 requests on a 4-vCPU host about 0.19 s to 0.07 s per step for a 7B head). The LM head
uses a fixed summation order (`pgw.engine.lm_head`), so a row's logits are bit-identical whichever
rows share its pass: the setting only trades speed. Default (unset): on when the native extension
`pgw_fast` is built, off with its slower pure-torch fallback.

Environment (`/etc/pgw/gateway.env`):

```bash
PGW_GATEWAY_HOST=0.0.0.0
PGW_GATEWAY_PORT=8080
PGW_GATEWAY_MODELS_FILE=/etc/pgw/models.json
PGW_GATEWAY_MASKSTORE_ROOT=/var/lib/pgw/masks
PGW_MASKSTORE_KEY_FILE=/etc/pgw/secrets/kek
PGW_GATEWAY_API_KEYS_FILE=/var/lib/pgw/gateway/api-keys.json
PGW_GATEWAY_AUDIT_LOG_FILE=/var/log/pgw/audit.jsonl
PGW_GATEWAY_QUOTA_STATE_FILE=/var/lib/pgw/gateway/quota.json
PGW_GATEWAY_TLS__CERT_FILE=/etc/pgw/secrets/gateway.crt
PGW_GATEWAY_TLS__KEY_FILE=/etc/pgw/secrets/gateway.key
PGW_GATEWAY_NOISE_MONITOR_REQUIRED=true
PGW_LOG_JSON=true
```

Read the admin token (`PGW_GATEWAY_ADMIN_TOKEN`) from your secret manager, not from a world-readable
file. Run `pgw config show gateway` to see every setting and its current value (secrets masked).

Production policy is fail-closed. Leave `PGW_GATEWAY_ALLOW_DEV_MODES` unset: setting it permits
plaintext dev models, sampled Freivalds checks and anonymous access. `/admin/status` reports
`"production_policy": true` when it is off.

## 7. Start and verify

```bash
pgw doctor --role gateway --settings --pack /var/lib/pgw/packs/qwen7b --maskstore /var/lib/pgw/masks \
    --service-url https://compute.example.net:8443 --service-ca-file /etc/pgw/secrets/service-ca.pem
pgw gateway serve
pgw gateway keys create --name analytics-app --rpm 120 --tpm 100000     # key shown once
curl -s https://pgw.internal:8080/readyz
```

Then send a test request ([API reference](../api/gateway-api.md)) and check:

* `GET /admin/status`: `ready: true`, the fingerprint matches, `noise_monitor.state` is active,
  `breaker` closed.
* `GET /admin/masks`: `available_slots` is above low water.
* `/metrics`: `pgw_gateway_requests_total`, `pgw_masks_consumed_slots_total` increase.

## 8. Firewall and egress

* Allow outbound connections **only** to the service origin(s). The gateway also enforces this
  (egress allowlist), but configure it in your firewall as well. That is your independent control
  against a malicious build (whitepaper section 9).
* Inbound: the API port from application networks only. Restrict `/metrics`, `/dashboard` and
  `/admin/*` to the operations network (`PGW_GATEWAY_METRICS_REQUIRE_ADMIN=true` if you cannot).
* The factory host needs no network.

## 9. Operations

| task | how |
|---|---|
| Monitor | alert on `pgw_masks_low_water == 1`, any increase of `pgw_gateway_integrity_failures_total`, `pgw_gateway_noise_alarm == 1`, `/readyz` 503, p95 `pgw_gateway_tpot_seconds` |
| Integrity failure | the request fails with `integrity_check_failed` and the breaker opens. Contact us with the `request_id` from the audit log. After investigation: `POST /admin/models/{name}/reset` |
| Noise alarm | masked traffic is blocked (fail-closed). Treat it as a security incident: possible unmasked send or mask reuse. Preserve logs, then reset |
| `mask_reuse` error | stop the gateway, preserve the store directory, open a security incident. Never retry |
| Key rotation | `pgw maskstore rotate-key NEW_KEY_FILE` (gateway and factory stopped); API keys: `pgw gateway keys create`, then `revoke` |
| Upgrades | stop the gateway (in-flight requests drain for `PGW_GATEWAY_DRAIN_TIMEOUT_S`), replace the image, run `pgw doctor`, start. The mask store format is versioned |
| Scaling | one gateway replica per mask store. Add capacity with more replicas, each with its own store and factory allocation. Masks are never shared between replicas |
| Backups | back up configuration, API key file and audit log. **Do not** back up mask segments or `state.sqlite` for restore purposes |

## 10. Uninstall and decommission

Stop the gateway and factory. Shred the mask store (`pgw maskstore gc`, then delete the volume and
destroy the KEK). Revoke the tenant token with us. Keep the audit log for your retention period.
