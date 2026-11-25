# PGW client library: build contract (internal, for the parallel build)

Goal: a client installs one Python package and changes one import:

```python
from pgw import OpenAI          # drop-in for openai.OpenAI (also AsyncOpenAI)
client = OpenAI(api_key="pgw_...")        # nothing else required
client.chat.completions.create(model="qwen2.5-7b-instruct", messages=[...], stream=True)
```

The gateway runs **in the client's process** (CPU only, no GPU, no model weights for the linears, no
large storage to manage). The provider (our GPU service) only ever sees masked data. Models: Qwen2.5
Instruct 0.5B / 1.5B / 3B / 7B for this release. Nothing here may weaken privacy or integrity; outputs stay
bit-identical to the pinned reference; every new mode defaults off unless proven exact and safe.

## Pieces and owners

| Piece | Location | Owner |
|---|---|---|
| Public API, in-process engine, background mask workers, cache dir | `src/pgw/sdk/` | SDK agent |
| Provisioning: non-secret model artifacts + pks-v3 hint (provider-built, client-verified) | `src/pgw/provision/` (client) and `src/pgw/service/provision.py` (provider) | Provisioning agent |
| Provider accounts: SDK keys, revocation, usage metering, rate limits; container image; RunPod tooling | `src/pgw/service/accounts.py`, `deploy/docker/`, `deploy/runpod/` | Provider agent |
| Developer console (keys, usage, request log), restyle of the existing console | `src/pgw/platform/`, `src/pgw/console/static/` | Console agent |
| Wheels (native extensions prebuilt), docs, examples, client hand-off kit | `pyproject.toml`, `.github/`, `docs/sdk/`, `examples/` | Packaging agent |

## Interfaces (keep these names; payload details are the owner's, documented in this file's appendix)

1. **SDK key.** `pgw_<env>_<base32>`: encodes the provider URL, the provider CA SHA-256 (pinned, as in an
   enrollment code) and a reusable, revocable credential. Parsed by `pgw.sdk.keys.parse_key` (SDK agent);
   issued and checked by the provider (`pgw.service.accounts`, provider agent). `PGW_API_KEY` env var works.
2. **Session bootstrap.** On first use the SDK reuses the enrollment machinery (`pgw.enroll`): CA pinned
   from the key, client key generated locally, `POST /pgw/v1/sdk/enroll` authenticated by the SDK key,
   returns what `pgw.enroll` writes today (service token, client cert, models.json). Cached under the
   cache dir (0600). Provider agent implements the endpoint; SDK agent consumes it.
3. **Provisioning.** `pgw.provision.client.provision(model, *, session, cache_dir, progress) ->
   ProvisionedModel` returns everything the in-process gateway needs that is not secret (tokenizer, the
   gateway's plain tensors, pack metadata + fingerprint) and a ready pks-v3 hint store. Every artifact is
   checked against the pack fingerprint; the hint is built by the provider from **client-chosen public
   seeds** and accepted only after a client-side verification with >= 40-bit soundness, or built locally
   (fallback / `hint="local"`). Provider endpoints under `/pgw/v1/provision/...`.
4. **Masks.** In-process background factory (pks-v3 by default for the SDK, with the library's explicit
   acknowledgement, documented as pending external review; `mask_scheme="mlwe"` and `"plain"` selectable),
   incremental reservations, conversation prefix reuse, p31 checks, native decode driver, masked head.

## Rules for every agent
- Never print or commit secrets. No Modal. No RunPod calls (no key yet). Kaggle is allowed for GPU
  validation only (the `kaggle` CLI is preconfigured).
- Tests: `pytest` for what you touch must pass; add tests for everything new.
- Commit messages end with the session's attribution lines; no model identifiers in commits.

## Appendix: payloads (owners fill in)

### A1. SDK key (provider agent; codec `pgw.enroll.sdk_key`, stdlib only)

```
pgw_<env>_<base32>
env    = [a-z][a-z0-9]{0,15}            (live | test | dev ...; the provider's PGW_SERVICE_SDK_KEY_ENV)
base32 = RFC 4648, lowercase, no padding, of:
  version (1 byte, = 1) | ca_sha256 (32) | key_id (8) | secret (24) | provider URL (UTF-8, <= 200 bytes) | check (4)
check  = SHA-256("pgw-sdk-key-v1" | env | 0x00 | everything before it)[:4]
```

* `ca_sha256`: SHA-256 of the DER of the provider CA, pinned exactly like an enrollment code (`pgw.enroll.code`):
  fetch `GET /pgw/v1/enroll/ca` unverified, compare, then verify TLS against that CA only. 32 zero bytes = no pin,
  accepted for `http://` to a loopback host only (tests).
* `key_id`: public, shown as `sk_<16 hex>` (listings, audit, usage). `secret`: 192 bits; the provider stores only
  `SHA-256("pgw-sdk-key-secret-v1" | key_id | secret)` and compares in constant time.
* `pgw.sdk.keys.parse_key` should wrap `pgw.enroll.sdk_key.SDKKey.parse(text) -> SDKKey(env, service_url,
  ca_sha256, key_id, secret)` (raises `SDKKeyError`, a `ConfigError`, whose message never contains the key).
  `str()`/`repr()` of an `SDKKey` never show the secret.
* Test vector: `SDKKey("test", "https://203.0.113.7:22443", bytes(range(32)), bytes.fromhex("0011223344556677"),
  bytes(range(100, 124))).encode()` ==
  `pgw_test_aeaacaqdaqcqmbyibefawdanbyhraeiscmkbkfqxdamrugy4dupb6aareizuivlgo5sgkzthnbuwu23mnvxg64drojzxi5lwo54hs6t3nb2hi4dthixs6mrqgmxdalrrgezs4nz2gizdinbtpqytwli`
  (stored digest `93825b55e599ec56935c715e19e9f1fe0dd0f1109564917224c183bbd5cb9ca3`).
* The key is sent only as `Authorization: Bearer <whole key>` and only on the two SDK routes below, over the
  pinned TLS. Never in a URL, a log line or a file other than the SDK's own 0600 cache.

### A2. `POST /pgw/v1/sdk/enroll` (provider agent)

Enabled when the provider sets `PGW_SERVICE_ACCOUNTS_DB` (SQLite, WAL). Request (JSON, at most 64 KiB):

```json
{"installation_name": "host-1234",
 "csr_pem": "-----BEGIN CERTIFICATE REQUEST-----...",
 "models": ["qwen2.5-7b-instruct"]}
```

* `installation_name`: `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` (`gateway_name` accepted as an alias).
* `csr_pem` (optional): EC P-256/P-384 or RSA >= 2048, key generated locally; signed when the provider has a
  client-issuing CA (as in `/pgw/v1/enroll`).
* `models` (optional): the subset wanted; default every served model in the key's scope.

200 reply (`Cache-Control: no-store`): exactly what `POST /pgw/v1/enroll` returns today, plus `sdk`:

```json
{"tenant": "sdk.<16 hex>", "service_token": "pgwt-...", "ca_bundle_pem": "-----BEGIN CERTIFICATE-----...",
 "service_url": "https://...",
 "models": [{"name": "...", "fingerprint": "...", "ring": "...", "rings": ["..."], "hf_repo": "...",
             "revision": "...", "license": "...", "plain_dtype": "...", "pack_name": "..."}],
 "client_cert_pem": "... (only when a CSR was sent and signed)", "client_cert_cn": "pgw-gw.sdk.<hex>.<name>",
 "sdk": {"key_id": "sk_<16 hex>", "org_id": "acme", "installation_id": "in_<12 hex>", "env": "live",
         "models_scope": ["qwen2.5-7b-instruct"],
         "limits": {"daily_token_quota": null, "org_daily_token_quota": null, "rows_per_s": null,
                    "burst_rows": null, "max_installations": null},
         "expires_at": null}}
```

(`models_scope` `[]` = every served model; limits `null` = none; `expires_at` unix seconds or `null`.)

`pgw.enroll.client.enroll_with_sdk_key(key, config_dir, installation_name=..., models=..., request_client_cert=True)`
does the whole flow (pin, local key + CSR, POST, write `secrets/service_token` 0600, `service-ca.pem`,
`secrets/client.key` 0600, `client.crt`, `models.json` 0600, `enrollment.json`) as `enroll_gateway` does for a
code. One tenant per SDK key (`sdk.<key hex>`); every installation of the key gets its own service token (and
client certificate when a CSR is sent). Revoking the key revokes all of them: tokens removed from the tokens file,
the tenant refused at once by the process that revoked it and within `PGW_SERVICE_ACCOUNTS_REFRESH_S` (2 s) by the
others; open compute channels and lane connections close.

Errors (`{"error": {"code", "message"}}`):

| HTTP | code | when |
|---|---|---|
| 401 | `unauthenticated` | malformed, unknown, revoked or expired key, wrong env, disabled organisation (one message for all: `invalid, revoked or expired SDK key`) |
| 403 | `permission_denied` | a requested model is outside the key's scope |
| 429 | `rate_limited` | failed attempts from this address (per minute), enrollments of this key per hour (`PGW_SERVICE_SDK_ENROLLS_PER_KEY_PER_HOUR`, 60), `max_installations`, or today's token quota (key or org) used up. `Retry-After` set |
| 503 | `unavailable` | none of the key's models is served right now |
| 400 | `protocol` | bad JSON, bad name, bad CSR |

Compute with the issued credentials is the normal gateway protocol (`/pgw/v1/compute`, `/pgw/v1/stream`,
`/pgw/v1/lane/open`, `/pgw/v1/maskgen`). For SDK tenants the provider additionally refuses a model outside the key's
scope with 403 `permission_denied`, and over-quota or over-rate calls with 429 `rate_limited` (+ `retry_after_s`).

### A3. `GET /pgw/v1/sdk/usage` (provider agent)

`Authorization: Bearer <SDK key>`; optional `?days=N` (1..92, default 31). Reply:

```json
{"key_id": "sk_...", "org_id": "acme", "state": "active", "models_scope": [], "limits": {},
 "today": {"tokens": 1234.0, "trips": 56789},
 "usage": [{"day": "2026-11-25", "model": "qwen2.5-7b-instruct", "trips": 56789, "rows": 61234,
            "tokens": 1234.0, "maskgen_cols": 0, "maskgen_requests": 0, "bytes_in": 1, "bytes_out": 2}]}
```

Units (metered by the provider, which never sees plaintext or masks): `trips` = compute round trips (HTTP
requests, stream frames and lane calls); `rows` = masked rows over all linears; `tokens` = rows of the model's first
linear kind divided by the number of linears of that kind (= token positions processed: prompt and generated,
speculative rows included); `maskgen_cols` / `maskgen_requests` = encrypted mask generation work (ciphertext
columns; the provider cannot see slots); bytes = masked payload bytes. Days are UTC. Usage reaches the database
every few seconds (`PGW_SERVICE_ACCOUNTS_FLUSH_S`, 5), so the last seconds may be missing.

### A4. Admin API (provider agent; off unless `PGW_SERVICE_ACCOUNTS_ADMIN_API=true`; for the console agent)

`Authorization: Bearer <admin token>` (`pgwa-...`, created with `pgw service admin-token create [--org ORG]` and
hashed at rest, or a bootstrap token whose SHA-256 is listed in `PGW_SERVICE_ACCOUNTS_ADMIN_TOKEN_SHA256`). A token
created with `--org` sees and changes only that organisation (403 otherwise); a global one sees all.

| method, path | body / query | reply |
|---|---|---|
| `POST /pgw/v1/admin/orgs` (global) | `{"org_id", "name"?, "daily_token_quota"?}` | `org` |
| `GET /pgw/v1/admin/orgs` | | `{"orgs": [org]}` (an org token: its own) |
| `GET /pgw/v1/admin/orgs/{org}` | | `org` |
| `POST /pgw/v1/admin/orgs/{org}/keys` | `{"name", "models"?, "env"?, "expires_in_s"?, "daily_token_quota"?, "rows_per_s"?, "burst_rows"?, "max_installations"?}` | `{"key": "pgw_...", "record": key}` (the key is shown once) |
| `GET /pgw/v1/admin/orgs/{org}/keys` | | `{"keys": [key]}` |
| `POST /pgw/v1/admin/orgs/{org}/keys/{key_id}/revoke` | | `key` |
| `GET /pgw/v1/admin/orgs/{org}/installations` | `?key_id=` | `{"installations": [inst]}` |
| `GET /pgw/v1/admin/orgs/{org}/usage` | `?since=YYYY-MM-DD&until=YYYY-MM-DD&key_id=` | `{"usage": [usage row + "key_id"]}` |
| `GET /pgw/v1/admin/orgs/{org}/audit` | `?limit=100&before=<id>` | `{"events": [event]}` |
| `POST /pgw/v1/admin/attest` (no auth) | `{"nonce": "<64 hex>"}` | `{"ca_sha256": hex, "macs": [hex]}` |

```text
org   = {"org_id", "name", "created_at", "disabled_at", "daily_token_quota", "keys_active"}
key   = {"key_id", "org_id", "name", "env", "models", "created_at", "expires_at", "revoked_at", "last_used_at",
         "state": "active|revoked|expired", "limits": {...as in A2...}, "installations": int}
inst  = {"installation_id", "key_id", "name", "tenant", "token_id", "cert_serial", "created_at", "revoked_at",
         "client"}
event = {"id", "ts", "event", "org_id", "key_id", "actor", "client", "detail": {...}}   (never a secret)
```

Audit events: `org.created`, `org.disabled`, `key.created`, `key.revoked`, `sdk.enroll.ok`, `sdk.enroll.failed`
(`detail.reason`: unknown_key, revoked, expired, wrong_env, org_disabled, model_scope, install_limit, enroll_rate,
quota), `quota.exceeded`, `admin.denied`. `attest` lets a launcher holding a bootstrap admin token check, before it
sends the token, that the TLS endpoint it reached holds the provider CA:
`mac = HMAC-SHA256(key = SHA-256(admin token) as 32 raw bytes, "pgw-admin-attest-v1" | nonce | ca_sha256)`.
