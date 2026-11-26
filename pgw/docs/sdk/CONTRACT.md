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
   enrollment code) and a reusable, revocable credential. One format and one codec (`pgw.enroll.sdk_key`, A1):
   issued and checked by the provider (`pgw.service.accounts`), parsed on the client by `pgw.sdk.keys.parse_key`
   (a wrapper of `SDKKey.parse` that also accepts pasted keys with whitespace, upper case or 0/1/8 look-alikes).
   `PGW_API_KEY` env var works.
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
* `pgw.sdk.keys.parse_key` wraps `pgw.enroll.sdk_key.SDKKey.parse(text) -> SDKKey(env, service_url,
  ca_sha256, key_id, secret)` and raises `pgw.sdk.keys.SdkKeyError` (a `SDKKeyError`, code `unauthenticated`:
  `AuthenticationError` at the API surface) whose message never contains the key; `pgw.sdk.keys.SdkKey` is
  `SDKKey`. `str()`/`repr()` of an `SDKKey` never show the secret. Views: `key_id_str` (`sk_...`), `key_id_hex`,
  `digest`, `bearer()`.
* Test vector: `SDKKey("test", "https://203.0.113.7:22443", bytes(range(32)), bytes.fromhex("0011223344556677"),
  bytes(range(100, 124))).encode()` ==
  `pgw_test_aeaacaqdaqcqmbyibefawdanbyhraeiscmkbkfqxdamrugy4dupb6aareizuivlgo5sgkzthnbuwu23mnvxg64drojzxi5lwo54hs6t3nb2hi4dthixs6mrqgmxdalrrgezs4nz2gizdinbtpqytwli`
  (stored digest `93825b55e599ec56935c715e19e9f1fe0dd0f1109564917224c183bbd5cb9ca3`; checked in
  `tests/sdk/test_keys.py` and `tests/service/test_accounts.py`).
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

200 reply (`Cache-Control: no-store`): exactly what `POST /pgw/v1/enroll` returns today, plus `sdk`, `lane` and a
`masked_head` flag per model:

```json
{"tenant": "sdk.<16 hex>", "service_token": "pgwt-...", "ca_bundle_pem": "-----BEGIN CERTIFICATE-----...",
 "service_url": "https://gpu.example.com",
 "models": [{"name": "...", "fingerprint": "...", "ring": "...", "rings": ["..."], "hf_repo": "...",
             "revision": "...", "license": "...", "plain_dtype": "...", "pack_name": "...", "masked_head": true}],
 "lane": {"transport": "tcp", "port": 8444},
 "client_cert_pem": "... (only when a CSR was sent and signed)", "client_cert_cn": "pgw-gw.sdk.<hex>.<name>",
 "sdk": {"key_id": "sk_<16 hex>", "org_id": "acme", "installation_id": "in_<12 hex>", "env": "live",
         "models_scope": ["qwen2.5-7b-instruct"],
         "limits": {"daily_token_quota": null, "org_daily_token_quota": null, "rows_per_s": null,
                    "burst_rows": null, "max_installations": null},
         "expires_at": null}}
```

(`models_scope` `[]` = every served model; limits `null` = none; `expires_at` unix seconds or `null`.
`masked_head`: the service also serves `head_fingerprint(fingerprint)` (`PGW_SERVICE_MASKED_HEAD=also`).
`lane`: the service's TCP+TLS trip lane as clients reach it (`lane_advertised_port`, else the listening
`lane_port`) when `PGW_SERVICE_LANE=tcp` is running, else `null`; the SDK then uses `lane="tls"` and allows
`https://<service host>:<port>` in its egress list. The SDK sends `gateway_name` (accepted alias of
`installation_name`), `client_name` and `sdk_version`; unknown fields are ignored.)

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

### A3b. `POST /pgw/v1/sdk/report` (content-free request records; integration)

Authenticated like `/pgw/v1/compute` (the installation's service token and/or client certificate; SDK tenants
only). Body `{"requests": [record, ...]}` (at most 256, 64 KiB), sent in batches by `pgw.sdk.reports.Reporter`
(`usage_reports=True`, the default; never delays or fails a request):

```json
{"ts": 1764100000.0, "model": "qwen2.5-7b-instruct", "status": "ok|error", "error_code": null,
 "prompt_tokens": 31, "completion_tokens": 200, "latency_ms": 8123.4, "ttft_ms": 412.0, "stream": true,
 "checks_ok": true}
```

Parsed by allow-list (`pgw.service.accounts._parse_report`; anything else is dropped, so no content can be stored);
a model outside the key's scope or a malformed record is dropped. Each record adds to the key's usage
(`requests`, `prompt_tokens`, `completion_tokens`, `errors`, `checks_ok`) and, unless
`PGW_SERVICE_ACCOUNTS_REQUEST_LOG_DAYS=0` (default 30), to the request log. Reply `{"accepted": n}`. The provider
already sees these counts and timings in the masked traffic (rows per trip, trips, arrival times); the report only
attributes them to requests. `checks_ok` is the client's statement that every integrity check of the request
passed: the provider cannot check a result itself, so the console labels it "checked client-side".

What the library puts in a record is limited to what the provider observes anyway (`pgw.sdk.reports.build_record`,
security review 2026-11): `prompt_tokens` counts only the prompt tokens sent to the provider (tokens reused from the
conversation's prefix cache are not counted), `latency_ms` is `null` with strict speculation (the answer returns
before the dummy passes), and `error_code` is one of the library's public codes (else `internal`). The provider
limits reports per SDK key (`PGW_SERVICE_ACCOUNTS_REPORT_RECORDS_PER_S`, default 20 records/s, burst 4096); over it
the reply is 429 and the library keeps the batch for its next send.

Only requests that reached the provider are reported: a request that failed before its mask session opened (no
masks in stock, prompt and `max_tokens` beyond the context, the model blocked after an integrity failure, the
model not ready) produces no record. The provider never saw it, and a record would tell it that a request was
made. Such failures are counted by error code in the client's own status only (`client.status()["local_errors"]`,
per model in `pgw sdk status --json`) and logged at INFO level, without content (security review 2026-11, S12).

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
| `GET /pgw/v1/admin/orgs/{org}/requests` | `?limit=50&before=<id>&key_id=&model=&status=` | `{"requests": [request], "next_before": id or null}` (newest first) |
| `GET /pgw/v1/admin/orgs/{org}/models` | | `{"models": [{"id", "name", "status", "context_length", "description"}]}` (what the service serves) |
| `POST /pgw/v1/admin/attest` (no auth) | `{"nonce": "<64 hex>"}` | `{"ca_sha256": hex, "macs": [hex]}` |

```text
org   = {"org_id", "name", "created_at", "disabled_at", "daily_token_quota", "keys_active"}
key   = {"key_id", "org_id", "name", "env", "models", "created_at", "expires_at", "revoked_at", "last_used_at",
         "state": "active|revoked|expired", "limits": {...as in A2...}, "installations": int}
inst  = {"installation_id", "key_id", "name", "tenant", "token_id", "cert_serial", "created_at", "revoked_at",
         "client"}
event = {"id", "ts", "event", "org_id", "key_id", "actor", "client", "detail": {...}}   (never a secret)
usage = {"day", "key_id", "model", "trips", "rows", "tokens", "maskgen_cols", "maskgen_requests", "bytes_in",
         "bytes_out"}  + when the client reported requests that day:
        {"requests", "prompt_tokens", "completion_tokens", "errors", "verified", "reported_by": "client"}
request = {"id", "ts", "key_id", "model", "status", "error_code", "prompt_tokens", "completion_tokens",
           "latency_ms", "ttft_ms", "stream", "private": true, "verified", "verification": "client-reported"}
```

Audit events: `org.created`, `org.disabled`, `key.created`, `key.revoked`, `sdk.enroll.ok`, `sdk.enroll.failed`
(`detail.reason`: unknown_key, revoked, expired, wrong_env, org_disabled, model_scope, install_limit, enroll_rate,
quota), `quota.exceeded`, `admin.denied`. `attest` lets a launcher holding a bootstrap admin token check, before it
sends the token, that the TLS endpoint it reached holds the provider CA:
`mac = HMAC-SHA256(key = SHA-256(admin token) as 32 raw bytes, "pgw-admin-attest-v1" | nonce | ca_sha256)`.

### A5. Provisioning (owner: provisioning agent; `src/pgw/provision/`, `src/pgw/service/provision.py`)

**Client API** (stable names; extra keyword arguments may be added with defaults):

```python
from pgw.provision.client import provision, ProvisionedModel
from pgw.provision.session import ProvisionSession

session = ProvisionSession(base_url="https://gpu.example.com:8443", token="<service token>",   # from the SDK bootstrap
                           ca_file=..., client_cert_file=..., client_key_file=...)      # all optional but base_url
pm = provision("qwen2.5-7b-instruct",            # model name (resolved via GET .../models) or a 64-hex fingerprint
               session=session, cache_dir=Path("~/.cache/pgw"),
               progress=None,                    # callable(dict) -> None, see "progress events"
               hint="provider",                  # "provider" (provider-built, client-verified) | "local" | "none"
               expected_fingerprint=None,        # pin; default: the session's catalog entry (see trust note)
               masked_head=True,                 # hint/store keyed by head_fingerprint(fp), head linear included
               source="provider",                # artifacts from the provider | "hf" (download HF + pack locally)
               hf_repo=None, hf_revision=None,   # source="hf": hub id or local checkpoint dir; packing parameters
               hf_name=None, hf_plain_dtype=None,  # default to the pinned manifest's (fetched from the provider)
               factory_pack=False)               # also keep the full pack (int8 weights) for the client's factory
```

Errors: `IntegrityError` (a chunk, an object or a hint operand failed its check; nothing is stored),
`FingerprintMismatchError` (the pin and what was served / derived differ), `ModelNotFoundError`,
`ServiceUnavailableError` (provider unreachable after retries), `ConfigError`. A failed hint check is not retried
automatically: retry (fresh seeds and challenges) or pass `hint="local"`.

`ProvisionedModel` (frozen dataclass):

| field | type | meaning |
|---|---|---|
| `name` | `str` | served model name (manifest) |
| `fingerprint` | `str` | base pack fingerprint (manifest; what `models.json` pins) |
| `factory_fingerprint` | `str` | `head_fingerprint(fingerprint)` with `masked_head`, else `fingerprint`: the key of the hint store and of mask segments |
| `pack_dir` | `Path` | weight-free gateway pack: `manifest.json`, the plain safetensors, `linear-meta.safetensors` (every linear's scale/bias) and `tokenizer/`; open with `pm.open_pack()` |
| `tokenizer_dir` | `Path` | `pack_dir / "tokenizer"` |
| `manifest` | `pgw.modelpack.Manifest` | verified against `fingerprint` |
| `hint_root` | `Path` or `None` | `PKSHintStore` root (`PKSHintStore.open(hint_root, factory_fingerprint, mac_key=...)`) |
| `hint_store` | `PKSHintStore` or `None` | opened store (holds the shared store lock; `close()` releases it) |
| `hint_key_file` | `Path` or `None` | customer key that MACs the store's `meta.json` (`<cache_dir>/keys/pks-hint.key`, 0600) |
| `hint_source` | `"provider-verified"`, `"local"` or `None` | how the hint was obtained |
| `masked_head` | `bool` | |
| `sizes` | `dict[str, int]` | bytes: `artifacts`, `hint`, `downloaded`, `weights_streamed`, `h_streamed` |
| `timings` | `dict[str, float]` | seconds per stage (`artifacts`, `factory_pack`, `hint`) |
| `factory_pack_dir` | `Path` or `None` | with `factory_pack=True`: `models/<fp[:16]>/factory-pack`, a full `ModelPack` (same fingerprint) for the background mask factory, built once from the fingerprint-checked `W8` (or kept from `source="hf"`); the hint check / local build then read `W8` from it |

`pm.open_pack()` returns a `pgw.provision.pack.GatewayPack`: a `ModelPack` (same `fingerprint`, `arch`,
`linear_specs()`, `load_plain`, `load_linear_meta`, `load_tensor`, `tokenizer_dir`, `manifest`) whose `load_linear`
raises `ConfigError` (the gateway holds no int8 weights). Wrap it in `MaskedHeadPack(pack, float_head=False)` for the
masked head. The in-process factory needs `W8` only for its private check vectors (`v W8`, once per process):
`pgw.provision.client.stream_weights(pm, session)` yields fingerprint-checked `LinearWeights` one at a time.

Cache layout (under `cache_dir`, created 0700): `models/<fp[:16]>/pack/`, `models/<fp[:16]>/hints/`,
`models/<fp[:16]>/.lock` (`flock`: concurrent `provision` calls for one model are serialised), partial downloads in
`models/<fp[:16]>/partial/` (resumed chunk by chunk). Every directory is built under a temporary name and renamed.

Progress events: `{"stage": "catalog"|"artifacts"|"weights"|"hint"|"done", "done": int, "total": int,
"unit": "bytes"|"linears", "item": str}`.

**Provider endpoints** (mounted by `pgw.service.app.create_app`; same authentication as `/pgw/v1/compute`:
bearer service token and/or verified client certificate, `PGW-Protocol: 1`; errors are the wire JSON error body).
`<fp>` is the base pack fingerprint.

| method, path | request | response |
|---|---|---|
| `GET /pgw/v1/provision/models` | - | `{"models": [{"name", "fingerprint", "masked_head": bool, "hint_build": bool, "device": "cuda" or "cpu"}]}` |
| `GET /pgw/v1/provision/<fp>/manifest` | - | provisioning manifest (below) |
| `GET /pgw/v1/provision/<fp>/object?id=<id>&chunk=<i>` | - | raw bytes of chunk `i` of object `id` (`application/octet-stream`, header `PGW-Chunk-Sha256`) |
| `POST /pgw/v1/provision/<fp>/hint` | `{"a_seed": <64 hex>, "masked_head": bool, "linears": [keys] or null}` | `{"job": str, "scheme": "pks-v3", "form": "h56", "rows_per_chunk": int, "device": str, "linears": [{"key", "m", "n", "chunks"}]}` |
| `GET /pgw/v1/provision/hint/<job>/h?linear=<key>&chunk=<i>` | - | rows `[i R, min(m, (i+1) R))` of `H = W8 A(a_seed) mod 2^56`, 7 bytes little endian per entry, row major (`rows x 4096 x 7`) |
| `DELETE /pgw/v1/provision/hint/<job>` | - | `{"ok": true}` |

Provisioning manifest: `{"format": "pgw-provision", "version": 1, "fingerprint", "name", "chunk_bytes",
"objects": [{"id", "kind": "manifest"|"tensor"|"tokenizer"|"weight", "name", "dtype", "shape", "size", "sha256",
"chunks": [sha256 hex per chunk]}]}`. `manifest.json` is checked against the pinned fingerprint first; every other
object's `sha256` must equal the value the fingerprint-bound pack manifest gives (tensor bytes / tokenizer file), so
the provider's own hashes are only used to reject a bad chunk early and to resume. `weight/<linear>` objects (int8
`W8`) are streamed for hint verification / local build / factory check vectors and never cached.

Hint: the client draws `a_seed` and `ks_seed` (CSPRNG). Only `a_seed` is sent (the provider sees it on every
maskgen request anyway); `ks_seed` never leaves the client. The provider computes the expensive `H = W8 A` (GPU when
available); the client verifies every linear's `H` against the fingerprint-checked `W8` with a 2-adic Freivalds
check (soundness error `<= 2^-41`: 42 binary challenge columns on each side), then computes the cheap deterministic
rest locally (`switch_hint`, key switch) and writes the same `PKSHintStore` a local build writes (bit-identical
hint files for the same seeds). The final key-switched hint itself is not checkable cheaply (two exact roundings
between `W8` and it); see `research/reviews/pks-v3-provider-hint.md`.

Optional (opt-in) hint streaming: `pgw.provision.streaming.StreamingHintStore.open(hint_root, factory_fingerprint,
source, mac_key=..., cache_bytes=...)` is a drop-in for the `PKSHintStore` the factory backend reads, with the hint
files in customer storage (`offload(store, dest)`; `DirectorySource` / `HttpRangeSource`) and an LRU of whole linears;
every fetched linear is checked against the HMAC'd local `meta.json`. It re-reads the whole hint once per segment
when the cache is smaller than the hint (about 55 MB/s at 7B and 25 slots/s): for disk-constrained hosts only.

Trust note: `expected_fingerprint` is the integrity anchor. Without it the catalog's fingerprint (provider-asserted,
like the enrolled `models.json`) is used; `source="hf"` re-derives the pack from the public checkpoint and checks that
it yields the same fingerprint.

### Console agent: the developer platform and what it reads

The developer platform (`src/pgw/platform/`, served at `/platform/`) reads everything through
`pgw.platform.client.AccountsClient`, a synchronous `Protocol` run on a worker thread: `authenticate(token)`,
`principal_for_sso(sub=, email=, role=)`, `org`, `list_keys`, `create_key`, `revoke_key`, `usage(since=, until=)`,
`requests(limit=, before=, key_id=, model=, status=)`, `models`, and optionally `audit(limit=)`. Every call is
scoped to the organisation of the signed-in session; no organisation id is taken from a request.
`pgw.platform.fixtures.FixtureAccounts` implements it with sample data (tests, screenshots,
`python -m pgw.platform --demo`).

`HttpAccountsClient` speaks the provider's **admin API (A4)**:

* Sign-in with an organisation admin token (`pgw service admin-token create --org ORG`): `GET /orgs` with that
  token must list exactly one, enabled organisation. Every later call uses the platform's own global admin token
  (`PGW_PLATFORM_ACCOUNTS_URL`, `PGW_PLATFORM_ACCOUNTS_TOKEN_FILE`). SSO users (`PGW_PLATFORM_OIDC_*`, redirect
  `.../platform/sso/callback`) belong to `PGW_PLATFORM_OIDC_ORG`.
* keys: `GET/POST /orgs/{org}/keys` (`{"key": "pgw_...", "record": key}`, the key shown once),
  `POST /orgs/{org}/keys/{key_id}/revoke`; usage: `GET /orgs/{org}/usage?since&until` (tokens and trips per key,
  model and UTC day); audit: `GET /orgs/{org}/audit`.
* Implemented in A4 since the integration (the platform still degrades gracefully against an older provider: a
  dash, "Request log not available"):
  1. `GET /orgs/{org}/models` -> `{"models": [{"id", "name", "status": "available|degraded|unavailable|preview",
     "params", "context_length", "description", "ttft_ms", "tokens_per_s"}]}` (404 -> empty Models page).
  2. `GET /orgs/{org}/requests?limit&before&key_id&model&status` -> `{"requests": [{"id", "ts", "key_id", "model",
     "status": "ok|error", "verified": bool, "private": bool, "latency_ms", "ttft_ms", "prompt_tokens",
     "completion_tokens", "stream", "error_code"}], "next_before"}`, newest first: **metadata only, never content**
     (the platform parses by allow-list and drops any other field).
  3. Usage rows add `requests`, `prompt_tokens`, `completion_tokens`, `verified`, `errors` when the client library
     reported requests that day (A3b). Without them the platform shows tokens and compute round trips and says
     requests and checks are counted in the client. `verified` is client-reported: pages say "checked
     client-side", never that the provider verified anything.

Mounting on the provider service (`pgw.service.app._mount_platform`, called by `create_app`): with
`PGW_PLATFORM_ENABLED=1` the console is mounted at `/platform` on
`pgw.service.platform_accounts.ServiceAccountsClient` (the same store and runtime as the admin API: sign-in with an
organisation admin token, `pgw service admin-token create --org ORG`), or, with `PGW_PLATFORM_ACCOUNTS_URL` (+
`_TOKEN_FILE`, `_CA_FILE`), on `HttpAccountsClient` against another provider's admin API. Enabling it without
accounts (`PGW_SERVICE_ACCOUNTS_DB`) or an accounts URL is a configuration error.

### Packaging (packaging agent)
- Distribution `pgw-sdk` (`pgw` on PyPI is an unrelated project), import name `pgw`, version from
  `pgw.__version__` only. Python >= 3.11 (abi3 wheels: one per platform). Base install = the client: add any
  new runtime dependency of `pgw.sdk` / `pgw.provision` to `[project].dependencies`; server-only packages go
  to extras (`gateway`, `service`, `factory`, ...). `pgw-sdk[client]` is an alias of the base install.
- Native extensions stay top-level modules `pgw_fast`, `pgw_ntt`, `pgw_pks` (setup.py, setuptools-rust);
  `PGW_REQUIRE_NATIVE=1` makes their build mandatory (CI, releases).
- CLI: `pgw doctor` reports the native extensions (check `native`); `pgw sdk status` / `pgw sdk doctor [--online]`
  (the SDK's group) are documented in `docs/sdk/configuration.md` (docs tests resolve every documented command).
- Release smoke test `scripts/smoke_installed.py --require-sdk` imports `from pgw import OpenAI, AsyncOpenAI`
  from the installed wheel; `examples/` and `tests/packaging/test_examples.py` assume the OpenAI surface
  (`chat.completions.create`, attribute access, sync/async streaming, `stream_options`, `usage`).
- Client options documented in `docs/sdk/configuration.md` (the full `SDKOptions` set: `api_key` / `PGW_API_KEY`,
  `timeout` (also per request: `create(timeout=...)`), `cache_dir`, `state_dir`, `mask_scheme` (`"pks-v3"`
  default), `hint` (`"local"` default, `"provider"` opt-in), `model_fingerprints`, `usage_reports`, ...); model ids
  `qwen2.5-{0.5b,1.5b,3b,7b}-instruct`.

### A1 (SDK side). Superseded by A1 above

The SDK's own key format (5-byte checksum) was dropped at integration: there is one format and one codec,
`pgw.enroll.sdk_key` (A1 above, stdlib only, with its published test vector); `pgw.sdk.keys.parse_key` wraps it.

### A2. `POST /pgw/v1/sdk/enroll` (consumed by `pgw.sdk.session`; provider agent implements)

Request: `Authorization: Bearer <the whole SDK key>`, TLS verified against the pinned CA only (the SDK first does
`GET /pgw/v1/enroll/ca` unverified and checks its SHA-256 against the key's pin; nothing secret is sent before).

```json
{"client_name": "<[A-Za-z0-9][A-Za-z0-9._-]{0,63}>", "gateway_name": "<same>", "csr_pem": "<PEM CSR, P-256>",
 "sdk_version": "0.1.0"}
```

Response 200 (same fields as `POST /pgw/v1/enroll`, plus `masked_head` per model and `lane`):

```json
{"tenant": "acme", "service_token": "<bearer for compute>", "client_cert_pem": "<PEM, optional>",
 "models": [{"name": "qwen2.5-7b-instruct", "fingerprint": "<64 hex>", "ring": "K32A12C27",
             "rings": ["K32A12C27"], "masked_head": true, "hf_repo": "...", "revision": "..."}],
 "lane": {"transport": "tcp", "port": 8444}}
```

* `models[].name` must be the lowercase names of `pgw.sdk.options.SUPPORTED_MODELS`
  (`qwen2.5-{0.5b,1.5b,3b,7b}-instruct`); the SDK refuses other models with `UnsupportedModelError`.
* `client_cert_pem`: sign the CSR as `/pgw/v1/enroll` does (map the CN to the tenant). Without it the SDK uses the
  token alone (issue it token-only then).
* `lane` (optional): the service's TCP+TLS trip lane (`PGW_SERVICE_LANE=tcp`, `lane_port`). The SDK adds
  `https://<service host>:<port>` to its egress allowlist and uses `lane = "tls"`; without it it uses the
  blocking WebSocket stream lane (`lane = "stream"`). `null` / absent = no lane.
  Lane format 2: a `POST /pgw/v1/lane/open` reply that has `"versions": [1, 2]`, `"max_items"` and
  `"max_multi_rows"` (next to `"version": 1`) offers `multi` frames (`rust/pgw-lane/src/proto.rs`): the SDK then
  sends the items of a batched trip (the rows of several concurrent answers, a prompt's rows) in one lane call,
  the same items, order, bytes and request id as the stream channel's compute request. A reply without
  `versions` (format 1) gets one item per call; several-item trips then take the stream lane.
  `PGW_SERVICE_LANE_MULTI=0` makes a service answer format 1.
* `masked_head` is informational: the SDK decides the masked LM head from the service's `GET /pgw/v1/models`
  (the derived fingerprint `pgw.modelpack.masked_head.head_fingerprint(fp)` must be served:
  `PGW_SERVICE_MASKED_HEAD=also`).
* Errors: 401 unknown/revoked/expired key -> `AuthenticationError`; 403 -> `PermissionDeniedError`; 429 ->
  `RateLimitError` (honour `Retry-After`); 400 -> `BadRequestError`; 404 (no such endpoint) -> `APIConnectionError`.
  Error body `{"error": {"code", "message"}}`.
* Re-enrollment: the SDK enrolls again (same key) when the cached service token is refused (401 at connect) or the
  provider adds a model; the provider should issue a fresh token each time (old tokens may stay valid until revoked).
* The SDK tests run against the real implementation (A2 above): `tests/sdk/provider_main.py` starts the real
  service with accounts, the admin API, provisioning and the console; keys are created and revoked through the
  admin API. The provider's reply carries `lane` and per-model `masked_head` as described here.

### A3. Provisioning (SDK side of interface 3; as integrated)

The default provisioner is `pgw.sdk.provisioning.ProviderProvisioner` (built from `SDKOptions`: `hint`,
`model_fingerprints`), a layer over the real `pgw.provision.client.provision` (A5), called in a worker thread once
per process and model:

```python
provision(model, *, session: pgw.sdk.session.Session, cache_dir: Path, progress: Callable[[str], None] | None,
          masked_head: bool)   # masked_head is passed to provisioners that accept it
```

* **Pins.** The provider's catalog is never trusted alone. `PINNED_MODELS` pins every supported model to its public
  checkpoint (repo, commit = `deploy/runpod/pod_boot.MODELS`) and, where derived for the release, its pack
  fingerprint (0.5B `d773a502...`, 1.5B `3d198de4...`, 3B `0ec0b8b2...`, 7B `f7c6fb4f...`:
  `scripts/pin_model_fingerprints.py`, i.e. `pgw pack <repo> --revision <commit> --name <model>` with default
  parameters, pack format 2; derived on a Kaggle CPU kernel, 0.5B / 1.5B also on the integration box, identical;
  the 0.5B pin is also re-derived and checked by `tests/sdk/test_e2e_product.py` when the checkpoint is cached). The
  larger models are pinned ahead of `SUPPORTED_MODELS`: 14B `d4afbe77...` (`cf98f3b3...`), 32B `a4cf6ba9...`
  (`5ede1c97...`, the provider image's commit), 72B `5e8001c0...` (`495f3936...`), derived on Kaggle CPU kernels with
  the shard-wise packer (`pgw.modelpack.shardwise`, byte-identical to `pgw pack`; at 14B also checked file by file
  against `pgw pack` on the whole checkpoint). A
  provider serving another fingerprint is refused (`FingerprintMismatchError`) before any download. A pin without a
  fingerprint (a future model) is checked on first use instead: the entry's
  `hf_repo`/`revision` must equal the pin, and the first provisioning runs `provision(source="hf", hf_repo=...,
  hf_revision=...)`, which packs the public checkpoint locally and requires the provider's fingerprint (verified in
  full); the result is remembered in `models/<fp16>/public-check.json`. That pack is built shard by shard
  (`stream_pack_checkpoint`: one checkpoint shard downloaded, packed and deleted at a time, resumable from
  `models/<fp16>/hf-stream`), so it needs about one shard (4 GB) plus the int8 pack in disk, not the bfloat16
  checkpoint (29 / 65 / 145 GB at 14B / 32B / 72B); a checkpoint already complete in the local Hugging Face cache is
  read there. `SDKOptions.model_fingerprints` adds or replaces pins (custom
  packs; tests with the tiny model).
* **Hint.** `hint="local"` (default; decision: the provider-built path streams ~44 GB of `H` at 7B):
  `provision(hint="none")` and the background factory builds the pks-v3 hint from the factory pack
  (`models/<fp16>/hint`). `hint="provider"`: `provision(hint="provider")`, checked against `W8` from the factory
  pack.
* **Packs.** The in-process gateway opens the weight-free gateway pack (`pack_path`; `pgw.gateway.runtime` opens a
  directory with `linear-meta.safetensors` as a `GatewayPack`, full verification); the factory opens the factory
  pack (`factory_pack_path`, a full `ModelPack` with the same fingerprint, verified in full by the factory).
* `cache_dir` passed to the provisioner is `<cache>/models`; `provision` is called with `<cache>` (its own
  `models/<fp16>/...` and `keys/pks-hint.key` layout).
* Raise `pgw.core.errors.ConfigError` / `IntegrityError` / `FingerprintMismatchError` on failures; the SDK maps
  them to its typed errors. `LocalPackProvisioner` (full packs on disk) stays for air-gapped installs and tests.

### A4. SDK defaults other agents should know (SDK agent)

* Ring `K32A12C27`; `local_linears="off"`, `float_ops="native"`, native trip / decode drivers
  (`PGW_TRIP_FAST/NATIVE/DRIVER`, `PGW_DECODE_NATIVE`, `PGW_TRIP_DECRYPT_AHEAD`, `PGW_LANE_SPIN_AUTO_US=400`,
  `PGW_LANE_WAIT_PREFETCH=1`, set with `setdefault`), p31 checks (factory `check_field=p31`), masked LM head when
  served, `mask_reservation="incremental"`, `freivalds_rate=1.0`, noise monitor fail-closed, `pack_verify="full"`.
* Prefix reuse (`prefix_cache="conversation"`) is proven exact only with the gateway's own LM head, so the SDK
  default is: masked head when the provider serves it, else prefix reuse (`prefix_cache=True` forces prefix reuse
  and turns the masked head off).
* Mask factory: `mask_scheme="pks-v3"` (with `PKS_V3_ACK`, pending external review), one supervised child process
  per (key, model) running `pgw factory run` settings, `workers = clamp(cpu_count // 4, 1, 8)`, `nice 10`, 1 torch
  thread per worker, target = `mask_store_gb` (8 GB) / slot bytes (64..16384 slots), low water = target / 2.
  pks-v3 needs `out_features % 32 == 0` and `in_features % 16 == 0` for every linear (true for Qwen2.5).
* Files: cache `~/.cache/pgw` (`PGW_CACHE_DIR`): `sessions/<key_id>/` (enrollment files, `masks/`, `factory/`,
  `status.json`), `models/` (provisioning). State `~/.local/state/pgw` (`PGW_STATE_DIR`):
  `sessions/<key_id>/masks.kek`, `masks.counter` (rollback counter), `pks-hint.key`.
