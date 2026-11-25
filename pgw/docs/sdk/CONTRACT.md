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

### A1. SDK key (SDK agent; `pgw.sdk.keys`) — provider issues with `SdkKey.new(...).encode()`

`pgw_<env>_<body>`; `env` = `[a-z0-9]{1,16}` (`live`, `test`, ...). `body` = lowercase RFC 4648 base32, no padding, of

```
version (1 byte = 1) | ca_sha256 (32) | key_id (8) | secret (24) | provider URL (UTF-8, <= 200 B) | checksum (5)
checksum = SHA-256("pgw-sdk-key-v1" | 0x00 | env | 0x00 | everything before it)[:5]
```

* `ca_sha256`: SHA-256 of the DER of the CA that issued the provider's TLS certificate (same pin as enrollment
  codes, `pgw.service` `enrollment_ca_file`). 32 zero bytes only for `http://` loopback.
* `key_id` (16 hex via `SdkKey.key_id_hex`): public, stable; the provider's record key (revocation, metering, logs).
* The provider stores only `SdkKey.secret_sha256()` and compares in constant time. The SDK never prints the key.
* Provider side: `pgw.sdk.keys.parse_key(token)` on the bearer value; refuse unknown / revoked `key_id` or a
  secret hash mismatch with **401**.

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
* `masked_head` is informational: the SDK decides the masked LM head from the service's `GET /pgw/v1/models`
  (the derived fingerprint `pgw.modelpack.masked_head.head_fingerprint(fp)` must be served:
  `PGW_SERVICE_MASKED_HEAD=also`).
* Errors: 401 unknown/revoked/expired key -> `AuthenticationError`; 403 -> `PermissionDeniedError`; 429 ->
  `RateLimitError` (honour `Retry-After`); 400 -> `BadRequestError`; 404 (no such endpoint) -> `APIConnectionError`.
  Error body `{"error": {"code", "message"}}`.
* Re-enrollment: the SDK enrolls again (same key) when the cached service token is refused (401 at connect) or the
  provider adds a model; the provider should issue a fresh token each time (old tokens may stay valid until revoked).
* A reference implementation used by the SDK tests is `tests/sdk/provider_main.py::_add_sdk_route` (reuses
  `pgw.enroll.service` token issuing and CSR signing).

### A3. Provisioning (SDK side of interface 3; provisioning agent implements `pgw.provision.client.provision`)

Called by name (`pgw.sdk.provisioning.PROVISION_FUNCTION = "pgw.provision.client:provision"`), in a worker thread,
once per process and model, with the model's canonical name:

```python
provision(model: str, *, session: pgw.sdk.session.Session, cache_dir: Path, progress: Callable[[str], None] | None)
```

* `session` attributes: `service_url`, `token()` (`SecretStr`), `token_file`, `ca_file` (pinned CA PEM or None),
  `client_cert_file` / `client_key_file` (or None), `models`, `model_entry(name)`, `tenant`, `key_id`, and
  `http_client(timeout_s=...)` (a blocking `httpx.Client` to the provider: CA pinned, client cert presented,
  bearer token set). Use these for `/pgw/v1/provision/...`.
* `cache_dir` = `<cache>/models` (0700; shared by all keys and processes on the host: lock your own files).
* Return any object (or mapping) with: `fingerprint` (64 hex; must equal the enrollment entry's, else the SDK raises
  `IntegrityError`), `pack_path` (alias `pack_dir` / `path`: a directory `ModelPack.open(verify="full",
  expected_fingerprint=fp)` accepts — the gateway re-verifies it at every start; the factory opens it with
  `components=("linears",)`), optional `hint_dir` (alias `hint_root` or `hint_store.root`: the root passed as
  `he_hint_dir`, i.e. `PKSHintStore.open(hint_dir, fp)` must find it), optional `hint_key_file`, optional `name`,
  `source`. Without `hint_dir` the factory builds the pks-v3 hint locally under `<cache>/models/<fp16>/hint`.
* With the masked LM head the factory opens `MaskedHeadPack(pack)` and needs the head's hint too (derived
  fingerprint); a provisioned hint store that holds only the base linears is completed locally by the factory.
* Raise `pgw.core.errors.ConfigError` / `IntegrityError` / `FingerprintMismatchError` on failures; the SDK maps
  them to its typed errors.

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
