# PGW (Private Gateway): Specification v1

Status: **normative for protocol version 1**. Owner: lead architect. Changes to anything marked
*normative* require an entry in `INTERFACE_NOTES.md` and a SPEC update in the same change.

PGW runs private LLM inference on public open-weight models (Qwen2.5, Llama 3). The heavy linear
algebra runs on our GPUs, but our compute service only ever sees uniformly random ring elements. The
customer keeps the plaintext and the small non-linear work, and holds the one-time masks.

---

## 1. Goals and non-goals

**Goals**
- Input privacy of every activation against the compute service: perfect secrecy for ideal one-time
  masks; computational at the PRG level as deployed (masks are expanded from ChaCha20 seeds, 7.3).
- Bit-exact results: masked inference gives exactly the logits of the same quantized model run locally.
- Integrity: wrong or malicious service results are detected with overwhelming probability.
- Production operations: an OpenAI-compatible API, API keys, rate limits, metrics, mTLS, crash-safe
  single-use mask accounting, reproducible model packs.
- Server cost close to ordinary int8 serving. Batching across sessions and tenants must work.

**Non-goals (v1)**
- Secret model weights. Weights are public; this is what makes the scheme cheap.
- Hiding metadata such as token counts and timing (see 3.4 for the mitigations offered).
- Protecting against a compromised gateway host. The gateway is inside the customer trust boundary.
- Training and fine-tuning. MoE and multimodal models.

## 2. Architecture

```
 customer trust boundary                                                   our infrastructure
┌──────────────────────────────────────────────────────────┐            ┌─────────────────────────┐
│  apps ──OpenAI API──▶ GATEWAY (pgw.gateway + pgw.engine)  │  x_hat     │  COMPUTE SERVICE        │
│                       plaintext: embed, norms, RoPE,      │ ─────────▶ │  (pgw.service)          │
│                       attention+KV cache, SiLU, LM head,  │  HTTP/2    │  y_hat = W8·x_hat mod2^K│
│                       sampling; mask/unmask; Freivalds    │ ◀───────── │  stateless, no logs     │
│                            ▲ reserve/take                 │  y_hat     │  public int8 weights    │
│                     MASK STORE (pgw.maskstore)            │            └─────────────────────────┘
│                     encrypted, single-use segments        │
│                            ▲ segments                     │
│   MASK FACTORY (pgw.factory) ── customer GPU/CPU, or ─────┼──▶ independent PARTNER factory
└──────────────────────────────────────────────────────────┘     (never the service operator)
          all three load the same MODEL PACK (pgw.modelpack), bound by its SHA-256 fingerprint
```

Per generated or prompt token, per layer, the four masked linear groups are `qkv` (q,k,v
concatenated), `o`, `gate_up` (gate,up concatenated) and `down`. That is `4·L` round trips per
forward pass. Prefill sends all prompt tokens of a group in one round trip.

**Protocol for one linear `y = W x (+ b)`**

| phase | where | operation |
|---|---|---|
| offline | factory | seed → `r ∈ Z_{2^K}^{n}` per slot. Store `W8·r mod 2^K` and the seed (encrypted). Store Freivalds `u`, `uᵀW8`. |
| online | gateway | `xq, sx = quantize(x)`; `x_hat = (xq + r) mod 2^K` → service |
| online | service | `y_hat = W8·x_hat mod 2^K` (K/8 int8 limb GEMMs) → gateway |
| online | gateway | check `u·y_hat == (uᵀW8)·x_hat`; `yq = centered(y_hat − W8 r)`; range check; `y = yq·sx·sw + b` |

## 3. Threat model (summary)

### 3.1 Parties and trust
| party | sees | trusted for |
|---|---|---|
| Gateway (customer) | everything in plaintext | everything (inside the customer boundary) |
| Mask store (customer) | masks, seeds | confidentiality of masks; single use |
| Mask factory (customer or partner) | seeds, masks, model | must **not collude** with the service operator |
| Compute service (us) | `x_hat`, `y_hat`, sizes, timing | nothing: assumed honest-but-curious **or malicious** |
| Network | TLS ciphertext | nothing |

### 3.2 Guarantees
- **Privacy.** Each `r` is uniform over `Z_{2^K}^n`, secret, and used exactly once, so `x_hat` is
  uniform and independent of `x`. For ideal masks this holds against unbounded adversaries; as
  deployed, masks come from HKDF-SHA-256 + ChaCha20 seeds (7.3), so the guarantee is computational at
  the PRG level (whitepaper 4.4). Either way it requires that (a) masks never reach the service, (b) no mask is reused, and (c) factory and service do
  not collude. The activation scales `sx` never leave the gateway.
- **Integrity.** Every result passes a range check and `c` Freivalds checks on every call.
  False-accept probability per call is at most `2^(−c·h)` with
  `h = K − ⌈log2(4·qmax·127·n)⌉` (`pgw.core.soundness_bits`). For K40A12, `h` is 8 at n = 4096,
  6 for Llama-3-8B `down_proj` (n = 14336) and 5 for Qwen2.5-7B `down` (n = 18944). The factory
  therefore picks `c` per pack by default (`checks` unset = smallest `c` in [2, 8] giving at least
  20 bits for the worst linear): `c = 3` for n = 4096 (24 bits) and `c = 4` for 8B-class layers
  (24 and 20 bits). Gateways and mask stores refuse segments whose checks give fewer than
  `min_soundness_bits` (default 20) before any mask row is used. **Any** failure aborts the
  request, raises `integrity_check_failed`, and trips a per-service circuit breaker (operator
  alert). An adaptive attacker who probes the check is caught with constant probability on each
  probe, and is then cut off.
  Freivalds covers the service product only. A wrong `W r` from the factory is caught by the
  factory's own self-check (7.4) and, for gross errors, by the gateway range check; an independent
  spot-check of `W r` by a party holding the weights is planned (threat model T-29). At K32A8 very
  wide layers have `h` of 1–2 bits, so even `c = 8` cannot reach the floor: such packs need K40A12
  or K48A20 (the factory warns and the gateway refuses the segments).
- **Correctness.** Results are exact modulo `2^K`. Decoding is unambiguous because
  `qmax·127·n < 2^(K−1)` (checked when a pack is loaded).

### 3.3 Catastrophic failures (must never happen; defended in depth)
1. **Mask reuse.** `x1_hat − x2_hat = x1 − x2` leaks. Defenses: durable consumption cursor before
   use (7.5), rollback detection (7.6), duplicate-row detection in the noise monitor (11.3).
2. **Unmasked send.** Defense: the outbound-noise monitor checks that payloads look uniform and fails closed.
3. **Masks or seeds reaching the service.** They are never in any wire message, and the service has no endpoint that accepts them.
4. **Factory/service collusion.** This is contractual: the partner factory must be independent of us, and customers can run the factory themselves.

### 3.4 Metadata leakage (accepted, with mitigations)
The service learns the model, the number of tokens per round (prompt length, generation length),
timing, and the tenant. Optional mitigations: batch several sessions per request (implemented),
pad prompt rows to buckets (`pad_rows_to`) and send constant-rate dummy rounds (both planned, not
implemented in v1). Each costs masks and compute.

## 4. Components and ownership

| package | dir | responsibility | runs at |
|---|---|---|---|
| `pgw.core` | `src/pgw/core` | ring math, exact ringmm (CPU/GPU), mask/unmask, Freivalds, CSPRNG, packing, fingerprints, errors, settings, logging | everywhere |
| `pgw.modelpack` | `src/pgw/modelpack` | HF checkpoint → int8 pack + manifest + fingerprint; loader | build time; all |
| `pgw.engine` | `src/pgw/engine` | plaintext transformer runtime (Qwen2, Llama), KV cache, sampling, generate/stream, `LinearBackend`s | gateway |
| `pgw.factory` | `src/pgw/factory` | mask generation (CPU/GPU), weight streaming, refill scheduler | customer / partner |
| `pgw.maskstore` | `src/pgw/maskstore` | encrypted single-use store, key providers, reservations, low-water alerts | customer |
| `pgw.wire` | `src/pgw/wire` | messages, msgpack codec, HTTP/2 client (pool, retries, mTLS), server helpers | gateway, service |
| `pgw.service` | `src/pgw/service` | stateless masked compute server: registry, kernels, batching, auth, health | us |
| `pgw.gateway` | `src/pgw/gateway` | OpenAI-compatible API, SSE, API keys, rate limits, admin | customer |
| `pgw.observability` | `src/pgw/observability` | Prometheus metrics, outbound-noise monitor, dashboard | all |
| `pgw.cli` | `src/pgw/cli` | `pgw pack/factory/maskstore/service/gateway/doctor` | all |

Dependency rule: `core` depends on nothing internal. `modelpack` depends on `core`. `wire` depends
on `core`. `maskstore` depends on `core`. `engine` depends on `core`, `modelpack`, `wire` and
`maskstore`. `factory` depends on `core`, `modelpack`, `maskstore`, `wire` and `hemask`. `hemask` (encrypted mask
generation) depends on `core` and `wire`. `service` depends on `core`, `modelpack`, `wire`, `hemask` and
`observability`. `gateway` depends on everything except `service` and
`factory`. `cli` may import anything, lazily. **`service` must never import `maskstore` or
`factory`.** CI enforces this with an import-graph test.

## 5. Arithmetic and quantization (normative)

- **Ring** `Z_{2^K}`, `K ∈ {32, 40, 48}`, multiple of 8 in [16, 56]. Presets: `K32A12C27`
  (**default**, see below), `K40A12` (previous default), `K48A20` (max quality), `K32A8` (cheapest, measured lossy: ppl 11.09 vs 10.93 for
  Qwen2.5-0.5B). `K32A12C27` (default): a 32-bit ring with a per-call output cap of 2^27. The gateway
  quantizes each token with the most activation bits (up to 12) for which the exact Cauchy-Schwarz bound
  `max_j ||W8_j||_2 * ||xq||_2 <= 2^27` holds. That makes any layer width decodable, and the range check
  and Freivalds use 2^27 (7 vectors give at least 20 bits). It is lossless on Qwen2.5-0.5B, 1.5B and 7B
  (perplexity identical to `K40A12`, no token reduced) and gives 20% less traffic, service work and mask
  storage. Encrypted mask generation uses q = 2^56 with it. Canonical values are `int64` in `[0, 2^K)`. Centered values are in `[−2^(K−1), 2^(K−1))`.
- **Weights:** int8, symmetric per output row: `sw[j] = max_i |W[j,i]| / 127` (1.0 for zero rows),
  `W8 = clamp(round(W/sw), −127, 127)`. The value −128 is never produced, but the kernels handle it.
- **Activations:** signed A-bit, per token: `qmax = 2^(A−1) − 1`, `sx[t] = max_i |x[t,i]| / qmax`
  (1.0 for zero rows), `xq = clamp(round(x/sx), −qmax, qmax)`. Non-finite input raises an error.
- **Output:** `y = yq · sx[t] · sw[j] + b[j]` in float64, then cast. Biases stay float and are applied by the gateway.
- **Headroom:** `qmax·127·n < 2^(K−1)`, checked by `RingParams.check_in_features`.
- **Ring matmul** (`pgw.core.ringmm`, exact mod `2^K` on every backend):
  - `cuda_int8`: K/8 `torch._int_mm` GEMMs on balanced base-256 digits `d ∈ [−128,127]`. Rows are
    padded to at least 32 and to a multiple of 8; `n` and `m` are padded to multiples of 8. Accumulation
    is int32, and the inner dimension is chunked at ≤131040 (a multiple of 32: CUDA operands are padded to multiples of 32).
  - `cpu_f64` (CPU default): wide balanced digits with `2^(b−1)·128·n ≤ 2^52`, giving 2 float64 GEMMs
    for K≤48 when n≤4096. Every partial sum is an exact integer.
  - `cpu_f32`: base-256 digits with float32 GEMMs over inner chunks of 1024 (partial sums ≤ 2^24).
  - Planned: fbgemm/oneDNN int8 path for CPU (0.3 TOPS measured), and a fused Triton kernel for GPU.
- **Integrity:** `check_output_range` (|yq| ≤ qmax·127·n) plus Freivalds over `Z_{2^K}` (`pgw.core.freivalds`).

## 6. Wire protocol (normative)

### 6.1 Transport: HTTP/2 + msgpack (chosen over gRPC)
- Payloads are a few large opaque byte blobs, up to hundreds of MB during prefill. msgpack `bin`
  decodes without per-element parsing and has no protobuf 2 GB / gRPC 4 MB default message ceilings.
- The gateway and OpenAI API already use FastAPI/hypercorn/httpx, so one stack covers TLS, mTLS,
  auth, metrics, load balancing and debugging (`curl`). gRPC would add a second stack and codegen.
- Plain HTTPS passes through corporate proxies, API gateways, PrivateLink/PSC and L7 load balancers
  that often mishandle gRPC trailers.
- HTTP/2 multiplexing gives the per-connection concurrency gRPC would. HTTP/1.1 keep-alive is
  supported as a fallback (uvicorn). Clients in Rust or Go need only an HTTP client and msgpack.
- Cost: there is no IDL. This spec plus strict codec validation and conformance tests
  (`tests/wire/`) serve as the schema.
- Server: uvicorn by default (HTTP/1.1 via httptools, and the `websockets` framing for the
  `/pgw/v1/stream` channel that clients use for compute calls: about half the per-frame CPU of
  hypercorn's pure-Python framing), or hypercorn (h2 and h2c) with `PGW_SERVICE_SERVER=hypercorn`
  for HTTP-only clients that want HTTP/2 multiplexing. Both publish the verified client-certificate
  subject to the app. Client: `httpx` with `http2=True` (ALPN falls back to HTTP/1.1) and a
  connection pool, plus the stream channel.

### 6.2 Endpoints (compute service)
| method | path | body | response |
|---|---|---|---|
| POST | `/pgw/v1/compute` | `ComputeRequest` | `ComputeResponse` or `ErrorResponse` |
| GET | `/pgw/v1/models` | – | `ModelsResponse` |
| GET (WebSocket) | `/pgw/v1/stream` | one `ComputeRequest` per binary frame | one `ComputeResponse` or `ErrorResponse` frame per request |
| GET | `/healthz` | – | `200 ok` (process alive) |
| GET | `/readyz` | – | `200` when packs are loaded and the device is OK, else `503` |
| GET | `/metrics` | – | Prometheus text (separate port or IP allowlist recommended) |

Headers: `Content-Type: application/vnd.pgw+msgpack`, `PGW-Protocol: 1`,
`Authorization: Bearer <tenant token>` (alongside or instead of mTLS), and optionally
`PGW-Request-Id`. Compression: none, because uniform data does not compress.

`/pgw/v1/stream` (v1.2) carries the same messages over one persistent WebSocket, removing the
per-call HTTP cost (about 1-3 ms per call; a 7B token makes 112 sequential calls). The handshake
carries the same `PGW-Protocol` and `Authorization` headers and the same mTLS identity, and is
refused (close 1008) when authentication or admission fails. Requests are multiplexed and replies
come in completion order, matched by the request `id`. Every frame passes the same limits, byte
budget, admission check, deadline (`deadline_ms`), response self-check and metrics hook as
`POST /pgw/v1/compute`. A frame whose `id` cannot be read closes the channel. Clients fall back
to `POST /pgw/v1/compute` when the channel cannot be opened, open a new channel after an
`unavailable` reply (e.g. draining), and retry requests that were in flight on a broken channel.
Over TLS the stream's ALPN offers only `http/1.1`. Services can turn the channel off
(`PGW_SERVICE_STREAM_CHANNEL=false`); clients with `stream=false`.

### 6.3 Messages (msgpack maps, string keys)
Ring tensors are `bin` in `pack_ring` format: K/8 little-endian bytes per element, row-major.

```
ComputeRequest  {"v":1,"type":"compute","id":str,"fp":hex64,"ring":"K40A12",
                 "items":[{"linear":"3.qkv","rows":int,"x_hat":bin(rows*n*K/8)}...],
                 "deadline_ms":int?}
ComputeResponse {"v":1,"type":"compute_result","id":str,
                 "items":[{"linear":"3.qkv","rows":int,"y_hat":bin(rows*m*K/8)}...],   # same order
                 "timing":{"queue_ms":f,"compute_ms":f,"batch_rows":int}}
ErrorResponse   {"v":1,"type":"error","id":str?,"code":str,"message":str,"retryable":bool,"details":map}
ModelsResponse  {"v":1,"type":"models","versions":[1],"server":str,
                 "models":[{"name":str,"fp":hex64,"rings":[str],"linears":[{"id","m","n"}],
                            "max_rows_per_item":int,"max_request_bytes":int}]}
```
The dataclasses are in `pgw.wire.messages`. The server validates `len(x_hat) == rows·n·K/8`.
Items in one request are independent and may target different linears. Requests are pure functions,
so retries are safe, and a retry resends the identical `x_hat` without consuming a new mask.

### 6.4 Versioning
- Major version `1` appears in the path (`/pgw/v1`), the `PGW-Protocol` header and the `"v"` key.
  A server rejects unknown majors with `426 version_mismatch` and lists `versions`.
- Within a major version, new optional keys may be added, and receivers ignore unknown keys.
  Removing or retyping a key requires a new major version.
- The model fingerprint binds weights. The gateway calls `/pgw/v1/models` at startup and refuses to
  run if the fingerprint or ring is not served.

### 6.5 Batching and limits
- Client side: the engine batches all prompt tokens per group (prefill) and concurrent sessions at
  the same group (`forward_many`) into one request.
- Server side: `BatchScheduler` coalesces rows across requests and tenants for the same
  `(fp, ring, linear)` while the batch waits for a device worker (plus an optional `batch_window_ms`, default 0: no idle wait), up to `max_batch_rows`. Mixing
  tenants is safe because every row is uniform noise.
- Defaults: `max_request_bytes` 256 MiB, `max_items` 256, `max_rows_per_item` 8192. The client
  splits larger requests by rows.

### 6.6 Error codes (`pgw.core.ErrorCode`; stable, never reused)
| code | HTTP | retry | meaning |
|---|---|---|---|
| `internal` | 500 | no | bug; details not leaked |
| `config` | 500 | no | invalid configuration |
| `bad_shape` | 400 | no | tensor shape/dtype/length mismatch |
| `ring_overflow` | 400 | no | value not representable in ring |
| `integrity_check_failed` | 502 | no | Freivalds/range check failed (gateway side) |
| `fingerprint_mismatch` | 409 | no | unknown/mismatched model fingerprint |
| `mask_exhausted` | 503 | no* | mask store empty (*retry after refill) |
| `mask_reuse` | 500 | never | attempted reuse; store must be investigated |
| `mask_store` | 500 | no | store I/O, decryption or format failure |
| `protocol` | 400 | no | malformed message |
| `version_mismatch` | 426 | no | unsupported protocol version |
| `backend_unavailable` | 503 | yes | compute backend not available |
| `unauthenticated` / `permission_denied` | 401/403 | no | auth |
| `rate_limited` | 429 | yes | honor `Retry-After` |
| `unavailable` | 503 | yes | overload/draining |
| `model_not_found` | 404 | no | unknown model |
| `payload_too_large` | 413 | no | exceeds limits |
| `deadline_exceeded` | 504 | yes | deadline passed |

### 6.7 Authentication
- Service: mTLS (customer client certificates issued by our CA, one per gateway) and/or bearer
  tenant tokens, stored server-side as SHA-256 hashes. The tenant id drives quotas and billing counters.
- Gateway: API keys `pgw-<key_id>-<secret>` (stored hashed) for apps, and a separate admin token
  for `/admin/*`.

## 7. Masks

### 7.1 Lifecycle
factory generates segment → store imports or writes it (encrypted) → gateway reserves slots
(durably consumed) → engine takes rows per linear → unused rows are burned at session end → fully
consumed segments are shredded (file deleted and wrapped DEK destroyed).
**Slot** = one token position. It contains one mask row for each of the `4·L` linears.

### 7.2 Segment file (`*.pgwm`)
```
magic "PGWMASK1" | u32le hlen | header JSON (SegmentHeader, canonical) | chunks...
chunk = AES-256-GCM(DEK, nonce = nonce_prefix(4) || u64be(chunk_index),
                    aad = SHA-256(header JSON) || u32be(chunk_index)) -> ciphertext || tag(16)
chunk 0          : seed(32) || for each linear in header order: uW packed (checks × n)
chunks 1..       : for each linear (header order), for each run of chunk_slots slots:
                   W8·r rows packed (rows × m)
```
The header (`pgw.maskstore.format.SegmentHeader`) holds segment_id, fingerprint, ring, slots,
chunk_slots, checks, linears [{id, m, n}], kek_id, wrapped_dek, nonce_prefix, factory_id and
created_at. All chunk offsets are computable from the header, so a reader decrypts only the
chunks it needs. Size per slot is `Σ m · K/8` bytes, for example 1.5 MB for Qwen2.5-0.5B at K40 and
6.9 MB for Llama-3-8B at K40 (8.3 MB at K48). `r` is **not stored**; it is regenerated from the seed.

### 7.3 Mask derivation (normative; `pgw.core.prng`)
```
key(labels) = HKDF-Expand(SHA-256, prk=seed, info="pgw/v1/prng" || Σ u32be(len)||typed(label), 32)
stream      = ChaCha20(key, counter=0, nonce=0)   ; ring elem = u64le(8 bytes) mod 2^K
r[slot, :]  = SeedExpander(seed).ring_uniform_rows(n, K, slot, 1, "r", linear_id)
u           = SeedExpander(seed).ring_uniform((checks, m), K, "u", linear_id)
```
Known-answer vectors are in `tests/core/test_prng.py`. Every implementation (for example a Rust or
CUDA factory) must reproduce them. The segment seed comes from `os.urandom(32)` and is unique per segment.

### 7.4 Factory
`MaskGenerator` computes `W8·r` with `PreparedWeight` (GPU `cuda_int8`; measured T4 8B K48 0.0066 s
per slot, about 550k slots/h). Weight streaming covers models larger than device memory (70B on T4:
0.047 s per slot at batch ≥1024). `FactoryScheduler` keeps every `(fingerprint, ring)` between
`low_water_slots` and `target_slots`. Output goes to a local store or to exported files (partner).

### 7.5 Single use (normative)
- The store keeps `state.sqlite` (WAL) with, per segment, `next_slot` and `status`, plus a global
  `generation` counter.
- `reserve(n)` runs `BEGIN IMMEDIATE` → advance cursors → bump generation → `COMMIT` + fsync,
  **before** any mask bytes are decrypted for the caller. A crash loses masks but never reuses them.
- A reservation hands out rows strictly in order, per linear. Rows are never returned. `close()` zeroizes buffers.
- A reservation that spans segments is split at extent boundaries, because `u` differs per segment.
- The gateway reserves `len(prompt) + max_tokens` slots per request. Unused slots are burned and counted in `pgw_masks_burned_slots_total`.

### 7.6 Keys and rollback
- Envelope encryption: a random 32-byte DEK per segment, wrapped by a KEK from a `KeyProvider`:
  `file` (0600 key file), `env` (base64), `pkcs11` (HSM, AES-KWP) or `kms` (AWS, GCP or Azure; stub in v1).
- Rollback protection: restoring an old `state.sqlite` would allow reuse. The generation counter is
  mirrored to a monotonic counter when `PGW_MASKSTORE_COUNTER` is set (`provider`, `file` on separate
  storage, `tpm` preliminary). MAC-capable providers also keep an HMAC-chained generation journal
  (default `<root>/generation.journal`). The journal detects a restore of the database alone, but
  not a restore of a snapshot of the whole volume (database and journal together): only a counter off
  the mask volume detects that. With `counter = none` the store refuses to hand out masks
  (`reserve()` raises `config`) unless `PGW_MASKSTORE_ALLOW_NO_EXTERNAL_COUNTER` is set (dev/staging
  only), and `pgw doctor` reports `fail`. Every shipped deployment configures a counter (Helm: a
  separate `<release>-counter` PVC excluded from backups; systemd/compose: a counter file on its own
  volume; VMware: an independent-persistent disk or the vTPM). If the mask volume is ever restored,
  burn every ACTIVE segment and rebind witnesses before serving.
- The store refuses to open on a network filesystem (NFS, CIFS, CephFS, GlusterFS, FUSE, ...):
  SQLite WAL and `flock` are unreliable there (`allow_network_filesystem` overrides). A store is
  never shared between hosts; factories on other hosts deliver segments for `import_segment`. A store with neither a MAC journal nor a counter refuses to open unless
  `allow_no_rollback_witness` is set (development, and the partner's write-only staging store).
  `verify_integrity()` runs at startup and refuses to serve on mismatch.

### 7.7 Partner delivery
The customer generates an X25519 `RecipientKeyPair` and gives the partner the public key. The
partner writes segments whose DEK is HPKE-sealed (RFC 9180, X25519/HKDF-SHA256/AES-256-GCM) with
`kek_id = "hpke:<key id>"`, and delivers them by any channel. `import_segment` verifies the header and
every chunk's AEAD, then re-wraps the DEK under the local KEK. The partner must be independent of the
service operator (3.1).
HPKE base mode does not authenticate the sender, so the partner also signs each segment with an
Ed25519 key (`pgw maskstore partner-key create`; factory `--partner-key` /
`PGW_FACTORY_PARTNER_SIGNING_KEY_FILE`). The signature sidecar `<file>.pgwm.sig` covers SHA-256 of the
header (which carries the sealed DEK and is the AAD of every chunk); the factory places it in the
outbox before the segment. Customers set `PGW_MASKSTORE_TRUSTED_PARTNER_KEYS`; unsigned, untrusted,
copied or invalid signatures are refused, and without trusted keys every HPKE segment is refused (fail
closed; `PGW_MASKSTORE_ALLOW_UNSIGNED_PARTNER_SEGMENTS` is a development-only escape hatch).
Each partner segment is also bound to exactly one store: the signed header carries
`destination = <store_id>` (factory `--destination-store` / `PGW_FACTORY_DESTINATION_STORE_ID`, taken
from `pgw maskstore status`), and `import_segment` refuses a missing or different destination
(`PGW_MASKSTORE_ALLOW_UNBOUND_PARTNER_SEGMENTS` for development). A recipient key pair must never be
shared by two stores. `pgw maskstore import` removes each imported file and its `.sig` by default
(`--keep-source` opts out). Segments whose Freivalds checks give fewer than `min_soundness_bits`
(default 20) for any linear are refused on import and by the gateway.

### 7.8 In-memory mask ring (optional; off by default)
`PGW_GATEWAY_MASK_RING__ENABLED=true` hands masks from a co-located factory to the gateway through
locked RAM (`pgw.maskstore.ringbuf`, design `research/plans/mask-ringbuffer-design.md`). Normative:
- **Never persisted, never exported, never demoted.** Ring bytes live only in `mlock`-ed,
  `MADV_DONTDUMP`, `MADV_DONTFORK` memory (each fails closed). No path leads from a ring slab to a
  file, the store, an export or a socket payload. A crash loses ring masks; nothing can be restored, so
  the external counter (7.6) applies only to persisted segments.
- **Slab rules.** R1: only FREE slabs are granted; only WIPING becomes FREE, after the wipe (v1:
  the arena slice is zeroed; v2: see below). R2:
  `reserve()` advances a slab's in-memory cursor under the ring lock **before** any byte is copied out
  (7.5); a fully reserved slab never becomes READY again. R3: a slab with open handles is never
  recycled. R4: every read checks, under the slab's own lock, that the ring is open, the slab
  generation matches and the slab is READY/DRAINING, and copies while holding that lock; closing the
  ring marks every slab WIPING first, so a late reader raises instead of reading zeros.
- **Seeds.** A fresh `os.urandom` seed per segment, unique across the ring and the store tier
  (shared registry plus the store's seed table). The registry pins the hashes of live ring slabs and
  keeps a bounded window of recent ones (memory does not grow with uptime). Every secrets read checks
  the seed against its hash recorded at publish and refuses an all-zero seed.
- **Process.** The gateway is non-dumpable (`PR_SET_DUMPABLE=0`) and refuses unencrypted disk swap
  (`swap_policy=warn` to override) whenever the ring is enabled. v2 (`factory_process`): the factory
  child gets one sealed-size memfd per granted slab over a private socketpair (no listening socket);
  publishing adds `F_SEAL_WRITE`, which fails while the child still maps the slab writable; the child
  is non-dumpable too, and on EOF the gateway kills and reaps it before dropping its slabs.
  **v2 freed slabs are not zeroed by the gateway**: a sealed slab cannot be written, so a consumed
  slab is only unmapped and closed, and its seed, `uW` and `W r` stay in the freed physical pages
  until the kernel reuses them (the kernel always zeroes a page before giving it to a process, but
  kernel-level readers can see it until then). Only a kernel booted with `init_on_free=1` clears
  them at free time; with v2 enabled the gateway warns at start-up when it cannot detect it
  (`init_on_free_policy=refuse` to refuse instead). Unsealed v2 slabs (aborted fills) are zeroed
  through the memfd before they are dropped. The supervisor takes the same host-wide per-device
  factory lock as the embedded factory, and the child inherits only `PGW_FACTORY_*` and runtime
  variables (no gateway secrets).
- **Persistence of ids.** Ring segment ids never reach a file: the factory's `factory-state.json`
  records ring segments as counts only.
- Freivalds vectors that are all zero (a reservation closed during a step) verify nothing: the
  engine treats them as `MaskReuseError`, never as a pass.

### 7.9 Encrypted mask generation (`pgw.hemask`; wire normative)
The factory can obtain `W8·r` from the compute service on module-LWE ciphertexts instead of
computing it itself (docs/operations/encrypted-masks.md). For each group of 256 slots of one
linear, the factory encrypts `b = A·s + e + 2^(q−K)·r mod 2^q`. A is public and expanded from a
seed the customer chooses. `s` is a fresh ternary secret, and `e` is fresh CBD(21) noise. The
modulus is q = 2^56 for K ≤ 32 and 2^64 for K = 40. The service returns `y = W8·b mod 2^q`. The
factory subtracts `H·s`, using the hint H = W8·A, which it builds locally. It then rounds away the
noise at the scaling factor Δ = 2^(q−K). Every decrypted row is verified with private check
vectors before it is stored. The online Freivalds and range checks do not change.

`POST /pgw/v1/maskgen` (msgpack, same headers, auth, limits and error bodies as compute):

```
v1 request  {"v":1,"type":"maskgen","id":str,"fp":hex64,"vs":[int]?,
             "items":[{"linear":str,"cols":int,"b":bin(n·cols·q/8),"q":56|64?,"drop":0|16?}]}
v1 reply    {"v":1,"type":"maskgen_result","id":str,"vs":[int]?,
             "items":[{"linear":str,"cols":int,"y":bin(m·cols·(q−drop)/8)}]}       # bytes drop/8..q/8−1 of y + 2^(drop−1)
v2 request  {"v":2,"type":"maskgen","id":str,"fp":hex64,"vs":[int]?,
             "items":[{"linear":str,"cols":int,"bb":int,"yb":int,"b":bin(n·cols·bb/8)}]}
v2 reply    {"v":2,"type":"maskgen_result","id":str,"vs":[int]?,
             "items":[{"linear":str,"cols":int,"y":bin(m·cols·yb/8)}]}
```

- `cols` is a multiple of 256, at most 4096. Matrices are row-major `(n, cols)` and `(m, cols)`.
- **v2 semantics.** The provider computes `Y = W8·b mod 2^bb` and returns
  `floor((Y + 2^(d−1)) / 2^d) mod 2^yb` with `d = bb − yb` (no offset when `d = 0`), with
  `8 ≤ yb ≤ bb ≤ 64` and `bb ≥ 32` (the provider unpacks `b` to 64-bit words; the lower bound
  caps that expansion at 2x; a v2 item still needs about 2-3x a v1 item's provider host memory).
- **v2 bit packing.** Both operands use the same packing. The `w`-bit values are split into fields
  of 64, 32, 16 and 8 bits, from bit 0, taking the widest that fits each time. Each field is one
  contiguous little-endian `uint{width}` array over all values. The last `t = w mod 8` bits of each
  value follow as an LSB-first bit stream: eight values per `t` bytes. Examples: 35 bits = u32 +
  3-bit tail, 53 bits = u32 + u16 + 5-bit tail.
- **Negotiation.** `vs` lists the versions the sender speaks. The client sends it on requests, and
  the service on every reply. A client sends v2 only after a reply has advertised 2. It falls back
  to v1 when a v2 request fails with `protocol` or `version_mismatch` (for example after a rolling
  downgrade), and then ignores advertisements for 300 s. A service rejects a version it does not
  serve with `426 version_mismatch`, `details.versions`. `PGW_SERVICE_MASKGEN_V2=false` turns v2 off.
  v1 is always served.
- **Customer choice of `(bb, yb)`, per linear.** The reply carries `yb = K + 3` bits (35 for K32,
  43 for K40). The upload is either `bb = q` or `bb = q − 3` (53 at q = 2^56). For `bb = q − 3`,
  the customer switches its own ciphertext down with randomised rounding: `b' = floor((b + d) / 8)`,
  where `d` is uniform in [0, 8), fresh from a CSPRNG-seeded generator for each coefficient, and
  never stored. It decrypts `2^(q−yb)·reply mod 2^q` with the unchanged hint. Of the formats that
  pass the noise budget, it picks the one with the fewest bytes.
- **Noise budget (normative).** v2 is used for a linear only if
  `tail·σ·‖w‖ + [bb<q]·tail·2^(q−bb−1)·‖w‖ + [yb<bb]·2^(q−yb−1) + 255·dim < 2^(q−K−1−margin)`,
  with `tail = 24`, `margin = 1` bit, and `‖w‖` the largest int8 row norm of the linear (from the
  hint store). The terms are, in order, `W8·e`, the upload switch, the reply rounding, and the
  hint's dropped low byte. The client checks this for every linear at startup and otherwise uses
  v1 for that linear. `‖w‖` comes from the customer's own hint store, not from the provider. The
  re-check before each v2 request only asserts that the stored plan is consistent; it is not an
  independent guard. For Qwen2.5-7B at K32 (rank 16), the
  bound is about 2^21.3 for linears with hidden-size inputs and 2^21.6 for the down projection,
  against a budget of 2^22. These figures are estimates from Qwen2.5-1.5B int8 row norms
  (‖w‖ / (127·√n) ≤ 0.37). Every linear then uses `(53, 35)`. Even all-±127 rows pass, with
  `(56, 35)` or `(53, 39)` on the down projection.
- **Why the rounding leaks nothing.**
  - The reply rounding is a deterministic function of `y`, which the provider computes anyway. It
    reduces only what the customer receives.
  - The upload switch is a randomised function of `b` alone. Its randomness `d` is independent of
    `s`, `e`, `r` and the weights. A provider holding `b'` therefore learns nothing it could not
    simulate from `b`, and `b` is an MLWE sample. Lattice security is that of the 2^q ciphertext.
  - Correctness: `8·b' = b − ε`, where `ε ∈ {c−8, c}` for `c = b mod 8`, with `E[ε | b] = 0`. The
    `ε` are independent across coefficients, so by Hoeffding `|W8·ε| < 24·4·‖w‖` except with
    probability `2·e^(−288)`.
  - The scaling factor is unchanged. Decrypted masks are exact and bit-identical to v1. The
    every-row check and the Freivalds soundness argument do not change.
  - Status: the `bb = q − 3` upload replaces the plan's q = 2^53 (E5, after WP7) with a
    customer-side randomised switch. It needs explicit research-owner sign-off; until then
    `upload_switch=False` keeps `bb = q`. The factory runtime does not yet pass `params` and
    `row_norms` to `RemoteMulmod64`, so production factories still send v1.
  - Not used: an upload that redefines the mask from a public value (the rejected 25-bit variant,
    which leaks noise), and low-bit `W·r` downloads (these need a new Freivalds soundness argument).
- **Bytes per Qwen2.5-7B slot (K32).** Σn = 831,488 and Σm = 1,390,592.
  - v1: 7 B up and 5 B down, 12.77 MB in total.
  - v2 (53/35): 6.625 B up and 4.375 B down, 11.59 MB in total (−9.2%).

## 8. Model pack

### 8.1 Layout
```
<pack>/manifest.json                     pgw.modelpack.Manifest
<pack>/linears-00000-of-0000N.safetensors   linears.<id>.weight int8 (m,n) | .scale f32 (m) | .bias f32 (m)
<pack>/plain.safetensors                 embed_tokens, final_norm, lm_head (absent if tied),
                                         layers.<i>.input_norm, layers.<i>.post_attn_norm
<pack>/tokenizer/                        tokenizer.json, tokenizer_config.json, ... (chat template)
```
The service and factory need only the `linears` shards. The gateway needs only `plain` and
`tokenizer`, plus `linears` for the `plain` and `float` dev modes.

### 8.2 Architecture mapping
`pgw.modelpack.arch.HF_LINEAR_PARTS` is normative: `qkv` rows = [q_proj; k_proj; v_proj], `gate_up` rows =
[gate_proj; up_proj]. Each `LinearEntry.parts` records `row_start` and `rows` so the engine can split
outputs. Qwen2 has q/k/v biases, and Llama has none. Supported architectures are `qwen2` and `llama`
(including Llama-3 `rope_scaling`).

### 8.3 Fingerprint (normative; `pgw.core.fingerprint`)
`fingerprint = hex(SHA-256("pgw-modelpack-v1\0" || canonical_json(manifest − {fingerprint, created_at, tool, notes})))`.
The manifest includes the SHA-256 of every tensor and tokenizer file, so the fingerprint covers all
bytes. Canonical JSON uses sorted keys, `(",", ":")` separators, UTF-8 and no NaN. Display form is the first 12 hex characters.

### 8.4 Verification
`ModelPack.open(verify="fingerprint")` (default) checks manifest self-consistency. `verify="full"`
also hashes every file; it is used at install time, by the service at startup and by `pgw doctor`.
Every party compares fingerprints: gateway against service (`/pgw/v1/models`), and gateway against
each mask segment header.

## 9. Gateway and engine

### 9.1 Public API (OpenAI-compatible)
`POST /v1/chat/completions`, `POST /v1/completions` (`stream=true` gives SSE `data:` chunks ending
with `data: [DONE]`, and supports `stream_options.include_usage`), and `GET /v1/models`. `n=1` only.
Errors use the OpenAI shape `{"error":{"message","type","code","param"}}` with `code` set to an
`ErrorCode` value. `system_fingerprint` is the short model fingerprint.
`GET /v1/models/{name}` returns one model card.
Admin endpoints (admin token, or an API key with scope `admin`): `GET /admin/status`,
`GET /admin/masks`, `GET /admin/keys`, `POST /admin/keys`, `DELETE /admin/keys/{key_id}`,
`POST /admin/models/{name}/reload`, `POST /admin/models/{name}/reset` (`{"breaker": bool,
"noise": bool}`), and the console data views `GET /admin/overview`, `GET /admin/usage`,
`GET /admin/audit` (these admin endpoints also accept a console session cookie; console viewers may
only read). Ops endpoints: `/healthz`, `/readyz`, `/metrics`, `/dashboard`. Operator console:
`/console/` (contract: `docs/api/console-api.md`).

### 9.2 Engine loop
`TransformerModel.forward` runs, per layer: RMSNorm → `qkv` → split, RoPE, causal GQA attention
over the KV cache → `o` → residual → RMSNorm → `gate_up` → `silu(gate)·up` → `down` → residual.
After the last layer: final norm and LM head (float, plaintext) → `Sampler`. The plaintext
non-linear work is under 1% of FLOPs.

### 9.3 Backends
`LinearBackend.forward(LinearCall) -> float32 (T, m)` (bias included). The backends are
`PlainBackend` (exact local integers), `FloatBackend` (quality baseline) and `MaskedBackend`
(production). **Requirement:** `MaskedBackend` logits are bit-identical to `PlainBackend`; this is
an integration test.

### 9.4 Mask reservation
Each request gets one `MaskReservation` of `prompt_len + max_tokens` slots, passed to the backend as
`LinearCall.session`. The backend takes `T` rows per call for that linear and splits calls at
segment-extent boundaries. If the store has too few slots, the request fails fast with
`mask_exhausted` before any compute.

### 9.5 Failure handling
- Service unavailable: retry with backoff (wire client), then fail the request (`unavailable`). KV
  state is per session, so nothing else is affected.
- Integrity failure: fail the request, increment `pgw_gateway_integrity_failures_total`, open the
  circuit breaker for that service endpoint (configurable cool-down; admin reset), and alert.
- Noise-monitor alarm in fail-closed mode: block all masked traffic until an admin resets it.

## 10. Compute service
- Stateless: no KV cache, no sessions, no request persistence, and no payload bytes in logs or traces.
- `ModelRegistry` loads packs (linears only, `verify="full"`), keyed by fingerprint.
  `ComputeKernel` wraps `PreparedWeight` per linear on the device. `BatchScheduler` does cross-request batching.
- CUDA execution (`pgw.service.gpu_exec`): one CUDA graph per (linear, K, row bucket) replays the
  offset-byte GEMM for calls with `rows·K/8 ≤ p`, `p ∈ cuda_graph_buckets` (default 32/64/128; larger calls
  run eagerly); captured at warm-up or first use and checked against the eager result once (a failed
  capture falls back to eager, logged once). Pinned staging, non-blocking copies, and int32 results for
  K ≤ 32 on the request path. Replays share per-device static buffers and a graph memory pool, so calls
  are serialized per device by a lock (the batcher already runs one GPU batch per device). Bit-exact.
  `PGW_SERVICE_CUDA_GRAPHS=false` disables graphs.
- Fused ring GEMV (`pgw.core.triton_gemv`): CUDA calls with K ≤ 32 and at most 16 rows run as one Triton
  kernel that streams the int8 weight once and computes `Σ u32(x)·u32(sext(W8)) mod 2^32` with unsigned
  32-bit wrapping arithmetic (no digits, no floats; any reduction order gives the same bits), then masks
  to `2^K`. Only the low 32 bits of `x̂` are copied to the device. Enabled per device only if triton imports
  and a startup self-check against the digit-GEMM path passes (random and extreme operands, rows 1–16,
  K ∈ {16, 24, 32}); the first call per (linear, K, rows) is cross-checked too, and any mismatch switches the
  device back to the digit-GEMM path. `PGW_SERVICE_TRITON_GEMV=false` disables it.
- Admission control: `max_inflight_requests` (beyond it, `unavailable` with `Retry-After`) and per-tenant quotas.
- Graceful drain on SIGTERM: `/readyz` returns 503, in-flight work finishes, then exit.
- Scaling: horizontally behind an L7 LB. Any replica can serve any request because the service is stateless.

## 11. Observability
### 11.1 Metrics
The names in `pgw.observability.metrics.METRIC_NAMES` are the contract. Components emit only listed
metrics. Labels never contain prompt data. Key SLO metrics: `pgw_gateway_ttft_seconds`,
`pgw_gateway_tpot_seconds`, `pgw_service_request_seconds`, `pgw_masks_available_slots`, `pgw_gateway_integrity_failures_total`.
### 11.2 Logging rules (normative)
Never log activations, payload bytes, masks, seeds, keys, API keys, prompt or completion text.
Log ids, shapes, sizes, timings and error codes. `pgw.core.log.configure_logging(json_lines=True)` in production.
### 11.3 Outbound-noise monitor
The gateway samples outbound `x_hat` and runs top- and low-byte chi-square tests (255 dof), a
top-bit balance test, correlation with `xq`, and duplicate-row detection. Thresholds are set for a
false-alarm rate below 1e-9 per window. An alarm sets `pgw_gateway_noise_alarm`, and in fail-closed
mode blocks sends.
### 11.4 Dashboard
A self-contained HTML page at `/dashboard` (no external assets) shows throughput, latency, mask
inventory, integrity and noise status.

## 12. CLI and configuration
`pgw version | doctor [--role] [--pack] [--ring] [--maskstore --key-file] [--service-url ...] [--strict] [--json] |
inspect PACK | pack SRC OUT | factory generate|run|status|benchmark | maskstore init|status|import|gc|rotate-key |
maskstore recipient-key create|show | maskstore partner-key create|show | service serve |
service token create|hash | gateway serve | gateway keys create|list|revoke | bench ring|factory|service |
config show ROLE | config render TEMPLATE OUT`. `python -m pgw.cli` is equivalent to `pgw`.
Configuration uses pydantic-settings classes with env prefixes `PGW_` (common: `PGW_LOG_LEVEL`,
`PGW_LOG_JSON`, `PGW_RING`), `PGW_SERVICE_`, `PGW_GATEWAY_`, `PGW_FACTORY_` and `PGW_MASKSTORE_`.
Nested keys use `__` (for example `PGW_SERVICE_TLS__CERT_FILE`). Secrets are `SecretStr` and can
come from files or env; they are never written to the command line. Exit codes: 0 ok, 1 `PGWError`, 3 not
implemented, 130 interrupted.

## 13. Deployment topologies
**A. On-prem gateway.** Gateway, mask store and (optionally) factory run in the customer data
center (container or VM; Docker image `pgw-gateway`). The service runs in our cloud. Traffic is mTLS
over the internet or a dedicated link. Masks come from the customer's own GPU factory or a partner
(7.7). Expect RTT of 10–30 ms, which gives 1.3–4 s per token for 8B at 128 rounds unless sessions are batched.

**B. Gateway in the customer's cloud account.** We ship Helm charts (Terraform modules planned, not in v0.1), and the customer deploys
it into their VPC in the same region as our service, over PrivateLink or PSC (RTT about 1 ms, so the
round-trip floor is about 0.13 s per token for 8B). The mask store uses an encrypted volume with
`kms` key provider and the customer's KMS key. A customer-account GPU instance can run the factory.

**C. Partner mask factory.** An independent partner (contractually non-colluding) runs factory GPUs
and delivers HPKE-wrapped segments (7.7) to customers' stores. This is the cheapest option for
customers without GPUs (T4 spot: about $0.36 per million 8B tokens at K48).

**Placement for latency (B).** A token's 112-128 round trips run one after another, so the gateway
should call a service replica in its own Availability Zone: the zonal endpoint DNS name first, the
regional name as fallback, cross-zone load balancing off and one service replica per AZ
(`deploy/marketplace/aws/README.md`). Same-AZ RTT adds an estimated 0.01-0.035 s per 7B token,
cross-AZ 0.05-0.11 s. The gateway pod runs with Guaranteed QoS (integer CPUs, requests equal to
limits) and, where the node uses `cpuManagerPolicy: static`, on exclusive cores; the factory has a
hard CPU limit (`deploy/helm/README.md`).

**Mask supply and alerts.** A new store is refilled with a ramp of small segments (one segment at a
time while below low water, sized `min(segment_slots, max(first_segment_slots, next_pow2(2·available)))`).
Encrypted mask generation (`pgw.hemask`, docs/operations/encrypted-masks.md) is link-bound: over slow links (for example the public tunnel of the
Modal demo) it cannot feed a decoding stream, so production supply runs over a private link or from a
local factory. `pgw doctor` times one mask chunk over the real path and checks certificate expiry. The
charts ship PrometheusRules (also `deploy/monitoring/pgw-alerts.yml`) for mask runway, low water,
factory stalls, burn ratio, model readiness, and paging alerts for integrity failures and the noise
alarm.

In every topology the service operator never receives seeds, masks or `u`.

Shipped artefacts live in `deploy/`: Docker images and entrypoint (`deploy/docker`), a docker-compose
demo of the whole pipeline on one machine (`deploy/compose`), Helm charts for gateway, service and
factory (`deploy/helm`), systemd units (`deploy/systemd`), a VMware appliance (`deploy/vmware`) and
marketplace listings (`deploy/marketplace`). `scripts/e2e_demo.sh` runs the pipeline by hand without
containers.

## 14. Performance targets (v1 release gates)
| item | target | measured basis |
|---|---|---|
| exactness | masked == plain logits bit-identical (all rings) | Qwen2.5-0.5B: diff 0.0 |
| service, 8B, T4, batch 1 | ≤ 0.5 s/token | 0.45 s (K48) |
| service, 8B, T4, batch 32 | ≤ 0.015 s/token amortized | 0.014 s |
| service, 8B, A100/H100 batch 32 | ≤ 2 ms/token amortized | estimate (int8 TOPS ratio) |
| gateway online crypto, 8B | ≤ 10 ms/token CPU (regen r 0.85M elems + unmask) | ChaCha20 ~13 ms per 1M elems in Python (vectorize/thread) |
| gateway plaintext work, 8B | ≤ 20 ms/token CPU at 4k context | — |
| factory, 8B, T4, K48 | ≥ 500k slots/h | 550k slots/h |
| mask storage | 6.9 MB/slot 8B K40, 1.5 MB/slot 0.5B K40 | analytic |
| wire per token 8B K40 | 4.3 MB up, 6.9 MB down; 128 round trips | analytic |
| service availability | 99.9% monthly | — |
| unit + integration test suite on 4-CPU box | < 5 min | — |

## 15. Engineering rules
- Python ≥3.12, src layout, full type hints, docstrings, `logging` (no prints in library code), and
  `PGWError` subclasses for all expected failures.
- Tests live in `tests/<subpackage>/`, run on CPU with tiny random models (`tests/conftest.py`), and
  finish in minutes. GPU tests are marked `@pytest.mark.gpu` and network tests `@pytest.mark.network`.
- Interface changes: append to `INTERFACE_NOTES.md` (who, what, why). The lead architect updates SPEC.
- `tests/e2e` runs the real CLI and real processes on localhost (pack, factory, encrypted store, service,
  gateway) with Qwen2.5-0.5B-Instruct (tiny random model offline) and checks exactness against
  QuantPlain, Freivalds, tamper detection, single use and that the service never sees plaintext.
- Cross-language vectors: `tests/vectors/core_v1.json` (generated from `pgw.core`) are checked by
  `pytest tests/vectors` and by `cargo test` in `rust/` next to the KATs of 7.3.
- No real secrets, personal data or customer data in code, tests or fixtures.

## 16. Open items / roadmap
- Fused GPU kernel (digit split, K/8 GEMMs and accumulation in one launch). A CPU int8 path via fbgemm.
- SmoothQuant-style activation smoothing so that K32A8 becomes lossless (25% less traffic, storage and compute).
- Mask compression: no gain is possible because masks are uniform. The way to reduce volume is to
  store a seed and recompute `W r` on the gateway GPU when one exists ("self-factory" mode).
- Speculative decoding and prefix caching on the gateway (fewer rounds per accepted token).
- Formal write-up of the security argument and an external audit.

## 17. Amendments v1.1 (integration review; folded from INTERFACE_NOTES.md)

All additive and backwards compatible unless noted.

**Core (5, 7.3).** `mask_activations` rejects `|xq| > qmax` (`ring_overflow`); `quantize_weight_rows`
rejects non-finite weights; `freivalds_check`/`verify` refuse `c = 0` (`config`). The mask store
already refuses to hand out check vectors of a `checks = 0` segment. These match the Rust core.

**Engine (9.2–9.4).** `Generator(model, tokenizer, *, max_batch=32, session_factory=None, batching=True)`;
`generate/stream/agenerate/astream(..., session=, own_session=False)`; with `own_session=True` the
engine closes (burns) the reservation at a safe point, also when request validation fails.
`generate_many`, `aclose`; `TokenEvent.top_logprobs`, `GenerationResult.logprobs`;
`SamplingParams.ignore_eos`, `include_stop_str_in_output`, `stop` may be a string.
`ContextOverflowError` (`bad_shape`). `PlainBackend` is exported as `QuantPlainBackend`.
`MaskedBackend(..., on_integrity_failure=fn, max_rows_per_item=8192)` checks the reservation's
fingerprint before taking rows, validates and quantizes every call of a batch before taking any mask
row, and wipes `W r` on every failure path. `LocalComputeClient`, `LocalMaskSource`, `InMemoryPack`
are in-process stand-ins for development and tests.

**Mask store (7.1–7.7).** `reserve(fp, ring, slots, *, session_id="")`, `inventory`, `status()` (the
`/admin/masks` payload), `metrics()`, `add_low_water_listener`, `export_segment`, `cleanup_partials`,
`rotate_kek` (offline, resumable). Reservation owners are recorded as `hostname|pid-namespace` and
live reservations are tracked process-wide. Rollback witness and partner signatures: 7.6, 7.7.

**Factory (7.4).** Self-check before each chunk is written: sampled rows verify
`u·(W8 r) == (u W8)·r (mod 2^K)` (`PGW_FACTORY_SELF_CHECK_ROWS`, default 8, -1 all, 0 off); a mismatch
aborts the segment (`IntegrityError`). On CUDA the `r` stream (7.3, key derived on the CPU) is expanded
by a GPU ChaCha20 kernel straight into the GEMM's digit planes (`PGW_FACTORY_GPU_PRNG`, default on after a
self-test against the CPU expander; same bytes, the self-check uses CPU-derived rows). `cpu_int8` backend
behind an exactness probe. Recipient keys
are validated (small-order X25519 points refused). `ExportSink.recover()` at every start publishes
segments stranded in staging and signs unsigned outbox segments. `Throttle.max_slots_per_hour`
accumulates no idle credit. Partner signing: 7.7.

**Service (10).** `create_app(settings, *, registry, authenticator, metrics)`, `serve()`,
`create_metrics_app(state)` serving `/metrics`, `/healthz` and `/readyz` on the internal metrics
listener (for probes when the main port requires client certificates). Settings: `devices`,
`placement`, `verify_packs`, backend `cpu_int8`, metrics listener, admission limits and tenant
quotas (`PGW_SERVICE_*`). Unknown fingerprint `fingerprint_mismatch`; ring not served `config`;
overload/drain `unavailable` + `Retry-After`; tenant quota `rate_limited`.

**Gateway (9.1, 9.5).** Extra endpoints `GET /v1/models/{name}`, `GET /admin/keys`,
`POST /admin/models/{name}/reset` (`{"breaker": bool, "noise": bool}`). Settings `allow_dev_modes`
(required for `plain`/`float`, `freivalds_rate < 1`, anonymous access), `require_service_tls` (plain
http only to loopback; with `false` a non-loopback http URL is allowed for service-mesh mTLS and
logged), `egress_allowlist`, `integrity_cooldown_s`, `audit_log_file`, quotas, limits, noise-monitor
options. `admin_token` needs at least 16 characters. Per model: `fingerprint` pin, queue limits,
service timeouts/retries/deadline, `device`, and `service_token_file` (exclusive with
`service_token`). The integrity breaker is checked before every compute round trip, so running
sessions stop too. Integrity failures and noise-monitor alarms are written to the audit log at once
(`safety.integrity_failure`, `safety.noise_alarm`; no content).

**Observability (11).** `PGWMetrics(registry=None, *, enabled, prefixes)`; `OutboundNoiseMonitor`
options (`alpha`, `always_sample_first`, `duplicate_*`, `feature_window_rows`, `metrics`, `model`,
`seed`), `add_alarm_listener`, `flush`, `NoiseReport.to_dict()`; the ring range check runs on every
call. `dashboard_html`, `mount_dashboard` (CSP, no-store, frame deny).

**Deferred.** AWS Marketplace metering hook (`PGW_GATEWAY_MARKETPLACE__*`); adopting `cpu_int8` as
the default of `pgw.core.ringmm.select_backend` (it stays the exact reference; service and factory
already select `cpu_int8` after their probes); copying `pgw/.github/workflows/ci.yml` to the
repository root (outside the `pgw/` tree).
