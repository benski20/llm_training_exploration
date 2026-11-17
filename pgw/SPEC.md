# PGW (Private Gateway): Specification v1

Status: **normative for protocol version 1**. Owner: lead architect. Changes to anything marked
*normative* require an entry in `INTERFACE_NOTES.md` and a SPEC update in the same change.

PGW runs private LLM inference on public open-weight models (Qwen2.5, Llama 3). The heavy linear
algebra runs on our GPUs, but our compute service only ever sees uniformly random ring elements. The
customer keeps the plaintext and the small non-linear work, and holds the one-time masks.

---

## 1. Goals and non-goals

**Goals**
- Information-theoretic input privacy of every activation against the compute service.
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
- **Privacy (information-theoretic).** Each `r` is uniform over `Z_{2^K}^n`, secret, and used
  exactly once, so `x_hat` is uniform and independent of `x`. This holds against unbounded adversaries
  as long as (a) masks never reach the service, (b) no mask is reused, and (c) factory and service do
  not collude. The activation scales `sx` never leave the gateway.
- **Integrity.** Every result passes a range check and `c` Freivalds checks (default `c = 2`,
  checking every call). False-accept probability per call is at most `2^(−c·h)` with
  `h = K − ⌈log2(4·qmax·127·n)⌉` (`pgw.core.soundness_bits`). For K40A12 and n = 4096, h = 8, so
  2⁻¹⁶ per call with c = 2. **Any** failure aborts the request, raises `integrity_check_failed`, and
  trips a per-service circuit breaker (operator alert). An adaptive attacker who probes the check
  is caught with constant probability on each probe, and is then cut off.
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
timing, and the tenant. Optional mitigations: pad prompt rows to buckets (`pad_rows_to`), batch
several sessions per request, and send constant-rate dummy rounds. Each costs masks and compute.

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
`maskstore`. `factory` depends on `core`, `modelpack` and `maskstore`. `service` depends on `core`,
`modelpack`, `wire` and `observability`. `gateway` depends on everything except `service` and
`factory`. `cli` may import anything, lazily. **`service` must never import `maskstore` or
`factory`.** CI enforces this with an import-graph test.

## 5. Arithmetic and quantization (normative)

- **Ring** `Z_{2^K}`, `K ∈ {32, 40, 48}`, multiple of 8 in [16, 56]. Presets: `K40A12`
  (**default**), `K48A20` (max quality), `K32A8` (cheapest, measured lossy: ppl 11.09 vs 10.93 for
  Qwen2.5-0.5B). Canonical values are `int64` in `[0, 2^K)`. Centered values are in `[−2^(K−1), 2^(K−1))`.
- **Weights:** int8, symmetric per output row: `sw[j] = max_i |W[j,i]| / 127` (1.0 for zero rows),
  `W8 = clamp(round(W/sw), −127, 127)`. The value −128 is never produced, but the kernels handle it.
- **Activations:** signed A-bit, per token: `qmax = 2^(A−1) − 1`, `sx[t] = max_i |x[t,i]| / qmax`
  (1.0 for zero rows), `xq = clamp(round(x/sx), −qmax, qmax)`. Non-finite input raises an error.
- **Output:** `y = yq · sx[t] · sw[j] + b[j]` in float64, then cast. Biases stay float and are applied by the gateway.
- **Headroom:** `qmax·127·n < 2^(K−1)`, checked by `RingParams.check_in_features`.
- **Ring matmul** (`pgw.core.ringmm`, exact mod `2^K` on every backend):
  - `cuda_int8`: K/8 `torch._int_mm` GEMMs on balanced base-256 digits `d ∈ [−128,127]`. Rows are
    padded to at least 32 and to a multiple of 8; `n` and `m` are padded to multiples of 8. Accumulation
    is int32, and the inner dimension is chunked at ≤131064.
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
- Server: hypercorn (h2 and h2c). Client: `httpx` with `http2=True` and a connection pool.

### 6.2 Endpoints (compute service)
| method | path | body | response |
|---|---|---|---|
| POST | `/pgw/v1/compute` | `ComputeRequest` | `ComputeResponse` or `ErrorResponse` |
| GET | `/pgw/v1/models` | – | `ModelsResponse` |
| GET | `/healthz` | – | `200 ok` (process alive) |
| GET | `/readyz` | – | `200` when packs are loaded and the device is OK, else `503` |
| GET | `/metrics` | – | Prometheus text (separate port or IP allowlist recommended) |

Headers: `Content-Type: application/vnd.pgw+msgpack`, `PGW-Protocol: 1`,
`Authorization: Bearer <tenant token>` (alongside or instead of mTLS), and optionally
`PGW-Request-Id`. Compression: none, because uniform data does not compress.

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
  `(fp, ring, linear)` within `batch_window_ms` (default 2 ms), up to `max_batch_rows`. Mixing
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
  mirrored to the provider's hardware monotonic counter where available (HSM, TPM, KMS-backed
  counter). The file and env providers fall back to an HMAC-chained journal and log a warning.
  `verify_integrity()` runs at startup and refuses to serve on mismatch.

### 7.7 Partner delivery
The customer generates an X25519 `RecipientKeyPair` and gives the partner the public key. The
partner writes segments whose DEK is HPKE-sealed (RFC 9180, X25519/HKDF-SHA256/AES-256-GCM) with
`kek_id = "hpke:<key id>"`, and delivers them by any channel. `import_segment` verifies the header and
every chunk's AEAD, then re-wraps the DEK under the local KEK. The partner must be independent of the
service operator (3.1).

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
Admin endpoints (admin token): `GET /admin/status`, `GET /admin/masks`, `POST/DELETE /admin/keys`,
`POST /admin/models/{name}/reload`. Ops endpoints: `/healthz`, `/readyz`, `/metrics`, `/dashboard`.

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
`pgw version | doctor [--pack] | pack SRC OUT | factory generate|run | maskstore init|status|import|gc |
service serve | gateway serve | gateway keys create|list|revoke`.
Configuration uses pydantic-settings classes with env prefixes `PGW_` (common: `PGW_LOG_LEVEL`,
`PGW_LOG_JSON`, `PGW_RING`), `PGW_SERVICE_`, `PGW_GATEWAY_`, `PGW_FACTORY_` and `PGW_MASKSTORE_`.
Nested keys use `__` (for example `PGW_SERVICE_TLS__CERT_FILE`). Secrets are `SecretStr` and can
come from files or env; they are never written to the command line. Exit codes: 0 ok, 1 `PGWError`, 3 not implemented.

## 13. Deployment topologies
**A. On-prem gateway.** Gateway, mask store and (optionally) factory run in the customer data
center (container or VM; Docker image `pgw-gateway`). The service runs in our cloud. Traffic is mTLS
over the internet or a dedicated link. Masks come from the customer's own GPU factory or a partner
(7.7). Expect RTT of 10–30 ms, which gives 1.3–4 s per token for 8B at 128 rounds unless sessions are batched.

**B. Gateway in the customer's cloud account.** We ship Terraform/Helm, and the customer deploys
it into their VPC in the same region as our service, over PrivateLink or PSC (RTT about 1 ms, so the
round-trip floor is about 0.13 s per token for 8B). The mask store uses an encrypted volume with
`kms` key provider and the customer's KMS key. A customer-account GPU instance can run the factory.

**C. Partner mask factory.** An independent partner (contractually non-colluding) runs factory GPUs
and delivers HPKE-wrapped segments (7.7) to customers' stores. This is the cheapest option for
customers without GPUs (T4 spot: about $0.36 per million 8B tokens at K48).

In every topology the service operator never receives seeds, masks or `u`.

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
- No real secrets, personal data or customer data in code, tests or fixtures.

## 16. Open items / roadmap
- Fused GPU kernel (digit split, K/8 GEMMs and accumulation in one launch). A CPU int8 path via fbgemm.
- SmoothQuant-style activation smoothing so that K32A8 becomes lossless (25% less traffic, storage and compute).
- Mask compression: no gain is possible because masks are uniform. The way to reduce volume is to
  store a seed and recompute `W r` on the gateway GPU when one exists ("self-factory" mode).
- Speculative decoding and prefix caching on the gateway (fewer rounds per accepted token).
- Formal write-up of the security argument and an external audit.
