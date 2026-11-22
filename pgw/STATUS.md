# PGW release status: v0.1.0

Date: 2026-11-17. This is the release manager's view of what is built, how it was verified, what is
missing, and what has to happen before a real customer pilot. The design is in [SPEC.md](SPEC.md).
The change log between teams is in [INTERFACE_NOTES.md](INTERFACE_NOTES.md). GPU numbers are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

**Verdict.** The protocol is implemented end to end and is exact. It has been verified on CPU here
and on a Kaggle T4 GPU with Qwen2.5-0.5B and Qwen2.5-7B. Security and reliability reviews are folded
in. The product is ready for an **internal or friendly-customer technical pilot in topology B**
(gateway in the customer cloud, same region as the service). It is **not** ready for a production
SLA pilot. Three reasons:

1. Per-token latency is about 3x off target. The cause is round trips and gateway CPU, not the GPU.
2. The HSM, KMS and TPM integrations have never run against real hardware or cloud services.
3. There has been no external security audit.

## Update 2026-11-22: strict speculation at a public mask charge (L3, off by default)

`spec_mask_rows` C (+ `spec_mask_credit` B), research and numbers in
[research/plans/speculation-no-compromise.md](research/plans/speculation-no-compromise.md): strict speculation's
filler rows and dummy passes are padding rows (no mask) and every decode pass charges the public schedule
`prompt + j + min(B, floor((C-1) j))`, so masks per generated token drop from R to about C (or 1 + B/L) while the
provider's view (rows, items, pass count, mask demand, per-call gateway work) stays independent of the drafts.
Exact: bit-identical to the plaintext reference in the new tests (greedy, seeded sampling, penalties, stop strings,
R 2/4, C 2/R, fractional C, credit caps, mask extents ending mid-pass).

- Four distinguishers of the existing padding-row code were found and closed before enabling it: item splits at
  mask-extent boundaries (now one item per padded call, and real rows kept within one extent), unverified dummy
  items (an active provider could learn which passes were dummies; now Freivalds-checked), a gateway turnaround
  that depended on the real/padding split (measured 466 / 529 / 654 us per call for dummy / real / mixed at R = 4;
  now one fixed-structure path, within noise), and window extension, ahead reservation and part retirement driven by
  mask use (`WindowedReservation(public=True)`: extensions only by `charge_to` on the schedule).
- Simulated on the Kaggle run 28 exact answers (`benchmarks/spec_charge_sim.py`): 1.5B target with the 0.5B draft at
  R = 4, C = 4, B = 160 keeps today's 2.84 tokens per pass at 2.50 masks per token instead of 3.96; the 0.5B with
  prompt lookup at C = 1.5 keeps its 1.08 tokens per pass at 1.48 instead of 3.96. Not measured on a GPU or Modal
  (commands in the plan, section 7; Modal: `launch.py --spec-mask C[/B]`).
- Pre-existing strict-mode timing residuals are documented, not fixed (attention cost grows with context during the
  answer and is flat during the dummy passes; the drafter's catch-up step).

## Update 2026-11-20: chat-speed phase A wired into the gateway

The phase-A packages now run on the gateway's production path (all per-model settings, off by default;
SPEC 9.4a). `local_linears`, `lane`, `spec_rows` + `spec_strict_passes` left `PENDING_SPEED_SETTINGS`;
`float_ops`, `head_mode`, `schedule` and `prefix_cache` stay refused (phase B). New per-model settings:
`local_rtt_ms`, `lane_uds_dir`, `lane_pool_wait_s`, `float_threads`. `/admin/status` reports them under
`speed` (lane state and fallbacks, local-linear decision, start-up RTT). Gateway-level tests show each
setting bit-identical to the plaintext reference on the tiny model in masked mode, including the lane
against the real service UDS lane (`tests/gateway/test_speed_wiring.py`).

- The lane client (WP-A3) and the service lane (WP-A1) were built against different framings
  (`/pgw/v1/lane_open` + `PGWT` vs `/pgw/v1/lane/open` + `PGWL`). `pgw.wire.service_lane` bridges them
  (Python channel; the native `TripChannel` speaks only the A3 framing, so the lane runs at the Python
  channel's cost until one side is changed).
- Strict speculation pads with masked filler rows (the model does not use the backend's mask-free padding
  rows yet), so it costs `spec_rows` times the mask rows per token.
- Modal demo: the customer's `models.json` gets `local_linears = "auto"` and `float_threads = 1` via
  `PGW_GATEWAY_SETUP_MODEL_OPTIONS`; no lane (remote provider) and no strict speculation (no draft model,
  little prompt overlap). Not run on Modal or Kaggle yet.
- `pgw doctor --maskstore-read-mb` sets the `maskstore.read_rate` probe size.

## Update 2026-11-20 (measurement and telemetry for run 26)

Run 26 (after the phase-1 merge) will be measured with corrected accounting; no new speed numbers yet.

- **Decode-only accounting** in `benchmarks/gpu_validate.py`. The run-25 "round trips 0.114 s/token"
  divided whole-request remote time (prefill included) by prompt + generated tokens, so it understated
  decode. The streaming phase now reports `decode_remote_s_per_token`, `decode_local_s_per_token`,
  `decode_remote_ms_per_trip` and `decode_round_trips_per_token` (from the backend's decode counters when
  the gateway has them, else from the stats delta after the first token), and how much of the measured
  s/token they account for. The old fields are kept as `*_incl_prefill`.
- **Per-trip split**: `/admin/status` now has `client_timing` (per rows-per-request class: client RTT,
  service queue and compute, and the rest = network, framing, Python), reported as `decode_trip_split`.
  The compute client wrapper now also times (and breaker-gates) the blocking `compute_sync` path that
  small decode calls use; before, those trips were untimed.
- **Metrics**: `pgw_gateway_round_trip_seconds` uses sub-ms buckets (0.25-25 ms plus a tail).
  `pgw_gateway_network_seconds`, `pgw_gateway_service_seconds` and `pgw_gateway_client_cert_expiry_seconds`
  are implemented but emit only once they are added to the metric contract
  (`pgw.observability.metrics.METRIC_NAMES`) and the gateway API metrics table.
- **Benches**: `benchmarks/net_floor_bench.py` (TCP ping-pong floor; loopback in this 4-vCPU container:
  p50 39 us at 64 B, 75 us at 80 KB, i.e. under 0.01 s/token over 112 trips). `profile_gateway.py`
  warms every layer. Kaggle jobs `pgw-run26-7b` (now `pgw-run27-7b`, 4/8/16 streams, net floor, profile) and
  `pgw-spec-sweep-7b` (k = 1, 2, 3 on cpu and cuda:1, `--spec-max-batch 1`) replace `pgw-spec-7b`
  (`--spec-tokens 8`).

## Update 2026-11-20 (two-party deployment on Modal)

`deploy/modal/` runs a real provider/customer pair as two Modal apps (one command: `launch.py`). The
customer is onboarded through the setup wizard over a public URL: enrollment with CA pin and an
issued client certificate, the model downloaded from Hugging Face and packed by the gateway itself
(fingerprint matches the provider's), encrypted mask generation over the network, finish without a
restart. A masked greedy chat was bit-identical (tokens and logprobs) to the plaintext reference.
Found on the way: a new store waited for a full 4096-token mask segment before the gateway could
serve (over an hour at the 1.5 MB/s the tunnel carries). The factory now makes a small first segment
for an empty store (`PGW_FACTORY_FIRST_SEGMENT_SLOTS`; the wizard sets 256), so the first masks arrive
after about 2 minutes. The service authenticates compute calls with the tenant token; client
certificates are verified and tenant-checked when presented but cannot be required on the port that
also serves enrollment.

## Update 2026-11-20 (deterministic multi-row LM head)

The float32 LM head (`norm(h) @ lm_head^T`) is now `pgw.engine.lm_head.lm_head_logits`: a fixed
summation order (16 partial sums over the hidden dim, fixed tree, no FMA), so a row's logits do not
depend on how many rows share the pass, the thread count or the CPU. Native kernel `pgw_fast.lm_head`
(SSE2/AVX, multi-threaded, streams the head once for all rows); pure-torch fallback with the same
order (bit-identical, tested; slower: about 0.45 s per row at 7B). Both the gateway's masked model and
the plaintext reference use it, so they still agree bit for bit; **the reference logits themselves
moved by ~1e-6 vs torch's old GEMV** (different but fixed order). Because batching is now exact,
`batch_lm_head` defaults on with the native kernel: one pass per step for every sequence's last row and
all rows of speculative verify passes (`RowLogits`); off (lazy per row) with the fallback.
Measured here (4 vCPUs, `benchmarks/lm_head_bench.py`, 152064 x 3584 fp32):

| rows per step | torch per row (old) | torch one GEMM (inexact) | deterministic, one pass |
|---|---|---|---|
| 1 | 0.050 s | 0.051 s | 0.043-0.054 s |
| 4 | 0.18-0.19 s | 0.09-0.11 s | 0.07-0.08 s |
| 8 | 0.39-0.42 s | 0.16-0.18 s | 0.10-0.11 s |
| 16 | 0.75 s | 0.11-0.12 s | 0.17 s |

## Update 2026-11-20 (gateway CPU per call, multi-session throughput)

Measured here (4 shared vCPUs, A/B interleaved, medians of 3; `benchmarks/profile_gateway.py --preset 7b
--ring K32A12C27`, `benchmarks/e2e_local.py`), outputs bit-identical to the plaintext reference:

| metric | before | after |
|---|---|---|
| gateway ms per decode call, 7B, 1 session (in-process CPU service) | 0.95 | 0.61 |
| same, 4 sessions per round trip | 2.71 | 2.01 |
| same, `--null-service` (no CPU matmul evicting caches; closer to a GPU service), 1 / 4 sessions | 0.57 / 1.86 | 0.30 / 1.01 |
| e2e small model, null service, 1 / 4 streams (tokens/s) | 56 / 93 | 94 / 156 |

What changed: `pgw_fast.finish_group` verifies and unmasks the chunks of all sessions whose masks come
from the same segment in one pass over the shared Freivalds vectors (each keeps its own `W r`); check
vectors and `W r` travel as 32-bit words for K=32; `W r` is zeroed inside the kernel; the request payload
replaces the int64 `x_hat`; per-(extent, linear) caches for stream keys, compact check vectors and the
current extent; `MaskReservation.take_checked` (boundary + checks + take under one lock);
`WindowedReservation` no longer rescans old parts; small decode round trips run on the event loop
(`inline_max_rows`, default 8; never for window extensions, a reservation's first pass or the LM head).
LM head (`benchmarks/lm_head_bench.py`, 7B float32): 47 ms per sequence, so 0.19 s per step with 4
sequences; the opt-in `batch_lm_head` (default off, changes logprob bits, not quality) takes 0.11 s.

## Update 2026-11-20: mask factory speed (GPU ChaCha20, native writer, encrypted factory)

Merged, each reviewed (outputs byte-identical, privacy unchanged):
- **GPU ChaCha20 + digit split** (`PGW_FACTORY_GPU_PRNG`, on for CUDA factories after a byte-identity self-test
  against the CPU path; the factory falls back to the CPU otherwise). Runs only on the customer's own factory GPU.
- **Factory quick wins:** native ChaCha on the CPU path, byte-gather digit split, in-place combine, auto chunk size,
  pinned copy-back on CUDA, per-stage stats.
- **Native segment writer** (Rust AES-GCM, same at-rest format): write stage about 30 -> 5 ms per 7B slot (local).
- **Encrypted factory, customer side:** native noise/secret sampling, more groups per pass, requests in flight.
- **Encrypted factory, provider side:** faster maskgen (about 2.1-2.6x on CPU locally), cached row sums, no global lock.

Kaggle T4, Qwen2.5-7B K32A12C27, all 28 layers, generation only (no store writes), 1024 slots:
**122 -> 356 slots/s (about 438k -> 1.28M slots/h) with the GPU PRNG**; rng time 8.7 s -> 0.07 s; GPU tests incl.
byte-identity pass on the T4 (`benchmarks/results/kaggle/factory_gpu_prng_v2`). The first attempt
(`factory_gpu_prng`) correctly fell back to the CPU path because of a cuda vs cuda:0 device comparison, since fixed.

## Update 2026-11-20: phase 2 of the speed plan (Kaggle run 27)

Four packages, each reviewed (no blocking issues): lean wire lane (service frame lane, client
`compute_fast`, zero-copy replies), native `prepare_many` plus single-use mask prefetch on a background
thread, a faster exact LM head (register-blocked; AVX-512 where available), sampler and RoPE, and
optional two-cohort ping-pong (`engine_pingpong`, off by default). Run 27, Qwen2.5-7B on one T4, same steps
as run 26, all outputs bit-identical (text and logprobs):

| metric | run 26 | run 27 |
|---|---|---|
| decode, 1 request | 0.339 s/token | 0.349 s/token (unchanged within noise) |
| time to first token, 40-token prompt | 2.37 s | 2.35 s |
| generated tokens/s, 4 / 8 / 16 streams | 4.16 / 4.70 / 4.61 | **4.80 / 5.72 / 5.74** (+15 / +22 / +25%) |
| gateway CPU per generated token, 4 / 8 / 16 streams | 0.184 / 0.177 / 0.179 s | **0.145 / 0.131 / 0.132 s** |
| decode round trip (remote), 1 request | 1.47 ms | 1.46 ms |

The single-request path is bound by its 112 sequential round trips (0.163 s/token) and did not move;
the gains are in gateway CPU, which is what limits aggregate throughput. Gateway plaintext work per token
on the Kaggle CPU (`benchmarks/profile_token.py`, 28 layers, stub service; this host has AVX2, not
AVX-512): 97.4 -> 82.5 ms at 1 row, 249 -> 227 ms at 4 rows, 967 -> 901 ms at 16 rows; sampling 1.6 ->
0.65 ms. Spinning OpenMP workers (`OMP_WAIT_POLICY=ACTIVE`) cut the layer work 29 -> 20 ms at 1 row but
slowed the LM head; a candidate for scoping to the layer ops only.

Ping-pong A/B (`pgw-pingpong-7b`, 32 tokens, same service and store): 4 streams 4.25 vs 4.29 tok/s
(control, not split), 8 streams 4.98 vs 5.10 (+2%), 16 streams 4.89 vs 5.29 (+8%), bit-identical. It stays
off by default; it is worth enabling for gateways that regularly run 16 or more concurrent sessions.

Raw outputs: `benchmarks/results/kaggle/run27_phase2_7b`, `pingpong_phase2_7b`, `run27_token_profile_cpu`.
(A first run 27 attempt is not recorded: the Rust toolchain check passed without cargo, so it measured
the pure-Python fallback. The check is fixed and a failed native build now stops the job.)

## Update 2026-11-20: phase 1 of the speed and readiness plan (Kaggle run 26)

Plan: `research/plans/speed-readiness-plan-2026-11-20.md` (55-agent sweep, adversarially verified).
Seven packages (wnorm in the manifest, noise-monitor lean path, GQA attention with a load-time
bit-equality check, speculation privacy fix with fixed row counts, service per-session batching,
telemetry, security and ops hardening) plus reviewer fixes. Run 26, Qwen2.5-7B on one T4, all outputs
bit-identical:

| metric | run 25 | run 26 |
|---|---|---|
| decode, 1 request | 0.365 s/token | **0.339 s/token** |
| time to first token, 40-token prompt | 2.89 s | **2.37 s** |
| generated tokens/s, 4 / 8 / 16 streams | 3.0 / - / - | **4.16 / 4.70 / 4.61** |
| decode round trip (remote) | | 1.47 ms (112 per token) |
| decode gateway work per trip | | 0.72 ms |

Decode time per token: 0.165 s round trips, 0.080 s gateway masking, 0.094 s other (LM head, float ops,
Python). Loopback socket floor: 33 us (64 B) to 81 us (80 KB) per exchange.

## Update 2026-11-20: bfloat16 LM head (exact)

Open checkpoints ship in bfloat16, so the float32 LM head's values are all exactly bfloat16. The
gateway now keeps the head in bfloat16 when that is lossless (checked at load) and the deterministic
kernel widens each weight to float32 on load (exact), so logits are bit-identical to the float32 head
while half the bytes are read per token (1.1 GB instead of 2.2 GB for Qwen2.5-7B, and 1.1 GB less
gateway memory). Local measurement, 7B shapes with 4 layers and a stub service (`benchmarks/e2e_local.py
--preset 7b4 --null-service`): 56 ms -> 42 ms per decode step. Kaggle run 23 showed no change (0.436
s/token) because its packs used `--plain-dtype float16`, which rounds the bfloat16 weights so the
head cannot be compacted; the Kaggle jobs now pack with `--plain-dtype bfloat16` (lossless). Run 24 with bfloat16 packs: 0.410 s/token
(from 0.436), TTFT 2.94 s, outputs bit-identical. Run 25 adds the fused Triton GEMV for 1-row calls and the
service wire-overhead cut: **0.365 s/token**, TTFT 2.89 s, 4 streams 3.0 tok/s, service 1-row call p50 1.39 ms
(0.70 ms GPU), round trips 0.114 s/token; outputs bit-identical. (The multi-row tensor-core Triton kernel does
not compile on T4/Triton 3.6; the startup self-check limits the fused kernel to 1 row, as designed.)

## Update 2026-11-19

**Speed (Qwen2.5-7B, one Kaggle T4, default ring K32A12C27, output bit-identical to the plaintext
reference in every run).** Raw outputs are in `benchmarks/results/kaggle/run13_*` to `run19_*` (run 19: all changes together).

| metric | 2026-11-17 | 2026-11-19 | what changed |
|---|---|---|---|
| decode, 1 request | 0.86 s/token | 0.339 s/token | persistent WebSocket channel (uvicorn), K=32 uint32 path, native (Rust) gateway kernels, numpy wipes, batcher fast path |
| time to first token, 40-token prompt | 8.4 s | 3.0-3.3 s | pipelined prefill (gateway and service overlap) |
| one service round trip, 1 row (load test) | 2.5 ms | 1.75 ms (0.80 ms GPU) | uvicorn WebSocket framing, inline small batches |
| decode with speculative decoding, 0.5B draft on a gateway GPU | n/a | 0.43 s/token | exact verification, coupled drafts, adaptive draft length; 2.6x fewer round trips |
| gateway CPU per masked call (Kaggle CPU) | about 2.9 ms | about 1.8 ms | `pgw_fast` kernels: prepare 0.05 ms, finish 0.2 ms; the rest is Python orchestration |

Where a decode token's time goes now: about 112 x 3.5 ms in round trips (of which about 0.9 ms is GPU
work; the rest is service and client Python and WebSocket framing), about 0.2 s gateway work, and
about 0.1 s for the float32 LM head. A draft model on a CPU-only gateway does not pay off yet.

**Concurrency on Kaggle: 2.8 generated tokens/s for 4 streams in run 22 (2.3 in run 21, before the multi-row LM head).** Run 21 breakdown: The 4 sessions
share round trips (7392 trips for 4 x 65 tokens), but each 4-session step takes about 1.7 s: about
0.7 s in round trips (gateway, service and mask factory share the VM's 4 vCPUs), about 0.5 s of gateway
crypto, and about 0.5 s of float32 LM heads computed one row at a time (2.2 GB of weights read per row).

**Fused Triton ring GEMV for one-row calls** (`pgw.core.triton_gemv`, on by default after a device
self-check, `PGW_SERVICE_TRITON_GEMV=false` turns it off). One kernel, int8 weight streamed once, wrapping
uint32 multiply-adds, bit-identical to the digit-GEMM path. Kaggle T4, 7B shapes, K32A12C27, rows=1
(`benchmarks/results/kaggle/triton_gemv/`): service call per layer 2.92 -> 1.92 ms (qkv 0.51 -> 0.33,
o 0.56 -> 0.28, gate_up 1.17 -> 0.81, down 0.68 -> 0.51); device time per layer 1.85 -> 1.26 ms (180-220
GB/s). Several rows stay on the digit-GEMM path on the T4: the CUDA-core kernel is compute-bound there
(4-8x slower) and the int8 `tl.dot` variant does not compile with Triton 3.6 on sm_75 (the self-check
detects it and limits the fused kernel to one row; GPUs where it compiles and passes get up to 16 rows).

**CUDA graphs are off by default** (`PGW_SERVICE_CUDA_GRAPHS=true` turns them on). On torch 2.11 with
a T4 most captures fail inside PyTorch's caching allocator and the service falls back to eager, and
where capture worked the gain was about 6% per layer (run 20). The graph tests skip when capture is
unsupported; exactness is still checked.

**New components.** `rust/pgw-fast` (native hot path, optional, bit-identical, falls back to Python);
PGW Console (`/console/`: operator web UI with sessions, admin/viewer roles, CSRF, strict CSP, usage,
audit, playground); prompt-lookup drafting; WebSocket compute channel with HTTP fallback.

**Fixed.** A shutdown spin in `Generator.wait_closed` that could hang model reload or gateway stop
under load. Speculative verification is now bit-identical (it differed by up to 0.054 in logprobs).

**Readiness.** Unchanged in substance: a technical pilot in topology B is reasonable; a production SLA
pilot still needs SSO for the console, the hardware key path, a GPU soak test, signed images and an
external security review.

## 1. What is built

About 26.8k lines of Python in `src/pgw` (96 modules, mypy strict, ruff clean), 89 test files, and a
2.5k-line Rust port of the core.

| component | state | notes |
|---|---|---|
| `pgw.core` | complete | Ring presets K40A12 (default), K48A20 and K32A8. Exact ring matmul on `cuda_int8` (T4-validated), `cpu_int8` (shared probe-gated int8 digit GEMM), `cpu_f64` and `cpu_f32`. Also masking with range check, Freivalds with soundness bound, the ChaCha20/HKDF mask PRG (known-answer tests), packing and fingerprints. |
| `pgw.modelpack` | complete | HF checkpoint (Qwen2 and Llama families) to an int8 pack with manifest and SHA-256 fingerprint. `verify=full` by default on the gateway. Unsupported RoPE types are refused. |
| `pgw.engine` | complete | Plaintext transformer with KV cache, batching, sampling, logprobs and streaming. Backends: `MaskedBackend` (production), `QuantPlain` (reference), and float (dev). |
| `pgw.factory` | complete | CPU and GPU mask generation, auto Freivalds check count (at least 20 bits), self-check, refill scheduler, partner export (HPKE plus Ed25519 signature plus destination binding), and health/metrics. |
| `pgw.maskstore` | complete (core); providers partial | Encrypted single-use segments, a durable consumption cursor, windowed reservations, crypto-shredding, rollback witnesses (HMAC journal plus external counter), refusal of network filesystems, and KEK rotation. See section 5 for the providers. |
| `pgw.wire` | complete | HTTP/2 plus msgpack, mTLS, retries with a deadline header and retry budget, and byte-budget admission. |
| `pgw.service` | complete | Stateless compute: registry keyed by fingerprint, GPU and CPU kernels, cross-request batching, token or mTLS-identity tenants, quotas, drain, and an internal metrics/probe listener. |
| `pgw.gateway` | complete | OpenAI-compatible `/v1/chat/completions`, `/v1/completions` and `/v1/models`, with SSE, API keys, rate limits and quotas, admin API, audit log, integrity circuit breaker, outbound-noise monitor (fail closed), dashboard, and graceful shutdown. |
| `pgw.observability` | complete | A Prometheus metric contract (`METRIC_NAMES`), the noise monitor and the dashboard. |
| `pgw.cli` | complete | `pgw pack/inspect/factory/maskstore/service/gateway/doctor/bench/config` (SPEC 12). |
| `rust/pgw-core-rs` | complete (reference) | Ring, PRG and Freivalds port, cross-checked against `tests/vectors/core_v1.json`. Optional pyo3 extension. |
| `deploy/` | complete, partly unverified | Five Dockerfiles (digest-pinned, hash-locked requirements), docker-compose demo, Helm charts for gateway, service and factory, systemd units, a VMware OVF/Packer appliance, and AWS/Azure marketplace READMEs. |
| `docs/` | complete | Security whitepaper, threat model, gateway API reference, operations guides (on-prem, customer cloud, air-gapped factory, sizing), compliance (audit readiness, FAQ), and a sizing tool. |
| CI | written | `.github/workflows/ci.yml` (lint, types, tests, audit, images) and `ci/check.sh`. The workflow sits under `pgw/` and has not been copied to the repository root yet. |

### 32-bit ring `K32A12C27` (new, 2026-11-18)

This is a capped ring with per-token adaptive activation bits; it decodes any layer width exactly.
Its quality is identical to `K40A12` on Qwen2.5-0.5B and 1.5B (perplexity 11.714 vs 11.714 on 1.5B,
with 0 tokens needing fewer bits). It cuts 20% from traffic, service GPU work and mask storage. With
encrypted masks it uses q = 2^56: 7B mask-supply traffic drops from 6.7/8.3 MB to 5.8/7.0 MB per
token and the hint from 39.9 to 34.2 GB (T4 run 11). The cost is about 1 ms more gateway CPU per
layer, for 7 Freivalds vectors instead of 3. Not done yet: a 7B quality check (blocked on memory
here) and making it the default.

### Encrypted mask generation (new, 2026-11-18)

`pgw.hemask` lets the mask factory get `W8 r` from the compute service on MLWE-encrypted masks
(`pgw factory run --he-service ...`, demo `--mask-source he`). Storage no longer grows with
token volume. Instead there is a one-time hint of about 9.3 GB for 0.5B and about 45 GB for 7B,
plus a just-in-time buffer, and there is no factory GPU. The privacy assumption is MLWE, with a
single provider.

- Verified: factory output is bit-identical to local generation, a tampered provider is
  rejected, and an end-to-end chat on 0.5B with encrypted masks gives the same answers with 0
  integrity failures.
- The refill daemon refilled the store through the service while the gateway was serving.
- T4: the CUDA `mulmod64` kernel is exact. 7B costs 0.020 s of provider GPU per token.
  Gateway CPU per token is 0.11 s with a gateway GPU and 0.116 s on an AMX CPU, after the
  int8 byte-plane optimisation (it was 1.18 s). The hint is 39.9 GB.
- Not done yet: a lattice-estimator confirmation of the parameters, int8 AMX for gateway
  encrypt and decrypt, and an external crypto review. See
  `docs/operations/encrypted-masks.md` and `research/mask-storage/`.

## 2. Test results (this release run, 4-CPU container, CPU-only torch)

| suite | result |
|---|---|
| `pytest tests` (all, including e2e on real Qwen2.5-0.5B-Instruct) | **1491 passed, 0 failed, 40 skipped** in 7 min 11 s, on the final run after the release fixes. The first run of this release gave 1490 passed. |
| skipped | 35 need CUDA, 3 need the `helm` binary, 2 need the optional Rust extension or `PGW_TEST_RUST=1` |
| `cargo test` (rust/) | **42 passed**, 0 failed |
| `ruff check src tests` | clean (after this release's fixes) |
| `python -m mypy src/pgw` (strict) | clean, 96 files (after this release's fixes) |
| `scripts/e2e_demo.sh` (Qwen2.5-0.5B, CPU) | **pass**: "Paris" in both non-streaming and streaming mode, 384 round trips, 384 Freivalds-checked, 0 integrity failures, 76 MB up / 128 MB down, noise monitor active |

### 2.1 Fixes made in this release pass
- **Demo was broken.** After the security fix that requires an external monotonic counter,
  `scripts/e2e_demo.sh` returned HTTP 500 (`config`) on its first chat request. The demo now
  configures a file counter outside the mask directory.
- **Gateway fails fast.** A gateway whose mask store has no external counter now refuses to start
  with a clear `ConfigError`, instead of starting "ready" and answering every request with 500.
  Regression test: `tests/gateway/test_config_safety.py::test_gateway_refuses_store_without_external_counter_at_startup`.
- **mypy.** Fixed a strict error in `MaskStore.metrics()`: the label tuple type now covers the
  store-wide series that have no labels.
- **ruff.** Fixed the 15 remaining findings: import sorting, line length, and a redundant `int()`.
- **Docs.** Freivalds statements in the whitepaper, threat model T-03 and docs/README now match the
  auto-checks default (c = 3 at width 4096 gives 2^-24, with a floor of 2^-20) instead of the old
  `c = 2` / 2^-16. The docs consistency test is updated to match.
- **PKCS#11 docstring.** It claimed the provider is tested against SoftHSM2 "in the release
  pipeline". No such pipeline exists, so the docstring now says so.
- **Packaging.** Added the optional extras `hsm`, `kms-aws`, `kms-gcp` and `kms-azure` for the
  key-provider SDKs, which `pyproject.toml` did not declare before.

## 3. GPU validation (Kaggle, Tesla T4, K40A12)

Full details are in `benchmarks/RESULTS.md`. Raw outputs are in `benchmarks/results/kaggle/`.

- **Exactness.** Qwen2.5-0.5B and Qwen2.5-7B produce tokens, logprobs and top-2 logprobs that are
  bit-identical to the QuantPlain reference. This holds for non-streaming, streaming and 4 or 8
  concurrent requests, with 0 integrity failures and 0 skipped checks. The 7B pack fits one 16 GB T4
  (12.9 GB used).
- **GPU tests.** `pytest -m gpu` gives 35 passed. The e2e suite with the service on CUDA gives 32 passed.
- **Three GPU bugs found and fixed.**
  1. cuBLASLt rejects a row-major mat2.
  2. `/readyz` probed a 1x8 product, which the T4 rejects.
  3. Dimensions must be padded to multiples of 32.

| metric (7B) | measured | SPEC 14 target | status |
|---|---|---|---|
| service GPU compute, batch 1 | 0.16 s/token (was 0.45) | ≤ 0.5 s | met |
| service GPU compute, 32 rows | 0.018 s/token (0.012 at 128 rows) | ≤ 0.015 at batch 32 | roughly met |
| end-to-end decode, 1 request | 0.86 s/token (was 1.48) | (implied ≤ 0.5) | **missed**: limited by 112 sequential round trips and per-call gateway CPU |
| gateway CPU per token | 0.38 CPU-s (was 1.32; 0.27 at 8 concurrent) | ≤ 10 ms crypto + ≤ 20 ms plaintext | **missed by about 40x** |
| factory, T4 | 169k tokens/h (7B, K40) | ≥ 500k slots/h (8B, K48) | **missed by about 3x** (GEMMs are only 36% of the time) |
| mask storage | 6.95 MB/token | 6.9 MB | met |
| wire | 10.96 MB/token | 11.2 MB | met |
| test suite, 4 CPUs | about 6 to 7.5 min | < 5 min | slightly missed (the e2e on real Qwen is about 4 min of it) |

Only K40A12 was measured end to end on GPU. K32A8 and K48A20 are covered by ring benchmarks and unit
tests. Multi-GPU placement passed its unit test but was not benchmarked. A full 7B re-run with the
final padding fix (bug 3) was not done. No real model shape is affected by that fix.

## 4. Known limitations

**Performance**
- One token is 4 x layers sequential round trips: 112 for 7B and 128 for 8B. On loopback this costs
  about 0.5 s/token. Over a 10–30 ms WAN link (topology A) it costs 1.3–4 s/token unless sessions
  are batched.
- The Python request path is the bottleneck. Masking and unmasking, msgpack, Freivalds and mask
  decryption cost the gateway about 1.3 CPU-s per 7B token. The service through HTTP tops out at
  about 13 token-rows/s on 4 vCPUs while the GPU is about 30% busy.
- The factory spends most of its time in the CSPRNG, digit arithmetic and serialization, not in GEMMs.

**Functional (rejected with HTTP 400)**
- Tools and function calling, structured output (`response_format`), `n > 1`, and multimodal inputs.

**Security**
- Privacy is computational at the PRG level (HKDF-SHA-256 plus ChaCha20 seeds), not information-theoretic.
- Metadata is visible to the service: model, tokens per round, timing and tenant. Row padding
  (`pad_rows_to`) and constant-rate dummy rounds are specified (SPEC 3.4) but not implemented.
- Freivalds covers only the service's product. A wrong `W r` from a partner factory is caught only by
  the factory's own self-check and by the gateway range check for gross errors. An independent
  spot-check (threat model T-29) is planned.
- After an integrity failure the gateway does not rotate to a segment with a fresh `u`. The breaker
  stays open until an admin resets it, which is the default.
- The soundness floor is enforced when a segment is first used, not by a scan of the store at start-up.

**Operations**
- Every production deployment needs an external monotonic counter on storage separate from the mask
  volume. Without one the store and gateway refuse to serve.
- The VMware Packer template does not create the separate counter disk. First boot fails with an
  explanation until an operator attaches it at `/var/lib/pgw-counter`.
- Velero exclusion of the Helm counter PVC relies on a label. CSI VolumeSnapshot policies must
  exclude it separately.
- Segments written before the reliability fixes keep their old 256-slot chunks, which use more gateway
  memory until they are consumed.
- Stale-lease detection assumes a single gateway per store, which is the supported topology.
- The hypercorn mTLS identity shim hooks hypercorn internals, so hypercorn is pinned below 0.19.
- The Helm charts render the service token into the models file. They could use the newer
  `service_token_file` instead.

## 5. What is stubbed or unverified

| item | state |
|---|---|
| `Pkcs11KeyProvider` (HSM, AES-KWP, HMAC) | **Stub-quality.** The code is written, but only construction and error paths are unit-tested. wrap/unwrap/mac have never run against SoftHSM2 or a real HSM. |
| `KmsKeyProvider` (AWS KMS, GCP KMS, Azure Key Vault) | The provider logic is tested with an injected fake client. The real SDK adapters (`_AwsKms`, `_GcpKms`, `_AzureKv`) have never run against a cloud account. |
| `ProviderCounter` (HSM or KMS hardware counter) | No shipped key provider implements `monotonic_counter()` yet, so `counter=provider` cannot be used in practice. |
| `TpmNvCounter` (tpm2-tools) | **Stub-quality** and untested on a TPM. A vSphere vTPM is captured in VM snapshots, so it does not detect a snapshot revert. |
| Metadata padding and dummy rounds (SPEC 3.4) | Specified, not implemented. |
| AWS Marketplace metering hook | Deferred (SPEC 17). Only listing READMEs exist under `deploy/marketplace/`. |
| fbgemm or oneDNN CPU int8 path and a fused GPU kernel | Roadmap (SPEC 16). `cpu_int8` uses torch `_int_mm` digit GEMMs. |
| Docker image builds, `helm lint`, shellcheck, pip-audit, cargo audit | Written and wired into CI, but **not run here**: no Docker daemon, helm or network audit tools. The charts were checked with the repository's Python Helm renderer, and compose with `docker compose config`. |
| CI workflow at the repository root | Not copied. It lives at `pgw/.github/workflows/ci.yml`. CI test jobs install unlocked `.[dev]`; only the image and audit jobs use the hashed locks. |
| Terraform for topology B | Mentioned in SPEC 13, but not in `deploy/`. Only Helm exists. |

## 6. Recommended next steps before a customer pilot

**Must do (blocking)**
1. **Hardware and cloud key path.** Run `Pkcs11KeyProvider` against SoftHSM2 in CI, then against the
   pilot customer's HSM or KMS. Implement and test at least one hardware or cloud-backed
   `ProviderCounter`, or validate the file-counter-on-a-separate-volume design for the customer's
   storage and backup tooling.
2. **Performance work on the request path.** Profile the gateway and service, and move masking,
   unmasking, Freivalds, msgpack and decryption into vectorised or native code (the Rust core is a
   starting point). Fewer per-call copies. Targets: under 50 ms of gateway CPU per 8B token, and under
   0.3 s/token end to end in topology B.
3. **External security review** of the whitepaper, threat model and implementation, focused on mask
   single-use under crash and restore, partner delivery, and the noise monitor.
4. **Build and scan the real images.** Run `docker build` for all five Dockerfiles, `helm lint` and a
   kind install, `pip-audit` and `cargo audit` on the locks, and an SBOM. Then sign the images.
5. **Full 7B or 8B soak on GPU** with the final code: hours of mixed concurrent load, gateway
   restarts, a service rollout during traffic, mask-store exhaustion and refill, and a disk-full test.
   Run it on K40A12 and on K48A20.

**Should do**
6. Topology B packaging: Terraform modules (VPC, PrivateLink or PSC, KMS key, gateway node pool) and
   a customer runbook.
7. Factory throughput: move the CSPRNG and digit split to the GPU to reach 500k or more slots/h on a T4.
   *Implemented, not yet measured on a GPU:* `pgw.factory.gpu_prng` computes the ChaCha20 keystream
   (key still derived on the CPU by HKDF) straight into the int8 digit planes of the GEMM operand, so
   `r` never crosses the host bus. It is on by default on CUDA after a self-test against the CPU
   expander (`PGW_FACTORY_GPU_PRNG=0` turns it off), and the output self-check re-derives its sampled
   rows on the CPU (so the GPU path is refused when `self_check_rows` is 0; a kernel error or a failed
   check switches the generator back to CPU ChaCha20). The Kaggle job `pgw-factory-gpu-prng` runs the CUDA byte-identity tests and the 7B
   K32A12C27 slots/h comparison with the setting at 1 and at 0.
8. Close the integrity gaps: rotate segments after an integrity failure, scan the soundness floor at
   start-up, and spot-check factory `W r` independently (T-29).
9. Metadata hardening: implement `pad_rows_to` and the optional dummy rounds.
10. Move CI to the repository root, run the test jobs against the hashed locks, and add a nightly
    Kaggle or GPU job for `pytest -m gpu` and the e2e suite on CUDA.
11. Bring the test suite back under 5 minutes, for example by marking the real-Qwen e2e as `slow` in
    the fast lane.

**Later (roadmap)**
- Fused GPU ring kernel, SmoothQuant so that K32A8 becomes lossless, a gateway-GPU "self-factory"
  mode, speculative decoding and prefix caching to cut round trips, and tools/structured output in
  the API.

### Phase A wiring check (2026-11-20)
- Commit 450b788 (marked WIP by a forced hook commit) is complete: full suite 3615 passed / 122 skipped, tests/e2e 32 passed.
- Modal e2e (CPU provider, tunnel, 0.5B): exact (first_mismatch null); local_linears auto chose qkv+o on the gateway, 49.5 round trips per token (was ~100). That placement had a 26 ms median trip, so the warm 32-token chat took 47 s; trip latency, not compute, dominates on Modal.
