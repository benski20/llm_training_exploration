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

## Update 2026-11-22: decode at 30 ms per token co-located (package S4)

Reached in the co-located demo: 29.9-30.3 ms per decoded token (33 tokens/s after the first, streamed), from 36,
exact. The remaining contention was the encrypted mask factory (its maskgen on the provider's GPU and its
ciphertext traffic), measured by pausing it inside one run (36 -> 32 ms per token). Details:
`deploy/modal/README.md`, "Decode rate, contention and the factory yield".

- `PGW_FACTORY_YIELD_FILE` (library default off; co-located demo on at 1000 slots): the factory waits before
  each encrypted batch and provider request while the gateway runs masked passes and the stock is above the
  threshold; below it, full speed. Same masks, requests and bytes; nothing new for the provider.
- `PGW_SERVICE_TRIP_STREAM_PRIORITY=high` (provider, library default off; demo on): decode graphs on the GPU's
  highest-priority stream. Scheduling only.
- `--lane uds` is now the co-located default (1-row trip 0.29 ms vs 0.44 ms on TLS in the combined config).
- The e2e reports TTFT and the streamed decode rate (`warm_stream_*`); `PGW_MODAL_AB` runs in-run A/B phases.
- Core pinning is impossible on Modal (`sched_setaffinity` EPERM under gVisor). Local GPU linears stay at about
  79 us per call with or without the factory: not contention.

## Update 2026-11-22: lean per-trip gateway path and the co-located UDS lane (package T3b)

Not reached: 30 tokens/s. Three exact Modal runs the same hour (T4, attention_down, native float ops, factory
capped; every run logprobs bit-identical, text equal, repeat identical): per decoded token 53-55 ms (TLS lane,
new paths off), 52-54 ms (TLS, new paths on), 53 ms (UDS lane, new paths on); warm chat 12.4 / 13.1 / 11.9
tokens/s. The gains are within Modal's host-to-host noise. Details: `deploy/modal/README.md`, "Lean per-trip path
and the UDS lane".

- `PGW_TRIP_FAST` (library default off; co-located demo on): a one-call pass (decode) skips the batch
  bookkeeping; the extent's soundness check runs inside `take_checked` (new `verify` hook) before any row is
  taken, so the separate pre-check lookup is gone; lean finish; cached constants for one-call local linears.
  Same masks, draws, payload bytes, checks on every row, outputs and wiping (tests/engine/test_trip_fast.py).
  Dev box, per trip: prepare 339 -> 279 us, finish 155 -> 153 us; local linear 330 -> 317 us per call.
- `PGW_NOISE_ASYNC_SAMPLED` (default off, also off in the co-located demo: the sampled payload leaves before its statistics finish): a sampled call's every-call checks stay inline, its
  statistics (about 400 us) run on a worker on wiped copies, in call order, joined before the next prepare (an
  alarm blocks the next payload, fail closed). Statistics identical to inline (tested).
- `launch.py --lane uds` (co-located, opt-in): the product's same-host UDS lane between the two party users;
  the service accepts the gateway user's uid only, the socket's directory is 0710 and the socket 0660 to that
  user's group. 1-row trip 0.44-0.47 ms (TLS: 0.54-0.57 ms); not made the default: no clear tokens/s gain in
  the measured run.
- What is left per trip on the dev box is mostly cache-cold native work (the finish kernel streams `u`, `W r`
  and the scale: 22 us warm, about 100 us in place) and the store's amortized derivation, not Python.

## Update 2026-11-22: trip latency floor and gateway per-trip CPU (package T3)

Not reached: the 1-row trip stays at 0.57 ms in three exact Modal runs (12.0-13.0 tokens/s back to back). The
provider side is already native; under Modal's gVisor the TLS lane transport alone costs 0.28 ms per trip (0.03-0.05
ms on a plain VM) and polling makes it worse. Spin knobs added, off by default; details in `deploy/modal/README.md`,
"Where a 1-row trip's 0.57 ms goes". Gateway per-trip CPU measured in place (dev box, `decode_cpu_bench.py`
wrappers): prepare about 340 us (mask take 85, of which the stage copy and amortized derivation; precheck 75;
request 28; noise guard 18 every call, 400 per sampled call; native prepare 30-40) and finish about 120 us (native 20);
one redundant `remaining()` per trip removed. Collapsing these into one native call per side is the remaining work.
## Update 2026-11-22: native pinned float path (package T3)

Per decoded token of one Qwen2.5-0.5B chat co-located: **46-53 ms (from 62-64 in a same-day torch A/B); back to
back 13.4-16.6 tokens/s (12.7), warm chat 13.7-14.9 (11.6)**, every run logprobs bit-identical, text equal, repeat
identical. Details: `deploy/modal/README.md`,
"Native pinned float path".

- **New pinned float reference (decision delegated by the user, 2026-11-22), opt-in.** Model setting
  `float_ops = "native"` (or `PGW_FLOAT_OPS=native`, `TransformerModel.set_float_ops`): RMSNorm, residual adds,
  RoPE, attention over the KV cache and the SiLU gate run in `pgw_fast` (`layer.rs`, `attn.rs`) in a fixed order
  of IEEE operations (no FMA, no libm `exp`; AVX2 and baseline builds give the same bits), one native call
  between consecutive linears. Every output row depends only on its own inputs and the cache prefix: the bits do
  not depend on the ISA, torch's thread count or the rows in a pass (speculative verify rows equal decode rows by
  construction). RMSNorm, attention and SiLU differ from torch in the last bits (RoPE and residual adds are
  torch's bits), so the gateway and its plaintext reference must both select it; the HF closeness tests pass with
  it (`tests/engine/test_native_float.py`, plus the existing suites run under `PGW_FLOAT_OPS=native`). The final
  norm and the LM head are unchanged. Library default stays `torch` (no golden changed); the co-located Modal demo
  now defaults to it (`launch.py --float-ops native`, reference via `plaintext_reference(float_ops=)`) after an
  exact e2e. Privacy unchanged: local gateway math only.
- Per decoded token now (about 50 ms): trips 13 ms, local linears about 10 ms, mask prepare/finish about 8-10 ms,
  LM head 4 ms, float ops and engine Python about 8-10 ms (from about 20). Not reached: 30 tokens/s.
## Update 2026-11-22: native local linears and the LM head on the gateway GPU (package T3)

Goal 30-40 tokens/s for one Qwen2.5-0.5B chat co-located with `--gateway-gpu`; reached **14.5-16.1 tokens/s back
to back** (12.1-13.5 warm chat; the same code with the new paths switched off: 10.9 on that run's host), every run
logprobs bit-identical, text equal and repeat identical. Details: `deploy/modal/README.md`, "Gateway GPU: native
local-linear call, fixed-order LM head".

- Local linears: one GIL-holding native call per linear (`pgw_fast.local_gpu_linear_into`: prepare into pinned
  memory, `cuGraphLaunch` through the driver, spin, range check and rescale, buffers wiped), a lean self-checked
  1-row path in the backend, and measured T4 launch configurations (down 66 -> 46 us). In the running gateway
  220 -> 99 us per call, 15.9 -> 7.1 ms per token (the device floor is about 22 us per call; the rest is the busy
  container). Integer results unchanged: start-up self-check per linear against the CPU reference.
- LM head (`lm_head_device = "cuda"`, opt-in, library default `cpu`; on in the co-located demo with
  `--gateway-gpu`): the existing fixed summation order as one Triton kernel with `mul.rn`/`add.rn` (no FMA, no
  flush to zero), so the logits are the CPU kernel's bit for bit; the plaintext reference needs no change and
  stays on the CPU. Start-up self-check, else the CPU. About 4 -> 1.4 ms per token. The final norm and sampling
  stay on the CPU. HF closeness tests and goldens unchanged (library default).
- Privacy unchanged: the GPU is the gateway's alone (per-UUID visibility, 0600 device nodes); nothing new
  reaches the provider.
- Per decoded token now (about 48 ms): trips 12 ms, the masked trips' gateway work 10 ms, local linears 7 ms,
  LM head 1.4 ms, float ops and Python about 17 ms. Not reached: 30 tokens/s.

## Update 2026-11-22: gateway GPU and pinned float threads (package T2)

Goal 30-40 tokens/s for one Qwen2.5-0.5B chat co-located; reached **13.3-13.6 tokens/s** (warm and back to back,
from 10-13.5), every run logprobs bit-identical and repeat identical. Details: `deploy/modal/README.md`, "Gateway
GPU, pinned float threads, factory cap".

- Exact local linears on a gateway-owned GPU (`local_linears_device = "cuda"`, `pgw.engine.local_gpu`; co-located
  `launch.py --gateway-gpu`, off by default, two T4s, one per party, each process sees only its own). Masked trip
  with `r = 0` on the gateway's device, self-checked per linear against the CPU. Halves the local GEMVs in
  isolation; in the running gateway about 10 ms/token either way, so little net gain for twice the GPU cost.
- **Pinned float formulation (decision delegated by the user, 2026-11-22).** Torch's CPU SDPA gives different bits
  at different thread counts on Modal's AVX-512 hosts, so the reference's bits depended on the host's default
  thread count. The co-located demo now pins both the gateway's process and the e2e's plaintext reference to the
  same torch thread count (`launch.py --float-threads`, default 1; 0 = the old behaviour): the reference's bits
  come from that pinned formulation, and the gateway's `float_threads = 1` compares 1 with 1. Privacy unchanged;
  the e2e compares against the pinned reference (bit-identical in the measured run). Library defaults (and the
  test suite's goldens and HF closeness tests) are unchanged: the pin is a deployment setting.
- Also: q and k rotated in one call (same bits), leaner GPU replay, factory cap knobs measured
  (`PGW_FACTORY_CPU_THREADS=3`, `PGW_FACTORY_NICE=10`: 120 slots/s on 3.9 cores while decoding, 9x the token rate),
  an opt-in stack sampler (`PGW_STACK_SAMPLER`), `benchmarks/local_gpu_bench.py`, `benchmarks/sdpa_threads_bench.py`.
- Per decoded token now (about 60-65 ms): trips 13 ms (24 x 0.52 ms), local linears 10 ms, mask prepare/take/finish
  about 8-10 ms (Python around the native kernels: reservation checks, noise monitor, staging copies), attention,
  norms, RoPE and Python about 20 ms, LM head 4 ms; time to first token 0.5-1.1 s. Not reached: 30 tokens/s.

## Update 2026-11-22: co-located chat throughput on Modal (package T1)

Goal was 30-40 tokens/s for one Qwen2.5-0.5B chat on the co-located Modal placement; reached **10-11 tokens/s
warm chat (12-13.5 tokens/s back to back), from 4.0-4.7**, every run logprobs bit-identical and repeat
identical. Details and the per-token breakdown: `deploy/modal/README.md`, "Co-located throughput".

- Co-located defaults: `attention_down` + TLS lane; the encrypted factory skips the local kinds
  (`PGW_FACTORY_SKIP_LINEAR_KINDS`, gate_up masks only): 138 slots/s on 2.5 cores while chats decode, so
  the factory no longer competes with decode (it was the 4.07 tok/s run's main loss) and keeps up 10x over.
- Local linears of any row count take the native kernel (prompt passes no longer use the two-digit torch
  int8 GEMM, 10-140 ms per call on Modal's AVX2 hosts); decode GEMVs split input features over the pool.
- Pool knobs `PGW_POOL_SPIN_US`, `PGW_POOL_WORKERS`; launcher `--gateway-env`, `--provider-env`,
  `--sustain-s`, loopback warm-chat figure; `gpu_bench.py` CPU targets `c8`/`c16`.
- What is left per decoded token (about 57-62 ms): round trips 12 ms, mask prepare/finish 10.6 ms, local
  GEMVs 17-25 ms (148 MB of int8 weights from DRAM; pool threads scale poorly under gVisor), LM head 3.5-4 ms,
  float ops and Python 12-15 ms; time to first token 520-640 ms. `float_threads = 1` stays off on Modal's
  AVX-512 hosts: torch's CPU SDPA gives different bits at 1 and 14-16 threads there (the self-check is right
  to refuse; the plaintext reference itself depends on the thread count on those hosts). Tried and dropped
  (no gain on Modal): cache warm-ahead of local weights, a 200 us GIL switch interval.
## Update 2026-11-21: encrypted mask factory, gateway side (package M3)

Design and numbers: [research/plans/he-gateway-decrypt.md](research/plans/he-gateway-decrypt.md).

- **Fused NTT decryption** (`PGW_FACTORY_HE_DECRYPT`, default `auto`): the hint product `H S` runs as NTT
  ring products (native `pgw_ntt`, exact by the CRT bound, checked per call) and decryption (subtract, round,
  transpose into the output) is fused into the same native pass, so the correction never leaves native scratch
  memory. `auto` uses it per pass size only when a start-up calibration on public random data found it
  bit-identical to the dense path and faster; any native error falls back to dense. Optional hints in NTT form
  (`PGW_FACTORY_HE_HINT_NTT_GB`, default 0; twice the hint's size in RAM). Measured locally (4 vCPUs with AMX,
  0.5B shapes, decrypt only): 7.6 -> 1.9 ms/slot at 16 groups per pass, 6.1 -> 5.1 at 2 groups (1.5-2.0 with
  hints in NTT form). Kaggle 4-vCPU Xeon without fast int8 (measured): 16.5 -> 8.2 ms/slot at 0.5B and 75 -> 38 at
  7B (16 groups), 5.9 and 27 with hints in NTT form; GPU-provider end-to-end numbers not measured (GPU session cap).
- **Masks of local linears** (`PGW_FACTORY_SKIP_LINEAR_KINDS=qkv,o`, opt-in): for gateways with
  `local_linears = attention` (and no `local_max_rows`) the factory no longer makes qkv/o masks (16% of mask
  bytes at 0.5B). The gateway declares at start-up which linears it sends (`MaskStore.require_linears`);
  segments lacking one are retired (written off, never reserved), so a later configuration change cannot run
  short or reuse a mask. Reservations may now span segments with different linear sets.
## Update 2026-11-21: mask data path at high bandwidth (package X2)

Memo: `research/plans/mask-ram-path-x2.md`. At 1k tokens/s a 7B gateway needs 5.6 GB/s of `W r`, a
70B deployment 27 GB/s, so masks go factory -> locked RAM ring -> gateway, with the encrypted disk
store as the outage reserve. Measured on the 4-vCPU dev box (A/B against the base commit, per full
slot): the factory writes into the ring at 0.85 ms per 7B slot (store: 9.6 ms). The gateway take was
the limit, and is now faster: **7B 172 -> 276 slots/s per core, 0.5B 350 -> 586** (store tier: 89 and
150, unchanged). Ring segments are now read in place under the slab lock
(`RamSegment.read_wr_rows`) instead of each session copying a 1 MiB chunk. The rows are unchanged,
and so are single use, wipe-on-consume and the recycled-slab checks. The remaining floor is the pad
keystream (1.8 ms per 7B slot, 8 keystream bytes per element by SPEC 7.2), not the `W r` bytes
(0.37 ms memcpy). `W r` is already K-bit packed at rest and in RAM, so packing cannot remove anything
more exactly. A pad-format v2 (4 keystream bytes per element at K <= 32) would halve the floor; it
needs a coordinated format change. Also new: an opt-in batched native claim over ring reservations
(`PGW_MASK_RING_BATCHED_CLAIM=1`; off because the native ChaCha20 is slower here), the sustained
producer/consumer bench `benchmarks/mask_ram_path_bench.py`, and an end-to-end test of the encrypted
factory -> ring -> gateway path (bit-identical to the local factory).
## Update 2026-11-21: aggregate throughput with many chats (X4)

Write-up: [research/plans/aggregate-throughput.md](research/plans/aggregate-throughput.md). Cross-session batching
already runs (one trip per linear for all live chats; the service concatenates rows), so aggregate tokens/s flattens
at one gateway CPU core: the plaintext thread does masking, float ops and the LM head for every row in series. Two
plaintext threads in one process are slower (GIL). Changes: `float_threads` defaults to 1 on CPU engines
(self-checked; engine step at 0.5B shapes 214 -> 136 ms with 32 chats, measured on 4 vCPUs); several gateway
processes per host over one mask store scale aggregate throughput (1/2/4 processes: 317/538/923 tokens/s on the dev
box, null service, measured). `gpu_validate.py` gained `--variant`, `--shard-gateways` and per-level CPU cores; GPU
numbers pending (command in the write-up, section 7).
## Update 2026-11-21: TCP+TLS trip lane (package L1, opt-in)

The service's native trip lane now also listens on TCP with TLS 1.3 (`PGW_SERVICE_LANE=tcp`,
`PGW_SERVICE_LANE_PORT`; rustls, the service's certificate, client certificates verified against its client
CA), and the gateway has `lane = "tls"`: decode trips use the native `pgw_fast.TripChannel` (rustls, a
48-byte binary header, one `writev`, the GIL released for the whole trip, spin-then-block wait) instead of
the WebSocket stream lane. Encryption and authentication are unchanged or stricter: the gateway verifies
the server against `service_ca_file` as the `service_url` host and connects only to that host; the
single-use ticket comes from `lane/open` on the regular channel (bearer token, client certificate) and is
bound to the client certificate that asked for it. Same masked bytes, request ids and call schedule; any
call the lane cannot take goes byte-identically to the blocking stream lane. Gateway-level tests show
greedy and seeded-sampled answers bit-identical (`tests/gateway/test_speed_wiring.py`); lane tests in
`tests/service/test_lane_tls.py`. On the service, lane calls without a CUDA graph now run inline on the
event loop when the device is idle (one thread hop instead of a task). Modal: `launch.py --placement
colocated --lane tls` (default off). Measured here per 1-row loopback TLS call (0.5B shapes, stub kernels,
load 30-40): stream lane p50/mean 346/795 us, TCP+TLS lane 219/395 us, transport alone 23/74 us; with two
busy Python threads in the gateway 2157/3100 vs 280/975 us. Details: `deploy/modal/README.md`.
## Update 2026-11-21: gateway CPU per decode token (package L2)

Single chat, decode, 1 row, on the gateway's production decode path (TripEngine, synchronous trips, noise
monitor, `local_linears` qkv + o). All changes are bit-identical (self-checked or tested against the
reference) and nothing sent to the provider changes (bytes, sizes, counts, schedule).

- **Native exact local linear** (`pgw_fast.local_linear_into`, `rust/pgw-fast/src/local.rs`): quantize with
  the TripEngine's constant-time quantizer (zero mask), int16 x int8 GEMV (AVX2 `vpmaddwd`, int32 partial
  sums flushed to int64 every `2^(24-A)` features), range check over every output, the finish kernel's
  rescale. Replaces numpy temporaries + a two-digit `torch._int_mm` (whose OpenMP threads also spun against
  the co-located provider). Works for any K multiple of 8 with A <= 15 (K40A12 now takes the fast path too).
  `PGW_LOCAL_NATIVE=0` restores the old form; the per-shape self-check is unchanged.
- **Native noise fingerprints** (`pgw_fast.noise_dup_keys`): `guard_packed`'s duplicate-row keys in one
  call (`PGW_NOISE_NATIVE=0`: numpy). Same keys bit for bit, same history and alarms.
- **Decode mask prefetch overlap**: with synchronous trips (lane `stream`/`uds`) a single session's 1-row
  decode stages its next 8 rows per (reservation, linear) on the prefetch thread, i.e. while the gateway
  thread waits in `recv`. Same rows, same order, same consumption (`PGW_PREFETCH_DECODE` overrides).
- Small: no logits copy in the sampler without penalties; request ids from `os.urandom(16).hex()` (same
  format); the deferred-check decision memoized per linear.
- Small: no logits copy in the sampler without penalties; the deferred-check decision memoized per
  linear. (Request ids stay `uuid4().hex` on every path: a review dropped an `os.urandom` id for the
  native path, whose missing version nibble would have told real trips from filler-only trips.)

Measured here (4 shared vCPUs at load 20-40 from other jobs, so medians of calling-thread CPU; GPU box
numbers pending, see the L2 report):

| item (1 row) | before | after |
|---|---|---|
| local linear 0.5B qkv 896x1152 / o 896x896 (`local_linear_bench.py`, 1 thread) | 137-175 / 114-255 us | 35 / 29 us |
| local linear 7B qkv 3584x4608 / down 18944x3584 (1 thread) | 2.6-4.6 / 25 ms CPU (5.4-8.2 / 45 ms wall) | 1.2 / 10.9 ms |
| noise `guard_packed` per trip, n = 896 ... 18944 | 17-21 us | 2.6-4.2 us |
| one 0.5B decode trip, gate_up / down (`trip_cpu_bench.py`, gateway thread CPU) | 175-231 / 281-350 us | 137-168 / 205-244 us |
| local linears in the decode loop (`decode_cpu_bench.py`, 0.5B, 4 layers, wall per token) | 12.0 ms | 2.6 ms |
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
  now one fixed-structure path in which a dummy call also takes one charged mask row, so every call makes the same store take; a dummy call is now within 0.5% of a 1-real-row call; the per-row take-versus-padding residual (4 real vs 1 real + 3 padding rows) is about 3% per call (~12 us on the 0.5B down linear, so ~0.5 ms per pass) with an in-memory reservation, reveals how many drafts a pass carries, and must be re-measured with the real mask store before production use), and window extension, ahead reservation and part retirement driven by
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
