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

## 2026-11-23 — 7B at 28 tok/s with a CPU-only, weight-free gateway (Modal, H100 provider)

Measured (Modal co-located, Qwen2.5-7B-Instruct, H100 provider with 12 cores, 32-core CPU-only gateway holding no
linear weights, all linears remote, masked LM head on the provider, native float ops, native decode driver, p31
checks, sampled noise checks inline before the next payload):
- **Streamed decode 27.97 tok/s median (18 back-to-back chats, 23.6-29.6), warm single chat 25.6 tok/s, TTFT ~1.9 s.**
- Exact: logprobs bit-identical (max diff 0.0), text equal, stream identical, every sustain chat identical.
- What changed vs the 19.0-19.3 tok/s runs: the gateway<->provider lane runs over the **shared-memory transport**
  (`PGW_LANE_SHM=1` / `PGW_SERVICE_LANE_SHM=1`: an anonymous sealed segment passed over the authenticated UDS
  socket; same frames and bytes), with adaptive reply spin (`PGW_LANE_SPIN_AUTO_US=400`), zero-copy replies and the
  lane's own device lock, the provider's read spin OFF and 12 provider cores. Inside Modal's gVisor sandbox the
  socket path was the cost; shared memory avoids its network stack. Per decoded token: ~15.7 ms waiting on the
  provider + ~10.8 ms gateway work (880 steps: 13.85 s remote, 9.51 s local).
- Masks: consumption 31.4 slots/s (2.71 slots per generated token, prompts included; incremental reservations +
  prefix reuse would cut it to ~1.2-1.3). The encrypted factory produced nothing inside the 60 s window (stock
  6547 -> 4657, refill not yet triggered), so keep-up at this rate is NOT shown; at ~35 slots/s it needs the factory
  sized accordingly (pks-v3: ~1.8 customer cores measured kernels; or more v2 workers).
- Flags: `launch.py --placement colocated --gpu H100 --model qwen2.5-7b-instruct --gateway-cache --gateway-cpus 32
  --provider-cpus 12 --local-linears off --float-ops native --trip-native --decode-native --masked-head --prime-check
  --gateway-env PGW_LANE_SPIN_AUTO_US=400 --gateway-env PGW_LANE_SHM=1 --gateway-env PGW_TRIP_DECRYPT_AHEAD=1
  --provider-env PGW_SERVICE_LANE_SHM=1 --provider-env PGW_SERVICE_LANE_ZERO_COPY=1 --provider-env
  PGW_SERVICE_LANE_OWN_LOCK=1` (plus the factory sizing env of the README's CPU-only 7B command).
## Update 2026-11-23: fused2 (2 trips per layer) quality gap closed; exact e2e harness ready for one 7B run (opt-in)

Diagnosis: the whole top-1 gap of fused2 (93.9% vs 96.9-97.1% for the int8 reference, 4k wikitext tokens) is the
int8 rounding of the folded weights `W diag(g * sw1)` (with them in float fused2 reaches 97.2%), spread over all
layers; the folded column scales spread 7x (up to 32x), so most columns get a 2.2x (up to 8x) coarser int8 grid.
Fix: **residual-grid migration**: public per-channel integers `hmul` (pack time, from public weights only,
`migration_multipliers`, 8 bits); the provider returns `H = R + hmul * (W1 x)` and the gateway keeps the residual
on the finer grid `sw1 / hmul`, so the folded matrix sees near-uniform column scales. Exact, same trips, same
elements and factory products, masks `r_R = v - hmul W1 r_x` (uniform), stage-1 check with `u * hmul`; costs ~8 bits
in `H`/`Z`: ring **K56A12C53** (7 bytes per element). Qwen2.5-0.5B, engine, 4088 tokens: top-1 = float **96.8%**
(int8 reference 97.1%, old fused2 94.1%), ppl 19.86 (int8 19.83, float 19.93), prompts 96.9% (= int8), every row
at 12 activation bits. Qwen2.5-1.5B (simulator, 2044 tokens): 98.0% (int8 97.7%, old fused2 96.5%).
Tests: migrated masked == plain bit for bit, tamper (H, consistent H, Z, Z without residual, a provider skipping
the multipliers), socket path, prefill == lazy masks.

E2E: `pgw.engine.fused2_remote` (the pair op as its own process on a 0600 Unix socket, CUDA ring products) +
`benchmarks/fused2_e2e.py` + `deploy/modal/fused2_e2e.py` (one container; masks from a gateway-side plain factory;
plain replay must match). Local 0.5B check: 49 trips/token, bit-identical, 1 slot per token (same elements as
today), 3.4 MB per token at K56. It does not use the production runtime, store, HE factory, lane, native decode
driver, p31 or the masked head (fused2 has no path there yet; list in `research/plans/fused2-two-trips-per-layer.md`
section 8 and `deploy/modal/README.md`). Defaults off; the gateway still refuses `schedule = "fused2"`.
## Update 2026-11-23: long contexts (prefill and decode): bounded attention scratch, fused/tiled kernels, chunked prefill

**Finding.** Real masked path, Qwen2.5-0.5B, in-process exact provider, 4-vCPU box: prompt passes of ~500-800 rows
took 88-138 s against 2.9 s for 228 rows. Reproduced (512 rows: 50-277 s). Two causes:

1. **Not the gateway.** In those passes `block_wait` (waiting on the in-process CPU provider) was 46 of 50 s, 150 of
   157 s, 268 of 278 s; the gateway's own work (attention and norms, mask takes, prepare, finish) was ~5 s for 512 rows
   and grows linearly. The provider's float64 BLAS GEMMs (2.9 GB of float64 weights, 4-thread MKL) take 3 s in
   isolation for 512 rows x 24 layers but were busy 64-288 s in the pass: two pipelined prompt blocks put two
   multi-threaded GEMMs in flight next to the gateway's threads, on a box shared with other sessions (load average
   6-11 on 4 vCPUs during the runs; the same GEMM alone went 125 -> 1621 ms per layer while another tenant ran).
   `LocalComputeClient` now runs one request's GEMMs at a time (same bytes); the rest is the test rig's CPU stand-in
   for the GPU service. Also the plain path: the one-time int8-vs-float64 timing probe could pick float64 for the
   whole process when the box was busy for a moment (plain linears ~5-10x slower at 128 rows). It now measures
   interleaved, best of 15, and re-measures a slow verdict (up to 4 rounds, ~1 s worst case, once per process).
2. **The gateway's native prompt attention was O(T^2) in memory.** One call kept `(rows, nh, start + rows)` float64
   scores: 1.9 GB per layer call at 4096 rows, 7.5 GB at 8192, 30 GB at 16k, 120 GB at 32k. Long prompts could not
   run, and attention was memory-bound (18 ns per score on one thread).

**Fixes (native float ops; each bit-identical, so default on).** Compared against the previous build's outputs on
276 problems (v1, v2 blocks 256 and 16, bf16 KV, threads 1 and 3) at the AVX-512, AVX2 and scalar levels: 0
mismatches. New Rust tests: fused == two-phase, long rows == the original one-row kernel, row tiles == one call.
- Prompt passes run as **fused (row, KV head) units**: phase A over every block, row maxima, phase B and the
  block-order merge in a per-thread scratch of `nh x len` scores (3.7 MB per thread at 32k) instead of one big buffer.
  Each row has the same operations in the same order, and units are independent, so any partition or thread count
  gives the same bits. One thread: 18 -> 9.5 ns per score. Decode-sized passes keep the (row, KV head, block) units.
  That path keeps **row tiles** under a 48 MB scratch budget.
- **Scores** are computed 16 positions at a time with the same per-score trees. Block maxima are vectorised (the
  same maximum). **Several heads per pass over V.** The AVX-512 value pass runs in **tiles of 256 positions**, with
  the accumulators and dens carried in memory between tiles (exact f32/f64 stores), so V is read from DRAM once per
  KV group instead of once per head batch.
- Decode-sized passes over >= 4096 positions use the attention pool when threads are automatic. The bits do not
  depend on the thread count.
- **KV cache**: a growth step within 25 % of the limit allocates the limit at once. The first decode step after a
  long prompt no longer copies the whole cache into a doubled buffer (2x peak).
- **Chunked prefill** (`prefill_chunk_rows` / `PGW_PREFILL_CHUNK_ROWS`, **default off**; native float ops only,
  validated): a pass in which a sequence has more than `c` new rows runs as sub-passes of at most `c` rows, each
  through every layer. Logits, KV and linear rows are bit-identical. Gateway memory is bounded by the chunk (masks,
  linear inputs and outputs). The provider sees the same rows and bytes per linear in total, but in
  `ceil(rows / c)` trips whose sizes depend on the public prompt length and this public setting. That changes the
  trip pattern, so it is opt-in. 2048 is a reasonable value.

**Gateway prompt pass vs length** (masked path, zero-cost stub provider, so this is gateway work only: synthetic
0.5B shapes, 2 layers, `native`; same box load for before and after; x12 for 24 layers). Wall s (attention + float
part) / peak RSS:

| rows | before | after | after, chunk 2048 |
|---|---|---|---|
| 128 | 0.17 (0.01) / 0.62 GB | 0.15 (0.01) / 0.62 GB | |
| 1024 | 0.55 (0.14) / 1.01 GB | 0.47 (0.10) / 0.81 GB | |
| 2048 | 1.25 (0.48) / 1.52 GB | 1.01 (0.25) / 1.05 GB | |
| 4096 | 3.34 (1.84) / 3.34 GB | 2.57 (0.99) / 1.74 GB | |
| 8192 | not run (7.5 GB scratch per call) | 6.41 (3.43) / 2.33 GB | |
| 16384 | impossible (30 GB) | 19.3 (13.3) / 3.90 GB | 19.5 (13.2) / **1.26 GB** |
| 32768 | impossible (120 GB) | | 58.9 (47.5) / **1.54 GB** |

Everything except attention is linear in rows: mask takes, prepare and finish together come to ~0.45 ms per row per
2 layers. Attention is quadratic: on this box a 0.5B 32k prompt is ~570 s of attention over 24 layers (~10 ns per
score per thread). The units are independent, so this scales with gateway cores. Projected 7B (per layer x4 for
28 heads x 128 vs 14 x 64, 28 layers): a 16k prompt is ~750 s of attention on 4 vCPUs. Long 7B prompts need a
many-core gateway, or prefix reuse (`prefix_cache = "conversation"`) so they are paid once. Plain exact-integer
passes (real 0.5B): the gateway float part at 4096 rows went 13.0 -> 9.0-12 s. The plain linears dominate and vary
3x run to run on this shared box.

**Decode attention, ms per token, all layers** (native kernel per layer x layers, automatic threads as the model
uses them; before: 1 thread for decode; same load). Before -> after:

| context | 0.5B v1 f32 | 0.5B v2-bf16 | 7B v1 f32 | 7B v1-bf16 | 7B v2-bf16 |
|---|---|---|---|---|---|
| 1k | 3.7 -> 2.2 | 3.2 -> 2.6 | 18 -> 14 | 14 -> 12.5 | 14.5 -> 12 |
| 4k | 26 -> 11 | 15 -> 8.5 | 165 -> 28 | 118 -> 33 | 66 -> 22 |
| 16k | 153 -> 32 | 65 -> 22.5 | 721 -> 142 | 509 -> 112 | 284 -> 101 |
| 32k | 330 -> 59 | 128 -> 39 | 1665 -> 365 | 1257 -> 308 | 754 -> 326 |

One thread alone (the V tiling): 7B 32k v1 1665 -> 667 ms. Decode cost, KV memory and masks are linear in context.
KV is 24.6 KB per token for 0.5B f32 (12.3 KB bf16) and 112 KB for 7B f32 (56 KB bf16), so 32k is 0.8/0.4 GB and
3.7/1.8 GB. Mask demand per decode row does not depend on context. 7B at 32k is memory-bound on this box: the floor
is ~0.2 s per token for 1.8 GB of bf16 KV. Use `-bf16kv` and more gateway bandwidth or cores there; 4-8k contexts
are 22-33 ms per token.

- Tests: `tests/engine/test_prefill_chunk.py`: one pass == chunks (700, 512, 1, n-1 rows; 0-3 threads) == token by
  token at 1500 positions, for v1, v2 (16-position blocks), and bf16 v1/v2; logits and KV bit-identical; a chunked
  pass on a cached prefix plus a batch of two lengths; captured hidden states; masked chunked == plain, with
  provider rows <= chunk and = prompt per linear in total and mask consumption unchanged; KV limit allocation; row
  tiles == single rows at 3000 rows. `tests/gateway/test_prefill_chunk_gateway.py`: multi-turn chat plain, with
  conversation reuse and strict speculation, bit-identical, with the same per-linear rows and masks consumed, and
  config validation. Existing native, decode-driver, tree, prefix-cache, thread-scope and gateway wiring suites
  pass (346 tests), as do core, factory and maskgen (661).

## Update 2026-11-23: decode driver keeps "alarm before the next payload"; few-row GEMV tuning (opt-in)

- **Native decode driver, noise statistics** (`PGW_DECODE_NATIVE`, still opt-in): sampled noise statistics came back
  once per driver call, so an alarm could surface after the rest of the pass's payloads had left. Now a sampled trip
  ends the driver call and the sampling draws are made one trip at a time (the per-trip path's order: the next trip's
  draw comes after the sampled trip's statistics). With `PGW_NOISE_ASYNC_SAMPLED=1` the sampled trip stays on the
  native lane and its statistics are joined before the next payload; with inline statistics (default) the driver
  hands the sampled trip back after its native prepare, the statistics run, and only then does its payload leave
  (ordinary path, same request id and bytes): the per-trip path's "before this payload". Same payloads, draws and
  statistics as without the driver (existing tests); new `test_sampled_alarm_stops_the_next_payload` (local client
  and native lane, sync and async: the driver stops traffic at exactly the payload count the per-trip path does;
  the previous code failed 6 of its 8 cases). Cost: one more driver call per sampled trip (5 % of trips), plus in
  inline mode that trip's send on the Python path; on this loaded 4-vCPU box (0.5B, 8 layers, `--lane-proc`,
  best of 3 runs) wall ms/token old/new: sync 19.2/20.8, async 21.5/20.9 (run-to-run spread 19-31 ms). Use
  `PGW_NOISE_ASYNC_SAMPLED=1` with the driver for the cheaper of the two.
- **Few-row GEMV tuning** (`PGW_SERVICE_TRITON_GEMV_TUNE_ROWS=1`, default off; `triton_gemv.tune_rows`): before the
  lane captures its 2..8-row CUDA graphs, each (rows, linear shape) times the untuned choice (on an H100 the
  16-row-padded tensor-core `dot` variant: 2-row graph trips were 168 us vs 89 us for 1 row, gate_up 389 vs 191 us)
  against memory-bound `stream` tiles (each weight byte loaded once for all rows) by CUDA-graph device time, after
  each candidate passed the exactness check against the digit-GEMM path on the device; a candidate replaces the
  baseline only if 3 % faster. The graphs then capture the pick and pass their own byte-equality self-check as
  before. One-row calls are never touched. Same bits (wrapping u32 arithmetic in every variant). CPU-tested in the
  Triton interpreter (`tests/core/test_triton_gemv.py -k tune`); **not yet run on a GPU**. H100 check (~1.5 min,
  under $0.15): `PGW_MODAL_APP_SUFFIX=-mr modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 5
  --cmd "python benchmarks/provider_scale_bench.py --presets 7b --tp 1 --rows 1,2,4 --iters 200 --modes graph
  --tune-rows --json /tmp/ps.json"` (`graph_report.tuning` has each shape's baseline and chosen device ms; the
  graph trip means use the picks; `device_us` is the graph-timed kernel alone).
## Update 2026-11-23: incremental mask reservations (opt-in), toward ~1.1-1.3 mask slots per generated token

Mask demand per generated token = prompt-pass rows (re-sent every turn) + decode rows + burned slots. The 7B e2e
measured about 2.7 (plain). `prefix_cache = "conversation"` (9.4b) removes the re-sent prompt rows;
**`mask_reservation = "incremental"`** (gateway setting, default `"window"`; `mask_chunk_slots`, default 8; SPEC 9.4c)
removes the burn: the prompt plus one public chunk up front, whole chunks reserved ahead on a background thread,
retired chunks closed off the token step, `< 1.5` chunks burned per request (finished, cancelled or failed) instead
of `max_tokens - generated` (up front) or up to 1-1.5 windows of 64. Admission keeps the up-front guarantee in
memory (`EntitlementLedger`: the store must cover every live request's remaining chunk-aligned prompt + max_tokens
plus one spare chunk), so extensions do not run short mid-answer; if one fails anyway the step fails closed before
any row is taken or sent and every part is burned. Privacy: unchanged in every provider-visible channel (calls,
row counts and order equal an up-front reservation's, chunks are multiples of `spec_rows` in strict mode; store
transactions happen at public pass indices; mask consumption, which drives the encrypted factory's `maskgen`
cadence, becomes rows used + `< 1.5 c`, no longer `prompt + max_tokens`, so `max_tokens` leaves that channel).
Strict speculation with `spec_mask_rows` keeps its public `charge_to` schedule (chunks as windows).

- Tests: `tests/gateway/test_incremental_masks.py` (bit-identical answers and logprobs vs up front, plain and strict
  R = 3, multi-turn with prefix reuse; provider calls identical call by call; consumption = rows used + < 2 chunks;
  admission refuses what live entitlements need; mid-answer exhaustion and a fault after the durable commit end the
  answer with an error, no partial pass, nothing reused, next answer exact; cancel burns every part; 4 concurrent
  tenants exact; default off), `tests/engine/test_trip_native.py::test_incremental_chunks_equal_plain_and_bytes_identical`
  (chunked chains with and without `PGW_TRIP_NATIVE` leases: masked == plain).
- 0.5B, real masked path (fsync store, file counter, every linear remote, in-process exact provider;
  `benchmarks/incremental_masks_bench.py`; Qwen2.5-0.5B-Instruct, 4-turn chat, max_tokens 256, greedy; same answers
  in every configuration): window 64 without reuse **2491 slots / 868 tokens = 2.87** slots/token (up front: 2619 =
  3.02; prompt rows alone 1.84); window 64 with reuse **1056 / 868 = 1.22** (up front with reuse 1184 = 1.36). The
  answers here mostly ran to max_tokens and the 64-windows happened to end close to the answers (burn 5, 1, 1, 25),
  so the window burn was small in this run; its expectation is ~window/2 per request on a plain store and ~1 window
  with the tiered store's ahead reservation. Incremental, measured turns: burn 5 and 9 slots (chunk 8, one answer
  at max_tokens: the spare chunk). Incremental with reuse = rows0 + generated - 1 + burn per turn (asserted by the
  tests): **~1.21-1.23 slots/token** for this chat, independent of window alignment and of max_tokens. The
  incremental-with-reuse run itself was killed by the shared box (load average high, factory at 4-6 slots/s), as
  was a 5-turn run; TTFT and decode times on this box were dominated by other load (decode 290-440 ms/token in all
  configurations, TTFT equal within noise for the same prompt-pass rows), so no timing difference is claimed.
- Projected 7B (plain decoding, chat answers of ~150-250 tokens, new message + template ~20-40 tokens per turn):
  1 (decode) + 0.1-0.25 (new prompt rows) + ~8/200 (burn) = **~1.15-1.3 slots/token**, vs 2.7 measured; with strict
  R = 2 about 2.1-2.3 (vs 4.2), with `spec_mask_rows` C about C + 0.15-0.3.
- Costs: one store transaction (reserve) and one close per chunk, both on background threads; a part re-decrypts
  the `W r` chunk its first slot lies in (1 MiB segment chunks: ~6 slots at 7B K32, so about 0.75 x one slot of
  extra AES-GCM per token at c = 8, ~0.4 ms on one core at the 11 GiB/s measured here; c = 16 halves it).
- Default: proposed as the default once it has run on the 7B co-located e2e (the ledger is stricter than the
  window mode's stock check: a store that covers the stock but not every live request's max_tokens refuses new
  requests instead of risking mid-answer exhaustion; deployments with large max_tokens need correspondingly more
  stock, as up front). Not done: chunk handoff of the decrypted segment chunk between consecutive parts.
## Update 2026-11-23: 1-row trip fixed cost: lane latency knobs, shared-memory lane transport (opt-in, CPU-tested)

Toward <= ~0.15 ms all-in per 1-row trip at 7B (today about 0.46 ms, device 0.04-0.09 ms). All default off; same
bytes, order, tickets, peer-uid check; what the provider sees is unchanged. Details, measurements and the Modal
commands: `deploy/modal/README.md`, "1-row trip fixed cost".

- Gateway (`rust/pgw-fast/src/lane.rs`, used by the native decode driver and the per-trip driver unchanged):
  adaptive reply spin (`PGW_LANE_SPIN_AUTO_US`: per (linear, rows) learned reply time x 1.25 + 20 us, capped), UDS
  spin polls by `recv(MSG_DONTWAIT)` (no `fcntl` per trip), `TripChannel.stats()` (spin hits, blocking waits,
  wait/trip time, the service's `queue_us`/`compute_us`), `LaneComputeClient.channel_stats()`.
- Provider (`rust/pgw-lane`): `queue_us` is now the device wait only (not the body read);
  `PGW_SERVICE_LANE_ZERO_COPY` (reply written from the graph's pinned `h_out`, non-blocking under the device lock,
  remainder after it); `PGW_SERVICE_LANE_OWN_LOCK` (lane graph replays never queue behind Python-path calls on the
  device lock; captures hold both locks); `PGW_SERVICE_LANE_READ_SPIN_US` (existing) recommended for the e2e.
- Shared-memory transport (`rust/pgw-lane/src/shm.rs`, `PGW_SERVICE_LANE_SHM` + `PGW_LANE_SHM`): a `shm` frame after
  the ticket `open`; the service passes a sealed, path-less memfd over the authenticated socket (`SCM_RIGHTS`);
  same headers and bodies in the segment, the socket as doorbell; the service copies and validates every frame
  as on the socket. A start-up race (first request stored before the service read the sequence) was found by
  the bench and fixed. **Not a win on this box** (floor 14.5 vs 12.3 us p50 for UDS); kept opt-in for the H100
  micro-benchmark to decide (gVisor syscalls/wake-ups are where it could pay).
- Local results (overloaded 4-vCPU box, medians): UDS transport floor 12.3 us p50; zero-copy 11.9; with H100-like
  device times and 140 us of gateway work between trips the overhead p50 28.7 -> 21.1 us with the adaptive spin
  (blocking waits 50 % -> 30 %); native decode driver `lane_us_per_trip` ~102 -> ~85 us. Spin-heavy variants
  (server read spin, shm) lose in the mean here because the spinners are preempted (p99 2-2.6 ms).
- New: `benchmarks/lane_gpu_trip_bench.py` (real lane trips on a GPU: variants x Python-path load x shared/own
  lock, exactness checked); `lane_tls_bench.py --stub-us/--zero-copy/--spin-auto-us/native-shm`;
  `decode_cpu_bench.py --spin-auto-us`; tests `tests/service/test_lane_shm.py`.
- Not run on a GPU. Next: the H100 micro-benchmark (~$0.1), then the 7B e2e with the chosen flags.

## Update 2026-11-22: native UDS lane channel (opt-in), persistent GPU worker's first H100 run

Toward the 7B CPU-only-gateway target (25 tok/s, <= ~0.35 ms per trip all-in; provider wait per trip today ~0.38 ms).

- **`lane_native`** (gateway model option, default `false`; `SETUP_MODEL_OPTION_KEYS`): the UDS lane (`lane = "auto"` /
  `"required"`) uses the native `pgw_fast.TripChannel.connect_service_uds` (A1 frames, GIL released for the whole
  trip, one `writev`, spin-then-block wait) instead of the Python socket channel; without the extension the Python
  channel as before. Same header fields and body bytes, same single-use tickets and peer-uid check by the service, same
  status mapping (closing statuses drop the lane and cool down), same byte-identical resend under the same request id on
  the stream channel, same `lane_stats` (`native: true`). It is what the per-trip native driver (`PGW_TRIP_DRIVER`)
  needs on the UDS lane (on the Python channel the driver stays off). Tests: `tests/gateway/test_speed_wiring.py`
  (`test_lane_native_over_the_service_uds_lane_is_bit_identical`, with and without `PGW_TRIP_FAST`/`NATIVE`/`DRIVER`:
  outputs and logprobs equal to the plaintext reference, every lane call counted by the service, the driver served
  trips). Local transport round trip (`benchmarks/lane_tls_bench.py --modes native-uds,py-uds --preset 7b`, stub
  executor, this shared 4-vCPU box): native-uds p50 22 us / mean 35 us, py-uds p50 23 us / mean 25 us: the socket hop
  itself is ~20 us either way, so the ~0.2 ms "wire" of the H100 run is not the channel; it is the gateway process
  around the call (GIL hand-offs: with 2 GIL-busy threads both channels' means rise to 0.4-0.6 ms) and, on the
  provider, the device-lock wait, which the lane reports as `queue_us` (not in `compute_us`, so the client counts it
  as wire). The native channel's gain is therefore expected from the driver (one native call for lane + finish).
- **Co-located demo**: `launch.py --lane uds` (the co-located default) now sets `lane_native` (opt out:
  `--lane-python`), and with `--trip-native` also `PGW_TRIP_DRIVER=1`.
- **Persistent GPU worker, first GPU run** (`deploy/modal/gpu_bench.py --target h100_c4`, new: H100, 4 cores, 12 min
  hard timeout; `provider_scale_bench.py --presets 7b --tp 1 --rows 1,2 --iters 300 --persistent-bps 1,2`; 40 s wall,
  ~$0.1): NVRTC compile, launch and the startup self-check all passed, and every timed persistent trip was **exact**
  (0 mismatches), but it was **slow: 2.94 ms per 1-row trip (5.66 ms with 2 blocks/SM)** vs graph replay 0.089 ms and
  the kernel path 0.296 ms. 1-row service time per 7B trip (us, mean): graph qkv 44 / o 40 / gate_up 191 / down 79
  (trip mean 89; rows 2: 168); kernel path 257 / 256 / 399 / 270 (296; rows 2: 351); persistent 2802 / 2799 / 3136 /
  3027. Cause (cost proportional to resident threads): every thread executed `__threadfence_system()` twice per trip.
  Fixed (one system fence per block after `__syncthreads`, the second grid barrier and the `d_y` round trip removed:
  warps store their outputs straight into the mapped reply; trip words read once per block; 16-byte input reads;
  4 rows per warp when 16 would idle warps): **not yet run on a GPU** (a second short run was not approved in this
  session); arithmetic unchanged (u32 wrap), and the start-up self-check still guards it. Not wired into the demo
  (it was not faster). Caveat for serving: while the resident kernel runs, any device-wide synchronisation on that
  GPU (`torch.cuda.synchronize`, `torch.cuda.graph` capture, `empty_cache`, Triton autotuning) blocks forever, so a
  provider with `PGW_SERVICE_EXEC=persistent` must not capture graphs or tune lazily after start-up.
  Graph replay's gate_up (191 us for 136 MB, ~0.7 TB/s) is the largest single provider cost per token.
  Re-run (~$0.1): `PGW_MODAL_APP_SUFFIX=-pw modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 6
  --cmd "python benchmarks/provider_scale_bench.py --presets 7b --tp 1 --rows 1,2,4 --iters 300 --persistent-bps
  1,2,4 --json /tmp/ps.json"` (the bench now records a failing path's error and still reports the others).

## Update 2026-11-23: 7B feature draft head trained on Modal (export bug found; gateway opt-in `draft_backend="feature_head"`)

Qwen2.5-7B-Instruct features (1.21M rows: 2,582 greedy UltraChat answers + 1,261 dataset replies; one A10G job,
~$0.75): the float head reaches validation top-1 0.46 (depth 1), but the exported int8 heads hold alpha 0.19 on
85 held-out answers because one per-row scale over `W_in`'s feature and token halves zeroed the feature half.
Exporter and loader fixed; a train-only rerun from the saved features (~$0.35) is pending approval. Gateway
loads a head file (`draft_backend = "feature_head"`, `draft_head_path`; strict speculation only, defaults off),
bit-identical to plain decoding in tests; 0.6-0.75 ms per head step on CPU (d = 384). Details:
`research/plans/tree-speculation.md` section 9.
## Update 2026-11-22: Qwen2.5-32B co-located Modal run prepared (CPU-only gateway, H100 provider; not run)

`launch.py --model qwen2.5-32b-instruct` (H100 only, `--gateway-cache` required): the provider's 37.5 GB pack from a
provider-side Volume (`prepare_provider_pack`, CPU container, download + pack inside Modal), the gateway side's pack and
114.4 GB hint (not about 78 GB: 4.65M output rows x 24,576 bytes) built once by `--prepare-cache` on a customer-side T4
container (hint on the T4, customer-chosen seed; never on the provider), a free fingerprint preflight before every run,
and a fail-fast in the container without a cached pack and hint. Sizing: provider 48 GiB, gateway 160 GiB (hint and
pack in the page cache), slot 18.6 MB, keep-up divisor 16; plaintext reference's exact integer products on 16 threads
(`PGW_MODAL_REF_INT_THREADS`, same bits); `--mask-source gpu-service` hook for the GPU mask service (refused until
wired); `--dry-run`. Commands, sizes and cost (prep about $1-2 once, short plain e2e about $1.5-2.5 at $7.51/h, TTL 30
min; expected plain 3.5-5 tok/s): `deploy/modal/README.md`, "Qwen2.5-32B co-located".
## Update 2026-11-22: provider for 32B/70B-class: output-row TP, persistent trip worker, lane coalescing (opt-in, CPU-tested)

Provider side of the scaling plan (targets 3-4); what the provider computes and sees is unchanged (masked `x_hat` in,
`y_hat` out, same bytes), and every knob defaults off.

- **Output-row tensor parallel** (`tp_devices`, `PGW_SERVICE_TP_DEVICES='["cuda:0",...]'`):
  `pgw.service.tensor_parallel.ShardedKernel` splits each linear's int8 weight by output rows (balanced, 16-aligned
  boundaries) over N devices in one process; the input is broadcast, shards run concurrently (caller thread + per-group
  helpers, each shard on its own device executor, stream, pinned staging and CUDA graphs), slices are concatenated in
  row order: byte-identical to one device (no reduction). Maskgen entry points concatenate row blocks too. A TP group
  is one scheduling unit for the batcher. A failed shard fails the call only after every shard finished (no partial
  reply). Lane: per-(linear, rows) graphs per shard reading one pinned input, registered as one lane graph; the native
  lane launches all shards and gathers rows in order (`cuda.rs` `GraphShard`, `gather_rows`), self-checked at startup.
- **Persistent trip worker** (`PGW_SERVICE_EXEC=persistent`): `pgw.service.persistent` (NVRTC-compiled resident kernel
  per GPU: block 0 polls a doorbell in mapped pinned memory, grid copies x, wrapping-u32 ring GEMV with 16-byte weight
  loads, coalesced writes of its output columns straight into the shared mapped reply buffer, `done` after a system
  fence) and `rust/pgw-lane/src/persistent.rs` (per trip: device locks, one body copy, doorbell per shard, spin on
  `done`; no launch, no Python). Rows 1-4; larger calls fall to graphs. Startup self-check vs the kernel path; a
  mismatch, an error status or a worker that does not answer disables it (graphs / kernel path serve).
- **Multi-tenant coalescing on the lane** (`PGW_SERVICE_COALESCE=1`): `rust/pgw-lane/src/coalesce.rs` merges calls of
  the same (model, linear, ring) that wait while that key runs into one device call (FIFO, <= lane_max_rows rows); an
  idle key runs at once and alone (window 0, no copy). Replies are cut per call; admission and per-tenant accounting
  stay per call; a failed merged call retries each member alone. (The stream/Python path already coalesced across
  tenants in `BatchScheduler` with window 0.)
- Tests (CPU; fake multi-device = repeated `cpu`, emulated workers on the same memory protocol, the real native lane):
  `tests/service/test_tensor_parallel.py`, `tests/service/test_lane_persistent.py`; Rust: `cargo test -p pgw-lane`
  (coalescer, persistent executor with device-emulating threads incl. TP gather, stall and error disable, graph shard
  tiling). Not reached: nothing ran on a GPU; the CUDA kernel, the TP graph capture and the timings are unverified.
- Ready to run (not run, no Modal spend): `modal run deploy/modal/gpu_bench.py --target h100x4_c32 --timeout-min 30
  --out provider_scale.json --cmd "python benchmarks/provider_scale_bench.py --presets 7b,32b,72b --tp 1,2,4
  --rows 1,2 --iters 400 --json /tmp/provider_scale.json"` (4 x H100, about $16/h plus CPU; ~10-15 min, ~$3-4).
  A 1-GPU subset (`--target h100_c16 --tp 1`) is ~$1. Reports per preset, TP and rows: kernel / graph / persistent
  service time per trip, the GEMV device floor and the overhead above it (target <= 15-25 us; gather <= 5 us).
## Update 2026-11-22: standard deployment in the customer's cloud account; GPU mask service (opt-in)

Decision: the standard deployment runs, in the **customer's own cloud account in the provider's region**, a CPU-only
gateway VM and, for 32B+, a small GPU "mask service" VM that computes the masks `W8 r` itself from the public
weights (no HE, no hint). Up to 14B the CPU-only encrypted factory stays the default. Premises run only a thin client.

- **`pgw factory mask-service`** (`PGW_FACTORY_MASK_SERVICE`, default off; SPEC 7.10; `pgw.factory.mask_service`):
  the existing plain factory (cuda_int8, GPU ChaCha20, per-chunk self-check) plus a **start-up check** (a throw-away
  segment on the production device and chunk size; sampled `W8 r` rows and the full `u W8` of sampled linears
  compared bit for bit with an independent CPU reference; any difference refuses start-up), **device scrubbing**
  after every segment, and refusals (encrypted generation, no self-check, CPU without `mask_service_allow_cpu`,
  unsigned or unbound export). Delivery: the gateway's store on the same host, or the partner export channel (HPKE +
  Ed25519 + destination store) over a small courier, `pgw factory outbox-serve` / `pgw maskstore pull` (unchanged
  import checks; delete local then remote; back-pressure).
- **Freivalds precomputation on the int8 kernel on CUDA** (`ring_vecmat(method="kernel")`, the CUDA default): `u W8`
  through the same digit-plane GEMM as `W8 r` instead of float64 GEMMs (1/32-1/64 rate on T4/L4/A10G). Same bits
  (tests; the start-up check compared `u W8` on the T4); the factory's per-segment check stage at 72B fell from
  0.76 to 0.05 s per layer (ring).
- **Exactness evidence**: CPU tests (rings K32/K40/K48, p31, device-PRNG emulation, corrupted `W r` / `u W` refused,
  buffers wiped, same-host store and export -> pull -> import -> reserve exact); on a Modal **T4** every start-up
  check passed (all linears of the timed layers, ring and p31, `gpu_prng=True`), and the CUDA kernel/mask service
  tests passed (83). Spend about $0.15.
- **Rates** (full model, extrapolated from synthetic layers, 1024-slot segments into an encrypted store; T4): 7B
  164-174, **32B 55-58, 72B 30-40 slots/s** (72B 47-50 with 4096-slot segments, generation only; before the vecmat
  change 32B 27-30, 72B 13-14). Estimated L4/A10G (1.5-1.9x T4 int8): 32B 85-110, 72B 60-90 slots/s vs decode
  demand 33-40 and with-prompts ~90 (scaling plan section 6). This box (4 shared vCPUs, load average ~40) as the CPU
  fallback, Qwen2.5-0.5B pack, cpu_int8, p31: 12-29 slots/s (34 slots per CPU-second); start-up check 200-245 s here
  (one 259-slot throw-away segment over all 96 linears at this box's rate).
- **`deploy/customer-cloud`**: Terraform module for AWS (region/zone, VPC or existing subnet, PrivateLink endpoint to
  the provider, gateway and GPU instance types, model sizing presets, mask supply `cpu_encrypted` / `separate_vm` /
  `same_host`, enrollment code as a sensitive variable in an SSM SecureString read once), security groups (client
  API TLS 8443 from given CIDRs only; gateway -> provider endpoint; gateway -> mask service 8470; no SSH; IMDSv2;
  encrypted volumes; separate counter volume), first-boot scripts (enroll, build the public pack, exchange only
  public keys and certificates via SSM, start containers), sizing and cost table (7B ~$0.85/h CPU-only; 32B ~$2.95/h
  with an L4; 72B ~$5.10/h), "Onboarding in about an hour", GPU-quota and region notes, teardown. `terraform fmt` and
  `validate` pass (Terraform 1.9.8, aws 6.73); `tests/deploy/test_customer_cloud_template.py` checks the contract
  statically. Not applied to any cloud account.
- **Privacy**: unchanged guarantees, now with everything secret in customer-controlled VMs: the provider never gets
  `r`, seeds or `u`; the weights the mask service uses are the provider's public model already; the mask service has
  no route to the provider.
- Not done: the gateway's in-memory mask ring is not fed by the mask service directly (segments go through the
  encrypted store, whose volume bandwidth binds at 72B); no real GPU instance (L4/A10G) measured; no cloud apply.
## Update 2026-11-22: parallel, deterministic gateway attention (v2 "blocked") and bf16 KV storage (opt-in, local)

The bound at 32B-72B (gateway attention over the private KV cache) now has a parallel native kernel. Defaults are
unchanged (`float_ops="torch"`; `"native"` keeps its bits). Code: `rust/pgw-fast/src/attn.rs`,
`NativeFloatOps` / `FLOAT_OPS_NAMES` in `engine/model.py`; tests: `tests/engine/test_native_attention_v2.py`,
`attn.rs` unit tests; benchmark: `benchmarks/attn_cpu_bench.py`.

- **Engine**: units (row, KV head, 256-position block) run on a persistent pool. Per-block partial sums (den in
  float64, acc in float32, each from 0 in position order) are merged in block order. The bits therefore do not
  depend on the thread count, the ISA (explicit AVX-512 and AVX2 kernels: the same 16-lane dot and tree, a
  vector `exp_f64`, no FMA) or the rows per pass. `float_ops="native-blocked"` is the new pinned formulation (v2);
  plain `"native"` (v1) runs on the same engine with one block per row: the same bits as before, now also
  parallel over KV heads. v2 equals v1 whenever the prefix fits in one block (< 256 positions). `PGW_ATTN_THREADS`
  sets the thread count (0 = automatic, as before).
- **bf16 KV** (`"native-bf16kv"`, `"native-blocked-bf16kv"`; each its own reference): the cache stores bfloat16,
  rounded to nearest even like torch's `.to(torch.bfloat16)`, and widens exactly to float32 when read. Half
  the bytes. Quality on Qwen2.5-0.5B is **not measured yet**. The check `test_bf16_kv_quality_qwen05b` (max |delta
  logprob| of the float32-KV top-1 token and top-1 agreement over about 570 positions, 3 blocks; also v2 vs v1)
  is written, but the memory cgroup killed it twice on this shared box. On the tiny models, bf16 KV stays within
  2e-2 of HF.
- **Exactness, tested**: 1-7 threads and ISA caps AVX-512/AVX2/scalar give identical logits; R-row prompt and
  verify passes equal 1-row decoding; masked == plain (tokens and logprobs); speculation identical; HF
  closeness at 1e-4 (2e-2 with bf16 KV); v2 vs v1 within 1e-5; v1 equals the original kernel bit for bit
  (Rust test against the kept `attend_row_ref`). 74 engine tests pass with the new extension.
- **Speed** (this box: 4 shared vCPUs with load average 45-52 from other jobs, so wall clocks are about 3x CPU time
  and thread scaling cannot be measured cleanly; numbers are single-thread CPU time per layer, decode row):
  72B at 2k context, v2: 4.3 ms float32 KV (15.6 GFLOP/s, 3.9 GB/s per core), 3.3 ms bf16 KV (20.5 GFLOP/s); v1
  engine: 8.1 ms. The pre-SIMD build of the same engine took 11.7 ms. 8k context: 19.6 ms float32 / 15.3 ms bf16
  (v1 36.2 ms). 7B at 2k: 2.0 / 1.6 ms. The best 4-thread wall time was 2.5x the 1-thread time (7B at 512 and
  2k). **Extrapolation** at 15-20 GFLOP/s per core: 600 GFLOP/s needs about 30-40 cores, so the target is not
  shown on this box. A 72B row at 2k has 64 units, and 256 at 8k: enough parallelism for 48 cores. Next: measure
  on an idle 16-48 core VM (`attn_cpu_bench.py --threads 1,8,16,32`).

## Update 2026-11-22: exact KV-cache reuse across chat turns (`prefix_cache = "conversation"`, opt-in)

SPEC 9.4b. A request whose prompt (after the chat template) starts with tokens of the previous turn of the same
conversation key (API key + OpenAI `user`, optionally narrowed by `x-pgw-conversation-id`) runs only the rest
through the prompt pass and reserves masks only for it (`(prompt - reused) + max_tokens`; strict speculation: the
same decode term, unchanged pass schedule). Longest-common-token-prefix match; anything else recomputes.
- **Exact by construction, refused elsewhere**: needs `float_ops = "native"` (row-invariant per-row kernels) and a
  `rows_exact` backend (`PlainBackend`, `MaskedBackend`); torch float ops, `mode = "float"` and GPU targets are
  refused by the config and by `Generator.prefix_reuse_supported()`. Tests (`tests/engine/test_prefix_cache.py`,
  `tests/gateway/test_prefix_cache.py`): token ids and logprobs bit-identical to recomputing over 3-5 turns for
  greedy, seeded sampling, variable-row draft and lookup speculation, strict speculation (masked filler and padded
  `spec_mask_rows`), masked and plain, chat and streaming through the gateway; partial/wrong prefix, cross-key /
  user / header isolation, LRU budget, TTL (sweeper thread), wipe, integrity failure (tampering service) and failed
  deferred check discard, cancelled turns not stored, masks and provider row counts per turn.
- **Security**: gateway RAM only (never spilled), one entry per key, taken exclusively, zeroed on eviction /
  expiry / mismatch / refusal, stored only after a clean end with the session's deferred checks joined
  (`release_session_checks` now returns whether they passed), all entries wiped on any integrity failure or noise
  alarm (epoch: running requests cannot store afterwards). New provider-visible signal, stated in SPEC 9.4b: a
  continued turn's prompt pass has the uncached rows only, and timing shows it continued a cached conversation
  (hit/miss bit); kept opt-in and not a wizard preset. Not turned on in the co-located demo: its exact e2e needs
  about 7.5 GB of disk for the 0.5B hint (4 GB free here); the gateway-level masked e2e tests above stand in.
- **Measured** (`benchmarks/prefix_reuse_bench.py`, Qwen2.5-0.5B-Instruct, native float ops + exact integer
  `PlainBackend` on this 4-vCPU box under a load average of about 45 from other jobs, greedy, 6 tokens per turn,
  chat with a system prompt; every turn bit-identical between the two runs). The run was cut by the 30-minute job
limit after turn 4 (each turn took 2-2.5 min on this box), so four turns are reported:

  | turn | prompt | reused | prompt rows (provider) | mask slots reserved | TTFT recompute | TTFT reuse |
  |---|---|---|---|---|---|---|
  | 1 | 68 | 0 | 68 / 68 | 74 / 74 | 50.3 s | 50.9 s |
  | 2 | 111 | 73 | 111 / 38 | 117 / 44 | 53.3 s | 41.8 s |
  | 3 | 139 | 116 | 139 / 23 | 145 / 29 | 51.3 s | 27.3 s |
  | 4 | 176 | 144 | 176 / 32 | 182 / 38 | 61.4 s | 26.6 s |

  Turns 2-4: mask slots 444 -> 111 (-75%), prompt rows 426 -> 93 (-78%), mean TTFT 55.3 -> 31.9 s (-42%); turns 1-4:
  slots 518 -> 185 (-64%). TTFT falls less than the rows because each pass here has a large fixed cost under this
  load (about 17 s per decode pass: 24 layers of call overhead plus the 151936-row LM head); with the provider, the
  saved rows also save masks taken, service GEMM rows and wire bytes of the prompt pass. Absolute times are not
  representative of an idle host.
## Update 2026-11-22: multi-core per-trip kernels for 32B/72B vectors (`PGW_TRIP_SPLIT`, opt-in, local CPU)

Component target 1 of [research/plans/scaling-32b-70b.md](research/plans/scaling-32b-70b.md) (prepare/finish of one trip
over many cores), first part. Default off; exact; no format change.

- **Split prepare** (`rust/pgw-fast/src/trip/split.rs`; `TripLease.prepare(..., split=c)`, `trip_prepare_into(...,
  split=c)`): each row is cut into a fixed partition of 2048-column blocks (depends on `n` only) over `c` tasks of the
  persistent pool, in three passes (max/sum x^2; exact integer sum of q^2 per candidate width; quantize with the
  selected width + mask + pack + x_hat + input-half partials). Merges on the calling thread in block order: max, the
  float `sum x^2` (window start only), wrapping integer sums, ring input half mod 2^K, prime input half mod p. Payload,
  `sx`, `x_hat`, `rhs` are bit-identical to the one-thread kernel for every `c` (plans with the float sum of squares,
  `n < 4096` or `c > 16` vectors run the one-thread kernel). The `r` keystream of a split call is filled over the same
  tasks (`fill_ring_split`, same bits).
- **Split checks**: the prime output half (`prime::output_half_split`, `TripTicket.finish(..., split=c)`,
  `trip_prime_output(..., split=c)`) by fixed 8192-column units, reduced partials summed mod p; the unmask/finish uses
  the existing column split of the pool with `c` tasks (exact).
- **r ahead for any rows**: a lease now derives the next trip's `r` for as many rows as the current trip (was 1-row
  trips only), on `PGW_TRIP_R_HELPERS` helper threads (default 1); single use (used only by the trip that takes
  exactly those rows, else zeroed; wiped on detach/close). Measured (`TripLease.prepare` wall, 300 us between trips, min of 38 trips, 72B
  shapes): down R = 1 220 -> 152 us, R = 2 433 -> 252 us; gate_up R = 1 57 -> 32 us, R = 2 108 -> 57 us (r is ~40%
  of a prepare). Medians on this overloaded box were not better (the helper is not scheduled in time).
- Wiring: `PGW_TRIP_SPLIT=c`, `PGW_TRIP_SPLIT_MIN_ELEMS` (default 8192) on the native per-trip path in `backends.py`.
- **Native lane + finish per trip** (`PGW_TRIP_DRIVER=1`, with `PGW_TRIP_NATIVE` and a lane client; default off):
  `TripTicket.lane_finish` sends the payload on the native `TripChannel` and reads the reply straight into the finish
  (one GIL-free native call after `lease.prepare`; no response objects, no zero-filled reply buffer, no copy), via
  `LaneComputeClient.trip_ticket` (same eligibility, pool, refusal/error handling and statistics; a call the lane does
  not serve goes to the wrapped client with the same request, the ticket untouched). Noise-monitor every-call checks
  stay inline between the two calls; sampled statistics unchanged. Test: `test_native_driver_over_the_lane` (masked ==
  plain, byte-identical payloads and noise statistics with/without it, refusals and a dropped connection fall back).
- Benchmarks: `native_trip_floor_bench.py --preset 7b|32b|72b --rows R --cores c [--r-offcore] [--stat min]`,
  `trip_native_bench.py --rows R --split c --r-ahead 0|1` (payload hashes compared across runs); presets for Qwen2.5-32B
  (64 L, 5120/27648, 40/8 heads) and Qwen2.5-72B (80 L, 8192/29568, 64/8), vocab 152064 (`PGW_BENCH_VOCAB` shrinks the
  vocabulary of the synthetic pack for disk).
- Tests: `tests/native/test_trip_split.py` (identical bits for c = 2, 3, 4, 7 vs one thread, ring and p31, genuine
  replies pass, tampered replies fail identically with zeroed outputs, r ahead for 1- and 2-row trips identical and
  single use, non-finite before any row, ticket single use), `test_native_split_equals_plain_and_bytes_identical`
  (engine: masked == plain, byte-identical payloads and noise statistics), cargo `trip::split::tests`.
- Measured single-core kernel cost per 72B trip (min of 150, K32A12C27, 4 vCPU dev box): gate_up R = 2 prepare 39 us,
  finish 79-84 us, p31 output half 41 us; down R = 2 prepare 143-172 us, finish 12 us; qkv R = 2 prepare 41-44 us,
  finish 15 us. Kernel floor per token on one core (r off-core, R = 2): 7B 12.9 ms, 32B 48 ms, 72B 61-64 ms; R = 1:
  8.4 / 30 / 34 ms. `r` (8 bytes of ChaCha20 per element, format-fixed) at 72B R = 2 is about 36 ms per token of helper
  time, so 72B needs about one dedicated helper core for `r` at 25 tok/s.
- **Not reached**: the multi-core speed-up could not be measured here: the box ran at load average 40-57 on 4 vCPUs
  (other jobs), the pool workers were rarely scheduled in time (the calling thread took ~80% of the tasks: 100 vs 121
  us of its own CPU for a 72B down prepare at c = 4) and wall times did not improve. Needs a quiet many-core VM
  (with `PGW_POOL_SPIN_US` raised so the workers stay awake between trips). The <= 20 us Python per trip target is
  not reached: profiled at 0.5B shapes over the in-process reference lane (`trip_native_bench.py --lane --driver
  --profile`), Python self time is still ~50-60 us per trip on this loaded box, mostly `_prepare_trip_lease` /
  `forward_many` / `_finish_lease` bookkeeping, and the driver did not lower main-thread CPU measurably (1.80 vs 1.94 ms
  per 8 trips, noise). A per-layer native driver (prepare -> lane -> finish for consecutive linears with no Python
  in between) was not built.

## Update 2026-11-22: scaling design for Qwen2.5-32B and 70B-class targets (model only, local)

Write-up: [research/plans/scaling-32b-70b.md](research/plans/scaling-32b-70b.md); model
`benchmarks/scaling_model.py` (tests: `tests/benchmarks/test_scaling_model.py`), calibrated on the measured 7B
native-kernel floor and the 7B H100 run (lane 0.26 ms per trip; tau 1.19 at R = 2 -> alpha 0.49). Estimates, ctx
2048, alpha 0.70 (0.49): today's paths 2.1 tok/s at 32B and 1.2 at 72B (single-core attention); engineered
gateway + service + network floors + fast draft: **32B 40 (33)**, 72B 27 (23); plus fused 2 trips per layer:
**72B 33 (29), Llama-3-70B 31 (27)**. The binding component at 32B-72B is the gateway's attention over the
private KV cache (655 KB per context token per pass at 72B), then trips x (RTT + service). Minimum targets:
<= 20 us orchestration per trip and trips split over 16 cores, attention >= 100 GB/s and 600 GFLOP/s, RTT <= 30
us, service <= 15-25 us per trip (persistent kernel), TP 4 with a <= 5 us gather (no all-reduce: output-row
split), draft step <= 4-5 ms; for 70B-class also fused chains. Masks at 72B: 28 MB per slot, 1.1 GB/s decode-only
and 2.6 GB/s with prompts; recommended factory: a plain `W8 r` factory on a customer-cloud GPU (0.2 of an L4, no
hint) instead of today's MLWE factory (~44 cores and a 172-345 GB hint per 72B stream). Not reached: nothing was
built or measured at 32B/72B.
## Update 2026-11-22: strict tree speculation and a few-MB target-feature draft head (engine, opt-in)

`research/plans/tree-speculation.md`. `Generator(spec_tree=[(parent, rank), ...])` verifies a public fixed token
tree of R rows in one pass with tree attention (torch and native float paths, no kernel change), bit-identical
to plain decoding (greedy and seeded), with R rows per pass and the strict L-pass schedule unchanged and fixed
gateway work per pass. `FeatureHeadDrafter` drafts from the target's final hidden state, which the gateway holds
in plaintext (5.5 / 9.7 / 13 MB per draft step at 1.5B / 32B / 72B vs 490 MB for the 0.5B draft). Findings:
for the 0.5B draft (alpha 0.81 on the 1.5B, 0.78 on the 3B) the best tree of <= 8 rows is the chain; a head
trained on local data overfits (held-out top-1 0.13), so a real head needs offline EAGLE-scale training; the
32B/72B model says a 0.5B draft is slower than plain, and 25-30 tok/s at 32B needs 0.3-0.5 ms trips, a 2L+1
schedule, per-row trip work spread over spare cores, and a head at alpha ~0.8 with R = 8.
## Update 2026-11-22: fused2 prototype, 2 provider trips per layer (WP-B4, opt-in, new reference; not promoted)

`pgw.engine.fused2` (default off; the gateway still refuses `schedule = "fused2"`): trip A = o + residual add +
gate_up, trip B = down + residual add + next qkv, RMSNorm's `1/rms` applied by the gateway after the product,
residual in fixed point on the o / down output grid, norm gains folded into the next linear at pack time.
Trips per token 4L (+1) -> **2L + 1** (70B 321 -> 161). Same elements on the wire and in the store; ring K40
(1.25x bytes) or K48 (1.5x); no extra factory products (masks reparametrized, `r_R = v - W1 r_x`); both stages
checked with Freivalds over F_(2^61-1). Plain and masked (in-process factory and provider) backends,
masked == plain bit for bit, tamper tests (`tests/engine/test_fused2.py`). Quality on Qwen2.5-0.5B (1020
wikitext tokens): ppl 23.63 (float 23.67, int8 reference 23.51), but top-1 = float 93.0% vs 96.4% for the int8
reference. **Not no-compromise as measured**; next: a folding variant that closes the top-1 gap, then 7B.
Design, numbers, follow-ups: `research/plans/fused2-two-trips-per-layer.md`.

## Update 2026-11-22: the 7B CPU-only-gateway speculation run is ready to launch (Modal launcher, opt-in)

One co-located run measures Qwen2.5-7B on an H100 provider with a CPU-only, weight-free gateway (every linear remote,
native float ops, native per-trip path, p31 check), strict speculation R = 2 with the Qwen2.5-0.5B draft
(`native_int8`, pipelined m = 0) and, in the same container, plain decoding as the A/B. Command, sizing, cost
(about $6.7/h, $2.5-3.5 per run, TTL 35 min) and the expected outcome: `deploy/modal/README.md`, "CPU-only 7B gateway,
strict speculation, H100 provider". Not run yet at 7B.

- Launcher: `--draft-backend`, `--spec-pipeline-steps`, run-time `--spec-mask`, `--trip-native`, `--prime-check`
  (refused with `--spec-mask`: padded calls refuse p31; unpadded R = 2 costs the same 2 slots per pass), `--ab-plain S`
  (restart the gateway without speculation on the same store and keys, then chats and an S-second window),
  `--gateway-cpus/--provider-cpus/--gateway-mem-gib/--provider-mem-gib`, `--gateway-cache`, `--prepare-cache`.
- Reports per phase: streamed decode tok/s, TTFT, tokens per verify pass, drafter hit rate and step time, trips per
  token and trip split, gateway CPU per trip, slots per token, factory production vs consumption; exactness of every
  phase is part of `ok`. New gateway-local admin counters `speed.spec_stats` / `speed.drafter_stats`.
- Gateway cache (a Volume of public artefacts): the pack (reused only with `PGW_SETUP_REUSE_PACK=1`, library default
  off, and only if it fully verifies to the enrolled fingerprint) and the hint `H = W8 A` (public weights times a
  public matrix whose seed the customer side chose; no secret). The 7B's ~20 min CPU hint build moves to a one-off T4
  job. Building the hint on the provider was rejected: a provider-supplied `H` would allow a selective-failure attack
  on the per-group secrets unless re-derived. The factory's segment log now names the check field.
- Validated on a T4 at 0.5B (self-draft, plumbing only): two runs exact in every phase, every segment p31, cache
  restore 2.7-9.2 s for 8.4 GB, plain-phase restart 10.5 s; about $0.2 spent.

## Update 2026-11-22: p31 extents through native trip leases, W r decrypted ahead, fused duplicate check (W2, opt-in)

- **p31 leases**: `TripLease(..., field=2**31 - 1)` serves prime-field extents on the native per-trip path; the prime
  input half is fused in `lease.prepare` and the output half (range + `u . yq == rhs mod p`) runs synchronously inside
  `ticket.finish` on the trip's `W r` *before* the finish kernel unmasks it (never deferred: no result is used before
  it is verified; a failure zeroes `W r` and the outputs). Leased only to a backend with `prime_check`
  (`trip_lease(..., prime_ok=True)`); without it the ordinary path refuses the extent before any row. Format confined
  to `tripnative::check_iface` (`field`, `sync_only`, `prime_output`).
- **`PGW_TRIP_DECRYPT_AHEAD=1`** (default off): the chunks the linear's *next* lease will take (same rows, in order,
  never past the reservation's extent; also the next extent / the next windowed part) are read, authenticated and
  decrypted on a helper thread with native AES-256-GCM, GIL released (`pgw_fast.chunk_open_into`, byte-identical to
  `SegmentFile.decrypt_into`). Each buffer goes to exactly one lease or is zeroed (close, mismatch, failure); an
  authentication failure raises when the lease needs the chunk (fail closed, no row of it handed out).
- **Fused duplicate check**: `lease.prepare(..., dup=noise.packed_dup_params(n))` computes the monitor's duplicate-row
  fingerprints in the same native call; `noise.guard_packed_keys` records them (same keys, bookkeeping, alarms).
- Measured (4 shared vCPUs, heavily loaded by other jobs; `decode_cpu_bench.py --local off --float-ops native
  --wnorm-frac 0.3 --service-delay-ms 0.25 --masked-head`, 2 layers, extrapolated, process CPU ms per 7B token):
  base native 75 / 134 (noisy), this branch ring 74-75, **p31 leases 59-68**, p31 + decrypt ahead 64-84 (no
  measurable gain here: the bench's 1-row chunks and window parts reserved lazily leave little to prefetch, and the
  helper's GIL hand-offs cost the main thread about what they save). 0.5B: 44-47 ring, 23-50 p31 variants (noise).
  `trip_native_bench.py --preset 7b --masked-head` (trips only, full model): main-thread CPU ring 25-33 ms, p31 24-31
  ms; process CPU ring 40-46 ms, **p31 29-37 ms** (no deferred-check helper). Not reached: <= 25 ms of gateway CPU per
  7B token on this box.
- Tests: `tests/engine/test_trip_native.py` (p31 leases: masked == plain, byte-identical payloads and statistics vs
  PGW_TRIP_FAST, tampering at any position / out of range detected through the lease with the row consumed, no lease
  without `prime_check`, field validation; decrypt ahead: native open == `decrypt_into`, ciphertext/tag tamper
  refused, leases fed from ahead buffers byte-identical over 330 tokens across parts and segments for ring and p31,
  ahead buffers zero after use and on close, tampered ahead chunk fails closed at the lease boundary).

## Update 2026-11-22: one-vector Freivalds check over F_p (package W2, opt-in, local only)

New opt-in segment format `check_field = "p31"` (SPEC 5.1, note `research/notes/prime-field-freivalds.md`):
one secret vector per linear over `F_p`, `p = 2^31 - 1`, checked on the unmasked `xq` / `yq`
(`u . yq == (u W8 mod p) . xq mod p`, plus the range check). Soundness per call `>= 30.99999999` bits with
one vector (needs `2B < p`, true for K32A12C27 at any width) against 21 bits for today's 7 ring vectors;
it also catches a wrong `W r`. Default off: factory `PGW_FACTORY_CHECK_FIELD=p31` writes it, gateways need
`prime_check = true` (`PGW_GATEWAY_PRIME_CHECK`) and otherwise refuse such segments before any row is used;
old segments keep the ring check and both formats run side by side. Kernels (`pgw_fast`):
`trip_prepare_into(..., field=2**31 - 1)` (input half fused in prepare), `trip_prime_output(y_hat, wr, u, rhs,
k, bound)` (range + output half, exact modular arithmetic, AVX-512 / AVX2 / portable). Scope: TripEngine paths,
disk store and in-process RAM ring; speculation padding, non-trip fallbacks, MoE oblivious and `ringshm`
refuse prime extents (fail closed). Tests: `tests/engine/test_prime_check.py` (exactness, single-entry /
structured / adversarial-within-cap tampering, model logits bit-identical, interop, defaults, store round
trip), Rust `prime::tests`.

Measured here (`native_trip_floor_bench.py --preset 7b --wnorm-frac 0.3`, 113 trips, ms per token, two runs):

| | ring, c = 7 (today) | p31, c = 1 |
|---|---|---|
| check vectors read per token | 66.6 MB | **9.5 MB** |
| `uW` per segment at rest / derived `u` in RAM | 23.4 MB / 43 MB | 3.3 MB / 6.2 MB |
| prepare (+ input half) | 5.9-6.5 | 6.1-6.2 |
| prime check (range + output half, synchronous) | - | 2.2 |
| finish | 4.0-4.5 (head checked inline) | 2.5-2.6 (no check) |
| **main core** | **14.7-16.2** | **15.6-15.9** |
| **helper core** (deferred output half) | **6.7-7.4** | **0** |

Per kernel (one row, warm): output half m = 4608: 9.1 -> 3.9 us; m = 37888: 44.6 -> 16.5 us; the 7B head
m = 152064: 376 -> 74 us; prepare n = 18944 with the input half 50-53 us (ring 7) vs 49-51 us (prime). The
total gateway CPU per token drops by the helper core's 6.7-7.4 ms; the check now runs synchronously (never
deferred). Run on a helper core instead (keep `W r` until it returns) it would take the main core to about
13.5 ms. `trip_cpu_bench.py --preset 7b --layers 1 --wnorm-frac 0.3 [--prime]` (Python TripEngine path, real
encrypted store, stub service; calling-thread CPU per call in us, two runs each): qkv 195-223 -> 165-179,
o 155-176 -> 125-154, gate_up 227-267 -> 209-222 (ring: plus its deferred output half on the helper),
down 326 -> 245-269; per layer 904-992 -> 769-799 us on the calling thread and nothing on the helper.

## Update 2026-11-22: factory HE on the gateway GPU (opt-in)

`PGW_FACTORY_HE_ENCRYPT=gpu_ntt` / `PGW_FACTORY_HE_DECRYPT=gpu_ntt` (library and demo default off) run the encrypted
factory's encryption/decryption on the gateway's own GPU (`pgw.hemask.gpuntt`), self-tested and cross-checked bit for
bit against the CPU paths at start-up, CPU fallback on any mismatch or error. Two exact co-located Modal runs
(`--gateway-gpu`, 240 s back to back): factory 2.0 -> 0.9 gateway cores, sustained decode 30.8-31.3 tok/s
token-weighted (median 31.3-31.6, min 25.5-25.7; 1-2 chats below 26 tok/s), keep-up unchanged (consumption 50-51
vs commit-to-commit 46-48 slots/s, stock min 3836-4028). Not the demo default: while the factory runs unyielded its
kernels time-slice with the decode graphs (separate processes; stream priorities do not help across contexts):
local linear 79 -> 108-119 us/call, warm single chat 25-27 tok/s during the stock ramp. The customer target is now
GPU-free, so this stays an opt-in. Details: `deploy/modal/README.md`, "Factory HE on the gateway GPU".
## Update 2026-11-22: native per-trip path, one native call per side of the lane call (package W2, local only)

`PGW_TRIP_NATIVE=1` (with `PGW_TRIP_FAST=1`; library default off). The mask reservation hands the gateway a
*lease* (`pgw_fast.TripLease`, `rust/pgw-fast/src/tripnative.rs`): the next rows of one linear in decrypted `W r`
chunks of one extent (at most `PGW_TRIP_LEASE_BYTES` = 1 MiB / `PGW_TRIP_LEASE_ROWS` = 16 rows), the extent's `r`
stream key and its Freivalds vectors. Per trip: `lease.prepare` (take the row, derive `r`, constant-time
quantize + mask + pack, Freivalds input half) before the lane call and `ticket.finish` (unmask with `W r` read in
place, range check, Freivalds sync or deferred, rescale, wipe) after it. `r` of a lease's next row is derived ahead
on a helper thread (`PGW_TRIP_R_AHEAD`, on by default with the native path). Not reached: <= 20 ms main-core CPU
per 7B token.

- Single use and accounting: the lease position advances under its lock before any secret is derived; while a
  lease is attached the linear's cursor is the lease's (`remaining`, `taken`, burn-on-close unchanged); any other
  path that takes rows of that linear detaches it first; `close()` closes leases (in-flight trips fail closed with
  `MaskReuseError`). Unused lease rows are zeroed with their chunk (never sent, like a dropped prefetch stage).
- Same order and checks as `PGW_TRIP_FAST`: window `ensure` / exhaustion pre-check and the extent's soundness check
  before any row is taken, non-finite activations refused before the row is taken, Freivalds then noise-sampling
  draws, every-call noise checks inline (sampled statistics inline, or async with `PGW_NOISE_ASYNC_SAMPLED`),
  uuid4 request ids. Freivalds format confined to `tripnative::check_iface` (calls the existing trip kernels).
- Served: `MaskReservation` on disk segments (also through `WindowedReservation`). In-memory ring segments and
  `ChainedReservation` fall back to the `PGW_TRIP_FAST` path.
- Tests (`tests/engine/test_trip_native.py`): masked == plain bit for bit, byte-identical provider payloads and
  identical noise statistics on/off (windowed store reservations crossing chunks, extents and parts; K32, K40,
  K48, K24), single use under transport failure / tampered / malformed replies, soundness and non-finite refusals
  take no row, close mid-trip fails closed, detach hand-over, ticket single use and wipe.
- Measured (dev box, 4 shared vCPUs, loaded; stub service): trips only (`benchmarks/trip_native_bench.py`, model
  order, instant stub, extrapolated to the full model): 7B 48 -> 27 ms per token (median per trip: qkv 214 -> 137,
  o 146 -> 101, gate_up 203 -> 144, down 372 -> 229 us; masked head about 1 ms either way); 0.5B 34 -> 13.5 ms.
  Decode (`decode_cpu_bench.py`, 2 layers + masked head, native float ops, `--wnorm-frac 0.3`, stub delay 0.25 ms,
  extrapolated): 7B 127 -> 78 ms process CPU per token (2-layer run 10.75 -> 7.61 ms), 0.5B 58 -> 38 ms; outputs
  repeat-identical and equal on/off.
- Left per trip on the main core: the lease refill (Python AES-GCM decrypt of `W r` chunks, about 1.5-2.5 ms per 7B
  token, inherent to a disk store; a RAM ring lease would remove it), the masked head (608 KB row per token, 1-row
  leases), and 40-60 us of Python glue per trip (noise `guard_packed` ~20 us, request object, statistics,
  deferred-check bookkeeping).

## Update 2026-11-22: weight-free CPU gateway at 7B shapes (package W1, local only)

Write-up: [research/plans/weight-free-cpu-gateway-7b.md](research/plans/weight-free-cpu-gateway-7b.md). Not reached:
<= 8 ms of gateway CPU per 7B token. All four linears remote plus a masked head (113 trips per token), measured on
the 4-vCPU dev box with a stub provider: about 2 ms of gateway time per layer (Python path), 57-65 ms per token
extrapolated, 9.5 tokens/s at a 0.25 ms round trip; the native kernels alone are ~16 ms main core + ~7 ms helper core
per token (66.6 MB of Freivalds vectors and 6.7 MB of r keystream per token dominate).

- **Masked remote LM head (opt-in)**: `pgw.engine.masked_head` (`MaskedHeadPack`, `enable_masked_head`), linear
  `"<L>.lm_head"` (new `LinearKind.LM_HEAD`, never in `LINEAR_ORDER`), int8 per-row head, domain-separated
  fingerprint, every row range- and Freivalds-checked (synchronously: the last linear). New pinned reference
  (int8 head); masked == plain bit for bit in `tests/engine/test_masked_head.py`. Wired end to end, opt-in (package
  W3, below): gateway `head_mode = "provider_int8"`, service `masked_head = "also"|"only"`, factory `masked_head`.
- AVX2 Freivalds output half (identical results): gate_up 140 -> 47 us, 7B head 668 -> 470 us per call.
- `PGW_EMBED_COMPACT=1` (opt-in): bfloat16-exact embedding table kept in bfloat16 (2.18 -> 1.09 GB RAM at 7B).
- Storage at 7B (K32): 0 linear weights, 1.09 GB embeddings, 6.17 MB of `W r` per token (185 MB/s at 30 tokens/s;
  a 10 s RAM ring is 1.85 GB), ~38 GB of hints only with a gateway-side encrypted factory, KV 115 KB per token.
- Next levers (not done): a native per-trip path, a one-vector Freivalds check over a prime field (66.6 -> 9.5 MB of
  check vectors per token), r derived off the critical path, `W r` stored mod 2^29 (-9.4%).
- Bench fix: zero synthetic weights made the width selection evaluate all 11 widths (prepare ~2x too slow in earlier
  zero-weight benches); `--wnorm-frac 0.3` gives real norms.
## Update 2026-11-22: Qwen2.5-1.5B and 7B co-located, measured (package L1)

Both larger targets now run the co-located demo end to end, exact in every run (logprobs bit-identical, text equal,
repeat, stream and all sustain chats identical). Details and tables: `deploy/modal/README.md`, "Larger targets".

| config (Modal) | decode tok/s | TTFT | trips/token | factory alone / consumption |
|---|---|---|---|---|
| 1.5B, 2xT4, gateway GPU, attention_down | 20.0 | 0.81 s | 28 | 118-142 / 43.6 slots/s |
| 7B, 2xT4, gateway GPU, attention_down | 10.0 | 1.49 s | 28 | 39-46 / 18.6 slots/s |
| 1.5B, 1xT4, CPU-only gateway, all linears remote | 9.5 | 1.72 s | 112 | about 57 (2 workers) / 22.8 slots/s |
| 7B, 1xT4, CPU-only gateway, all linears remote | 5.0 | 3.1 s | 112 | 18 / 9.1 slots/s |

- Fixed: the co-located enrollment code listed the 0.5B whatever the target (1.5B/7B pairs could not enroll).
- One thread-safe `ByteBudget` per service now covers the compute routes, `/pgw/v1/maskgen` (which had none) and the
  bulk maskgen channel; the demo uses the bulk channel by default (library default off).
- `PGW_HINT_DEVICE=cuda` (library default off): the one-time encryption hint on the gateway's GPU, same bits,
  self-checked; 7B 213 s vs about 20 min on 16 vCPUs.
- 7B customer storage as run: hint 34.2 GB, int8 pack 7.6 GB, embedding + LM head 2.2 GB, masks 5.56 MB per slot.
- Not reached: 30 tokens/s at 7B. The T4s are bandwidth-bound (about 3.8 GB per token on each side with a gateway
  GPU); the CPU-only gateway pays 112 trips of about 1 ms plus about 0.8 ms of gateway CPU per trip.

## Update 2026-11-22: mask factory keeps up with back-to-back chats (package S5)

The S4 demo ran out of masks after 130 s of back-to-back chats (HTTP 503 `mask_exhausted`): all-or-nothing yield
at 1000 slots, refill only below low water 1000, 30 s scheduler poll, 4096-slot segments. Now 302 s of
back-to-back chats (146 chats, about 51 slots/s consumed) with no exhaustion: the stock settles into a sawtooth
of about 4100-6300 slots (a 2048-slot segment every 40-45 s, segment-to-segment production 50.6 slots/s),
streamed decode 31.7 tokens/s token-weighted (median 32.4), every chat exact. Details: `deploy/modal/README.md`,
"Factory keep-up under back-to-back chats".

- `PGW_FACTORY_YIELD_FULL_SLOTS` / `_PERIOD_S` / `_HORIZON_S` / `_FULL_HORIZON_S` (`pgw.core.factory_yield`;
  library defaults off = all or nothing as before; co-located demo on): partial yield whose duty cycle grows as the
  stock falls, never below the stock covering 45 s of observed consumption, full yield when the stock covers 120 s
  (occasional chats unaffected). Same masks, requests and bytes; the timing depends on decode activity and stock,
  both of which the provider can already derive from its own trips and maskgen requests.
- Demo factory sizing: 2048-slot segments, refill below 6144 to 8192, 2 s poll.
- `--sustain-s` streams every chat, checks all are identical, samples the stock and reports consumption vs the
  factory's production (`keep_up`); `factory.committed` fixes the empty `segments` list for long segments.
- Over a window that starts at the top of the sawtooth the factory's committed slots read below consumption
  (40.9-47.6 vs 49-52 slots/s): the stock starts above its equilibrium; slope -1.1 to -2.2 slots/s. Not done:
  moving the factory's encryption/decryption to the gateway GPU (lever 3).

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
## Update 2026-11-22: strict speculation on a CPU-only gateway (off by default)

Details: [research/plans/strict-speculation-cpu-gateway.md](research/plans/strict-speculation-cpu-gateway.md).
Every linear remote + `float_ops = native` + padded strict speculation with a draft on the gateway CPU.

- Exact: verify rows of R = 2/4/8 passes (with padding rows) equal single-row decoding bit for bit (masked,
  all remote, native ops; also at real 0.5B shapes with both float paths); spec == plain bit for bit for five
  draft sources incl. the pipelined drafter; R rows per pass and L passes on the wire. The GPU refusal is
  about a CUDA target and does not touch this configuration (`float_ops = native` is CPU-only).
- `draft_backend = "native_int8"` (`NativeInt8DraftBackend`, `pgw_fast.local_linear_into`): Qwen2.5-0.5B
  draft 75 ms/token on 1 core, 46-54 on 4 shared vCPUs (torch int8: 390 / 112); memory-bound, ~10-20 ms
  expected on 16 cores.
- `spec_pipeline_steps = m` (`PipelinedDrafter`): the next block is drafted in the background during the
  pass; exactly m draft steps between passes hit or miss (no acceptance timing signal; tested).
- 7B model (`benchmarks/spec_7b_cpu_gateway_model.py`): 25 tokens/s target met with margin by R = 2, C = 2,
  m = 0 at 0.3 ms trips on a 16-core gateway (29-32 tok/s, <= 2 mask slots per token); 0.4 ms needs R = 3;
  0.5 ms only at high acceptance. Plain CPU-only decoding tops out near 22. No 7B acceptance measured.
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
