# PGW implementation plan: speed and readiness (after run 25)

## 0. Starting point and limits

**Current numbers, Kaggle run 25** (commit f2e02ef: fused Triton 1-row GEMV plus the wire cut)

| metric | value |
|---|---|
| single-stream decode | 0.365 s/token |
| TTFT, 40-token prompt | 2.89 s |
| 4-stream aggregate | 3.0 tok/s |
| service 1-row call p50 | 1.39 ms (0.70 ms of it on the GPU) |
| round trips | 0.114 s/token as reported |

- The round-trip figure is understated. `gpu_validate.py:351` divides by `n_prompt + max_tokens`, and decode-only remote time is closer to 0.13-0.15 s/token.
- Some proposal texts quote run 23/24 baselines (0.41-0.44 s/token). Every gain below is re-based to run 25.

**What sets the floor.** One forward pass needs 112 round trips that must run one after another (28 layers × 4 linears). Merging or splitting them is not possible without breaking bit-exactness, because each linear's input is requantized with a whole-row absmax and nonlinearities sit between the linears. So:

`s/token ≥ 112 × (per-trip latency) + gateway CPU that cannot overlap (masking + float ops + LM head + Python)`

- Co-located today: 112 × about 1.15 ms ≈ 0.13 s, plus about 0.23 s of gateway CPU.
- Realistic best case on this hardware: per trip about 0.8 ms (0.09 s/token), masking about 0.05, bf16 LM head about 0.025, float ops and Python about 0.04. That gives **about 0.20-0.22 s/token single stream, about 4.5-5 tok/s**.
- In a real two-host deployment add 112 × network RTT: +0.01 to +0.035 s same-AZ, +0.05 to +0.11 s cross-AZ.
- The only way below 112 × trip is multi-token passes (speculation). That pays only with a GPU draft on the gateway, for about 10-15%.

**There is no exponential speedup available, and the measurements do not support one.** Realistic results:

- **Single stream:** about 1.15× after phase 1, about 1.35-1.5× after phase 2.
- **Aggregate:** grows more than single-stream, because the T4 is about 90% idle. About 1.2× at 4 streams in phase 1 and about 1.4-1.5× in phase 2. Beyond that, scaling needs more streams and gateway cores separate from the service.

**On "spin up 10s to 100 agents".** More than about 8 parallel agents does not help here. The hot files (handler.py, backends.py, model.py, generate.py) can each have only one writer at a time, and every speed claim must be checked on Kaggle (T4) one run at a time. The plan below is the most parallelism that still merges cleanly: 7 packages now, then 3 follow-ups after a combined Kaggle run.

## 1. Ranking by realistic gain per effort

Gains are against run 25. "1s" is single stream, "4s" is 4-stream aggregate.

| # | item | gain | effort | package |
|---|---|---|---|---|
| 1 | wnorm precomputed at pack/startup: first request no longer reads 7 GB of int8 | first-request TTFT 52.7 s → about 3 s (-90%) | S | K |
| 2 | Client-cert expiry fail-safe + renewal (every gateway has an outage on day 30) | readiness, critical | S (fail-safe), M (renewal) | X |
| 3 | Noise monitor: build x_hat/xq only for the ~5% of calls it samples | TTFT -15-25%; decode -0.02 to -0.03 s | S-M | K |
| 4 | GQA SDPA (`enable_gqa=True`) with a load-time bit-equality self-check | -0.02 to -0.025 s at about 100 tokens of context; -0.15 to -0.3 s at 1k | S | M |
| 5 | Service: one kernel per linear per multi-session request | 4s +9-10%; 1s 0 | S | S |
| 6 | Speculation: lookup privacy fix, fixed k, spec_max_batch, draft timer | 4s: removes the spec-on regression; 1s with GPU draft -10-15% after the sweep | S | G |
| 7 | glibc mallopt (+ uvloop on the service and gateway main loops) | TTFT -0.15 to -0.35 s; spec -3-5%; 4s +1-3% | S | S |
| 8 | Mixed-batch prefill pipelining | latecomer TTFT -10-15%; running-stream stall -1.5 to -2 s per admission | S | M |
| 9 | `_read_wr` reads that cross chunks via native uint32 | TTFT -3-5% | S | K |
| 10 | Per-trip accounting, sub-ms RTT histograms, network-floor bench | readiness: decides the next priorities | S | T |
| 11 | Mask-segment ramp, alert rules, sizing docs, Modal launcher safety | readiness, high | S each | O |
| 12 | Service mTLS required for compute; trusted-proxy rate limits; token revoke | readiness, high | S+S+M | X |
| 13 | Gateway node sizing (whole cores, Guaranteed QoS), AZ pinning | deployment: -0.04 to -0.07 s/token on real hardware; avoids +0.02 to +0.06 s cross-AZ | S | O |
| 14 | *Phase 2:* lean decode lane (frame-level service fast path + client `compute_fast`) | 1s -0.03 to -0.045 s (7-10%) | L | P2-a |
| 15 | *Phase 2:* two-cohort ping-pong in `_run_loop` | 4s +8-12%; more at 8-32 streams on a dedicated gateway | M | P2-b |
| 16 | *Phase 2:* native `prepare_many` + single-use mask prefetch | 4s +8-12%; 1s -3-4% plus less p99 jitter | M-L | P2-c |
| 17 | *Phase 2, gated:* CUDA-graph draft runner (only if measured draft_ms > 30 ms) | -5-10% with a GPU draft | M | P2-b |
| 18 | *Phase 2:* zero-copy bulk wire for prefill (server memoryview send, client no-copy receive; no raw TCP side channel) | TTFT -10-15% | M | P2-a |

**Do not pursue:**
- Tighter packing, compression, or 27-bit reply truncation. Truncation sets Freivalds soundness to 0 bits, so it is unsafe.
- Shared memory, busy-polling, a custom binary format alone. Each is under 1%.
- A bigger GPU class: at most 7% at 2-7× the cost.
- 4 MB mask chunks.
- Splitting `quantize_mask_pack` across threads.
- A layer-0 QKV table.
- CPU pinning except as one Kaggle experiment.

## 2. Combined estimates

| | single-stream decode | 4-stream aggregate | TTFT (40 tokens, warm) | first request after start |
|---|---|---|---|---|
| run 25 | 0.365 s/token (2.74 tok/s) | 3.0 tok/s | 2.89 s | about 52 s |
| after phase 1 (packages below) | **0.31-0.33 s/token** (3.0-3.2 tok/s) | **3.4-3.6 tok/s** (3.6-3.8 with a GPU spec draft only for single stream) | **2.0-2.4 s** | **about 3 s** |
| after phase 2 | **0.25-0.28 s/token** (0.22-0.25 with a GPU draft at k=2) | **4.0-4.4 tok/s** on Kaggle; 5-6 at 8-16 streams on a dedicated 8-core gateway | 1.7-2.0 s | about 3 s |
| structural floor (co-located, no speculation) | about 0.20-0.22 s/token | GPU-bound only beyond about 100 rows/s; the gateway CPU limits first | — | — |

Phase-1 single-stream breakdown: GQA -0.022, noise lean path -0.02, uvloop/mallopt -0.005 to -0.01, variance ±0.01. Every estimate must be confirmed by the combined Kaggle run 26, run from package T after the merge.

## 3. Phase-1 work packages (run in parallel, files do not overlap)

**Rules for every package:**
- Use your own git worktree off `claude/charming-carson-qxlk5v` at f2e02ef.
- Edit only your listed files. New files go only in your listed paths.
- Read `pgw/STATUS.md` first.
- This container is 4 vCPU / 15 GB: run only the targeted test files named in your brief, never the full suite, and no 7B jobs. Speed claims are validated on Kaggle T4 by package T after the merge.
- Outputs must stay bit-identical to the plaintext quantized reference. Pads are single-use and per session, the provider sees only masked data, and Freivalds and range checks stay.
- Commit with the session's attribution lines. Report measured local numbers and the Kaggle checks needed.

**Interfaces between packages** (every caller guards with `hasattr` or `getattr` so merge order does not matter):
- C1: package K adds `MaskedBackend.warm_weight_norms()`, which returns when every linear's wnorm is known. Package T's `runtime.py` calls it (if present) in a worker thread before setting model_ready. It runs only during startup; there is no cost on later requests.
- C2: package K adds `BackendStats` fields `decode_calls`, `decode_remote_seconds`, `decode_local_seconds` (counted for calls with rows ≤ `inline_max_rows`, and after the first decode step when available). Package T reads them with `getattr(..., 0)`.
- C3: package G adds config field `spec_max_batch: int = 1` to `GatewaySettings`. Package T forwards `--spec-max-batch` and `--spec-min-tokens` into the gateway config only when the flag is given.
- C4: package X adds `CommonSettings.trusted_proxy_cidrs: list[str] = []`, env `PGW_TRUSTED_PROXY_CIDRS`, default trust nothing. Package O maps Helm value `trustedProxyCidrs` to that env var.
- C5: package S adds `pgw/core/proc_tuning.py` with `apply_malloc_tuning()` and `new_event_loop()` (uvloop if importable, else asyncio). Only package S calls it in phase 1.

---

### Package K: gateway masking path (wnorm, noise-monitor lean path, chunk reads, decode stats)

**Files:**
- `src/pgw/engine/backends.py`
- `src/pgw/observability/noise.py`
- `src/pgw/modelpack/manifest.py`, `packer.py`, `loader.py`
- `src/pgw/maskstore/store.py`
- `rust/pgw-fast/src/store.rs`, `rust/pgw-fast/src/prepare.rs` (only if needed), `rust/pgw-fast/src/lib.rs` (registration only)
- New: `tests/engine/test_wnorm_manifest.py`, `tests/engine/test_noise_lean.py`, `tests/maskstore/test_read_wr_span.py`

**Brief:**
1. **wnorm (largest single win).**
   - The packer writes an integer `row_norm_sq_max` per `LinearEntry`, computed exactly as the int64 sum of int32 squares. The fingerprint covers it.
   - `load_linear_meta` returns it.
   - `MaskedBackend.conv` (backends.py about 666-674) uses `sqrt(float(row_norm_sq_max))`. That is bit-identical to `_row_norm_max`: the float64 sum is exact because it stays below 2^53. Assert this in a test.
   - Packs without the field: implement `warm_weight_norms()` (C1). It computes the norms chunked, with an int64 accumulator, and writes a sidecar `wnorm.<fingerprint>.json` next to the pack (or in a cache dir if the pack directory is read-only), so the next start reads no weights.
   - The gateway must never call `pack.load_linear` on the hot path. Add a counter or mock test for that.
2. **Noise-monitor lean path.**
   - Split `noise.guard` into `plan_call() -> bool` (the same RNG draw and `always_sample_first` as today, drawn in the same order) and the checks.
   - On non-sampled calls `_prepare_native` stays lean: no int64 x_hat, no `xq`, and the lean `finish_group`. Run the ring range check (only when K<32) and the duplicate-row fingerprint on the packed uint32 payload, using the same columns, top bits and keyed multipliers. Test that the fingerprints are equal to today's for random payloads.
   - Sampled calls keep the full path and drop x_hat after `guard`, so the finish group stays uniform.
   - Monitor tests must still alarm on an injected reused mask and on an unmasked send.
3. **`_read_wr` across chunks** (store.py about 474-510).
   - Allocate the uint32 output (rows, out) once and decrypt plus native-unpack each chunk piece into its slice. Add a Rust `unpack_rows_u32_into(buf, out_slice)` or loop with per-piece wipes.
   - Remove the bytearray + `unpack_ring` int64 path for K=32.
   - Wipe the same buffers the current wipe paths do.
   - Do not change the chunk size.
4. **Optional, small:** send the wide `down` rows (cols 18944 > 4096 in `_mask_rows_keyed`) to native `mask_rows_with_key` if the bits match. Measure first.
5. Add the C2 decode stats.

**Validation (local):**
- New tests above.
- `tests/engine` masked-vs-plain exactness tests on the tiny pack (only the files that touch MaskedBackend).
- `tests/maskstore/test_store*.py` wipe tests.
- `benchmarks/profile_gateway.py --preset 7b --rows 1` and `--rows 40 --noise`. Expected: 40-row prepare about 15 → about 6-7 ms per call; `_read_wr` about 2.75 → about 1.2 ms per call.

**Kaggle (run 26):**
- First `chat` ttft_ms about 52 s → about 3 s.
- Gateway CPU for the first chat about 118 → about 27 s.
- Tokens and logprobs identical.
- noise_monitor samples still increasing.

---

### Package M: float model path (GQA, mixed-batch pipelining, CUDA LM-head guard)

**Files:**
- `src/pgw/engine/model.py`
- `src/pgw/engine/lm_head.py`
- New: `tests/engine/test_attention_gqa.py`, `tests/engine/test_pipeline_mixed.py`

**Brief:**
1. **GQA.** In `TransformerModel._attention` (model.py 222-237), replace `repeat_interleave` with `F.scaled_dot_product_attention(..., enable_gqa=True)`, only if torch ≥ 2.5 and a one-time per-process self-check passes. The self-check runs both forms at the configured thread count over T=1 and T>1 with a causal mask, several L (1, 37, 100, 513), a sliced non-contiguous cache view (as `KVCache.append` returns), and the serial_tail per-row path. Use the GQA form only if every output is `torch.equal`; otherwise keep the old path and log once.
2. **Mixed-batch pipelining** (`_aforward_split.hop`, about line 507).
   - Today `all(rows >= 2)` disables the split. Instead, split each call with ≥2 rows, put each 1-row call wholly in half A, and give half B no item for it.
   - Rejoin results by call position, not with the strict zip.
   - Mask rows per session must still be taken in order: half A first. Check that `MaskExhaustedError` and discard-on-failure (`owned`) still work.
   - Keep `pipeline_min_rows=16`. Do not add the 4-way split yet.
3. **LM head on CUDA.** `lm_head_logits` falls back to the slow per-chunk reference when the head is on CUDA (about 17k kernel launches). Keep the head on CPU (bf16 compact) when the gateway device is cuda, and log a warning. Do not write a Triton head in phase 1.

**Validation (local):**
- GQA equality grid test.
- `forward_batch` with [1-row decode, 40-row prompt, 40-row prompt] gives identical logits with `pipeline_min_rows=16` and with 0, and mask rows consumed per session equal its row count.
- Existing `tests/engine/test_model*.py` and `test_generate.py` exactness tests.
- Microbench `_attention` at L=100 and L=1000 (expect 1.14 → 0.07 ms per layer per thread at L=100).

**Kaggle:** decode about -20-25 ms/token, concurrent latecomer TTFT lower, identical outputs. Add a 1000-token-prompt case to show the gain grows with context.

---

### Package G: speculation (privacy fix plus a cost-model retune)

**Files:**
- `src/pgw/engine/generate.py`
- `src/pgw/gateway/config.py`
- `tests/engine/test_speculative.py`

**Brief:**
1. **Prompt-lookup privacy** (`_lookup_drafts` 113-127, `_plan_drafts` 450-456).
   - Minimum match length `spec_lookup_min_n` (default 3); no 1-gram matches.
   - A proposal is exactly k tokens: pad a short match by continuing the draft model from the lookup tokens when a draft exists; otherwise send 0 drafts. So every pass carries rows in {1, 1+k}.
   - `_spec_budget`: near the end of the window, budget or mask runway, drop k to 0 rather than to a value that depends on acceptance history.
   - Update the 1-gram unit test (line 210) to match.
2. **Retune.**
   - New defaults `spec_tokens=2`, `spec_min_tokens=2` (one fixed size).
   - When `spec_min_tokens >= spec_tokens`, skip the per-token softmax/max and its host sync in `take()`.
   - Add `spec_max_batch=1` (C3): `_plan_drafts` returns no drafts when the step has more sequences than that.
   - `spec_row_budget` default 2.0, plus a test that k=2 does not starve.
   - Speculation stays off by default (no draft model, `spec_lookup=0`). In the config description, say a CPU-only draft does not pay.
3. **Metrics.** Add `draft_ms`, `spec_rows`, `spec_accepted` per pass to `GenerationResult.timings_ms` and stats. Package T and the phase-2 graph-draft decision depend on this.

Exactness: verification is unchanged; a draft is accepted only when the target's sampler emits it.

**Validation (local):**
- `tests/engine/test_speculative.py`: identity tests at k=1,2; a per-pass row-count invariant {1, 1+k} with lookup on, with and without a draft; a 1-gram match no longer pre-empts the draft; `_plan_drafts` is empty when `len(batch) > spec_max_batch`.

**Kaggle (via T's build.py):**
- Sweep `--spec-tokens 1,2,3` with cuda:1 and cpu drafts.
- Expect cuda:1 at k=2 about 0.30-0.32 s/token against no-spec about 0.33 after phase 1.
- spec_concurrent equals non-spec concurrent.
- Then fix the default k from the sweep.

---

### Package S: service throughput and process runtime

**Files:**
- `src/pgw/service/handler.py`, `batcher.py`, `server.py`, `gpu_exec.py` (optional stage timings, off by default)
- New: `src/pgw/core/proc_tuning.py`
- `src/pgw/gateway/cli.py`
- `src/pgw/factory/cli.py`
- `deploy/docker/**`, `deploy/systemd/**` (MALLOC_* env only)
- `benchmarks/wire_overhead_bench.py`
- New: `tests/service/test_grouped_compute.py`, `tests/core/test_proc_tuning.py`

**Brief:**
1. **Grouped kernel per linear.** In `ServiceHandler._compute` (handler.py 255-307), when `len(items) > 1`:
   - Group items by linear key, unpack each, and make one `batcher.submit_parts(parts)` call per group. `_run_kernel` already concatenates parts.
   - Return the row slices in order and pack each item as today.
   - Keep the `len(items)==1` fast path, including the 1-row Triton route.
   - On T4, a group of exactly 2 one-row items may be faster as 2 Triton calls than as one digit GEMM. Choose by `max_rows()` and a measured threshold; the default is to group when ≥3 rows.
   - `submit_parts` must also work on the queued `_add` path (spec groups can exceed `inline_max_rows=8`) and must respect `max_batch_rows` chunking.
   - A wrong-shape item or an exception fails every item in its group with the item's request id. Keep the first-call cross-check per (linear, K, rows) for the new row counts.
   - Wire format and gateway are unchanged.
   - Also remove the `asyncio.timeout` around the inline kernel. The inline kernel cannot suspend, so a plain remaining > 0 deadline check is enough.
2. **`proc_tuning.py`** (C5).
   - `apply_malloc_tuning()`: only on glibc (detect `gnu_get_libc_version`), via ctypes `mallopt`: M_TRIM_THRESHOLD=256 MiB, M_TOP_PAD=64 MiB, M_MMAP_THRESHOLD=32 MiB, all three together. Do nothing if `PGW_MALLOC_TUNING=0` or the platform is not glibc.
   - `new_event_loop()` uses uvloop if importable.
   - Call both at the top of the service `serve()` (`asyncio.run(..., loop_factory=...)`), the gateway `cli.serve`/`_main`, and the factory `main`.
   - Set the equivalent MALLOC_* env vars in the Dockerfiles and systemd units.
   - The wire-client thread's loop (wire/client.py) is left to phase 2, because package X owns that file.

**Validation (local):**
- Grouped request (4 items on linear A + 1 on B) gives y_hat byte-identical to 5 single-item requests, and `stats.batches` rises by 2, not 5. Also cover the spec-sized group above `inline_max_rows`, and error fan-out.
- `tests/service/test_batcher.py`, `test_handler*.py` only.
- `wire_overhead_bench.py --rows 1,20,40 --calls 200`, A/B interleaved, with tuning on and off. Expect rows=40 about 9 → about 4.5-5 ms; rows=1 unchanged or better with uvloop.
- Unit test that the mallopt helper does nothing on a non-glibc platform (mock it).

**Kaggle:**
- concurrent 4 and spec concurrent: tokens identical; remote s per round trip at 4 streams about 6 → 3-3.5 ms; service `mean_rows` rises.
- Peak RSS per process recorded.
- TTFT lower by 0.15-0.35 s.

---

### Package T: measurement, benchmarks and gateway telemetry

**Files:**
- `benchmarks/gpu_validate.py`
- `benchmarks/kaggle/**`
- `benchmarks/profile_gateway.py`
- New: `benchmarks/net_floor_bench.py`
- `src/pgw/gateway/metrics.py`, `src/pgw/gateway/runtime.py`
- `pgw/STATUS.md`, `benchmarks/RESULTS.md`

**Brief:**
1. **Accounting fix** in `gpu_validate.py`.
   - Report decode-only remote/local s per token and s per trip, using C2 when present; otherwise take the stats delta after the first decode token.
   - Keep the old fields, renamed to `*_incl_prefill`.
   - Add per-trip client RTT, service queue_ms and compute_ms, and wire = RTT - queue - compute, clamped at 0. In multi-item responses `compute_ms` is a sum and `queue_ms` a max, so label by rows.
   - Record lscpu, CPU count and topology.
   - Forward `--spec-min-tokens` and `--spec-max-batch` (C3), only when given.
2. **Telemetry.**
   - New per-hop buckets (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 25 ms) for `pgw_gateway_round_trip_seconds`, plus new `pgw_gateway_network_seconds` and `pgw_gateway_service_seconds`.
   - `_InstrumentedClient` (runtime.py:92) must also time `compute_sync`. Today `__getattr__` passes it through untimed, and decode uses it via backends.py:1238.
   - Add the `pgw_client_cert_expiry_seconds` gauge, set at startup and hourly from the configured client cert.
   - Call C1 `warm_weight_norms()` before model_ready.
3. **Benches.**
   - `net_floor_bench.py`: TCP ping-pong for 64 B and 80 KB over loopback.
   - `profile_gateway.py`: warm up every layer, not just layer 0.
   - Kaggle `build.py`: add the spec sweep (k=1,2,3 × cpu and cuda:1), the net floor, and `--concurrent 4 8 16`. Move off `--spec-tokens 8`.
4. **After all phase-1 packages merge:** run Kaggle run 26 and update STATUS.md with real numbers, including whether decode remote per trip × 112 + local + LM head adds up to s/token within 10%.

**Validation (local):**
- A stub service with fixed compute_ms and an injected delay gives the expected network histogram.
- `gpu_validate.py --help` and a dry run on the 0.5B tiny pack if one exists locally (CPU, a few tokens).

---

### Package X: security readiness (TLS identity, auth, rate limits, revocation)

**Files:**
- `src/pgw/enroll/service.py`, `src/pgw/enroll/client.py`
- `src/pgw/wire/client.py`, `src/pgw/wire/server.py`
- `src/pgw/service/auth.py`, `src/pgw/service/config.py`, `src/pgw/service/app.py`
- `src/pgw/console/sessions.py`, `src/pgw/console/routes.py`
- `src/pgw/gateway/setup.py`, `src/pgw/gateway/app.py`
- `src/pgw/core/settings.py`
- New: `src/pgw/core/client_ip.py`
- `src/pgw/cli/main.py`
- `tests/service/test_auth*.py`, `tests/enroll/`, `tests/console/`, `tests/wire/test_tls*.py` (new or extended)

Priority order (land item 1 first, as its own commit):

1. **Critical: client cert expiry on day 30.**
   - (c) In `wire/client.py`, when building the SSL context, do not present a client cert that has expired or expires within 60 s. Log an error and fall back to token-only, which works under CERT_OPTIONAL.
   - (a) Add `POST /pgw/v1/enroll/renew`, authenticated with bearer token + current unexpired cert + new CSR. Same CN, tenant-checked, audited `enroll.renewed` with the serial. Never accept an expired cert.
   - (b) Add a gateway renewer at 2/3 of the cert lifetime with jittered retry. It writes the new pair atomically (key 0600), then rebuilds the SSL context and the stream channel between calls, never during a hop.
   - Note: under `require_client_cert=true` only (a) + (b) prevent the outage.
2. **Require mTLS for compute.**
   - Add `ServiceSettings.require_client_cert_for_compute`: default true when `client_issuing_ca_*` is configured.
   - In `TenantAuthenticator.authenticate`, raise PERMISSION_DENIED when the flag is set and no verified cert maps to a tenant. Tokens enrolled without a CSR are marked token-only and exempt.
   - The TLS layer stays CERT_OPTIONAL so the enroll routes and probes keep working.
3. **Rate limits behind proxies.**
   - Add `core/client_ip.py`, a resolver: take the right-most X-Forwarded-For hop not in `trusted_proxy_cidrs` (C4); ignore the header from untrusted peers.
   - Use it in console login, OIDC callback, the setup wizard, the audit `client` field and the enroll limiter.
   - Count failures only, plus a global failure ceiling (300/min) and a cheap global callback cap.
   - Do not also enable uvicorn proxy_headers: use one mechanism.
4. **Token revocation.**
   - New format `tenant:token_id:sha256[:not_after]`. The old `tenant:sha256` lines still load.
   - `pgw service token list|revoke ID`, with atomic rewrite.
   - Reload by mtime, rebuilding the whole set from the file.
   - Close open WebSocket channels of a revoked token_id with 1008.
   - Audit records carry token_id, never the token.

Masks, numerics and the wire data path are untouched.

**Validation (local):**
- Fake-clock test: a 2 s cert is renewed before expiry, and the next connection presents the new serial.
- In-process TLS test: an expired cert gives a handshake failure before the fix and a token-only success after it.
- With the flag on: token without cert → 403; token + mapped cert → ok; enroll routes → 200 without a cert.
- Trusted-proxy bucket tests.
- A revoked token gets 401 within one reload interval, and its channel is closed.

**Modal follow-up:** `client_cert_validity_days=0.01`, observe one renewal cycle during greedy chat with bit-identical output.

---

### Package O: ops, deployment and docs readiness

**Files:**
- `src/pgw/factory/scheduler.py`
- `src/pgw/cli/doctor.py`
- `deploy/helm/**`
- New: `deploy/monitoring/pgw-alerts.yml`
- `deploy/modal/**`
- `deploy/marketplace/aws/README.md`
- `docs/operations/*.md`, `docs/tools/pgw_sizing.py`, `docs/measurements.json`
- `SPEC.md` (section 13 only)
- `tests/factory/test_scheduler*.py`, `tests/docs/`, new `tests/deploy/test_alert_rules.py`

**Brief:**
1. **Segment ramp.**
   - While `avail < low_water`, plan exactly one segment of `min(segment_slots, max(first_segment_slots, next_pow2(2*avail)))`: 256 → 512 → 1024 → 2048 → 4096.
   - Re-plan after each segment lands; return to full segments at or above low water.
   - Test plan() for avail 0, 256, 512, 1000 and 5000; state the boundary behaviour.
2. **Alerts.**
   - A PrometheusRule template per chart (gated by `serviceMonitor.enabled`) plus `deploy/monitoring/pgw-alerts.yml`, covering runway, low water, factory stalled, integrity failure (page), noise alarm (page), model not ready, and burn ratio (use `pgw_masks_burned_slots_total`).
   - Aggregate by (fingerprint, ring).
   - Add `alerts.runwayMinSeconds` instead of the nonexistent `refill_lead_s`.
   - Test that every metric named in a rule exists in METRIC_NAMES.
3. **Doctor.**
   - Mask-supply check: time one hemask chunk in a temporary store (reuse factory/bench.py), bound by measured link throughput, compare against expected load, and check disk.
   - Cert-expiry check: warn under 7 days.
4. **Sizing docs.**
   - K32A12C27 rows (5.56 MB per slot for 7B).
   - Link to the encrypted-mask costs.
   - Speculation can use up to `spec_row_budget` × masks (off by default).
   - Reconcile the factory rates in slots/h for one model and ring.
   - Note that K40 rows over-size the store; they do not under-size it.
   - Per-tier RTT table marked as estimates.
   - Network caveat in encrypted-masks.md: the HE supply cannot keep up over the Modal tunnel.
   - `pgw_sizing.py --ring K32A12C27 --encrypted --spec-row-budget`.
5. **Helm.**
   - Gateway: integer CPU requests equal to limits, memory requests equal to limits (Guaranteed QoS), `priorityClassName`, and documented `cpuManagerPolicy: static`.
   - Factory: hard CPU limit and node separation.
   - Service: zone `topologySpreadConstraints` example.
   - Gateway: zone nodeSelector example.
   - `trustedProxyCidrs` → `PGW_TRUSTED_PROXY_CIDRS` (C4).
   - Marketplace README: zonal vpce DNS with the regional name as fallback, and the cost of cross-zone NLB off (a replica per AZ).
6. **Modal launcher.**
   - Call `stop()` before SystemExit on the start timeout.
   - Dead-peer detection by run_id/FunctionCall status, only after the peer has been seen once.
   - Print the setup token once and then overwrite it with "(shown)". Document that `show()` can no longer reprint it.
   - Optional `bench_colocated.py`: one app, both functions pinned to the same region, i6pn, with its own RTT ping (`pgw doctor --rtt` does not exist).

**Validation (local):**
- Scheduler and docs tests.
- Render the Helm charts with the repo's Python renderer test.
- `promtool check rules` if available.
- `python deploy/modal/local_rehearsal.py` with the provider killed mid-run.

---

### File ownership check

Every listed path has exactly one owner:
- K: backends, noise, modelpack, maskstore/store, rust store.rs and prepare.rs
- M: model, lm_head
- G: generate, gateway/config
- S: service handler, batcher, server, gpu_exec; core/proc_tuning; gateway/cli; factory/cli; docker, systemd
- T: gpu_validate, benchmarks/kaggle, profile_gateway, gateway/metrics, gateway/runtime, STATUS.md
- X: enroll, wire client and server, service auth/config/app, console, gateway setup and app, core/settings, core/client_ip, cli/main
- O: factory/scheduler, cli/doctor, helm, monitoring, modal, marketplace, docs, SPEC

Any package can merge first. Cross-package links are only C1-C5, each guarded by `hasattr`/`getattr`.

**Merge order:** X item 1 (critical) as soon as it is ready, then K, M, S, G, T, O, rest of X. T then runs Kaggle run 26.

## 4. Phase 2 (after run 26; three parallel packages)

**P2-a, lean wire lane.**
- Files: `service/server.py`, `wire/server.py`, `service/handler.py`, `batcher.py`, `metrics.py`, `wire/client.py`, `wire/codec.py`, `engine/backends.py` (`send_prepared` only).
- Frame-level synchronous handling of single-item, small, idle-device requests. Per-connection tenant/auth/TLS-peer identity, plus the same drain, budget, rate, request-limit and validation checks.
- Client `compute_fast` that falls back to `_call` with a byte-identical resend.
- uvloop on the wire-client thread.
- Zero-copy server send and client receive for bulk prefill payloads; no raw TCP side channel.
- Needs differential byte-identity tests for the two lanes and a security review.
- Target: -0.03 to -0.045 s/token and TTFT -10-15%.

**P2-b, scheduling.**
- Files: `engine/generate.py`, `engine/model.py`.
- Two-cohort ping-pong behind a flag, off by default, enabled only with more than 4 live sessions. Keep the MaskExhaustedError isolation and round-robin invariants.
- If run 26's `draft_ms` is above 30 ms, add a CUDA-graph draft runner. It needs a static KV with index-reset truncate and on-device argmax chaining with the vocab clamp.
- A/B on Kaggle at 4, 8 and 16 streams.

**P2-c, gateway CPU parallelism.**
- Files: `engine/backends.py`, `maskstore/store.py`, `rust/pgw-fast/src/prepare.rs`, `store.rs`.
- Native `prepare_many` over sessions with a persistent pool.
- Single-use prefetch staging per (reservation, linear), active only for rows ≥ 3 or for multi-session steps. close() counts staged rows as burned and zeroes them; never cross an extent boundary.
- Validate against the real service, not `--null-service`.

**Not scheduled (unless run 26 changes the picture):**
- Gateway-GPU Triton LM head: opt-in customers only, about 5%.
- 4-way prefill split.
- Bigger GPU class.
- `PGW_NATIVE_THREADS`: Rust already honours the cgroup quota once Helm sets limits.

## 5. Risks and checks to keep

- Any change to row order or row count per session must keep per-session mask consumption equal to the rows sent. Every package touching this has a test for it.
- The GQA and wnorm changes depend on exact float equality. Both are guarded: GQA by the self-check and fallback, wnorm by the < 2^53 integer argument plus an equality test.
- Speculation row counts must stay within {1, 1+k} so the privacy claim of at most 1 bit per pass holds again.
- Bigger RSS from mallopt (up to about 320 MiB per process) must be recorded on Kaggle.
- No package may claim a speed number from this 4-vCPU container. Only Kaggle run 26 (and later the Modal co-located bench) counts.
