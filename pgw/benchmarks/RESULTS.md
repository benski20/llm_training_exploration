# PGW GPU validation on Kaggle (Tesla T4)

Date: 2026-11-17. Hardware: Kaggle GPU kernel, 2x Tesla T4 (16 GB; one used), 4 vCPUs, 29 GiB RAM.
Software: Python 3.13.16, torch 2.11.0+cu130 (CUDA 13.0), pgw 0.1.0 from this tree.
All runs go through the real CLI and real processes on 127.0.0.1: `pgw pack` -> `pgw maskstore init` ->
`pgw factory generate --device cuda` (encrypted store) -> `pgw service serve` (device cuda) ->
`pgw gateway serve` (engine on CPU, masked mode, Freivalds on every response, noise monitor on).
Ring K40A12 (default) unless stated otherwise. The QuantPlain reference is an in-process
`pgw.engine.PlainBackend` engine on the same pack (on CPU for 0.5B and on CUDA for 7B).

Drivers: `benchmarks/gpu_validate.py` (one model end to end) and `benchmarks/kaggle/build.py`
(builds the kernels with the source embedded). Raw outputs are in `benchmarks/results/kaggle/`:
`run2_tests/` (GPU tests, full suite, ring benches, 0.5B), `run3_7b/` and `run4_7b/` (7B; run4 adds
batched load tests), `run1_7b/` and `run2_7b/` (the failed runs that found bugs 1 and 2 below),
and `run5_verify/` (re-run after the fixes, see "Tests").

## Exactness

| model | pack (int8) | answer (greedy, 32 tokens, ignore_eos) | tokens vs QuantPlain | logprobs and top-2 | Freivalds-checked items | integrity failures |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.92 GB | "The sky is blue because of the scattering of sunlight by tiny particles in the Earth's atmosphere. ..." | identical (non-streaming, streaming, 8 concurrent) | identical, bit for bit (max diff 0.0) | 3072 per request, 0 skipped | 0 |
| Qwen2.5-7B-Instruct | 8.72 GB | "The sky appears blue because the Earth's atmosphere scatters sunlight in all directions, ..." | identical (non-streaming, streaming, 4 and 8 concurrent) | identical, bit for bit (max diff 0.0) | 3584 per request, 0 skipped | 0 |

7B fits a single 16 GB T4: the service holds 6.5 GB of int8 weights and the GPU shows 12.9 GB used
after warm-up (13.4 GB at the end of the batched load tests). Qwen2.5-3B was therefore not needed.

## Performance (T4, K40A12)

| metric | Qwen2.5-0.5B | Qwen2.5-7B | how it was measured |
|---|---|---|---|
| round trips per token (layers x 4) | 96 | 112 | pack manifest |
| **mask storage per token** | 1.52 MB | **6.95 MB** (6.96 MB on disk, encrypted) | manifest; store size / slots |
| **wire bytes per token** (up + down) | 2.39 MB (0.89 + 1.50) | **10.96 MB** (4.10 + 6.86) | gateway counters / (prompt + generated tokens) |
| **service GPU compute, batch 1** | 1.6 ms per call, 0.15 s/token | 4.0 ms per call, **0.45 s/token** | service-reported compute time x round trips (load test 1x1) |
| service GPU compute, 32 rows per call | n/a | 5.2 ms per call, **0.018 s/token** | load test 1x32 |
| service GPU compute, 128 rows per call | n/a | 13.7 ms per call, **0.012 s/token** | load test 1x128 |
| service over HTTP, batch 1 (one session, sequential) | 0.63 s/token | 1.08 s/token | load test 1x1 (client and service share 4 vCPUs) |
| service over HTTP, best batched throughput | 0.25 s/token (8 sessions) | **0.075 s/token** (4 sessions x 32 rows) | load tests; CPU-bound on the 4 vCPUs, the GPU is mostly idle |
| end-to-end decode, 1 request | 0.86 s/token (1.16 tok/s) | **1.48 s/token** (0.68 tok/s) | streaming, gaps between tokens |
| end-to-end prefill (40-token prompt) | 2.9 s TTFT | 12.0 s TTFT (0.30 s per prompt token) | streaming, time to first token |
| end-to-end, 8 concurrent requests | 3.06 generated tok/s | 0.98 generated tok/s | wall clock; the gateway batches all sessions into one round trip per linear |
| **gateway CPU per token, batch 1** | 0.58 CPU-s/token | **1.32 CPU-s/token** | /proc CPU time of the gateway process / tokens (prompt + generated) |
| gateway CPU per token, 8 concurrent | 0.12 CPU-s/token | 0.39 CPU-s/token | same |
| gateway local engine time (plaintext parts) | 0.075 s/token | 0.20 s/token | `local_seconds` counter |
| **factory throughput, GPU** | 1.40 M tokens/h (2.6 ms/token) | **169 k tokens/h** (21 ms/token) | `pgw bench factory --pack --device cuda` (GEMM share 70% / 36%) |
| factory incl. process start, encryption into the store | 139 k tokens/h (736 slots in 19 s) | 31 k tokens/h (736 slots in 85 s) | `pgw factory generate` wall clock |
| pack time (download + pack) | 8 s + 19 s | 76 s + 270 s | `pgw pack` |
| service / gateway start-up | 4 s / 8 s | 14 s / 12 s | until /readyz |

What the numbers say:

* GPU compute matches the prototype: 7B batch 1 is 0.45 s/token (prototype, Llama-3-8B: 0.45 s), and
  128 rows per call amortize it to 0.012 s/token (prototype batch 32: 0.014 s).
* End-to-end latency is dominated by round trips and CPU, not by the GPU. One 7B token is 112
  sequential HTTP round trips (about 4.6 ms each on loopback, 0.52 s/token in `remote_seconds`)
  plus 1.3 CPU-seconds of gateway work. Through HTTP the service tops out at about 13 token-rows/s
  on this 4-vCPU host while the GPU is about 30% busy, so batched throughput is limited by
  per-request CPU work (msgpack, ring packing, Python), not by the GPU.
* The gateway is the bottleneck under concurrency. At 8 concurrent 7B requests its engine time
  (129 s) is larger than the service time (85 s).
* The factory on a T4 produces masks for about 169 k 7B tokens per hour; only 36% of that time is
  GEMMs (the rest is CSPRNG, digit arithmetic and serialization).

### Speed pass (run 7, same T4, Qwen2.5-7B, K40A12)

Changes: digit planes stacked into one GEMM (the weight is streamed once per call, not K/8 times), no
idle batch window on the service, passive OpenMP threads, zero-copy uint64 Freivalds/unmask on the
gateway, one worker-thread hop per round trip. Output is still bit-identical to the reference (text
and logprobs); all 35 GPU tests pass. Raw output: `results/kaggle/run7_speed/`.

| metric (7B) | before (run 4) | after (run 7) | change |
|---|---|---|---|
| **end-to-end decode, 1 request** | 1.48 s/token | **0.86 s/token** (1.16 tok/s) | 1.7x faster |
| time to first token (40-token prompt) | 12.0 s | 8.4 s | 1.4x |
| service GPU compute per call, batch 1 | 4.0 ms (0.45 s/token) | **1.4 ms (0.16 s/token)** | 2.8x |
| service queue wait per call, batch 1 | 2.2 ms | 0.05 ms | batch window removed |
| `remote_seconds` per token (stream) | 0.52 s | 0.27 s | 1.9x |
| **gateway CPU, one stream** | 91.3 CPU-s (1.32/token) | **26.0 CPU-s (0.38/token)** | 3.5x less |
| gateway CPU per token, 8 concurrent | 0.39 | 0.27 | 1.4x less |
| 8 concurrent requests | 0.98 generated tok/s | 1.45 generated tok/s | 1.5x |
| service over HTTP, 1x1 load test | 1.08 s/token | 0.57 s/token | 1.9x |
| service over HTTP, 4 sessions x 32 rows | 13.3 token-rows/s | 14.1 token-rows/s | unchanged (CPU-bound) |

Remaining decode time per token is roughly 0.27 s remote (112 round trips at about 2.4 ms, of which
1.4 ms is GPU) plus about 0.59 s on the gateway (plaintext engine about 0.14 s, the rest is per-call
Python overhead, the fp32 LM head and event-loop scheduling). Further large single-stream gains need
fewer sequential round trips per token (speculative decoding), not faster calls.

### Ring matmul kernel (`pgw bench ring`, T4, ms per call, all results exact)

| shape (out x in) | ring | 1 row | 32 rows | 256 rows | CPU int8 (4 vCPU), 1 row |
|---|---|---|---|---|---|
| Llama-3-8B qkv 6144x4096 | K32A8 / K40A12 / K48A20 | 1.25 / 1.61 / 1.77 | 1.39 / 1.53 / 1.76 | 7.4 / 8.6 / 10.1 | 34 / 48 / 53 |
| Llama-3-8B gate_up 28672x4096 | K32A8 / K40A12 / K48A20 | 4.62 / 6.16 / 7.31 | 4.91 / 6.12 / 7.34 | 24.3 / 30.0 / 36.2 | 164 / 216 / 240 |
| Llama-3-8B down 4096x14336 | K32A8 / K40A12 / K48A20 | 2.74 / 3.05 / 4.11 | 2.78 / 3.30 / 4.12 | 14.1 / 17.5 / 20.9 | 79 / 101 / 119 |
| Qwen2.5-0.5B gate_up 9728x896 | K32A8 / K40A12 / K48A20 | 1.09 / 1.33 / 1.58 | 1.09 / 1.31 / 1.57 | 6.0 / 7.6 / 9.2 | 13 / 15 / 19 |

Peak 1.24 T ring-MAC/s (K32A8, 256 rows). CPU int8 on the Kaggle host is about 0.6 G ring-MAC/s.

## Tests on GPU

| run | result |
|---|---|
| `pgw doctor` | all ok: cuda_int8 self test exact for K32A8/K40A12/K48A20, Freivalds accepts genuine and rejects tampered results (after fix 1; before it the CUDA probe failed) |
| `pytest -m gpu` before the fixes (`run2_tests/pytest_gpu.txt`) | 14 passed, 21 failed: every GPU test with an output dimension that is not a multiple of 32 (bug 3) |
| full suite with the e2e service on CUDA, before fix 3 (`run2_tests/pytest_full.txt`) | 1395 passed, 24 failed, 24 skipped (21 GPU small-shape failures, `pgw doctor` self test, factory bench test, and the vectors test that needs `rust/tools`, left out of the first tarball). **All 7 e2e tests passed with the service on the T4** (Qwen2.5-0.5B, real processes, recording proxy) |
| **after all fixes** (`run5_verify/`) | `pytest -m gpu`: **35 passed, 0 failed**; the suite's earlier failures (`tests/core/test_misc.py`, `tests/factory/test_bench_cli.py`, `tests/vectors`): 22 passed, 2 skipped; `tests/e2e` with the service on CUDA (Qwen2.5-0.5B): **32 passed** |

## Bugs found and fixed

1. **CUDA int8 GEMM layout (engine and probe).** On T4 with CUDA 13, cuBLASLt rejects `torch._int_mm`
   with a row-major `mat2` for most shapes (`CUBLAS_STATUS_NOT_SUPPORTED`, for example 40 rows of
   7B qkv). `pgw.engine.backends.exact_int_matmul` (used by `PlainBackend` on CUDA, the QuantPlain
   reference) passed a row-major `(n, m)` copy, and `pgw.core.ringmm.cuda_int8_available()` probed
   that layout, so the probe reported "no cuda_int8" on a working GPU (doctor warn, and an explicit
   `backend=cuda_int8` would refuse). Fix: the engine keeps CUDA weights row-major `(m, n)` and passes
   the transposed (column-major) view, the layout the service and factory already used; the probe
   tests that layout.
2. **Compute service never ready on T4.** The `/readyz` device check ran a 1x8 product, which the T4
   rejects, so readiness failed forever and the gateway could not start. Fix: the check now uses a
   64x64 product with extreme digits, checked against a Python-int reference, and logs the exception
   message.
3. **Small or odd dimensions on CUDA.** cuBLASLt int8 GEMMs on T4/CUDA 13 also reject output or inner
   dimensions that are multiples of 8 but not of 32 (measured: m in {8, 24, 40} and m=16 with k=24;
   see `run3_7b/probe_int_mm.txt`). All CUDA paths (`core.ringmm.PreparedWeight`, service and factory
   kernels, engine) now pad output and inner dimensions to multiples of 32 (`_CUDA_ALIGN`), and inner
   chunks are 131040 wide (a multiple of 32) instead of 131064. Real models were not affected (their
   dimensions are multiples of 64), but the self test, small models and the GPU unit tests were.

The 7B and 0.5B end-to-end runs above already included fixes 1 and 2; fix 3 does not change any real
model shape (all are multiples of 64).

Regression tests: `tests/engine/test_backends.py::test_exact_int_matmul_cuda` (both weight layouts,
shapes up to 7B qkv) and `test_plain_backend_cuda_matches_cpu`. The existing GPU tests in core,
service and factory cover bug 3. `tests/e2e` can now put the service on a GPU with
`PGW_E2E_SERVICE_DEVICE=cuda`.

## Reproduce

```bash
python benchmarks/kaggle/build.py /tmp/kernels            # embeds the current source tree
kaggle kernels push -p /tmp/kernels/pgw-gpu-tests          # tests, ring benches, 0.5B end to end
kaggle kernels push -p /tmp/kernels/pgw-gpu-7b             # 7B end to end (about 18 minutes)
python benchmarks/gpu_validate.py --model Qwen/Qwen2.5-0.5B-Instruct --label qwen0.5b --device cuda --out results/
```
