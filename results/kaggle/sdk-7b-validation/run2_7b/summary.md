# SDK end to end on Qwen2.5-7B-Instruct, Kaggle T4: run 2

**Result: every check passed on attempt C.** Attempt C ran the code at `e73fec8`; attempts A and B are kept as the record of what failed and why.

The setup: the provider service ran on one T4 (16 GB, `cuda:0`), and the client library (`from pgw import OpenAI`) ran in the same VM over loopback (TCP+TLS lane, 4 vCPUs, 32 GB RAM). The client used the library's defaults: `mask_scheme="pks-v3"`, `hint="local"`, `masked_head="auto"` (on), and an empty cache. The harness is `pgw/benchmarks/sdk_e2e_gpu.py` (Kaggle job `pgw-sdk-e2e-7b`).

**T4 numbers check correctness; they do not measure speed.** A T4 has no int8 tensor-core path for the fused GEMV (compute capability 7.5). The client's 4 vCPUs also build the hint and run the mask factory on the same VM as the provider. Speed has to be measured on the target hardware.

## Attempts

| | code | kernel | outcome |
|---|---|---|---|
| A | `bfee514` (run1's OOM fix) | v2, 00:14-03:16 UTC | The provider loaded. 66 min in, the provider process died while serving pks-v3 maskgen (old host hint cache). The first request then waited until the 3 h step timeout. |
| B | `1bdfd27` (+ provider device-hint fix `3e347e1`, harness watchdog) | v3, 03:27-04:09 | The provider stayed up and the first answer arrived. The next request was refused with `mask_exhausted`: 42 slots needed, 14 in stock. |
| C | `e73fec8` (+ harness waits for mask stock) | v4, 04:19-05:59 (95 min of e2e) | **All checks passed.** |

## Attempt C checks (`c_e73fec8/sdk_e2e/results.json`)

| check | result |
|---|---|
| GPU unit tests (`test_kernels.py`, `test_kernels_memory.py`, `test_ringmm.py`, `-m gpu`) | PASS (21 passed) |
| provider pack fingerprint == baked pin (`f7c6fb4f…`) | PASS |
| provider loads on the T4 | PASS. 6.54 GB on `cuda:0`; the memory plan says it fits; startup took 47 s |
| enroll + provision (gateway and factory packs, fingerprint-checked) | PASS |
| local pks-v3 hint built | PASS. 115 files, 2.37 GB |
| bit-identical vs the gateway path (text + log-probs incl. top alternatives), non-streaming: greedy, turn 0/1/2 | PASS |
| bit-identical, streaming: sampled (T=0.8, top_p 0.95, seed), long greedy stream x2 | PASS |
| `bit_identical_all` | PASS |
| 3-turn conversation | PASS. "Nice to meet you, Ada, and enjoy your tea!" / "Your name is Ada, and you like tea." / "Crackers with cheese and tea." |
| no integrity failures (22,352 Freivalds checks, 0 skipped) | PASS |
| factory pks-v3, not halted, 0 provider integrity failures | PASS |
| no content in the provider's usage / request records | PASS |
| revoked key refused: connected client | PASS (`AuthenticationError: unauthenticated`) |
| revoked key refused: fresh client | PASS (`AuthenticationError: unauthenticated`) |

The bit-identity comparison is the same as in `tests/sdk/test_e2e_product.py`. For each recorded answer it compares the content and the full `logprobs.content` list with `==` against the gateway's plaintext reference mode (`pgw gateway` app, `head_mode=provider_int8`, same model settings).

## Attempt C numbers (T4; correctness run, not speed)

| phase | time |
|---|---|
| provider: checkpoint download / pack from the pinned checkpoint | 73 s / 310 s |
| provider start (pack verified, kernels, warm-up, lane graphs) | 47 s |
| client `warmup()`: enroll + provision + connect | 233 s |
| provisioning (client log) | 182 s: catalog 24 s, artifacts 2.2 GB in 26 s, **weight stream 6.5 GB in 116 s** (~56 MB/s, loopback, verified) |
| **local hint build** (factory start to the last hint file) | **17.3 min** (1,039 s; 04:33:08 to 04:50:27) |
| first mask segment (64 slots) | 196 s; then 128 slots in 188 s and 512 slots in 266 s (1.9 slots/s overall) |
| first answer after `warmup()` (= hint build + first segment + request) | 1,246 s |
| **TTFT**, 64-token greedy stream, ~30-token prompt | 3.95 s / 3.91 s |
| **steady decode**, same stream | 2.21 tok/s (factory segment running on the same T4) / **8.14 tok/s** (factory idle) |
| turns (<= 32 tokens, growing history) | 5.4 s / 7.5 s / 8.6 s |
| gateway reference on CPU (plaintext 7B, 4 vCPUs) | 3,247 s |

Other numbers:
- Lane: 22,240 trips, 56,512 rows, 1.68 GB up and 2.90 GB down.
- Provider usage: 6 requests, 230 prompt and 168 completion tokens, all verified.
- Memory peaks: provider RSS 2.6 GB, GPU 9.1 GB of 15 GB, host MemAvailable never below 11.9 GB.
- Two mask-stock waits were recorded: sampled stream 60 s, long stream 140 s (`mask_stock_waits` in results.json).

## What failed in A and B, and what changed

**Run 1.** The CUDA OOM at startup (int64 weight copy in `_rowsum`, duplicate device weight) is fixed by `bfee514`. Attempt A confirms the fix: the provider loaded in 49 s.

**Attempt A: what ran out was host RAM, from the provider's pks-v3 hint cache.**
- The client provisioned in 232 s, and its factory started at t = 755 s.
- From t ~ 2150 s the provider served pks-v3 maskgen. The old path cached the switched hint `m x 4096` uint64 per (seed, linear) on the host: 50.5 GB per seed at 7B, against an 8 GB cache. For the 152k-row LM head, each item built multi-GB host temporaries, plus a 5 GB device result.
- Host MemAvailable swung between 4 and 15 GB, and GPU use reached 13.1 GB.
- At t ~ 3976 s the provider process died. GPU use fell to 413 MiB, and host MemAvailable jumped from 3.9 to 16.7 to 26.4 GB.
- The step timeout killed the harness, so `provider.log` was never copied. The kernel cannot read `dmesg`, so "host OOM killer" is inferred from the RAM low and from what the exit freed.
- The provider GPU agent's `3e347e1` ("pks-v3 provider hint recomputed per item on the device") removes that cache. With it (attempts B and C), provider RSS stayed <= 2.6 GB and GPU use <= 9.1 GB.

**Attempt A, client side: reported, not fixed.** After the provider died, the client's first request kept waiting for the rest of the 3 h budget (the harness passes `timeout=7200`, `ready_timeout_s=7200`). The factory reported `ServiceUnavailableError: compute service transport error (ConnectError)` but did not count it as a failure (`consecutive_failures` stayed 0), and the request was not failed. That is client reliability (sdk/factory) behaviour, so it is left to the reliability work.

**Attempt B.** The second request was refused with `mask_exhausted` after the default `mask_wait_s` (120 s): it needed 42 slots and 14 were in stock. This is correct library behaviour: a refusal happens at admission and uses no mask. On a T4 the factory, not the request, is the bottleneck. The harness now waits for the factory's next segment and sends a refused request again. Each wait is recorded, timings are per successful attempt, and the library options stay at their defaults. No check was dropped or weakened.

**Harness changes (`1bdfd27`, `bcb1226`):**
- The sampler records per-process RSS and copies `provider.log` every minute.
- A watchdog ends the run, with results written, when the provider exits (after 300 s of grace) or when `--budget-s` is spent.
- The 7B step timeout is now 2 h 45 min, with a 2 h 35 min budget, so a kernel stays under 3 h.
- The script exits with `os._exit` after writing its results. One local 0.5B smoke run segfaulted in interpreter teardown after `DONE`, with every check passed; the cause was not investigated.

## Files

- `a_bfee514/`, `b_1bdfd27/`, `c_e73fec8/`: raw kernel output. Each has `summary.json`, the step logs, and `pgw-sdk-e2e-7b.log` (the Kaggle log). `sdk_e2e/` holds `results.json`, `status.jsonl` (client status, host memory, RSS and GPU every 15 s), `client.log`, and `provider.log` (B and C only), plus `sdk_requests.json` and `reference.json` (C only).
- The pack fingerprint differs between A/B (`d3db0ca8…`) and C (`f7c6fb4f…`) because `e1603aa` (pack format 2: bf16 embeddings and LM head kept as in the checkpoint, with new pins) landed between the runs. This is also why the artifacts went from 4.38 to 2.20 GB. In every attempt the provider-derived pack matched the pin baked into that code.
