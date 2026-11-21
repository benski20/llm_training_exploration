# X4: aggregate online throughput with many concurrent chats

Date: 2026-11-21. Scope: how many generated tokens/s one co-located gateway + one T4 provider serve with 1 to 64
concurrent chats (masks pre-filled, so the mask factory is not the limit), what limits it, and exact, privacy-neutral
fixes. Tags as in `scale-north-star.md`: **measured**, **derived**, **estimate**.

## 1. Summary

1. **Cross-session batching already exists and works.** The engine's continuous-batching loop puts the rows of every
   live chat into one round trip per linear (`Generator._run_loop` -> `forward_many`), and the service concatenates the
   rows of one linear into one kernel call (`service/batcher.py`). Trips per step stay `4 L`; rows per trip = live chats.
2. **The limit is one CPU core of the gateway.** Every engine step runs its plaintext work (finish/verify of the
   previous trip, norms, RoPE, attention per sequence, the next trip's masking) on one plaintext thread, so the step
   time grows linearly with the number of chats and aggregate tokens/s flattens at about `1 / (gateway CPU per row)`.
   At 0.5B shapes that is about 15 ms of one core per generated token on the 4-vCPU dev box (measured: masking 7.9 ms
   per row over 96 trips, `profile_gateway --sessions 32`; float ops + LM head + sampling about 6 ms per row,
   `profile_token --rows 32`). Neither the service nor the network is close to its limit.
3. **More threads in one process do not help.** Two plaintext threads with two cohorts (ping-pong) were slower than
   one (158 vs 283 tokens/s, measured, 32 chats, null service): the per-sequence Python work holds the GIL.
4. **Fix 1 (in product, exact): `float_threads` now defaults to 1** on CPU engines. Decode float ops on one torch
   thread instead of the process count; the existing start-up bit-identity self-check still gates it and falls back.
   Measured (`profile_token --preset 0.5b`, 4 vCPUs): engine step 13.6 -> 11.6 ms with 1 chat, 56 -> 40 ms with 8,
   214 -> 136 ms with 32 (attention 72 -> 20 ms).
5. **Fix 2 (deployment, exact): several gateway processes per host.** Processes share the mask store (its file lock
   serializes reservations) and the service (which coalesces rows across connections). Measured on the dev box with
   time-aligned windows (`e2e_local --null-service --start-at`, 32 chats split evenly, synthetic 4-layer 0.5B-width
   model): 1 process 317, 2 processes 538 (1.7x), 4 processes 923 tokens/s (2.9x). Kaggle T4 numbers: section 3.

## 2. Method

- `benchmarks/gpu_validate.py` (real processes, real CLI): pack, plaintext reference, factory fills the store, then
  `pgw service serve` on the GPU and `pgw gateway serve` on 127.0.0.1. Each concurrency level sends n identical
  greedy requests at once; every answer is compared token for token with the reference. New: level 1, gateway and
  service cores used per level (`gateway_cores` = gateway CPU seconds / wall), `--variant NAME=JSON` (a model-config
  A/B on a fresh gateway over the same store), `--shard-gateways N` (N gateway processes on one store and service,
  request i to gateway i mod N).
- `benchmarks/e2e_local.py --null-service` isolates the gateway (zero weights, a service stub returning zeros).
  New: `--float-threads`, `--profile`, `--start-at` (align the timed window of several processes).
- Kaggle job `pgw-x4-concurrency` (`benchmarks/kaggle/build.py`): 0.5B at 1/8/32/64 chats (default and
  `float_threads 1`), 64 chats over 2 and 3 gateways, service-only load tests at 1/8/32/64 sessions; 1.5B at 1/64 and
  64 over 2 gateways.

## 3. Results

CPU-side only so far (all **measured** on the shared 4-vCPU dev box, null service; GPU runs were not done in this
package: the Kaggle quota was busy and GPU measurements are run centrally after merge).

| what | 1 chat | 8 chats | 32 chats |
|---|---|---|---|
| plaintext engine step, 0.5B shapes, default threads (`profile_token`) | 13.6 ms | 56 ms | 214 ms |
| same, `float_threads 1` | 11.6 ms | 40 ms | 136 ms |
| gateway masking per row-token, 0.5B, 96 trips (`profile_gateway --sessions N`) | 15.5 ms | 8.9 ms | 7.9 ms |
| e2e tokens/s, synthetic 4-layer model, 1 gateway process (`e2e_local --null-service`) | 143-163 | 202-232 | 234-317 |

Gateway processes sharing 32 chats (aligned windows, `float_threads 1`): 1 process 317, 2 processes 538, 4 processes
923 tokens/s. Two plaintext threads in one process: 158 vs 283 tokens/s (slower).

**Estimate** for the T4 run (0.5B, real service): one gateway process flattens near `1 / 15 ms` = about 60 generated
tokens/s from about 16 chats on, and 2-3 processes should approach 2x on the 4-vCPU Kaggle VM (2 cores, plus the
service's own CPU). The command to measure it is in section 7.

## 4. Privacy and exactness

- **float_threads = 1** changes only how many OpenMP threads the gateway uses for its own plaintext float ops of
  decode passes. Nothing the provider sees changes (same trips, rows, bytes, schedule). Exactness: the per-row-count
  self-check (`float_threads_self_check`, rows 1..8, cache lengths to 4100) compares the bits with the process
  thread count at start-up and keeps the old count if any output differs; the gateway tests compare answers and
  logprobs with the plaintext reference for the default, explicit and environment settings.
- **Sharded gateways** are independent, complete gateways: each holds its own sessions, reservations (from the shared
  store, each slot used once: the store's cross-process lock and cursors), engine and checks. The provider sees N
  connections whose trips carry `rows = live chats of that gateway`; the split of chats over gateways is round-robin
  (or by API key), independent of any content, and the per-trip row count was already public. No secret leaves any
  gateway; Freivalds and range checks run unchanged in each. Outputs are bit-identical because a sequence's logits do
  not depend on what shares its pass (tested by the engine suite; checked on every request in the benchmark).
- Speculation's strict-pass conversation gate (`spec_strict_passes`) is per gateway process: requests of one
  conversation (API key + OpenAI `user`) must stay on one gateway, so route by API key, not per request.

## 5. Deployment guidance (co-located gateway host)

- Run one gateway process per 1-2 physical cores, each on its own port with its own audit log file, all on the same
  mask store and API keys file, behind an L4 balancer with affinity by API key. `max_concurrent_sessions` per process
  = expected chats / processes.
- Keep `float_threads` at its default (1).
- Leave cores for the service process when it is on the same host (it used CPU on every trip in the Kaggle run).

## 6. What is left

- An in-product multi-process engine (one gateway front end, N engine worker processes) so that quotas, the console
  and the conversation gate stay global. Today sharding is a deployment pattern.
- Per-row gateway CPU: the masking path (about half) and per-sequence attention (exact batching across sequences of
  different cache lengths is not bit-identical in general, so it needs a deterministic attention kernel).
- 7B and larger: run the same job (`--shard-gateways`) to size gateway cores per 1k tokens/s.

## 7. GPU measurement to run

```
python benchmarks/gpu_validate.py --model Qwen/Qwen2.5-0.5B-Instruct --label qwen05_conc --device cuda \
  --ref-device cuda --ring K32A12C27 --plain-dtype bfloat16 --delete-hf --max-tokens 24 \
  --concurrent 1 8 32 64 --variant 'ft_off={"float_threads": null}' \
  --shard-gateways 2 3 --shard-concurrent 64 \
  --loadtest 1x1 8x1 32x1 64x1 --loadtest-tokens 4 --factory-bench-slots 64 --out results/x4
```

Measures generated tokens/s, gateway and service cores and bit-identity at 1/8/32/64 concurrent chats on one gateway
(new default `float_threads 1`), the same on the old thread count (`ft_off_*`), 64 chats over 2 and 3 gateway
processes (`shard<N>_concurrent_64`), and the service alone at 1-64 sessions (`service_loadtest`). About 16 GB of
masks; roughly 20-30 min on a T4 (pack, reference and factory included), less on an H100 (**estimate**). Kaggle job
entries with the same steps: `pgw-x4-concurrency` (0.5B + 1.5B) and `pgw-x4-concurrency-7b`.
