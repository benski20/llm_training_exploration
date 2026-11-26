# Scaling: many users and conversations on one client machine

One `pgw.OpenAI` client serves many threads and tasks at once: concurrent requests are batched into the same
round trips (one trip per layer step carries a row for every running answer). This page says what limits a
client machine as the number of concurrent streams grows, how to size it, and how to run several worker processes.

## What limits a client, in order

1. **Mask production (CPU).** Every generated token and every prompt token that is not reused from the previous
   turn consumes one mask slot, and each slot costs client CPU in the background factory: about 7 ms at 0.5B,
   25 ms at 7B, 80 ms at 32B and 120 ms at 72B (pks-v3, one core). At 7B, 16 streams of 17 tokens/s with
   500-token prompts need about 800 slots/s, i.e. 20 cores of mask workers. This is the dominant cost at every
   size: it is linear in tokens and has no batching gain. Prompt tokens cost as much as answer tokens, so
   conversation prefix reuse (below) is the main lever.
2. **Mask stock (disk).** The stock absorbs bursts above the factory rate. With the default admission, a request
   starts only when the stock covers the whole remaining prompt + `max_tokens` of every running request, so N
   concurrent requests need about N x (prompt + max_tokens) slots on disk. One slot is 1.8 MB at 0.5B, 6.2 MB at
   7B, 19 MB at 32B and 28 MB at 72B. The default `mask_store_gb = 8` holds about 1,300 slots at 7B: **one** request
   with a 500-token prompt and the default `max_tokens = 512` at a time. Raise `mask_store_gb` (table below) or use
   `mask_admission="overcommit"`.
3. **Gateway CPU, one core per process.** The masking, checks, float ops and sampling of every running answer
   run on one plaintext thread per process (the GIL). Batching amortises the per-trip part (measured: client CPU
   per token falls from 7.2 to 2.4 ms between 1 and 64 streams on the tiny model), but the per-row part (unmask,
   checks, sampling over 152k logits) grows with the streams: one process tops out at about 4-5 streams at 7B-72B
   decode speeds. Above that, run several worker processes (below); they share one mask store and one factory
   (measured, tiny model, 64 streams on 4 vCPUs: 211 tokens/s in one process, 296 in four).
4. **The fast trip lane: all answers of a step in one call** (lane format 2, on in the SDK). When several answers
   share a decode step (or a prompt has more rows than one lane call serves), the batched trip goes over the
   TCP+TLS lane as one `multi` call: the same items, order, bytes and request id as the WebSocket channel's
   request, under the connection's tenant (`rust/pgw-lane/src/proto.rs`). Before, any trip with two items fell
   back to the stream channel (0.35 vs 0.22 ms per idle trip, 2.2 vs 0.28 ms with busy Python threads). Measured
   on this 4-vCPU box (CPU provider, `benchmarks/sdk_concurrency_bench.py`, before / after): tiny at 4 streams
   245-263 / 294-297 trips/s, 99-107 / 119-120 tokens/s, every trip on the lane (before: 1 of 633); 0.5B at 4
   streams 156 / 179 trips/s, 4.1 / 4.7 tokens/s, at 16 streams 56 / 61 trips/s. One multi call costs the client
   about what one single call does (`benchmarks/lane_multi_bench.py`, 0.5B gate_up, 16 answers: 0.32 ms client
   CPU vs 1.1 ms on the stream channel and 1.5 ms for 16 parallel single calls over 16 connections). A provider
   without format 2 (an older one, or `PGW_SERVICE_LANE_MULTI=0`) gets one item per call, as before.
   The per-stream cost that remains with many streams is the plaintext thread (3).
5. **Provider batching** is not a client limit: the service concatenates the rows of a trip and of concurrent
   clients (CPU kernels here; on a GPU the per-row cost is small).

The GIL is not a separate limit from 3: the client's other threads (event loop, decrypt-ahead, mask retirement)
use 0.1-0.3 of a core next to the plaintext thread.

## Sizing table

`python benchmarks/sdk_scale_sizing.py` (arithmetic on the model shapes and on per-unit costs measured on CPU,
see the script). Assumptions: each stream decodes at the single-stream speed of an H100 provider at 0.2 ms RTT
(0.5B 30, 7B 17, 32B 6.7, 72B 5.1 tokens/s), 500-token prompts with no prefix reuse, 256 generated tokens per
request, `max_tokens = 512`, a stock that also absorbs 30 s of full-rate demand. "Cache" is the one-time model
data and mask setup (not per stream). Cores are physical cores with AVX-512 VNNI or AMX for the factory.

| model | N | mask slots/s | factory cores | gateway processes | gateway cores | cores total | RAM GB | mask stock GB (entitled / overcommit) | cache GB |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B | 1 | 89 | 0.6 | 1 | 0.6 | 1.3 | 1.6 | 7 / 6 | 2.2 |
| 0.5B | 4 | 354 | 2.6 | 1 | 1.0 | 3.6 | 1.7 | 27 / 23 | 2.2 |
| 0.5B | 16 | 1,418 | 10.2 | 4 | 4.1 | 14.3 | 6.8 | 108 / 93 | 2.2 |
| 0.5B | 64 | 5,670 | 40.8 | 13 | 14.8 | 55.6 | 22.7 | 431 / 371 | 2.2 |
| 7B | 1 | 50 | 1.3 | 1 | 0.5 | 1.8 | 3.4 | 16 / 12 | 13 |
| 7B | 4 | 201 | 5.0 | 1 | 0.9 | 5.9 | 3.9 | 63 / 50 | 13 |
| 7B | 16 | 803 | 20.0 | 3 | 3.1 | 23.1 | 12.5 | 251 / 200 | 13 |
| 7B | 64 | 3,213 | 80.0 | 11 | 11.8 | 91.8 | 46.9 | 1,004 / 799 | 13 |
| 32B | 1 | 20 | 1.6 | 1 | 0.5 | 2.1 | 4.9 | 31 / 21 | 45 |
| 32B | 4 | 79 | 6.3 | 1 | 0.8 | 7.2 | 7.5 | 125 / 85 | 45 |
| 32B | 16 | 317 | 25.3 | 3 | 2.9 | 28.2 | 25.7 | 501 / 341 | 45 |
| 32B | 64 | 1,266 | 101.3 | 10 | 10.6 | 111.9 | 94.5 | 2,005 / 1,365 | 45 |
| 72B | 1 | 15 | 1.8 | 1 | 0.5 | 2.3 | 7.0 | 42 / 27 | 91 |
| 72B | 4 | 60 | 7.3 | 1 | 0.8 | 8.1 | 10.2 | 167 / 109 | 91 |
| 72B | 16 | 241 | 29.1 | 3 | 2.9 | 32.0 | 34.8 | 668 / 434 | 91 |
| 72B | 64 | 964 | 116.4 | 10 | 10.7 | 127.2 | 127.1 | 2,671 / 1,738 | 91 |

Multi-turn chat with prefix reuse roughly halves the factory and the stock: with 80 % of each prompt reused, 16
streams need 9.4 / 11.9 / 13.7 factory cores and 133 / 282 / 381 GB of stock (entitled) at 7B / 32B / 72B
(`--cached-frac 0.8`). The factory may run on other cores than the gateway (it runs at low priority); disk is
local SSD next to the cache (never a network filesystem: the store refuses one).

RAM per process is dominated by the float32 LM head (0.6 / 2.2 / 3.1 / 5 GB) plus about 1 GB of runtime; per
stream, the KV cache (float32: 25 KB per token at 0.5B, 115 KB at 7B, 524 KB at 32B, 655 KB at 72B; a 4k-token
conversation is 0.5 / 2.1 / 2.7 GB at 7B / 32B / 72B) and about one mask slot of decrypted masks.

## Settings for many streams

```python
from pgw import OpenAI

client = OpenAI(
    max_concurrent=8,             # answers generating at once per process (more wait in a FIFO queue)
    mask_workers=16,              # factory processes (default: a quarter of the CPUs, at most 8)
    mask_store_gb=250,            # stock on local disk (see the table)
    # mask_admission="overcommit",  # opt-in, below
)
```

* `max_concurrent` (default 8) bounds the streams batched per process; requests above it wait in a FIFO queue
  (`timeout` bounds the wait). Keep it at 4-8 for 7B and larger (gateway CPU, lane rows).
* `mask_workers` and `mask_store_gb`: from the table. `pgw sdk status` shows the stock, the factory's
  throughput and how many requests wait for masks (`requests_waiting_for_masks`).
* `prefix_cache` (default `"auto"`): continuing a conversation reuses the previous turn's KV state, and its
  tokens take no masks. It is on when the output layer runs in your process (`masked_head="off"`, or when the
  service does not serve the masked head). With the masked head (the default where served) every turn
  recomputes the whole conversation and pays masks for it. `prefix_cache=True` turns the masked head off: the
  output layer then runs on the client CPU (about 47 ms per generated token at 7B, float32), while every reused
  token saves a mask slot (25 ms at 7B) and its prompt pass. A turn that reuses 2,000 tokens and generates 256
  saves about 50 CPU-seconds of masks for about 12 of head work: for long multi-turn chats, prefer
  `prefix_cache=True`; for single-turn traffic keep the default. Reuse is per conversation key (API key, OpenAI
  `user`, `extra_body={"conversation_id": ...}`): never across users or conversations.
* KV memory: the prefix cache keeps at most 1 GiB of conversations (least recently used are wiped first).

## When the stock runs short (bursts)

* Requests that find too few masks wait (back-pressure) up to `mask_wait_s` (120 s), then fail with
  `NotReadyError` (retryable). Waiting requests are served **in arrival order**: once one waits, later requests
  queue behind it, so a large request is never starved by a stream of small ones. Only running answers keep
  generating while the queue waits.
* A request that could never fit (its prompt + `max_tokens` exceed what the stock can hold, `mask_store_gb`) fails
  at once with `BadRequestError` naming `mask_store_gb`, instead of waiting.
* `mask_admission="overcommit"` (opt-in, `PGW_MASK_ADMISSION`): a request starts when the stock covers its prompt
  and one chunk (8 slots) for every running answer; an answer that finds the stock short waits for the factory
  (up to 30 s, then fails closed). Under a burst every answer slows to the factory's rate instead of new requests
  queueing, and the stock no longer has to hold every request's `max_tokens`. The waits happen inside the batched
  step, so a stalled answer stalls the others of its process for that time.
* Single use of masks never depends on any of this: the store advances its durable cursor before any mask is
  used, so a crash or a failed request loses masks but never reuses them.

## Several worker processes (gunicorn, uvicorn workers, Celery)

Processes on one machine with the same key, `cache_dir` and `state_dir` share:

* **one enrollment** and the model cache;
* **one mask store**: every reservation advances the store's cursors in one SQLite `BEGIN IMMEDIATE` transaction
  (fsync) under an exclusive `flock`, so no slot is ever handed to two processes, and rollback of the store is
  detected (HMAC journal, monotonic counter);
* **one factory per model**: whichever process holds the factory's owner lock runs it; the others take over
  when it exits. Workers do not duplicate the factory;
* **one admission ledger** (`<cache>/sessions/<key>/mask-ledger/`): each process publishes the masks promised to
  its running answers, and an admission counts every live process's promises under a cross-process lock, so one
  worker's new request never takes the stock another worker's running answer will need.

Recommended pattern: **one client per worker process, created after the fork** (in gunicorn's `post_fork` or
lazily on first use; a client created before a fork refuses to run in the child), `max_concurrent` 4-8 per worker
for the most tokens per core or 1-2 for the lowest latency per stream (the plaintext thread, above), all on the same
`cache_dir` / `state_dir` on local disk. The factory's `mask_workers` count once per machine, not per worker
(every worker passes the same value; only the owner runs them). No separate mask daemon
is needed: the factory already runs once per machine and model. Do not share one cache directory between
machines (each machine enrolls and keeps its own store).

```python
# app.py (gunicorn -w 4 app:app): one client per worker, created on first use, after the fork
import pgw
client = None

def get_client():
    global client
    if client is None:
        client = pgw.OpenAI(max_concurrent=6, mask_workers=16, mask_store_gb=250)
    return client
```

## Fairness between users of one client

* Admission is FIFO across all users of a client (`max_concurrent` running, the rest queued in arrival order);
  within a step every running answer gets one token, so a long answer does not slow a short one's tokens.
* The OpenAI `user` field keys the conversation prefix cache (no reuse across users) and, with `speculation`,
  orders a user's requests behind the previous one's remaining dummy rounds; it never delays another user.
* There is no per-user quota inside the client: one user with many concurrent requests can occupy all
  `max_concurrent` slots. Apply per-user limits in your application if users are untrusted.

## Many clients per provider

The service admits by tenant (one tenant per SDK key): `tenant_max_inflight` (concurrent requests),
`tenant_rows_per_s` / `tenant_burst_rows` (token bucket on rows), the global `max_inflight_requests` (256) and
`max_queued_rows`; a refused call gets `unavailable` with `Retry-After` and the client retries it. Each client
process holds one lane connection set per model; a single lane call serves at most 8 rows, a multi call up to 64 items and 256 rows. Size the provider by rows
per second (sum of all clients' tokens/s, prompt rows included); the client side never needs changes for it.

## Measuring

`benchmarks/sdk_concurrency_bench.py` runs N concurrent streams (and `--procs P` worker processes) against a
local provider, checks every answer against the same request run alone, and reports aggregate and per-stream
tokens/s, TTFT, rows per trip, lane fallbacks and CPU per token by component.

Measured on a shared 4-vCPU box with the provider on the same CPU (CPU kernels; other jobs running, load 4-28),
full stock, streamed sampled chats, every answer bit-identical to the same request alone, no errors:

| model | N | procs | tokens/s total | tokens/s per stream | TTFT p50 / p95 s | rows per trip | lane trips | client CPU ms/token | provider CPU ms/token |
|---|---|---|---|---|---|---|---|---|---|
| tiny | 1 | 1 | 45 | 49 | 0.07 / 0.07 | 2.2 | 280 of 312 | 6.5 | 3.1 |
| tiny | 4 | 1 | 98 | 28 | 0.20 / 0.20 | 7.4 | 1 of 345 | 3.9 | 2.7 |
| tiny | 16 | 1 | 173 | 12 | 0.35 / 0.51 | 27 | 1 of 378 | 2.6 | 1.2 |
| tiny | 64 | 1 | 211 | 4.1 | 1.39 / 2.30 | 75 | 1 of 543 | 2.4 | 0.7 |
| tiny | 64 | 4 | 296 | 5.6 | 1.00 / 1.50 | 25 | 13 of 1638 | 2.8 | |
| 0.5B | 1 | 1 | 2.1 | 2.6 | 1.9 / 1.9 | 2.7 | 71 of 1840 | 123 | 268 |
| 0.5B | 4 | 1 | 4.4 | 1.7 | 5.6 / 5.6 | 8.7 | 1 of 2225 | 95 | 131 |
| 0.5B | 8 | 1 | 4.9 | 1.1 | 12.0 / 12.0 | 17 | 1 of 2225 | 78 | 114 |

At 0.5B on this box the CPU provider saturates (its kernels, two workers); the client's plaintext thread is 20-27 %
busy. Client CPU per token includes the prompt pass (about 30 prompt rows per 16 generated tokens). Burst with a
stock of 512 slots and 64 tiny streams (factory running): every stream served, bit-identical, TTFT p95 14.3 s with
the default admission and 11.6 s with `overcommit` (129 vs 142 tokens/s).
