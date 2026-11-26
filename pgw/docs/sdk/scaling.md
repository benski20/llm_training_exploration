# Scaling: many users and conversations on one client machine

One `pgw.OpenAI` client serves many threads and tasks at once: concurrent requests are batched into the same
round trips (one trip per layer step carries a row for every running answer). This page says what limits a
client machine as the number of concurrent streams grows, how to size it, and how to run several worker processes.

## What limits a client, in order

1. **Mask production (CPU).** Every generated token and every prompt token that is not reused from the previous
   turn consumes one mask slot, and each slot costs client CPU in the background factory: about 11 ms at 0.5B,
   34 ms at 7B, 103 ms at 32B and 157 ms at 72B (pks-v3, one core, every part counted; the breakdown is
   [below](#mask-cost-per-slot-by-component)). At 7B, 16 streams of 17 tokens/s with 500-token prompts need
   about 800 slots/s, i.e. 27 cores of mask workers. This is the dominant cost at every size: it is linear in
   tokens and has no batching gain. Prompt tokens cost as much as answer tokens, so conversation prefix reuse
   (below) is the main lever.
2. **Mask stock (disk).** The stock absorbs bursts above the factory rate. With the default admission, a request
   starts only when the stock covers the whole remaining prompt + `max_tokens` of every running request, so N
   concurrent requests need about N x (prompt + max_tokens) slots on disk. One slot is 1.8 MB at 0.5B, 6.2 MB at
   7B, 19 MB at 32B and 28 MB at 72B (`4 x sum(out_features)`: the masks are uniform 32-bit values, so this is
   the floor; see below). The default `mask_store_gb = 8` holds about 1,300 slots at 7B: **one** request
   with a 500-token prompt and the default `max_tokens = 512` at a time. Raise `mask_store_gb` (table below) or use
   `mask_admission="overcommit"`. The stock's disk must also take the factory's writes: slots/s x slot bytes,
   5 GB/s at 7B with 16 streams (column "stock writes" below), twice that while `shred_zero_fill` (the default)
   zeroes consumed segments; on SSDs set it off (`PGW_MASKSTORE_SHRED_ZERO_FILL=false`: the crypto-erase of the
   segment's key header is what deletes the masks; the zero-fill is useless there).
3. **Gateway CPU, one core per process.** The masking, checks, float ops and sampling of every running answer
   run on one plaintext thread per process (the GIL). Batching amortises the per-trip part (measured: client CPU
   per token falls from 7.2 to 2.4 ms between 1 and 64 streams on the tiny model), but the per-row part (unmask,
   checks, sampling over 152k logits) grows with the streams: one process tops out at about 4-5 streams at 7B-72B
   decode speeds. Above that, run several worker processes (below); they share one mask store and one factory
   (measured, tiny model, 64 streams on 4 vCPUs: 211 tokens/s in one process, 296 in four).
4. **The fast trip lane carries one answer per call.** A lane call takes one item (one answer's rows, at most 8).
   As soon as two answers share a decode step in one process, the batched trip carries two items and goes over
   the WebSocket stream channel instead (correct and bit-identical, slower per trip: measured on a 4-vCPU box
   0.35 vs 0.22 ms per idle trip, 2.2 vs 0.28 ms with busy Python threads; the benchmark below shows 280 of 312
   trips on the lane with one stream and 1 of 345 with four). With 4L + 1 trips per token (113 at 7B, 321 at
   72B) this costs more per-stream speed than batching saves; for the lowest latency per stream, keep few
   streams per process and add processes; for the most tokens per core, batch.
5. **Provider batching** is not a client limit: the service concatenates the rows of a trip and of concurrent
   clients (CPU kernels here; on a GPU the per-row cost is small).

The GIL is not a separate limit from 3: the client's other threads (event loop, decrypt-ahead, mask retirement)
use 0.1-0.3 of a core next to the plaintext thread.

## Sizing table

`python benchmarks/sdk_scale_sizing.py` (arithmetic on the model shapes and on per-unit costs measured on CPU,
see the script). Assumptions: each stream decodes at the single-stream speed of an H100 provider at 0.2 ms RTT
(0.5B 30, 7B 17, 32B 6.7, 72B 5.1 tokens/s), 500-token prompts with no prefix reuse, 256 generated tokens per
request, `max_tokens = 512`, a stock that also absorbs 30 s of full-rate demand. "Cache" is the one-time model
data and mask setup (not per stream). Cores are physical cores with AVX-512 VNNI or AMX for the factory. Mask CPU
per slot is the complete figure of the [breakdown](#mask-cost-per-slot-by-component) (factory 10.6 / 33.5 / 103 /
157 ms; gateway cores include 1.25 / 3.7 / 11 / 16 ms per slot of mask opening and retirement on the gateway's
helper threads). "Stock writes" is the disk write bandwidth of the factory at that rate.

| model | N | tok/s each | mask slots/s | factory cores | gateway procs | gateway cores | cores total | RAM GB | mask stock GB (entitled / overcommit) | stock writes GB/s | cache GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B | 1 | 30 | 89 | 0.9 | 1 | 0.7 | 1.7 | 1.6 | 7 / 6 | 0.16 | 2.2 |
| 0.5B | 4 | 30 | 354 | 3.8 | 1 | 1.5 | 5.2 | 1.7 | 27 / 23 | 0.65 | 2.2 |
| 0.5B | 16 | 30 | 1,418 | 15.0 | 4 | 5.8 | 20.9 | 6.8 | 108 / 93 | 2.59 | 2.2 |
| 0.5B | 64 | 30 | 5,670 | 60.1 | 13 | 21.9 | 82.0 | 22.7 | 431 / 371 | 10.34 | 2.2 |
| 7B | 1 | 17 | 50 | 1.7 | 1 | 0.7 | 2.4 | 3.4 | 16 / 12 | 0.31 | 13 |
| 7B | 4 | 17 | 201 | 6.7 | 1 | 1.6 | 8.3 | 3.9 | 63 / 50 | 1.24 | 13 |
| 7B | 16 | 17 | 803 | 26.9 | 3 | 6.0 | 32.9 | 12.5 | 251 / 200 | 4.96 | 13 |
| 7B | 64 | 17 | 3,213 | 107.6 | 11 | 23.7 | 131.4 | 46.9 | 1,004 / 799 | 19.83 | 13 |
| 32B | 1 | 6.7 | 20 | 2.0 | 1 | 0.7 | 2.8 | 4.9 | 31 / 21 | 0.38 | 45 |
| 32B | 4 | 6.7 | 79 | 8.2 | 1 | 1.7 | 9.9 | 7.5 | 125 / 85 | 1.52 | 45 |
| 32B | 16 | 6.7 | 317 | 32.7 | 3 | 6.4 | 39.1 | 25.7 | 501 / 341 | 6.08 | 45 |
| 32B | 64 | 6.7 | 1,266 | 130.9 | 10 | 24.6 | 155.6 | 94.5 | 2,005 / 1,365 | 24.34 | 45 |
| 72B | 1 | 5.1 | 15 | 2.4 | 1 | 0.8 | 3.1 | 7.0 | 42 / 27 | 0.42 | 91 |
| 72B | 4 | 5.1 | 60 | 9.4 | 1 | 1.8 | 11.2 | 10.2 | 167 / 109 | 1.69 | 91 |
| 72B | 16 | 5.1 | 241 | 37.8 | 3 | 6.6 | 44.4 | 34.8 | 668 / 434 | 6.76 | 91 |
| 72B | 64 | 5.1 | 964 | 151.0 | 10 | 25.8 | 176.8 | 127.1 | 2,671 / 1,738 | 27.04 | 91 |

Before the 2026-11-26 factory changes, with every part counted, the factory needed 20.1 / 53.2 / 73.1 / 104.4
cores at N = 16 (0.5B / 7B / 32B / 72B; `--mask-cpu before`), now 15.0 / 26.9 / 32.7 / 37.8. The earlier version
of this table showed 10.2 / 20.0 / 25.3 / 29.1 (`--mask-cpu documented`): its per-slot figures left out the masked
LM head, the segment's check vectors, the every-row self-check and the segment writer. The stock in GB is unchanged
(bytes per slot are at their floor, below).

Multi-turn chat with prefix reuse roughly halves the factory and the stock: with 80 % of each prompt reused, 16
streams need 12.7 / 15.4 / 17.8 factory cores and 133 / 282 / 381 GB of stock (entitled) at 7B / 32B / 72B
(`--cached-frac 0.8`). The factory may run on other cores than the gateway (it runs at low priority); disk is
local SSD next to the cache (never a network filesystem: the store refuses one).

RAM per process is dominated by the float32 LM head (0.6 / 2.2 / 3.1 / 5 GB) plus about 1 GB of runtime; per
stream, the KV cache (float32: 25 KB per token at 0.5B, 115 KB at 7B, 524 KB at 32B, 655 KB at 72B; a 4k-token
conversation is 0.5 / 2.1 / 2.7 GB at 7B / 32B / 72B) and about one mask slot of decrypted masks.

## Mask cost per slot, by component

`benchmarks/pks_slot_breakdown.py` runs every step the factory and the store perform for one mask slot, with the
functions the production path calls, at the real shapes of each model (one layer of each linear kind plus the
masked LM head, times the layer count; random int8 weights, a stand-in reply of the right size), 1024-slot
segments, p31 check vectors, one thread. Client CPU ms per slot, before -> after the 2026-11-26 changes, measured
on a shared 4-vCPU AMX box under load (4-19):

| part | 0.5B | 7B | 32B | 72B |
|---|---|---|---|---|
| `r` keystream (ChaCha20) | 1.40 -> 1.28 | 7.7 | 30.2 | 45.3 |
| upload encryption + reply keys | 4.78 -> 4.93 | 16.3 | 56.9 | 84.0 |
| reply keys' wire form | 0.38 -> 0.37 | 0.48 | 1.16 | 1.36 |
| reply decryption | 3.14 -> 3.49 | 10.7 | 33.3 | 52.7 |
| int64 copy of `W8 r` | 0.86 -> **0** | 2.9 -> **0** | 9.0 -> **0** | 13.2 -> **0** |
| every-row 40-bit self-check (14 private vectors) | 1.97 -> **0.85** | 8.0 -> **3.1** | 25.4 -> **10.0** | 38.1 -> **15.4** |
| segment check vectors `u W8 mod p` (per segment / 1024) | 2.43 -> **0.15** | 37.9 -> **1.2** | 172 -> **4.8** | 383 -> **9.6** |
| private `v W8` redrawn per segment (cache overflow) | 0 | 0 | 0 | 6.6 -> **0** |
| writer: pack + AES-256-GCM + pwrite + fsync | 2.38 -> 1.89 | 6.1 | 19.1 | 27.6 |
| **factory total, this box** | **17.3 -> 13.0** | **90 -> 46** | **347 -> 155** | **652 -> 236** |
| factory total, scaled to the earlier audit's box | 14.2 -> 10.6 | 66 -> 33.5 | 231 -> 103 | 433 -> 157 |
| (earlier documented figure: r, upload, keys, decryption, no head) | 7.2 | 24.9 | 80 | 121 |
| gateway: AES-GCM open + unpack of the slot's rows | 1.04 -> 1.05 | 3.6 | 11.9 | 16.5 |
| gateway: retirement (crypto-erase, zero-fill, unlink) | 0.70 -> 0.48 | 1.5 | 4.9 | 6.9 |
| private `v W8` at factory worker start (s, once) | 0.3 | 4.2 | 18.9 | 42.3 |
| bytes per slot (MB) | 1.82 | 6.17 | 19.2 | 28.1 |

How measured: 0.5B, before and after, every part measured. 7B / 32B / 72B after: every part measured at the real
shapes. 7B before: the decoder layers measured (`PGW_VECMAT_NATIVE=0 PGW_PKS_COMPACT=0`), the LM head's three
changed parts at the layers' per-element rates (its float64 product does not fit this box's free RAM), the int64
copy at the 0.5B per-element rate (7B measured it 7.9x slower, under memory pressure). 32B / 72B before: the after
figures plus the three changed parts at the 7B-measured rates (5.5 ns per multiply-add for the float64 `u W8`,
3.4 ns per element for the old self-check). Parts without an arrow did not change (one figure: after; run to run
noise is about 10-20 % on this box). "Scaled" multiplies by the ratio of the earlier audit's HE parts
(`pks_client_slot_bench.py`, quieter box) to the same parts measured here (0.82 / 0.73 / 0.67 / 0.66).

What changed (outputs bit-identical; scheme, parameters, single use, secrecy and check soundness unchanged):

* **Segment check vectors natively.** `u W8 mod p` (and `u W8 mod 2^32` for ring-check segments) was a float64
  copy of every `W8` per segment (5.5 ns per multiply-add, the largest single term from 7B up: 38 / 172 / 383 ms
  per slot at 7B / 32B / 72B). `pgw_pks.vecmat_p31` / `vecmat_wrap32` compute the same integers with exact 32-bit
  lane sums on AVX-512, reading `W8` once per 16-row block (0.17 ns per multiply-add; 1.2 / 4.9 / 9.8 s per
  segment at 7B / 32B / 72B). `u` stays per segment (SPEC 7.3).
* **Self-check on 32-bit lanes, 16 rows at a time** (`pgw_pks.check_rows32`): every row is still checked with
  the same 40-bit private vectors; 2.5x less CPU.
* **No int64 copy of `W r`**: the pks-v3 backend hands the decryption's uint32 buffer on as the int32 compact
  form that every segment writer takes for K <= 32 (`PGW_PKS_COMPACT=0` restores the copy). The writer then
  packs from 4 bytes instead of 8 (about 20 % of its time).
* **Private self-check vectors cached as uint32**: at 72B the uint64 set (1.27 GB) overflowed the 1 GiB cache, so
  the last layers' `v W8` were recomputed for every segment; the uint32 set (0.63 GB) fits.

Not changed, with the reason:

* **Bytes per slot are at their floor.** A slot stores `W8 r mod 2^32` of every output feature, uniform 32-bit
  values (that is what hides `W8 x`), so no narrower or compressed form exists; the format adds 16 bytes per
  ~1 MiB chunk and the segment's `uW` (`4 x sum(in_features)` per segment: 3.3 KB per slot at 7B, 0.05 %).
* **Store seeds or ciphertexts and expand at use time:** a seed alone needs `W8 r` at use time, a pass over all
  of `W8` (6.5 GB at 7B) per token on the gateway's critical path: not viable. The provider's reply is larger
  than the masks (36-bit packed words: 4.5 bytes per element, +12.5 %) and moves the decryption onto the gateway.
  Both lose on bytes and on use-time CPU.
* **Share check vectors across segments:** they are already shared by every slot of a segment; across segments
  `u` derives from each segment's own fresh seed (SPEC 7.3), so sharing it would be a scheme change. Not needed
  any more: the term is now 3 / 3 / 4 % of the factory total at 7B / 32B / 72B.
* **Masked-head rows of prompt slots** ("lazy prefill masks"): the masked head uses one row per prefill (the last
  token's), yet every prompt slot carries head masks that are produced, stored and burned unused. A separate
  head stock (the generator can already produce body-only segments, `skip_kinds`) consumed only by decode tokens
  would save, at 500-token prompts and 256 generated tokens, 22 / 6.5 / 2 / 1.4 % of the stock's bytes and
  11 / 3 / 1 / 0.7 % of the factory CPU (0.5B / 7B / 32B / 72B). Single use and secrecy are unaffected (separate segments, same derivation),
  but it changes the store's reservations and the gateway: a decision, not done here.
* **AES-GCM speed:** the native writer and reader use RustCrypto AES-GCM (VAES AES, 128-bit PCLMUL GHASH):
  about 1.2 GB/s here against 7.4 GB/s for OpenSSL on the same core. `ring`'s VAES/VPCLMUL path would cut the
  writer and the gateway's open about 4x (7B: 9.7 -> ~2.5 ms per slot), but it does not wipe its key schedule
  on drop, which the writer guarantees today: a decision.
* The HE kernels themselves (upload, reply keys, decryption: hand-written AVX-512 IFMA, earlier audit) and the
  `r` keystream (normative SPEC 7.3 derivation, 8 keystream bytes per element) are unchanged; they are now 70-80 %
  of the total.

## Keeping no weights on the client

`factory_weights="stream"` (opt-in, [Configuration](configuration.md#keeping-no-weights-factory_weightsstream))
replaces the int8 weights on disk by a stock of pre-drawn check values, refilled by a weight pass whenever it falls
to a quarter. Measured on Qwen2.5-0.5B-Instruct (CPU, the real provisioning endpoint on loopback, the SDK's
settings: masked head, p31 checks, 40-bit private sets; `benchmarks/stock_pass_bench.py`): one pass received
357.9 MB (the 357.8 MB of layer weights plus 0.03 % of protocol; the output layer's int8 weights are derived from the
model data the client keeps anyway) and wrote a stock of 128.6 MB (64 segment sets of 0.73 MB, 8 private sets of
10.2 MB) in place of the 359 MB of int8 weights, in 14-22 s on this shared 4-vCPU box. Extrapolated with the same
formulas to the real shapes:

| model | int8 weights it replaces | stock on disk (64 segments + 8 private sets) | weights per pass | pks-v3 traffic per 1024-slot segment | extra network, default (48 segments per pass) | extra network, 64 per pass |
|---|---|---|---|---|---|---|
| 0.5B (measured) | 0.36 GB | 0.13 GB | 0.36 GB | 4.1 GB | +0.18 % | +0.14 % |
| 7B | 6.5 GB | 0.59 GB | 6.5 GB | 13.9 GB | +0.98 % | +0.73 % |
| 32B | 31.2 GB | 1.94 GB | 31.2 GB | 43.7 GB | +1.49 % | +1.12 % |
| 72B | 70.2 GB | 3.06 GB | 70.2 GB | 65.7 GB | +2.23 % | +1.67 % |

* **Disk** drops by the weights minus the stock: 5.9 / 29.3 / 67.2 GB at 7B / 32B / 72B. More workers need more
  private sets (`max(8, 2 x workers + 2)`, 47 / 155 / 243 MB each at 7B / 32B / 72B).
* **Network** is a full weight download per pass, every `check_stock_segments - low water` segments. A bigger
  `check_stock_segments` lowers it (disk: 3.3 / 11 / 17 MB per extra segment). Every worker start and every aborted
  segment uses a private set, so frequent restarts or a flaky link bring passes sooner.
* **Availability**: the factory depends on reaching a weight source; with the stock empty and no source, masks stop
  (requests wait, then fail closed). A pass at 72B moves 70 GB: at 1 Gbit/s about 10 minutes, during which the
  quarter of the stock left (16 segments, about 16 K token rows) must cover the traffic.
* **CPU**: a pass hashes the weights (SHA-256) and computes 64 + 112 check vectors over them, once per 48
  segments: small against pks-v3's own per-segment client work.

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
  token saves a mask slot (about 32 ms at 7B without the head's masks) and its prompt pass. A turn that reuses
  2,000 tokens and generates 256 saves about 64 CPU-seconds of masks for about 12 of head work: for long multi-turn chats, prefer
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
for the most tokens per core or 1-2 for the lowest latency per stream (the fast lane, above), all on the same
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
process holds one lane connection set per model; a lane serves at most 8 rows per call. Size the provider by rows
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
