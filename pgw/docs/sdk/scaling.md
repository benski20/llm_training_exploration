# Scaling: many users and conversations on one client machine

One `pgw.OpenAI` client serves many threads and tasks at once: concurrent requests are batched into the same
round trips (one trip per layer step carries a row for every running answer). This page says what limits a
client machine as the number of concurrent streams grows, how to size it, and how to run several worker processes.

## What limits a client, in order

1. **Mask production (CPU).** Every generated token and every prompt token that is not reused from the previous
   turn consumes one mask slot, and each slot costs client CPU in the background factory. Since the third round of
   2026-11-26 ([below](#round-3-wire-keystream-buffers)), counting also the request and reply handling the earlier
   figures left out, one core spends about 35 ms per slot at 7B, 112 ms at 32B and 166 ms at 72B (per-part
   benchmark; the factory as production chains it, decoder layers only: 40 / 127-137 / 220-240 ms), i.e. 25-29 /
   7.3-8.9 / 4.2-6.0 slots/s per core; before that round the same measurements gave 56 / 199 / 299 ms (68 / 227-238
   / 379-410 end to end). The [breakdown](#mask-cost-per-slot-by-component) below has every part. At 7B, 16 streams
   of 17 tokens/s with 500-token prompts need about 800 slots/s, i.e. about 28-32 cores of mask workers. This is the dominant cost at every size: it is linear in
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
   Since 2026-11-26 the library cuts this per-stream cost, exactly (same tokens, logprobs and provider-visible
   trips): a decode step of several answers runs as one synchronous pass on the plaintext thread when one lane multi
   call serves it (no event-loop hand-offs per trip, `PGW_SYNC_MULTI`); its trips take one native lease per answer,
   prepared and finished in one GIL-free native call each, spread over the native pool for large trips
   (`PGW_TRIP_NATIVE_MULTI`); the answers of a step sample on a thread pool (`PGW_SAMPLE_THREADS`, each answer has its
   own generator); and `torch.multinomial` is replaced by an exact native draw (same index and generator state,
   3.9 -> 1.6 ms at a 152k vocabulary; 5.5-6.1 -> 3.1-3.3 ms CPU per sampled token with logprobs). Measured on this
   box (before / after): tiny 16 streams 215-223 / 279-282 tokens/s at 3.2 / 2.0 ms client CPU per token, 4 streams
   117-119 / 128-130; 0.5B 16 streams 7.3-8.1 / 9.3-10.1 tokens/s, 4 streams 5.5-5.7 / 6.2-6.3 (CPU provider
   saturated). Since the second round (same day, also exact and on by default): the decoder layers of a
   several-answer step run in the native decode driver (`PGW_DECODE_NATIVE_MULTI`: one native call, GIL released,
   per stretch of trips; per trip every answer's prepare, the duplicate-row checks, one lane multi call, every
   answer's finish and float ops), and the next trip lease of each linear of such answers is prepared on a background
   thread (`PGW_TRIP_LEASE_AHEAD`; rows the session already holds, attached only at its first row, zeroed otherwise).
   Measured (before / after, interleaved pairs): tiny 16 streams 279-290 / 337-360 tokens/s at 2.0-2.1 / 1.36-1.47 ms
   client CPU per token (plaintext thread 1.55-1.64 / 0.86-0.95), 4 streams 123-126 / 140-147 tokens/s at 2.9 / 1.9-2.1
   ms; 0.5B 4 streams 6.4-6.6 / 6.9-7.7 tokens/s at 62-65 / 49-54 ms, 16 streams 10.4-11.0 / 11.9-12.4 at 50-52 /
   44-46 ms (the CPU provider limits tokens/s there); one stream unchanged. What remains on the plaintext thread
   per trip: the noise monitor's statistics of a sampled answer (a
   step of 16 answers has one in about half its trips), the leases' window bookkeeping, and the masked LM head's trip
   (still the per-trip Python path); per answer and token: the top-p sort.
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
data and mask setup (not per stream). Cores are physical cores with AVX-512 VNNI or AMX for the factory. Mask CPU
per slot is the complete figure of the [breakdown](#mask-cost-per-slot-by-component) (factory 10.6 / 33.5 / 103 /
157 ms; gateway cores include 1.25 / 3.7 / 11 / 16 ms per slot of mask opening and retirement on the gateway's
helper threads). These figures predate the switch of the store's AES-GCM to AWS-LC (same day), which cuts the
writer by about 35-45 % and the open by about 45 % ([below](#the-stores-aes-256-gcm-writer-open-retirement)); the
table is not recomputed: the factory total falls by about 5 %, the gateway's open + retirement by about a third. "Stock writes" is the disk write bandwidth of the factory at that rate.
The table also predates [round 3](#round-3-wire-keystream-buffers) (same day): its factory figures left out the
request and reply handling (about 13 / 44 / 67 ms per slot at 7B / 32B / 72B scaled the same way), and with it
counted the factory now costs about 25 / 75 / 110 ms scaled (35 / 112 / 166 on this box), so the factory cores at a
given N are about 25-30 % lower than the table says.

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
| writer: pack + AES-256-GCM + pwrite + fsync (AWS-LC AES-GCM, below: -> **after**) | 2.38 -> 1.89 -> **1.19** | 6.1 -> **4.0** | 19.1 -> **11.8** | 27.6 -> **16.7** |
| **factory total, this box** | **17.3 -> 13.0** | **90 -> 46** | **347 -> 155** | **652 -> 236** |
| factory total, scaled to the earlier audit's box | 14.2 -> 10.6 | 66 -> 33.5 | 231 -> 103 | 433 -> 157 |
| (earlier documented figure: r, upload, keys, decryption, no head) | 7.2 | 24.9 | 80 | 121 |
| gateway: AES-GCM open + unpack of the slot's rows (AWS-LC: -> **after**) | 1.04 -> 1.05 -> **0.55** | 3.6 -> **1.9** | 11.9 -> **5.9** | 16.5 -> **8.9** |
| gateway: retirement (crypto-erase, zero-fill, unlink) | 0.70 -> 0.48 | 1.5 | 4.9 | 6.9 |
| (retirement with `shred_zero_fill=False`: crypto-erase + unlink) | 0.15 | 0.45 | 1.3 | 2.0 |
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
* The HE kernels themselves (upload, reply keys, decryption: hand-written AVX-512 IFMA, earlier audit) and the
  `r` keystream (normative SPEC 7.3 derivation, 8 keystream bytes per element) are unchanged; they are now 70-80 %
  of the total.

### Round 3: wire, keystream, buffers

Measured 2026-11-26 after the changes above, profiling first (`benchmarks/pks_slot_breakdown.py`, now also counting
the upload's wire form and the reply's decode; `benchmarks/pks_factory_e2e.py`, the real factory chain; `pks-bench
stages`, the IFMA kernels part by part). One core (`taskset`), 1024-slot segments, p31, memory touched once before
measuring (on this VM the first touch of a page costs ~20x a normal fault), best of 3; before = the branch head's
build and code; ms per slot:

| part | 7B | 32B | 72B |
|---|---|---|---|
| `r` keystream (ChaCha20) | 5.8 -> **1.6** | 19.6 -> **5.2** | 31.1 -> **8.1** |
| upload encryption + reply keys | 15.6 | 55.4 | 82.6 |
| reply keys' wire form | 0.5 | 1.7 | 2.1 |
| upload wire form (`pack_bits(b)`, msgpack request) | 14.3 | 54.3 | 82.1 |
| the three above, now one native call (`encrypt_wire`) | 30.4 -> **17.9** | 111.3 -> **55.4** | 166.7 -> **83.5** |
| reply decode (msgpack, `y` handed to the decryption) | 3.5 -> **0.0** | 11.0 -> **0.0** | 19.0 -> **0.0** |
| reply decryption | 9.5 -> **7.7** | 30.4 -> **25.8** | 45.0 -> **36.9** |
| every-row 40-bit self-check | 3.1 | 10.2-11.1 | 14.9-15.4 |
| segment check vectors `u W8 mod p` | 1.2-1.3 | 5.7-7.8 | 9.0-9.2 |
| writer (pack, AES-GCM, `pwrite`, fsync) | 3.0-3.2 | 8.9 | 12.9-13.3 |
| **factory total** | **56.4 -> 34.8** | **199.3 -> 112.0** | **299.2 -> 165.8** |
| slots/s per core | 17.7 -> 28.7 | 5.0 -> 8.9 | 3.3 -> 6.0 |
| end to end, decoder layers (`pks_factory_e2e.py`) | 68.1 -> 39.5 | 227-238 -> 127-137 | 379-410 -> 220-240 |
| gateway: open + retirement (unchanged) | 3.9 | 13.1-13.6 | 17.4-17.7 |

What the profile showed: the earlier per-slot figure left out the upload's wire form and the reply's decode, which
were the largest single parts from 7B up (numpy `pack_bits` of the 56-bit bodies, 10 ns per value, the msgpack
request copy, 9 ns per value, and the reply's copy: 101 ms of 299 at 72B); the `r` keystream ran at 4-8 ns per element (the `chacha20`
crate and a fresh array per chunk) where AVX-512 ChaCha20 takes 1.6; and every fresh buffer (r, bodies, decryption
output, reply copies) cost its page faults and the kernel's zeroing again, about 0.16 ns per byte on a warm box.
Changes (each bit-identical, tested; scheme, parameters, check material, single use and wiping unchanged):

* **`r` keystream**: `pgw_fast`'s ring stream (factory and gateway) runs a 16-block AVX-512 ChaCha20 with the ring
  mask in the store (same bytes; `PGW_FAST_CHACHA512=0`: the crate), and the factory fills each chunk's `r` into a
  zeroed buffer of the same shape from an earlier chunk (`mask_rows_into`; `PGW_FACTORY_R_REUSE=0`: fresh arrays).
* **Upload written in place** (`pgw_pks.encrypt_wire`, `Customer.encrypt_wire`, `RemotePKS.frame`): the encryption
  kernel writes the bodies' three bit planes (`pack_bits(b, 56)`) and the reply keys' wire form straight into the
  request's `bytes` between the msgpack head and middle (`wire.maskgen_v3_frame`): no `b` array, packing pass or
  message copy; the same request bytes for the same seeds (`PGW_PKS_WIRE_FUSED=0`: the old path).
* **Reply in place**: `decode_maskgen_response_view` hands the decryption a read-only view of `y` in the response
  body (the rest of the message decoded and checked by the same code; anything unusual takes the copying decode);
  the native decryption reads any byte buffer.
* **Decryption output reused**: the factory's segment writer hands each zeroed output back
  (`PKSMaskBackend.release_output`), the next call of that shape decrypts into it (`decrypt_into`; several passes
  write their rows straight into it); at most one per shape, 1 GiB (`PGW_PKS_OUT_REUSE=0`: fresh arrays).

Tried and not kept (measured no gain on this box): smaller pad primes (2^36) with fully lazy NTT butterflies
(8 instead of 10-12 vector operations per butterfly), radix-4 passes of the inverse NTTs (half the loads and
stores), other L1 block sizes, prefetching the cross-block NTT columns, an IFMA (`vpmadd52luq`) copy of the
self-check and an L2-tiled one. The HE kernels themselves are unchanged: encryption about 15 ns per input value and
slot at 4 groups per request (pad inverse NTTs 6 per group, noise 2.2, output 1.8, the public `a`'s keystream and
forward NTTs 8 per request), decryption about 5 ns per output value and slot (inverse NTT mod `q1` 3.1, combine 0.8,
hint load 1 per request).

Open: reply keys per (linear, group) ~3.5 ms (scalar NTT of `t'` per prime, the noise layout loop): ~4 ms per slot
at 72B; the encryption's request `bytes` is a fresh allocation per call (httpx takes `bytes`); the HTTP layer's own
copies and TLS are not counted here; with the default 8 GB stock, 32B / 72B segments hold one 256-slot group per
request, so the public `a`'s forward NTT is paid per group (about 6 ns more per input value and slot than at 1024
slots): a larger `mask_store_gb` gives 1024-slot segments.

### The store's AES-256-GCM (writer, open, retirement)

Since 2026-11-26 the native segment writer (`pgw_fast.seal_chunks` / `seal_ring_chunks`) and the gateway's
chunk open (`chunk_open_into`) use AWS-LC's AES-256-GCM (`aws-lc-rs`, statically linked) instead of RustCrypto
`aes-gcm` 0.11. Same at-rest bytes, same nonces (`prefix || u64be(index)`) and AAD, same store contents.
Measured with `benchmarks/maskstore_aead_bench.py` (the store rows of the breakdown, same calls and shapes;
best of two interleaved runs on this shared 4-vCPU box, load 3-8):

| ms per slot | 0.5B | 7B | 32B | 72B |
|---|---|---|---|---|
| writer (range check + pack + AES-GCM + pwrite + fsync), RustCrypto -> AWS-LC | 1.79 -> 1.19 | 6.14 -> 4.00 | 18.8 -> 11.8 | 29.4 -> 16.7 |
| gateway open + unpack | 1.05 -> 0.55 | 3.52 -> 1.88 | 10.8 -> 5.92 | 16.4 -> 8.89 |
| gateway open + unpack + retirement (default zero-fill) | 1.69 -> 1.04 | 5.24 -> 3.45 | 15.4 -> 10.8 | 24.4 -> 15.6 |
| retirement, default / `shred_zero_fill=False` (no AES; best of all four runs) | 0.48 / 0.15 | 1.57 / 0.45 | 4.6 / 1.3 | 6.7 / 2.0 |

* **Why RustCrypto was slow.** Not a configuration problem: `aes` 0.9 already picks its VAES AVX-512 backend
  and `polyval` 0.7 its PCLMULQDQ backend at run time (`cpufeatures`; no `RUSTFLAGS` or `.cargo/config` in the
  build, and `-C target-cpu=native` does not help: 2.0 GB/s). `aes-gcm` runs CTR and GHASH as two separate
  passes over the chunk, CTR through the generic `ctr` crate (5.5 GB/s) and GHASH as POLYVAL with a byte
  reversal per block, four 128-bit carry-less multiplies at a time (6 GB/s): 2.8 GB/s for the AEAD on an idle
  core, the 1.2 GB/s quoted before under load. Chunks are already ~1 MiB, so per-call overhead is nil (64 KiB
  chunks: the same rate). AWS-LC (like OpenSSL) runs one stitched pass with VAES and VPCLMULQDQ on 512-bit
  registers.
* **Options measured** (one core, 1 MiB chunks, seal / open): RustCrypto aes-gcm 0.11 2.8 / 2.5 GB/s;
  **AWS-LC (aws-lc-rs 1.19) 13.8 / 12.9**; OpenSSL 3.0 via the `openssl` crate 12.3 (seal); graviola 0.4
  (pure Rust + assembly, zeroizes) 9.8 / 8.4; `ring` excluded (no key-schedule wipe). AWS-LC is the fastest and
  wipes its key schedule on drop: the round keys and GHASH table live in the `EVP_AEAD_CTX` that
  `EVP_AEAD_CTX_free` releases with `OPENSSL_free`, which `OPENSSL_cleanse`s the whole allocation (references and
  tests in `rust/pgw-fast/src/writer.rs`). OpenSSL would need a vendored libcrypto in the wheel (or an
  auditwheel-bundled `.so`); graviola panics on CPUs without ADX / AVX2 and is 30 % slower.
* **Where the writer's time goes now** (7B, one `gate_up` linear, 155 MB of `W r`, one thread): packing the
  uint32 rows into the chunk buffer plus AES-GCM 29 ms (5.4 GB/s; the AEAD alone is about 12 ms of it), the
  `pwrite` into the page cache 30-48 ms (3-5 GB/s: the kernel's copy, into freshly allocated pages for a new
  file), the range check 0: it cannot fail for the uint32 / int32 form at K = 32 (the pks-v3 compact output) and
  is now skipped there (`ALWAYS_IN_RANGE_FROM_K`; it was a separate 24 ms pass over the array; every other
  type / K is still checked in full). So the writer is now bound by memory copies and the page cache, not the
  cipher. The fsync (one per segment, at publish, plus the directory's) costs the factory almost no CPU (the
  writeback runs in the kernel's flusher threads: `seal` and `seal_nofsync` agree within noise), but doubles the
  segment's wall time (7B linear: 132 -> 268 ms); it is what makes a published segment durable before its cursor
  is used, so it stays. `O_DIRECT` would skip the page-cache copy but needs 4 KiB-aligned chunk offsets and
  lengths: a format change, not done. Several writer threads per call cost more CPU than one (111 vs 94 ms for
  that linear) for less wall time (132 vs 126 ms on this 4-vCPU box shared with other jobs; more on an idle
  one); the default (`PGW_WRITER_THREADS=0`: one per MiB, at most the CPU count and 8) is unchanged.
* **Retirement** is the crypto-erase (64 KiB of random bytes over the header that holds the wrapped DEK, fsync),
  then, with `shred_zero_fill` (default on), zeros over the whole body and a second fsync, then truncate and
  unlink: the zero-fill is 70 % of the retirement CPU (6.7 vs 2.0 ms per slot at 72B) and doubles the disk writes.
  The masks are deleted by the crypto-erase; the zero-fill only helps on disks that overwrite in place (not SSDs,
  not copy-on-write filesystems), so the default stays on (the conservative choice for spinning disks) and the
  guide's advice stays: turn it off on SSDs (`PGW_MASKSTORE_SHRED_ZERO_FILL=false`).

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
* `prefix_cache` (default `"auto"`): a request whose prompt starts with tokens this process has already
  computed (the earlier turns of its conversation, a shared system prompt) runs only the rest: the cached tokens
  take no masks and no round-trip rows. `"auto"` turns it on only when the output layer runs in your process
  (`masked_head="off"`, or when the service does not serve the masked head). `prefix_cache=True` turns it on
  with the masked head as well (since 2026-11-26; before, it turned the masked head off): opt-in, because the
  service can then tell that a request continued an earlier one ([configuration](configuration.md#prefix-cache)).
  For long multi-turn chats it is the main lever on mask production (a turn that reuses 2,000 tokens saves about
  2,000 mask slots, 64 CPU-seconds at 7B). Reuse is within one OpenAI `user` (`prefix_cache_scope="client"`:
  across the whole client).
* KV memory: the prefix cache keeps at most `prefix_cache_mb` (1 GiB) of KV states per model and process (least
  recently used are wiped first); each worker process has its own, so route a conversation to one worker.

* Client CPU per stream (on by default in the library; environment variables, `0` turns each off, nothing else
  changes): `PGW_SYNC_MULTI` (several-answer decode steps as one synchronous pass), `PGW_TRIP_NATIVE_MULTI` (one native
  lease per answer, batched native prepare / finish), `PGW_SAMPLE_THREADS` (`auto`: sampling threads, at most 8),
  `PGW_NATIVE_MULTINOMIAL` (the exact native draw for `temperature > 0`, checked against torch at first use),
  `PGW_DECODE_NATIVE_MULTI` (several-answer steps in the native decode driver), `PGW_TRIP_LEASE_AHEAD` (trip leases
  of several-answer steps prepared on a background thread: about 1.5-2 ms of CPU per token moves off the plaintext
  thread, the total stays about the same).

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
* The OpenAI `user` field scopes the prefix cache (no reuse across users unless `prefix_cache_scope="client"`) and, with `speculation`,
  orders a user's requests behind the previous one's remaining dummy rounds; it never delays another user.
* There is no per-user quota inside the client: one user with many concurrent requests can occupy all
  `max_concurrent` slots. Apply per-user limits in your application if users are untrusted.

## Many clients per provider

The service admits by tenant (one tenant per SDK key): `tenant_max_inflight` (concurrent requests),
`tenant_rows_per_s` / `tenant_burst_rows` (token bucket on rows), the global `max_inflight_requests` (256) and
`max_queued_rows`; a refused call gets `unavailable` with `Retry-After` and the client retries it. Each client
process holds one lane connection set per model; a single lane call serves at most 8 rows, a multi call up to 64 items and 256 rows. Size the provider by rows
per second (sum of all clients' tokens/s, prompt rows included); the client side never needs changes for it.

## Consumer devices: the fused schedule (`schedule="fused"` / `"auto"`, opt-in)

On a high-latency link the round trips per token, not the GPU, set the speed. `schedule="fused"`
([Configuration](configuration.md#schedule-standard-or-fused)) halves them (`2 x layers` instead of `4 x layers +
1`) and is a different fixed reference (not bit-identical to `"standard"`); `"auto"` picks it per model when the
client starts, from the measured RTT and bandwidth, and keeps it for the client's life.

What it costs a client machine, per generated token:

* **Bytes:** 1.74x the standard schedule's (7-byte values; the same number of values). Qwen2.5-0.5B: 1.27 MB up and
  3.18 MB down; 7B: 5.8 / 10.8 MB. On a consumer uplink this is often the limit (7B at 100 Mbit/s: ~0.47 s per
  token for the upload alone); `"auto"` counts it.
* **Masks:** one K56 slot per token row (the same number of values as a standard slot at 7 bytes each: 3.2 MB per
  slot at 0.5B). The encrypted `pks-v3` factory makes each 56-bit mask from 3-4 ordinary 32-bit products (the
  top-digit plan), so a fused slot costs about 3-4x a standard slot's client factory CPU and provider maskgen;
  size `mask_workers` accordingly. The first use of a model builds a second one-time hint (over the fused weights)
  next to the standard one.
* **Client CPU:** the native fused driver runs a whole decode pass (layer 0's qkv computed locally, every pair trip,
  the folded output layer) in one call; measured client CPU per token in `benchmarks/sdk_fused_wan.py` (tiny test
  model on loopback: 3.6-5.9 ms fused vs 6.8-10.3 ms standard). The wide element-wise stages of a one-row pass
  (unmask of the second stage, its check, the gate/up and output-layer scaling) can be split over the native pool
  with `PGW_FUSED2_SPLIT=<parts>` (same bits).
* **Speculation:** strict speculation works on the fused schedule; filler rows are masked rows there, so a verify
  pass of `spec_rows` rows costs `spec_rows` mask slots.

Measured on this 4-vCPU box with the CPU test provider behind an emulated link (`benchmarks/sdk_fused_wan.py`,
`benchmarks/results/fused2/sdk_wan_*_cpu_2026-11-27.json`): the trips per token are exact (tiny: 9 standard, 4
fused; Qwen2.5-0.5B: 97 and 48) and the fused answers equal the fused reference bit for bit. The decode rate on the
tiny model at RTT 0 / 5 / 20 ms: standard 37 / 13.2 / 4.6 tok/s, fused 114 / 33.5 / 10.8 tok/s. The 0.5B numbers
are bounded by the CPU provider's own products (a GPU's job), see `STATUS.md`.

## Consumer devices: mask cost per token (research, opt-in switches)

`research/reviews/mask-cost-consumer.md` (2026-11-27) sizes pks-v3 masks for a consumer device at 20 tok/s.

**Bandwidth comes first.** At 32B a slot costs 30.3 MB of online trips and 42.7 MB of masks: ~23 Gbit/s per user
at 2 slots per token. At 7B it is ~7 Gbit/s. A consumer link carries neither, whatever the mask compute costs.

**Provider throughput depends on the client's stock.** A client that holds one 256-slot group (32B with the default
8 GB stock) makes every provider item recompute the public hint for 256 slots: ~48 slots/s per H100 at 32B. Two
switches, both off by default, let one hint computation serve many groups:

* `PGW_SERVICE_PKS_COALESCE_MS` on the provider: items of one linear and one public seed arriving within that many
  milliseconds run as one device item. The replies are byte-identical to running each item alone.
  `PGW_SERVICE_PKS_COALESCE_GROUPS` (16) caps a batch.
* `PGW_PKS_CRS_A` on the client: new hint stores use the model's common public upload seed
  (`pgw.hemask.pks.crs_a_seed`) instead of a fresh one, so the groups of different customers share one hint.
  This is pending the external review (question Q8 of the note). Each customer keeps its own private
  key-switching seed and fresh secrets.

With both on, the estimate at 32B is 143-508 slots/s per H100 and 3.4-13.5 users per H100 at 2 slots per token
(1.2 today). On a T4, 8 items took 2.2-3.7x less time coalesced than one by one.

The client's NTT kernels port to a GPU in exact integer arithmetic. The torch prototype is
`pgw.hemask.pks_torch`, bit-identical to the CPU kernels; fused, it is estimated at ~2 ms per slot on an RTX
4070-class GPU, against 113 ms of one core.

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

The table above predates lane format 2 and the client CPU changes of 2026-11-26. With both (same box, two
interleaved pairs, ranges; 64 tokens per tiny answer, 24 per 0.5B answer):

| model | N | tokens/s total | rows per trip | lane trips | client CPU ms/token | plaintext thread ms/token |
|---|---|---|---|---|---|---|
| tiny | 1 | 45 | 1.6 | 600 of 600 | 3.9-4.2 | 2.7-2.8 |
| tiny | 4 | 128-130 | 5.9 | 633 of 633 | 2.7-2.8 | 2.2 |
| tiny | 16 | 279-282 | 22 | 666 of 666 | 1.9-2.1 | 1.4-1.6 |
| 0.5B | 1 | 2.9 | 2.2 | 2616 of 2616 | 69-70 | 46-47 |
| 0.5B | 4 | 6.2-6.3 | 7.5 | 3001 of 3001 | 68-72 | 43-46 |
| 0.5B | 16 | 9.3-10.1 | 30 | 3001 of 3001 | 55-59 | 32-35 |
