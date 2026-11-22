# Mask data path at high bandwidth (package X2)

Status: 2026-11-21. Scope: the bytes path factory -> RAM ring -> gateway, with the disk store as the
outage reserve. Not in scope (other groups): overlapping encrypt/provider/decrypt inside
`HEMaskBackend` (M1), the provider's maskgen GPU path (M2), gateway HE decrypt and skipping masks for
gateway-local linears (M3). Code: `src/pgw/maskstore/ringbuf.py`, `src/pgw/maskstore/store.py`;
tests `tests/maskstore/test_ringbuf.py`, `tests/factory/test_he_ring_path.py`; benches
`benchmarks/mask_pipeline_bench.py` (phases) and `benchmarks/mask_ram_path_bench.py` (new: producer
and consumers running at the same time). Labels: **[m]** measured on the 4-vCPU dev box (shared with
other agents; load average 10-14 during these runs, so treat single numbers as +-20%), **[d]** derived,
**[e]** estimate.

## 1. Requirement

`benchmarks/scale_sizing.py`: one 7B token uses 5.56 MB of `W r`; one 70B token uses 26.9 MB. At
1k tokens/s that is 5.6 GB/s (7B) or 27 GB/s (70B), written once and read once. Disk cannot carry this
in the hot path (scale plan section 8). The in-memory ring (WP8, `mask-ringbuffer-design.md`) already
existed. The open question was what it delivers per slot and per core, and what limits it.

## 2. Where the time goes per slot (before this package)

`mask_pipeline_bench.py --slots 512 --segments 2 --sessions 32 --tokens 32` (7B: 2 of 28 layers
materialised, scaled to a full slot; 0.5B: all layers). "take" is the gateway's decode take: one
`take_checked` row per (session, linear) per step, which yields `r` (regenerated from the seed) and
`W r`. Medians of 3, A/B interleaved against the base commit [m]:

| per full slot | 0.5B store | 0.5B ring (before) | 7B store | 7B ring (before) |
|---|---|---|---|---|
| factory write | 3.0 ms (334/s) | 0.24 ms (about 4,200/s) | 9.6 ms (104/s) | 0.85 ms (about 1,180/s) |
| gateway take | 6.7 ms (150/s) | 2.9 ms (350/s) | 11.2 ms (89/s) | 5.8 ms (172/s) |
| W r bytes | 1.22 MB | 1.22 MB | 5.56 MB | 5.56 MB |

The factory write into RAM ran at 4.4-6.6 GB/s, close to a memcpy. **The gateway take was the limit.**
For the 7B ring, the cProfile breakdown of 8,192 takes was: pad `r` regeneration 44%, copying the
`W r` chunk out of the slab into a per-reservation cache 18% (each of 32 sessions copied its own
1 MiB chunk to read one row of it), decoding 9%, zeroing the caches 7%, the rest Python.

Floors on this box, one thread, per 7B slot [m] (`mask_ram_path_bench.py` prints them): a memcpy of
the 5.56 MB of `W r` takes 0.37 ms (about 15 GB/s); the ChaCha20 keystream for every pad row takes
1.8 ms. The pads cost more than the `W r` bytes because SPEC 7.2 expands each ring element from 8
keystream bytes (`u64 mod 2^K`), so a 7B slot needs 6.6 MB of keystream for its 831k input elements.

## 3. What changed

1. **Zero-copy reads from the ring** (`RamSegment.read_wr_rows`, used by `MaskReservation._read_wr`
   for RAM segments, so `take`, `take_checked` and `prefetch` all use it). The rows are decoded
   straight from the locked slab into a fresh, writable numpy array (`uint32` for the compact K <= 32
   form, else `int64`), under the slab's R4 lock with the same checks as `decrypt_into` (handle open,
   ring open, slab generation, READY/DRAINING). There is no per-reservation chunk cache for RAM
   segments, so there is nothing extra to zero. This was the "view_rows fast path" the ring design
   left open (section 15, "Not done"). The v2 shared-memory slabs (memfd) take the same path
   (`FdMapping.address`).
2. **Batched native claim over ring reservations** (`take_checked_many`, opt-in with
   `PGW_MASK_RING_BATCHED_CLAIM=1`). One `pgw_fast.claim_rows_many` call serves many
   (session, linear) requests with the GIL released. The `W r` rows are read from the slabs while
   every involved slab lock is held (fixed lock order; reservation locks first, as the sequential take
   path already does) and after each slab was checked again under its lock. **It stays off by default**
   because it measured slower here: 140-185 vs 255-275 slots/s with 1 native thread, and 70-80
   with 3 threads, on the loaded box. Its pads come from RustCrypto ChaCha20, which is 2-2.5x slower
   per byte than OpenSSL's AVX-512 ChaCha20 in the sequential path. It becomes worthwhile once the
   native keystream is as fast as OpenSSL's (section 6).
3. **`benchmarks/mask_ram_path_bench.py`**: the factory writer and the gateway consumers run at the
   same time (steady state, with the ring primed). It reports the delivered slots/s and GB/s, producer
   and consumer stall shares, CPU ms per slot and the two per-slot floors.
4. **End-to-end test of the north-star path**: encrypted (MLWE) factory -> `RingSink` -> ring ->
   sequential and batched takes. The masks are bit-identical to the local factory's, the pads are the
   normative ones, and the provider stand-in saw only q-bit ciphertext words
   (`tests/factory/test_he_ring_path.py`).

## 4. After (same benches, A/B interleaved with the base commit)

| per full slot | 0.5B ring before | 0.5B ring after | 7B ring before | 7B ring after |
|---|---|---|---|---|
| gateway take, phases bench [m] | 350/s (2.9 ms) | **586/s (1.7 ms)**, 3 runs 506-669 | 172/s (5.8 ms) | **276/s (3.6 ms)**, 3 runs 267-288 |
| sustained, writer + 1 consumer concurrently [m] | 256/s, 3.9 ms CPU | **410/s, 2.5 ms CPU** | 95/s, 10.8 ms CPU | **135/s, 7.5 ms CPU** |
| delivered W r, sustained [m] | 0.31 GB/s | 0.50 GB/s | 0.53 GB/s | 0.75 GB/s |

The store tier did not change, as expected [m]: 7B take 89 slots/s, write 104 slots/s with fsync;
0.5B 150 and 334. The sustained 7B numbers are lower than the phases bench because the writer and the
consumer share the loaded cores.

Several consumer threads in one process did not scale on this box [m]: 2 consumers delivered 76-105
slots/s against 135-155 for one. CPU per slot went up, which points to contention between threads,
not idle waiting. A microbench that only generated pads showed the same with both ChaCha
implementations. With the load average at 10-14 on 4 vCPUs, this cannot be separated from the other
agents' load. It has to be re-measured on an idle multi-core host before anyone plans on in-process
thread scaling. Until then, scale out with processes: gateways are sharded by user in the scale plan
anyway.

## 5. What this means at the north-star rates ([d] from the measured per-slot times, one core each)

| | 7B, 1k tokens/s | 70B, 1k tokens/s |
|---|---|---|
| W r bytes into and out of RAM | 5.6 GB/s | 27 GB/s (5.4 GB/s per gateway over 5 gateways) |
| factory write into the ring (0.85 ms per 7B slot) | 0.9 core | about 4 cores in total [e, x4.8 bytes] |
| gateway take before / after | 5.8 / 3.6 cores | about 28 / 17 cores in total [e] |
| floor: pad keystream + W r memcpy | 2.2 cores | about 10 cores [e] |
| disk store as the hot path (for comparison) | 11 cores to read, plus 10 to write | not feasible |

So the RAM path is no longer limited by bytes. The gateway's pad regeneration is the floor, and it
costs more than the `W r` bytes themselves. The ring holds plaintext only in locked RAM. A 30 s buffer
is 167 GB for 7B at 1k tokens/s, so production rings should hold seconds of supply (for jitter) and
leave outages to the disk reserve. The scheduler already plans "store below low water" ahead of "ring
not full", so the reserve refills before the ring is topped up.

## 6. Packing of mask words (evaluation; no format change made)

* **`W r` at rest and in the ring is already K-bit packed**: `K/8` little-endian bytes per element
  (4 at K32, 5 at K40, 6 at K48; `RingParams` only allows K that are multiples of 8). This is also the
  "48-bit packing" for K48 rings. An element of `W r` is uniform mod `2^K` whenever its row of `W` has
  an odd entry, which almost every row of a real checkpoint has, so K bits per element is the
  information-theoretic minimum. Rows whose entries are all even would leave low zero bits, but
  exploiting them needs per-row bookkeeping for a negligible gain. Nothing more can be removed exactly,
  and narrower storage (`mod 2^C`) is rejected because the
  Freivalds and range checks read the high bits (scale plan section 8).
* `r` is never stored (32-byte seed per segment). `uW` is `checks x n x K/8` per linear per segment,
  0.5% of a 7B slot at 256 slots per segment.
* **Remaining slack is in transient gateway arrays, not stored bytes**: `r` is handed out as `int64`
  (50% unused at K <= 32), because `pgw_fast.quantize_mask_pack` takes `int64` pads; `W r` is compact
  `uint32` at K <= 32 and `int64` above.
* **The one real byte cut is in the pad expansion, and it would change the format.** SPEC 7.2 draws
  each pad element from 8 keystream bytes. A segment-format flag "4 keystream bytes per element when
  K <= 32" keeps `r` exactly uniform mod `2^K` (the low K bits of a uniform u32), so privacy is
  unchanged. It would halve the gateway's largest per-slot cost: 1.8 -> 0.9 ms per 7B slot [e]. It
  needs the factory (CPU and GPU PRNG), the gateway take path, `pgw_fast` and a header version to
  change together, so it is listed as the next step, not done here.

## 7. Privacy and exactness

* **Exact.** The direct read decodes the same bytes the copy-out path decoded. Tests compare every
  slot range (inside one chunk and across chunks, int64 and compact, K32/K40/K48) with the copied
  chunks, compare ring and store tiers on the same seed, compare the batched claim with the sequential
  takes across extents and with prefetched rows, and run the encrypted factory end to end through the
  ring. The existing ring, ringshm, ring-sink and tiered tests all pass unchanged.
* **Single use and wipe on consume are unchanged.** Cursors still advance before any byte is
  derived; slabs are still wiped (v1) or dropped (v2, sealed) only after every reservation on them
  has closed; burn accounting is unchanged. The direct read makes fewer copies of secrets: the
  per-reservation 1 MiB chunk copy of `W r` (which held rows of other sessions too) is gone, and the
  only copy is the caller's output array, which the engine zeroes as before.
* **R4 (no read of a recycled slab) is kept.** Every direct read, and every native batched read, runs
  under the slab lock after re-checking the ring, the generation and the state, exactly like
  `decrypt_into`. `MaskRing.close` still marks slabs WIPING under their locks before force-closing
  reservations and unmapping. A read that loses that race raises (tested by bumping the generation
  under the lock and by closing the ring).
* **No new data leaves the gateway, and the disk tier is unchanged.** At-rest AES-GCM, the rollback
  journal and counter, and crypto-shredding still apply to every byte that reaches disk. The ring
  never writes to disk.
* **Provider view: unchanged.** Mask sizes, counts and schedules depend only on public configuration.

## 8. What is left

1. A faster native keystream (AVX2/AVX-512 ChaCha20 in `pgw_fast`, or the system OpenSSL from Rust),
   then turn on the batched ring claim and wire `take_checked_many` into the engine's multi-session
   step (`engine/backends.py`, outside this package). This is what scales one gateway process across
   cores.
2. Pad format v2 (section 6): about half of the remaining per-slot floor.
3. Re-measure thread scaling on an idle multi-core host (section 4).
4. Factory side at 70B rates: register the ring arena as CUDA pinned memory (`cudaHostRegister`) so the
   GPU's device-to-host copy lands in the slab directly, saving one host memcpy per slot (about 0.4 ms
   per 7B slot [d]). This needs a GPU run and a change in `factory/generator.py`, a shared file.
5. The store tier's take decrypts a 1 MiB chunk per session per linear (32 sessions decrypt the same
   chunk 32 times). This matters only during outages, so it was left alone.
