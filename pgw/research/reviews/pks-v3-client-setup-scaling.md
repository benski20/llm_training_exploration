# pks-v3 client setup and per-token CPU at 32B-72B: what was measured, changed and left

Date: 2026-11-26. Scope: the client side of the SDK with the default `mask_scheme="pks-v3"`, `hint="local"`: the
one-time setup (weights + the local pks-v3 hint build) and the steady per-token client CPU of the mask factory. The
scheme and its parameters are unchanged (`pks-v4-prime-upload.md`): every change here computes the same values
faster or in another order, and every integrity check is as strong as before. Box: this 4-vCPU Sapphire Rapids
(AVX-512, AMX, IFMA), shared with other jobs; CPU-seconds at one thread are the robust figures, wall times are this
box's at four threads. Benchmarks: `benchmarks/provision_hint_bench.py` (`--path old|new`, `--threads`, `--reps`;
"old" runs the previous `pgw_pks` build), `benchmarks/pks_client_slot_bench.py`, `benchmarks/prepare_real_bench.py`.

## 1. Results (before -> after)

The per-model summary is in `STATUS.md` (section "2026-11-26: client setup and per-token CPU at 32B-72B"); this note
keeps the reasoning and the measured pieces.

End to end, measured (`prepare_real_bench.py`, Qwen2.5-0.5B with the masked head, provider on loopback, 4 vCPU):
the client's whole setup (gateway pack, factory pack, hint) took 145.4 s wall / 383 CPU-s / 2.78 GB peak RSS before
and 67.9 s / 189 CPU-s / 2.14 GB after; 0.915 GB downloaded and 2.17 GB on disk in both.

### 1.1 Hint build: CPU per hint row

Hint rows (output rows of every masked linear, the masked head included): 7B 1.54M, 32B 4.81M, 72B 7.01M.

Where the time went (7B, one thread, CPU-s for the whole model, `provision_hint_bench.py --threads 1 --reps 2`):

| part | before | after | change |
|---|---|---|---|
| `H = W8 A mod 2^56` (int8 GEMM, AMX via oneDNN) | 1635 | 373 + 44 (planes) | `A`'s byte planes built once per linear (were rebuilt and transposed for every 1024-row block), 7 planes instead of 8 (the top byte of `A < 2^56` is zero), 512-column GEMM blocks (AMX runs 2x faster on them than on 28672 columns), native recombination |
| switch `round(q1 H / 2^56) mod q1` | ~250 | ~3 (fused) | native, fused into the key switch (no switched copy of `H`) |
| key switch + ModDown + NTT form | ~340 | ~164 | hand-written AVX-512 forward NTTs (were the auto-vectorised portable ones), one residue pass for four primes, pairwise lane sums, constant-prime remainders instead of 64-bit divisions; also speeds the provider's reply key switch (same kernel) |
| **total** | **2226** (1443 us/row) | **582** (377 us/row) | **3.8x** |

Everything is bit-identical: the new kernels against the numpy reference for every ISA copy and thread count, and a
whole store against `pks.compact_hint` (`tests/hemask/test_hint_build.py`); the fused provisioning build against a
separate build with the same seeds, file for file (`tests/provision/test_prepare.py`).

**Per CPU kind.** The int8 GEMM is exact and fast only where oneDNN has VNNI/AMX kernels. With
`ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI` (no AMX) the 7B build is 934 CPU-s (605 us/row; before 2562). On AVX2-only CPUs oneDNN's int8
kernel saturates on extreme operands (`cpu_int8_available()` says so on this box when capped at AVX2), so the exact
float64 path runs: ~5800-6800 CPU-s at 7B (3.7-4.4 ms/row) before and after (the GEMM is the whole cost there). An exact AVX2 kernel (7-bit planes: `vpmaddubsw` cannot saturate with operands
`< 2^7`) would be at most ~2x faster than the float64 path on paper; not built (open item).

### 1.2 One pass over the weights, resumable, ahead of time

* `provision(hint="local", factory_pack=True)` builds each linear's hint while that linear's `W8` (downloaded and
  checked against the pinned manifest's SHA-256, exactly as before) is in memory for the factory pack, instead of
  writing the factory pack first and re-reading and re-hashing all of it for the hint. The factory pack's final full
  verification against the fingerprint is unchanged.
* `resume=True`: the factory pack keeps `factory-pack.partial` with a list of the objects written and synced; the
  hint keeps `<store>.partial` whose state (seeds, per-linear SHA-256 and row norm) is HMAC-SHA256-authenticated
  under the cache's hint key, with its own domain label (`HintBuild`). A state that does not verify is discarded
  with its seeds; a finished linear whose file changed is rebuilt; a store becomes visible (HMAC'd `meta.json`,
  rename) only when complete, so the hint still exists before any reply secret. Tested: interruption and resume
  (finished linears neither rebuilt nor downloaded again), an edited state, a flipped byte in a finished file.
* `pgw sdk prepare --model ...` (`pgw.sdk.prepare`) runs this ahead of the first request, on every core, with
  progress; a client started later finds the prepared store (`find_prepared_hint`: the cache's hint key must verify
  its `meta.json`; without the key nothing is used; a store that fails the MAC raises `IntegrityError`) and hands it
  to its factory instead of building a second hint. Tested end to end with the real provider (tiny model): prepared
  hint used, answers bit-identical to the plaintext reference, tampered metadata refused.
* **Big box once, copy the cache** (module docstring of `pgw.sdk.prepare`): copy `<cache>/models/<fp16>/` and
  `<cache>/keys/pks-hint.key`. On the target, `meta.json` must verify under that key (HMAC over seeds, row norms and
  every file's SHA-256), the factory re-hashes every hint file when it opens the store, and the packs are verified
  against the pinned fingerprint as always. The key authenticates, it does not encrypt; nothing in the cache is
  secret from the provider (the seeds are public), but the key must travel over a trusted channel and stay 0600.

### 1.3 The factory pack: needed, kept

The factory needs `W8` after the hint exists: its private check vectors `v W8` (40-bit check of every decrypted row
against a provider that picks its errors) are drawn per process and **redrawn after every rejected segment**
(`MaskGenerator._wipe_private`: a provider must never learn anything about vectors still in use). Dropping the
factory pack would mean re-downloading `W8` (6.5 / 32 / 70 GB) at every factory start and after every rejection, or
keeping `v W8` from a fixed `v` across a rejection, which the soundness argument forbids. So it stays: 6.5 / 31 / 70 GB
on disk (int8 weights; the plain-only files are hard links to the gateway pack). Streaming it per segment is not
viable either (a segment touches every linear). What changed is that it is now written in the same pass as the hint
and resumable.

## 2. The byte-plane dial (v3 note, section 8): evaluated, not implemented

The dial: the client computes the low `j` byte planes of `H` (`j/7` of the local GEMM) and the provider sends the
upper `56 - 8j` bits, checked mod `2^(56-8j)` (same 2-adic argument). Client CPU and bytes are both linear in `j`,
between the provider path (`j = 0`) and the local build (`j = 7`); the weight stream (6.5 GB at 7B) is needed for the
check at every `j`.

Measured endpoints at 7B (one thread, after this work): local GEMM + planes 417 CPU-s (AMX) / ~770 (VNNI) / ~5200
(AVX2, float64); provider-path check (`A X`, `U^T W8`, decode + `U^T H X`) ~370-440 CPU-s plus 44.2 GB; the finish
(~165 CPU-s) is common.

* **AMX and VNNI hosts:** the local build costs about as much CPU as the provider path's check alone (417-770 vs
  ~400 CPU-s) and needs no `H` at all. Every `j < 7` adds bytes and saves at most ~370 CPU-s (~1.5 min on 4 cores); the
  dial cannot win.
* **AVX2 hosts (no exact int8 GEMM):** wall time with download and CPU overlapped is `max(CPU(j)/cores,
  bytes(j)/link)`, the maximum of an increasing and a decreasing linear function, so the optimum can be interior.
  4 cores, 1 Gbit/s, 7B: `j = 0` 405 s (network bound), `j = 7` ~1550 s (CPU bound), best `j = 1` ~350 s: **13 %**
  faster than the existing provider path, in a band of link speeds of roughly 0.5-2 Gbit/s; at 10 Gbit/s `j = 0`
  wins, below ~0.3 Gbit/s `j = 7` does.
* Cost of building it: a new `H` form on the provider and in the wire format, the check mod `2^(56-8j)`, a review
  delta to Q7. For at most ~13 % in a narrow band on CPUs that are also the slowest for everything else, this is not
  worth the review surface. **Not implemented.** A cheaper lever for those hosts is an exact AVX2 int8 kernel
  (section 1.1).

## 3. Hint cache size: no lossless gain

The stored hint is `c1` in the NTT domain mod the two ~2^24 primes, 3 bytes per residue: 48 bits per coefficient pair
against `log2(p0 p1) = 47.98` bits of entropy (the values are computationally indistinguishable from uniform: that is
what the hint's security rests on). The `.npy` files carry a 128-byte header and no padding. At most 0.04 % could be
saved; nothing to do. Sizes: 2.37 / 7.38 / 10.77 GB (7B / 32B / 72B with the masked head; 1.5 KB per output row).
(The 10 / 22 GB of the planning table were overestimates.)

## 4. Per-token client CPU (factory, customer side)

`benchmarks/pks_client_slot_bench.py` runs the customer's per-segment work exactly as `PKSMaskBackend.matmul` does
(r keystream, encryption + reply keys, the reply keys' wire form, decryption with the compact hint) at Qwen2.5-0.5B
shapes, with a stand-in provider reply, and extrapolates per part: `r` and the upload with the input widths, the
decryption with the output widths, the reply keys with the number of linears.

Measured at 0.5B (one thread, ms per mask slot): segments of 64 slots 34.0, 256 slots 9.9, 1024 slots 8.0. The 256-slot
group is the unit of encryption, and the upload's public forward NTTs are amortised over the groups of one request:

| CPU ms per mask slot (1 thread) | 0.5B | 7B | 32B | 72B |
|---|---|---|---|---|
| before: 256-slot segments, reply-key wire form in numpy | 8.4 | 28.4 | 90.8 | 137.4 |
| 256-slot segments, native wire form | 7.8 | 27.6 | 89.0 | 135.2 |
| 1024-slot segments (four groups), native wire form | 7.2 | 24.9 | 80.0 | 120.8 |
| of which at 1024: upload / decrypt / r / reply keys + wire form | 2.3 / 2.0 / 0.6 / 2.4 | 10.7 / 8.9 / 2.6 / 2.8 | 35.3 / 29.8 / 8.5 / 6.4 | 55.5 / 44.0 / 13.3 / 8.0 |
| cores at 25 slots/s, before -> after | 0.21 -> 0.18 | 0.71 -> 0.62 | 2.27 -> 2.0 (2.23 at the default stock) | 3.43 -> 3.02 (3.38 at the default stock) |

(0.5B measured, best of 5; the other columns extrapolated per part. A 64-slot segment costs 30.2 ms per slot at
0.5B: the first segment pays for a whole group, once.)

Changes:

* SDK default `mask_segment_slots` 256 -> 1024 (still capped by the stock target), and pks-v3 segments rounded down to
  whole 256-slot groups (`pgw.sdk.factory.segment_slots`): a partly filled group costs as much as a full one. At 7B
  the default 8 GB stock (~1400 slots) allows 1024-slot segments; at 32B / 72B the 8 GB stock holds ~430 / ~290 slots,
  so the gain there needs `mask_store_gb` of ~20 / 28 GB (documented in the option). Masks and outputs do not depend
  on the segmentation.
* The reply keys' wire form natively (`pgw_pks.pack_ksk`, same bytes, tested): 2.0 -> 0.67 ms per (linear, group);
  10 % of the per-slot CPU at 0.5B, 1.5-3.5 % at 7B-72B.
* The `r` keystream (OpenSSL ChaCha20, 2.0 ns per element here) is not the 25 ms/slot the earlier model assumed
  (5.8 ns per element); 14 ms/slot at 72B. Left as is.
* Not changed: the upload and decryption kernels themselves (hand-written AVX-512/IFMA already, section 7.7 of the
  plan), decrypt-ahead (a latency question, not CPU), threads (the factory child already uses every core at low
  priority). The first segment (64 slots, fast first masks) pays a whole group: 3.4x per slot, once.

## 5. Open items

* Exact AVX2 int8 GEMM for hosts without VNNI/AMX (7-bit planes), ~2x on the hint GEMM there.
* The public-check path for unpinned models (`source="hf"`) downloads the whole bf16 checkpoint into the Hugging Face
  cache and packs it (15 / 65 / 145 GB of download and peak disk at 7B / 32B / 72B). A shard-at-a-time packer that
  deletes each shard once its tensors are packed would bound the disk to one shard plus the pack; better, bake the
  32B / 72B fingerprints into the pin table so this path is never taken.
* The masked head's `W8` is derived from the plain f32 head (5 GB of transient RAM at 72B); a row-block derivation
  would bound it.
* Per-slot upload cost at 32B / 72B with the default 8 GB stock (one group per request): caching the upload's public
  NTTs across requests needs ~3 KB per input feature (13 GB at 72B); not worth it, use a larger stock.
