# Weight-free CPU gateway at 7B: CPU per trip, storage and the per-token budget (package W1)

Date: 2026-11-22. Local measurements only (no Modal, no Kaggle): 4 shared vCPUs (Xeon @ 2.3 GHz, AVX-512,
AMX), synthetic Qwen2.5-7B shapes (zero linear weights so a stub provider's zero answer is exact; real
weight-row norms injected with `--wnorm-frac 0.3`), K32A12C27, 7 Freivalds vectors (20 bits).

Goal: 30+ tokens/s for 7B and larger without the customer needing a GPU or large storage, privacy and
exactness unchanged. Deployment assumed: a CPU-only gateway the customer controls (own cloud account or
confidential VM) in the provider's region, the customer premises a thin client.

## 1. What changed in this package

* **Masked remote LM head (opt-in, `pgw.engine.masked_head`)**. `MaskedHeadPack(pack)` adds the head as one
  more masked linear `"<L>.lm_head"` `(vocab, hidden)`: int8 per-row quantization (SPEC 5) of the pack's
  plain `lm_head`, a deterministic function of a tensor the base fingerprint covers; its fingerprint is
  `SHA-256("pgw/v1/masked-lm-head\0" || base)`, so its masks and provider pack never mix with the base
  pack's. `enable_masked_head(model)` installs a `HeadMode` that sends the final-normed rows as one masked
  call per sequence (prepare, range check, Freivalds on every row; the head is the model's last linear, so
  its check is synchronous before the logits are sampled) and releases the float head (1.09 GB at 7B).
  The logits are those of the int8 head: a **new pinned reference** (like `float_ops = "native"`). Tests
  (`tests/engine/test_masked_head.py`): masked == plain bit for bit (sync, async, TripEngine one-call path),
  a tampered head result is refused, close to the float head (same argmax on the tiny model). Library API
  only: the service registry, factory CLI and gateway runtime do not wire it (both parties must wrap the
  same pack).
* **AVX2 Freivalds output half** (`trip.rs`: `fo_run` dispatches an AVX2 build; wrapping integer arithmetic,
  identical results). Warm, per call: gate_up 140 -> 47 us, qkv 17 -> 9 us, the 7B head 668 -> 470 us.
* **`PGW_EMBED_COMPACT=1`** (opt-in): a bfloat16-exact embedding table stays bfloat16 in RAM (1.09 instead of
  2.18 GB at 7B); looked-up rows are widened exactly (test: same logits bit for bit).
* `WindowedReservation.ensure`: the per-part sum of reserved rows stops once both decisions are known (same
  decisions; was 5-9% of the decode prepare's samples).
* Benches: `decode_cpu_bench.py --masked-head --float-ops native --wnorm-frac`, per-kind prepare/finish
  breakdown; `trip_cpu_bench.py --wnorm-frac`; new `native_trip_floor_bench.py` (the native kernels of one
  token at real shapes, cache-realistic, no Python orchestration).

Measurement caveat found: with all-zero synthetic weights the weight-row norm is 0, and the constant-time
width selection then evaluates all 11 widths instead of 2; earlier zero-weight benches overstated prepare
by about 2x. `--wnorm-frac 0.3` (Qwen2.5 int8 rows measure <= 0.37) fixes that.

## 2. Gateway CPU at 7B shapes, all linears remote + masked head

Native kernels alone, warm cache, one row (`trip_prepare_into` with the Freivalds input half,
`trip_finish_into` split mode, `trip_freivalds_output`), microseconds:

| linear (n -> m) | prepare + input half | finish | Freivalds output (after AVX2; before) |
|---|---|---|---|
| qkv 3584 -> 4608 | 9.8 | 5.0 | 8.7 (17) |
| o 3584 -> 3584 | 9.5 | 4.3 | 7.8 (15) |
| gate_up 3584 -> 37888 | 9.9 | 29 | 47 (140) |
| down 18944 -> 3584 | 50 | 4.3 | 7.9 (16) |
| lm_head 3584 -> 152064 | 10 | 167 | 470 (668) |

Native floor per token, cache-realistic (`native_trip_floor_bench.py`, 113 trips, distinct per-linear
buffers; includes the r keystream and the `W r` read):

| stage, ms per token | c = 7 (today) | c = 1 (traffic of a one-vector check, see 5) |
|---|---|---|
| r (ChaCha20, 6.68 MB keystream) | 3.8 | 4.4 |
| W r read (6.17 MB) | 1.2-1.3 | 1.5 |
| prepare (+ input half) | 6.3-6.7 | 4.8 |
| finish (head with full check) | 4.3-5.1 | 4.7 |
| **main core** | **15.6-16.9** | 15.5 |
| Freivalds output (helper core) | 7.2-8.1 | 1.6 |
| check vectors read | 66.6 MB | 9.5 MB |

The measured Python path (`decode_cpu_bench.py`, 2 layers + masked head, `PGW_TRIP_FAST=1`, native float
ops, stub provider with no delay), per call wall: prepare qkv 305, o 218, gate_up 300, down 440, head
525 us; finish qkv 78, o 65, gate_up 131, down 135, head 672 us. Per layer about 1.7 ms of trip
prepare/finish plus about 0.5 ms of float ops and model glue; one head trip + sampling 1.8-1.9 ms.
Extrapolated to 28 layers: **about 57-75 ms of gateway time per token** (this box; 57-65 ms with
`PGW_NOISE_ASYNC_SAMPLED=1`; with a 0.25 ms stub round trip the 2-layer run gives 9.3 ms per token, i.e.
~105 ms or 9.5 tokens/s at 28 layers; every run's outputs repeat-identical), of which the native
kernels are about 16 ms. Where the rest goes (py-spy): the reservation's take (copies out of the prefetch
stage and their wipes, extent check vectors), window bookkeeping, the outbound-noise monitor's sampled
statistics (5% of calls, ~400 us each; `PGW_NOISE_ASYNC_SAMPLED` moves them off the critical path), numpy
allocations and Python dispatch, around 150-400 us per trip.

**Not reached: <= 8 ms per token.** Even with no Python at all the 7B all-remote token needs ~16 ms on the
main core here (plus 7 ms on a helper core). Getting to ~8 ms needs, in order of size:

1. A native per-trip path (take from the RAM ring + prepare + send + receive + finish in one call per
   trip; the payload written straight into the lane's buffer): removes ~50 ms of the 65-75.
   *Done in part (package W2, `PGW_TRIP_NATIVE`, see STATUS.md):* one native call per side of the lane call
   fed by a reservation lease (disk segments), `r` of the next row derived ahead on a helper thread. Measured
   7B trips 48 -> 27 ms per token, decode process CPU 127 -> 78 ms per token (extrapolated, loaded box). Not
   yet: leases on RAM-ring segments, the `W r` decrypt moved off the main core, the noise every-call check and
   request object in native code.
2. A one-vector Freivalds check over a prime field (see 5): check-vector traffic 66.6 -> 9.5 MB per token,
   helper core 7-8 -> 1.6 ms, prepare -1.5 ms.
3. r off the critical path: the r keystream (3.8-4.4 ms) is independent of the activations, so the
   prefetch thread can derive it ahead on another core (it already stages decode rows; make it the only
   path). A pad format with 4 keystream bytes per element at K <= 32 halves it (coordinated format change,
   STATUS X2).
4. Then the critical path is W r read + prepare + finish, about 8-10 ms on this box, nearer 5-6 ms on a
   current server core with ~2x the single-core memory bandwidth.

## 3. Storage on the gateway (Qwen2.5-7B, K32)

| item | size | note |
|---|---|---|
| linear weights | **0** | all four linears remote; the gateway reads only scales and norms |
| LM head | **0** (masked head) / 1.09 GB bf16 (local head) | the masked head is opt-in |
| embedding table | 1.09 GB on disk (bf16); 1.09 GB RAM with `PGW_EMBED_COMPACT=1` (2.18 GB f32 today) | see 4 |
| norms, scales, biases | ~0.8 MB norms + ~5.6 MB per-row scales | |
| masks per slot (token) | **6.17 MB** `W r` (5.56 layers + 0.61 head) | `r` is never stored: derived from the segment seed (SPEC 7.2/7.3, already so) |
| per segment | 23.4 MB `uW` (c = 7) at rest; 43 MB of `u` in RAM while active (derived) | amortized over the segment's slots |
| demand at 30 tokens/s | **185 MB/s** of `W r` | 1.5 Gbit/s from a partner factory, or the gateway's own encrypted factory |
| rolling buffer | 2 s = 0.37 GB, 10 s = 1.85 GB, 60 s = 11 GB | the RAM ring (SPEC 7.8) sized in seconds of demand; the encrypted disk store as reserve only if wanted |
| hint store (encrypted factory on the gateway) | 34.2 GB layers + ~3.7 GB head = **~38 GB** (2x in NTT form) | only with gateway-side encrypted mask generation; a partner factory needs none |
| KV cache (float32) | 115 KB per token: 0.47 GB at 4k context, 3.8 GB at 32k | bfloat16 KV would halve it (not exact against today's reference) |

Narrower exact `W r` encoding: the gateway needs `W r` only to unmask, and `y` is bounded by the cap
(|y| <= 2^27), so `W r mod 2^29` (the low C + 2 bits) determines `y` exactly while Freivalds keeps checking
the full `x_hat`/`y_hat` (it never reads `W r`): 6.17 -> 5.59 MB per slot (-9.4%). Not implemented (segment
format change: packing, factory, ring, store readers).

Mask production at 30 tokens/s is the dominant cost of a gateway without a GPU: a gateway-side encrypted
factory needs ~38 GB of hints and, from the measured 7B decrypt (38 ms/slot on a 4-vCPU Xeon without fast
int8, 27 with NTT hints) plus encryption, roughly 2-3 cores per 30 tokens/s stream; a partner factory moves
the work off the gateway at 1.5 Gbit/s of inbound masks per stream.

## 4. Embeddings: store the table, do not look it up privately

* Stored table: 1.09 GB bf16 on disk (memory-mapped, only touched rows hot), exact, one 7 KB row read per
  token, no trip, no privacy question (the gateway's own memory).
* Private lookup as a masked linear on a one-hot input (`W = E^T`, 3584 x 152064): one more sequential trip
  per token (the next token's row depends on sampling), a 152064-wide masked upload (608 KB per token at
  K32), 1.2 MB of r keystream (~0.7 ms CPU), 14 KB of `W r` per slot, and a 152064 x 3584 int8 weight on the
  provider. And it is not exact: the int8 per-row quantization of `E^T` changes the hidden state (a new
  reference for every layer, not just the head), and the one-hot also needs a 152064-wide activation, so
  `A` bits fall to keep the cap. Rejected: storing 1.09 GB is cheap; `PGW_EMBED_COMPACT=1` keeps the RAM at
  1.09 GB.

## 5. Freivalds over a prime field (recommended next format change)

Today's check runs over Z_{2^32}, where a vector catches an error only to the error's 2-adic valuation, so
7 vectors are needed for 20 bits. Checking `u . y == (u W mod p) . xq (mod p)` on the unmasked integers
(`|y| <= 2^27` after the range check, `xq` from prepare) with one secret vector `u` uniform mod a prime
`p` (2^31 - 1 or 2^61 - 1) catches any wrong `y` except with probability `1/p`: 31 or 61 bits with one
vector. Both halves fuse into passes the gateway already makes (prepare has `xq`, finish has `y`), the
provider sees nothing new, and every row is still checked. Needs `u W mod p` per segment from the factory
(an integer product with `W8`) instead of `u W mod 2^K`: a normative segment/factory change (SPEC 7.2,
encrypted factory), not done here. Effect measured with the one-vector traffic: 66.6 -> 9.5 MB of check
vectors per token, helper core 7-8 -> 1.6 ms per token.

## 6. Per-token budget, 7B, CPU-only gateway

Trips per token: 28 x 4 + 1 (masked head) = **113** sequential round trips. Wire per token: 3.34 MB up,
6.17 MB down (the head's reply alone 608 KB: 0.49 ms at 10 Gbit/s, 0.19 ms at 25 Gbit/s).

| configuration | trips (0.25 ms / 0.5 ms) | gateway CPU on the critical path | float ops + glue | ms per token | tokens/s |
|---|---|---|---|---|---|
| today's Python path (measured here, extrapolated from 2 layers + head) | 28 / 57 | ~45 | ~14 | 105 at 0.25 (measured: stub delay 0.25 ms); 57-65 at 0 | **9.5** (0.25) / ~7.5 (0.5) |
| native per-trip path, c = 7, r inline | 28 / 57 | ~16 | ~3 | 47 / 76 | 21 / 13 |
| + r prefetched on core 2, prime check c = 1 | 28 / 57 | ~9 | ~3 | 40 / 69 | 25 / 14.5 |
| same on a 2x faster core | 28 / 57 | ~5 | ~2 | 35 / 64 | 28.5 / 15.6 |

The trip column includes the provider's GPU time and the wire serialization only if they fit in the
quoted round trip. 30 tokens/s at 7B with 113 trips needs the round trip itself at or below ~0.2 ms *and*
under ~5 ms of everything else per token; at 0.5 ms the ceiling is 17.7 tokens/s with zero gateway CPU.

**Strict speculation (reviewed mode, L3).** Per pass of R rows: 113 trips, gateway mask work for R rows
(about R x the per-row native cost, masks charged at C per token), draft work for R - 1 tokens. Per token:
divide by tau (accepted tokens per pass). With tau = 2.84 (measured 1.5B target / 0.5B draft, R = C = 4)
trips per token fall to ~40 (10 ms at 0.25 ms), which is what makes 30 tokens/s reachable at 0.25-0.5 ms,
but on a CPU-only gateway the draft costs: a 0.5B int8 draft streams ~0.5 GB of weights per draft token
(25-50 ms per token on one core here), far more than it saves; an EAGLE-style one-layer drafter (~0.2-0.4
GB at 7B) still costs ~10-20 ms per draft token, and the drafter weights are storage on the gateway. Mask
demand per token rises to C/tau of the R rows' cost (about 1.4x at C = 4, tau = 2.84) and wire per pass
x R. Prompt-lookup drafting (no model, tau ~ 1.08 on chat) is nearly free but gains ~8%. Net: speculation
helps only with a drafter that is cheap on CPU (a few ms per draft token) and a high tau; otherwise it adds
CPU, storage and mask demand.
