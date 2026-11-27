# Consumer devices on home links: bytes per row and per mask, their floors, and what can be cut

Date: 2026-11-27. Setting: the client is a consumer device on the internet. For every linear of every row it sends
the masked input `x + r` and receives the masked output (ring K32A12C27 on the standard schedule, K56C53 on fused2).
The question here is bandwidth, not latency: how many bytes a row and a mask slot move, how far that can go down
without weakening privacy, integrity or fidelity, and what that means per link class.

Everything below is exact counting or measured on this box (4 vCPU, CPU only). The exceptions are the GPU rates
and the latency terms of `benchmarks/round_trip_model.py`, which are estimates from earlier reviews.

* Numbers: `research/reviews/consumer_bandwidth_numbers.py` (writes
  `benchmarks/results/bandwidth/consumer_bandwidth_numbers.json`).
* Measurements: `benchmarks/narrow_wire_bench.py` (writes
  `benchmarks/results/bandwidth/narrow_wire_qwen05b_2026-11-27.json`).
* Prototypes, default off:
  * `pgw.core.narrow`;
  * `pgw.engine.narrow_backend.NarrowMaskedBackend`;
  * `pgw.hemask.pks_narrow`;
  * the `--links` bandwidth term of `benchmarks/round_trip_model.py`.

## 0. Summary

1. **Today, per decode row** (standard schedule 4L + 1, masked int8 head, 4 bytes per element). The prefill row
   column has no head outputs. The pks-v3 slot is what the factory uploads and downloads per row.

   | model | up | down | online total | prefill row | pks-v3 slot up | pks-v3 slot down | online + masks |
   |---|---|---|---|---|---|---|---|
   | 0.5B | 0.73 MB | 1.82 MB | 2.55 MB | 1.94 MB | 1.95 MB | 2.05 MB | 6.55 MB |
   | 1.5B | 1.53 | 3.19 | 4.71 | 4.10 | 3.45 | 3.59 | 11.8 |
   | 3B | 2.48 | 4.74 | 7.21 | 6.60 | 5.34 | 5.33 | 17.9 |
   | 7B | 3.34 | 6.17 | 9.51 | 8.89 | 6.63 | 6.94 | 23.1 |
   | 32B | 11.03 | 19.22 | 30.25 | 29.62 | 21.08 | 21.62 | 72.9 |

   * fused2 (2L trips) and parallel blocks (L + 1) move almost the same element counts at K56, so 1.75x the bytes:
     4.4 / 16.6 / 52.9 MB per row at 0.5B / 7B / 32B for fused2.
   * Lane headers are 0.08-0.36 % of a row and TLS records 0.13 %. Ethernet, IP and TCP framing take 5.9 % of the
     line rate, and ACKs send 2.1 % of the downstream bytes back up.

2. **Floors.** I proved these in the model where the provider sees one-time-padded values in a ring `Z_N`, applies
   `W8` over the integers, and the client decodes each output coordinate from its own residue (sections 3-4).
   * Down: each output needs at least `log2(2 B_j + 1)` bits, with `B_j` the public bound on `|y_j|`. At K32A12C27
     that is at most **C + 1 = 28 bits**; today's wire sends 32.
   * Up, for a client without weights: each input needs at least the output modulus, so **28 bits**.
   * Up, for a client that holds `W8`: there is no fixed floor. It is a trade between bytes and client work. With
     `m`-bit inputs the client corrects a sparse carry vector itself.
   * The masked values are uniform by construction, so no lossless compressor can shrink them.

3. **Sound reductions, prototyped, exact and tested.** Outputs are bit-identical to the plain reference on the tiny
   model and on Qwen2.5-0.5B.

   | reduction | who can use it | 0.5B measured | all sizes (provable) |
   |---|---|---|---|
   | narrow rings, weight-free (inputs at `max_j w_j` <= 28 bits, outputs at per-row widths `w_j` <= 28) | every client, existing K32 masks truncated | up -13.1 %, down -15.5 %, total **-14.8 %** | -12.5 % |
   | + carry-corrected inputs at m = 16 (client holds `W8`) | clients with the factory pack | up **-50 %**, total **-25.3 %** | -23 to -26 % |
   | + the LM head computed on the device | clients with `W8` | total -44 % | -28 to -44 % |
   | pks-v3 with a narrow message ring (30-bit replies) | pks-v3 clients | mask download -16.7 % | same margin as v3 or better |
   | masks made on the device (`W8 r` locally) | clients with `W8` | mask traffic **-100 %** | needs 5-71 ms of one CPU core per slot at 0.5B-7B |

   Best sound case per row, with the device holding `W8`, carry16, the local head and device-made masks, compared
   with today (online + pks-v3):

   | model | today | best case | reduction |
   |---|---|---|---|
   | 0.5B | 6.55 MB | 1.42 MB | 4.6x |
   | 7B | 23.1 MB | 6.5 MB | 3.5x |
   | 32B | 72.9 MB | 21.8 MB | 3.3x |

4. **It is not enough for interactive chat at 7B or 32B on any home link.**
   * At the best case and 1.3 rows per token, 20 tok/s needs about 4.5 Gbit/s at 32B and 1.4 Gbit/s at 7B.
   * A 500-token prompt with no cache hit takes:
     * 0.5B on cable 100/10: 373 s today, 211 s best;
     * 7B on 1 Gbit fibre: 41 s today, 31 s best.
   * A typical chat day (40 requests) moves 293 / 1034 / 3268 GB today at 0.5B / 7B / 32B, and 88 / 317 / 1001 GB in
     the best case. A 1.2 TB/month data cap allows about 40 GB per day.
   * Bandwidth-only tok/s, one stream (section 7):
     * 0.5B on symmetric gigabit fibre: 17.7 today, 58.9 best;
     * 7B on fibre: 5.0 today, 16.4 best;
     * 7B on cable 100/10: 0.10 today, 0.51 best.
   * At RTT 20 ms the trips x RTT term dominates first: at most ~3 tok/s for every size and link
     (`consumer-round-trips.md`).

5. **Recommendations.**
   * Ship the weight-free narrow wire: lane format 3. It saves 12.5-15 % for every client, keeps today's stocks and
     check, and changes no output bit.
   * For clients that already hold the factory pack, add carry-corrected inputs (m = 16-20) and device-made masks.
     They are the only large cuts (2-3x per row with masks).
   * Treat 7B+ on home links as bandwidth-infeasible for interactive use under this protocol class. Section 9 says
     what would have to change.

## 1. Exact accounting (today's wire)

**Wire formats.**
* Lane `call` / `multi` frames carry `rows x n x K/8` bytes of `x_hat` and `rows x m x K/8` bytes of `y_hat`
  (`rust/pgw-lane/src/proto.rs`; `pgw.core.packing.pack_ring`, little-endian `K/8` bytes per element).
* Headers are 48 B per request and 40 B per reply, plus 8 B per item of a multi frame.
* The WebSocket stream channel carries the same packed bytes inside msgpack (`pgw.wire.codec`, a few dozen bytes of
  framing).
* The pks-v3 maskgen request carries `b` at 56 bits per input value and slot, plus the reply keys
  (`16 x 8192 x (24 + 24 + 30 + 30)` bits = 1.77 MB per 256-slot group and linear, 6.9 KB per slot). The reply is
  `c0` at 36 bits per output value and slot (`pgw.hemask.wire`, `V3_*`).

**Per row and linear** (one row, bytes, K32). Up is `4n` and down is `4m`. At 7B the largest is gate_up at
75.8 KB down per layer: 2I outputs, 76 % of a layer's download. The head adds 152k outputs (0.61 MB) per row that
needs logits. Prefill rows other than the last need none.

**Schedules** (MB per decode row, up / down):

| model | standard K32 (4L+1) | fused2 folded K56 (2L) | parallel K56 (L+1) |
|---|---|---|---|
| 0.5B | 0.73 / 1.82 | 1.27 / 3.18 | 1.12 / 3.04 |
| 7B | 3.34 / 6.17 | 5.82 / 10.77 | 5.14 / 10.10 |
| 32B | 11.03 / 19.22 | 19.27 / 33.59 | 17.01 / 31.34 |

Fusion regroups the same masked elements:
* standard: `L(3d + I)` up and `L(2d + 2I + q)` down;
* fused2: the residual `R` replaces the separate inputs (`consumer-round-trips.md` 2.4).

It needs K56 for quality, so **fused schedules cost 1.75x the bytes of the standard one**. On a bandwidth-bound link
the standard schedule is faster, and on an RTT-bound link fused is faster (section 7).

**Per mask slot (pks-v3).**
* Up: 7 B per input value plus 6.9 KB of reply keys per linear.
* Down: 4.5 B per output value.
* Masks therefore more than double a row's traffic (0.5B: 4.0 MB of masks vs 2.55 MB online). The note
  `mask-cost-consumer.md` section 3 shows that any provider-made mask needs at least about 4 B per input and output.

**Overheads that are not payload.**

| item | share | reducible? |
|---|---|---|
| lane headers | 4L + 1 trips x 96 B = 9.3 KB per pass at 0.5B: 0.36 % of a row (0.08 % at 32B) | no, already fixed-size and public |
| msgpack | tens of bytes per message | no |
| TLS 1.3 | 22 B per 16 KiB record: 0.13 % | no |
| TCP/IPv4 + Ethernet | 1448 B of payload per 1538 B on the wire: 5.9 % | not on the public internet (no jumbo frames) |
| TCP ACKs | about 2.1 % of the downstream volume sent upstream | modelled, see below |

None of these is worth a format change. The model (`LINK_EFF`, `ACK_UP`) includes them.

## 2. The model the floors are proved in

The protocol class of the current design:
* (i) the provider sees, per element, a value that is uniform and independent of the data: a one-time pad over a
  ring `Z_N`;
* (ii) the provider applies `W8` to what it receives. It can only use ring arithmetic, because any non-affine
  function of a padded value cannot be unmasked (`consumer-round-trips.md` section 5);
* (iii) the client decodes each output coordinate from the provider's value for that coordinate and data it holds,
  with work linear in `n + m` per row. Doing work like `W8 x` itself is exactly what is being offloaded;
* (iv) every decoded integer is verified exactly (range + Freivalds over `F_p`) before use.

Within (iii) there are two client classes:
* **weight-free**: no `W8` on the device, as in `factory_weights="stream"` and the gateway pack;
* **W8-holding**: the factory pack, the default client, about 7 GB at 7B.

## 3. Outputs: the floor is C + 1 bits, and we send 32

**Bounds.** At K32A12C27 the gateway quantizes each token with the most activation bits `<= 12` such that
`wnorm ||xq||_2 (1 + 1e-9) <= 2^C`, where `wnorm` is the largest row norm (`pgw.core.ring._quantize_capped`).

* The slack factor is far above float64 error (about `1e-15` relative). So for every row `j`:
  `|y_j| <= ||W8_j|| ||xq|| < 2^C ||W8_j|| / wnorm`.
* Also `|y_j| <= qmax ||W8_j||_1`, with `qmax = 2047`.
* The integer bound is `B_j = min(ceil(2^C rho_j (1 + 2^-40)) - 1, 2047 ||W8_j||_1) <= 2^C - 1`, with
  `rho_j = ||W8_j|| / wnorm` (`pgw.core.narrow.plan_linear`).
* `B_j` depends only on `W8` and the ring: it is public, data-independent and fixed per linear.

**Theorem 1 (output floor).**
* *Setting.* Under (i)-(iii), the provider's message for coordinate `j` is a function `f(y_hat_j)` of its value
  `y_hat_j = y_j + c_j` (mod `N`), where the correction `c_j` is uniform and known only to the client.
* *Claim.* Zero-error decoding needs `H(f(y_hat_j)) >= log2(2 B_j + 1)` bits.
* *Proof.*
  1. For every `c`, the map `y -> f(y + c)` must be injective on `S = [-B_j, B_j]`.
  2. So `f(a) != f(b)` whenever `a - b` lies in `S - S \ {0}`, a cyclic interval of half-width `2 B_j`.
  3. Each value of `f` is therefore taken on a set whose points are at least `2 B_j + 1` apart.
  4. Such a set has at most `N / (2 B_j + 1)` points, so under uniform `y_hat_j` each value has probability at most
     `1 / (2 B_j + 1)`.
  5. Hence `H >= log2(2 B_j + 1)`.
* *Achievability.* The residue mod `2^(w_j)` with `w_j = bitlen(2 B_j)` reaches the floor to within less than one
  bit (power-of-two rings).
* *Variable-length codes do not help.* The provider's value is uniform to the provider, and a variable-length code of
  an input whose value classes all have probability at most `1/(2B+1)` cannot average fewer bits.

**Truncation is exact.**
* The provider computes `W8 x_hat mod 2^32` as today, then keeps the low `w_j` bits.
* The client subtracts `(W8 r)_j mod 2^(w_j)`: the low bits of the stored K32 correction.
* It then centres in `w_j` bits. Every `y` in `[-B_j, B_j]` is recovered because `2 B_j < 2^(w_j)`.

**Integrity is unchanged.**
* A tampered value decodes to some `y'` with `|y'| < 2^(w_j - 1)`. The range check against `B_j` is free.
* An accepted wrong row has `|y' - y| < 2^(w_j) <= 2^(C+1) < p = 2^31 - 1`. So it is non-zero mod `p`, and the p31
  Freivalds check misses it with probability exactly `1/p` per vector, the same bound as today
  (`pgw.core.freivalds`).
* The **ring** check over `Z_(2^K)` relies on headroom above the outputs, so it does not apply. Narrow rings require
  the p31 check, which is already on every trip in the client library.

**What K32A12C27 already exploits, and what is left.**
* The cap is what made K32 possible instead of K40 (2026-11-19: -20 %).
* The wire still spends `K - (C + 1) = 4` bits per element on headroom that only the ring check needed.
* The "48-bit" figures in the mask notes are the provider's switched hint (q1 ~ 2^48) and the reply modulus. They do
  not apply to the online wire.
* Measured on 0.5B:
  * the per-row bounds give **26.9 bits on average** for qkv / o / gate_up / head, and 27.8 for down;
  * rows differ in their `||W8_j|| / wnorm` (every int8 row reaches 127 at its maximum, but the row norms differ);
  * so the measured saving is 15.5 % of the download, against the provable 12.5 % at a uniform 28.

**Ruled out, with reasons.**
* *Data-dependent widths*, such as the client announcing `bitlen(||W8|| ||xq||)` per call: they reveal the norm of
  the activations.
* *Retry when a narrow guess overflows*: the retry is a data-dependent event the provider sees.
* *Lossy truncation of low output bits*: the client then knows `y` only to within `2^t`. An exact Freivalds check is
  impossible, and a tolerance check lets the provider choose the error within the tolerance. That weakens integrity,
  so it is rejected.
* *A smaller cap C*: a fidelity knob, not a free saving. Each bit of C removed costs one bit per element up and down,
  but makes the cap bind on more tokens (fewer activation bits). It is not bit-identical, so it was not pursued here.

**fused2 (K56C53).** The same argument gives `H` and `Z` at C + 1 = 54 bits instead of 56 (-3.6 %), from the
strict bounds of `quantize_pair` (same slack). Design only: the fused2 lane items are not in the client library yet.

## 4. Inputs: 28 bits for weight-free clients, near the activation width for clients holding W8

**Theorem 2 (weight-free input floor).**
* *Setting.* Under (i)-(iii) with no `W8` on the client, the client's unmasking cannot depend on the data through
  `W8`. So the provider's output must equal `W8 x + c(r)` modulo the output modulus.
* *Claim.* Each input element must be masked over a ring of at least the output modulus, at least `C + 1` bits.
* *Proof.*
  1. The provider computes `W8 x_hat` over a ring `Z_N`, with `x_hat` known modulo `N_in`.
  2. Adding `N_in` to `x_hat_i` changes the output by `N_in W8[:, i]`. The output is therefore determined only modulo
     `N_in` (times the gcd of the column, generically 1), so `N_out` must divide `N_in`.
  3. The client needs `N_out > 2 B` (Theorem 1), hence `N_in >= 2^(C+1)` for a power-of-two ring.

  The bound is met: inputs at `max_j w_j` bits (28, measured 27.8 average over 0.5B's linears). Today's K32 masks can
  be **truncated**, because `W8 (r mod 2^m) = W8 r mod 2^m` (tested: `test_weight_free_mode_reuses_k32_masks`).

**Carry-corrected inputs (client holds W8): the construction.**
1. *Masking.* Send `x_hat = (xq + r) mod 2^m` with `r` uniform in `[0, 2^m)` and `2^m > qmax`.
2. *Provider.* It multiplies the integers `x_hat` in `[0, 2^m)` with its unchanged K32 kernel and returns
   `y_hat mod 2^(w_j)`.
3. *Carry identity.* With the carry vector `c = (xq + r - x_hat) / 2^m` in `{-1, 0, 1}`,
   `W8 xq = W8 x_hat - W8 r + 2^m W8 c`.
4. *Client.* It knows `c` (it has `xq` and `r`), so it adds `2^m (W8 c)_j mod 2^(w_j)` on the rows with
   `w_j > m`. This is a sparse column sum over `W8`.

**Properties.**
* *Exact*: an integer identity (tests: every mode, flat, heavy-tailed and aligned inputs, real 0.5B).
* *Private*:
  * `x_hat` is exactly uniform in `Z_(2^m)` for every `xq` (`r -> x_hat` is a bijection; tested exhaustively at
    m = 12);
  * `c` never leaves the device;
  * the masks are `W8 r` for `r < 2^m`: device-made, or pks-v3 with `r` restricted (pks-v3 hides any message).
* *Timing*:
  * each carry is non-zero with probability `|xq_i| / 2^m <= qmax / 2^m`, independently per element. So the count
    is dominated by `Bin(n, qmax / 2^m)` **for every input**;
  * the gather is padded to a data-independent budget (`carry_budget`: overflow probability `2^-40`). Its work does
    not depend on the data;
  * overflow rows stay exact and are counted (0 in every run here).
* *Integrity*: unchanged (the same p31 check on the decoded integers).

**Cost** (client gather as a % of one full int8 product of the row; padded budget / expected at the measured mean
`|xq|` of about 100):

| m (bits up) | 0.5B | 7B | 32B |
|---|---|---|---|
| 14 | 19.8 / 0.61 % | 15.9 / 0.61 % | 15.3 / 0.61 % |
| 16 | 7.3 / 0.15 % | 5.0 / 0.15 % | 4.7 / 0.15 % |
| 18 | 3.2 / 0.04 % | 1.8 / 0.04 % | 1.6 / 0.04 % |
| 20 | 1.7 / 0.01 % | 0.8 / 0.01 % | 0.7 / 0.01 % |
| 24 | 0.7 / 0.001 % | 0.25 / 0.001 % | 0.19 / 0.001 % |

Measured on 0.5B decode rows:

| m | non-zero carries (of 182,144 inputs) | padded gather slots |
|---|---|---|
| 16 | 274 | 11,063 |
| 20 | 17.5 | 2,153 |
| 24 | 2 | 799 |

Mean `|xq|` per linear was 101 / 184 / 142 / 64 / 101 (qkv / o / gate_up / down / head).

The padded budget is the honest cost. It is a sparse gather of `budget x m_out` int8 values per row and linear.
* At 7B and m = 16 that is 5 % of the row's product.
* On a GPU it is negligible.
* On one CPU core at 20 GB/s it is ~16 ms per row, below the RTT at consumer RTTs. m = 18-20 halves or quarters it.

**Why the other carry ideas fail (proved).**
* *Choosing `r` so that no carries occur*: no-carry means `x_hat >= xq`, so `P(x_hat = 0 | xq) = 0` for `xq > 0`. That
  contradicts uniformity. Any distribution of `r` that avoids carries makes `x_hat` depend on `xq`.
* *Making carries predictable*: the client already knows `c`. The obstacle is `W8 c`, an `O(mn)` product without
  `W8`.
* *Sending carries to the provider masked*: `c + s` over `Z_(2^(w-m))` is exactly a second limb of a `w`-bit mask
  (`x_hat + 2^m (c + s) = xq + r'` with `r' = r + 2^m s`). Total bits `m + (w - m) = w`, so nothing is saved
  (any number of levels). Sending `c` in the clear, or `c XOR s`, reveals the carry pattern or needs
  `W8 diag(s) c_hat` (client-side `O(mn)`). Rejected.
* *Carries on the fused2 residual `R`*: possible in principle (the client corrects `H` elementwise and `Z` with
  `W2' c_R`). But `|R|` reaches `2^41`, and a data-independent budget needs the worst case `2^53`, so there is no
  saving. Activations in fused2 can be carry-corrected, with the carry term folded into `R_hat`: `R_hat` gets
  `+ hmul W1 2^m c` (design).

## 5. Masks: download size, idle time, or none at all

**pks-v3 with a narrow message ring (prototype `pgw.hemask.pks_narrow`, reference round trip exact, tested).**

*What changes.*
* With narrow rings a mask is needed only mod `2^k`, with `k = max w_j <= 28`.
* The client encodes `B = a t + e + 2^(56 - k) M` (q = 2^56 unchanged) and decrypts with `round(2^k v / q1) mod 2^k`.
* The provider rounds `c0` to `yb` bits. Its GEMM, hint, key switch and reply key are unchanged.

*Noise.* The budget is `q1 / 2^(k+1)`: 4 bits larger at k = 28. The reply-rounding term `q1 / 2^(yb+1)` grows by 2^6
at yb = 30. The other terms do not change. At the worst row norm of SPEC 7.9 (72B widest input):

| | v3 (k 32, yb 36) | narrow (k 28, yb 30) |
|---|---|---|
| bound | 2^13.35 | 2^17.07 |
| budget | 2^14.98 | 2^18.98 |
| margin | 1.63 bits | **1.91 bits** |

*Result.* `smallest_reply_bits(28) = 30`, so the mask download goes from 36 to 30 bits per output: **-16.7 %**.

*Security.* The same RLWE instance (n, q, sigma, fresh ternary secret) and the same reply key. The message scale is
the client's private choice. Unchanged.

*Upload.* It stays 56 bits. Moving to `q = 2^52` (28 + 24) would cut 7 % and is more secure at the same n and sigma,
but it is a provider parameter change for the external review. Not done.

**Seeded public parts.** Already done everywhere it is sound:
* `a_C` comes from the client's public seed;
* the key-switching `a`-parts come from `ks_seed`;
* the compact hint is public and built locally (`hint="local"`), never downloaded.

What travels is bodies (pseudorandom to the provider) and replies (functions of them). Neither can be seeded.

**Idle-time stock.** The mask factory already runs ahead, but volume makes idle time insufficient for a regular
user. Take a typical day of 40 requests: 600 new prompt tokens and 400 generated tokens x 1.3 rows each, so 44.8k
slots.
* At 0.5B:
  * 179 GB of pks-v3 traffic per day;
  * of which 87 GB is upload: about 21 hours of a 10 Mbit/s uplink;
  * 82 GB of stock to hold a day.
* At 7B the volumes are 4x larger.

Idle-time masks keep the link free during a chat only for light use within the stock (8 GB is 4,400 slots at 0.5B,
about 4 requests).

**Masks made on the device (no mask traffic at all).**
* *How.* A device holding the fingerprint-checked `W8` computes `W8 r` itself, with `ceil(m/8)` byte planes of `r`.
* *Cost per slot*, measured CPU rate 112-247 G int8 MAC/s per core here, 200 used:

  | model | `W8` on disk | m = 28 (weight-free widths) | m = 16 (carry mode) | RTX-4070-class GPU |
  |---|---|---|---|---|
  | 0.5B | 0.5 GB | 10 ms | 5 ms | < 0.1 ms |
  | 1.5B | 1.5 GB | 31 ms | 15 ms | < 0.1 ms |
  | 3B | 3.1 GB | 62 ms | 31 ms | 0.2 ms |
  | 7B | 7.1 GB | 141 ms | 71 ms | 0.2-0.4 ms |
  | 32B | 32 GB | 640 ms | 320 ms | 1-2 ms (weights streamed from RAM; the 4070's 12 GB cannot hold them) |

* *In practice.* The typical day costs 4 core-minutes (0.5B, m = 16) to 53 core-minutes (7B), done in idle time.
  Carry mode halves the mask work, because `r < 2^16` has two byte planes instead of four.
* *Security.*
  * `r` comes from the device's ChaCha20 stream and `W8 r` is computed on the device. No mask material ever leaves it.
  * The provider still sees only the one-time pad `x + r`.
  * Integrity is unchanged: the check vectors `u` and `u W8` are made locally already.
  * A wrong local `W8` would only make decoding fail, and the fingerprint check prevents it.
  * This is the existing local factory (`he_urls` empty) with `W8` on the device.
* *The catch.* A device that can do this can often run the model itself (decode at 7B is ~14 ms per token on a
  500 GB/s GPU). The protocol's consumer value is for models the device cannot run, and those are exactly the cases
  with the largest bytes per row.

## 6. Bit packing: CPU per row (measured, one core)

`benchmarks/narrow_wire_bench.py` uses the maskgen v2 layout (`pack_bits`: whole-byte fields plus a bit tail):

| per decode row | K32 cast pack / unpack | 28-bit numpy pack / unpack | 16-bit pack / unpack |
|---|---|---|---|
| 0.5B up / down | 0.19 / 0.12, 0.46 / 0.35 ms | 0.82 / 0.83, 1.86 / 2.19 ms | 0.09 / 0.05, 0.32 / 0.21 ms |
| 7B up / down | 0.87 / 0.64, 1.78 / 1.23 ms | 3.5 / 3.9, 6.8 / 7.5 ms | 0.67 / 0.37, 1.28 / 0.94 ms |
| 32B up / down | 4.3 / 2.5, 6.9 / 5.5 ms | 22.8 / 18.3, 37.1 / 39.9 ms | 2.4 / 1.9, 4.7 / 5.0 ms |

* 28 bits costs about 8 ns per value in numpy. The existing native `he_unpack_shift` is no faster, because of a copy
  of the input.
* At a 1 Gbit/s link (35.7 M values/s at 28 bits) that is ~0.3 of a core. Every consumer link is slower than the
  CPU.
* Widths that are multiples of 8 (16, 24) are plain casts, cheaper than today's K32.
* A SIMD 28-bit kernel (4-bit tails) should be memory-bound like the cast (estimate, not built).

## 7. Throughput per link class

The bandwidth term is now in the model (`python benchmarks/round_trip_model.py --links`; `row_bytes`, `link_ms`).
For one stream, a pass costs:
* trips x (RTT + transport + client work);
* plus device time;
* plus `rows x ((up + ACK) / uplink + down / downlink)` at 94 % goodput;
* plus the masks' own traffic when they are downloaded while chatting.

The cells below give tok/s as **bandwidth only (1 row) / RTT 5 ms / RTT 20 ms**. The RTT columns use the best of
speculation off and the shipped draft head at R = 2-4 (tau 1.45-1.72). Client work and multi-row kernels are the
model's targets; 32B runs on 2 H100s.

| model | schedule, wire, masks while chatting | DSL 25/3 | cable 100/10 | cable 300/30 | cable 1000/50 | fibre 1000/1000 |
|---|---|---|---|---|---|---|
| 0.5B | standard, K32, pks-v3 (today) | 0.11 / 0.10 / 0.09 | 0.37 / 0.32 / 0.22 | 1.12 / 0.72 / 0.39 | 1.99 / 1.00 / 0.49 | 17.7 / 2.46 / 0.79 |
| 0.5B | standard, K32, masks in stock | 0.36 / 0.30 / 0.21 | 1.24 / 0.77 / 0.41 | 3.71 / 1.40 / 0.59 | 6.84 / 1.84 / 0.69 | 45.3 / 2.92 / 0.84 |
| 0.5B | standard, narrow, pks-v3 narrow | 0.12 / 0.11 / 0.09 | 0.39 / 0.33 / 0.22 | 1.18 / 0.75 / 0.40 | 2.08 / 1.03 / 0.50 | 19.7 / 2.52 / 0.80 |
| 0.5B | standard, narrow, masks in stock | 0.41 / 0.34 / 0.23 | 1.41 / 0.83 / 0.43 | 4.24 / 1.50 / 0.62 | 7.82 / 1.93 / 0.70 | 51.8 / 2.98 / 0.85 |
| 0.5B | standard, carry16, device masks | 0.60 / 0.46 / 0.28 | 2.11 / 1.03 / 0.50 | 6.32 / 1.78 / 0.68 | 12.3 / 2.21 / 0.75 | 58.9 / 3.03 / 0.85 |
| 0.5B | fused2 2L, K56, device masks | 0.21 / 0.20 / 0.17 | 0.71 / 0.60 / 0.42 | 2.13 / 1.40 / 0.76 | 3.93 / 2.00 / 0.98 | 26.0 / 4.50 / 1.53 |
| 0.5B | fused2 2L, carry16 + 54-bit, device masks | 0.35 / 0.32 / 0.26 | 1.25 / 0.96 / 0.57 | 3.75 / 1.95 / 0.97 | 7.47 / 2.82 / 1.20 | 31.7 / 4.77 / 1.57 |
| 0.5B | parallel L+1, carry16 + 54-bit, device masks | 0.42 / 0.40 / 0.35 | 1.51 / 1.26 / 0.86 | 4.52 / 2.85 / 1.53 | 9.28 / 4.22 / 2.01 | 34.4 / 7.66 / 2.76 |
| 3B | standard, K32, pks-v3 (today) | 0.04 / 0.04 / 0.03 | 0.13 / 0.12 / 0.09 | 0.39 / 0.30 / 0.18 | 0.69 / 0.46 / 0.25 | 6.49 / 1.38 / 0.49 |
| 3B | standard, carry16, device masks | 0.19 / 0.17 / 0.12 | 0.67 / 0.45 / 0.25 | 2.02 / 0.84 / 0.37 | 3.83 / 1.14 / 0.44 | 21.5 / 1.84 / 0.55 |
| 3B | parallel L+1, carry16, device masks | 0.14 / 0.14 / 0.13 | 0.50 / 0.46 / 0.36 | 1.50 / 1.16 / 0.71 | 2.97 / 1.89 / 1.02 | 12.9 / 4.13 / 1.66 |
| 7B | standard, K32, pks-v3 (today) | 0.03 / 0.03 / 0.03 | 0.10 / 0.10 / 0.08 | 0.31 / 0.26 / 0.18 | 0.54 / 0.41 / 0.24 | 5.03 / 1.47 / 0.57 |
| 7B | standard, K32, masks in stock | 0.08 / 0.08 / 0.07 | 0.29 / 0.25 / 0.17 | 0.86 / 0.57 / 0.31 | 1.55 / 0.82 / 0.41 | 12.2 / 1.95 / 0.66 |
| 7B | standard, narrow, masks in stock | 0.10 / 0.09 / 0.08 | 0.33 / 0.28 / 0.19 | 0.99 / 0.63 / 0.34 | 1.78 / 0.87 / 0.43 | 13.9 / 2.03 / 0.67 |
| 7B | standard, carry16, device masks | 0.14 / 0.13 / 0.11 | 0.51 / 0.39 / 0.23 | 1.52 / 0.80 / 0.40 | 2.86 / 1.13 / 0.49 | 16.4 / 2.11 / 0.68 |
| 7B | fused2 2L, carry16 + 54-bit, device masks | 0.09 / 0.08 / 0.08 | 0.30 / 0.28 / 0.23 | 0.91 / 0.72 / 0.45 | 1.75 / 1.15 / 0.64 | 8.89 / 2.77 / 1.11 |
| 7B | parallel L+1, carry16 + 54-bit, device masks | 0.11 / 0.10 / 0.10 | 0.38 / 0.35 / 0.31 | 1.13 / 0.96 / 0.68 | 2.22 / 1.65 / 0.97 | 9.88 / 3.99 / 1.82 |
| 32B | standard, K32, pks-v3 (today) | 0.01 / 0.01 / 0.01 | 0.03 / 0.03 / 0.03 | 0.10 / 0.08 / 0.06 | 0.17 / 0.14 / 0.09 | 1.59 / 0.56 / 0.23 |
| 32B | standard, carry16, device masks | 0.04 / 0.04 / 0.04 | 0.16 / 0.13 / 0.09 | 0.47 / 0.29 / 0.15 | 0.87 / 0.40 / 0.19 | 5.18 / 0.84 / 0.29 |
| 32B | parallel L+1, carry16 + 54-bit, device masks | 0.03 / 0.03 / 0.03 | 0.12 / 0.11 / 0.10 | 0.35 / 0.31 / 0.24 | 0.69 / 0.55 / 0.36 | 3.15 / 1.49 / 0.74 |

How to read the table:
* The **uplink** binds on asymmetric links: on cable 1000/50 a 0.5B row's 0.73 MB up takes 8x as long as its 1.82 MB down.
  That is why carry16 (half the upload) nearly doubles the bandwidth-only rate there.
* Speculation costs bytes (R rows per pass for tau tokens). On bandwidth-bound links the best setting is often
  speculation off. The model picks per cell.
* **Fused schedules lose on bandwidth-bound links** (1.75x bytes per row; bandwidth only, standard always wins) and
  win at RTT >= 5 ms on fast links. At RTT 20 ms the crossover is around cable 100/10 (25/3 at 0.5B).
* The local head (not in the table) drops a further 0.53 MB down per row: 27 % / 7.5 % / 2.4 % of a carry16 row at
  0.5B / 7B / 32B.

**TTFT, 500-token prompt, no cache hit, RTT 20 ms** (prompt rows have no head outputs; seconds):

| | DSL 25/3 | cable 100/10 | cable 300/30 | cable 1000/50 | fibre 1000/1000 |
|---|---|---|---|---|---|
| 0.5B today / carry16 | 1274 / 730 | 373 / 211 | 126 / 72 | 71 / 39 | 10.3 / 8.1 |
| 7B today / carry16 | 5835 / 3337 | 1705 / 961 | 570 / 322 | 319 / 173 | 40.6 / 30.5 |
| 32B today / carry16 | 19352 / 11079 | 5651 / 3189 | 1887 / 1066 | 1055 / 573 | 133 / 99 |

**GB per day** (light 10 / typical 40 / heavy 150 requests, as in section 5):

| | 0.5B | 1.5B | 3B | 7B | 32B |
|---|---|---|---|---|---|
| K32 + pks-v3 masks (today) | 73 / 293 / 1100 | 132 / 526 / 1974 | 200 / 801 / 3004 | 258 / 1034 / 3877 | 817 / 3268 / 12256 |
| K32, masks in stock or on the device | 29 / 114 / 429 | 53 / 211 / 792 | 81 / 323 / 1212 | 107 / 426 / 1598 | 339 / 1355 / 5082 |
| carry16 + device masks | 22 / 88 / 329 | 40 / 159 / 597 | 60 / 241 / 904 | 79 / 317 / 1188 | 250 / 1001 / 3752 |

## 8. Prototypes and tests

**`pgw.core.narrow`** (default off; nothing imports it on an existing path):
* `plan_linear`: per-row bounds and widths, input width, a public wire order grouped by width;
* `mask_inputs` and `carry_term`: carries, the padded gather and `carry_budget`;
* `mask_correction`, `decode_outputs`: range-checked against `B_j`;
* `pack_up` / `unpack_up`, `pack_down` / `unpack_down`: length-checked, with strict zero padding;
* `provider_apply`: the provider's whole job, the unchanged K32 product and the low bits.

**`pgw.engine.narrow_backend.NarrowMaskedBackend`**: a `LinearBackend` that runs each call as a narrow trip in
process. It counts the narrow and K32 bytes.
* Client steps: quantization identical to `PlainBackend`; a fresh ChaCha20 mask per call; the device-made correction;
  the carry term; the range check; the p31 Freivalds check with a secret `u`.
* Provider step: it receives only the packed bytes.

**`pgw.hemask.pks_narrow`**: `PKSNarrowParams` (k, yb) over the unchanged reference functions;
`smallest_reply_bits`.

**`benchmarks/round_trip_model.py --links`** and the functions behind it: `row_bytes`, `link_ms`,
`mask_bytes_per_slot`, `LINKS`, `WIRES`, `MASKS`.

**Measured, Qwen2.5-0.5B-Instruct** (`narrow_wire_bench.py`: chat prompt of 19 tokens and 6 greedy decode tokens
with the masked int8 head; every logit compared with the plain reference: **bit-identical in all four modes**):

| mode | decode row up | decode row down | total vs K32 | prefill row up / down |
|---|---|---|---|---|
| K32 today | 728,576 B | 1,824,256 B | - | 725,181 / 1,248,499 B |
| weight-free narrow | 633,360 B | 1,541,988 B | -14.8 % | 630,389 / 1,053,311 B |
| carry m = 24 | 546,432 B | 1,541,988 B | -18.2 % | 543,885 / 1,053,311 B |
| carry m = 20 | 455,360 B | 1,541,988 B | -21.8 % | 453,238 / 1,053,311 B |
| carry m = 16 | 364,288 B | 1,541,988 B | -25.3 % | 362,590 / 1,053,311 B |

**Tests.**
* `tests/core/test_narrow.py` (42):
  * bounds hold for normal, heavy-tailed, flat (cap-binding) and row-aligned (Cauchy-Schwarz) inputs;
  * widths are minimal;
  * exactness in every mode and on an uncapped ring;
  * K32 masks truncate;
  * exact uniformity: `r -> x_hat` is a permutation of `Z_(2^12)` for extreme inputs, and the carry count is
    exactly `|x|`;
  * chi-square on the wire bytes for zero and extreme inputs, and data-independent payload sizes;
  * the carry budget is never exceeded at the worst-case density (20,000 trials);
  * 64 of 64 single-bit reply tamperings are caught;
  * bad inputs are refused;
  * the tiny model is bit-identical to `PlainBackend` in four modes over prefill and decode, with fewer bytes;
  * a tampering provider raises `IntegrityError` end to end.
* `tests/hemask/test_pks_narrow.py` (3): the reply width and margin; the reference round trip at k = 28 / yb = 30 is
  exact for 28-bit and 16-bit `r`; the production class keeps its guard.
* `tests/benchmarks/test_round_trip_model.py` (+5): the row bytes equal the wire formats, the link term adds per row,
  and the table prints.

**Not built.**
* Lane format 3: narrow `call` / `multi` frames whose widths come from the ticket's public plan, and the provider's
  truncation in `rust/pgw-lane`.
* A native `quantize_mask_pack` for narrow widths and carries.
* The narrow pks-v3 reply in the provider (`yb` is a wire constant today) and the client's decryption scale.
* Carry mode in the native decode driver.
* fused2 at 54 bits.

The prototype is an exactness and measurement vehicle. Its in-process provider is not a speed number.

## 9. What is impossible (in this protocol class)

1. **Compressing masked traffic.** Every masked value is uniform over its ring by construction, so no lossless code
   shortens it on average.
2. **Outputs below `log2(2 B_j + 1)` bits** for a client that decodes per coordinate: Theorem 1. That means at least
   ~27 bits per output at K32A12C27. The download is 63-71 % of today's online bytes, so **any sound scheme keeps at
   least ~53-60 % of today's online bytes** (87.5 % for a client without weights), unless the client takes over
   linears (the local head is one).
3. **Inputs below the output modulus without `W8` on the client**: Theorem 2. Clients without weights cannot go below
   28 bits up.
4. **Lossy output truncation** (a few high bits only). Integrity needs the exact integer; a tolerance lets a
   malicious provider steer results within it.
5. **Data-dependent widths or retries**: they reveal activation norms or overflow events.
6. **Provider-made masks below ~4 B per input and output** (`mask-cost-consumer.md` section 3). Masks downloaded
   while chatting at least double the link load. pks-v3 with 30-bit replies is close to that floor; the gap is the
   upload's 7 B per input.
7. **Interactive chat at 7B or 32B on home links.** In the best sound case, 20 tok/s needs ~1.4 Gbit/s at 7B and
   ~4.5 Gbit/s at 32B of combined link. On top of that, the trips x RTT wall caps a single stream at ~0.3-3 tok/s at
   RTT 20 ms. A 500-token prompt at 7B takes at least 30 s even on symmetric gigabit fibre. A typical usage day moves
   at least ~300 GB at 7B.

What would change the picture is outside this protocol class.
* Move linears to the device:
  * the head (done here as an option);
  * the attention linears, about 20 % of the bytes; the device then computes them;
  * at the limit, the whole model, where the device can run it.
* Smaller boundary activations: an architecture with fewer, narrower MLP crossings, since every gate/up value must
  cross the wire twice.
* Cryptographic offload that does not return per-coordinate values (HE with packed outputs or FSS). These have been
  quantified elsewhere at 10^3-10^5x the compute (`research/plans/fewer-round-trips.md`).

## 10. Next steps (if the product pursues consumers)

1. **Lane format 3 with weight-free narrow widths.**
   * The plan comes from the pack (public): `in_bits`, the per-row widths, the wire order.
   * The provider applies `& (2^w - 1)` after its K32 kernel. The client truncates its K32 corrections.
   * The p31 check is required.
   * Saves 12.5-15 % for everyone, with bit-identical outputs and no new masks.
2. **Carry mode in the native client** for factory-pack clients (m = 16 on GPUs, m = 20 on CPU-only devices), with
   device-made masks at `ceil(m/8)` planes.
3. **The 30-bit pks-v3 reply** (`yb` negotiated per request) for pks-v3 clients. Fold it into the external review
   packet with the noise table of section 5.
4. **A local LM head option** for factory-pack clients: -27 % / -7.5 % / -2.4 % of a carry16 row at 0.5B / 7B / 32B.
5. **Product.** Price consumer plans by bytes per row; show TTFT honestly. Target 0.5B-3B on fibre or metro links,
   or keep consumers on a gateway next to the GPUs (topology B), where the bytes stay in the data centre.
