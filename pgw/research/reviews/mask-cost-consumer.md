# Mask cost at consumer scale: 20 tok/s up to 32B (cost model, ideas, prototypes)

Date: 2026-11-27. Scope: what one pks-v3 mask slot costs on the provider GPU and on the client device, and what
can be cut without weakening privacy or integrity and with bit-identical outputs. A slot is one masked row,
i.e. one fresh `r` for every linear of the model (masked LM head included), and it is used once. Numbers come
from `research/reviews/mask_cost_numbers.py`, which drives the calibrated device-item model in
`pgw.service.memplan` (H100 datasheet rates, kernels calibrated on a T4) and the measured client CPU of
`STATUS.md` (2026-11-26, round 3). Two Kaggle T4 runs of `pgw-maskcost-t4` measured the prototypes (section 9).
**Device numbers are estimates unless marked measured.**

Prototypes (default off, exactness-tested):
* coalesced provider items: `PKSProvider.run_items` and the group commit behind `PGW_SERVICE_PKS_COALESCE_MS`;
* the opt-in common upload seed `pks.crs_a_seed` (`PGW_PKS_CRS_A`);
* the customer kernels in exact torch, `pgw.hemask.pks_torch`, for a client GPU.

## 0. Verdict

1. **Bandwidth, not mask compute, is the binding constraint for a consumer device on the internet.** Two
   streams make it up:
   * the online trips move every masked activation of every linear: 30.3 MB per row at 32B, 9.5 MB at 7B;
   * pks-v3 masks add 42.7 MB per slot at 32B and 13.6 MB at 7B (7 B up per input value, 4.5 B down per
     output value).

   At 20 tok/s and 2 slots per token, a 32B user needs **9.7 Gbit/s for the online trips plus 13.7 Gbit/s for
   masks**. A 7B user needs 3.0 + 4.3 Gbit/s. No mask-cost change fixes the online part.
   * The provider-made mask stream has a floor of about 4 B per input value and 4 B per output value, so masks
     made by the provider at least double a user's traffic. pks-v3 is at 1.75x / 1.1x of that floor.
   * Only masks made on the device remove the mask traffic. A device with a GPU and the int8 weights can compute
     `W8 r` itself, which is exact and private by construction (section 6.4).
   * **The 20 tok/s / 32B / consumer-internet target is not reachable with this protocol family** without a
     change to the online trips. That change belongs to the trip owners, not to mask generation. Section 3 has
     the numbers.
2. **Provider maskgen, the part that can be fixed: 3.5-11x at consumer stock sizes, with a stated review
   condition.**
   * A consumer device can hold only one 256-slot group of masks. At 32B a slot's masks are 19.2 MB, so a group
     is 4.9 GB.
   * With one group per request, every item recomputes the hint `W8 A` and the key switch's forward NTTs for
     256 slots. That gives 48 slots/s at 32B per H100, not the 135 quoted for 1024-slot items.
   * The hint is public and depends only on (weights, public seed). Coalescing the groups of many requests under
     one seed into one device item runs the hint once for all of them (prototyped; bit-identical).
   * Across users this needs a common public seed. That uses the same standard assumption but raises a
     multi-user precomputation question, so it is opt-in and goes to the external review (Q8, section 4.2).
   * Result at 32B: 48 -> 143-508 slots/s per H100 (8-GPU node, upload hidden behind a second item). At 7B:
     164 -> 473-1684. The range is the memplan model as is at the top, and at the bottom the model with the
     per-group cost multiplied by the 3-5x the T4 measured (section 9).
   * **Users per H100 at 20 tok/s and 2 slots/token: 32B 1.2 -> 3.4-13.5, 7B 4.1 -> 12-48.** That is 3.5-11x.
     Against the SDK's 1024-slot items (stock a consumer cannot hold at 32B) it is 1.1-3.7x.
   * Measured on a T4 with 8 items: coalescing gives 2.2-3.7x per linear, with byte-identical replies.
   * Beyond the top of the range, the provider is bound by its NIC: 500 slots/s x 42.7 MB = 171 Gbit/s per GPU at
     32B.
3. **Client: from cores to a fraction of a GPU.**
   * Today the client spends 113 ms of one core per slot at 32B (34.5 ms at 7B): 4.5 / 1.4 cores per user at
     2 slots/token and 20 tok/s.
   * The NTT-heavy 80 % (encryption, reply keys, decryption, self-check) is exact integer work that ports to a
     GPU. Prototyped in exact torch, bit-identical to the reference on CPU and CUDA.
   * Measured on a T4, eager torch int64 is slower than one tuned CPU core: 159 ms per slot at 32B. The
     decryption with the provider's fused NTT kernel takes 27.5 ms; the eager int64 tail (not timed separately)
     is the likely remainder.
   * Fully fused kernels, estimated from the provider kernels' calibrated T4 rates, come to ~4-6 ms per slot on a
     T4 and ~2 ms on an RTX 4070-class GPU (section 9).
   * The CPU keeps the wire framing, the writer and the keystream: ~22 ms per slot at 32B, 0.9 cores per user.
   * A Metal / wgpu port needs only 32-bit integer multiplies with high halves (section 6.2).
4. **K56 masks for fused trips (section 12).**
   * Today's 7-digit factory makes a fused2 slot 7x a K32 slot.
   * Without new cryptography, two changes bring it to 1.2-1.6x on the provider at consumer stock:
     * the top-digit plan: 3-4 K32 products, because the top 32 bits need only `W d mod 2^32`;
     * stacking the planes as extra slots of one item, so the hint is computed once.
     Both are prototyped, bit-identical, and tested on the real pks-v3 factory.
   * A direct K56 parameter set (`q = 2^80`, `q1 ~ 2^72`, `P ~ 2^84`; noise identical to v3 term for term;
     148 / 153-bit core-SVP, inside the HE-standard tables) costs ~1.45x everywhere.
     * Prototyped as an exact reference; it needs review (Q9).
     * It wins by ~2x when the common seed is also adopted.
   * No stage of the fused trip can stay at K32 exactly.
5. Ideas rejected for consumer use:
   * a larger slot group (fewer key-switch products per value, but 4x the client stock);
   * reusing the upload secret or the reply key across linears (covers a passive provider only);
   * sharing mask stock across a user's devices (risks reuse);
   * the previously rejected pks-v4, transposed hint and secret basis B, which stay rejected.

## 1. Where the work of a slot goes today

Shapes per slot, Qwen2.5 with the masked head:

| | linears | sum of out rows `m` | sum of in features `n` | masks stored per slot |
|---|---|---|---|---|
| 7B | 113 | 1.543M | 0.835M | 6.2 MB |
| 32B | 257 | 4.805M | 2.758M | 19.2 MB |

**Provider, H100 estimate, ms per slot by stage.** One item covers one linear and `G` groups (256 slots each).
`G = 1` is a consumer at 32B with the default 8 GB stock; `G = 4` is the SDK's 1024-slot segments.

| stage | 7B G=1 | 7B G=4 | 32B G=1 | 32B G=4 | depends on |
|---|---|---|---|---|---|
| key switch (fused NTTs) | 3.06 | 0.99 | 9.54 | 3.09 | ~85 % on the hint (forward NTTs of `A~`, 16 x 4 primes per 32 rows), ~15 % per group (products, inverse NTTs, ModDown) |
| hint GEMM `W8 A` (4096 columns, 7 int8 planes) + switch | 2.33 | 0.58 | 9.12 | 2.28 | the hint only |
| `y = W8 b` (7 planes) | 0.15 | 0.15 | 0.59 | 0.59 | per group |
| upload `b` + keys over PCIe (exposed) | 0.08 | 0.32 | 0.13 | 0.96 | per group (hidden by a second item) |
| planes, keys, reply, ring rows | 0.49 | 0.18 | 1.39 | 0.49 | mostly per group |
| **total ms per slot** | **6.1** | **2.2** | **20.8** | **7.4** | |
| slots/s per H100 | 164 | 455 | 48 | 136 | |

The model is affine in `G`: a full-model cycle costs `c + g G` seconds. For 7B, `c = 1.33 s` and `g = 0.23 s`.
For 32B, `c = 4.58 s` and `g = 0.74 s`. **At consumer stock sizes the hint is 85-87 % of the provider's
work**, and it is public and identical for every group under the same seed.

**Client, measured, one core, ms per slot** (STATUS 2026-11-26 round 3):

| part | 7B | 32B | kind of work |
|---|---|---|---|
| `r` keystream | 1.6 | 5.2 | ChaCha20 |
| upload: pad `a_C t` (3-prime NTTs + CRT), CBD(21) noise, reply keys, wire | 17.9 | 55.4 | NTT, sampling |
| decryption `c0 + c1 t'` (2-prime inverse NTTs, rounding) | 7.7 | 25.8 | NTT |
| 40-bit private self-check | 3.1 | 10.2 | dot products |
| `u W8` (p31 check material) | 1.2 | 7.8 | GEMV over `W8` |
| writer | 3.0 | 8.9 | memory |
| **total** | **34.5** | **113** | |

## 2. The theoretical minimum, and how far we are from it

Per slot at 32B (`research/reviews/mask_cost_numbers.py`, `ops`):

| quantity | count | note |
|---|---|---|
| plaintext `W8 r mod 2^32` (local, no HE) | 1.28e11 int8 MAC | 4 byte planes of `r` |
| provider `y = W8 b mod 2^56` | 2.24e11 int8 MAC | floor for any upload of 56 bits |
| provider hint `W8 A` | 9.17e14 int8 MAC per hint | per slot: divided by `256 S`, with `S` the groups sharing it |
| key switch per group | ~94 modmul per output value: 4.5e8 | 64 products + 26 inverse butterflies + ModDown |
| key-switch forward NTTs | 5.1e11 modmul per hint | divided by `256 S` |
| client pad / decryption | 5.8e7 / 7.2e7 modmul | about 21 per input value and 15 per output value |

* **Upload width (why 56 bits).** It is the floor for this scheme. `W8 e` must stay below the `2^24` message
  scale: `24 sigma ||w||` reaches `2^20.7` in the worst case, so the scale needs 24 bits on top of the 32-bit
  ring. Fewer planes would need a smaller noise, but CBD(21) with `sigma 3.24` is already the HE-standard
  minimum. A 53-bit randomised upload switch (as in v2) saves 5 % of the upload at +0.2 bit of noise; it is not
  worth a scheme change.
* **Provider floor.** `y` plus the per-group key switch: ~0.5 ms per slot at 32B on an H100 (2.24e11 MAC at
  1.2 POPS, plus 4.5e8 modmul).
  * Today at `G = 1`: 20.8 ms, **about 40x the floor**. At `G = 4`: 7.4 ms (15x).
  * The gap is almost entirely the per-hint work spread over too few slots (section 4.1). After that, the
    modelled key-switch kernels are the gap: they run at ~2 % of the H100's integer rate, being bound by
    shuffles and layout changes (memplan note).
* **Client floor.** Every input value needs `r`, noise and its share of the pad; every output value needs its
  share of one inverse NTT and two roundings. That is ~25 + 15 integer operations per value, ~1.5e8 per slot at
  32B, or 15-40 ms of one AVX-512 core.
  * Today: 113 ms, ~3-7x off on CPU. The CPU kernels were tuned (round 3) and their remaining bound is not
    pinned down.
  * On a GPU the same 1.5e8 operations take about a millisecond (section 6.2).
* **Bandwidth floor** (section 3). It is the floor that matters for consumers, and pks-v3 is already close to
  it.

## 3. Bandwidth: the consumer blocker

**Per slot:**

| | 7B | 32B |
|---|---|---|
| online trip up (masked inputs, 4 B) | 3.3 MB | 11.0 MB |
| online trip down (masked outputs, 4 B) | 6.2 MB | 19.2 MB |
| mask upload (7 B per input + reply keys) | 6.6 MB | 21.1 MB |
| mask download (4.5 B per output) | 6.9 MB | 21.6 MB |

**Per user at 20 tok/s** (slots per token: 1.3 for decode-only strict speculation, 2.0 midpoint, 2.6 measured
including prompts):

| model | slots/token | online trips | pks-v3 masks |
|---|---|---|---|
| 7B | 1.3 / 2.0 / 2.6 | 2.0 / 3.0 / 4.0 Gbit/s | 2.8 / 4.3 / 5.6 Gbit/s |
| 32B | 1.3 / 2.0 / 2.6 | 6.3 / 9.7 / 12.6 Gbit/s | 8.9 / 13.7 / 17.8 Gbit/s |

* The masked values are uniform in `Z_2^32`, so nothing compresses.
* Information-theoretically, a provider-made mask stream needs at least `4 B x (n + m)` per slot. The client
  must receive `m` words it cannot compute cheaply, and the provider must receive an encryption of `n` words.
  So **any provider-made mask scheme at least doubles the user's traffic**.
* Merging masks into the online trip ("HE online": send `Enc(x)` instead of `x + r`) would halve the bytes, but
  it fails. The reply key (2 MB per linear) and the hint both amortise only over a 256-slot group sharing one
  upload secret. Rows of one trip cannot share that secret: equal `a` and `t` leak `x_1 - x_2`. So it is
  rejected.
* The provider NIC: masks alone at 32B allow 29 / 73 / 293 / 1171 slots/s at 10 / 25 / 100 / 400 Gbit/s. At
  7B: 92 / 230 / 921 / 3685.
* What actually reduces bytes:
  1. **Masks made on the device** when it has the weights and a GPU (section 6.4). No mask traffic at all.
  2. A smaller ring: K28 instead of K32 is possible if the accumulator bound (C27) allows it; that is the trip
     owners' call, and it would cut ~12 %.
  3. A topology where the "consumer" gateway sits next to the GPU: what topology B already is.

## 4. Provider ideas

### 4.1 Coalesced items: one hint for many groups (prototyped, sound, default off)

`PKSProvider.run_items(fp, kern, items, a_seed)` runs several items of one linear under one public seed as one
device item:
* `pks_gpu.item` takes a `MultiUpload`: each item's wire bytes are unpacked on its own, with no repacking;
* the reply keys are concatenated;
* the hint rows (`W8 A`, the switch, the key switch's forward NTTs) are computed once per row block for all
  groups;
* `pks_gpu.split_reply` cuts the packed reply back per item.

`PGW_SERVICE_PKS_COALESCE_MS` (default `0`: off) turns `run_item` into a group commit. The first item of a
(model, kernel, linear, seed) waits that long, or until `PGW_SERVICE_PKS_COALESCE_GROUPS` (16) groups have
joined, then runs the batch and hands every caller its own reply.

* **Fidelity.** In the key switch, each reply ciphertext depends only on its own 32 rows, its group's `y` and
  its group's reply keys, plus the public hint (`pks_gpu.item`'s docstring and the per-row-block structure).
  So every item's bytes equal the item alone. Tested on CPU:
  * `MultiUpload` replies equal the items alone and the host path, for array and wire-byte parts and for
    `PackedReply` and array outputs;
  * `run_items` gives three customers exactly what `run_item` gives each, on both the device and the cache
    path, and each customer decrypts `W8 r` with its own hint;
  * the group commit gives each of three concurrent callers its own reply.

  On CUDA the bench compares coalesced against one-by-one bytes; see section 9 for the measured result.
* **Security.**
  * Nothing the provider receives changes. It already holds every item.
  * Only public data is shared between items: the seed's hint, a function of `W8` and the public `a`.
  * No customer's data enters another's reply. Replies are separate wire objects, and a group's ciphertexts are
    computed from its own inputs.
  * A refusal is isolated. If any item's reply keys are out of range, the batch is rerun item by item, so only
    that item is refused (tested). A tenant that keeps sending bad keys only makes its batches fall back. Rate
    limiting by tenant would bound even that.
* **Cost.** `T = c + g S` for `S` groups instead of `S (c + g)`. The batch's uploads stay on the device for the
  item: `7 x (4096 + 256 S) x n_pad` bytes, 2.4 GB at 32B `down` with `S = 32`. That is why the default cap is 16
  groups. The joiners' worker threads wait while the opener runs. Their maskgen slots stay held, so the batch is
  one slot's work; the per-device byte budget sees one item.
* **Within one customer** (per-customer seed): coalescing helps only when that customer has several requests of
  one linear in flight, which happens with large stocks. A consumer at 32B has one group in flight per linear.
  So the gain at consumer scale needs 4.2.

### 4.2 A common public upload seed across customers (opt-in, needs sign-off: Q8)

`pks.crs_a_seed(fingerprint, epoch)` is `SHA-256(domain || fingerprint || 0 || epoch)`. `PGW_PKS_CRS_A=1` (default
off) makes new hint stores and provisioning use it instead of a fresh seed (`pks.upload_seed`). Each customer
still:
* builds its own compact hint;
* draws its own private `ks_seed`, so its own `k`, which never leaves it (F13);
* draws fresh `t`, `e`, `t'` and noise per (linear, group).

Only the public `a_C` is shared, so the provider's `W8 A` is the same for all customers of the model and
coalescing (4.1) spans customers.

**Security argument.**
* **Standard model: unchanged assumption.**
  * Each upload is a decision-RLWE instance `(a_C, a_C t + e)` with a fresh ternary `t` and fresh CBD(21) `e`.
  * `a_C` is already shared by all of one customer's groups (millions of instances).
  * The hybrid of the packet (section 3: replace `beta` by uniform, then `b` by uniform) runs per instance, with
    `a` public in both. It never used that `a` is unique to a customer.
  * A common `a` across customers changes only the number of instances per `a`. That number enters the
    standard multi-instance bound linearly, exactly as the existing per-customer reuse does.
  * The reply-key instances are untouched: `k` stays per customer and private.
* **Seed integrity.** The seed is a hash of public strings, computed locally by every client and never accepted
  from the provider.
  * A provider-chosen `a` could be `0`, which makes `b = e + 2^24 M` reveal `M`. That is why the client derives
    the seed itself.
  * Nothing-up-my-sleeve: SHA-256 output under a fixed domain. Nobody chooses it, as with Kyber's per-key seed,
    except that this one is global.
* **What does change: multi-user precomputation.**
  * Kyber and FrodoKEM use per-key seeds partly so that no one lattice is worth attacking for everyone
    ("all-for-the-price-of-one", backdoors).
  * With a common `a` per model, a lattice-reduced basis of the `a`-part lattice, computed once, would help
    against every customer's uploads (BDD with preprocessing). Today it helps against one customer's.
  * Known preprocessing algorithms do not bring the upload's 238.9-bit classical core-SVP (216.8 quantum) near
    128. They change constants and exponents in the BDD phase, not the basis-reduction cost that dominates.
  * This is still a different model (non-uniform, multi-user), so it needs the reviewer's sign-off.
  * Mitigation: an `epoch` in the seed. Rotating it re-keys everybody, at the cost of a hint rebuild per
    customer: provider-built `H` with the client-verified check, `pks-v3-provider-hint.md`.
  * Multiparty-HE practice is a precedent: a common reference polynomial `a` shared by all parties, as in
    multiparty / threshold BFV and CKKS.
* **Fidelity:** unchanged. The seed is just another public seed, so the hint and replies are deterministic as
  before (tested: two customers on the common seed decrypt exactly; a customer's hint does not decrypt another's
  reply).

**Q8 for the reviewer:** with `a_C` common to all customers of a model (SHA-256 of a public label) instead of
per customer, are the upload instances (n = 4096, q = 2^56, ternary, CBD(21)) still at the packet's security
level, given that an attacker may precompute on that one `a` for all customers? Is an epoch (re-key) period
needed, and of what length?

### 4.3 What the provider gains (H100 estimates; `mask_cost_numbers.py`)

Slots per second per H100 against `S`, the groups sharing one hint:

| S | 1 (today, consumer) | 4 (today, SDK) | 16 | 64 | 256 | infinity |
|---|---|---|---|---|---|---|
| 7B | 164 | 455 | 819 | 1023 | 1091 | 1116 |
| 7B, upload hidden by a second concurrent item | 166 | - | 1173 | 1684 | - | 1971 |
| 32B | 48 | 136 | 249 | 314 | 337 | 345 |
| 32B, upload hidden | 48 | - | 350 | 508 | - | 597 |

"Upload hidden": the service runs 2 items per device by default, and the item-by-item model does not count the
overlap. The RunPod stage in section 10 measures it.

**Where `S` comes from.** Every user keeps one group (stock) in flight; the provider sweeps the linears, and each
linear's hint serves every waiting group. Consumption sets a latency bound: a user drains 256 slots in
`256 / (20 x slots per token)` seconds (6.4 s at 2.0). The sweep must finish within it. An 8-GPU node splits the
linears, so its sweep is 8x shorter at the same per-GPU throughput. Users per H100 under this model (each user one
group of stock):

| model, slots/token | today S=1 | today S=4 (stock a consumer cannot hold at 32B) | coalesced, 1 GPU | coalesced, 8-GPU node | coalesced, 8 GPUs, upload hidden |
|---|---|---|---|---|---|
| 7B, 2.0 | 4.1 | 11.4 | 22.1 | 27.2 | **47.9** |
| 7B, 1.3 / 2.6 | 6.3 / 3.2 | 17.5 / 8.8 | 37.1 / 15.7 | 42.2 / 20.7 | 74.4 / 36.5 |
| 32B, 2.0 | 1.2 | 3.4 | 2.5 | 7.9 | **13.5** |
| 32B, 1.3 / 2.6 | 1.9 / 0.9 | 5.2 / 2.6 | 7.1 / 0.5 | 12.5 / 5.9 | 21.6 / 10.1 |

* These are the memplan model's numbers. On a T4 the per-group part measured 3-5x the model, which gives the lower
  bounds of section 9: 32B 3.4 and 7B 12 users per H100 at 2.0 slots/token.
* At 32B a single GPU cannot sweep the model's hint within a consumer's 4.9-6.4 s: its hint alone is 4.6 s.
  Hence the 8-GPU column.
* Other GPUs, 32B, `S = 1 / 64`, upload not hidden:

  | GPU | S = 1 | S = 64 | why |
  |---|---|---|---|
  | H100 | 48 | 314 | |
  | H200 | 48.5 | 316 | HBM does not bind the fused kernels: they are SM-bound |
  | B200 | 68 | 384 | int8 4.5 POPS, 1.25x SM rate assumed |
  | MI300X | 60 | 360 | |

  B200 and MI300X need their kernels ported or validated (MI300X: Triton on ROCm, int8 `_int_mm`). None of them
  changes the picture as much as `S`.

### 4.4 Device hint cache (with 4.2; design, not built)

* With a common seed, the switched hint of a model is one object for all users: 34 GB at 7B and 114 GB at 32B
  (48-bit packed).
* It fits next to the int8 weights:
  * at 7B on one H100 (34 + 7.6 GB) or an H200;
  * at 32B on a B200 or MI300X (114 + 32 GB of 192), or split over a TP pair of H100s.
* It removes the hint GEMM entirely: `c` loses `hint_gemm_switch`, 0.22 ms per slot at `S = 64` on 32B. The
  forward NTTs of `A~` stay unless they are cached too, at 64 KB per row (300 GB at 32B), which is too large.
* It is worth it only after 4.1 and 4.2, and it is worth less than they are: with `S >= 64` the hint is already
  < 10 % of the work.
* Security and fidelity: public data, deterministic. Same as 4.1.

### 4.5 The per-group key switch on tensor cores (design, estimate)

* After 4.1-4.2, the per-group key switch (1.08 ms per slot at 32B) and `y` (0.86) are what is left.
* With many groups per hint, the products `sum_l A~_l beta_{l,g}` are, for each NTT slot and prime, a
  `(blocks x 16) x (16 x S)` matrix product. Exact on int8 tensor cores with an Ozaki-style split:
  * 30-bit residues in 4 bytes on each side give 16 int8 products;
  * the sums are below 2^21 for `K = 16`;
  * the result is recombined mod `p` with 7 shift weights.
* That is ~16 int8 MACs per modular MAC, at ~60x the CUDA-core modmul rate on an H100.
* Expected: the products, about two thirds of the per-group key switch's arithmetic, become nearly free. The
  inverse NTTs and ModDown remain: **~1.4-1.8x on the key switch, ~1.2-1.3x overall.**
* Not built: the fused kernels are bound by shuffles and layout, and a gain must be measured on an H100 before it
  is claimed. Exactness is a plain integer identity, testable bit for bit like the other fused kernels.

### 4.6 Fewer limbs, smaller moduli: no room without loss (shown)

The noise budget is `q1 / 2^33 = 2^14.98` against a worst case of `2^13.35`, with 1 bit kept as margin
(refusal at `2^13.98`).

| option | effect |
|---|---|
| a 1-prime `P` (30 bits) | key-switch noise `(q1/2) sigma sqrt(16 x 8192) / P` grows by `2^30`: far over |
| a 2-digit gadget instead of hybrid | 3 primes but 32 products instead of 16: more work |
| reply 35 bits instead of 36 | +1 bit on the reply-rounding term: total `2^13.7`, inside the refusal threshold with 0.3 bit to spare; saves 2.8 % of the download and none of the compute |
| upload `q < 2^56` | needs a smaller `sigma` (not LWE-standard) |
| a 4-prime forward key switch ("fwd4") | exists; slower on the T4 |

### 4.7 Larger slot groups (fewer key-switch products): rejected for consumers

* Splitting the 4096-coefficient upload ring as 4 features x 1024 slots (rank 4) instead of 16 x 256 has two
  effects:
  * the hint is amortised over 4x the slots;
  * the products drop from 64 to 16 per output value.

  Security is unchanged (the same RLWE over `R_u`). But the client must hold 4x the stock (20 GB at 32B per
  group) and a 4x larger compact hint (28 GB at 32B).
* The opposite direction (64 slots per group) helps consumer stock but costs the provider 4x in key-switch
  products and hint amortisation. It is only worth having where 4.2 is refused.

### 4.8 Previously rejected ideas: still rejected

* **pks-v4 prime upload** (`pks-v4-prime-upload.md`). Every variant loses 22-83 bits of reply-key core-SVP or
  v3's noise margin. The cheap check of the sketch is unsound: the lift to `P` is not linear. It was about
  provisioning bytes anyway, not per-slot cost.
* **Transposed hint / secret basis B** (`weight-free-client-checks.md`). These concern the client's check
  material (`u W8`, 7.8 ms per slot at 32B).
  * The transposed product is unsound without an anchor.
  * Basis B relies on a long-lived secret, whose snapshot defeats every later check. Today a snapshot exposes only
    the segments in stock.
  * No sound variant was found here: any anchor must be `W8`, a long-lived secret, or a public commitment plus
    proofs. The GPU port (6.2) makes `u W8` cheap instead.
* **One reply key or upload secret for all linears of a group.** Reply keys are ~4 % of the client upload work.
  This is covered for a passive provider only (plan 7.8), so it stays out.

## 5. Fewer masked vectors per token

* **Fusion (fused2: 2L+1 trips; with parallel blocks L+1).** Trips are fewer, but the masked elements per token
  are the same counts regrouped: `L(3d + I)` up and `L(2d + 2I + q)` down.
  * The reparametrisation `r_R = v - W1 r_x` keeps the factory at the same single-stage products `W1 r_x` and
    `W2' v`, with no chain. The composite `W2' = int8(W2 diag(g sw1))` has `W2`'s shape.
  * **So the hint does not grow, and the mask cost per slot is unchanged in shape.**
  * What changes is the ring: fused2's quality fix needs K56 (7 B per element), and pks-v3 covers K32 only. A
    K56 variant at the same noise ratios needs the following (all noise terms equal to v3's, so the same
    2^13.35 against 2^15):

    | parameter | K32 (today) | K56 variant |
    |---|---|---|
    | upload `q` | 2^56 | 2^80 |
    | `q1` | ~2^48 (2 primes) | ~2^72 (3 primes) |
    | `P` | ~2^60 | ~2^84 |
    | reply | 36 bits | 60 bits |

  * Security of that variant: the upload at n = 4096, log q 80 drops to **148 / 135 bits core-SVP** (from 239).
    That meets 128, but it is a loss against v3, so it is excluded by "same or stronger". At n = 8192 it is
    369 bits. The reply key at n = 8192 and log `q1 P` 156 is 153 bits (from 251). It needs n = 16384 (383 bits),
    which doubles reply-key and key-switch work per value.
  * Cost of the K56 variant at equal security: roughly
    * provider 1.45x per element (10 planes instead of 7, 6 primes instead of 4);
    * client 1.5-2x (3 decryption primes, the larger rings);
    * bytes 1.5x (10 B up, 7.5 B down).

    That stacks on top of the online trips' 1.75x (K56 instead of K32). *Superseded by section 12:* the
    stacked top-digit plan gets K56 masks at 1.2-1.6x a K32 slot with no new parameters, and the direct set's
    cost there is ~1.45x.
* **Speculation rows.** Every row of a strict-speculation pass is its own slot with a fresh `r`.
  * Can one hint serve a batch of row masks without reusing randomness? It already does: a group is 256 slots
    under one upload secret, one reply key and the public hint. Rows of one pass are slots of a group, with
    nothing more to share.
  * Sharing `r` between rows leaks `x_i - x_j`. Deriving row masks as linear combinations of fewer base masks
    leaks the same combinations of the inputs.
  * The provider learns the row count per pass, which is public today.
  * The only lever is fewer wasted rows (better drafts): slots per token = rows per pass / tokens accepted per pass.
* **Prompt rows** dominate in chat (2.6 slots per generated token measured, including prompts). The prefix cache
  (`prefix_cache=True`) already removes cached prefixes' rows and masks.

## 6. Client ideas

### 6.1 Offline / online split (exists in effect; no new code)

* The factory already makes stock while idle.
* Decryption's `c1 t'` part does not depend on the reply, so it could be computed before the reply arrives.
  That would put only `c0 + s mod q1` and the rounding after the reply. It shortens the factory's critical path,
  not its total work.
* Storage bounds idle-time stock at 32B: 19.2 MB per slot, so 8 GB is ~420 slots, about 10 s of generation.
* An SSD stock wears the drive: ~1 GB/s of writes at 20 tok/s.

### 6.2 The client factory on the device's GPU (prototyped in exact torch, measured on a T4)

`pgw.hemask.pks_torch` implements the four kernels in exact int64 torch, each bit-identical to the numpy
reference (`tests/hemask/test_pks_torch.py`, CPU here and CUDA on Kaggle):

| kernel | what it is | reference |
|---|---|---|
| `pad` | three 31-bit-prime NTTs plus the centred Garner CRT done mod 2^56 with 28-bit halves (no reliance on wrap-around), tested at the range extremes | `pks.negacyclic_mul_mod2` |
| `encrypt_body` | the upload bodies | `pks.encrypt_rlwe` |
| `reply_key` | the reply keys | `pks.make_reply_key` |
| `decrypt` | the masks | `pks.decrypt_pks` |

The secrets come from the existing samplers, so the streams and request bytes are unchanged. Measured
ms per slot: section 9.

**Portable path** (design). The arithmetic is 30/31-bit modular multiplication. Metal has `mulhi` on 32-bit
integers, so Montgomery or Shoup 32-bit modmul is native. WGSL has no `mulhi`, so a 16-bit-split emulation costs
about 4x. Apple GPUs (M-series) and integrated GPUs then run the same NTTs:
* ~1.5e8 modmul per slot at 32B;
* ~60 MB of data touched per slot.

That is about 1 ms per slot on an M2-class GPU, ~5 % of the GPU at 40 slots/s.

**Sampling on the GPU** (design). The repo's device ChaCha20 (`pks_fused.ring_rows`) already reproduces the host
stream bit for bit. Porting CBD(21) (two popcounts per 64-bit word) and the ternary rejection sampler would put
the noise on the device too, with the same streams.

**Security:**
* the same secrets, computed on a device the user owns, with no new party involved;
* wipe the device buffers after use, as the CPU path does;
* the device GPU is inside the user's trust boundary, as the CPU is.

**Fidelity:** integer arithmetic only, bit-identical (tested).

### 6.3 Width-class `a` (design; sound; needs a hint rebuild and a wire flag)

* Today `a_C` is derived per linear. Deriving it per (input width, block) instead lets the client keep the
  public transforms `â_C` for all linears: (5120 + 27648) / 16 x 4096 x 3 x 8 B ≈ 200 MB at 32B, against 16.9 GB
  per linear-derived set.
* At one group per request, the client pays these forward NTTs per group. STATUS measures ~6 ns per input value,
  ~16 ms per slot at 32B (15 %); in the torch prototype, caching them removes 36-51 % of the encryption time.
* Security: `a` shared across layers with fresh `t` per (linear, group) is the same multi-instance use as `a`
  shared across groups (already the case), with no new assumption.
* Fidelity: deterministic.
* Cost of adoption: a new hint (the hint depends on `A`), a request flag so the provider derives the same `A`,
  and the native kernels' stream key. Not prototyped. With 6.2 it matters less (GPU memory holds the
  transforms).

### 6.4 Masks made on the device: no HE, no mask traffic (the bandwidth answer where the hardware allows)

* A device holding the int8 weights (7.6 GB at 7B, 32 GB at 32B) and a GPU can compute `W8 r mod 2^32` itself:
  four byte planes of `r` times `W8`, int32 sums. It already holds them for the hint build and the checks
  (`factory_weights="stream"` streams them).
* Cost: 1.28e11 int8 MAC per slot at 32B (2.8e10 at 7B).
  * On an RTX 4070-class GPU (~116 T int8 MAC/s dense) that is ~1.1 ms per slot of compute.
  * The weights stream per batch of slots: 32 GB per 256-slot pass from host RAM at ~25 GB/s, ~5 ms per slot.
    At 40 slots/s that is ~20 % of PCIe time.
* On GPUs without int8 matrix units (Apple) the exact route is fp16 matrix multiply with fp32 accumulation:
  * `|W8 x r_byte| <= 2^14`, so chunks of 1024 terms stay below 2^24 and are exact;
  * the chunks are summed in int32.

  That is ~18 ms per slot on an M3 Max-class GPU. Feasible only on the top parts, where pks-v3 on the GPU (6.2) is
  ~10-20x cheaper but bandwidth-bound.
* Security: nothing leaves the device, which is perfect privacy by construction. Integrity: no provider involved
  in the masks.
* Fidelity: exact integer identity.
* It costs the provider nothing: users per GPU are then bounded by the online trips alone.

### 6.5 Mask stock shared across a user's devices: not allowed

Copying or syncing stock between devices cannot guarantee single use: two devices may both consume a slot before
the sync, and a one-time pad used twice leaks `x_1 - x_2`. Each device keeps its own factory and its own stock.
A protocol with a server-side "used" ledger would move the reuse guarantee to a party the design does not trust.
Rejected.

## 7. Before / after per token (20 tok/s, 2.0 slots per token; ranges for 1.3-2.6 in the tables above)

| | 7B before | 7B after | 32B before | 32B after |
|---|---|---|---|---|
| provider H100 ms per token (consumer stock, G = 1) | 12.2 | 1.2-4.2 (S = 64, upload hidden) | 41.6 | 3.9-14.0 |
| users per H100 | 4.1 (11.4 at G = 4) | **12-48** (8-GPU node, common seed, coalesced, upload hidden) | 1.2 (3.4 at G = 4) | **3.4-13.5** |
| client core-seconds per token | 0.069 | 0.015 CPU + ~2-3 ms of a T4-class GPU (fused, estimated) | 0.227 | 0.045 CPU + ~8-12 ms of a T4-class GPU |
| client cores per user | 1.38 | 0.29 + 5-8 % of a T4-class GPU | 4.53 | 0.90 + 16-24 % of a T4-class GPU (~8 % of an RTX 4070) |
| mask bandwidth per user | 4.3 Gbit/s | unchanged (0 with 6.4) | 13.7 Gbit/s | unchanged (0 with 6.4) |
| online bandwidth per user | 3.0 Gbit/s | unchanged | 9.7 Gbit/s | unchanged |

## 8. Security and fidelity per idea

| idea | status | privacy | integrity | assumptions | fidelity |
|---|---|---|---|---|---|
| coalesced items (4.1) | prototype, off | unchanged: same inputs, public hint shared | p31 / self-check unchanged; refusals isolated | none new | bit-identical (tested) |
| common seed (4.2) | prototype, off | same standard RLWE; more instances per `a` | unchanged (`k` private per customer) | **Q8: multi-user preprocessing** | bit-identical |
| device hint cache (4.4) | design | public data | unchanged | as 4.2 | deterministic |
| tensor-core key switch (4.5) | design | unchanged | unchanged | none | exact integer identity |
| client GPU kernels (6.2) | prototype | same secrets, on the user's device | unchanged | none | bit-identical (tested) |
| width-class `a` (6.3) | design | same multi-instance use as across groups | unchanged | none | deterministic |
| masks made on the device (6.4) | design (exists for data-centre gateways) | perfect | no provider in the masks | none | exact |
| rank 4 / 1024-slot groups (4.7) | rejected for consumers | unchanged | unchanged | none | exact |
| shared reply key / secret across linears | rejected | passive only | - | - | - |
| stock across devices (6.5) | rejected | reuse risk | - | - | - |
| pks-v4, transposed hint, basis B (4.8) | stay rejected | - | unsound or lossy | - | - |

## 9. Measurements (Kaggle T4, `pgw-maskcost-t4`, two runs; `results/kaggle/maskcost-t4{,-r2}`)

Run setup: torch 2.11 + cu130, 4 vCPUs. Run 2 merges one-group parts into 512-column plane blocks and adds the
baselines and the fused decryption.

**Exactness on CUDA.** `pytest` with `PGW_TEST_CUDA=1`: 13 passed, 3 skipped (the three tests that need `pgw_pks`,
which the job does not build; they pass here on CPU). That covers:
* `test_pks_torch.py` on CPU and CUDA: NTTs, the pad at the range extremes, the upload bodies and reply keys
  against the reference streams, and the full round trip;
* `test_pks_coalesce.py`.

The bench's byte comparisons on CUDA:
* coalesced replies equal the items run one by one, at every shape;
* one 8-group item's bytes equal the concatenated coalesced replies;
* the fused decryption equals the eager one at every 7B / 32B shape.

**Coalescing, 8 one-group items of one linear, T4, seconds:**

| shape | one by one | coalesced | one item of 8 groups (one customer) | speed-up | model (T4) |
|---|---|---|---|---|---|
| 7B gate_up 37888 x 3584 | 2.96 | 1.35 | 1.08 | **2.19x** | 5.17x |
| 7B down 3584 x 18944 | 1.44 | 0.39 | 0.39 | **3.71x** | 5.44x |
| 32B gate_up 55296 x 5120 | 5.71 | 2.26 | 1.85 | **2.52x** | 5.29x |
| 32B down 5120 x 27648 | 2.88 | 0.79 | 0.78 | **3.64x** | 5.53x |
| 32B qkv 7168 x 5120 | 0.81 | 0.27 | 0.27 | **3.03x** | 5.12x |

Fitting `T = c + g S` to these:
* **The hint part `c` matches the model** (measured / model 0.74-1.15).
* **The per-group part `g` is 3.1-5.3x the model's**, mostly on the wide-output linears. The reply's device ->
  host copy, the host copies into the wire buffer and the `y` GEMMs at 256-512 columns are what a T4 does slowly;
  memplan already noted that the T4 model under-predicts device-wire items 1.6-1.9x.
* Coalescing is as fast as one customer's 8-group item, except at the widest output (gate_up: the 8 separate
  staged uploads).

So the H100 projection of section 4.3 is a range, from the model as is to `g` x 4:

| H100, slots/s | S = 1 | S = 16 | S = 64 | users per GPU at 2.0 slots/token (8-GPU node, upload hidden) |
|---|---|---|---|---|
| 7B, model | 166 | 1173 | 1684 | 47.9 (today 4.1) |
| 7B, `g` x 4 | 132 | 421 | 473 | 12.0 (today 3.3) |
| 32B, model | 48 | 350 | 508 | 13.5 (today 1.2) |
| 32B, `g` x 4 | 39 | 127 | 143 | 3.4 (today 1.0) |

The provider gain is therefore **3.5-11x**. The upper end needs the per-group path (reply copies, `y` GEMM shapes,
the key-switch products, 4.5) to be as cheap on an H100 as the model says. The RunPod stage (section 10) measures
it.

**Client kernels on the T4, ms per slot, whole model** (one tuned CPU core for reference: 34.5 / 113):

| part | 7B | 32B |
|---|---|---|
| upload bodies, eager torch (cached `a` transforms) | 21.8 (12.7) | 72.2 (41.7) |
| reply keys, eager | 7.5 | 17.0 |
| decryption, eager | 20.4 | 63.1 |
| **decryption, fused NTT kernel + eager tail** | **9.2** | **27.5** |
| noise proxy (CBD popcounts) | 2.0 | 6.5 |
| total eager (cached `a`) | 51.7 (42.6) | 158.9 (128.3) |

* Eager torch int64 on a T4 is slower than one tuned AVX-512 core. Every butterfly stage is several kernels with
  64-bit `remainder`, which is integer division, slow on GPUs.
* The provider's fused NTT kernel (Shoup / Montgomery, 32-bit) with one product per value cuts decryption 2.3x.
  The eager int64 tail (rounding, CRT, long division) was not timed separately and is the likely remainder.
* The provider's own fused kernels show what a fully fused client reaches. Their calibrated T4 rates are
  60-120 G modmul/s, against the client's ~1.5e8 modmul per slot at 32B. That puts **all client kernels fused at
  about 4-6 ms per slot on a T4** and about 2 ms on an RTX 4070-class GPU: 10-20 % / ~8 % of the GPU at 40
  slots/s.
* That is the estimate to use until the fused client kernels exist (pad: 31-bit primes need the kernels'
  `p < 2^30` bound lifted or 30-bit pad primes, `|a t| < 2^69` fits 3 x 30 bits; tail: one fused pass, like the
  provider's `_tail_kernel`).

## 10. Proposed RunPod stage (not launched)

`deploy/runpod/RUNBOOK.md`, session 7 (`maskcost`; the fusion study proposed a session 6): coalescing `S = 1, 4, 16, 64` on an H100 at 7B and 32B shapes
with two concurrent items (upload hidden or not), the group commit under N synthetic customers on the common
seed, and the client kernels on a consumer GPU (RTX 4090 community pod).

## 11. Files

* `src/pgw/service/pks_gpu.py`: `MultiUpload`, `split_reply`, and `item` taking several uploads.
* `src/pgw/service/pks_provider.py`: `run_items` and `_Coalescer` (`PGW_SERVICE_PKS_COALESCE_MS`,
  `PGW_SERVICE_PKS_COALESCE_GROUPS`).
* `src/pgw/hemask/pks.py`: `crs_a_seed`, `upload_seed` (`PGW_PKS_CRS_A`); used by `PKSHintStore.build`,
  `HintBuild` and provisioning.
* `src/pgw/hemask/pks_torch.py`: the client kernels in torch.
* `tests/service/test_pks_coalesce.py`, `tests/hemask/test_pks_torch.py`.
* `benchmarks/pks_client_gpu_bench.py`, `benchmarks/kaggle/build.py` job `pgw-maskcost-t4`.
* `research/reviews/mask_cost_numbers.py`: every number of sections 1-7 and 12 (`k56_fused2`).
* `src/pgw/factory/fused2_he.py`: `digit_plan`, `split_plan`, `recombine_plan`, `TopDigitProducts`
  (`PGW_FUSED2_HE_DIGITS=top`); `tests/factory/test_fused2_factory.py`.
* `src/pgw/hemask/pks56.py` (the direct K56 reference); `tests/hemask/test_pks56.py`.

## 12. K56 masks for fused trips (added 2026-11-27, after the fusion study)

Input: `research/reviews/consumer-round-trips.md`. fused2, which halves trips (o -> gate_up and down -> next qkv
in one trip, folded head), runs at ring K56C53 with p61 checks. Its masks are `W1 r_x mod 2^56` and
`W2' v mod 2^56`: two single-stage products with the int8 shapes of today's linears, so the hint does not grow.
Today they come from pks-v3 (K32) by digit decomposition (`pgw.factory.fused2_he`): 7 exact 8-bit digit
products per mask, i.e. **7x provider maskgen and ~7x client factory work per slot** against a K32 slot.

### 12.1 (b) Can part of the fused trip stay at K32? No, but the masks can mostly be K32 products

* Stage 1 returns `H = R + hmul W1 x`, which reaches 41.5 bits measured (52-53 a priori). Stage 2 returns
  `Z = W2' H`, up to 48.6 bits.
* The provider forms `H_hat` and multiplies it in the same trip, so stage 1 must be computed in stage 2's ring.
* Hiding needs `r_x` and `v` uniform mod `2^56`. Any K32 input mask, for example `x_hat = aq + r_x mod 2^32` lifted
  to `[0, 2^32)`, creates a data-dependent carry `2^32 W1 c` that only `W`-sized client work removes.
* A statistically hiding integer mask needs 12 + 40 + 7 + log n bits, more than 56.
* So no stage stays at K32 with exact results and perfect hiding.

**What can stay K32 is most of the mask product itself.** A digit of `r` at shift `s` contributes
`2^s (W d) mod 2^56`, so only `W d mod 2^(56 - s)` is needed.
* The top 32 bits (`s = 24`) need exactly `W d mod 2^32`, which is what pks-v3 returns for any `d`, with no
  exactness bound.
* Only the low 24 bits need exact products. Balanced digits (`|d| <= 2^(b-1)`) are exact while
  `128 x 2^(b-1) x n < 2^31`: `b = 12` at `n = d` (3584 / 5120) and `b = 10` at `n = I` (18944 / 27648).

**The top-digit plan** (`pgw.factory.fused2_he.digit_plan`, `TopDigitProducts`, `PGW_FUSED2_HE_DIGITS=top`,
default off): **3 products per mask for the `d`-input linears and 4 for `down`, instead of 7.** For the mlwe v2
factory at K40 it is 2 instead of 4.
* Exactness: `W r = sum_s 2^s W d_s` over the integers. Each low product is centred exactly
  (`|W d| < 2^31`, strict), and the top one is taken mod `2^32`. The sum is formed in wrapping uint64.
* Tested:
  * against `W r mod 2^56` at every bound-tight width (n = 5120 / 27648 / 29568, extreme weight rows, masks
    `2^56 - 1`, 0, and balanced carries into the top digit);
  * the planes recombine to `r`;
  * every plane is a valid K32 ring element;
  * on the real pks-v3 factory, the segments are bit-identical to plain generation (`tests/factory/test_fused2_factory.py`).
* Privacy is the digit factory's argument, unchanged. Each plane is an ordinary K32 mask row encrypted under fresh
  secrets; the scheme hides any message. The provider sees 3-4 instead of 7 times a K32 segment's rows, a public
  count.

**Stacked planes** (`TopDigitProducts(stack=True)`, the default of that class). The planes of a chunk go to the
backend as one call of `planes x t` rows, i.e. as more slots of the same linear. The provider then serves them in
one item. Its public hint (85 % of a one-group item) is computed once instead of once per plane, and short chunks
share groups. Tested: 1 request per linear and chunk instead of 7.

This combines with coalescing (section 4.1) and needs neither the common seed nor any new cryptography.

### 12.2 (a) A direct K56 parameter set: `pks-v3-K56` (reference implementation; new parameters, needs review)

`pgw.hemask.pks56` is pks-v3 with every modulus scaled so that one product is `W8 r mod 2^56`. Each noise term
keeps its v3 value relative to the budget, because the terms depend on the moduli only through `q1 / 2^q`,
`q1 / P` and `q1 / 2^yb`:

| | pks-v3 (K32) | pks-v3-K56 |
|---|---|---|
| upload modulus | 2^56 | **2^80** (message scale 2^24 in both) |
| q1 | 2 primes < 2^24 (2^47.98) | 3 primes < 2^24 (2^71.96) |
| P | 2 primes < 2^30 (2^60) | 3 primes < 2^28 (2^84.0) |
| key-switch RNS | 4 primes (2^108) | 6 primes (2^156); all in the fused kernels' range 2^23 < p < 2^30 |
| reply | 36 bits | 60 bits |
| pad primes (client) | 3 x 31 bits | 4 x 31 bits (`|a t| < 2^92`) |
| worst-case noise (all-127 rows, n = 29568) / budget | 2^13.35 / 2^14.98 | **2^13.33 / 2^14.96** |

Other properties:
* Rings, ranks, packing, the hint form, the hybrid key switch and the hint-integrity rule are pks-v3's.
* Tested (`tests/hemask/test_pks56.py`, exact Python integers, numpy NTTs):
  * full round trips at 256 and 77 slots decrypt to exactly `W8 r mod 2^56` (extreme rows, `r = 2^56 - 1`);
  * the measured error is inside the budget;
  * every noise term is within 0.05 bit of v3's relative to its budget;
  * the 1-bit refusal margin holds at the worst case.

**Security** (the repo's primal-uSVP core-SVP model; ternary secret, CBD(21); the external estimator still to run):

| instance | n | log2 q | core-SVP classical / quantum | HE-standard 128-bit limit (classical / quantum) |
|---|---|---|---|---|
| v3 upload | 4096 | 56 | 238.9 / 216.8 | 109 / 101 |
| v3 reply key | 8192 | 108.0 | 251.1 / 227.9 | 218 / 202 |
| **K56 upload** | 4096 | 80 | **148.3 / 134.6** | 109 / 101 |
| **K56 reply key** | 8192 | 156.0 | **153.0 / 138.9** | 218 / 202 |
| K56 upload at n = 8192 | 8192 | 80 | 368.8 / 334.7 | 218 / 202 |
| K56 reply key at n = 16384 | 16384 | 156.0 | 383.4 / 347.9 | 438 / 411 |

* **Same assumption** (decision-RLWE, ternary, CBD(21)), and both instances sit inside the HomomorphicEncryption.org
  128-bit tables with slack.
* **The concrete margin is lower than v3's**: 148 / 153 bits classical against 239 / 251, still above 128 classical
  and quantum.
* Under "same or stronger" it is therefore a reviewed-parameters decision, not a free change: **Q9 for the
  reviewer.**
* Keeping v3's margins needs n = 8192 for the upload (rank 32) and n = 16384 for the reply ring (64 rows per
  ciphertext). That doubles the hint GEMM and roughly triples the key-switch products per value: ~2.5-3x a K32
  slot instead of ~1.45x. It is listed as the conservative variant.

**Cost of the direct set against one K32 product** (operation counts, not benchmarked):
* provider ~1.45x: 10 instead of 7 byte planes in the `y` and hint GEMMs; 6 instead of 4 primes in the key
  switch's products and NTTs; 10 / 7 upload bytes; 60 / 36 reply bits;
* client ~1.4x: 4 / 3 pad primes, 6 / 4 reply-key primes, 3 / 2 decryption primes;
* bytes 1.43x up and 1.67x down.

Tensor cores do not change the ratio: the extra planes are int8 GEMMs and FP8 is not faster than int8 on
H100/B200.

### 12.3 (c) Cost per K56 slot and per token under fused2 (H100 model, upload hidden; `mask_cost_numbers.py`, `k56_fused2`)

Provider slots/s per H100 at 32B. Ranges are the model as is ... per-group part x 4 (T4-measured). Users per H100
are at 20 tok/s and 2 slots per token.

| option | K32 slot today (reference) | digit x7 (today's K56) | top-digit, separate | **top-digit, stacked** | digit x7, stacked | **direct K56** |
|---|---|---|---|---|---|---|
| consumer stock, per-customer seed (S = 1) | 48 / 39 | 6.9 / 5.6 | 15 / 12 | **41 / 25** | 33 / 15 | 33 / 27 |
| users per H100 | 1.2 / 1.0 | 0.17 / 0.14 | 0.38 / 0.31 | **1.0 / 0.6** | 0.8 / 0.4 | 0.8 / 0.7 |
| common seed, coalesced (S = 64) | 508 / 143 | 73 / 20 | 162 / 46 | 180 / 47 | 83 / 21 | **350 / 99** |
| users per H100 | 12.7 / 3.6 | 1.8 / 0.5 | 4.0 / 1.1 | 4.5 / 1.2 | 2.1 / 0.5 | **8.8 / 2.5** |

At 7B (S = 1 / S = 64, model):

| | K32 | digit x7 | top-digit, stacked | direct |
|---|---|---|---|---|
| slots/s | 166 / 1684 | 24 / 241 | 141 / 598 | 114 / 1162 |
| users per H100 | 4.2 / 42 | 0.6 / 6.0 | 3.5 / 15.0 | 2.9 / 29.0 |

* **Without new cryptography:** the stacked top-digit plan makes a K56 slot cost about 1.2-1.6x a K32 slot at
  consumer stock, against 7x today. This is the immediate fix: 6x more fused2 users per GPU.
  * The 7-digit factory stacked would also drop to 1.5-2.6x. Its 7 planes multiply the per-group part, which is
    why the top-digit plan matters once the hint is shared.
* **With the common seed (S = 64):**
  * the hint is shared anyway, and the per-group part times the plane count dominates;
  * the direct K56 set is then the cheapest, ~1.45x a K32 slot and **~2x the stacked top-digit plan**;
  * that is the case for reviewing it (Q9).
* **Client per K56 slot** (one core, before GPU offload):

  | model | K32 | digit x7 | top-digit (3.41 K32 slots of work) | direct |
  |---|---|---|---|---|
  | 32B | 113 ms | 793 | 387 | ~159 |
  | 7B | 34.5 ms | 242 | 118 | ~48 |

  With the GPU offload of section 6.2, scale the GPU share by the same factors.
* **Per token, fused2 against today**, same slots per token (rows):
  * the fusion halves the trips (2L instead of 4L + 1);
  * the online bytes grow 1.75x (K56 words);
  * the masks stored on the client grow 1.75x (7 B per value);
  * the mask stream grows 1.43x up and 1.67x down with the direct set, or 3.4x with the top-digit plan (each plane
    is a K32 upload and reply).
  * At 32B and 20 tok/s with 2 slots per token: online 17 Gbit/s; masks 21 Gbit/s (direct) or 47 Gbit/s
    (top-digit) per user.
  * So for a consumer on the internet, fused2 trades RTT for bandwidth that is already the binding constraint
    (section 3). Its masks are cheap only in GPU terms.
* **Recommendation:**
  1. Ship the stacked top-digit plan as the fused2 K56 factory once it passes a GPU end-to-end run: no new
     assumption, bit-identical, ~6x the current K56 throughput.
  2. Put `pks-v3-K56` (n 4096 / 8192, 148 / 153-bit core-SVP) to the external review as Q9, with the conservative
     n 8192 / 16384 variant as the fallback. It doubles fused2 users per GPU over the top-digit plan only when the
     common seed (Q8) is also adopted.
