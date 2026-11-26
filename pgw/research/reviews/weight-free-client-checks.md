# Weight-free client checks: can the client drop `W8` without weakening anything?

Date: 2026-11-26. Scope: the SDK client's factory under the default `mask_scheme="pks-v3"`, `check_field = "auto"`
(p31 online check), with the masked head. Inputs: `pks-v3-client-disk.md` (open option 3, the "transposed pks-v3
hint"), `pks-v3-review-packet.md`, `pks-v3-adversarial-review.md` (F1-F17), `pks-v3-provider-hint.md`,
`pks-v4-prime-upload.md` (the bar), SPEC 3.2, 5.1, 7.3-7.6, 7.9, `factory/generator.py`, `hemask/pks*.py`.
Numbers: `research/reviews/weight_free_numbers.py` (run `PYTHONPATH=src python research/reviews/weight_free_numbers.py
--measure`; same estimator and noise model as the v3 packet and `pks_v4_numbers.py`; Monte Carlo checks of every
closed form at small moduli; two attack demonstrations). Status: **analysis only, nothing implemented.**

## 0. Verdict

No candidate removes `W8` from the client while keeping every property and every cost of today. The reason is
structural (section 2): the online check needs `u W8` for a `u` the provider cannot predict, and something trusted
must tie that vector to the true `W8`. There are only three possible anchors: `W8` itself, a secret derived from
`W8` that lives for a long time, or a public commitment with a proof per use. The first keeps `W8` (on disk or in
transit). The second weakens forward security. The third is a new proof system.

| candidate | online check soundness, per check | factory self-check | MLWE / noise | what it costs | new trust or assumption | verdict |
|---|---|---|---|---|---|---|
| **A0** transposed pks-v3 product, decrypted `u W8` used as is | **broken**: an additive shift of the decrypted `u W8` makes the verdict a function of the private activation (error 1 for a 1-bit leak per breaker trip, demonstrated) | same problem for `v W8` | upload instance unchanged (238.9 bits), but more ring samples than the packet states (up to 9504 for the head) and the head's worst-case column norm is refused (2^14.19 > 2^13.98) | transposed hint 1.28 / 4.24 / 6.67 GB; +0.4 % traffic (amortised) | "the provider can only add constants" (non-standard), and still unsound | **not viable** |
| **A1** A0 + secret sketch `S = W8 X` to verify `u W8` | 2^-31 (unchanged) + 2^-62 for a wrong `u W8` (d = 2) | 2^-42 (unchanged) with an integer anchor | as A0 | as A0 + 6-28 MB per sketch column, plus a full pks-v3 path on `W8^T` | long-lived secret `X`; packet's sample-count statement amended; head needs measured norms (0.5B: 2^13.81, 0.17-bit margin) | **viable with a stated loss**; dominated by B |
| **A2** A0 + public commitment and an opening proof | unchanged (plus the PCS soundness error) | unchanged | as A0 | as A0 + a polynomial-commitment system: ~23 / 50 / 74 MB per opening, provider `O(N)` per opening, client re-encodes all of `W8` once | binding of a hash commitment (SHA-256, already relied on for the pin) and the code's proximity bounds; a new protocol | **not viable now** (no security loss in principle; a research project and an external review) |
| **B** secret long-lived basis: `u = B c`, `v = B_v c` with fresh `c`; store `G = B^T W8`, `V = B_v^T W8` | 2^-31 + 2^-155 (k = 5): **unchanged** while `B` is secret | 2^-42 (k_v = 28): **unchanged** | untouched (no HE change) | **0.11 / 0.36 / 0.57 GB** instead of 6.5 / 31 / 70 GB; factory CPU on `W8` per segment 14 / 64 / 143 G MAC -> 12 / 38 / 57 M MAC | one long-lived secret: a snapshot of `B` (or of `G`, or of `k` past segments' `u`) lets the provider pass every later check (demonstrated: 2000/2000); today a snapshot exposes only the segments in stock | **viable with a stated loss** (forward security of integrity) |
| **C** keep no `W8` at rest; stream it from any untrusted source, verified against the pin, once per `R` segments (stock of pre-drawn sets in between) | unchanged | unchanged | untouched | `R = 64`: +0.8 / 1.1 / 1.7 % network per segment, stock 0.6 / 1.9 / 3.1 GB on disk; `R = 1`: +49 / 73 / 108 % network, +23 / 33 / 48 % client CPU (SHA-256) | none (the stock argument of `pks-v3-client-disk.md` section 3) | **viable, no security loss, stated operational cost** (recurring 6.5-70 GB downloads, a runtime dependency on a `W8` source) |

None passes the gate "no loss", so **no prototype was written**. Recommendation in section 8.

## 1. What needs `W8`, per segment (the inputs to every candidate)

Verified against `generator.generate()` and SPEC 5.1 / 7.3:

1. **`uW = u W8 mod p`, p31, `c = 1` vector per linear, fresh per segment.** `u` comes from the segment seed
   (`os.urandom(32)`, stream `"u-p31"`); the gateway regenerates `u` and checks `u . yq == uW . xq (mod p)` on every
   row. This is the online integrity check, and since F4 it also catches wrong masks. 3.3 / 11.0 / 17.4 MB per
   segment. Computing it is two exact passes over all of `W8` (16-bit limbs): 14 / 64 / 143 G MAC per segment,
   head included.
2. **No mask-side term.** `W8 r` is computed by the provider under pks-v3 and decrypted with the hint;
   `PKSMaskBackend.matmul` never reads `W8`. The hint was built from `W8` once, at setup.
3. **Private `v W8`, 14 uniform vectors over `Z_2^32` per linear** (40-bit target against chosen errors, F1).
   Kept in RAM, redrawn at worker start and after every aborted segment. 94 / 309 / 486 MB per set (u64).
4. Non-per-segment uses: hint (re)build, `ModelPack.open(verify="full")` at factory start, and the head `W8`
   derived from the plain head (0.54 / 0.78 / 1.25 GB, not stored: the plain head stays on disk for the gateway).

So the question is exactly: **can the client obtain `u W8 mod p` (for fresh, provider-unpredictable `u`) and
`v W8 mod 2^32` without `W8`, with today's soundness and secrecy?**

## 2. Why an anchor is unavoidable

The gateway accepts `yq` iff `u . yq == z . xq` with `z` claimed to equal `u W8`. Suppose the client obtains
`(u, z)` from a state `sigma` it keeps after setup.

* **`sigma` public (or computable by the provider) and `z` computed locally.** Local computation of `u W8` from
  `sigma` works only for `u` in the subspace `sigma` determines (for a linear sketch `sigma = Y^T W8`, `u` must be in
  `col(Y)`). The provider knows that subspace and picks `e` orthogonal to it: soundness error 1.
* **`z` delegated to the provider** (A0: it computes `Enc(u) W8` under pks-v3). The decrypted `z` must then be
  checked against something that encodes `W8`: either `W8`, a secret sketch, or a public commitment with a proof.
  Without a check the provider controls `z - u W8` (F1: pks-v3 replies are additively malleable to the exact
  entry), and section 3.2 shows this is enough to leak private data.
* **`sigma` secret.** Works (candidate B, and A1's sketch), but `sigma` is then a secret that lives as long as the
  client does not see `W8` again.

The three anchors are `W8` (C), a long-lived secret (A1, B) or a public commitment plus proofs (A2). Only C and
A2 avoid a long-lived secret.

## 3. Candidate A: the transposed pks-v3 product ("transposed hint")

### 3.1 Construction

Per linear, run pks-v3 on `W8^T` (`n x m`): the client uploads an encryption of `M` with slots carrying `u`'s
limbs (and, when needed, the private `v`), the provider computes `W8^T b`, and the client decrypts with a second
hint built from `W8^T` (1.54 KB per input feature of `W8`: 1.28 / 4.24 / 6.67 GB, built at setup from `W8` or with
the 2-adic provider check of `pks-v3-provider-hint.md`).

* **Limbs.** pks-v3 decrypts `mod 2^32`; `u W8` over the integers has 53 bits. With centred limbs of `b` bits,
  `|W8^T u_l| <= 127 m 2^(b-1) < 2^31` needs `b <= 9` for layers and `b <= 7` for the head (m = 152064): **5 limbs**
  for a 31-bit `u`. `v W8 mod 2^32` needs no limbs (it is exactly what pks-v3 computes).
* **Amortisation.** One 256-slot group carries 51 segments' worth of `u` (a stock of pre-drawn `u`, security-neutral
  as in `pks-v3-client-disk.md` section 3). Per group: upload `sum m x 1792 B` plus one reply key per linear,
  2.96 / 9.07 / 13.14 GB; amortised 58 / 178 / 258 MB per segment, **+0.4 %** of pks-v3's per-segment traffic
  (13.2 / 42.9 / 64.9 GB). Client CPU: one extra encryption of `sum m` inputs per 51 segments (~0.15 s at 7B),
  negligible against today's 14 G MAC `prime_precompute`.
* **Lattice parameters.** Same instances: upload core-SVP 238.9 / 216.8, reply key 251.1 / 227.9 (recomputed).
  Two changes for the reviewer:
  * The upload now has `m/16` ring samples per (linear, group): 2368 / 3456 / 3696 for the widest layer and **9504
    for the head**, while the packet states `<= 1848`. The primal attack uses fewer than one ring sample (optimum
    m ~ 3809 LWE samples), so the estimate does not move, but the stated assumption changes.
  * The noise bound is in terms of the row norm of the matrix being applied, here a **column** of `W8` (length `m`).
    All-127 columns pass for every layer (2^13.47-2^13.68 < 2^13.98) but **not for the head** (127 sqrt(152064) =
    49524 > 40931, 2^14.19). Measured on Qwen2.5-0.5B: max column L2 35034 for the head (vs 1480 for its rows), noise
    2^13.81: accepted, with 0.17 bit of the 1-bit margin left. Larger heads are not measured.

### 3.2 A0, no anchor: not viable

The decrypted `z` differs from `u W8` by whatever the provider adds (F1; for pks-v3 an exact shift of one entry
costs it nothing). Take `delta = e_j` on `z` and an honest product `yq = W8 xq`. The gateway accepts iff
`u . W8 xq == (u W8 + delta) . xq`, i.e. iff `xq_j == 0` (`|xq| < p`). The verdict, and so the circuit breaker, is
a function of a private activation: the leak F4/R2 removed comes back, without any error on the output. The script
demonstrates it (`xq_5 = 0: accept, 3: REJECT, -7: REJECT, ...`).

* Proving anything better would need an assumption like "a provider can change a decryption only by a constant
  independent of the plaintext". That is not implied by RLWE, and the shift above shows that even granting it is not
  enough.
* Scaling attacks (doubling every reply) do not work against pks-v3, because the provider does not know `c1 t'`.
  Weight substitutions `W' = W8 + D` with `D A == 0 (mod 2^56)` need short SIS solutions (root Hermite factor
  ~1.0002 for the noise budget): infeasible. These are informal observations, not an argument.

### 3.3 A1, secret sketch: viable, with B's loss and more cost

Store `S = W8 X` (`X` uniform, secret, `n x d`, kept with the KEK) and accept a decrypted `z` only if
`z X == u S (mod p)`. Wrong `z` passes with probability `(1/p)^d`: 2^-62 at `d = 2`, 6.2 / 19.2 / 28.1 MB per column.
For `v` (mod `2^32`), anchor the exact integer product instead (`|v W8| < 2^61`, checked modulo a 61-bit prime
after recombining limbs and a range check), to avoid the 2-adic weakness of `Z_2^32` sketches (F1).

* **Soundness:** unchanged while `X` is secret (fresh full-dimensional `u` per segment, and `z` verified).
* **Loss:** `X` is a long-lived secret. If it leaks, the provider can again shift `z` by `delta` with `delta X = 0`.
  That gives the data-dependent verdict of 3.2 (one bit per breaker trip), though not silent corruption (the
  provider still does not know `u`).
* **Cost:** the whole transposed path (second hint 1.28-6.67 GB, a wire mode, provider support, a second noise
  refusal and its review) for the same class of loss as B, which needs none of it.

### 3.4 A2, public commitment: no long-lived secret, but a research project

Commit to `W8` (a hash-based polynomial commitment, e.g. Ligero / Brakedown). The client checks the commitment once
at setup, when it has `W8`. Per use:

1. The client picks a random `rho` and asks for `w = W8 eq(rho)` (public data).
2. The provider proves `eq(rho')^T w = W~(rho', rho)` by one opening.
3. The client checks `z~(rho) == u . w`.

The provider committed to `z` (in its pks-v3 reply) before `rho` was drawn, so a wrong `z` survives with probability
at most `log2(n) / |F|`. Use an extension field, or repeat.

* **Assumptions:** SHA-256 binding (already relied on for the pin) and the code's proven proximity bounds. The
  protocol is interactive, so no random oracle is needed.
* **Cost (rough):** about 23 / 50 / 74 MB per opening (t = 300 queries), `O(N)` provider work per opening, a
  re-encode and hash of all `W8` at setup, plus all of A's costs.
* **Amortisation:** openings can be rare if they only rotate a basis (B with rotation every `R` segments). That is
  the only design here that removes `W8` with no security loss.
* **Verdict:** it is a new subsystem with its own soundness analysis and an external review. Not viable as a change
  now.

## 4. Candidate B: a secret long-lived basis

### 4.1 Construction

At setup (with `W8` present), the client draws a secret `seed_B` and, per linear, `B` (`m x k` over `F_p`) and
`B_v` (`m x k_v` over `Z_2^32`). It stores `G = B^T W8 mod p` and `V = B_v^T W8 mod 2^32` encrypted under the KEK
(`G` and `W8` determine a kernel vector, see 4.3, so `G` is as secret as `B`). Then it deletes `W8`. Per segment:

* `c` fresh from the segment seed, `u = B c`, `uW = c^T G`. Exact: `(B c)^T W8 = c^T (B^T W8)`.
* The gateway derives `u = B c` (it needs `seed_B`), or the segment stores `u` instead of a seed.
* Fresh `c_1..c_14` give the private vectors `v_i = B_v c_i` and `v_i W8 = c_i^T V`. The redraw-on-abort rule
  redraws the `c_i` and costs nothing.

### 4.2 Soundness against an adaptive provider (closed forms; Monte Carlo agrees at p = 31 and 2^8)

* **p31 check, error `e != 0` independent of `(B, c)`:** `B^T e` is uniform in `F_p^k`, so
  `P[accept] <= (1/p)^k + (1/p)^c` (with the `1 + 2^-33` sampling slack).
  * k = 1: 2^-30, one bit lost.
  * k >= 2: 2^-31.00, which equals today.
  * The same error re-tried in every later segment has a floor of `p^-k` instead of today's `p^-T`. That floor is
    2^-155 at k = 5.
  * Adaptivity: a rejection tells the provider only that `c . B^T e != 0`. It changes the posterior of
    `B^T e' = 0` by a factor `1/(1 - Q 2^-31)` after `Q` rejections. With F3's halt after 3 failures and the
    gateway's breaker, `Q` is tiny.
* **Private check over `Z_2^32`, error of valuation `t <= 29`:** `w = B_v^T delta` has `k_v` coordinates, each
  uniform on `2^t Z`. Hence `P = sum_s P[val(w) = s] 2^-14(32-s) + 2^-(32-t) k_v`. Results:

  | `k_v` | bits |
  |---|---|
  | 14 | 2^-40.00 (2 bits worse than today's 2^-42, still at the 40-bit target) |
  | 20 | 2^-41.98 |
  | 28 | 2^-42.00 (equal) |

* **Secrecy of `u`:** the provider sees exactly today's bytes. `B`, `c`, `G` and `V` never leave the client.
* **Lattice parameters and noise:** untouched (no HE change at all).

### 4.3 The loss: one long-lived secret instead of per-segment secrets

Small-subspace attacks need knowledge of `col(B)`. Ways an attacker gets it:

* **Leaked `u` of `k` segments** (the store at rest under the KEK, a backup, a partner export, gateway or factory
  RAM). They span `col(B)`. A kernel vector `e` then passes every later check (demonstrated: 2000/2000).
* **`G` plus the public `W8`.** For `m <= n`, `B^T = G W8^+`. For `m > n`, `e = W8 x'` with `x' in ker G`.
* **`seed_B`.**

Today the same snapshot exposes only the `u` of the segments in stock. Segments drawn after the KEK is rotated are
safe again, because the factory still has `W8`. Under B, a one-time, undetected snapshot breaks integrity for
**every future segment** until `B` is redrawn, and that requires `W8` again (a 6.5-70 GB download). With the
integrity break comes the data-dependent breaker behaviour, which is a privacy channel (F1, F4). This is a loss of
forward security: small and clearly stated, but a weakening, so B fails the gate.

* **Mitigation:** redraw `B` on a schedule by re-streaming `W8`, which is C with a different stock. The exposure
  window then becomes the rotation period.

### 4.4 Costs

* **Disk:** `G` (k = 5) + `V` (k_v = 28), uint32: 0.11 / 0.36 / 0.57 GB instead of `W8` at 6.5 / 31.2 / 70.2 GB.
* **Factory CPU on weights per segment:** 14 / 64 / 143 G MAC drop to 12 / 38 / 57 M MAC. The gateway adds `k m` per
  linear to derive `u`.
* **Download:** unchanged (`W8` once at setup, then deleted).
* **Spec:** SPEC 7.3's normative derivation of `u` changes, so the segment format changes and needs external review.

## 5. Candidate C: no `W8` at rest, streamed and verified when needed

Keep `W8` on any untrusted store: the provider, a mirror, or a NAS in the customer network. Stream it through
SHA-256 against the pinned manifest, and compute everything that needs it during that pass:

* `uW` for the next `R` segments (an `R`-column GEMM, more efficient than today's per-segment GEMVs);
* a few private sets;
* the head `W8` from the plain head, as today.

A derived value is used only after its tensor's hash matched, so a wrong stream is a DoS and never a wrong check
vector. Between passes, the factory consumes the pre-drawn stock. `pks-v3-client-disk.md` section 3 showed this is
as secret and as sound as drawing on demand: same seeds, same derivation, KEK envelope, durable single use.

| | 7B | 32B | 72B |
|---|---|---|---|
| `W8` per pass | 6.53 GB | 31.2 GB | 70.2 GB |
| `R = 1` (a pass per segment): network vs pks-v3's per-segment traffic | +49 % | +73 % | +108 % |
| `R = 1`: SHA-256 CPU (1.34 GB/s per core measured) vs pks-v3 client CPU | +23 % | +33 % | +48 % |
| `R = 64`: network | +0.77 % | +1.14 % | +1.69 % |
| `R = 64`: stock on disk (64 `uW` sets + 8 private sets, u32) | 0.59 GB | 1.94 GB | 3.06 GB |

* **Security:** no change. Soundness, secrecy, parameters and the provider's view are as today. The provider may
  see when a pass happens, which follows factory activity it already sees through maskgen.
* **Cost:** recurring downloads and a runtime dependency on a `W8` source (the factory stalls if the stock runs out
  and no source is reachable). Private sets are consumed by every aborted segment, so a flaky link burns them
  faster. Private sets at rest are new secret material, under the same KEK envelope as the segments' seeds and `uW`.
* **Verdict:** this is lever 1 of the client-disk note. Counted against pks-v3's own traffic, it is cheap at
  `R = 64` (+1-2 %), which that note did not show. It is not lossless (recurring 6.5-70 GB downloads,
  availability), it lives in the factory (another agent's area), and it is a product decision. It is not
  prototyped here.

## 6. Other ideas considered

* **Redundant queries with a hidden scalar** (send `x_hat` and `lambda x_hat + r2`, check `y2 - W8 r2 == lambda y`).
  This uses masks instead of `uW`, but the masks need the factory check, which needs `v W8`, so it is circular. It
  also doubles provider work and masks per row and gives about 3 bits per repetition over `Z_2^32`. Not viable.
* **Sparse `u` with privately retrieved rows** (PIR of the rows of `W8` in `u`'s support). A single-coordinate
  error escapes unless its coordinate is in the support: soundness `|S|/m`. Not viable.
* **A larger basis** (k in the thousands) only defends against leaked `u`, not against leaked `G`, and costs as
  much as `W8`.

## 7. Against the pks-v4 bar

pks-v4 failed on correctness (noise 2^20.9 vs budget 2^15), security bits (22-83 bits lost) and forgeability
(soundness error 1). The comparison:

| | correctness | security bits | forgeable check |
|---|---|---|---|
| A0 | layers pass; head worst case refused | unchanged | **yes** (leak, demonstrated) |
| A1 | layers pass; head worst case refused | unchanged | no, while `X` is secret |
| B | unchanged | unchanged | no, while `B` is secret (2^-31 / 2^-42 as today); yes after a one-time leak, forever |
| C | unchanged | unchanged | no change |

Unlike pks-v4, B and C fail nothing numerically. B fails on a property the numbers do not show (forward security),
and C on cost.

## 8. Recommendation

1. **Keep `W8` on the client by default.** Nothing here changes the default.
2. **If a customer's disk forces it, the route with no security change is C** with a stock: about 64 segments, a
   `W8` pass per refill from any source verified against the pin, +1-2 % traffic, 0.6-3.1 GB of disk. It belongs in
   the factory (the generator's `u` / `v` source and a verifying weight streamer). It needs no review beyond the
   stock argument already written down.
3. **B is the cheapest design** (0.1-0.6 GB, less factory CPU), but it trades per-segment secrets for one long-lived
   secret. Offer it only as an explicit, documented opt-in that states the forward-security loss, with rotation
   (a `W8` re-stream) on a schedule and on any suspected compromise. It changes SPEC 7.3 and the segment format and
   needs external review.
4. **Do not pursue the transposed hint** (A0 / A1). It is unsound without an anchor, and with a secret anchor it
   is dominated by B. A2 is the only no-loss route without `W8` and is a research item.
5. For the reviewer, if A is ever revived: the `m/16` sample count (up to 9504) and column-norm noise refusal for
   the head (0.17-bit margin at 0.5B) are the two new items for the pks-v3 parameter statement.
