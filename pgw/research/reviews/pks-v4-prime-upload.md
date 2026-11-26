# pks-v4 (prime upload modulus, provider-built checked hint): analysis and decision

Date: 2026-11-26. Input: the sketch in `pks-v3-provider-hint.md`, section 8, second bullet. Numbers:
`research/reviews/pks_v4_numbers.py` (same estimator and noise model as the v3 packet: `core_svp_beta`,
`PKSParams.noise_terms`). Status: **analysis only, not implemented.** The gate for implementing was "no loss in
security or correctness versus pks-v3"; every variant below fails it. Sections 1-6 cover what the task asked for. The
reasons are in sections 2 (correctness), 3 (security) and 4 (soundness of the check).

## 0. Verdict

| variant | correctness, worst case (v3: 2^13.35, budget 2^14.98) | upload core-SVP (v3: 238.9) | reply-key core-SVP (v3: 251.1) | cheap check sound? |
|---|---|---|---|---|
| V0: upload mod `q1` (the sketch) | **2^20.91** (fails; Qwen2.5-0.5B already 2^16.98) | 290.8 | 251.1 | **no** (soundness error 1, section 4.2) |
| V1a: upload mod `q1 * 65537`, same reply key | 2^13.25 | 238.9 | **209.1** | no (needs V3's change) |
| V1b: same, reply-key `q/sigma` kept | **2^20.47** (fails) | 238.9 | 251.1 | no |
| V1c: same, error split for equal correctness | 2^13.35 | 238.9 | **221.3** | no |
| V2: upload mod `q1 * 2^8` | 2^14.05 / **2^14.47** (above the 2^13.98 refusal threshold) | 238.9 | 228.6 / 251.1 | no |
| V3: V1 plus key switch on the unreduced integer `W8 A` | 2^13.25 | 238.9 | **167.9-173.7** | yes, error <= 2^-45 |

No variant both keeps v3's correctness margin and security bits and has a sound check costing about 5 GB.
The sketch's two premises do not hold:

1. **"Upload modulus = q1" breaks correctness.** In v3, switching `2^56 -> q1` divides the upload noise `W8 e`
   by `2^8.02`. Without that step the noise is 6-7 bits over budget, even at real row norms. Keeping v3's
   `sigma / q` at modulus `q1` would need `sigma = 0.0125`, which is not LWE.
   * The repair is an upload modulus `Q = q1 f` with `f >= 2^8`. But the key-switching modulus then becomes
     `Q P`, not `q1 P`.
   * This trade-off is invariant: (key-switch noise) x (reply-key `q / sigma`) grows by `f` compared with v3,
     whatever `P` and `sigma_ks` are. Either the reply key loses 22-42 bits of core-SVP, or the noise leaves
     the budget.
2. **"C is linear in W8" holds only modulo `q1`.** The hybrid key switch lifts `A~ = centred(H mod Q)` to the
   special primes of `P` (basis extension). That lift is not linear in `W8`.
   * The witness `[C]_P` can be checked only through `C mod q1`. Because `P > q1`, every `c1'` has a matching
     in-range witness, so the sketched check accepts any hint (section 4.2, verified numerically).
   * Making the check sound requires one of two things:
     * per-entry lift quotients plus a range argument for each entry of `H`. That is 17.3 GB of quotients at 7B,
       and the range argument is the open problem of v3, section 2;
     * a key switch on the unreduced `W8 A` (V3), which costs 77-83 bits of reply-key security.
   * Every variant also has to give the provider `ks_seed`. v3 keeps it private (F13; section 5).

So pks-v4 is **not implemented**, as the gate requires. The SDK default stays pks-v3 with `hint="local"`. The
v3 provider-`H` mode (44 GB, verified) stays as it is.

## 1. Scheme delta from v3 (the sketch, V0)

| step | pks-v3 | pks-v4 sketch |
|---|---|---|
| upload | `B_C = a_C t + e_C + 2^24 M_C mod 2^56`, CBD(21) | `B_C = a_C t + e_C + round(q1 M_C / 2^32) mod q1` (`q1` is odd, so the encoding has a rounding) |
| provider body | `round(q1 y / 2^56)` with `y = W8 b mod 2^56` | `y = W8 b mod q1`, no switch |
| hint operand | `hs = round(q1 H / 2^56)`, `H = W8 A mod 2^56` | `hs = W8 A mod q1` (linear) |
| key switch | `C = sum_l A~_l k_l mod q1 P`, with `A~ = centred(hs)` lifted to `P`'s primes | same |
| hint | client: `c1 = round(C / P) mod q1` from `hs` it computes or checks | provider: `c1` and the witness `w = [C]_P`; client checks with prime-field Freivalds |
| who knows `k` (`ks_seed`) | client only (F13) | **provider too** (it computes `C`) |
| decryption | `v = c0 + c1 t'`, `round(2^32 v / q1)` | unchanged |

The variants (section 0) change only the upload modulus (`Q = q1 f`), the ModDown divisor (`P f`, so that `c1`
stays mod `q1`), and, in V3, the key-switch operand (the integer `W8 A_centred` instead of its residue mod `Q`).

## 2. Correctness (noise budget, honest worst case)

Budget `q1 / 2^33 = 2^14.98`. The factory refuses a linear above `2^13.98` (1-bit margin). Units are `1 mod q1`.
The tails are v3's: 24 sigma for Gaussian-like terms, Hoeffding at `t = 24`.

v3 terms at the worst case (all-127 rows, `n = 29568`):

| term | log2 |
|---|---|
| `W8 e` | 12.68 |
| `H s` switch | 9.58 |
| ModDown `c1 t'` | 10.08 |
| key switch | 4.46 |
| reply rounding | 10.98 |
| **total** | **13.35** |

**V0.** Upload at `q1`. Without the `q1 / 2^56` factor, `W8 e` is `24 * 3.24 * ||w||` at full scale. The new
encoding rounding adds `W8 eps` (`|eps| <= 1/2`), and the `H s` switch term disappears.

| row | `W8 e` | total | v3 total |
|---|---|---|---|
| worst case (72B) | 2^20.70 | **2^20.91** | 2^13.35 |
| Qwen2.5-7B `down_proj`, real norms (0.37 x max) | 2^18.94 | **2^19.15** | 2^12.51 |
| Qwen2.5-0.5B q/k/v/o, real norms | 2^16.74 | **2^16.98** | 2^12.08 |

* CBD(1), the smallest noise the sampler offers, still gives 2^19.28 / 2^17.54 / 2^15.42.
* Every model is refused, and v3's margin of 1.6 bits becomes a deficit of 2-6 bits.
* The ratio is fixed: decryption noise relative to `q1` is about `24 ||w|| sigma / Q_upload`. v3 sits at
  `sigma / Q = 3.24 / 2^56`. At `Q = q1` that needs `sigma = 0.0125`.

**V1/V2.** Upload modulus `Q = q1 f`, `sigma_up = 3.24 Q / 2^56` (v3's `sigma / q`; CBD with eta about 1.3e6 for
`f = 65537`, so a different sampler), ModDown by `P f`.

* `W8 e / f` equals v3's term.
* The `H s` term is gone.
* The encoding term is at most `||w||_1 / (2 f)`: 2^4.84 for `f = 65537`, 2^12.84 for `f = 2^8` (2^9.3 by Hoeffding).
* The key-switch term is `(Q / 2) sigma_ks / (P f) = (q1 / 2) sigma_ks / P`, as in v3, **but the key switch now runs
  modulo `Q P = q1 f P`**.

Results:

* V1a: 2^13.25. That is 0.1 bit better than v3, at the cost in section 3.
* V2: 2^14.05 with the worst-case encoding bound, 2^13.39 with Hoeffding. Both are worse than v3.
* Keeping the reply key's `q / sigma` (multiply `sigma_ks` by `f`) costs a factor `f` in the key-switch term:
  2^20.46 for `f = 65537` (fails), 2^12.46 for `f = 2^8`. The total is then 2^14.47, which is above the refusal
  threshold.

**The invariant.** The key-switch noise is `~ q1 sigma_ks / P` and the reply-key ratio is `q1 f P / sigma_ks`. Their
product is `q1^2 f`, which is `f` times v3's, whatever `P` and `sigma_ks` are. With `f >= 2^8`, and in fact
`f >= 65537` for a prime that is NTT-friendly at degree 8192, no choice of parameters matches v3 on both counts.
v3 avoids the factor because it rounds `H` to `q1` before the key switch, and that rounding is exactly the step
the sketch wants to remove.

## 3. Security at the new moduli (same method as the v3 packet: primal uSVP core-SVP, ternary secret)

| instance | log2 q | sigma | beta | core-SVP classical / quantum |
|---|---|---|---|---|
| v3 upload (n 4096) | 56 | 3.24 | 818 | 238.9 / 216.8 |
| v3 reply key (n 8192) | 107.98 | 3.24 | 860 | 251.1 / 227.9 |
| V0 upload | 47.98 | 3.24 | 996 | 290.8 / 263.9 (higher, but incorrect, section 2) |
| V1 upload (`Q = q1 * 65537`, sigma scaled) | 63.98 | 819 | 818 | 238.9 / 216.8 (equal) |
| V1a reply key (`Q P`) | 123.98 | 3.24 | 716 | **209.1 / 189.7** (-42 bits) |
| V1c reply key (equal correctness: `sigma_ks x 35.7`) | 123.98 | 116 | 758 | **221.3 / 200.9** (-30 bits) |
| V2 reply key (`f = 2^8`, same `sigma_ks`) | 115.98 | 3.24 | 783 | **228.6 / 207.5** (-22 bits) |
| V3 reply key (`P` grows by `||w||_1` or by its 24-sigma bound) | 142.2-145.8 | 3.24 | 575-595 | **167.9-173.7 / 152.4-157.7** (-77 to -83 bits) |

All of these stay above 128 bits and inside the HomomorphicEncryption.org table (`log q <= 218` at 8192). In absolute
terms none is weak. Each is still a measurable loss against v3, and the gate excludes that. The upload instances keep
v3's `sigma / q`. They also need a new noise sampler (v3's CBD(21) cannot give sigma 819 cheaply) and a wider upload
word (64 bits instead of 56, 14 % more upload bytes). Both would need their own review.

## 4. Soundness of the check

### 4.1 The Freivalds part is fine

Per RNS prime `p` and linear, suppose the client compares an `F_p`-linear projection `pi(C mod p)` with
`pi(f_p(W8))`, where `f_p` is `F_p`-linear and the projection uses `d` independent uniform challenges in `F_p` from
the OS CSPRNG, never sent. Then a nonzero error matrix `E` passes with probability at most `2/p` per challenge pair:
the left vector misses `E x != 0` with probability `<= 1/p`, and `E x = 0` has probability `<= 1/p`. With `d`
independent pairs the bound is `(2/p)^d`.

* For `p ~ 2^24`, `d = 2` gives `2^-46`.
* For `f = 65537`, `d = 3` gives `2^-45`.
* Any wrong hint has a wrong linear, so the bound holds per provisioning. Fresh challenges per linear mean that
  corrupting more linears only lowers it.
* The NTT-slot projection `u^T (.) x` is rank-1 in (row, column) of `hs`, so `f_p(W8)` costs one streamed pass over
  `W8`, as the sketch says.

This part is not the problem.

### 4.2 What it checks is not what determines `c1`

`c1 = (C - [C]_P) / P mod q1`. To recompute `c1`, the client needs `C` modulo every prime of `q1 P`.

* **Modulo `p0, p1`:** `C = sum_l A~_l k_l` depends only on `hs mod q1`, which is linear in `W8` in the sketch.
* **Modulo the special primes `p2, p3`:** `C` depends on the centred integer lift of `hs`. That lift is not linear:
  `centred(a + b) != centred(a) + centred(b) mod p2` whenever the sum wraps.
* So the client can check only `C mod q1 = c1 P + w mod q1`.

**Forgery.** Take any `c1'`, for example `c1 + delta` with a small structured `delta`. Set
`w' = (C - c1' P) mod q1`. Because `w'` lies in `[0, q1)`, inside the witness range `[-P/2, P/2)`, it is a
well-formed witness that satisfies every equation the client can evaluate.

* The soundness error of the sketched check is **1**.
* This is verified with the v3 kernels: honest `C` for one 32-row block, forged `delta = 3` on 32 coefficients, all
  `mod q1` equations hold, and the lift is not linear mod `p2`.
* Such a `delta` is exactly the selective-failure delta of v3's section 5.

**Ways to make it sound:**

* *(a) Lift quotients.* The provider also sends `z = (W8 A - A~) / Q` for every entry of `H`: about 22 bits x 4096
  per output row, **17.3 GB at 7B**. The client checks `A~ = W8 A - Q z` linearly.
  * But a non-canonical `z` (`A~ +- Q`) is still consistent. So each `A~` needs a range argument (`|A~| <= Q/2`).
  * That is the succinct-range-proof problem v3 left open (`pks-v3-provider-hint.md` section 2), only smaller.
  * Without the range argument, the provider picks the lift. With `k` known (section 5) it can search for lifts that
    move `c1` by small amounts (`Q sum delta_l k_l / P` small mod `q1`: an SIS-type search). We have no argument
    that this search is hard.
* *(b) V3.* Key-switch the unreduced integer `H = W8 A_centred` (`|H| <= ||w||_1 Q / 2`). Then `C` is linear in
  `W8` modulo every prime. `(c1 mod q1, w in [0, P f))` maps one-to-one onto `C mod Q P`, so the check in 4.1 is
  sound (error `<= 2^-45`) with no range argument.
  * But the key-switch operand grows by `||w||_1` (2^21.8 worst case, 2^18.2 at 24 sigma). Keeping the key-switch
    noise needs `P` larger by that factor, and the reply key drops to 168-174 bits of core-SVP (section 3).
  * The download would be `c1` plus a witness of about 94-98 bits per coefficient: about 7 GB at 7B, not 5.

## 5. The provider's view compared with a local build

In every variant the provider must compute `C = sum_l A~_l k_l`, so it must learn `k`, that is `ks_seed`.

* In v3 (local build and the provider-`H` mode alike), `ks_seed` never leaves the client. The adversarial review
  records "the provider never needs `k`" (F13).
* Formally nothing breaks: the packet's hybrid H1 is decision-RLWE, where `k` is public anyway.
* But it removes a v3 defence-in-depth property, and it gives the provider what the 4.2(a) search needs.
* The provider also learns `a_seed` at provisioning, as in v3's provider mode (Q7). It also learns `c1`, which a local
  build never shows it.
* No secret exists at provisioning time (`t`, `t'` are drawn later), and the pass/fail bit concerns public data only.
  These points are as in Q7.

## 6. What an adversarial hint can still do

* **Sketch (V0/V1/V2 with the `[C]_P` witness):** everything in v3's section 5, *including selective failure*.
  * The provider ships `c1 + delta` with a forged witness (4.2). Decryption gains `delta t'`.
  * With `|delta t'|` tuned near the rounding margin, the failure pattern leaks bits of fresh reply secrets.
  * The online checks cannot see it, and the 3-failure halt bounds only the count.
  * This is precisely what v3's hint-integrity rule exists to exclude. **An integrity loss, not only a parameter
    cost.**
* **V3:** with probability `<= 2^-45` per provisioning, the same as above. Otherwise nothing: the accepted hint is
  bit-identical to the honest one.
  * Gross errors are caught by the check, and a wrong seed or wrong linear changes `f_p(W8)`.
  * The cost is the security loss of section 3 and the `k` exposure of section 5.

## 7. Costs the variants would have had (estimates at Qwen2.5-7B, 1.54M hint rows; not benchmarked)

| | download | client check CPU | per-request change |
|---|---|---|---|
| v3 `hint="local"` | `W8` 6.5 GB | 3029 CPU-s build | - |
| v3 `hint="provider"` | `W8` + `H` 44.2 GB | 1699 CPU-s | - |
| sketch (unsound) | `c1` + `[C]_P` 5.3 GB | one `W8` pass plus projections (about 100-200 CPU-s, as v3's `U^T W8` and `A X`) | upload +14 % bytes, reply key +25 % (5 primes), provider key switch +25 % |
| V3 (sound, -80 bits reply key) | about 7 GB | as above | reply key and provider key switch about +50 % (6 primes), upload +14 % |

Customer decryption (`c0 + c1 t' mod q1`) is unchanged in every variant: two primes, same hint format.

## 8. Recommendation

* Keep pks-v3. The SDK default `hint="local"` is unchanged.
* For fewer provider-mode bytes without a scheme change, the first option of `pks-v3-provider-hint.md` section 8 remains
  available: the client computes the low byte planes of `H` itself, and the same 2-adic check runs mod `2^(56-8j)`.
  It is sound with no new assumption, and offers a linear dial between 44 GB and 0 GB of download.
* If smaller downloads become worth a reviewed parameter change, V3 is the only variant here with a sound cheap check.
  Taking it would need:
  * an explicit decision to accept about 170 bits (classical core-SVP) on the reply key;
  * a new upload noise sampler;
  * sending `ks_seed` to the provider;
  * its own external review.
* Q7 of the v3 packet is unaffected. No Q8 was added, since nothing was built.
