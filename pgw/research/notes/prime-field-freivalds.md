# Freivalds over a prime field: one check vector per linear (segment format `"p31"`)

Date: 2026-11-22. Status: implemented, opt-in (factory `check_field = "p31"`, gateway `prime_check = true`),
default off. Normative text: SPEC 5.1. Code: `pgw.core.freivalds` (reference), `rust/pgw-fast/src/prime.rs`
(kernels), `trip.rs` (`trip_prepare_into(..., field=p)`, `trip_prime_output`), `pgw.engine.backends`
(TripEngine wiring), `pgw.maskstore` (format, store, in-process RAM ring), `pgw.factory.generator`.

## Why

At 7B (Qwen2.5-7B shapes, K32A12C27, 113 trips per token) the gateway read 66.6 MB of Freivalds check
vectors per token: 7 vectors over `Z_{2^32}` per linear (`u` over the outputs, `uW` over the inputs), because
over a ring of 2-power order a vector only sees an error up to its 2-adic valuation. After the range check
(`|yq| <= 2^27`) an error can still be divisible by `2^29`, so each vector contributes `h = 3` bits and
7 vectors give 21 bits per call. Those reads cost 7-8 ms of a helper core per token and main-core memory
bandwidth, and the segments hold 23.4 MB of `uW` at rest plus 43 MB of derived `u` in RAM while active.

## The check

`p = 2^31 - 1`. For each linear the factory draws secret `u in F_p^m` (one vector by default) and stores
`uWp = u^T W8 mod p`. The gateway checks the unmasked integers it already holds:

```
accept row t  iff  |yq_t| <= B   and   u . yq_t == uWp . xq_t  (mod p)
xq_t = quantized input (prepare), yq_t = centered(y_hat_t - W r_t) (finish), B = output bound (cap)
```

The check is on `yq` and `xq`, not on the masked `y_hat` / `x_hat`: the masked values only satisfy the
product relation modulo `2^K` (the provider returns `W8 x_hat mod 2^K`; the carries are unknown), which is
not checkable modulo an odd prime. Unmasking first turns the ring identity into an integer identity.

## Soundness (per call, at least as strong as today's, never weaker)

1. Completeness: `|W8 xq| <= B < 2^31`, so the genuine decoded `yq` equals `W8 xq` over `Z`, and
   `u . W8 xq = sum_l (u^T W8)_l xq_l == uWp . xq (mod p)`.
2. Lifting: an accepted row with a wrong output has `e = yq - W8 xq != 0` in `Z^m`, and
   `|e_j| <= |yq_j| + |(W8 xq)_j| <= 2B`. The format requires `2B < p` per linear (`prime_soundness_bits`
   returns 0 otherwise, so `min_soundness_bits` refuses the segment), hence `e mod p != 0`. Without this
   condition an error of exactly `p` would pass (the tests show the range check catching that case).
3. Independence: `u` is derived from the secret segment seed and never leaves the gateway / encrypted
   segment. The provider's view is `x_hat = xq + r` with fresh uniform `r` (independent of `u`); `xq` and `W8`
   are independent of `u`. So `e` (any function of the provider's view, `xq` and `W8`) is independent of `u`.
4. Probability: pick `j` with `e_j != 0 mod p`. For any values of the other coordinates exactly one value of
   `u_j` satisfies `u . e == 0`. `u_j = (ChaCha20 u64) mod p` has point probability at most
   `ceil(2^64 / p) / 2^64 <= (1 + 2^-33) / p`. With `c` independent vectors:
   `Pr[accept] <= ((1 + 2^-33) / p)^c`.
5. Per call: a call is accepted only if every wrong row is, so the per-row bound is the per-call bound.
6. Across calls: honest calls always pass, any failure aborts the request and trips the circuit breaker
   (SPEC 3.2). The first wrong call is accepted with probability `<= ((1 + 2^-33)/p)^c`; this is the same
   adaptive argument as for the ring check.

| | ring check (today) | prime check, c = 1 | prime check, c = 2 |
|---|---|---|---|
| bits per call, K32A12C27, any 7B linear | `c * h = 7 * 3 = 21` | **30.99999999** | 61.99999999 |
| condition | `h = K - ceil(log2 4B) > 0` | `2B < p` (K32 rings; `B <= 2^30 - 1`) | same |
| what it covers | the service product `W8 x_hat mod 2^K` | the decoded output `yq` (also a wrong `W r`) | same |

So one prime vector is strictly stronger than the 7 ring vectors it replaces. `2B < p` holds for
K32A12C27 at every width (`B = 2^27`) and for K32A8 up to `n = 66572` (not for a K32A8 LM head of width
152064, which is refused: use K32A12C27, as today's defaults do). Rings wider than 32 bits keep the ring check
(a `p = 2^61 - 1` variant would cover them; not implemented).

## Kernels (exact modular arithmetic)

Values are `i32`-range (`|xq| < 2^31`, `|yq| <= 2^31` at K32), words `u < 2^32`. `sum u v mod p` is computed
as `(sum u (v + 2^31) - sum u) mod p` (`2^31 = 1 mod p`): `v + 2^31` is a non-negative 32-bit value, so every
product is one `u32 x u32 -> u64` (`vpmuludq`), folded once (`x -> (x & p) + (x >> 31)`, `< 2^34`) and
summed lazily in `u64` (blocks of 2^20 terms, reduced between blocks); the final reduction folds twice and
subtracts `p` with a mask. LLVM vectorizes the loop; AVX-512F (8 lanes), AVX2 (4 lanes) and portable builds
are dispatched at run time and give identical integers (tests force each with `PGW_PRIME_ISA`). Loops run over
public sizes only, every vector of every row is compared, and the `yq` scratch is wiped.

* Input half: fused in `trip_prepare_into(..., field=2**31 - 1)`: after the constant-time width selection
  the row's `xq = centered(x_hat - r)` (exact) is dotted with `uWp` (`n` words).
* Output half: `trip_prime_output(y_hat, wr, u, rhs, k, bound) -> (status, bad_rows)`: decodes `yq` from
  `y_hat` and `W r` (read only), range-checks it and compares `u . yq` with `rhs`. The backend runs it
  synchronously before `trip_finish_into` (which then runs with no check and wipes `W r`). The call shape is
  the ring kernels' (inputs, outputs/W r, check data -> status), so a native one-call-per-trip path can call
  it in place of `trip_freivalds_output`; it can also run on a helper core if `W r` is kept until it returns.

## Format and wiring

* Segment header: optional `check_field: "p31"` (absent = ring check, byte-identical old headers; readers that
  do not know it refuse the header). Chunk 0 layout unchanged (`uW` slot holds `uWp`, 4-byte words).
  `u` = `ring_uniform((c, m), 64, "u-p31", linear) mod p`.
* Factory: `MaskGenerator(check_field="p31")` / `PGW_FACTORY_CHECK_FIELD=p31`; `checks_for()` returns 1.
  `uWp` is an exact limb product (16-bit limbs of `u`, float64 GEMMs on the weight's device).
* Store / RAM ring: per-extent format; `checks_compact` / `take_checked` serve prime vectors with a token
  ending in `"p31"`; `checks()` / `checks_u64()` (ring-check consumers) raise for prime extents (fail closed).
  The shared-memory ring (`ringshm`, factory child process) is not wired.
* Gateway: `prime_check = true` (`PGW_GATEWAY_PRIME_CHECK`) turns the TripEngine on and accepts prime
  extents; off, they are refused before any row is used. Ring segments keep the ring check on the same
  backend. Padded / fixed-structure calls (speculation), non-trip fallbacks and the MoE oblivious path refuse
  prime extents.

## Measured (this box: 4 shared vCPUs, Xeon 2.3 GHz with AVX-512; synthetic 7B shapes, `--wnorm-frac 0.3`)

See the table in STATUS.md (package W2) for `native_trip_floor_bench.py` and `trip_cpu_bench.py`
before/after numbers.
