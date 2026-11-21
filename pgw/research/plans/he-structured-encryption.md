# Gateway encryption cost of the encrypted mask factory: structure, exact fast path, opt-in parameters

Status: prototype, 2026-11-21. Package M1 ("cut gateway encryption cost by an order of magnitude").
Code: `pgw/src/pgw/hemask/mlwe_ring.py`, tests `pgw/tests/hemask/test_mlwe_ring.py`, benchmark
`pgw/benchmarks/hemask_encrypt_bench.py`, Kaggle job `pgw-he-encrypt-ntt`.

## 1. Finding: the public matrix is already structured; the dense cost is an implementation choice

The scheme in `hemask/lwe.py` is module-LWE over `R = Z[X]/(X^256+1)`, rank 16, `q = 2^56` (K32):
for input feature `c` of a linear and a group of 256 token slots (coefficient encoding),

    b_c = sum_{l<16} a_{c,l} * s_l + e_c + 2^(q-K) r_c        (mod q, in R)

with `a_{c,l}` uniform (expanded from a gateway-chosen seed), `s` a fresh uniform-ternary secret per
(linear, group), `e` CBD(21) noise. Row `c` of the "dense n x 4096 matrix A" is exactly the module
element `(a_{c,0}, ..., a_{c,15})`; the dense code multiplies it by the secret's 4096 x 256
block-negacyclic matrix `S`, i.e. it evaluates a ring product with a matrix product:

| per row and 256-slot group           | operations                         |
|--------------------------------------|------------------------------------|
| dense (today, `encrypt_masks`)        | 4096 x 256 x 7 planes = 7.34e6 int8 MACs |
| NTT, A transformed per call (`ntt.big_planes_mm_small`, used today when the int8 engine is not fast) | 48 forward NTTs + MACs, ~1e5 modmuls |
| **NTT, A transformed once (`ntt_pre`)** | 3 primes x (16 x 256 pointwise MAC + 1024 inverse butterflies) = 1.5e4 modmuls |

So "a structured public matrix (A made of negacyclic 256x256 blocks)" is not a new scheme: it is
the current scheme. The order-of-magnitude saving is available **without any cryptographic change**,
by keeping `A` (public, fixed per hint store) in NTT form and only doing pointwise products and one
inverse NTT per row and group. That is what `PretransformedEncryptor` does.

## 2. The exact fast path (`he_encrypt = "ntt_pre"`, default `"dense"`)

* `A[:n_max]` is transformed once with the existing native crate `pgw_ntt`
  (`ntt.TransformedOperand`, 12 bytes per coefficient: 0.24 GB at 0.5B, 0.93 GB at 7B
  (n_max = 18944), 1.41 GB for a 70B-class model (n_max = 28672); it replaces the 7-plane int8 copy
  of `A`, 0.54 GB at 7B, which the backend no longer builds in this mode).
* Per pass of `G` groups (default 4): sample the per-group seeds (OS CSPRNG), secrets and noise with
  the *same* samplers as `encrypt_masks` (`sample_secret`, `sample_noise`), one native call computes
  `A[:n] @ [S_1 | ... | S_G] mod 2^64` from the transformed `A` (pointwise MAC over the rank, inverse
  NTT, Garner CRT over three 29-bit primes), and the pad becomes the ciphertext in place
  (`+ e + (r^T << (q-K))`, `& (2^q - 1)`).
* **Exactness.** The CRT recovers every coefficient `|Z| <= (2^56 - 1) * L1(s) <= 2^56 * 4096 = 2^68`
  exactly (`HALF ~ 2^86`, checked natively on every call, see `hemask/ntt.py` and
  `rust/pgw-ntt/src/arith.rs`). Hence the pad equals the dense pad mod 2^64, and the ciphertexts are
  **bit-identical** to `encrypt_masks` for the same per-group seed: tested for K32/q56 and K40/q64,
  rank 16 and 12, 1..700 slots, against both the plane and the uint64 reference
  (`test_bit_identical_to_encrypt_masks`), and through `HEMaskBackend` against the dense backend
  (`test_backend_ntt_pre_exact_and_same_ciphertexts`). Round trips encrypt -> provider `W8 b mod 2^q`
  -> decrypt with `H = W8 A` give `W8 r mod 2^32` exactly (`test_round_trip_exact`, worst-case
  weight rows). The provider, the wire, the hint and decryption are unchanged.
* **Privacy.** Same distribution of ciphertexts as before, so the security argument is unchanged.
  Secret hygiene: the pad (`A s`) is never kept separately (it is turned into `b` in place and wiped
  on error), secrets and their compact copies are wiped, per-group seeds are dropped after use.
* Switch: `FactorySettings.he_encrypt` (`PGW_FACTORY_HE_ENCRYPT=ntt_pre`), passed to
  `HEMaskBackend(encrypt=...)`. Default stays `"dense"` (no behaviour change); since it is proven
  exact it can be made the CPU default after one end-to-end run. On a gateway with a GPU the dense
  path runs on the GPU (`engine() == "cuda"`) and may stay faster; see the measurements.

### Measurements

Local box (shared 4-vCPU Xeon @ 2.1 GHz with AVX-512/AMX flags, `fastmm.engine() == cpu_f64` because
the int8 timing probe fails here, so *today's* encryption takes the NTT path that transforms `A` on
every call). Numbers are full encryptions (sampling + pad + combine), extrapolated per slot from
2048 rows; the box was shared with other agents' test runs, so expect +-30 %:

(see section 6, filled in from `benchmarks/hemask_encrypt_bench.py`)

## 3. Security of the current parameters (and why nothing below rank 12 is offered)

Method: primal uSVP, "2016 estimate" / core-SVP (NewHope, Kyber, Saber): smallest BKZ block size
`beta` with `sigma_e sqrt(beta) <= delta(beta)^(2 beta - d - 1) Vol^(1/d)`, Bai-Galbraith embedding with
the ternary secret rescaled (`sigma_s = sqrt(2/3)`), number of samples optimised (the provider sees
`n x 256` samples per secret, at least 896 x 256, far more than the optimum ~3000). Cost
`2^(0.292 beta)` classical, `2^(0.265 beta)` quantum. Implemented in `mlwe_ring.core_svp_beta` (tested).
Calibration: the HomomorphicEncryption.org standard's 128-bit entries (ternary secret, sigma 3.2)
come out at core-SVP 95-101 classical in this model (n=1024/log q=27: 101.3; n=2048/54: 98.4;
n=4096/109: 95.5), i.e. that standard's cost model adds ~30 bits of overhead on top of core-SVP.

| profile   | n = 256 x rank | log2 q | noise | secret   | beta | core-SVP classical | quantum | HE-std 128-bit max log q |
|-----------|----------------|--------|-------|----------|------|--------------------|---------|-------------------------|
| rank 8    | 2048           | 56     | CBD(21), sigma 3.24 | ternary | 320 | **93.4** | 84.8 | 54 (so 56 is *below* 128-bit) |
| rank 9    | 2304           | 56     | same  | ternary  | 378  | 110.4              | 100.2   | (~61, interpolated)       |
| rank 10   | 2560           | 56     | same  | ternary  | 437  | 127.6              | 115.8   | (~68)                     |
| **rank 12** (opt-in) | 3072 | 56  | same  | ternary  | 560  | **163.5**          | 148.4   | (~81)                     |
| rank 14   | 3584           | 56     | same  | ternary  | 687  | 200.6              | 182.1   | (~95)                     |
| **rank 16** (default) | 4096 | 56 | same  | ternary  | 818  | **238.9**          | 216.8   | 109                       |
| rank 16, q=2^64 (K40) | 4096 | 64 | same | ternary | 687 | 200.6             | 182.1   | 109                       |

Consequences:

* The default (rank 16, q = 2^56) has a very large margin: core-SVP ~239 classical, and
  `log q = 56` is about half of what the HE standard allows at n = 4096.
* **Existing exposure, fixed here:** `FactorySettings.he_rank` accepted any rank >= 8. Rank 8 at
  q = 2^56 is below 128-bit security by the HE standard (54 bits allowed) and by core-SVP (93). The
  settings now refuse every rank below 16 except the opt-in rank-12 profile, which needs an explicit
  acknowledgement (`mlwe_ring.check_rank`). `HEParams(rank >= 8)` itself is unchanged (tests and
  benchmarks use it directly); production configuration goes through `FactorySettings`.

## 4. Opt-in reduced-dimension profile `mlwe-r12` (a cryptographic change; OFF by default)

* Parameters: R = Z[X]/(X^256+1), module rank 12 (LWE dimension 3072), q = 2^56 (K32; q = 2^64 for
  K40), secret uniform ternary per (linear, 256-slot group), noise CBD(eta=21) per coefficient
  (sigma = 3.24), public matrix uniform from a gateway-chosen 32-byte seed, scaling 2^(q-K).
* Same assumption family (decision MLWE with ternary secret, power-of-two cyclotomic, many samples
  per secret); only the dimension shrinks. Estimated core-SVP 163.5 bits classical / 148.4 quantum
  (Kyber-768 is quoted at core-SVP ~182, Kyber-512 at ~118); comfortably above 128 even without the
  HE-standard overheads.
* Correctness is unchanged: the decryption noise does not depend on the rank except the hint-drop
  term `255 x dim`, which shrinks (`HEParams.noise_bound`); at K32 the worst 7B row (n = 18944,
  ||w|| <= 127 sqrt(n)) gives a bound of 2^21.05 (rank 16: 2^21.22) vs the 2^23 budget; a 70B-class row (n = 28672) 2^21.25 (rank 16: 2^21.39).
  Round trips are tested at rank 12 (`test_round_trip_exact[12]`).
* Gain: encryption, decryption and the one-time hint (`W8 A`, the gateway's largest object: ~33 GB at
  7B with 6 byte planes) all scale with the rank, so -25 % for each. The provider's work (`W8 b`) does
  not depend on the rank.
* Enabling: `PGW_FACTORY_HE_RANK=12` **and**
  `PGW_FACTORY_HE_PARAMS_ACK=i-accept-unreviewed-he-params:mlwe-r12` (`mlwe_ring.R12_ACK`); without
  the exact string the factory settings refuse to load. Hint stores are keyed by rank, so the two
  profiles never mix.

## 5. Designs considered and not implemented (larger changes)

1. **Ring-LWE N = 4096, rank 1** (same dimension). The pad then costs 1 pointwise MAC + a length-4096
   inverse NTT (12 levels) per coefficient instead of 16 MACs + 8 levels: ~2-3x fewer modular
   operations than `ntt_pre` at rank 16. Provider-compatible (a 4096-slot group is sent as 16
   ordinary 256-column groups; `W8 b` is still column-wise), hint shape unchanged. Needs: primes
   = 1 mod 8192 (the current ones are only 1 mod 512), a length-4096 native NTT (the `pgw_ntt` kernels
   are specialised to 256), decryption of `H s` as degree-4096 products, 4096-slot granularity. The
   assumption (RLWE, power-of-two cyclotomic, n = 4096, log q = 56) is *the* setting the HE standard
   tabulates, so the security memo is easy; the engineering is the cost. Recommended next step only
   if CPU encryption is still a bottleneck after `ntt_pre` (it mostly is not; decryption and the hint
   are bigger, see 6).
2. **NTT-friendly prime q with q = 1 mod 2^K** (BFV-style scaling; one 60-bit prime instead of three
   CRT primes). Saves the CRT, but the provider must then reduce mod q instead of wrapping mod 2^64:
   provider kernel and wire change. Not pursued.
3. **Secret reuse across groups or linears.** The secret transform is already negligible (rank x 3
   NTTs per group); reusing secrets saves nothing and needs a new argument. Rejected.
4. **Compressed / seeded A.** `A` is already expanded from a 32-byte seed and never sent; nothing to gain.
5. **GPU NTT.** A customer GPU already runs the dense byte-plane GEMM fast (`engine() == "cuda"`);
   an NTT kernel on the GPU would make encryption negligible there too. Left to the gateway-GPU work.

## 6. Numbers

Filled in below from the local and Kaggle runs.

## 7. What a human cryptographer must check

For `ntt_pre` (no change to the scheme): only that the ciphertext distribution is unchanged, i.e.
that the bit-identity tests are convincing and that the per-group seeds are fresh (`new_seed()` per
group, `_seeds` is a test hook). Nothing else.

For `mlwe-r12` (and before any other rank is allowed):

1. Re-run the estimate with the lattice estimator (`LWE.estimate(LWE.Parameters(n=3072, q=2**56,
   Xs=ND.UniformMod(3), Xe=ND.CenteredBinomial(21), m=oo))`), including dual, hybrid-dual and
   meet-in-the-middle / hybrid attacks on the ternary secret; also at q = 2^64 for K40.
2. Module structure: rank 12 over degree-256 power-of-two cyclotomic; no known attack exploits the
   module structure at these sizes, but confirm (MLWE -> LWE reduction used as the heuristic).
3. Sample count per secret: `n_linear x 256` coefficients (up to 28672 x 256 for 70B-class down
   projections) under the *same* secret, with fresh noise per coefficient. Confirm no issue with
   that many samples (standard for LWE estimates, which assume m unbounded).
4. Leakage beyond ciphertexts: the provider sees only `b`; it never sees decryption results, but
   the gateway's Freivalds self-check rejects a bad `W8 b`. A malicious provider could return a
   tampered `y` and learn whether the gateway accepted (one bit per request). This holds for all
   ranks today; confirm the self-check rejects *before* any observable behaviour depends on the
   decrypted value, and that decryption failure probability (24 sigma tail) is negligible.
5. Seed handling: secrets/noise from `SeedExpander(os.urandom(32))` (ChaCha20); hygiene of wiped
   buffers.
