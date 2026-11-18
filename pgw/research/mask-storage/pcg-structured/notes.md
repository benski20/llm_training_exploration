# Track pcg-structured: compressing the (r, W r) correlation

Labels: [M] measured here (this 4-core Xeon 2.1 GHz with AVX-512, shared, 2 threads); [C] cited; [E] estimated.

## 0. Reframing that matters

The service learns `W x_hat` regardless, so the server's half of any correlation can be **zero**.
What we need is one of the following:
1. a pseudorandom r for which the gateway can compute `W r` cheaply (one party, so no PCG protocol is needed), or
2. someone other than the customer holds `W r` in a form that does not let the service unmask.

PCG machinery (DPF + LPN expansion) is useful here only through its LPN expansion step. Section 3 explains why the two-party part does not apply.

Sizes used (Qwen2.5-7B, from the task statement): sum over linears of m*n = 6.525e9; sum of m = 1,390,592 per token; sum of n = 831,488 per token.
Llama-3-70B: sum m*n = 68.4e9; sum m = 6.72e6; sum n = 4.26e6 [E, from the shapes].

## 1. Options that survive

### Option B: the provider stores the incompressible half, blinded ("z-store")
- **Factory.** Per slot it derives `r = PRG(s_r)` and `c = PRG(s_c)` (both m- and n-vectors from seeds held only by the customer), computes `z = W r - c mod 2^K` and uploads z to the provider. The customer keeps only the seeds (32 B per segment or slot).
- **Online.** The gateway sends `x_hat = x + r`. The service returns `y' = W x_hat - z = W x + c`. The gateway computes `y = y' - c` (PRG regeneration, about m*5 B per linear).
- **Integrity.** The gateway runs an end-to-end Freivalds check, `u.y == (uW).x mod 2^K`. The check is on x rather than x_hat. The error the gateway sees is the server deviation plus any factory error, and an abort depends only on (error, u), never on x. As a result this option *also* closes threat T-29 (a wrong `W r` from the factory is now caught by Freivalds).
- **Privacy reduction.** The service sees (x_hat, z) = (x + PRG_r, W PRG_r - PRG_c). Replace PRG_c with uniform: z becomes uniform and independent. Then replace PRG_r with uniform: x_hat becomes uniform. So privacy reduces to PRG security, the same assumption as today. Trust is unchanged from today: the partner factory knows r, so partner and provider must not collude, exactly as now.
- **Required:** the c blind. Storing `z = W r` without it leaks `W x = W x_hat - z` (rejected item R10).
- **Costs.** Customer storage per token goes to ~0. Customer compute: 0 if a partner runs the factory. Provider storage is 6.95 MB/token (7 TB per 1M tokens of 7B, now our cost). Factory-to-provider upload is 6.95 MB/token. Factory GPU time is unchanged (T4 at about 169k tokens/h). Gateway extra work: about 1.39M PRG words per token (~7 MB of ChaCha, [E] ~3-5 ms per core).
- **Maturity.** Deployed-pattern: this is the "client-aided / dealer preprocessing" pattern. The novelty here is only the c blind plus the end-to-end check. Confidence: high.

### Option C: LPN-structured masks, generated just in time at the gateway with no per-token storage
- **Construction.** `r = H e mod 2^K`, with `H = [I_n | H']` public and seeded, and `H'` of size n x (N-n) uniform over Z_{2^K}. Here e is in Z_{2^K}^N with exactly t nonzeros, a uniformly random support, and uniform **odd** values.
- **Computing W r.** `W r = sum_{j in supp, j<n} e_j W[:,j] + sum_{j>=n} e_j T[:,j-n]`, where `T = W H' mod 2^K` is a one-time table. T is public information (W and H are both known to the service), so the provider may compute it. The customer verifies it once with Freivalds: `u^T T == (u^T W) H'`.
- **Cost per token per linear.** About m*t ring multiply-adds, plus n*t*(N-n)/N for r itself, instead of m*n dense MACs. Exactness was checked against pgw `ringmm` [M] (`lpn_mask_proto.py`).
- **Security.** The service sees `x + H e` with a fresh e per (token, linear). This is the decisional dual-LPN / syndrome-decoding assumption over Z_{2^K}, via a standard hybrid over samples (H is public).
  - The systematic form is without loss of generality: `A H e` with A invertible.
  - Over Z_{2^K}, the mod-2 reduction halves the effective weight when noise values are uniform. See Liu, Wang, Yang, Yu, "The Hardness of LPN over Any Integer Ring and Field for PCG Applications", EUROCRYPT 2024 [C]: prior Z_{2^k} PCGs overestimated security by up to 40 bits. That paper also gives search-to-decision equivalence and a reduction from F2-LPN.
  - Parameter rule used here: Prange cost `log2 C(N,w)/C(N-k',w)` at weight **w = t/2** must be at least 150 bits. For sublinear weight all ISD variants share the Prange exponent [C: Canto Torres and Sendrier, "Analysis of information set decoding for a sub-linear error weight", PQCrypto 2016]. The 150 target leaves about 19 bits for multi-target (DOOM, Sendrier, PQCrypto 2011, gain about sqrt(Q)) with Q <= 3e11 samples per H over 10 years at 1M tokens/day when one H' is shared per input width.
  - Before deployment, re-check with the Liu et al. estimator (artifacts.iacr.org/eurocrypt/2024/a1).
  - Resulting t [M, `prange_t.py`, independent of n for n >= 3.5k]: N = 1.5n gives t = 500-512; N = 2n gives t = 296-300; N = 4n gives t = 150.
  - The values must be uniform mod 2^K, not +-1. With +-1 values, e becomes short and lattice/ISIS attacks would need a separate analysis (R11).
- **One-time gateway storage, 7B [E from shapes].** N = 2n: T is 6.525e9 x 5 B = 32.7 GB, plus H' shared per width (3584^2 + 18944^2) x 5 B = 1.9 GB. N = 4n: 98 GB + 5.6 GB. 70B: 342 GB (N = 2n) or 1.03 TB (N = 4n).
  - Compare 7 TB/day today.
  - T can live on NVMe: the batched kernel streams each table tile once per batch of B tokens, so N = 2n with B = 4096 needs about 8 MB/token of sequential reads [E].
- **Ops per token, 7B.** N = 2n, t = 300: 4.2e8 (W r) + 1.25e8 (r) = 5.4e8. N = 4n, t = 150: 2.1e8 + 0.9e8 = 3.0e8. Dense W r is 6.5e9 MACs.
- **Measured CPU time, 2 threads, naive numba kernel** (`lpn_spmm_bench.py`, `lpn_spmm_tiled.py`):
  - Sparse: 0.48-0.54 ns per 64-bit multiply-add [M].
  - pgw dense `ringmm` (cpu_f64 backend, batched): 48-50 ps/MAC in a quiet run [M], 67-220 ps/MAC under load [M].
  - So 7B costs about 0.27 s/token (N = 2n) or 0.15 s/token (N = 4n) per 2 cores [E from M], against about 0.31 s/token for **dense** just-in-time `W r` [E from M].
  - The win is only 1.2-2x with this kernel.
  - An AVX-512 IFMA kernel (vpmadd52luq gives exact low 40 bits of 40x40 products, 8 lanes) could plausibly gain 5-10x [E, not built].
  - 70B, N = 4n: about 1.5e9 ops, roughly 0.75 s/token per 2 cores, against about 3.3 s dense [E].
- **Integrity.** Use the same end-to-end Freivalds check as in B. A table error is then caught online too.
- **Maturity.** Dual-LPN expansion is deployed inside every silent-OT/VOLE PCG (Boyle et al., CCS 2019 and CRYPTO 2019; Silver, CRYPTO 2021, later broken, so use **uniform random H, not structured codes**). Applying it to make `W r` cheap is new here (prototype only). Confidence: medium on security (standard assumption, but the parameters need the official estimator), high on correctness.

### Reference point (not crypto-structured, so outside this track, but it frames C)
Dense just-in-time generation of (r, W r) at the gateway removes per-token storage with no change to the security model. The cost is about 0.31 s/token per 2 cores for 7B [E from M]: 1M tokens/day takes about 7 cores. For 70B it is about 3.3 s/token per 2 cores, or about 77 cores. Option C cuts this compute by 1.2-4.5x as measured, and maybe more with IFMA. B removes it entirely by putting the storage on our side.

## 2. Rejected ideas, with attacks

- **R1. Subspace masks, r in span(B) with dim d < n** (including "r = sum of a few stored basis vectors with fresh coefficients"). The service computes a left-kernel matrix P with P B = 0 mod 2^K, so P x_hat = P x. That gives n-d exact linear equations in a short x (|x_i| < 2^11). For n-d >= d plus a few, x is the unique short solution, recovered by Babai/LLL on a d-dimensional coset (or directly when d is small). Distinguishing needs only one P row. **Insecure.**
- **R2. Sparse r.** The coordinates where r_i = 0 are sent in the clear. **Insecure.**
- **R3. Masks reused across tokens or layers** (`r_t = r`). Then `x_hat_t - x_hat_s = x_t - x_s` exactly, and skewed activations (outlier channels) reveal the content. **Insecure.**
- **R4. Reuse plus fresh sparse noise** (`r_t = r + e_t`). Differences reveal `x_t - x_s` outside at most 2t coordinates. **Insecure.** "Session masks with re-randomisation" fails the same way unless the re-randomiser is itself a full-entropy mask, which brings back the original cost.
- **R5. Scalar re-randomisation** (`r_t = a_t r`, cheap since `W r_t = a_t W r`). From `x_hat_t - lambda x_hat_s = x_t - lambda x_s` with `lambda = a_t / a_s`: guess (x_t[0], x_s[0]) in 2^24 tries, giving lambda (when the denominator is a unit), then confirm that all coordinates are small. Recovers `x_t - lambda x_s`; three tokens give everything. **Insecure.** Diagonal or permutation re-randomisation gives no cheap `W r_t` anyway.
- **R6. Primal LPN, `r = A s + e`** (s uniform, dimension k). Secure under LPN, but `W r = (WA) s + W e` costs m(k+t). With only n samples, k must be about n/4 to n/2 (t about 350 at k = n/4), so the saving is at most 4x and dominated by the dual form, C. (HNF-LPN, Applebaum, Cash, Peikert, Sahai, CRYPTO 2009, shows dual-with-systematic-H equals primal LPN with a noise-distributed secret, which is why C is the right form.) **Dominated.**
- **R7. LWE-style masks** (r = A s + e with a small dense e). `W e` is a dense matmul, so no saving. Ring-LPN/Ring-LWE masks don't help either: W is not a ring element (it doesn't commute with polynomial multiplication). **No benefit.**
- **R8. Two-party PCGs for (r, W r - s)** (silent VOLE/OLE: Boyle, Couteau, Gilboa, Ishai, Kohl, Scholl, CRYPTO 2019; Ring-LPN OLE, CRYPTO 2020, ePrint 2022/1035; PCFs from VDLPN, FOCS 2022; matrix-triple PCGs: Bombar, Couteau, Melissaris, Modi, ePrint 2026/2298 preprint).
  - VOLE compression works because `Delta * e` stays sparse for a *scalar* Delta. With the matrix W, `W H e` is dense, so the DPF-shared object is a dense m-vector: no compression.
  - With DPF shares of e itself, the customer must evaluate `W H d0` on a dense share. That costs more than today.
  - Ring-LPN OLE needs W to be a ring element. Matrix-triple PCGs produce *random* A and B, not a fixed public W known to the server.
  - All of these also need a two-party setup with the service.
  - **Not applicable.** The only reusable piece, the LPN expansion, is Option C.
- **R9. The service computes W r from seeds obliviously (no HE, no second party).** That is single-server oblivious linear evaluation, which implies OT, so it needs public-key assumptions. It does not amortise because each slot needs fresh r, and amortisation is exactly what R8 shows is unavailable.
  - Single-server PIR over the table T to fetch the t columns privately is LWE-HE-based (excluded). It is also impractical: about 300 x 112 column queries per 7B token, each scanning the database ([E]; SimplePIR, USENIX Security 2023, reports about 10 GB/s/core server throughput [C], so about 3 s per query at 33 GB).
  - **Rejected.**
- **R10. Provider stores `W r` without a blind.** It leaks `W x = W x_hat - W r`, and x itself for tall layers (qkv, gate_up). **Insecure.**
- **R11. Option C with +-1 noise values** (adds only, no multiplies). e becomes ternary and short, so `H e = r` is an ISIS instance with a very short solution (norm about 17 against a Gaussian heuristic of about 2^24 at N = 7168, n = 3584, q = 2^40). Sparse-secret lattice and hybrid attacks are not analysed. **Research-only.**
- **R12. Moving H' to a cheap structure** (quasi-cyclic) to avoid storing or regenerating H'. The block lengths 3584 = 2^9 * 7 and 18944 = 2^8 * 74 give badly factoring X^n - 1 mod 2 (folding attacks). It would need padding to a prime p with 2 primitive mod p, plus truncation (a non-standard assumption). **Research-only.** C instead stores a uniform H' (1.9 GB).

## 3. Artifacts
- `lpn_mask_proto.py`: exactness of `W r` via table against pgw ringmm [M: OK].
- `lpn_spmm_bench.py`, `lpn_spmm_tiled.py`: kernel timings [M].
- `prange_t.py`, `prange_t.json`: noise weights.
- The full TII `cryptographic_estimators` SD estimator was too slow here (>5 min per point); it was killed.
