# pgw-ntt

Exact NTT products for the customer side of the encrypted (MLWE) mask factory (plan E3, option a:
q stays 2^56, wire and provider unchanged). Python module `pgw_ntt`, wrapped by `pgw.hemask.ntt`.

The customer multiplies a big operand by the block-negacyclic matrix `S` of a ternary secret:

| product | big operand | shape per 256-slot group | dense cost per row |
|---|---|---|---|
| encrypt pad `A[:n] @ S` | public matrix, 64-bit words (8 int8 planes) | `n x 4096` (rank 16) | 8 planes x 4096 x 256 int8 MACs |
| decrypt correction `H @ S` | hint `W8 A`, bytes 1..6 (6 planes) | `m x 4096` | 6 planes x 4096 x 256 int8 MACs |

Here each row is `sum_l a_l * s_l` in `Z[X]/(X^256+1)`: residues mod three primes `~2^29`
(536870401, 536865281, 536856577, all `1 mod 512`), forward NTTs of the row's `rank` polynomials,
`rank` pointwise multiply-accumulates per coefficient and secret, one inverse NTT per secret, and
Garner CRT with a centred last digit, then reduction mod `2^64`. Exactness proof: `src/arith.rs`
(`|Z| <= (2^b - 1) * L1(s) <= p0 p1 (p2 - 1) / 2 ~ 2^86`; `2^76` at rank 16), checked on every call.
The result is the dense result bit for bit (`tests/hemask/test_ntt.py`, `cargo test -p pgw-ntt
--no-default-features`).

Rows are processed 16 at a time, one row per vector lane (`[u32; 16]`), so every butterfly is
full width. AVX-512 kernels (`src/avx512.rs`: even/odd `vpmuludq` Montgomery, register
accumulators, in-register 16x16 transposes) and auto-vectorised portable kernels (compiled for
AVX2 and baseline x86-64) are picked at run time; all give the same bits.

Build: `python rust/tools/build_ntt.py` (installs `src/pgw_ntt.abi3.so`). Microbenchmark:
`cargo run --release -p pgw-ntt --no-default-features --example ntt_bench -- 512 1 [avx512|avx2|scalar]`.
Model-level comparison with the dense path: `python benchmarks/hemask_ntt_bench.py`.
