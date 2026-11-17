# pgw-core-rs

Small, dependency-light Rust reimplementation of the **security-critical core** of PGW, written to be
audited independently of the Python package and kept bit-compatible with it (SPEC.md 5, 7.3, 7.5).

| module | what it implements |
|---|---|
| `ring` | `RingParams` (`K32A8`, `K40A12` default, `K48A20`; K multiple of 8 in [16,56]), reduce/center/add/sub/mul, output bound and headroom check, balanced base-2^b digits |
| `quant` | per-token activation and per-row weight quantization (float64, round-half-even, clamp), rescale |
| `packing` | `K/8`-byte little-endian ring encoding (wire and mask store) |
| `ringmm` | exact `x @ W8^T mod 2^K`: reference (wrapping u64) and the server algorithm (int8 limbs, int32 accumulators, inner chunks <= 131064) |
| `masking` | `x_hat = xq + r`, `yq = center(y_hat - W r)`, range check |
| `freivalds` | `u^T W8` precompute, per-row check, `soundness_bits` |
| `prng` | `SeedExpander`: HKDF-SHA256 + ChaCha20 (RFC 8439) stream per typed label list, seekable `ring_uniform_rows`, `derive_seed`/`child`, `uniform_int`; seeds zeroized on drop |
| `reservation` | `Ledger` (all-or-nothing reserve, oldest segment first, generation counter, write-off, shredding, invariant `verify`, rollback checks `check_successor` / `check_witness`) and `ReservationCursor` (per-linear in-order rows, no extent crossing, burn on close) |

`#![forbid(unsafe_code)]`; runtime dependencies are only `chacha20`, `hkdf`, `sha2` and `zeroize`.
Errors carry the stable `pgw.core.ErrorCode` strings. Nothing is logged or formatted from secret data.

## Test

```bash
cd pgw/rust && cargo test          # unit tests + tests/vectors.rs (cross-language vectors)
cargo clippy --all-targets && cargo fmt --check
```

## Cross-language vectors

`pgw/tests/vectors/core_v1.json` is produced from the Python reference and must never be edited by hand:

```bash
cd pgw && python rust/tools/gen_vectors.py          # regenerate
python rust/tools/gen_vectors.py --check            # verify (also done by pytest tests/vectors)
```

It covers label encoding, HKDF keys, ChaCha20 streams (several seeds/labels), ring elements for
K = 16..56, seeked mask rows (including offsets past 3 GB), `uniform_int`, `derive_seed`, ring constants and
soundness bits, balanced digits (8 and 26 bits), packing, and end-to-end masked linears for K32A8, K40A12 and K48A20
(quantization bit patterns, `r`, `u`, `x_hat`, `y_hat`, `W r`, `yq`, `u^T W`, tampered results and verdicts).
A diff in this file means `pgw.core` changed behaviour: a protocol break for every independent implementation.

## Python bindings (optional)

Feature `python` exposes `pgw_core_rs` (`random_bytes`, `ring_uniform_rows`, `mask_activations`, `unmask_outputs`,
`ringmm`, `freivalds_check`, `pack_ring`) via pyo3 (abi3, Python >= 3.12). They are not built by the tests:

```bash
maturin develop -m rust/pgw-core-rs/Cargo.toml --features python
# or: PYO3_PYTHON=python3 cargo build --release --features python && cp target/release/libpgw_core_rs.so pgw_core_rs.abi3.so
```

When importable, `pytest tests/vectors` checks the extension against the vectors too.

## Deliberate differences from the Python core

The Rust crate is the auditable reference, so it is stricter wherever the Python core silently
accepts input that would break a guarantee. All outputs on valid input are bit-identical (the
vectors prove it).

| function | Rust | Python (`pgw.core` / `pgw.maskstore`) |
|---|---|---|
| `mask_activations` | rejects non-canonical `r` and `\|xq\| > qmax` (`ring_overflow`) | rejects `\|xq\| > qmax` (`ring_overflow`) since the integration review; `r` is not range-checked |
| `quantize_weight_rows` | rejects non-finite weights (`ring_overflow`) | same (adopted in the integration review) |
| `freivalds_check` / `verify` | `c == 0` is a `config` error | same (adopted in the integration review); the mask store also refuses to hand out checks for `checks=0` segments |
| `SeedExpander::stream_bytes` | refuses a stream that *ends* past the 2^32-block counter | checks the start only |
| `ReservationCursor` | not `Clone`; refuses overlapping extents (`mask_reuse`); `take` after `close` is `mask_reuse` for any linear | n/a |
| `Ledger::write_off(Exported)` | only an unused active segment (`mask_reuse` otherwise), as `export_segments` | same |
| `Ledger::write_off(Missing/Quarantined)` | only active or consumed segments | only on-disk segments |
