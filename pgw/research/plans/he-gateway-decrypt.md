# Gateway side of the encrypted mask factory: fused NTT decryption, skipped local-linear masks

Status: 2026-11-21, package M3 of the mask-factory throughput work. Code: `rust/pgw-ntt` (`kernel::decrypt`,
Python `pgw_ntt.decrypt_planes` / `Transformed.decrypt`), `pgw.hemask.ntt.decrypt_into`,
`pgw.hemask.backend.HEMaskBackend(decrypt=..., hint_ntt_bytes=...)`, `pgw.factory.MaskGenerator(skip_kinds=...)`,
`pgw.maskstore.MaskStore.require_linears`, `pgw.gateway.runtime` (`mask_linears_needed`). Tests:
`tests/hemask/test_he_decrypt.py`, `tests/maskstore/test_skip_linears.py`,
`tests/gateway/test_speed_wiring.py::test_local_attention_serves_from_segments_without_qkv_o`, and
`cargo test -p pgw-ntt --no-default-features` (`decrypt_matches_product`). Benchmarks:
`benchmarks/hemask_decrypt_bench.py`, `benchmarks/gateway_gpu_bench.py --mode he` (new `--decrypt`,
`--hint-ntt-gb`, `--he-device none`, `--skip-kinds`). Kaggle job: `pgw-he-decrypt-m3`.

## 1. Where the decryption time went

Modal (one container, 6 vCPUs, T4 provider, Qwen2.5-0.5B, K32A12C27, `he_encrypt = ntt_pre`) measured
`512 slots in 16.25s ... he: 192 groups, encrypt 1.61s, provider 9.40s, decrypt 4.12s`. A 512-slot segment
(the store's ramp) gives every linear one pass of 2 groups. The CPU decryption was the dense path: per pass,
`torch._int_mm` of the hint byte planes (`m x 6 x 4096` int8) with the secrets, then the native combine.
Local profile (4 vCPUs with AMX, 0.5B shapes): the GEMM is about half and the combine (6 int32 planes per
element, shifted and summed in 64 bits, subtracted, rounded, transposed) the other half. With a larger pass
(16 groups) the dense cost per slot does not fall: it is compute-bound, `6 x 4096` MACs per row and slot.

## 2. Fused NTT decryption (`decrypt = "ntt"`, and `"auto"` by default)

The hint row `H_c = W8_c A` is, like `A`, a module element of `rank` polynomials, so the correction
`H S` of one row and group is `sum_l h_l * s_l` in `Z[X]/(X^256 + 1)`: `rank` pointwise products per
coefficient after forward NTTs, one inverse NTT, and CRT over three ~2^29 primes (the existing `pgw_ntt`
kernels). The new native `decrypt` runs the same product block by block (16 hint rows) and, instead of
writing `H S` out, finishes decryption in place: for every reply column `c` with output slot `s`,
`out[s, row] = ((y[row, c] - (H S)[row, c] + 2^(delta-1)) >> delta) & (2^K - 1)`. Several replies of one
pass are read in place (their columns concatenate), slots may be skipped (padding), and the hint can be
the stored planes (`Big::Planes`, forward NTTs per pass, no extra memory) or a hint kept in NTT form
(`Big::Pre`, `decrypt = "ntt_pre"` within `hint_ntt_bytes`: 12 bytes per coefficient, twice the stored hint).

**Exactness.** `|H S| <= (2^56 - 1) * L1(s) <= 2^68` (`2^76` at q = 2^64) is inside the CRT range
`p0 p1 (p2 - 1) / 2 ~ 2^86`, checked natively on every call, so `H S mod 2^64` is the dense product bit for
bit and the subtraction, rounding and masking are the dense path's own formula. Tested: native
`decrypt_matches_product` (planes with a row offset, u64, pre-transformed, every instruction set, skipped
columns, two replies), `test_decrypt_into_matches_dense` (q = 2^56 and 2^64, partial groups), and the whole
factory output equal to local generation in every mode (`test_factory_output_identical_in_every_mode`).
Every decrypted row is in addition verified by the factory's every-row Freivalds self-check, so a wrong
decryption could never reach the store.

**Default on, with a self-check and a fallback.** `auto` decides once per pass size (number of groups) on
public random data (256 hint rows, random secrets and replies): it runs the dense and the NTT decryption,
compares their outputs, and uses NTT only when the bits agree **and** it was faster; else dense. Any error
of a native NTT call (exactness bound, shape, a missing symbol) logs, switches the backend to dense for
good, and redoes the pass densely. `decrypt = "dense"` restores the old behaviour exactly.

**Secret hygiene.** The correction never leaves native scratch memory (the CRT buffer is wiped with volatile
stores after each block; the NTT-domain products of every worker are wiped when it ends). The compact
secrets copied for the call are zeroed. The plaintext `W8 r` goes into the caller's buffer as before.

## 3. Masks of local linears (`skip_linear_kinds`, opt-in)

With `local_linears = "attention"` the gateway computes qkv and o itself on every step, so their masks
(16% of the mask bytes and of the decryption at 0.5B, 24% of the encryption) are made and never used.

* **Factory** (`PGW_FACTORY_SKIP_LINEAR_KINDS=qkv,o`, default empty): the generator leaves those linears
  out; the segment header lists only the generated linears. The other linears' masks are unchanged
  (`r`, `u` derive from the seed and the linear id alone; tested bit for bit). Refused by the embedded
  factory (its in-memory rings have a fixed layout) and for every kind at once.
* **Gateway** (always on): at start-up, after the local-linear decision, it declares to the disk store the
  linears it may send to the service (`mask_linears_needed`: every linear, minus the local kinds only when
  `local_max_rows` is unset, since otherwise large steps send them too). From public configuration only.
* **Store** (`MaskStore.require_linears`): active segments whose header lacks a declared linear are
  **retired**: their remaining slots are written off (cursor to the end, status consumed, counted as burned,
  shredded like any consumed segment), at declaration time and again whenever `reserve` meets one (a
  misconfigured factory). They are never handed out. Reservations may span segments with different linear
  sets (each extent reads a linear at its own header index); a reservation covers the linears all its extents
  have, and taking any other raises (`ConfigError`), never a short or reused mask.

Restart scenarios: local attention with old full segments -> full segments are used (their qkv/o rows are
burned on close as before). Local linears turned off while partial segments are stocked -> they are retired
at start-up, the inventory drops, the factory (with the setting unset) refills; requests fail with "mask
exhausted" in between rather than run without masks (`test_local_attention_serves_from_segments_without_qkv_o`).
The declaration is per process (not persisted), so every start-up re-applies the current configuration.

## 4. Privacy

Nothing the provider sees depends on data. Decryption is purely gateway-local: the ciphertexts, request
sizes, request count and order are unchanged; only the gateway's own speed changes (its inter-request
gaps were never data-dependent). With `skip_linear_kinds` the factory sends no maskgen requests for qkv/o:
a function of public configuration (the provider already sees online that qkv/o never arrive with local
attention). Retired segments only burn masks; no mask is ever used twice.

## 5. Numbers

Local, this 4-vCPU dev box (Sapphire Rapids-class with AMX, shared and noisy; `--max-rows 2048`), decrypt
only, Qwen2.5-0.5B shapes, **measured**, ms per slot (all linears):

| groups per pass | dense (AMX GEMM + combine) | ntt (fused, stored hint) | ntt_pre (hint in NTT form) |
|---|---|---|---|
| 2 (512-slot segments) | 6.09 | 5.10 | 2.00 |
| 16 (4096-slot chunks) | 7.57 | 1.86 | 1.54 |

Same box, real factory loop (`gateway_gpu_bench.py --mode he`, 1 layer of 0.5B, `encrypt = ntt_pre`, CPU
gateway, provider in process **on the same CPU**), **measured**: 4096-slot chunk, decrypt 2.15 s -> 0.54 s and
wall 6.6 s -> 3.2 s (`dense` -> `auto`); 512-slot chunk, decrypt 0.45 s -> 0.30 s.

Kaggle, **measured**, CPU-only machine (4 vCPUs, Intel Xeon @ 2.20 GHz, no fast int8 engine, so `dense` is the
`cpu_f64` engine's product): the `pgw-he-decrypt-m3` steps run as a CPU-only variant (`pgw-he-decrypt-m3-cpu`,
provider in process on the same CPU) because the account's two GPU sessions were taken for over an hour.
Raw output: `benchmarks/results/kaggle/he_decrypt_m3_cpu/`. Exactness tests there: 46 passed. All paths
bit-identical in the bench.

Decrypt only, ms per slot (all linears):

| model, groups per pass | dense | ntt | ntt_pre |
|---|---|---|---|
| 0.5B, 2 | 33.4 | 26.7 | 8.2 |
| 0.5B, 16 | 16.5 | 8.2 | 5.9 |
| 7B, 2 | 161 | 123 | 37.7 |
| 7B, 16 | 75.2 | 37.7 | 27.2 |

Factory loop, 4 layers of 0.5B (scaled to 24), `encrypt = ntt_pre`, provider **on the same 4 CPUs** (so the
wall time is dominated by the CPU provider and is not a deployment number; the decrypt seconds are the point):

| chunk | decrypt mode | decrypt s | wall s | slots/s (full model, CPU provider) |
|---|---|---|---|---|
| 512 | dense (before) | 3.53 | 8.29 | 10.3 |
| 512 | auto (-> ntt) | 2.93 | 7.77 | 11.0 |
| 512 | ntt_pre (4 GB) | 0.93 | 6.05 | 14.1 |
| 4096 | dense (before) | 11.7 | 44.8 | 15.2 |
| 4096 | auto (-> ntt) | 6.26 | 39.5 | 17.3 |
| 4096 | ntt_pre (4 GB) | 4.63 | 37.7 | 18.1 |
| 4096 | auto + skip qkv,o | 4.87 | 32.9 | 20.8 |

GPU (T4 provider) end-to-end numbers: not measured (GPU session cap); the GPU job is `pgw-he-decrypt-m3`.

**Estimate for the Modal line** (512 slots: encrypt 1.61, provider 9.40, decrypt 4.12 s; serial): decrypt
falls by 1.2x (`auto`) to about 3.4 s, or to about 1.2 s with hints in NTT form (needs about 15 GB of RAM at
0.5B; the demo container has 8 GB), and with 4096-slot passes by 2-4x more per slot. Because the stages are
serial, the slot rate gains at most decrypt's share (25%): about 1.05x (`auto`) to 1.25x (`ntt_pre`) at 512
slots. The 5x target needs the provider round trip and the serial stage order fixed (other packages).

## 6. What is left

* The 2-group passes of the store's ramp segments pay the hint's forward NTTs per pass (about two thirds of
  the NTT decryption at 2 groups). Larger passes (bigger first segments, or several segments per pass)
  or `ntt_pre` on hosts with the RAM remove it.
* The NTT kernels use Montgomery multiplication on even/odd lanes (`vpmuludq`); a Shoup-style or IFMA kernel
  would cut the butterflies further. Not done.
* Encryption, the provider round trip and decryption still run one after the other per linear (the pipeline
  overlaps only requests of the same linear); overlapping linears is the other packages' work.
* `skip_linear_kinds` is not wired into `pgw.gateway.setup`'s wizard (the Modal demo uses `local_linears =
  auto`, which depends on a start-up RTT measurement and therefore must keep full segments).
