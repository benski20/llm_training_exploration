# Encrypted mask generation (no mask factory GPU, no large mask store)

In this mode the customer's mask factory never multiplies masks by the model weights itself.
It sends the masks **encrypted** (module-LWE) to the compute service. The service multiplies the
ciphertexts by the int8 weights, and the factory decrypts `W8 r`. The service learns nothing
about the masks. Its view is a stream of MLWE samples, which are pseudorandom under a standard
post-quantum lattice assumption. No second provider or partner is involved.

Everything downstream is unchanged: the mask store, the gateway, the online protocol and its
Freivalds checks. Masks are made just in time by the refill daemon, so the store only needs to
hold minutes to hours of traffic instead of days.

## Turning it on

```bash
pgw factory run --pack /packs/qwen2.5-7b --store /var/lib/pgw/masks --key-file /etc/pgw/kek \
    --he-service https://compute.example.net:8443 --he-token-file /etc/pgw/service_token \
    --he-hint-dir /var/lib/pgw/hemask --low-water 2000 --target 8000 --chunk-slots 1024
```

The same settings are available as `PGW_FACTORY_HE_SERVICE_URL`, `PGW_FACTORY_HE_TOKEN_FILE`,
`PGW_FACTORY_HE_CA_FILE`, `PGW_FACTORY_HE_HINT_DIR`, `PGW_FACTORY_HE_RANK` (default 16),
`PGW_FACTORY_HE_GROUPS_PER_CALL` (default 4) and `PGW_FACTORY_HE_TIMEOUT_S`. The client demo
takes `--mask-source he`.

## The one-time hint

On first use, the factory builds `H = W8 A mod 2^64` for every linear from the pack and a fresh
public seed it chooses itself. It does this locally, so no provider input is trusted. The hint is
stored as int8 byte planes of bytes 1-7. Byte 0 is not needed for decryption, and its absence is
counted in the noise budget. The hint is reused until the model pack changes.

| model | hint (rank 16) | build: gateway T4 / AMX CPU (4 vCPU) / CPU without fast int8 |
|---|---|---|
| Qwen2.5-0.5B | 8.7 GB | 54 s / about 90 s / about 2 min |
| Qwen2.5-7B | 39.9 GB | 6.5 min / 10.5 min / 26 min |
| Llama-3-70B | about 190 GB (estimate) | about 1 h on a gateway GPU (estimate) |

After that, storage no longer grows with token volume.

## Security properties

- **Privacy:** decision-MLWE with ring degree 256, rank 16 (dimension 4096), q = 2^64, a ternary
  secret and centred-binomial noise (η = 21, σ ≈ 3.2). This sits inside the HomomorphicEncryption.org
  standard's 128-bit table (log q ≤ 109 at n = 4096). **The parameters must still be confirmed
  with the lattice estimator before production use.** The secret and noise are fresh for every
  256-slot group and come from the OS CSPRNG.
- **Public matrix:** its seed is always chosen by the customer. A provider-chosen matrix could
  expose the masks.
- **Integrity:** every decrypted row is verified before it is stored, using private check
  vectors that are never written to the segment. A provider that returns a wrong product causes
  the segment to be rejected, and it cannot learn the online check vectors from those rejections.
  Online, the existing Freivalds check and range check apply unchanged.
- **Metadata the provider sees:** linear ids, group counts and timing of mask generation.
  Generation runs ahead of use, so this reveals at most the volume of masks being stocked.

## Costs per token (measured)

The provider multiplies on its GPU; on a T4 this is exact and fast. The gateway's side (encrypt
and decrypt) is two GEMMs with a small ternary secret matrix. `pgw.hemask.fastmm` runs them as
int8 GEMMs over byte planes on the fastest available engine: a GPU on the gateway, a CPU with
AMX/VNNI, or exact float64 on other CPUs. Override with `PGW_HEMASK_ENGINE`.

| Qwen2.5-7B, per token | before optimisation | gateway T4 (`cuda`) | AMX CPU, 4 vCPU (`cpu_int8`) | CPU without fast int8 (`cpu_f64`) |
|---|---|---|---|---|
| gateway encrypt + decrypt | 1.18 s | **0.11 s** (0.069 + 0.042) | **0.116 s** (0.051 + 0.065) | 0.88 s (0.39 + 0.49) |
| provider GPU (T4) | 0.020 s | 0.019 s | | |

| Qwen2.5-7B, per token | 40-bit ring (q = 2^64) | 32-bit ring (q = 2^56) |
|---|---|---|
| mask-supply upload / download | 6.7 / 8.3 MB | **5.8 / 7.0 MB** |
| one-time hint | 39.9 GB | **34.2 GB** |
| online traffic (masked calls) | 4.2 / 7.0 MB | **3.3 / 5.6 MB** |
| gateway CPU and provider GPU | 0.11 s / 0.019 s | 0.11 s / 0.019 s |

Results are in `benchmarks/results/kaggle/run11_k32/`. All results are exact, and all GPU tests pass.

Qwen2.5-0.5B: 0.026 s per token with a gateway T4, 0.031 s on the AMX CPU.

At 1M tokens/day of 7B, the gateway needs about 1.3 AMX cores, or a share of one small GPU.

Network is the remaining cost at volume. On the 32-bit ring, mask supply takes about 12.8 MB per
7B token, or 11.6 MB with [wire v2](#wire-format-v2-fewer-bytes-per-slot). The masked calls
themselves add 8.9 MB.

**Network caveat: the supply is link-bound on slow links.** Every slot crosses the link once for
supply (12.8 MB for 7B) before it crosses it again in the masked calls. Over the public Modal tunnel
of the two-party demo (`deploy/modal`, about 1.5 MB/s measured) that is about 420 slots per hour for
7B and about 1,900 for Qwen2.5-0.5B (derived): enough to fill a 256-slot first segment for a demo, but
**the encrypted supply cannot keep up with even one decoding stream over that tunnel** (one 7B stream
at 0.365 s/token consumes about 9.9k slots per hour). For sustained load, put the factory and the
service on a private link of 1 Gbit/s or more (about 35k 7B slots per hour, estimate), or run a local
factory. `pgw doctor --role factory --pack <pack> --mask-supply` times one encrypted chunk over the real link and
compares the rate with `PGW_DOCTOR_EXPECTED_TOKENS_PER_S`. Offline, the calculator gives the same
figures: `python docs/tools/pgw_sizing.py --model qwen2.5-7b --encrypted --link-mb-per-s 1.5`. See also
[sizing.md](sizing.md#one-model-one-ring-qwen25-7b-at-k32a12c27).

## Wire format v2 (fewer bytes per slot)

`POST /pgw/v1/maskgen` has a second message version (SPEC 7.8). It sends only the bits the
customer needs:

* **Reply: `K + 3` bits per element instead of `q − 16`.** The provider rounds its own result
  `y = W8 b` to the bits that decryption reads, plus three guard bits. That gives 35 bits on the
  K32 ring (v1: 40) and 43 on K40 (v1: 48).
* **Upload: `q − 3` bits per coefficient** (53 at q = 2^56). The customer switches its ciphertext
  down with randomised rounding, `b' = floor((b + d) / 8)`, where `d` is fresh and uniform in
  [0, 8). The scaling factor and the hint stay the same.

Both operands are bit-packed into fields of 32, 16 and 8 bits plus a sub-byte tail. Each field is a
plain little-endian array.

| Qwen2.5-7B, per slot (Σn = 831,488, Σm = 1,390,592) | v1 | v2 | change |
|---|---|---|---|
| K32 (q = 2^56): upload / download | 5.82 / 6.95 MB (7 + 5 bytes) | **5.51 / 6.08 MB** (53 + 35 bits) | **12.77 → 11.59 MB, −9.2%** |
| K40 (q = 2^64): upload / download | 6.65 / 8.34 MB (8 + 6 bytes) | 6.34 / 7.47 MB (61 + 43 bits) | 14.99 → 13.81 MB, −7.9% |

Without the 53-bit upload (`upload_switch=False`), K32 is 11.90 MB (−6.8%). msgpack framing adds
under 200 bytes per request.

**Negotiation.** v1 is always served and is the fallback. The service advertises the versions it
serves (`vs`) on every reply. The factory advertises them on its requests and moves to v2 only
after a reply has advertised it. If a v2 request is refused, for example during a rolling deploy
or downgrade, the factory resends the same ciphertext as v1 and stays on v1 for 5 minutes. To
turn v2 off on the service, use its maskgen v2 environment switch (SPEC 7.8). The client enables v2 only when it
is given the encryption parameters and the hint store's row norms:
`RemoteMulmod64(..., params=HEParams(k, rank), row_norms=hints.meta["row_norms"])`.
**Status:** the factory runtime (`pgw.factory.runtime.build_he_backend`) does not pass them yet. It
needs one line, which the factory package owns. Until that change lands, the factory keeps
sending v1.

**Noise budget and refusal.** For each linear, the client chooses the smallest format whose
decryption error bound stays below a quarter of the scaling factor. That is half of Δ, with one
more bit of margin:

```
24·σ·‖w‖  +  24·4·‖w‖ (53-bit upload)  +  2^(q−yb−1) (reply rounding)  +  255·dim (hint's low byte)  <  2^(q−K−2)
```

The client checks this at startup for every linear and again before every v2 request. Linears that
do not fit, or have no recorded norm, stay on v1. They are logged at startup. For Qwen2.5-7B at
K32 and rank 16, the bound comes to about 2^21.3 (hidden-size inputs) to 2^21.6 (down projection)
against 2^22. These figures are estimated from Qwen2.5-1.5B int8 row norms, so every linear uses
53/35. Even worst-case rows (every |w| = 127) fit, with 56/35 or 53/39 on the down projection.

**Why the rounding leaks nothing.**
- The provider's rounding is a deterministic function of a value the provider computes anyway.
  It only gives the customer less.
- The customer's upload switch is a randomised function of the ciphertext alone. Its randomness
  is independent of the secret, the noise, the masks and the weights. So `b'` tells the provider
  nothing that `b` would not, and `b` is an MLWE sample at q = 2^56. Security does not rest on a
  smaller modulus.
- Correctness: `8·b' = b − ε`, where each `ε` takes two values 8 apart with mean exactly 0 given
  `b`, independently per coefficient. Hoeffding bounds `|W8 ε|` by `24·4·‖w‖`, with failure
  probability `2·e^(−288)`. That is the same tail as the `W8 e` noise term.
- Decrypted masks are exact and identical to v1 for the same randomness. The tests compare v1 and
  v2 end to end at K32, rank 16, q = 2^56, on 7B-shaped linears. The every-row check and the
  online Freivalds checks do not change.
- v2 is not the rejected 25-bit upload variant, which redefined the mask from a public value and
  leaked noise. It also does not ship low-bit `W·r` downloads, which would need a new Freivalds
  soundness argument.

**CPU cost.** In this release, packing and unpacking run in numpy on both sides. The measurements
below come from a shared 4-vCPU host, single-threaded per call, so they are noisy.

| | per 7B slot |
|---|---|
| customer, extra over v1 | ≈ +7 ms (v1 ≈ 12 ms, v2 ≈ 19 ms) |
| provider, v1 | 0 ms (raw bytes go to the device) |
| provider, v2 (CUDA or CPU kernel) | ≈ 20-25 ms |

On a T4 provider, the v2 numpy work is therefore larger than the GEMM itself (2-4 ms). v2 pays off
where the link is the limit, which is any link of 1 Gbit/s or less: about 100 ms per slot on the
wire. On a co-located link of 10 Gbit/s or more, keep v1 by turning v2 off on the service. The
other option is to move the v2 unpack and pack into the device kernel (`ComputeKernel`) or
`pgw_fast`. That is a follow-up: the format is plain field casts, so it maps directly to a GPU
kernel.

Planned: the K32 ring (less network and hint), a smaller rank once the lattice estimate allows it,
and moving hint building to the provider with Freivalds verification (faster setup for 70B).
