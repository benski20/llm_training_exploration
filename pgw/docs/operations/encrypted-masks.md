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

Network is the remaining cost at volume. On the 32-bit ring it is about 12.8 MB per 7B token for
mask supply plus 8.9 MB for the masked calls themselves.

Planned: the K32 ring (less network and hint), a smaller rank once the lattice estimate allows it,
and moving hint building to the provider with Freivalds verification (faster setup for 70B).
