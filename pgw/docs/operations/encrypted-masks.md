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
reused until the model pack changes.

| model | hint (rank 16) | equals today's masks for | build time (4-core CPU) |
|---|---|---|---|
| Qwen2.5-0.5B | 9.3 GB (measured) | about 6,100 tokens | about 2-3 min (measured, incl. the first 256 masks) |
| Qwen2.5-7B | about 45 GB | about 6,600 tokens | tens of minutes (estimate) |
| Llama-3-70B | about 220 GB | about 6,600 tokens | hours on CPU (estimate); can be built once per model version |

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

## Costs (Qwen2.5-0.5B, this 4-vCPU container, measured)

- One encrypted group (256 slots) for the largest 0.5B linear: encrypt 0.20 s, service 0.23 s
  (CPU), decrypt 1.05 s.
- Refill daemon at nice 19, 1 thread, 50% duty cycle: 256 slots in 256 s wall (127 s compute).
- Network per 0.5B token: about 1.5 MB up and 2.4 MB down (64-bit coefficients).

Planned speedups: AMX/VNNI int8 decryption, a GPU service path (the CUDA `mulmod64` kernel is
written but not yet validated on a GPU), 44-bit download truncation, and a smaller rank once the
lattice estimate allows it.
