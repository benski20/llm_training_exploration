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
`PGW_FACTORY_HE_GROUPS_PER_CALL` (default 4), `PGW_FACTORY_HE_MAX_INFLIGHT` (provider requests at
once, default 2) and `PGW_FACTORY_HE_TIMEOUT_S`. The client demo takes `--mask-source he`.

`PGW_FACTORY_HE_PIPELINE_DEPTH` (default 0) overlaps the factory's stages across chunks and linears:
with depth `d`, up to `d` chunks are outstanding, so the gateway encrypts the next linear's masks and
decrypts the previous one's while the provider works on the current one. With depth 0 the stages of a
chunk run in series, and a small segment (two 256-slot groups per linear, one request) never overlaps
them. The provider receives the same requests (same ciphertexts, sizes and counts); only their start
times move, and those follow from shape-dependent stage times. The decrypted masks are bit-identical and
every row is still self-checked. `benchmarks/he_factory_overlap_bench.py` measures serial against
overlapped configurations against a real service. Measured on a Kaggle T4 VM (4 vCPUs shared by
gateway and service, 24 layers of Qwen2.5-0.5B shapes, `ntt_pre`): 512-slot segments 21.1 slots/s serial,
26.4 at depth 4 with `PGW_FACTORY_HE_MAX_INFLIGHT=4`, 28.6 at depth 8 / inflight 4 (1.36x); depth 2 or 4
with inflight 2 was *slower* (17.4 slots/s). 4096-slot segments: 20.0 -> 25.4 slots/s (depth 4 /
inflight 4). All configurations were bit-identical to the plaintext factory. Keep depth 0 unless the
gateway has spare cores, and pair a non-zero depth with inflight 4.

`PGW_FACTORY_HE_ENCRYPT=ntt_pre` (default `dense`) keeps the public matrix in NTT form and encrypts
about 10x faster on a CPU than the dense path on a T4, with bit-identical ciphertexts (needs the
native `pgw_ntt` module). Ranks below 16 are refused; the unreviewed rank-12 profile needs
`PGW_FACTORY_HE_PARAMS_ACK` set to its acknowledgement string. See
`research/plans/he-structured-encryption.md`. With a gateway GPU, `ntt_pre` now decrypts as device passes
on that GPU (self-tested at start; the CPU engines otherwise).

`PGW_FACTORY_HE_WIRE_V2=true` (default off) lets the maskgen client use wire v2 (randomised modulus switch
to `q - 3` upload bits, `K + 3` reply bits, noise budget planned per linear from the hint store's row
norms, v1 fallback). Same decrypted masks; at 7B on a Kaggle T4x2 it cut the bytes by 11% and raised the
whole factory from 5.5 to 6.4 slots/s (`benchmarks/he_e2e_bench.py`, `benchmarks/results/kaggle/x3_he_e2e_7b_r2`).

Device hint cache (gateway GPU): hints used in the last `PGW_HEMASK_GPU_HINT_STALE_S` seconds (default 600)
are not evicted; a hint that does not fit is streamed with the next block prefetched
(`PGW_HEMASK_HINT_PREFETCH=0` turns that off). A decryption pass that runs out of device memory drops cached
hints and reruns (same bits).

Decryption on a CPU gateway: `PGW_FACTORY_HE_DECRYPT` (default `auto`) picks, per decryption-pass size,
the faster of the byte-plane GEMM (`dense`) and the fused native NTT decryption (`ntt`), measured at
start-up on public random data and used only when both give the same bits; a native error falls back to
`dense`. `PGW_FACTORY_HE_HINT_NTT_GB` (default 0) keeps the hints of as many linears as fit in NTT form in
RAM (`ntt_pre`: 12 bytes per coefficient, twice the stored hint, about 15 GB for all of Qwen2.5-0.5B),
another 1.2-2.5x less decryption work. Every mode gives the same bits. Details and numbers:
`research/plans/he-gateway-decrypt.md`.

A gateway whose `local_linears` is fixed to `attention` (and `local_max_rows` unset) never sends qkv or o
to the service, so their masks are never used. `PGW_FACTORY_SKIP_LINEAR_KINDS=qkv,o` (opt-in) makes the
factory skip them (about 16% of the mask bytes and of the decryption at 0.5B). The gateway declares at
start-up which linears it sends; segments lacking one of them are retired (written off, never reserved),
so changing the gateway's setting later is safe: unset the factory setting and it refills with full
segments.
### Several workers and several provider GPUs

One factory process runs encrypt, provider call and decrypt for one segment at a time. To use more
cores and more provider GPUs, run a fleet of worker processes on the same store:

```bash
PGW_FACTORY_WORKERS=4 \
PGW_FACTORY_HE_SERVICE_URLS='["https://compute-0.example.net:8443", "https://compute-1.example.net:8443"]' \
pgw factory run --pack /packs/qwen2.5-7b --store /var/lib/pgw/masks --key-file /etc/pgw/kek ...
```

(`pgw factory run --workers 4`; Helm: `scaleOut.workers`, `encrypted.serviceUrls`.) The workers
coordinate through an in-flight ledger under `<store>/fleet`, so together they stop at the target.
Worker `k` sends every request of a segment to service `k mod len(urls)`; segments are sharded,
never split, so each service sees the same request format and sizes as with one service. A segment
that fails on one service is aborted (its seed is never used) and the worker's next segment goes to
the next service. With fewer workers than services, each worker rotates per segment. The store
serialises segment registration and hand-out across processes, and each writer locks its partial
file, so a restarting worker never deletes a peer's segment in progress. Details and measurements:
`research/plans/factory-scale-out.md`.

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
  vectors that are never written to the segment. A provider can shift a decrypted mask entry by an
  exact amount of its choosing (the reply is additively malleable), so the factory uses enough
  vectors for 40 bits against chosen errors (14 at capped K32 widths, SPEC 7.9), redrawn after any
  rejected segment. A provider that returns a wrong product causes the segment to be rejected, and
  it cannot learn the online check vectors from those rejections. Each rejection is a CRITICAL
  alert; after 3 (`PGW_FACTORY_MAX_PROVIDER_INTEGRITY_FAILURES`) the factory halts until an operator
  has investigated the provider and runs `pgw factory clear-halt --yes`. Segments are written in the
  p31 check format by default (`PGW_FACTORY_CHECK_FIELD=auto`), so the gateway also catches a wrong
  mask online, with the same reaction for every input (the gateway accepts p31 on a trip lane or with `PGW_GATEWAY_PRIME_CHECK=true`; gateways without either, or
  with padded speculation, need `PGW_FACTORY_CHECK_FIELD=ring`).
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

`POST /pgw/v1/maskgen` has a second message version (SPEC 7.9). It sends only the bits the
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
turn v2 off on the service, use its maskgen v2 environment switch (SPEC 7.9). The client enables v2 only when it
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

The client checks this at startup for every linear. Linears that do not fit, or have no recorded
norm, stay on v1. The startup plan is the guard: it uses the row norms from the customer's own hint
store, which is built from `W8` and is not provider-controlled. Before each v2 request the client
re-asserts the stored plan, which is a consistency check rather than independent protection. They are logged at startup. For Qwen2.5-7B at
K32 and rank 16, the bound comes to about 2^21.3 (hidden-size inputs) to 2^21.6 (down projection)
against 2^22. These figures are estimated from Qwen2.5-1.5B int8 row norms, so every linear uses
53/35. Even worst-case rows (every |w| = 127) fit, with 56/35 or 53/39 on the down projection.
For all-±127 down-projection rows (n = 18944) the 53/39 bound is 2^21.98, just under 2^22. That
budget already sits one bit below the decryption limit (failure at 2^23), and the bound is
conservative: simulated all-±127 rows at K32/q56 and K40/q64 reach 2^20.05 against a non-noise
bound of 2^20.76.

**Design status.** The 53-bit upload departs from the original plan (E5 assumed q = 2^53 after
WP7). Here the customer instead rounds its own q = 2^56 ciphertext down with fresh dither. The
argument below was reviewed and holds, but as a new design it still needs explicit sign-off from
the research owner. Until then, `RemoteMulmod64(..., upload_switch=False)` keeps 56-bit uploads
(11.90 MB per 7B slot instead of 11.59 MB).

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

**Provider memory.** The service unpacks a v2 upload into a full `uint64` array plus a scratch
array (and a byte copy on the CPU kernel). For the largest item (n = 18944, cols = 4096) each is
about 0.6 GB, so a v2 item needs roughly 2-3x the host memory of a v1 item while in flight. The
service rejects uploads narrower than 32 bits (`bb < 32`), which caps the unpack expansion at 2x
the request body. Concurrency limits, `cols ≤ 4096` and authentication bound the rest; size
provider host memory for the configured maskgen concurrency accordingly.

**CPU cost.** In this release, packing and unpacking run in numpy on both sides. The measurements
below come from a shared 4-vCPU host, single-threaded per call, so they are noisy.

| | per 7B slot |
|---|---|
| customer, extra over v1 | ≈ +7 ms (v1 ≈ 12 ms, v2 ≈ 19 ms) |
| provider, v1 | 0 ms (raw bytes go to the device) |
| provider, v2 (CUDA or CPU kernel) | ≈ 20-25 ms |

On a T4 provider, the v2 numpy work is therefore larger than the GEMM itself (2-4 ms). v2 pays off
where the link is the limit, which is any link of 1 Gbit/s or less: about 100 ms per slot on the
wire. The service serves v2 by default and the client has no link-speed setting, so on a
co-located link of 10 Gbit/s or more, keep v1 by turning v2 off on the service
(the service's maskgen v2 environment switch, SPEC 7.9); otherwise the provider CPU becomes the bottleneck. The
other option is to move the v2 unpack and pack into the device kernel (`ComputeKernel`) or
`pgw_fast`. That is a follow-up: the format is plain field casts, so it maps directly to a GPU
kernel.

Planned: the K32 ring (less network and hint), a smaller rank once the lattice estimate allows it,
and moving hint building to the provider with Freivalds verification (faster setup for 70B).

## Provider throughput at scale

On CUDA the provider runs each maskgen item end to end on the GPU (`pgw.service.maskgen_gpu`). The
v2 bit-packing stores whole-byte fields as plain arrays, so they are already byte planes. Only the
`bb mod 8` tail bits need decoding, and that happens on the device. The provider runs `ceil(bb/8)`
planes (7 for the default 53-bit upload; the previous path ran 8). A Triton kernel recombines the
planes after the int8 GEMM. Before it serves, it is checked once per device against the torch
reference. The v1 byte cut and the v2 rounding and bit-packing also run on the device. Only the
packed reply (`yb/8` bytes per element) comes back to the host. The request upload is one
multi-threaded pinned copy, and the reply goes out without another host copy. The items of a
request run concurrently, with `maskgen_concurrency` slots for each CUDA device, so layers placed
on several GPUs work in parallel. The output bytes are identical to the previous path. The
`gpu` tests and the benchmark compare them with it and with the CPU `matmul_mod64` reference.

| Switch (service env) | Default | Effect |
|---|---|---|
| `PGW_SERVICE_MASKGEN_FUSED` | on | `0` restores the previous CUDA path (host-side v2 unpack/pack) |
| `PGW_SERVICE_MASKGEN_YIELD_MS` | 0 (off) | Each maskgen block (64 MB of partial sums) waits up to this long while decode rows are queued |
| `PGW_SERVICE_MASKGEN_CONCURRENCY` | 2 per CUDA device | Concurrent items per device |

Measured on Kaggle 2x T4 (4 vCPU) with `benchmarks/maskgen_provider_bench.py`, which runs one layer
of each linear at real shapes and scales by the layer count. Default upload v2 53/35 bits, best
`conc`. Raw results are in `benchmarks/results/kaggle/maskgen_provider_run{1,2}`.

| Model | Before (1 T4) | After, 1 T4 | After, 2 T4 (layers sharded) |
|---|---|---|---|
| Qwen2.5-0.5B | 217 slots/s | 1,910 slots/s (g4) to 2,150 (g16) | 3,840 slots/s (g16) |
| Qwen2.5-7B | 43 slots/s (4.4 TOPS) | 255 slots/s (23 TOPS) | 450 slots/s (41 TOPS) |
| Llama-3-70B shapes | 8.3 slots/s | 23 slots/s (22 TOPS) | 46 slots/s (44 TOPS) |

The same box measured about 28 to 37 TOPS for a bare 8192³ int8 GEMM (the T4 throttles at 70 W).
At 7B and 70B the provider now runs at 70 to 80% of that GEMM ceiling. Per 7B gate_up item at 4
groups: GEMM 48 ms, device pack 21 ms, D2H 13 ms, combine 6 ms (the torch ops took 30 ms), unpack
5 ms.

Decode interference: with saturating maskgen at 7B, the p50/p99 latency of an 8-row qkv decode
GEMM was 12 / 84 ms (alone: 0.7 / 1.1 ms). With `PGW_SERVICE_MASKGEN_YIELD_MS=5` it was
2.0 / 19.7 ms. The yield is opt-in, because it lowers maskgen throughput when decode is busy.

**Projection to 70B dense.** One slot is 6.85e10 MACs per plane (80 layers), or 9.6e11 int8 ops
with 7 planes:

* T4: about 23 slots/s per GPU (measured above).
* A10G: about 150 slots/s, assuming about 150 effective TOPS.
* H100: about 1,200 slots/s, assuming about 1,200 effective dense TOPS.

The wire then becomes the limit. At 70B a slot is 28 MB up and 29 MB down (v2 53/35:
Σn = 4.26M, Σm = 6.72M), so a 100 Gbit/s link carries only about 200 slots/s. Encrypted masks at
70B throughput therefore need the provider and the customer's mask factory close together
(same region, multi-100G), or masks generated ahead of time in off-peak hours. Reducing the
per-slot wire is a cryptographic change. Such a change needs its own reviewed scheme and stays
off by default.

## Bulk maskgen channel (transport, not crypto)

Measured with `benchmarks/maskgen_wire_bench.py` (the real service app, TLS on loopback, a stub provider
that answers at zero cost): the factory's per-pass "provider" time on the HTTP route is almost all
transport and customer-side bit packing. For Qwen2.5-0.5B at 512 slots it was 9.8 s (v1) / 7.8 s (v2)
with **no GPU work at all** (4 shared vCPUs), against 9.40 s measured on Modal with the T4. One slot moves
about 2.5 MB (0.5B) or 11.6 MB (7B, v2), and one asyncio event loop per side (httpx/h11 and uvicorn,
with TLS inside the loop) carries roughly 150-350 MB/s.

The bulk channel carries the same request and reply bodies over blocking TLS sockets, one thread per
connection, so OpenSSL runs with the GIL released and the reply is written straight from the pinned
device-copy buffer:

| Setting | Default | Effect |
|---|---|---|
| service `PGW_SERVICE_MASKGEN_BULK_PORT` | unset (off) | Listener of the bulk channel (TLS with the main listener's certificate and client-certificate policy; plain TCP only on a loopback host without TLS) |
| service `PGW_SERVICE_MASKGEN_BULK_MAX_CONNS` | 16 | Open bulk connections at most (more get one `unavailable` reply) |
| factory `PGW_FACTORY_HE_BULK_PORT` | unset (off) | Use the service's bulk port (host and TLS from `he_service_url`); a transport failure resends the request over HTTP |
| factory `PGW_FACTORY_HE_BULK_SPLIT` | 1 | Send a multi-group request as up to N concurrent whole-group requests |
| factory `PGW_FACTORY_HE_WIRE_V2` | on | Use wire v2 where the noise budget allows (the factory did not pass the parameters before, so it always used v1) |

Nothing new reaches the provider: the bodies are the HTTP route's, and authentication (token, client
certificate, `require_client_cert_for_compute`), admission, draining, tenant limits, wire versions,
the GPU path and its decode yield are the handler's. With `pgw_fast`, the customer's v2 modulus switch
and packing, and the reply unpacking, run natively (byte-identical to numpy for the same dither; the
production dither is ChaCha20 from a fresh `os.urandom` seed, never stored).
