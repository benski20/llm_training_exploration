# pks-v3 client disk: can the factory pack go? (no: W8 is consumed per segment, not per rejection)

Date: 2026-11-26. Scope: the SDK client's steady-state cache (`<cache>/models/<fp16>/`: weight-free gateway pack,
factory pack, pks-v3 hint) with the default `mask_scheme="pks-v3"`, masked head, p31 segment checks. Question: can the
factory pack (int8 `W8`, ~1 byte per parameter) be replaced by a small stock of secret check-vector sets drawn at
setup, with a fingerprint-checked re-provision when the stock runs out?

**Verdict: the hypothesis does not hold, so nothing was implemented on the factory.** A stock of pre-drawn sets would
be exactly as secret and as sound as on-demand draws (section 3), so no property needs weakening; but the
premise that only rejections consume sets is wrong. The factory needs `W8` for every segment it produces (the
segment's own Freivalds vectors, `uW = u W8 mod p`, from a fresh seed per segment, SPEC 7.3 / 7.6), and its private
self-check vectors are drawn again at every worker start and after every aborted segment, transport errors
included. A stock sized against the halt rule (3 + margin) would be used up after a few segments, about a few
thousand tokens. A stock as large as the factory pack would last ~2-4 M token rows before the client has to
download all of `W8` again. That path would be routine, not rare. Section 5 lists the levers that exist and what
they would cost. One duplicate copy was found and removed: the plain tensors on the `source="hf"` path.

## 1. Every use of `W8` after setup (evidence)

| user | what it reads | when | file |
|---|---|---|---|
| factory, per segment: segment check vectors | `uW = u W8 mod p` (p31, 1 vector) or `u W8 mod 2^K` (ring), `u` from the segment seed | **every segment, every linear** | `factory/generator.py` `generate()` (`prime_precompute(u, w)` / `ring_vecmat(u, w)`) |
| factory: private self-check | `v W8` for `c = 14` uniform vectors per linear (40 bits against chosen errors) | each generator (= each worker process) start; after **every** failed or abandoned segment (`_wipe_private` in `generate()`'s `except BaseException`); every segment for linears that do not fit the 1 GiB `PRIVATE_CACHE_BYTES` (72B needs 1.18 GiB) | `generator.py` `_private_check`, `_wipe_private` |
| factory: weight streamer | iterates every linear's `W8` per segment (feeds the two rows above; `_RemoteKernel` itself ignores it) | every segment | `generator.py` `self.streamer.iter_device_weights()` |
| factory start | `ModelPack.open(verify="full")`: hashes the whole factory pack against the pinned manifest | every factory child start (= every SDK process that owns the factory) | `factory/runtime.py` `open_packs` |
| masked head (factory side) | head `W8` derived from the plain head (`quantize_head`) | every factory start | `modelpack/masked_head.py` `_head` |
| masked head (gateway side) | scales and row norm derived from the plain head; no `W8` kept | every gateway start | `masked_head.py` `load_linear_meta` |
| hint store | not `W8`: `row_norms` and the noise check come from the HMAC'd `meta.json`; decryption uses the hint | per segment | `hemask/pks_backend.py` |
| hint (re)build | `W8` | only if the store is missing (`open_or_build`) | `factory/runtime.py` `_build_pks_backend` |
| gateway, local linears, `pgw sdk doctor`, prepare/resume | gateway: never (`GatewayPack`, `load_linear` raises); doctor: no; prepare: setup only | n/a | `provision/pack.py`, `sdk/cli.py`, `sdk/prepare.py` |

`pks_backend.PKSMaskBackend.matmul` itself never reads `W8` (the provider computes `W8 r` under encryption, the
client decrypts with the hint). Opt-in paths (mlwe `HEMaskBackend`, fused2) read it the same way or more.

## 2. What consumes a check-vector set

A "set" here is everything the factory derives from `W8` for one use:

* **Segment set**: a fresh 32-byte seed (`os.urandom`; it also derives the segment's `r`) and `uW` of every linear.
  SPEC 7.3 makes `u` a normative function of the segment seed (the gateway regenerates it), and 7.6 makes the seed
  unique per segment. **One set per segment, always.** The pks-v3 path produces a segment per `mask_segment_slots`
  (default 1024) slots, and one slot is one token row (prompt or generated).
* **Private set**: `v W8` for 14 vectors per linear (`v` itself can be regenerated from a 32-byte seed). Consumed
  at each worker start (`mask_workers`, default a quarter of the CPUs, up to 8; the factory child is restarted with
  every SDK process that owns it). Also consumed by each aborted segment: a provider integrity failure (at most 3,
  then the halt), any transport error, timeout or rate limit (retried with back-off, unbounded,
  `scheduler.is_provider_integrity_failure`), SIGTERM at application exit while a segment is in flight, and a crash.
  At 72B it is also consumed every segment for the linears beyond the 1 GiB in-RAM cache.

Sizes (computed from the real shapes; `benchmarks`-free script, K32A12C27, `c = 14` everywhere):

| | 0.5B | 7B | 32B | 72B |
|---|---|---|---|---|
| segment set (`uW`, p31, 4 B per input feature) | 0.73 MB | 3.34 MB | 11.0 MB | 17.4 MB |
| private set (`v W8`, uint64 `(n, 14)`) | 20 MB | 94 MB | 309 MB | 486 MB |
| factory pack `W8` (layers; the head comes from the plain head) | 0.36 GB (measured) | 6.53 GB | 31.2 GB | 70.2 GB |
| segments a pack-sized stock holds (ignoring private sets) | ~490 | ~1950 | ~2840 | ~4030 |
| = token rows before a full `W8` re-download | ~0.5 M | ~2.0 M | ~2.9 M | ~4.1 M |

(SPEC 5.1 quotes the same 3.3 MB of `uW` per segment at 7B.) Sized against the halt rule (3 + 5 = 8 sets), the
stock would last 8 segments, about 8 K token rows, or fewer: each application start and each network blip uses a
private set. A stock of 1 GB at 7B is ~250 segments (~250 K rows, ~3 h of continuous decode at 25 tok/s) before a
6.5 GB re-download. That is a disk-for-bandwidth trade, and the re-provision path would run in normal operation.

## 3. Would a pre-drawn stock be as secure? (yes: the plan fails on size only)

For the record, since this is what would make a bounded variant acceptable:

* **Same distribution.** A segment set is `(seed, uW(seed))` with `seed = os.urandom(32)` drawn at setup instead of
  at segment start. The segment then uses that seed for `r` and `u` exactly as SPEC 7.3 says. The bytes are
  identical to an on-demand segment with the same seed, and a private set `(v_seed, v W8)` likewise. Nothing about
  the per-segment check changes: the gateway's p31 soundness argument (SPEC 5.1) needs `u` uniform and independent
  of what the provider sees, which a seed drawn earlier and kept secret satisfies.
* **Secrecy at rest.** Seeds and `uW` already live on disk inside every mask-store segment (chunk 0, AES-GCM under
  a per-segment DEK wrapped by the store KEK, `maskstore/format.py`). A stock would use the same envelope, plus
  0600 and an HMAC'd meta under the cache's hint key with its own domain label. The hint key authenticates and
  does not encrypt, so encryption needs the KEK. The private `v W8` sets are new at rest: today they never leave
  the factory's RAM. They would be secret material on disk for the first time, under the KEK.
* **Single use and crash safety.** This needs a durable "taken" record per set, written before the set is used
  (fsync), so that a crash can never hand out a set again. For private sets, a durable "in flight" mark that
  burns the set on restart. Both are the store's existing counter / reservation pattern.

None of this weakens a property. It just does not save disk at a useful re-provision rate.

## 4. On-disk breakdown, measured at 0.5B and extrapolated

Measured: `provision(source="hf", hint="local", masked_head=True, factory_pack=True)` of Qwen2.5-0.5B-Instruct (plain
float32, as pinned), files listed and `du` of the cache. Extrapolated per model from the same layout (plain
float32 embeddings + untied float32 head at 7B+, `W8` 1 B per layer parameter, hint 1.54 KB per output row).

| file | needed after setup by | 0.5B measured | 7B | 32B | 72B |
|---|---|---|---|---|---|
| `pack/plain.safetensors`: embeddings (f32) | gateway, every token | 0.545 GB (tied: also the head) | 2.18 | 3.11 | 4.98 |
| same file: untied float head (f32) | gateway start (scales + row norm of the masked head), factory start (head `W8`) | (tied) | 2.18 | 3.11 | 4.98 |
| `pack/linear-meta`, manifest, tokenizer | gateway | 0.013 GB | ~0.02 | ~0.03 | ~0.04 |
| `factory-pack/linears-*.safetensors` (`W8`) | factory, every segment (section 1) | 0.359 GB | 6.53 | 31.2 | 70.2 |
| `factory-pack/plain.safetensors`, tokenizer | factory start (head) | hard link on the provider path; **a second copy on `source="hf"`: 0.557 GB** | 4.36 dup. | 6.23 dup. | 9.97 dup. |
| `hints/*-pks-v3-*` | factory, every segment (decrypt) | 0.700 GB | 2.37 | 7.38 | 10.77 |
| **total, provider path (SDK default)** | | **1.63 GB** | **13.3** | **44.8** | **90.9** |
| **total, `source="hf"` before** | | **2.17 GB (measured)** | 17.6 | 51.0 | 100.9 |
| **total, `source="hf"` after (this change)** | | **1.63 GB (measured, `du`)** | 13.3 | 44.8 | 90.9 |

(`benchmarks/prepare_real_bench.py` reports 2.17 GB for the provider path too: its `_du` counts every directory
entry, so hard-linked files are counted twice. On disk the provider path holds one copy, `link_or_copy` in
`_install_factory_pack`.)

**Change made:** `_install_from_full_pack` (the `source="hf"` path) hard-links a plain-only file into the gateway
pack when the gateway needs the whole file. The full pack it came from is kept as the factory pack, so a copy
stored the embeddings and float head twice. It falls back to the old rewrite when a link is not possible. Both
packs are still verified in full against the pinned fingerprint. Test:
`tests/provision/test_provision_e2e.py::test_public_source_factory_pack_shares_plain_files`.

Nothing else in the gateway pack or the hint can go:

* Embeddings: read every token.
* Hint: read every segment. Lossless compaction was ruled out in `pks-v3-client-setup-scaling.md`, section 3.
* Float head: still read at every gateway and factory start, to derive the masked head's scales and row norm
  and the factory's head `W8`.
* `W8`: measured order-0 entropy 7.06 bits per weight (zlib-9: 88.5 %). Lossless compression would save at
  most ~12 %, at the price of decompressing all of `W8` for every segment.

## 5. Levers that exist (none implemented; each is a decision)

1. **Bounded segment-set stock + routine `W8` re-stream** (security-neutral, section 3): disk `S x` (3.3 / 11 / 17 MB)
   plus private sets, against a full fingerprint-checked `W8` download (6.5 / 31 / 70 GB) every `S` segments and
   more often with restarts. Only worth it for clients with a fast link, a tight disk budget and low volume. It
   needs a stock writer in the fused build, KEK-encrypted single-use files with durable consumption, and factory
   changes in the area the multi-user agent is working in (the generator's `u`/`v` source).
2. **Drop the float head after setup** (7B+, untied heads): keep the int8 head (`W8`, 1/4 of the f32 head) in the
   factory pack and the gateway's head scales + row norm as HMAC'd sidecars. That saves 1.6 / 2.3 / 3.7 GB (12 % /
   5 % / 4 %). The cost is that the gateway pack could no longer be verified in full against the public pin, and
   two derived artifacts would be trusted under the local cache key instead, which is a change to the integrity
   model. It also touches the gateway's start-up path.
3. **A transposed pks-v3 hint** (the only route to no `W8` on the client): get `u W8` and `v W8` from the provider
   under encryption, like `W8 r`, from a hint of `W8^T` (1.54 KB per input feature: 1.28 / 4.24 / 6.67 GB instead
   of `W8`'s 6.5 / 31 / 70 GB). This is a scheme change. It needs mod-p limbs (`u` is 31-bit and products exceed
   `2^32`) and a new soundness argument for checking the check vectors themselves, so it needs a design and an
   external review. It is not a factory-narrow change.
4. **Plain tensors in bfloat16** when the checkpoint is bf16 (widening is exact): halves the 4.4 / 6.2 / 10 GB of
   plain tensors. It changes the pack format and so the fingerprints and pins. It is a provider/pin decision.
