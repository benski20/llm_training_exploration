# PGW release status: v0.1.0

Date: 2026-11-17. This is the release manager's view of what is built, how it was verified, what is
missing, and what has to happen before a real customer pilot. The design is in [SPEC.md](SPEC.md).
The change log between teams is in [INTERFACE_NOTES.md](INTERFACE_NOTES.md). GPU numbers are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

**Verdict.** The protocol is implemented end to end and is exact. It has been verified on CPU here
and on a Kaggle T4 GPU with Qwen2.5-0.5B and Qwen2.5-7B. Security and reliability reviews are folded
in. The product is ready for an **internal or friendly-customer technical pilot in topology B**
(gateway in the customer cloud, same region as the service). It is **not** ready for a production
SLA pilot. Three reasons:

1. Per-token latency is about 3x off target. The cause is round trips and gateway CPU, not the GPU.
2. The HSM, KMS and TPM integrations have never run against real hardware or cloud services.
3. There has been no external security audit.

## Current state (2026-11-26)

How to read this file: the verdict above is the v0.1.0 release view (2026-11-17). Workstreams append dated
sections. The 2026-11-25 sections follow this summary. The 2026-11-19 to 2026-11-23 updates come next, then the
release report's numbered sections (1-6). The **2026-11-26 sections are at the end**, ending with the RunPod
runbook's launch-readiness pass. A note in *italics* marks a statement that a later section superseded.

**What the product is now.** The client library (`pip install pgw-sdk`, `from pgw import OpenAI`) runs the gateway
in the application's process against our provider service. Defaults:
* pks-v3 masks with the hint built locally (`pgw sdk prepare` does it ahead of time);
* the masked LM head on the provider;
* the native TCP+TLS trip lane, with lane format 2: several answers of one step in one call;
* for several answers per step: one synchronous pass, one native lease per answer with batched native prepare /
  finish, sampling on a thread pool, and an exact native `torch.multinomial` (section "client CPU at many concurrent
  streams" at the end); since the second round, the decoder layers of such a step in the native decode driver (one
  native call per stretch of trips, one lane multi call per trip) and trip leases prepared on a background thread
  (the last section);
* the p31 check on every trip;
* cross-process mask admission (`SharedLedger`);
* stall detection of the mask factory (`factory_stall_s`), and the maskgen watchdog for a provider whose GPU stops
  finishing mask generation (`maskgen_stall_s`, 30 s; `ProviderStalledError`; see the last 2026-11-26 section).

Opt-in:
* the client KV prefix cache with the masked LM head (`prefix_cache=True`; cached prompt prefixes take no trips
  rows and no masks, bit-identical; the provider can tell a hit: see the last 2026-11-26 section);
* strict speculation (`speculation="lookup" | "draft_head"`; the shipped heads are trained on commercially usable
  prompt data only: see the last 2026-11-26 section);
* `mask_admission="overcommit"`;
* the weight-free factory (`factory_weights="stream"`).

Pack format 2 (bf16 plain tensors, exact) is the default. Fingerprints for 0.5B, 1.5B, 3B and 7B are baked in
(`PINNED_MODELS`), and for 14B / 32B / 72B ahead of support (usable for measurements only through
`PGW_SDK_PREVIEW_MODELS`, off by default).

On the provider, the pks-v3 hint is recomputed per item on the device. This runs as fused Triton kernels after
bit-exact self-tests, with the 4-prime forward NTT off by default. Since the third round the reply keys are
unpacked and range-checked on the device too, and uploads are staged by a host thread pool overlapped with PCIe
(estimated H100 maskgen 450 / 135 / 81 slots/s at 7B / 32B / 72B, from 264 / 87 / 56; the device wire forms and
keys validated on a Kaggle T4, every item byte-identical). The mask store uses AES-256-GCM on AWS-LC
(same format). The security review's findings S1-S15 and S20 are fixed or closed.

On the client, the mask factory's CPU per slot (one core, every part including the request and reply handling,
1024-slot segments) is 35 / 112 / 166 ms at 7B / 32B / 72B, about 29 / 9 / 6 slots/s per core (was 56 / 199 / 299
counted the same way; last section: upload and reply written in place, AVX-512 keystream for `r`, buffer reuse).

**Verified:**
* CPU end to end at 0.5B, every answer bit-identical to the gateway path;
* the SDK at 7B against a Kaggle T4 provider: all checks pass; a correctness run, not a speed run;
* the pks-v3 device item and the fused kernels on a T4: bit-identical; memplan calibrated on two T4 runs;
* soak: 31 min with injected faults, 0 errors.

**Not yet measured on target hardware.** The H100 numbers below are modelled or estimated, not measured:
* provider memory at 7B/32B/72B;
* per-trip device time;
* the lane's fixed cost over a real NIC;
* pks-v3 maskgen throughput;
* the SDK's TTFT, tok/s and trips per token against a remote GPU provider;
* 32B and 72B at all.

`deploy/runpod/RUNBOOK.md` measures all of these in four sessions under $60 (`deploy/runpod/jobs.py`; $39.74 at
full TTL, ~$31-36 stopped when done), launch-ready since the last 2026-11-26 section: 32B runs through the
measurement-only preview opt-in `PGW_SDK_PREVIEW_MODELS` (pinned models only, still pin-checked; `SUPPORTED_MODELS`
unchanged), the shipped draft head is used by default, and the day's provider device paths, the maskgen watchdog's
bound, the client CPU rounds, the prefix cache and the client mask cost per server core have stages of their own.

**Open:**
* the external cryptographic review of pks-v3 and the external audit of the Rust `unsafe` code;
* whether 14B / 32B / 72B join `SUPPORTED_MODELS` (a product decision; their pins and the shard-wise packer ship);
* a draft head for 32B / 72B (0.5B and 7B ship);
* provider images rebuilt and pushed with torch 2.15.0;
* HSM/KMS/TPM never run on real services;
* the per-token latency target. The model gives 7B at about 17 tok/s at RTT 0.2 ms without speculation; to be
  confirmed by the RunPod sessions.

## 2026-11-25 — Client library integrated: one product, proven end to end on CPU (real Qwen2.5-0.5B-Instruct)

The five parts of the client library (SDK, provisioning, provider accounts, console, packaging) were built against
stubs of each other; they now run as one product (`docs/sdk/CONTRACT.md`, appendix updated):
- **One SDK key codec**: `pgw.enroll.sdk_key` (stdlib, published test vector); `pgw.sdk.keys.parse_key` wraps it
  (pasted-key leniency only). The SDK's 5-byte-checksum format is gone.
- **Enrollment**: the SDK uses the real `POST /pgw/v1/sdk/enroll`; the reply now carries `lane: {transport, port}`
  (the advertised TCP+TLS lane port) and `masked_head` per model. The SDK tests run the real accounts-enabled service
  (keys created and revoked through the admin API; the test-only route is gone).
- **Provisioning**: the default provisioner is the real `pgw.provision.client.provision` behind
  `pgw.sdk.provisioning.ProviderProvisioner`. **Fingerprints pinned in the library**: 0.5B and 1.5B derived here
  with the pack tool from the public checkpoints at the pinned commits (`scripts/pin_model_fingerprints.py`); 3B/7B
  pinned by commit and cross-checked on first use by re-deriving the pack from the public checkpoint
  (`source="hf"`, remembered in the cache) until their fingerprints are baked in. A provider serving another pack
  is refused before any download. **`hint="local"` is the default** (the provider-built hint streams ~44 GB at 7B;
  opt-in `hint="provider"`). The gateway runs on the weight-free gateway pack (`GatewayPack` in
  `pgw.gateway.runtime`); the background factory gets a **factory pack** (the fingerprint-checked int8 weights,
  ~6.5 GB at 7B, `provision(factory_pack=True)`) for its private check vectors and the local hint build.
- **Console on the provider**: `PGW_PLATFORM_ENABLED=1` mounts `pgw.platform` at `/platform` on the same store as
  the admin API (`pgw.service.platform_accounts`). New, content-free: `POST /pgw/v1/sdk/report` (the client's
  per-request record: model, token counts, latency, status, whether every check passed; allow-list parsed),
  request counts and the input/output split per key/model/day in usage, a request log (`/orgs/{org}/requests`,
  kept `PGW_SERVICE_ACCOUNTS_REQUEST_LOG_DAYS`, 30) and `/orgs/{org}/models`. "Verified" is shown as **checked
  client-side** (the provider cannot check results; it records the client's statement).
- **Architecture rule**: the fused2 table derivation moved to `pgw.modelpack.fused2_weights` (code unchanged, the
  engine re-exports it); the service no longer imports the engine.
- **torch 2.15.0** (the measured Modal stack) in every lock and image (CPU `+cpu`, CUDA `+cu132`, provider base
  `nvidia/cuda:13.2.1`); locks regenerated, all constrained by `pack-pins.in`. Docker images not rebuilt here.
- SDK: `create(timeout=...)` bounds the request (a short timeout never cancels a model start-up in progress); during
  a model's first fill the request waits for the one-time mask setup instead of failing after `mask_wait_s`.
- **End to end** (`tests/sdk/test_e2e_product.py`, CPU, loopback TLS, this 4-vCPU box): real provider service
  (accounts, provisioning, console, TCP+TLS lane, masked head), org + console token via the CLI, key via the admin
  API, empty cache -> `from pgw import OpenAI` -> enroll -> provision (real endpoints; the provider's pack of
  Qwen2.5-0.5B-Instruct equals the baked pin) -> chat non-streaming and streaming **bit-identical** to the gateway
  path -> usage, split and request log via the admin API and the console, no content anywhere -> key revoked, next
  call refused. 5 passed in 72 s; first request on an empty cache 23.8 s (plain masks). The library's default scheme
  (pks-v3, hint built locally from the factory pack) is covered bit-identically on the tiny model
  (`tests/sdk/test_e2e_cpu.py`); at 0.5B the client built the hint locally (24 layers + head in ~4 min on this box)
  but the **CPU provider's pks-v3 maskgen peaked at ~10 GB and was OOM-killed** on this 15 GB box, so that check is
  opt-in (`PGW_SDK_E2E_PKS=1`). Open: provider memory of CPU pks-v3 maskgen; 3B/7B fingerprints not yet baked in;
  images not rebuilt with torch 2.15.0; mypy baseline errors moved with the fused2 code are unchanged.
  *(Since then: 3B/7B pins baked in, pack format 2; the provider hint is recomputed per item on the device instead
  of cached on the host. See the 2026-11-26 sections at the end.)*


## 2026-11-25 — Hosted provider: SDK keys, usage metering, provider image, RunPod tooling (not run on RunPod)

For the in-process client library (`docs/sdk/CONTRACT.md`; payloads in its appendix A1-A4):
- **Accounts** (`pgw.service.accounts`, on with `PGW_SERVICE_ACCOUNTS_DB`, SQLite WAL): organisations, SDK keys
  `pgw_<env>_<base32>` (provider URL + pinned CA SHA-256 + key id + 192-bit secret; codec `pgw.enroll.sdk_key`;
  stored as a digest, constant-time compare, model scope, expiry, per-key row rate, key/org daily token quotas,
  installation and enrollment-rate limits), installations, per key/model/UTC-day usage (trips, rows, tokens,
  maskgen columns, bytes; metered in memory, flushed by a thread) and an audit log without secrets.
  `POST /pgw/v1/sdk/enroll` (authenticated by the key) reuses the enrollment machinery
  (`EnrollmentService.issue_credentials`) and returns exactly what `/pgw/v1/enroll` returns plus `sdk`;
  `pgw.enroll.client.enroll_with_sdk_key` is the client side. Revoking a key removes its installations' tokens and
  refuses its tenant at once in-process (within `accounts_refresh_s` elsewhere); open stream and lane channels close.
  SDK tenants get 403 outside their model scope and 429 over rate or quota on every compute path (HTTP, stream, lane
  open and lane drain, maskgen). Admin API `/pgw/v1/admin/...` (off by default; global or org-scoped tokens).
  CLI: `pgw service orgs|keys|usage|audit|admin-token`. Other tenants are untouched.
- **Images**: `deploy/docker/provider.Dockerfile` (CUDA base, hash-locked torch 2.5.1+cu124 / triton, `pgw_fast`
  built in a builder stage with a pinned Rust toolchain, pack tool, root only to hand over the volume, service as
  uid 10001 with no_new_privs, health check) and `provider-cpu.Dockerfile`; locks `requirements/provider-*.txt`;
  `deploy/docker/provider-smoke.sh`; GHCR workflow `.github/workflows/provider-image.yml` (not run).
- **RunPod** (`deploy/runpod/`): launcher with cost estimate and a required hard `--max-usd`, Secure Cloud default,
  GPU / data center / volume selectable, server-side TTL self-stop plus a detached client-side watchdog, attested
  bootstrap of an SDK key, `--client-pod` (CPU pod in the same data center), `--status`, `--stop`; `rtt_probe.py`.
  Provider tuning for remote clients: TLS trip lane on 8444 (`lane_advertised_port`, new, default off), trip stream
  priority high, masked head `also`, maskgen v3. **No RunPod call has been made** (no API key yet): unit-tested
  against a mocked API; the bootstrap against a real loopback TLS service.
- Tests: `tests/service/test_accounts.py`, `tests/deploy/test_runpod.py`, `tests/deploy/test_provider_image.py`.

## 2026-11-25 — SDK provisioning: fingerprint-checked artifacts and a provider-built, client-verified pks-v3 hint

For the in-process client library (`docs/sdk/CONTRACT.md`, appendix A3). New: `src/pgw/provision/` (client),
`src/pgw/service/provision.py` (provider, `/pgw/v1/provision/...`, off by default: `PGW_SERVICE_PROVISION`).
- **Artifacts:** exactly what the weight-free masked-head gateway reads (manifest, plain tensors, every linear's
  scale/bias, tokenizer), derived from the pinned manifest, fetched in chunks (resumable, per-chunk SHA-256), each
  object checked against the manifest the fingerprint covers, written in place and renamed after a full verification;
  `GatewayPack` (a `ModelPack` without int8 weights). Fallback `source="hf"`: download + pack locally, same pin.
- **Hint:** the key-switched hint is not cheaply checkable (two exact roundings after the GEMM). The provider computes
  `H = W8 A mod 2^56` (the GEMM: 60 % of a local build's CPU here, more without fast int8 GEMMs) for the client's fresh `a_seed` (`ks_seed` stays local); the client accepts
  each linear's `H` only after a 2-adic Freivalds check against the pinned `W8` (42 binary columns per side, error
  `<= 2^-41` for any `H` the provider picks) and computes the switch and key switch itself: the store is bit-identical
  to a local build with the same seeds (tested). Default `hint="provider"` for the SDK (equivalence argument in
  `research/reviews/pks-v3-provider-hint.md`, question Q7 added to the review packet); `hint="local"` stays.
- **Costs at 7B:** client CPU 1700 CPU-s (~10 min on this 4-vCPU box) instead of 3030 CPU-s (~22 min) for a local build,
  of which the check is 500 CPU-s; but 44.2 GB of `H` streamed (plus 6.5 GB `W8`, both never stored); stored hint
  2.37 GB with the head. Pays off with slow int8 GEMMs / few cores and a fast link; `hint="local"` otherwise.
- **0.5B end to end on this box:** Qwen2.5-0.5B + masked head, provider and client on loopback (CPU, load ~10):
  24.5 min including the provider's CPU GEMM, 13.1 GB `H`, hint files equal to a local build's (spot-checked linears).
- Optional `pgw.provision.streaming` (hint from customer storage, bounded LRU, digest-checked): opt-in; it re-reads the
  whole hint per segment when the cache is smaller than the hint, so local disk stays the default.
- Tests: `tests/provision/` (check soundness incl. 2^55 / structured / wrong-seed / wrong-weight errors; tampered chunk,
  lying listing, wrong fingerprint, adversarial provider hints, resume, local and public-source fallbacks, streaming).

## 2026-11-23 — pks-v3 review fixes on the default path; pks-v3 the CPU-only demo preset (pending external review)

Fixes for `research/reviews/pks-v3-adversarial-review.md` (F1, F3, F4, F6, F7, F10), now on **every** encrypted
factory, the default mlwe v1/v2 path included:
- **F1 (high), all backends:** any product from a provider (mlwe v1/v2 over HTTP or the bulk channel, fleet
  `SegmentShardedRemote`, pks-v3, encrypted fused2 digits) is self-checked on every row with enough private
  vectors for **40 bits** against errors the provider chooses (14 at capped K32 widths; was 2: a `2^31` error passed
  with 1/4, a `2^27` one with 2^-10). A backend may ask for more, never less. Local masks (plain CPU/GPU factory,
  mask service) keep 2 vectors: their errors are faults, not choices, and gross ones fail the range check.
  Private vectors and `v W8` are reused across accepted segments and wiped/redrawn after any failed or aborted one.
  Partner imports are unchanged (a partner is trusted with the seed; it runs this factory itself).
- **Cost (this 4-vCPU box, m = 18944, n = 3584, 256 rows, one core):** the new 32-bit SIMD row check
  (`pgw_fast.he_check_rows`, K <= 32, up to 16 vectors) runs 14 vectors in **2.2 ns per (slot, m+n)**, below the
  former 2-vector uint64 check (2.3 ns; 14 vectors on the old path: 7.3 ns). 7B (Σ(m+n) = 2.2M): ~4.8 ms CPU per
  slot, as before. The per-segment `v W8` (3.8 s CPU per 7B segment with 14 vectors, 2.1 s with 2) is now paid once
  per process instead of per segment. Real v2 factory (`he_factory_overlap_bench.py`, 0.5B shapes, 2 layers,
  in-process provider, 4 x 512-slot segments, 2 runs each): segments bit-identical (same digest); check stage
  0.158 / 0.163 s with 14 vectors vs 0.173 / 0.188 s with the old 2, of 3.4-3.7 s wall. Rebuild `pgw_fast` (`rust/tools/build_fast.py`) to get it;
  an older build falls back to the uint64 path (correct, slower).
- **F3:** provider-attributable integrity failures (self-check mismatch, malformed reply) are a CRITICAL alert,
  counted in `factory-state.json` (never reset by success or restart), and after `max_provider_integrity_failures`
  (3) the factory **halts** (every fleet worker; `/healthz` unhealthy; alerts `PGWFactoryProviderIntegrityFailure`,
  `PGWFactoryHalted`) until `pgw factory clear-halt --yes`. Transport errors keep retrying with back-off.
- **F4:** `check_field = "auto"` (new default) writes **p31** segments for encrypted factories at K32; gateways accept
  p31 by default (`prime_check` unset = auto: TripEngine already on, native prime kernels, K <= 32, no padded speculation). A wrong mask is
  then caught online with the same reaction for every input (tested; the ring format lets it through silently).
  The factory keeps its 40-bit check (not relaxed for pks-v3: the online check is a backstop, not a proof).
- **F6/F7/F10:** pks-v3 hint `meta.json` HMAC under a customer key (`he_hint_key_file`, outside the hint dir;
  without it the trust boundary is documented); hint store `flock` (a rebuild never replaces seeds under a running
  factory); doc fixes (reply-key core-SVP 251/228, 32 rows per reply ciphertext at n = 8192).
- **Defaults:** pks-v3 is the preset of the CPU-only co-located demo (`launch.py --he-scheme auto`) and of the
  customer-cloud template's `cpu_encrypted` presets (`he_scheme = "auto"`); both pass the library's
  acknowledgement explicitly. **pks-v3 remains pending external review (lattice-estimator run and cryptographer
  sign-off).** The library default stays mlwe. Not run on Modal (no Modal in this session).
- Tests: `tests/factory/test_provider_integrity.py`, `tests/hemask/test_factory_he.py` (chosen errors on mlwe,
  vector floor for every remote, reuse/wipe), `tests/hemask/test_native_sampling.py` (SIMD path == uint64 == numpy),
  `tests/hemask/test_pks.py` (hint MAC, lock), `tests/engine/test_prime_check.py` (wrong mask: p31 vs ring, defaults),
  `tests/gateway/test_setup.py`, `tests/deploy/*` (launcher, template, alert rules).

## 2026-11-23 — 7B at 28 tok/s with the encrypted mask factory keeping up (Modal, H100, CPU-only gateway)

Same stack as the 28 tok/s entry (shared-memory lane, native decode driver, masked head, p31, weight-free gateway),
48 gateway cores, encrypted (MLWE v2) factory with 8 workers x 4 threads, refill active through the window:
- **Sustained 150 s of back-to-back chats: decode 28.27 tok/s median (min 24.1)**, warm single chat 22.2 tok/s;
  logprobs bit-identical, text equal, every sustain chat identical.
- **Masks keep up:** consumption 24.8 slots/s (2.62 slots per generated token incl. prompts), factory production
  26.9 slots/s in the window (25.5 commit-to-commit), stock 14634 -> 14950 (min 14438), keeps_up = true.
- Customer side: CPU only, no linear weights on the gateway. The default v2 factory still keeps its ~34 GB encryption
  hint at 7B; the compact pks-v3 scheme (~2.1 GB, ~1 core per 50 slots/s) needs an independent review before it can
  be the default.
## Update 2026-11-23: Qwen2.5-32B at 25 tok/s? Per-trip gateway CPU at 32B widths, the co-located budget, 32B commands

Goal checked: about 25 tok/s (<= 40 ms per token) streamed for Qwen2.5-32B with a CPU-only gateway that holds no
weights, exact and private. Everything here is local (4-vCPU dev box, shared with other jobs; no Modal, no Kaggle).
Every new knob is off by default and changes no bits.

**Measured: gateway work per trip at 32B widths** (the native drivers' own counters, 2 layers at the preset's widths,
stub provider in another process; medians of interleaved runs, spread about +-15% on this box):

| path | dev box, 1 core | what it is | calibrated to the Modal gateway (x0.87) |
|---|---|---|---|
| standard decode driver, 7B widths | 110 us | qkv/o/gate_up/down prepare + finish + float ops | 0.096 ms (the 7B run's measured per-trip gateway work, by construction) |
| standard decode driver, 32B | 137 us (qkv 102, o 65, gate_up 214, down 205) | same | **0.119 ms** |
| + `PGW_LANE_WAIT_PREFETCH=1` | 121 us | `W r` rows and check vectors prefetched while the lane waits | **0.105 ms** |
| + trip split, extrapolated to 16 warm pinned cores | ~80 us | kernel-level fit (cold floor 125 -> 85.6 us at 3 cores, hot floor 59 us) | ~0.07 ms |
| native fused2 driver, 32B, mean pair trip | 824 -> **671 us** after this update's 8-byte pack/unpack (o+gate_up 839, down+qkv 503) | quantize, mask, 7-byte pack, unmask, both p61 checks, float ops | **0.58 ms** |
| + `PGW_FUSED2_SPLIT=12`, extrapolated | ~363 us (o+gate_up 133 serial + 706/12; down+qkv's quantization of n = 27648 stays serial) | | ~0.32 ms |

Target "<= ~0.12 ms per standard trip": **met on one core** (0.105-0.119 ms calibrated). Target "<= ~0.2 ms per fused2
pair trip": **not met** (0.58 ms on one core, ~0.32 ms extrapolated with the split): the down+qkv pair's serial
quantization (float64 passes with numpy's pairwise sums over 27648 inputs, a division per H element) is the next
item, then the memory-bound K56 unmask (8-byte mask words). On this 4-vCPU box the split itself does not pay end to
end (`PGW_TRIP_SPLIT` 137 -> 137-139 us; with 3 spinning workers the stub provider starves: 304 us): the workers,
the driver thread, the spinning lane client and the provider share 4 vCPUs; the kernels alone scale 1.46x at 3 cores
(`benchmarks/trip_split_scaling_bench.py`), the pooled finish does not (column split slower than one thread on this
VM), and the down prepare does (79 -> 44 us at 4 cores). The fused2 split is inconclusive to adverse end to end here (o+gate_up
after the reply: 760 -> 1500-2700 us with 2-3 spinning workers over shm, 768 -> 2413 us over UDS, but 1004 -> 673 us
in a pair of runs under a load average of 14): that bench's provider is an in-process `LaneService` whose server and
Python executor need the same 4 vCPUs, other jobs shared the box, and a fork-join stalls whenever one worker is
preempted, so the driver thread waits; the split numbers above are
therefore extrapolations from the per-phase single-core times (unmask, checks, float are element-wise or exact
partial sums mod p), to be measured on a gateway with its own cores.

**Built (opt-in, exact):**
- Native pool (`rust/pgw-fast/src/prepare.rs`): `PGW_POOL_ACTIVE_SPIN_US` (workers keep polling while a decode driver
  call is active, `pool::Warm` guard held by both native drivers for each call; outside a pass the ordinary 50 us spin
  then sleep, so the factory keeps its cores), `PGW_POOL_WORKERS` may now raise the pool above 7 (up to 47 and the
  CPU count - 1), `PGW_POOL_CPUS=4-15` pins worker `i` to the list's `i`-th CPU.
- Lane wait prefetch (`PGW_LANE_WAIT_PREFETCH=1`, `lane::wait_prefetch`): after the request is sent and before the
  reply is awaited, the lane issues prefetch hints for what the driver reads next (standard: this trip's `W r` rows
  and `u`, the next trip's `uw`; fused2: `W1 r_x`, `v`, `W2' v`, both stages' check vectors, the next trip's `r_x`).
  Hints only (never fault, never write).
- fused2 driver: `PGW_FUSED2_SPLIT=c` (+ `PGW_FUSED2_SPLIT_MIN_ELEMS`, default 8192): the unmask of `Z`, the stage-2
  check (`p61::check_split`: range check + dot products in fixed ranges, canonical partial sums mod p, the same
  verdict) and the normed gate/up + SiLU of one-row pair trips in fixed element ranges on the pool; 8-byte
  pack/unpack of the 7-byte K56 words (same bytes; the per-element `memcpy` was 60% of the unmask).
- Phase counters for benchmarks: `pgw_fast.decode_phase_ns()` (standard driver: prepare / finish / float per linear
  kind), `pgw_fast.f2_phase_ns()` (fused2: before / after the lane, unmask / checks / float per trip kind); reported
  by `decode_cpu_bench.py` and the new `benchmarks/fused2_driver_cpu_bench.py` (native fused2 driver over a native
  `LaneService`, UDS or shm, zero weights, every check passing) and `benchmarks/trip_split_scaling_bench.py`.
- fused2 tables without SVDs where they are not read: `Fused2Weights(..., sigma=False)` for the provider's lane entry
  and the factory (one dense SVD per `o` / `down` for `sigma` took 27 s each at 32B widths here: about an hour per
  party single-threaded); the lane fingerprint, the pair op and the masks use `w8` / `hmul` only.
- Launcher: `--prepare-cache --schedule fused2` also derives the full fused2 tables once into the gateway cache
  (`<model>/fused2/fused2-m<bits>.pt`, `PGW_MODAL_FUSED2_PREP`); the run restores them and the gateway
  (`fused2_weights_path`, now an allowed wizard preset: base fingerprint checked, the lane fingerprint binds `w8` /
  `hmul`), the factory (`PGW_FACTORY_FUSED2_WEIGHTS_PATH`) and the e2e's plaintext reference load the same file (one
  `sigma`, the same bits everywhere).

**The 32B budget** (`python benchmarks/scaling_model.py --colocated`; 1 decode row, short context; lane 0.09 ms per
trip and device time scaled by weight bytes from the measured 7B run, which the model reproduces: 15.8 vs 15.7 ms of
waiting, 35.8 vs 35.7 ms per token; serving outside the passes 9.2 ms, the 7B run's residual; planes pair op 1.07-1.30x
the two K32 singles, measured on an H100; TP gather 0.01 ms per trip, estimate):

| configuration (trips) | lane | device | gateway | other | ms/token | tok/s | bound |
|---|---|---|---|---|---|---|---|
| standard, 1 H100, gateway 1 core (257) | 23.1 | 25.5 | 27.2 | 9.2 | 85.0 | 11.8 | gateway |
| standard, 1 H100, gateway split 16 (extrap.) | 23.1 | 25.5 | 18.1 | 9.2 | 75.9 | 13.2 | device |
| standard, 2 H100 TP, gateway 1 core | 23.1 | 15.3 | 27.2 | 9.2 | 74.8 | 13.4 | gateway |
| standard, 2 H100 TP, gateway split 16 (extrap.) | 23.1 | 15.3 | 18.1 | 9.2 | 65.7 | 15.2 | lane |
| fused2 1.07-1.3x, 1 H100, gateway 1 core (130) | 11.7 | 27.3-33.0 | 75.5 | 9.2 | 124-129 | 7.7-8.1 | gateway |
| fused2 1.07x, 1 H100, split 12 (extrap.) | 11.7 | 27.3 | 41.0 | 9.2 | 89.1 | 11.2 | gateway |
| fused2 1.07x, 2 H100 TP, split 12 (extrap.) | 11.7 | 14.9 | 41.0 | 9.2 | 76.8 | 13.0 | gateway |
| target: fused2 1.07x, 2 H100, 0.06 ms/pair, other 2 ms | 11.7 | 14.9 | 7.8 | 2.0 | **36.5** | **27.4** | device |
| target: standard, 2 H100, 0.03 ms/trip, other 2 ms | 23.1 | 15.3 | 7.8 | 2.0 | 48.3 | 20.7 | lane |

(fused2 counts 130 trips: 2L + 1 plus the masked head as one K32 trip, which fused2 does not combine with yet; without
it the gateway streams the 1.56 GB bfloat16 head, +50-80 ms per token.)

Reading: with today's measured components 32B co-located is **11-15 tok/s**. No single component reaches 25: lane +
device alone are 38-49 ms on the standard path, so 25 tok/s needs **fused2 (lane 11.7 ms) + 2 H100s (device ~15 ms)**
or a device time per byte much better than the 7B scaling (the 7B singles reach 1.25 TB/s; at 2.5 TB/s one H100's
32B device time is 12.5 ms), **plus** (1) gateway work <= ~0.06-0.09 ms per pair trip (5-10x below one core today,
3.5-5x below the extrapolated split: split the pair quantization, 4-byte mask words where K allows, more cores),
(2) the serving residual from 9.2 to ~2 ms per token (outside the forward pass: scheduler, sampling over 152k logits,
streaming; not yet profiled), (3) fused2 combined with the masked head. The bound today is the gateway on fused2 and
lane + device on the standard path.

**32B co-located commands** (README "Qwen2.5-32B with the full fast stack"; both `--dry-run` tested; nothing run):
standard full stack, TTL 20 min, $7.70/h, **$1.8-2.6** expected ($2.57 cap); fused2 with the planes op, the plain
customer-side fused2 factory and cached fused2 tables, TTL 30 min, $8.08/h, **$2.7-3.8** expected ($4.04 cap), after a
one-time `--prepare-cache --schedule fused2` (tables, ~$0.3-0.45). Readiness checks: H100 memory with fused2 (K32
kernels + pair-op weights, ~63 of 80 GB), provider host RAM for its derivation (`--provider-mem-gib 96`), the masked
head's hint extension at start (est. 2-4 min, CPU), `PGW_MODAL_CACHE_SAVE=0` (the extension's hard-linked base files
would be copied into the cache, 114 GB), pinned split workers on CPUs 4-15 apart from the factory.

Tests: `tests/native/test_scale32_knobs.py` (decode driver with prefetch + warm pool + pinned workers + split at a low
threshold, fused2 driver with the split + prefetch: the existing bit-for-bit and tamper suites in fresh processes),
cargo `p61::split_check_equals_check`, `decode_f2::center_and_pack` (8-byte paths == byte copies), pool CPU-list
parsing and warm guard, `tests/benchmarks/test_scaling_model.py` (the co-located model reproduces the 7B run; trip
counts; TP), `tests/deploy` (fused2 tables cache restore, `PGW_MODAL_FUSED2_PREP` in the deploy env). Engine, fused2,
decode-native, factory, gateway, deploy suites: 108 + 407 passed.

## Update 2026-11-23: fused2 in the gateway service, the factory scheduler and the HE factory (opt-in)

- **Gateway service** (`schedule = "fused2"`, no longer pending; `pgw.gateway.fused2`, `ModelRuntime`): public fused2
  weights derived from the verified pack at start-up (`fused2_migrate_bits`, or `fused2_weights_path`), the fused2
  lane fingerprint pinned (store inventory key, lane model; a service serving other weights is never used), a lane
  ticket for it over the authenticated compute client (UDS, `PGW_LANE_SHM` shm, or `lane = "tls"`), the
  `Fused2MaskedBackend` on weight-free tables with masks from per-request fused2 reservations (window or
  incremental, `reserve(..., schedule="fused2")`), the native fused2 driver for single-sequence passes
  (`fused2_decode_native = auto`), the noise monitor on every payload and the circuit breaker on any failed check
  (Python path and native driver). `/admin/status`: `schedule`, `fused2` (lane / base fingerprint, transport, native
  driver, trips, checks, bytes). The plain dev mode with the setting is the fused2 reference. Validation: capped K56
  ring, native float ops (implied), lane auto/required/tls, `freivalds_rate = 1`, p61 soundness vs
  `min_soundness_bits`; refused: masked head (not combined yet: the head stays on the gateway), speculation, local
  linears, prefix cache, chunked prefill, the in-memory mask ring.
- **Factory**: `PGW_FACTORY_SCHEDULE=fused2` / `--schedule fused2` (`run`, `generate`, `mask-service`, `status`):
  `Fused2MaskGenerator` per pack and ring under `FactoryScheduler` (low water / target / ramp, throttle and stop per
  chunk, chunk settings, disk budget) and the mask-service start-up check (`--check-only` report, p61).
- **HE without new cryptography** (`pgw.factory.fused2_he`, plan section 12): K56 products as independent uniform
  digits on the existing encrypted factories, recombined exactly mod 2^56 (mlwe v2 at K40, 4 x 14-bit; pks-v3 at
  K32, 7 x 8-bit); each digit is an ordinary v2 / v3 maskgen request of public shape; the provider serves its fused2
  linears for maskgen with `PGW_SERVICE_FUSED2_MASKGEN=1`. Bit for bit equal to the plain factory's segments (real
  HEMaskBackend and PKSMaskBackend with the service's kernels in-process). Customer CPU per full 7B slot, this 4-vCPU
  box (wall, provider excluded; `benchmarks/fused2_he_split_bench.py`, 2048 of each linear's output rows kept and
  scaled, an upper bound): plain CPU fused2 factory **0.13 s**, mlwe digits **0.29 s** (2.2x), pks-v3 digits
  **0.17 s** (1.3x).
- **Launcher**: `launch.py --schedule fused2` (provider fused2 lane entry + maskgen, wizard presets, factory, fused2
  reference in the e2e), dry-run tested; the 7B command is in `deploy/modal/README.md` ("fused2 in the co-located
  demo"). Not run (no Modal in this session).
- Tests: `tests/gateway/test_fused2_gateway.py` (masked == plain logprobs bit for bit through the store, native on/off,
  window/incremental, tamper -> breaker, other weights refused, config), `tests/factory/test_fused2_factory.py`
  (scheduler refill, throttle/stop, mask service + CLI, digit split exact, v2 and pks-v3 == plain bit for bit,
  service resolution), `tests/deploy/test_modal_launch.py::test_schedule_fused2_flag_and_dry_run`; gateway, fused2
  engine, factory, deploy and lane suites: 919 passed.

## Update 2026-11-23: fused2 pair masks from the encrypted mask store (opt-in; format, factory, gateway)

- **Format** (`pgw.maskstore.format`, SPEC 7.2): fused2 pair-mask segments are today's `*.pgwm` files with
  `schedule = "fused2"` + `check_field = "p61"` (omitted from standard headers, whose bytes are unchanged; readers
  without the fields refuse them). One entry per fused2 linear in pass order (`r_x` / `W1 r_x` for a trip's first
  linear, `v` / `W2' v` for a pair's second; the gateway forms `r_R = v - hmul W1 r_x`), `r` derived as today,
  `uW = (u * hmul) W mod (2^61 - 1)` as 8-byte words. Header fingerprint = the fused2 lane fingerprint;
  `reserve(..., schedule="fused2")`, a mismatched schedule is refused (fail closed).
- **Factory** (`pgw.factory.fused2_gen`, `python -m pgw.factory.fused2_gen generate|bench`): exact `W r mod 2^K`
  (`cpu_f64` on CPU, `cuda_int8` on a customer GPU), exact p61 words (float64 limbs), start-up check vs numpy uint64 /
  Python-int references, per-chunk private self-check, abort on error, device scrub on CUDA. CPU, 4 vCPUs, Qwen2.5-7B
  shapes: **5.3 slots/s** full model into an encrypted store (6.1 without the write; 7.2 at K32 with the same code),
  + 16 s of `uW` per segment. GPU not measured (no GPU here): ~95-100 slots/s estimated on a T4 (K56 = 7 int8 digit
  GEMMs vs 4 at K32; host ChaCha20/PCIe ceiling ~600 slots/s).
- **Gateway** (`pgw.engine.fused2_store`): `Fused2MaskedBackend(ring, provider, weights=tables)` takes pair rows from
  the call's store session (`MaskReservation` / `WindowedReservation`, incremental: `open_fused2_session`), no
  in-process source, no weights; takes never cross a segment extent (trips run in row pieces), check vectors copied
  per segment and wiped; the native driver (`Fused2NativeDecode`) runs a pass natively when it fits one extent and
  installs the segment's check vectors on change (`wipe_checks()`).
- **Measured**: `tests/engine/test_fused2_store.py` (10 tests: format round trip / old readers refuse, exact segment
  content, self-check abort, masked == plain via the store on the Python path and the native driver over the lane,
  incremental session, 1 slot per token, tamper + rows never reused, crashed gateway process, burn on close);
  maskstore + factory + gateway mask suites unchanged (668 passed). End to end, Qwen2.5-0.5B, `benchmarks/fused2_e2e.py
  --masks store`: factory process -> encrypted store -> weight-free gateway, logits bit-identical (4/4), 1.0 slot per
  token.
- **HE**: v2 / pks-v3 cannot make K56 masks (`q <= 2^64`, 24 bits of noise room: K40 / K32). Plan section 11: a direct
  scheme needs `q >= 2^80` (RNS, new parameters, review); the practical path is digit decomposition on the existing
  schemes (4 v2-K40 products or 7 pks-v3 products per mask, recombined exactly; p61 words need no HE).
- Not done: the gateway service still refuses `schedule = "fused2"` (request path -> `open_fused2_session`), the
  factory scheduler / mask-service CLI do not run the fused2 generator yet, K56 device PRNG, GPU numbers.

## 2026-11-23 — 7B at 28 tok/s with a CPU-only, weight-free gateway (Modal, H100 provider)

Measured (Modal co-located, Qwen2.5-7B-Instruct, H100 provider with 12 cores, 32-core CPU-only gateway holding no
linear weights, all linears remote, masked LM head on the provider, native float ops, native decode driver, p31
checks, sampled noise checks inline before the next payload):
- **Streamed decode 27.97 tok/s median (18 back-to-back chats, 23.6-29.6), warm single chat 25.6 tok/s, TTFT ~1.9 s.**
- Exact: logprobs bit-identical (max diff 0.0), text equal, stream identical, every sustain chat identical.
- What changed vs the 19.0-19.3 tok/s runs: the gateway<->provider lane runs over the **shared-memory transport**
  (`PGW_LANE_SHM=1` / `PGW_SERVICE_LANE_SHM=1`: an anonymous sealed segment passed over the authenticated UDS
  socket; same frames and bytes), with adaptive reply spin (`PGW_LANE_SPIN_AUTO_US=400`), zero-copy replies and the
  lane's own device lock, the provider's read spin OFF and 12 provider cores. Inside Modal's gVisor sandbox the
  socket path was the cost; shared memory avoids its network stack. Per decoded token: ~15.7 ms waiting on the
  provider + ~10.8 ms gateway work (880 steps: 13.85 s remote, 9.51 s local).
- Masks: consumption 31.4 slots/s (2.71 slots per generated token, prompts included; incremental reservations +
  prefix reuse would cut it to ~1.2-1.3). The encrypted factory produced nothing inside the 60 s window (stock
  6547 -> 4657, refill not yet triggered), so keep-up at this rate is NOT shown; at ~35 slots/s it needs the factory
  sized accordingly (pks-v3: ~1.8 customer cores measured kernels; or more v2 workers).
- Flags: `launch.py --placement colocated --gpu H100 --model qwen2.5-7b-instruct --gateway-cache --gateway-cpus 32
  --provider-cpus 12 --local-linears off --float-ops native --trip-native --decode-native --masked-head --prime-check
  --gateway-env PGW_LANE_SPIN_AUTO_US=400 --gateway-env PGW_LANE_SHM=1 --gateway-env PGW_TRIP_DECRYPT_AHEAD=1
  --provider-env PGW_SERVICE_LANE_SHM=1 --provider-env PGW_SERVICE_LANE_ZERO_COPY=1 --provider-env
  PGW_SERVICE_LANE_OWN_LOCK=1` (plus the factory sizing env of the README's CPU-only 7B command).
## Update 2026-11-23: fused2 tensor-core pair op ("planes") and native fused2 decode driver (opt-in, not yet on GPU)

- Provider: `PGW_SERVICE_FUSED2_VARIANT=planes` (`pgw.core.ring64`): input as 8 balanced int8 digit planes per row
  (written by the fused unpack and by stage 1's epilogue), ONE int8 tensor-core dot per tile with an int32 accumulator
  over the whole split (exact: split length <= 131071 enforced), recombined once in uint64; epilogue fused into the
  GEMV when unsplit; CUDA graphs unchanged. Exact vs the Python-int reference (interpreter: 1-16 rows, K 32-64,
  splits, extremes). Target 1.0-1.3x the two K32 singles (today 2.0x); H100 numbers pending this one command
  (<= ~2 min, ~$0.13, not run): `modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 2 --cmd "python
  benchmarks/fused2_pair_bench.py --rows 1,2,4,8 --reps 30 --variants planes,sum --budget-s 50 --json /tmp/f2planes.json"`.
- Gateway: `pgw_fast.Fused2Driver` (`rust/pgw-fast/src/decode_f2.rs`, `pgw.engine.fused2_native`,
  `PGW_FUSED2_DECODE_NATIVE=1`): a whole fused2 pass (2L + 1 trips, float work, native 7-byte unpack/unmask, native
  p61 checks of both stages, native TripChannel UDS/shm, segment masks: 1 slot per token, weight-free gateway) in
  one GIL-free call; sampled noise trips end the call (statistics before the next payload). Bit-identical to the
  Python fused2 path on the native float ops (now allowed) and to the plain reference; byte-identical provider
  bodies; tamper (H, consistent H, Z, Z without residual, skipped multipliers) caught. Head still after the layers.
- Per-token budgets 7B/32B/72B (co-located shm and 0.1-0.2 ms RTT): plan section 10. 7B co-located wait: standard
  15.3 ms, fused2 today 15.5 ms, fused2 at 1.0-1.3x 10.3-11.9 ms.

## Update 2026-11-23: fused2 production pieces (opt-in): graphed pair op, pair items on the lane, segments, native p61

- Provider: `pgw.core.ring64` Triton pair op over Z/2^K (K <= 64): `H = R + hmul * W1 x`, `Z = W2' H`, split-K uint64
  (CUDA cores) or int8 digit-plane tensor-core GEMV, fused epilogue, one CUDA graph per (item, rows) replayed by the
  native lane with no Python. Exact (wrapping 64-bit arithmetic only), self-checked at start-up against the
  Python-int reference and each graph through the serving path. GPU speed not measured yet:
  `modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 4 --cmd "python benchmarks/fused2_pair_bench.py
  --rows 1,4 --reps 30 --json /tmp/f2pair.json"`.
- Lane: fused2 weight sets as their own lane model (`pgw.service.fused2_lane`, one item per trip, pair items with
  concatenated widths: the same framing, tickets, UDS / shm / TLS; no `pgw-lane` change). Gateway client
  `pgw.engine.fused2_lane`. Masks from a separate plain-factory process as pair-mask segments
  (`pgw.engine.fused2_masks`; gateway holds no int8 matrices). Native p61 check `pgw_fast.p61_check`
  (0.17 ms per 7B layer vs 2.8 ms numpy).
- Tests (CPU, Triton interpreter, real native lane): masked == plain over UDS and shm, identical provider-visible
  bodies across transports, tamper (H, consistent H, Z, Z without residual, skipped multipliers) caught, 1 slot per
  token with segment masks, segments == in-process masks. Not done: fused2 in the native decode driver (the gateway
  is still the Python model), maskstore / HE factory for pair segments. Plan section 9.

## Update 2026-11-23: fused2 (2 trips per layer) quality gap closed; exact e2e harness ready for one 7B run (opt-in)

Diagnosis: the whole top-1 gap of fused2 (93.9% vs 96.9-97.1% for the int8 reference, 4k wikitext tokens) is the
int8 rounding of the folded weights `W diag(g * sw1)` (with them in float fused2 reaches 97.2%), spread over all
layers; the folded column scales spread 7x (up to 32x), so most columns get a 2.2x (up to 8x) coarser int8 grid.
Fix: **residual-grid migration**: public per-channel integers `hmul` (pack time, from public weights only,
`migration_multipliers`, 8 bits); the provider returns `H = R + hmul * (W1 x)` and the gateway keeps the residual
on the finer grid `sw1 / hmul`, so the folded matrix sees near-uniform column scales. Exact, same trips, same
elements and factory products, masks `r_R = v - hmul W1 r_x` (uniform), stage-1 check with `u * hmul`; costs ~8 bits
in `H`/`Z`: ring **K56A12C53** (7 bytes per element). Qwen2.5-0.5B, engine, 4088 tokens: top-1 = float **96.8%**
(int8 reference 97.1%, old fused2 94.1%), ppl 19.86 (int8 19.83, float 19.93), prompts 96.9% (= int8), every row
at 12 activation bits. Qwen2.5-1.5B (simulator, 2044 tokens): 98.0% (int8 97.7%, old fused2 96.5%).
Tests: migrated masked == plain bit for bit, tamper (H, consistent H, Z, Z without residual, a provider skipping
the multipliers), socket path, prefill == lazy masks.

E2E: `pgw.engine.fused2_remote` (the pair op as its own process on a 0600 Unix socket, CUDA ring products) +
`benchmarks/fused2_e2e.py` + `deploy/modal/fused2_e2e.py` (one container; masks from a gateway-side plain factory;
plain replay must match). Local 0.5B check: 49 trips/token, bit-identical, 1 slot per token (same elements as
today), 3.4 MB per token at K56. It does not use the production runtime, store, HE factory, lane, native decode
driver, p31 or the masked head (fused2 has no path there yet; list in `research/plans/fused2-two-trips-per-layer.md`
section 8 and `deploy/modal/README.md`). Defaults off; the gateway still refuses `schedule = "fused2"`.
## Update 2026-11-23: long contexts (prefill and decode): bounded attention scratch, fused/tiled kernels, chunked prefill

**Finding.** Real masked path, Qwen2.5-0.5B, in-process exact provider, 4-vCPU box: prompt passes of ~500-800 rows
took 88-138 s against 2.9 s for 228 rows. Reproduced (512 rows: 50-277 s). Two causes:

1. **Not the gateway.** In those passes `block_wait` (waiting on the in-process CPU provider) was 46 of 50 s, 150 of
   157 s, 268 of 278 s; the gateway's own work (attention and norms, mask takes, prepare, finish) was ~5 s for 512 rows
   and grows linearly. The provider's float64 BLAS GEMMs (2.9 GB of float64 weights, 4-thread MKL) take 3 s in
   isolation for 512 rows x 24 layers but were busy 64-288 s in the pass: two pipelined prompt blocks put two
   multi-threaded GEMMs in flight next to the gateway's threads, on a box shared with other sessions (load average
   6-11 on 4 vCPUs during the runs; the same GEMM alone went 125 -> 1621 ms per layer while another tenant ran).
   `LocalComputeClient` now runs one request's GEMMs at a time (same bytes); the rest is the test rig's CPU stand-in
   for the GPU service. Also the plain path: the one-time int8-vs-float64 timing probe could pick float64 for the
   whole process when the box was busy for a moment (plain linears ~5-10x slower at 128 rows). It now measures
   interleaved, best of 15, and re-measures a slow verdict (up to 4 rounds, ~1 s worst case, once per process).
2. **The gateway's native prompt attention was O(T^2) in memory.** One call kept `(rows, nh, start + rows)` float64
   scores: 1.9 GB per layer call at 4096 rows, 7.5 GB at 8192, 30 GB at 16k, 120 GB at 32k. Long prompts could not
   run, and attention was memory-bound (18 ns per score on one thread).

**Fixes (native float ops; each bit-identical, so default on).** Compared against the previous build's outputs on
276 problems (v1, v2 blocks 256 and 16, bf16 KV, threads 1 and 3) at the AVX-512, AVX2 and scalar levels: 0
mismatches. New Rust tests: fused == two-phase, long rows == the original one-row kernel, row tiles == one call.
- Prompt passes run as **fused (row, KV head) units**: phase A over every block, row maxima, phase B and the
  block-order merge in a per-thread scratch of `nh x len` scores (3.7 MB per thread at 32k) instead of one big buffer.
  Each row has the same operations in the same order, and units are independent, so any partition or thread count
  gives the same bits. One thread: 18 -> 9.5 ns per score. Decode-sized passes keep the (row, KV head, block) units.
  That path keeps **row tiles** under a 48 MB scratch budget.
- **Scores** are computed 16 positions at a time with the same per-score trees. Block maxima are vectorised (the
  same maximum). **Several heads per pass over V.** The AVX-512 value pass runs in **tiles of 256 positions**, with
  the accumulators and dens carried in memory between tiles (exact f32/f64 stores), so V is read from DRAM once per
  KV group instead of once per head batch.
- Decode-sized passes over >= 4096 positions use the attention pool when threads are automatic. The bits do not
  depend on the thread count.
- **KV cache**: a growth step within 25 % of the limit allocates the limit at once. The first decode step after a
  long prompt no longer copies the whole cache into a doubled buffer (2x peak).
- **Chunked prefill** (`prefill_chunk_rows` / `PGW_PREFILL_CHUNK_ROWS`, **default off**; native float ops only,
  validated): a pass in which a sequence has more than `c` new rows runs as sub-passes of at most `c` rows, each
  through every layer. Logits, KV and linear rows are bit-identical. Gateway memory is bounded by the chunk (masks,
  linear inputs and outputs). The provider sees the same rows and bytes per linear in total, but in
  `ceil(rows / c)` trips whose sizes depend on the public prompt length and this public setting. That changes the
  trip pattern, so it is opt-in. 2048 is a reasonable value.

**Gateway prompt pass vs length** (masked path, zero-cost stub provider, so this is gateway work only: synthetic
0.5B shapes, 2 layers, `native`; same box load for before and after; x12 for 24 layers). Wall s (attention + float
part) / peak RSS:

| rows | before | after | after, chunk 2048 |
|---|---|---|---|
| 128 | 0.17 (0.01) / 0.62 GB | 0.15 (0.01) / 0.62 GB | |
| 1024 | 0.55 (0.14) / 1.01 GB | 0.47 (0.10) / 0.81 GB | |
| 2048 | 1.25 (0.48) / 1.52 GB | 1.01 (0.25) / 1.05 GB | |
| 4096 | 3.34 (1.84) / 3.34 GB | 2.57 (0.99) / 1.74 GB | |
| 8192 | not run (7.5 GB scratch per call) | 6.41 (3.43) / 2.33 GB | |
| 16384 | impossible (30 GB) | 19.3 (13.3) / 3.90 GB | 19.5 (13.2) / **1.26 GB** |
| 32768 | impossible (120 GB) | | 58.9 (47.5) / **1.54 GB** |

Everything except attention is linear in rows: mask takes, prepare and finish together come to ~0.45 ms per row per
2 layers. Attention is quadratic: on this box a 0.5B 32k prompt is ~570 s of attention over 24 layers (~10 ns per
score per thread). The units are independent, so this scales with gateway cores. Projected 7B (per layer x4 for
28 heads x 128 vs 14 x 64, 28 layers): a 16k prompt is ~750 s of attention on 4 vCPUs. Long 7B prompts need a
many-core gateway, or prefix reuse (`prefix_cache = "conversation"`) so they are paid once. Plain exact-integer
passes (real 0.5B): the gateway float part at 4096 rows went 13.0 -> 9.0-12 s. The plain linears dominate and vary
3x run to run on this shared box.

**Decode attention, ms per token, all layers** (native kernel per layer x layers, automatic threads as the model
uses them; before: 1 thread for decode; same load). Before -> after:

| context | 0.5B v1 f32 | 0.5B v2-bf16 | 7B v1 f32 | 7B v1-bf16 | 7B v2-bf16 |
|---|---|---|---|---|---|
| 1k | 3.7 -> 2.2 | 3.2 -> 2.6 | 18 -> 14 | 14 -> 12.5 | 14.5 -> 12 |
| 4k | 26 -> 11 | 15 -> 8.5 | 165 -> 28 | 118 -> 33 | 66 -> 22 |
| 16k | 153 -> 32 | 65 -> 22.5 | 721 -> 142 | 509 -> 112 | 284 -> 101 |
| 32k | 330 -> 59 | 128 -> 39 | 1665 -> 365 | 1257 -> 308 | 754 -> 326 |

One thread alone (the V tiling): 7B 32k v1 1665 -> 667 ms. Decode cost, KV memory and masks are linear in context.
KV is 24.6 KB per token for 0.5B f32 (12.3 KB bf16) and 112 KB for 7B f32 (56 KB bf16), so 32k is 0.8/0.4 GB and
3.7/1.8 GB. Mask demand per decode row does not depend on context. 7B at 32k is memory-bound on this box: the floor
is ~0.2 s per token for 1.8 GB of bf16 KV. Use `-bf16kv` and more gateway bandwidth or cores there; 4-8k contexts
are 22-33 ms per token.

- Tests: `tests/engine/test_prefill_chunk.py`: one pass == chunks (700, 512, 1, n-1 rows; 0-3 threads) == token by
  token at 1500 positions, for v1, v2 (16-position blocks), and bf16 v1/v2; logits and KV bit-identical; a chunked
  pass on a cached prefix plus a batch of two lengths; captured hidden states; masked chunked == plain, with
  provider rows <= chunk and = prompt per linear in total and mask consumption unchanged; KV limit allocation; row
  tiles == single rows at 3000 rows. `tests/gateway/test_prefill_chunk_gateway.py`: multi-turn chat plain, with
  conversation reuse and strict speculation, bit-identical, with the same per-linear rows and masks consumed, and
  config validation. Existing native, decode-driver, tree, prefix-cache, thread-scope and gateway wiring suites
  pass (346 tests), as do core, factory and maskgen (661).

## Update 2026-11-23: decode driver keeps "alarm before the next payload"; few-row GEMV tuning (opt-in)

- **Native decode driver, noise statistics** (`PGW_DECODE_NATIVE`, still opt-in): sampled noise statistics came back
  once per driver call, so an alarm could surface after the rest of the pass's payloads had left. Now a sampled trip
  ends the driver call and the sampling draws are made one trip at a time (the per-trip path's order: the next trip's
  draw comes after the sampled trip's statistics). With `PGW_NOISE_ASYNC_SAMPLED=1` the sampled trip stays on the
  native lane and its statistics are joined before the next payload; with inline statistics (default) the driver
  hands the sampled trip back after its native prepare, the statistics run, and only then does its payload leave
  (ordinary path, same request id and bytes): the per-trip path's "before this payload". Same payloads, draws and
  statistics as without the driver (existing tests); new `test_sampled_alarm_stops_the_next_payload` (local client
  and native lane, sync and async: the driver stops traffic at exactly the payload count the per-trip path does;
  the previous code failed 6 of its 8 cases). Cost: one more driver call per sampled trip (5 % of trips), plus in
  inline mode that trip's send on the Python path; on this loaded 4-vCPU box (0.5B, 8 layers, `--lane-proc`,
  best of 3 runs) wall ms/token old/new: sync 19.2/20.8, async 21.5/20.9 (run-to-run spread 19-31 ms). Use
  `PGW_NOISE_ASYNC_SAMPLED=1` with the driver for the cheaper of the two.
- **Few-row GEMV tuning** (`PGW_SERVICE_TRITON_GEMV_TUNE_ROWS=1`, default off; `triton_gemv.tune_rows`): before the
  lane captures its 2..8-row CUDA graphs, each (rows, linear shape) times the untuned choice (on an H100 the
  16-row-padded tensor-core `dot` variant: 2-row graph trips were 168 us vs 89 us for 1 row, gate_up 389 vs 191 us)
  against memory-bound `stream` tiles (each weight byte loaded once for all rows) by CUDA-graph device time, after
  each candidate passed the exactness check against the digit-GEMM path on the device; a candidate replaces the
  baseline only if 3 % faster. The graphs then capture the pick and pass their own byte-equality self-check as
  before. One-row calls are never touched. Same bits (wrapping u32 arithmetic in every variant). CPU-tested in the
  Triton interpreter (`tests/core/test_triton_gemv.py -k tune`); **not yet run on a GPU**. H100 check (~1.5 min,
  under $0.15): `PGW_MODAL_APP_SUFFIX=-mr modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 5
  --cmd "python benchmarks/provider_scale_bench.py --presets 7b --tp 1 --rows 1,2,4 --iters 200 --modes graph
  --tune-rows --json /tmp/ps.json"` (`graph_report.tuning` has each shape's baseline and chosen device ms; the
  graph trip means use the picks; `device_us` is the graph-timed kernel alone).
## Update 2026-11-23: incremental mask reservations (opt-in), toward ~1.1-1.3 mask slots per generated token

Mask demand per generated token = prompt-pass rows (re-sent every turn) + decode rows + burned slots. The 7B e2e
measured about 2.7 (plain). `prefix_cache = "conversation"` (9.4b) removes the re-sent prompt rows;
**`mask_reservation = "incremental"`** (gateway setting, default `"window"`; `mask_chunk_slots`, default 8; SPEC 9.4c)
removes the burn: the prompt plus one public chunk up front, whole chunks reserved ahead on a background thread,
retired chunks closed off the token step, `< 1.5` chunks burned per request (finished, cancelled or failed) instead
of `max_tokens - generated` (up front) or up to 1-1.5 windows of 64. Admission keeps the up-front guarantee in
memory (`EntitlementLedger`: the store must cover every live request's remaining chunk-aligned prompt + max_tokens
plus one spare chunk), so extensions do not run short mid-answer; if one fails anyway the step fails closed before
any row is taken or sent and every part is burned. Privacy: unchanged in every provider-visible channel (calls,
row counts and order equal an up-front reservation's, chunks are multiples of `spec_rows` in strict mode; store
transactions happen at public pass indices; mask consumption, which drives the encrypted factory's `maskgen`
cadence, becomes rows used + `< 1.5 c`, no longer `prompt + max_tokens`, so `max_tokens` leaves that channel).
Strict speculation with `spec_mask_rows` keeps its public `charge_to` schedule (chunks as windows).

- Tests: `tests/gateway/test_incremental_masks.py` (bit-identical answers and logprobs vs up front, plain and strict
  R = 3, multi-turn with prefix reuse; provider calls identical call by call; consumption = rows used + < 2 chunks;
  admission refuses what live entitlements need; mid-answer exhaustion and a fault after the durable commit end the
  answer with an error, no partial pass, nothing reused, next answer exact; cancel burns every part; 4 concurrent
  tenants exact; default off), `tests/engine/test_trip_native.py::test_incremental_chunks_equal_plain_and_bytes_identical`
  (chunked chains with and without `PGW_TRIP_NATIVE` leases: masked == plain).
- 0.5B, real masked path (fsync store, file counter, every linear remote, in-process exact provider;
  `benchmarks/incremental_masks_bench.py`; Qwen2.5-0.5B-Instruct, 4-turn chat, max_tokens 256, greedy; same answers
  in every configuration): window 64 without reuse **2491 slots / 868 tokens = 2.87** slots/token (up front: 2619 =
  3.02; prompt rows alone 1.84); window 64 with reuse **1056 / 868 = 1.22** (up front with reuse 1184 = 1.36). The
  answers here mostly ran to max_tokens and the 64-windows happened to end close to the answers (burn 5, 1, 1, 25),
  so the window burn was small in this run; its expectation is ~window/2 per request on a plain store and ~1 window
  with the tiered store's ahead reservation. Incremental, measured turns: burn 5 and 9 slots (chunk 8, one answer
  at max_tokens: the spare chunk). Incremental with reuse = rows0 + generated - 1 + burn per turn (asserted by the
  tests): **~1.21-1.23 slots/token** for this chat, independent of window alignment and of max_tokens. The
  incremental-with-reuse run itself was killed by the shared box (load average high, factory at 4-6 slots/s), as
  was a 5-turn run; TTFT and decode times on this box were dominated by other load (decode 290-440 ms/token in all
  configurations, TTFT equal within noise for the same prompt-pass rows), so no timing difference is claimed.
- Projected 7B (plain decoding, chat answers of ~150-250 tokens, new message + template ~20-40 tokens per turn):
  1 (decode) + 0.1-0.25 (new prompt rows) + ~8/200 (burn) = **~1.15-1.3 slots/token**, vs 2.7 measured; with strict
  R = 2 about 2.1-2.3 (vs 4.2), with `spec_mask_rows` C about C + 0.15-0.3.
- Costs: one store transaction (reserve) and one close per chunk, both on background threads; a part re-decrypts
  the `W r` chunk its first slot lies in (1 MiB segment chunks: ~6 slots at 7B K32, so about 0.75 x one slot of
  extra AES-GCM per token at c = 8, ~0.4 ms on one core at the 11 GiB/s measured here; c = 16 halves it).
- Default: proposed as the default once it has run on the 7B co-located e2e (the ledger is stricter than the
  window mode's stock check: a store that covers the stock but not every live request's max_tokens refuses new
  requests instead of risking mid-answer exhaustion; deployments with large max_tokens need correspondingly more
  stock, as up front). Not done: chunk handoff of the decrypted segment chunk between consecutive parts.
## Update 2026-11-23: 1-row trip fixed cost: lane latency knobs, shared-memory lane transport (opt-in, CPU-tested)

Toward <= ~0.15 ms all-in per 1-row trip at 7B (today about 0.46 ms, device 0.04-0.09 ms). All default off; same
bytes, order, tickets, peer-uid check; what the provider sees is unchanged. Details, measurements and the Modal
commands: `deploy/modal/README.md`, "1-row trip fixed cost".

- Gateway (`rust/pgw-fast/src/lane.rs`, used by the native decode driver and the per-trip driver unchanged):
  adaptive reply spin (`PGW_LANE_SPIN_AUTO_US`: per (linear, rows) learned reply time x 1.25 + 20 us, capped), UDS
  spin polls by `recv(MSG_DONTWAIT)` (no `fcntl` per trip), `TripChannel.stats()` (spin hits, blocking waits,
  wait/trip time, the service's `queue_us`/`compute_us`), `LaneComputeClient.channel_stats()`.
- Provider (`rust/pgw-lane`): `queue_us` is now the device wait only (not the body read);
  `PGW_SERVICE_LANE_ZERO_COPY` (reply written from the graph's pinned `h_out`, non-blocking under the device lock,
  remainder after it); `PGW_SERVICE_LANE_OWN_LOCK` (lane graph replays never queue behind Python-path calls on the
  device lock; captures hold both locks); `PGW_SERVICE_LANE_READ_SPIN_US` (existing) recommended for the e2e.
- Shared-memory transport (`rust/pgw-lane/src/shm.rs`, `PGW_SERVICE_LANE_SHM` + `PGW_LANE_SHM`): a `shm` frame after
  the ticket `open`; the service passes a sealed, path-less memfd over the authenticated socket (`SCM_RIGHTS`);
  same headers and bodies in the segment, the socket as doorbell; the service copies and validates every frame
  as on the socket. A start-up race (first request stored before the service read the sequence) was found by
  the bench and fixed. **Not a win on this box** (floor 14.5 vs 12.3 us p50 for UDS); kept opt-in for the H100
  micro-benchmark to decide (gVisor syscalls/wake-ups are where it could pay).
- Local results (overloaded 4-vCPU box, medians): UDS transport floor 12.3 us p50; zero-copy 11.9; with H100-like
  device times and 140 us of gateway work between trips the overhead p50 28.7 -> 21.1 us with the adaptive spin
  (blocking waits 50 % -> 30 %); native decode driver `lane_us_per_trip` ~102 -> ~85 us. Spin-heavy variants
  (server read spin, shm) lose in the mean here because the spinners are preempted (p99 2-2.6 ms).
- New: `benchmarks/lane_gpu_trip_bench.py` (real lane trips on a GPU: variants x Python-path load x shared/own
  lock, exactness checked); `lane_tls_bench.py --stub-us/--zero-copy/--spin-auto-us/native-shm`;
  `decode_cpu_bench.py --spin-auto-us`; tests `tests/service/test_lane_shm.py`.
- Not run on a GPU. Next: the H100 micro-benchmark (~$0.1), then the 7B e2e with the chosen flags.

## Update 2026-11-22: native UDS lane channel (opt-in), persistent GPU worker's first H100 run

Toward the 7B CPU-only-gateway target (25 tok/s, <= ~0.35 ms per trip all-in; provider wait per trip today ~0.38 ms).

- **`lane_native`** (gateway model option, default `false`; `SETUP_MODEL_OPTION_KEYS`): the UDS lane (`lane = "auto"` /
  `"required"`) uses the native `pgw_fast.TripChannel.connect_service_uds` (A1 frames, GIL released for the whole
  trip, one `writev`, spin-then-block wait) instead of the Python socket channel; without the extension the Python
  channel as before. Same header fields and body bytes, same single-use tickets and peer-uid check by the service, same
  status mapping (closing statuses drop the lane and cool down), same byte-identical resend under the same request id on
  the stream channel, same `lane_stats` (`native: true`). It is what the per-trip native driver (`PGW_TRIP_DRIVER`)
  needs on the UDS lane (on the Python channel the driver stays off). Tests: `tests/gateway/test_speed_wiring.py`
  (`test_lane_native_over_the_service_uds_lane_is_bit_identical`, with and without `PGW_TRIP_FAST`/`NATIVE`/`DRIVER`:
  outputs and logprobs equal to the plaintext reference, every lane call counted by the service, the driver served
  trips). Local transport round trip (`benchmarks/lane_tls_bench.py --modes native-uds,py-uds --preset 7b`, stub
  executor, this shared 4-vCPU box): native-uds p50 22 us / mean 35 us, py-uds p50 23 us / mean 25 us: the socket hop
  itself is ~20 us either way, so the ~0.2 ms "wire" of the H100 run is not the channel; it is the gateway process
  around the call (GIL hand-offs: with 2 GIL-busy threads both channels' means rise to 0.4-0.6 ms) and, on the
  provider, the device-lock wait, which the lane reports as `queue_us` (not in `compute_us`, so the client counts it
  as wire). The native channel's gain is therefore expected from the driver (one native call for lane + finish).
- **Co-located demo**: `launch.py --lane uds` (the co-located default) now sets `lane_native` (opt out:
  `--lane-python`), and with `--trip-native` also `PGW_TRIP_DRIVER=1`.
- **Persistent GPU worker, first GPU run** (`deploy/modal/gpu_bench.py --target h100_c4`, new: H100, 4 cores, 12 min
  hard timeout; `provider_scale_bench.py --presets 7b --tp 1 --rows 1,2 --iters 300 --persistent-bps 1,2`; 40 s wall,
  ~$0.1): NVRTC compile, launch and the startup self-check all passed, and every timed persistent trip was **exact**
  (0 mismatches), but it was **slow: 2.94 ms per 1-row trip (5.66 ms with 2 blocks/SM)** vs graph replay 0.089 ms and
  the kernel path 0.296 ms. 1-row service time per 7B trip (us, mean): graph qkv 44 / o 40 / gate_up 191 / down 79
  (trip mean 89; rows 2: 168); kernel path 257 / 256 / 399 / 270 (296; rows 2: 351); persistent 2802 / 2799 / 3136 /
  3027. Cause (cost proportional to resident threads): every thread executed `__threadfence_system()` twice per trip.
  Fixed (one system fence per block after `__syncthreads`, the second grid barrier and the `d_y` round trip removed:
  warps store their outputs straight into the mapped reply; trip words read once per block; 16-byte input reads;
  4 rows per warp when 16 would idle warps): **not yet run on a GPU** (a second short run was not approved in this
  session); arithmetic unchanged (u32 wrap), and the start-up self-check still guards it. Not wired into the demo
  (it was not faster). Caveat for serving: while the resident kernel runs, any device-wide synchronisation on that
  GPU (`torch.cuda.synchronize`, `torch.cuda.graph` capture, `empty_cache`, Triton autotuning) blocks forever, so a
  provider with `PGW_SERVICE_EXEC=persistent` must not capture graphs or tune lazily after start-up.
  Graph replay's gate_up (191 us for 136 MB, ~0.7 TB/s) is the largest single provider cost per token.
  Re-run (~$0.1): `PGW_MODAL_APP_SUFFIX=-pw modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 6
  --cmd "python benchmarks/provider_scale_bench.py --presets 7b --tp 1 --rows 1,2,4 --iters 300 --persistent-bps
  1,2,4 --json /tmp/ps.json"` (the bench now records a failing path's error and still reports the others).

## Update 2026-11-23: 7B feature draft head trained on Modal (export bug found; gateway opt-in `draft_backend="feature_head"`)

Qwen2.5-7B-Instruct features (1.21M rows: 2,582 greedy UltraChat answers + 1,261 dataset replies; one A10G job,
~$0.75): the float head reaches validation top-1 0.46 (depth 1), but the exported int8 heads hold alpha 0.19 on
85 held-out answers because one per-row scale over `W_in`'s feature and token halves zeroed the feature half.
Exporter and loader fixed; a train-only rerun from the saved features (~$0.35) is pending approval. Gateway
loads a head file (`draft_backend = "feature_head"`, `draft_head_path`; strict speculation only, defaults off),
bit-identical to plain decoding in tests; 0.6-0.75 ms per head step on CPU (d = 384). Details:
`research/plans/tree-speculation.md` section 9.
## Update 2026-11-22: Qwen2.5-32B co-located Modal run prepared (CPU-only gateway, H100 provider; not run)

`launch.py --model qwen2.5-32b-instruct` (H100 only, `--gateway-cache` required): the provider's 37.5 GB pack from a
provider-side Volume (`prepare_provider_pack`, CPU container, download + pack inside Modal), the gateway side's pack and
114.4 GB hint (not about 78 GB: 4.65M output rows x 24,576 bytes) built once by `--prepare-cache` on a customer-side T4
container (hint on the T4, customer-chosen seed; never on the provider), a free fingerprint preflight before every run,
and a fail-fast in the container without a cached pack and hint. Sizing: provider 48 GiB, gateway 160 GiB (hint and
pack in the page cache), slot 18.6 MB, keep-up divisor 16; plaintext reference's exact integer products on 16 threads
(`PGW_MODAL_REF_INT_THREADS`, same bits); `--mask-source gpu-service` hook for the GPU mask service (refused until
wired); `--dry-run`. Commands, sizes and cost (prep about $1-2 once, short plain e2e about $1.5-2.5 at $7.51/h, TTL 30
min; expected plain 3.5-5 tok/s): `deploy/modal/README.md`, "Qwen2.5-32B co-located".
## Update 2026-11-22: provider for 32B/70B-class: output-row TP, persistent trip worker, lane coalescing (opt-in, CPU-tested)

Provider side of the scaling plan (targets 3-4); what the provider computes and sees is unchanged (masked `x_hat` in,
`y_hat` out, same bytes), and every knob defaults off.

- **Output-row tensor parallel** (`tp_devices`, `PGW_SERVICE_TP_DEVICES='["cuda:0",...]'`):
  `pgw.service.tensor_parallel.ShardedKernel` splits each linear's int8 weight by output rows (balanced, 16-aligned
  boundaries) over N devices in one process; the input is broadcast, shards run concurrently (caller thread + per-group
  helpers, each shard on its own device executor, stream, pinned staging and CUDA graphs), slices are concatenated in
  row order: byte-identical to one device (no reduction). Maskgen entry points concatenate row blocks too. A TP group
  is one scheduling unit for the batcher. A failed shard fails the call only after every shard finished (no partial
  reply). Lane: per-(linear, rows) graphs per shard reading one pinned input, registered as one lane graph; the native
  lane launches all shards and gathers rows in order (`cuda.rs` `GraphShard`, `gather_rows`), self-checked at startup.
- **Persistent trip worker** (`PGW_SERVICE_EXEC=persistent`): `pgw.service.persistent` (NVRTC-compiled resident kernel
  per GPU: block 0 polls a doorbell in mapped pinned memory, grid copies x, wrapping-u32 ring GEMV with 16-byte weight
  loads, coalesced writes of its output columns straight into the shared mapped reply buffer, `done` after a system
  fence) and `rust/pgw-lane/src/persistent.rs` (per trip: device locks, one body copy, doorbell per shard, spin on
  `done`; no launch, no Python). Rows 1-4; larger calls fall to graphs. Startup self-check vs the kernel path; a
  mismatch, an error status or a worker that does not answer disables it (graphs / kernel path serve).
- **Multi-tenant coalescing on the lane** (`PGW_SERVICE_COALESCE=1`): `rust/pgw-lane/src/coalesce.rs` merges calls of
  the same (model, linear, ring) that wait while that key runs into one device call (FIFO, <= lane_max_rows rows); an
  idle key runs at once and alone (window 0, no copy). Replies are cut per call; admission and per-tenant accounting
  stay per call; a failed merged call retries each member alone. (The stream/Python path already coalesced across
  tenants in `BatchScheduler` with window 0.)
- Tests (CPU; fake multi-device = repeated `cpu`, emulated workers on the same memory protocol, the real native lane):
  `tests/service/test_tensor_parallel.py`, `tests/service/test_lane_persistent.py`; Rust: `cargo test -p pgw-lane`
  (coalescer, persistent executor with device-emulating threads incl. TP gather, stall and error disable, graph shard
  tiling). Not reached: nothing ran on a GPU; the CUDA kernel, the TP graph capture and the timings are unverified.
- Ready to run (not run, no Modal spend): `modal run deploy/modal/gpu_bench.py --target h100x4_c32 --timeout-min 30
  --out provider_scale.json --cmd "python benchmarks/provider_scale_bench.py --presets 7b,32b,72b --tp 1,2,4
  --rows 1,2 --iters 400 --json /tmp/provider_scale.json"` (4 x H100, about $16/h plus CPU; ~10-15 min, ~$3-4).
  A 1-GPU subset (`--target h100_c16 --tp 1`) is ~$1. Reports per preset, TP and rows: kernel / graph / persistent
  service time per trip, the GEMV device floor and the overhead above it (target <= 15-25 us; gather <= 5 us).
## Update 2026-11-22: standard deployment in the customer's cloud account; GPU mask service (opt-in)

Decision: the standard deployment runs, in the **customer's own cloud account in the provider's region**, a CPU-only
gateway VM and, for 32B+, a small GPU "mask service" VM that computes the masks `W8 r` itself from the public
weights (no HE, no hint). Up to 14B the CPU-only encrypted factory stays the default. Premises run only a thin client.

- **`pgw factory mask-service`** (`PGW_FACTORY_MASK_SERVICE`, default off; SPEC 7.10; `pgw.factory.mask_service`):
  the existing plain factory (cuda_int8, GPU ChaCha20, per-chunk self-check) plus a **start-up check** (a throw-away
  segment on the production device and chunk size; sampled `W8 r` rows and the full `u W8` of sampled linears
  compared bit for bit with an independent CPU reference; any difference refuses start-up), **device scrubbing**
  after every segment, and refusals (encrypted generation, no self-check, CPU without `mask_service_allow_cpu`,
  unsigned or unbound export). Delivery: the gateway's store on the same host, or the partner export channel (HPKE +
  Ed25519 + destination store) over a small courier, `pgw factory outbox-serve` / `pgw maskstore pull` (unchanged
  import checks; delete local then remote; back-pressure).
- **Freivalds precomputation on the int8 kernel on CUDA** (`ring_vecmat(method="kernel")`, the CUDA default): `u W8`
  through the same digit-plane GEMM as `W8 r` instead of float64 GEMMs (1/32-1/64 rate on T4/L4/A10G). Same bits
  (tests; the start-up check compared `u W8` on the T4); the factory's per-segment check stage at 72B fell from
  0.76 to 0.05 s per layer (ring).
- **Exactness evidence**: CPU tests (rings K32/K40/K48, p31, device-PRNG emulation, corrupted `W r` / `u W` refused,
  buffers wiped, same-host store and export -> pull -> import -> reserve exact); on a Modal **T4** every start-up
  check passed (all linears of the timed layers, ring and p31, `gpu_prng=True`), and the CUDA kernel/mask service
  tests passed (83). Spend about $0.15.
- **Rates** (full model, extrapolated from synthetic layers, 1024-slot segments into an encrypted store; T4): 7B
  164-174, **32B 55-58, 72B 30-40 slots/s** (72B 47-50 with 4096-slot segments, generation only; before the vecmat
  change 32B 27-30, 72B 13-14). Estimated L4/A10G (1.5-1.9x T4 int8): 32B 85-110, 72B 60-90 slots/s vs decode
  demand 33-40 and with-prompts ~90 (scaling plan section 6). This box (4 shared vCPUs, load average ~40) as the CPU
  fallback, Qwen2.5-0.5B pack, cpu_int8, p31: 12-29 slots/s (34 slots per CPU-second); start-up check 200-245 s here
  (one 259-slot throw-away segment over all 96 linears at this box's rate).
- **`deploy/customer-cloud`**: Terraform module for AWS (region/zone, VPC or existing subnet, PrivateLink endpoint to
  the provider, gateway and GPU instance types, model sizing presets, mask supply `cpu_encrypted` / `separate_vm` /
  `same_host`, enrollment code as a sensitive variable in an SSM SecureString read once), security groups (client
  API TLS 8443 from given CIDRs only; gateway -> provider endpoint; gateway -> mask service 8470; no SSH; IMDSv2;
  encrypted volumes; separate counter volume), first-boot scripts (enroll, build the public pack, exchange only
  public keys and certificates via SSM, start containers), sizing and cost table (7B ~$0.85/h CPU-only; 32B ~$2.95/h
  with an L4; 72B ~$5.10/h), "Onboarding in about an hour", GPU-quota and region notes, teardown. `terraform fmt` and
  `validate` pass (Terraform 1.9.8, aws 6.73); `tests/deploy/test_customer_cloud_template.py` checks the contract
  statically. Not applied to any cloud account.
- **Privacy**: unchanged guarantees, now with everything secret in customer-controlled VMs: the provider never gets
  `r`, seeds or `u`; the weights the mask service uses are the provider's public model already; the mask service has
  no route to the provider.
- Not done: the gateway's in-memory mask ring is not fed by the mask service directly (segments go through the
  encrypted store, whose volume bandwidth binds at 72B); no real GPU instance (L4/A10G) measured; no cloud apply.
## Update 2026-11-22: parallel, deterministic gateway attention (v2 "blocked") and bf16 KV storage (opt-in, local)

The bound at 32B-72B (gateway attention over the private KV cache) now has a parallel native kernel. Defaults are
unchanged (`float_ops="torch"`; `"native"` keeps its bits). Code: `rust/pgw-fast/src/attn.rs`,
`NativeFloatOps` / `FLOAT_OPS_NAMES` in `engine/model.py`; tests: `tests/engine/test_native_attention_v2.py`,
`attn.rs` unit tests; benchmark: `benchmarks/attn_cpu_bench.py`.

- **Engine**: units (row, KV head, 256-position block) run on a persistent pool. Per-block partial sums (den in
  float64, acc in float32, each from 0 in position order) are merged in block order. The bits therefore do not
  depend on the thread count, the ISA (explicit AVX-512 and AVX2 kernels: the same 16-lane dot and tree, a
  vector `exp_f64`, no FMA) or the rows per pass. `float_ops="native-blocked"` is the new pinned formulation (v2);
  plain `"native"` (v1) runs on the same engine with one block per row: the same bits as before, now also
  parallel over KV heads. v2 equals v1 whenever the prefix fits in one block (< 256 positions). `PGW_ATTN_THREADS`
  sets the thread count (0 = automatic, as before).
- **bf16 KV** (`"native-bf16kv"`, `"native-blocked-bf16kv"`; each its own reference): the cache stores bfloat16,
  rounded to nearest even like torch's `.to(torch.bfloat16)`, and widens exactly to float32 when read. Half
  the bytes. Quality on Qwen2.5-0.5B is **not measured yet**. The check `test_bf16_kv_quality_qwen05b` (max |delta
  logprob| of the float32-KV top-1 token and top-1 agreement over about 570 positions, 3 blocks; also v2 vs v1)
  is written, but the memory cgroup killed it twice on this shared box. On the tiny models, bf16 KV stays within
  2e-2 of HF.
- **Exactness, tested**: 1-7 threads and ISA caps AVX-512/AVX2/scalar give identical logits; R-row prompt and
  verify passes equal 1-row decoding; masked == plain (tokens and logprobs); speculation identical; HF
  closeness at 1e-4 (2e-2 with bf16 KV); v2 vs v1 within 1e-5; v1 equals the original kernel bit for bit
  (Rust test against the kept `attend_row_ref`). 74 engine tests pass with the new extension.
- **Speed** (this box: 4 shared vCPUs with load average 45-52 from other jobs, so wall clocks are about 3x CPU time
  and thread scaling cannot be measured cleanly; numbers are single-thread CPU time per layer, decode row):
  72B at 2k context, v2: 4.3 ms float32 KV (15.6 GFLOP/s, 3.9 GB/s per core), 3.3 ms bf16 KV (20.5 GFLOP/s); v1
  engine: 8.1 ms. The pre-SIMD build of the same engine took 11.7 ms. 8k context: 19.6 ms float32 / 15.3 ms bf16
  (v1 36.2 ms). 7B at 2k: 2.0 / 1.6 ms. The best 4-thread wall time was 2.5x the 1-thread time (7B at 512 and
  2k). **Extrapolation** at 15-20 GFLOP/s per core: 600 GFLOP/s needs about 30-40 cores, so the target is not
  shown on this box. A 72B row at 2k has 64 units, and 256 at 8k: enough parallelism for 48 cores. Next: measure
  on an idle 16-48 core VM (`attn_cpu_bench.py --threads 1,8,16,32`).

## Update 2026-11-22: exact KV-cache reuse across chat turns (`prefix_cache = "conversation"`, opt-in)

SPEC 9.4b. A request whose prompt (after the chat template) starts with tokens of the previous turn of the same
conversation key (API key + OpenAI `user`, optionally narrowed by `x-pgw-conversation-id`) runs only the rest
through the prompt pass and reserves masks only for it (`(prompt - reused) + max_tokens`; strict speculation: the
same decode term, unchanged pass schedule). Longest-common-token-prefix match; anything else recomputes.
- **Exact by construction, refused elsewhere**: needs `float_ops = "native"` (row-invariant per-row kernels) and a
  `rows_exact` backend (`PlainBackend`, `MaskedBackend`); torch float ops, `mode = "float"` and GPU targets are
  refused by the config and by `Generator.prefix_reuse_supported()`. Tests (`tests/engine/test_prefix_cache.py`,
  `tests/gateway/test_prefix_cache.py`): token ids and logprobs bit-identical to recomputing over 3-5 turns for
  greedy, seeded sampling, variable-row draft and lookup speculation, strict speculation (masked filler and padded
  `spec_mask_rows`), masked and plain, chat and streaming through the gateway; partial/wrong prefix, cross-key /
  user / header isolation, LRU budget, TTL (sweeper thread), wipe, integrity failure (tampering service) and failed
  deferred check discard, cancelled turns not stored, masks and provider row counts per turn.
- **Security**: gateway RAM only (never spilled), one entry per key, taken exclusively, zeroed on eviction /
  expiry / mismatch / refusal, stored only after a clean end with the session's deferred checks joined
  (`release_session_checks` now returns whether they passed), all entries wiped on any integrity failure or noise
  alarm (epoch: running requests cannot store afterwards). New provider-visible signal, stated in SPEC 9.4b: a
  continued turn's prompt pass has the uncached rows only, and timing shows it continued a cached conversation
  (hit/miss bit); kept opt-in and not a wizard preset. Not turned on in the co-located demo: its exact e2e needs
  about 7.5 GB of disk for the 0.5B hint (4 GB free here); the gateway-level masked e2e tests above stand in.
- **Measured** (`benchmarks/prefix_reuse_bench.py`, Qwen2.5-0.5B-Instruct, native float ops + exact integer
  `PlainBackend` on this 4-vCPU box under a load average of about 45 from other jobs, greedy, 6 tokens per turn,
  chat with a system prompt; every turn bit-identical between the two runs). The run was cut by the 30-minute job
limit after turn 4 (each turn took 2-2.5 min on this box), so four turns are reported:

  | turn | prompt | reused | prompt rows (provider) | mask slots reserved | TTFT recompute | TTFT reuse |
  |---|---|---|---|---|---|---|
  | 1 | 68 | 0 | 68 / 68 | 74 / 74 | 50.3 s | 50.9 s |
  | 2 | 111 | 73 | 111 / 38 | 117 / 44 | 53.3 s | 41.8 s |
  | 3 | 139 | 116 | 139 / 23 | 145 / 29 | 51.3 s | 27.3 s |
  | 4 | 176 | 144 | 176 / 32 | 182 / 38 | 61.4 s | 26.6 s |

  Turns 2-4: mask slots 444 -> 111 (-75%), prompt rows 426 -> 93 (-78%), mean TTFT 55.3 -> 31.9 s (-42%); turns 1-4:
  slots 518 -> 185 (-64%). TTFT falls less than the rows because each pass here has a large fixed cost under this
  load (about 17 s per decode pass: 24 layers of call overhead plus the 151936-row LM head); with the provider, the
  saved rows also save masks taken, service GEMM rows and wire bytes of the prompt pass. Absolute times are not
  representative of an idle host.
## Update 2026-11-22: multi-core per-trip kernels for 32B/72B vectors (`PGW_TRIP_SPLIT`, opt-in, local CPU)

Component target 1 of [research/plans/scaling-32b-70b.md](research/plans/scaling-32b-70b.md) (prepare/finish of one trip
over many cores), first part. Default off; exact; no format change.

- **Split prepare** (`rust/pgw-fast/src/trip/split.rs`; `TripLease.prepare(..., split=c)`, `trip_prepare_into(...,
  split=c)`): each row is cut into a fixed partition of 2048-column blocks (depends on `n` only) over `c` tasks of the
  persistent pool, in three passes (max/sum x^2; exact integer sum of q^2 per candidate width; quantize with the
  selected width + mask + pack + x_hat + input-half partials). Merges on the calling thread in block order: max, the
  float `sum x^2` (window start only), wrapping integer sums, ring input half mod 2^K, prime input half mod p. Payload,
  `sx`, `x_hat`, `rhs` are bit-identical to the one-thread kernel for every `c` (plans with the float sum of squares,
  `n < 4096` or `c > 16` vectors run the one-thread kernel). The `r` keystream of a split call is filled over the same
  tasks (`fill_ring_split`, same bits).
- **Split checks**: the prime output half (`prime::output_half_split`, `TripTicket.finish(..., split=c)`,
  `trip_prime_output(..., split=c)`) by fixed 8192-column units, reduced partials summed mod p; the unmask/finish uses
  the existing column split of the pool with `c` tasks (exact).
- **r ahead for any rows**: a lease now derives the next trip's `r` for as many rows as the current trip (was 1-row
  trips only), on `PGW_TRIP_R_HELPERS` helper threads (default 1); single use (used only by the trip that takes
  exactly those rows, else zeroed; wiped on detach/close). Measured (`TripLease.prepare` wall, 300 us between trips, min of 38 trips, 72B
  shapes): down R = 1 220 -> 152 us, R = 2 433 -> 252 us; gate_up R = 1 57 -> 32 us, R = 2 108 -> 57 us (r is ~40%
  of a prepare). Medians on this overloaded box were not better (the helper is not scheduled in time).
- Wiring: `PGW_TRIP_SPLIT=c`, `PGW_TRIP_SPLIT_MIN_ELEMS` (default 8192) on the native per-trip path in `backends.py`.
- **Native lane + finish per trip** (`PGW_TRIP_DRIVER=1`, with `PGW_TRIP_NATIVE` and a lane client; default off):
  `TripTicket.lane_finish` sends the payload on the native `TripChannel` and reads the reply straight into the finish
  (one GIL-free native call after `lease.prepare`; no response objects, no zero-filled reply buffer, no copy), via
  `LaneComputeClient.trip_ticket` (same eligibility, pool, refusal/error handling and statistics; a call the lane does
  not serve goes to the wrapped client with the same request, the ticket untouched). Noise-monitor every-call checks
  stay inline between the two calls; sampled statistics unchanged. Test: `test_native_driver_over_the_lane` (masked ==
  plain, byte-identical payloads and noise statistics with/without it, refusals and a dropped connection fall back).
- Benchmarks: `native_trip_floor_bench.py --preset 7b|32b|72b --rows R --cores c [--r-offcore] [--stat min]`,
  `trip_native_bench.py --rows R --split c --r-ahead 0|1` (payload hashes compared across runs); presets for Qwen2.5-32B
  (64 L, 5120/27648, 40/8 heads) and Qwen2.5-72B (80 L, 8192/29568, 64/8), vocab 152064 (`PGW_BENCH_VOCAB` shrinks the
  vocabulary of the synthetic pack for disk).
- Tests: `tests/native/test_trip_split.py` (identical bits for c = 2, 3, 4, 7 vs one thread, ring and p31, genuine
  replies pass, tampered replies fail identically with zeroed outputs, r ahead for 1- and 2-row trips identical and
  single use, non-finite before any row, ticket single use), `test_native_split_equals_plain_and_bytes_identical`
  (engine: masked == plain, byte-identical payloads and noise statistics), cargo `trip::split::tests`.
- Measured single-core kernel cost per 72B trip (min of 150, K32A12C27, 4 vCPU dev box): gate_up R = 2 prepare 39 us,
  finish 79-84 us, p31 output half 41 us; down R = 2 prepare 143-172 us, finish 12 us; qkv R = 2 prepare 41-44 us,
  finish 15 us. Kernel floor per token on one core (r off-core, R = 2): 7B 12.9 ms, 32B 48 ms, 72B 61-64 ms; R = 1:
  8.4 / 30 / 34 ms. `r` (8 bytes of ChaCha20 per element, format-fixed) at 72B R = 2 is about 36 ms per token of helper
  time, so 72B needs about one dedicated helper core for `r` at 25 tok/s.
- **Not reached**: the multi-core speed-up could not be measured here: the box ran at load average 40-57 on 4 vCPUs
  (other jobs), the pool workers were rarely scheduled in time (the calling thread took ~80% of the tasks: 100 vs 121
  us of its own CPU for a 72B down prepare at c = 4) and wall times did not improve. Needs a quiet many-core VM
  (with `PGW_POOL_SPIN_US` raised so the workers stay awake between trips). The <= 20 us Python per trip target is
  not reached: profiled at 0.5B shapes over the in-process reference lane (`trip_native_bench.py --lane --driver
  --profile`), Python self time is still ~50-60 us per trip on this loaded box, mostly `_prepare_trip_lease` /
  `forward_many` / `_finish_lease` bookkeeping, and the driver did not lower main-thread CPU measurably (1.80 vs 1.94 ms
  per 8 trips, noise). A per-layer native driver (prepare -> lane -> finish for consecutive linears with no Python
  in between) was not built.

## Update 2026-11-22: scaling design for Qwen2.5-32B and 70B-class targets (model only, local)

Write-up: [research/plans/scaling-32b-70b.md](research/plans/scaling-32b-70b.md); model
`benchmarks/scaling_model.py` (tests: `tests/benchmarks/test_scaling_model.py`), calibrated on the measured 7B
native-kernel floor and the 7B H100 run (lane 0.26 ms per trip; tau 1.19 at R = 2 -> alpha 0.49). Estimates, ctx
2048, alpha 0.70 (0.49): today's paths 2.1 tok/s at 32B and 1.2 at 72B (single-core attention); engineered
gateway + service + network floors + fast draft: **32B 40 (33)**, 72B 27 (23); plus fused 2 trips per layer:
**72B 33 (29), Llama-3-70B 31 (27)**. The binding component at 32B-72B is the gateway's attention over the
private KV cache (655 KB per context token per pass at 72B), then trips x (RTT + service). Minimum targets:
<= 20 us orchestration per trip and trips split over 16 cores, attention >= 100 GB/s and 600 GFLOP/s, RTT <= 30
us, service <= 15-25 us per trip (persistent kernel), TP 4 with a <= 5 us gather (no all-reduce: output-row
split), draft step <= 4-5 ms; for 70B-class also fused chains. Masks at 72B: 28 MB per slot, 1.1 GB/s decode-only
and 2.6 GB/s with prompts; recommended factory: a plain `W8 r` factory on a customer-cloud GPU (0.2 of an L4, no
hint) instead of today's MLWE factory (~44 cores and a 172-345 GB hint per 72B stream). Not reached: nothing was
built or measured at 32B/72B.
## Update 2026-11-22: strict tree speculation and a few-MB target-feature draft head (engine, opt-in)

`research/plans/tree-speculation.md`. `Generator(spec_tree=[(parent, rank), ...])` verifies a public fixed token
tree of R rows in one pass with tree attention (torch and native float paths, no kernel change), bit-identical
to plain decoding (greedy and seeded), with R rows per pass and the strict L-pass schedule unchanged and fixed
gateway work per pass. `FeatureHeadDrafter` drafts from the target's final hidden state, which the gateway holds
in plaintext (5.5 / 9.7 / 13 MB per draft step at 1.5B / 32B / 72B vs 490 MB for the 0.5B draft). Findings:
for the 0.5B draft (alpha 0.81 on the 1.5B, 0.78 on the 3B) the best tree of <= 8 rows is the chain; a head
trained on local data overfits (held-out top-1 0.13), so a real head needs offline EAGLE-scale training; the
32B/72B model says a 0.5B draft is slower than plain, and 25-30 tok/s at 32B needs 0.3-0.5 ms trips, a 2L+1
schedule, per-row trip work spread over spare cores, and a head at alpha ~0.8 with R = 8.
## Update 2026-11-22: fused2 prototype, 2 provider trips per layer (WP-B4, opt-in, new reference; not promoted)

`pgw.engine.fused2` (default off; the gateway still refuses `schedule = "fused2"`): trip A = o + residual add +
gate_up, trip B = down + residual add + next qkv, RMSNorm's `1/rms` applied by the gateway after the product,
residual in fixed point on the o / down output grid, norm gains folded into the next linear at pack time.
Trips per token 4L (+1) -> **2L + 1** (70B 321 -> 161). Same elements on the wire and in the store; ring K40
(1.25x bytes) or K48 (1.5x); no extra factory products (masks reparametrized, `r_R = v - W1 r_x`); both stages
checked with Freivalds over F_(2^61-1). Plain and masked (in-process factory and provider) backends,
masked == plain bit for bit, tamper tests (`tests/engine/test_fused2.py`). Quality on Qwen2.5-0.5B (1020
wikitext tokens): ppl 23.63 (float 23.67, int8 reference 23.51), but top-1 = float 93.0% vs 96.4% for the int8
reference. **Not no-compromise as measured**; next: a folding variant that closes the top-1 gap, then 7B.
Design, numbers, follow-ups: `research/plans/fused2-two-trips-per-layer.md`.

## Update 2026-11-22: the 7B CPU-only-gateway speculation run is ready to launch (Modal launcher, opt-in)

One co-located run measures Qwen2.5-7B on an H100 provider with a CPU-only, weight-free gateway (every linear remote,
native float ops, native per-trip path, p31 check), strict speculation R = 2 with the Qwen2.5-0.5B draft
(`native_int8`, pipelined m = 0) and, in the same container, plain decoding as the A/B. Command, sizing, cost
(about $6.7/h, $2.5-3.5 per run, TTL 35 min) and the expected outcome: `deploy/modal/README.md`, "CPU-only 7B gateway,
strict speculation, H100 provider". Not run yet at 7B.

- Launcher: `--draft-backend`, `--spec-pipeline-steps`, run-time `--spec-mask`, `--trip-native`, `--prime-check`
  (refused with `--spec-mask`: padded calls refuse p31; unpadded R = 2 costs the same 2 slots per pass), `--ab-plain S`
  (restart the gateway without speculation on the same store and keys, then chats and an S-second window),
  `--gateway-cpus/--provider-cpus/--gateway-mem-gib/--provider-mem-gib`, `--gateway-cache`, `--prepare-cache`.
- Reports per phase: streamed decode tok/s, TTFT, tokens per verify pass, drafter hit rate and step time, trips per
  token and trip split, gateway CPU per trip, slots per token, factory production vs consumption; exactness of every
  phase is part of `ok`. New gateway-local admin counters `speed.spec_stats` / `speed.drafter_stats`.
- Gateway cache (a Volume of public artefacts): the pack (reused only with `PGW_SETUP_REUSE_PACK=1`, library default
  off, and only if it fully verifies to the enrolled fingerprint) and the hint `H = W8 A` (public weights times a
  public matrix whose seed the customer side chose; no secret). The 7B's ~20 min CPU hint build moves to a one-off T4
  job. Building the hint on the provider was rejected: a provider-supplied `H` would allow a selective-failure attack
  on the per-group secrets unless re-derived. The factory's segment log now names the check field.
- Validated on a T4 at 0.5B (self-draft, plumbing only): two runs exact in every phase, every segment p31, cache
  restore 2.7-9.2 s for 8.4 GB, plain-phase restart 10.5 s; about $0.2 spent.

## Update 2026-11-22: p31 extents through native trip leases, W r decrypted ahead, fused duplicate check (W2, opt-in)

- **p31 leases**: `TripLease(..., field=2**31 - 1)` serves prime-field extents on the native per-trip path; the prime
  input half is fused in `lease.prepare` and the output half (range + `u . yq == rhs mod p`) runs synchronously inside
  `ticket.finish` on the trip's `W r` *before* the finish kernel unmasks it (never deferred: no result is used before
  it is verified; a failure zeroes `W r` and the outputs). Leased only to a backend with `prime_check`
  (`trip_lease(..., prime_ok=True)`); without it the ordinary path refuses the extent before any row. Format confined
  to `tripnative::check_iface` (`field`, `sync_only`, `prime_output`).
- **`PGW_TRIP_DECRYPT_AHEAD=1`** (default off): the chunks the linear's *next* lease will take (same rows, in order,
  never past the reservation's extent; also the next extent / the next windowed part) are read, authenticated and
  decrypted on a helper thread with native AES-256-GCM, GIL released (`pgw_fast.chunk_open_into`, byte-identical to
  `SegmentFile.decrypt_into`). Each buffer goes to exactly one lease or is zeroed (close, mismatch, failure); an
  authentication failure raises when the lease needs the chunk (fail closed, no row of it handed out).
- **Fused duplicate check**: `lease.prepare(..., dup=noise.packed_dup_params(n))` computes the monitor's duplicate-row
  fingerprints in the same native call; `noise.guard_packed_keys` records them (same keys, bookkeeping, alarms).
- Measured (4 shared vCPUs, heavily loaded by other jobs; `decode_cpu_bench.py --local off --float-ops native
  --wnorm-frac 0.3 --service-delay-ms 0.25 --masked-head`, 2 layers, extrapolated, process CPU ms per 7B token):
  base native 75 / 134 (noisy), this branch ring 74-75, **p31 leases 59-68**, p31 + decrypt ahead 64-84 (no
  measurable gain here: the bench's 1-row chunks and window parts reserved lazily leave little to prefetch, and the
  helper's GIL hand-offs cost the main thread about what they save). 0.5B: 44-47 ring, 23-50 p31 variants (noise).
  `trip_native_bench.py --preset 7b --masked-head` (trips only, full model): main-thread CPU ring 25-33 ms, p31 24-31
  ms; process CPU ring 40-46 ms, **p31 29-37 ms** (no deferred-check helper). Not reached: <= 25 ms of gateway CPU per
  7B token on this box.
- Tests: `tests/engine/test_trip_native.py` (p31 leases: masked == plain, byte-identical payloads and statistics vs
  PGW_TRIP_FAST, tampering at any position / out of range detected through the lease with the row consumed, no lease
  without `prime_check`, field validation; decrypt ahead: native open == `decrypt_into`, ciphertext/tag tamper
  refused, leases fed from ahead buffers byte-identical over 330 tokens across parts and segments for ring and p31,
  ahead buffers zero after use and on close, tampered ahead chunk fails closed at the lease boundary).

## Update 2026-11-22: one-vector Freivalds check over F_p (package W2, opt-in, local only)

New opt-in segment format `check_field = "p31"` (SPEC 5.1, note `research/notes/prime-field-freivalds.md`):
one secret vector per linear over `F_p`, `p = 2^31 - 1`, checked on the unmasked `xq` / `yq`
(`u . yq == (u W8 mod p) . xq mod p`, plus the range check). Soundness per call `>= 30.99999999` bits with
one vector (needs `2B < p`, true for K32A12C27 at any width) against 21 bits for today's 7 ring vectors;
it also catches a wrong `W r`. Default off: factory `PGW_FACTORY_CHECK_FIELD=p31` writes it, gateways need
`prime_check = true` (`PGW_GATEWAY_PRIME_CHECK`) and otherwise refuse such segments before any row is used;
old segments keep the ring check and both formats run side by side. Kernels (`pgw_fast`):
`trip_prepare_into(..., field=2**31 - 1)` (input half fused in prepare), `trip_prime_output(y_hat, wr, u, rhs,
k, bound)` (range + output half, exact modular arithmetic, AVX-512 / AVX2 / portable). Scope: TripEngine paths,
disk store and in-process RAM ring; speculation padding, non-trip fallbacks, MoE oblivious and `ringshm`
refuse prime extents (fail closed). Tests: `tests/engine/test_prime_check.py` (exactness, single-entry /
structured / adversarial-within-cap tampering, model logits bit-identical, interop, defaults, store round
trip), Rust `prime::tests`.

Measured here (`native_trip_floor_bench.py --preset 7b --wnorm-frac 0.3`, 113 trips, ms per token, two runs):

| | ring, c = 7 (today) | p31, c = 1 |
|---|---|---|
| check vectors read per token | 66.6 MB | **9.5 MB** |
| `uW` per segment at rest / derived `u` in RAM | 23.4 MB / 43 MB | 3.3 MB / 6.2 MB |
| prepare (+ input half) | 5.9-6.5 | 6.1-6.2 |
| prime check (range + output half, synchronous) | - | 2.2 |
| finish | 4.0-4.5 (head checked inline) | 2.5-2.6 (no check) |
| **main core** | **14.7-16.2** | **15.6-15.9** |
| **helper core** (deferred output half) | **6.7-7.4** | **0** |

Per kernel (one row, warm): output half m = 4608: 9.1 -> 3.9 us; m = 37888: 44.6 -> 16.5 us; the 7B head
m = 152064: 376 -> 74 us; prepare n = 18944 with the input half 50-53 us (ring 7) vs 49-51 us (prime). The
total gateway CPU per token drops by the helper core's 6.7-7.4 ms; the check now runs synchronously (never
deferred). Run on a helper core instead (keep `W r` until it returns) it would take the main core to about
13.5 ms. `trip_cpu_bench.py --preset 7b --layers 1 --wnorm-frac 0.3 [--prime]` (Python TripEngine path, real
encrypted store, stub service; calling-thread CPU per call in us, two runs each): qkv 195-223 -> 165-179,
o 155-176 -> 125-154, gate_up 227-267 -> 209-222 (ring: plus its deferred output half on the helper),
down 326 -> 245-269; per layer 904-992 -> 769-799 us on the calling thread and nothing on the helper.

## Update 2026-11-22: factory HE on the gateway GPU (opt-in)

`PGW_FACTORY_HE_ENCRYPT=gpu_ntt` / `PGW_FACTORY_HE_DECRYPT=gpu_ntt` (library and demo default off) run the encrypted
factory's encryption/decryption on the gateway's own GPU (`pgw.hemask.gpuntt`), self-tested and cross-checked bit for
bit against the CPU paths at start-up, CPU fallback on any mismatch or error. Two exact co-located Modal runs
(`--gateway-gpu`, 240 s back to back): factory 2.0 -> 0.9 gateway cores, sustained decode 30.8-31.3 tok/s
token-weighted (median 31.3-31.6, min 25.5-25.7; 1-2 chats below 26 tok/s), keep-up unchanged (consumption 50-51
vs commit-to-commit 46-48 slots/s, stock min 3836-4028). Not the demo default: while the factory runs unyielded its
kernels time-slice with the decode graphs (separate processes; stream priorities do not help across contexts):
local linear 79 -> 108-119 us/call, warm single chat 25-27 tok/s during the stock ramp. The customer target is now
GPU-free, so this stays an opt-in. Details: `deploy/modal/README.md`, "Factory HE on the gateway GPU".
## Update 2026-11-22: native per-trip path, one native call per side of the lane call (package W2, local only)

`PGW_TRIP_NATIVE=1` (with `PGW_TRIP_FAST=1`; library default off). The mask reservation hands the gateway a
*lease* (`pgw_fast.TripLease`, `rust/pgw-fast/src/tripnative.rs`): the next rows of one linear in decrypted `W r`
chunks of one extent (at most `PGW_TRIP_LEASE_BYTES` = 1 MiB / `PGW_TRIP_LEASE_ROWS` = 16 rows), the extent's `r`
stream key and its Freivalds vectors. Per trip: `lease.prepare` (take the row, derive `r`, constant-time
quantize + mask + pack, Freivalds input half) before the lane call and `ticket.finish` (unmask with `W r` read in
place, range check, Freivalds sync or deferred, rescale, wipe) after it. `r` of a lease's next row is derived ahead
on a helper thread (`PGW_TRIP_R_AHEAD`, on by default with the native path). Not reached: <= 20 ms main-core CPU
per 7B token.

- Single use and accounting: the lease position advances under its lock before any secret is derived; while a
  lease is attached the linear's cursor is the lease's (`remaining`, `taken`, burn-on-close unchanged); any other
  path that takes rows of that linear detaches it first; `close()` closes leases (in-flight trips fail closed with
  `MaskReuseError`). Unused lease rows are zeroed with their chunk (never sent, like a dropped prefetch stage).
- Same order and checks as `PGW_TRIP_FAST`: window `ensure` / exhaustion pre-check and the extent's soundness check
  before any row is taken, non-finite activations refused before the row is taken, Freivalds then noise-sampling
  draws, every-call noise checks inline (sampled statistics inline, or async with `PGW_NOISE_ASYNC_SAMPLED`),
  uuid4 request ids. Freivalds format confined to `tripnative::check_iface` (calls the existing trip kernels).
- Served: `MaskReservation` on disk segments (also through `WindowedReservation`). In-memory ring segments and
  `ChainedReservation` fall back to the `PGW_TRIP_FAST` path.
- Tests (`tests/engine/test_trip_native.py`): masked == plain bit for bit, byte-identical provider payloads and
  identical noise statistics on/off (windowed store reservations crossing chunks, extents and parts; K32, K40,
  K48, K24), single use under transport failure / tampered / malformed replies, soundness and non-finite refusals
  take no row, close mid-trip fails closed, detach hand-over, ticket single use and wipe.
- Measured (dev box, 4 shared vCPUs, loaded; stub service): trips only (`benchmarks/trip_native_bench.py`, model
  order, instant stub, extrapolated to the full model): 7B 48 -> 27 ms per token (median per trip: qkv 214 -> 137,
  o 146 -> 101, gate_up 203 -> 144, down 372 -> 229 us; masked head about 1 ms either way); 0.5B 34 -> 13.5 ms.
  Decode (`decode_cpu_bench.py`, 2 layers + masked head, native float ops, `--wnorm-frac 0.3`, stub delay 0.25 ms,
  extrapolated): 7B 127 -> 78 ms process CPU per token (2-layer run 10.75 -> 7.61 ms), 0.5B 58 -> 38 ms; outputs
  repeat-identical and equal on/off.
- Left per trip on the main core: the lease refill (Python AES-GCM decrypt of `W r` chunks, about 1.5-2.5 ms per 7B
  token, inherent to a disk store; a RAM ring lease would remove it), the masked head (608 KB row per token, 1-row
  leases), and 40-60 us of Python glue per trip (noise `guard_packed` ~20 us, request object, statistics,
  deferred-check bookkeeping).

## Update 2026-11-22: weight-free CPU gateway at 7B shapes (package W1, local only)

Write-up: [research/plans/weight-free-cpu-gateway-7b.md](research/plans/weight-free-cpu-gateway-7b.md). Not reached:
<= 8 ms of gateway CPU per 7B token. All four linears remote plus a masked head (113 trips per token), measured on
the 4-vCPU dev box with a stub provider: about 2 ms of gateway time per layer (Python path), 57-65 ms per token
extrapolated, 9.5 tokens/s at a 0.25 ms round trip; the native kernels alone are ~16 ms main core + ~7 ms helper core
per token (66.6 MB of Freivalds vectors and 6.7 MB of r keystream per token dominate).

- **Masked remote LM head (opt-in)**: `pgw.engine.masked_head` (`MaskedHeadPack`, `enable_masked_head`), linear
  `"<L>.lm_head"` (new `LinearKind.LM_HEAD`, never in `LINEAR_ORDER`), int8 per-row head, domain-separated
  fingerprint, every row range- and Freivalds-checked (synchronously: the last linear). New pinned reference
  (int8 head); masked == plain bit for bit in `tests/engine/test_masked_head.py`. Wired end to end, opt-in (package
  W3, below): gateway `head_mode = "provider_int8"`, service `masked_head = "also"|"only"`, factory `masked_head`.
- AVX2 Freivalds output half (identical results): gate_up 140 -> 47 us, 7B head 668 -> 470 us per call.
- `PGW_EMBED_COMPACT=1` (opt-in): bfloat16-exact embedding table kept in bfloat16 (2.18 -> 1.09 GB RAM at 7B).
- Storage at 7B (K32): 0 linear weights, 1.09 GB embeddings, 6.17 MB of `W r` per token (185 MB/s at 30 tokens/s;
  a 10 s RAM ring is 1.85 GB), ~38 GB of hints only with a gateway-side encrypted factory, KV 115 KB per token.
- Next levers (not done): a native per-trip path, a one-vector Freivalds check over a prime field (66.6 -> 9.5 MB of
  check vectors per token), r derived off the critical path, `W r` stored mod 2^29 (-9.4%).
- Bench fix: zero synthetic weights made the width selection evaluate all 11 widths (prepare ~2x too slow in earlier
  zero-weight benches); `--wnorm-frac 0.3` gives real norms.
## Update 2026-11-22: Qwen2.5-1.5B and 7B co-located, measured (package L1)

Both larger targets now run the co-located demo end to end, exact in every run (logprobs bit-identical, text equal,
repeat, stream and all sustain chats identical). Details and tables: `deploy/modal/README.md`, "Larger targets".

| config (Modal) | decode tok/s | TTFT | trips/token | factory alone / consumption |
|---|---|---|---|---|
| 1.5B, 2xT4, gateway GPU, attention_down | 20.0 | 0.81 s | 28 | 118-142 / 43.6 slots/s |
| 7B, 2xT4, gateway GPU, attention_down | 10.0 | 1.49 s | 28 | 39-46 / 18.6 slots/s |
| 1.5B, 1xT4, CPU-only gateway, all linears remote | 9.5 | 1.72 s | 112 | about 57 (2 workers) / 22.8 slots/s |
| 7B, 1xT4, CPU-only gateway, all linears remote | 5.0 | 3.1 s | 112 | 18 / 9.1 slots/s |

- Fixed: the co-located enrollment code listed the 0.5B whatever the target (1.5B/7B pairs could not enroll).
- One thread-safe `ByteBudget` per service now covers the compute routes, `/pgw/v1/maskgen` (which had none) and the
  bulk maskgen channel; the demo uses the bulk channel by default (library default off).
- `PGW_HINT_DEVICE=cuda` (library default off): the one-time encryption hint on the gateway's GPU, same bits,
  self-checked; 7B 213 s vs about 20 min on 16 vCPUs.
- 7B customer storage as run: hint 34.2 GB, int8 pack 7.6 GB, embedding + LM head 2.2 GB, masks 5.56 MB per slot.
- Not reached: 30 tokens/s at 7B. The T4s are bandwidth-bound (about 3.8 GB per token on each side with a gateway
  GPU); the CPU-only gateway pays 112 trips of about 1 ms plus about 0.8 ms of gateway CPU per trip.

## Update 2026-11-22: mask factory keeps up with back-to-back chats (package S5)

The S4 demo ran out of masks after 130 s of back-to-back chats (HTTP 503 `mask_exhausted`): all-or-nothing yield
at 1000 slots, refill only below low water 1000, 30 s scheduler poll, 4096-slot segments. Now 302 s of
back-to-back chats (146 chats, about 51 slots/s consumed) with no exhaustion: the stock settles into a sawtooth
of about 4100-6300 slots (a 2048-slot segment every 40-45 s, segment-to-segment production 50.6 slots/s),
streamed decode 31.7 tokens/s token-weighted (median 32.4), every chat exact. Details: `deploy/modal/README.md`,
"Factory keep-up under back-to-back chats".

- `PGW_FACTORY_YIELD_FULL_SLOTS` / `_PERIOD_S` / `_HORIZON_S` / `_FULL_HORIZON_S` (`pgw.core.factory_yield`;
  library defaults off = all or nothing as before; co-located demo on): partial yield whose duty cycle grows as the
  stock falls, never below the stock covering 45 s of observed consumption, full yield when the stock covers 120 s
  (occasional chats unaffected). Same masks, requests and bytes; the timing depends on decode activity and stock,
  both of which the provider can already derive from its own trips and maskgen requests.
- Demo factory sizing: 2048-slot segments, refill below 6144 to 8192, 2 s poll.
- `--sustain-s` streams every chat, checks all are identical, samples the stock and reports consumption vs the
  factory's production (`keep_up`); `factory.committed` fixes the empty `segments` list for long segments.
- Over a window that starts at the top of the sawtooth the factory's committed slots read below consumption
  (40.9-47.6 vs 49-52 slots/s): the stock starts above its equilibrium; slope -1.1 to -2.2 slots/s. Not done:
  moving the factory's encryption/decryption to the gateway GPU (lever 3).

## Update 2026-11-22: decode at 30 ms per token co-located (package S4)

Reached in the co-located demo: 29.9-30.3 ms per decoded token (33 tokens/s after the first, streamed), from 36,
exact. The remaining contention was the encrypted mask factory (its maskgen on the provider's GPU and its
ciphertext traffic), measured by pausing it inside one run (36 -> 32 ms per token). Details:
`deploy/modal/README.md`, "Decode rate, contention and the factory yield".

- `PGW_FACTORY_YIELD_FILE` (library default off; co-located demo on at 1000 slots): the factory waits before
  each encrypted batch and provider request while the gateway runs masked passes and the stock is above the
  threshold; below it, full speed. Same masks, requests and bytes; nothing new for the provider.
- `PGW_SERVICE_TRIP_STREAM_PRIORITY=high` (provider, library default off; demo on): decode graphs on the GPU's
  highest-priority stream. Scheduling only.
- `--lane uds` is now the co-located default (1-row trip 0.29 ms vs 0.44 ms on TLS in the combined config).
- The e2e reports TTFT and the streamed decode rate (`warm_stream_*`); `PGW_MODAL_AB` runs in-run A/B phases.
- Core pinning is impossible on Modal (`sched_setaffinity` EPERM under gVisor). Local GPU linears stay at about
  79 us per call with or without the factory: not contention.

## Update 2026-11-22: lean per-trip gateway path and the co-located UDS lane (package T3b)

Not reached: 30 tokens/s. Three exact Modal runs the same hour (T4, attention_down, native float ops, factory
capped; every run logprobs bit-identical, text equal, repeat identical): per decoded token 53-55 ms (TLS lane,
new paths off), 52-54 ms (TLS, new paths on), 53 ms (UDS lane, new paths on); warm chat 12.4 / 13.1 / 11.9
tokens/s. The gains are within Modal's host-to-host noise. Details: `deploy/modal/README.md`, "Lean per-trip path
and the UDS lane".

- `PGW_TRIP_FAST` (library default off; co-located demo on): a one-call pass (decode) skips the batch
  bookkeeping; the extent's soundness check runs inside `take_checked` (new `verify` hook) before any row is
  taken, so the separate pre-check lookup is gone; lean finish; cached constants for one-call local linears.
  Same masks, draws, payload bytes, checks on every row, outputs and wiping (tests/engine/test_trip_fast.py).
  Dev box, per trip: prepare 339 -> 279 us, finish 155 -> 153 us; local linear 330 -> 317 us per call.
- `PGW_NOISE_ASYNC_SAMPLED` (default off, also off in the co-located demo: the sampled payload leaves before its statistics finish): a sampled call's every-call checks stay inline, its
  statistics (about 400 us) run on a worker on wiped copies, in call order, joined before the next prepare (an
  alarm blocks the next payload, fail closed). Statistics identical to inline (tested).
- `launch.py --lane uds` (co-located, opt-in): the product's same-host UDS lane between the two party users;
  the service accepts the gateway user's uid only, the socket's directory is 0710 and the socket 0660 to that
  user's group. 1-row trip 0.44-0.47 ms (TLS: 0.54-0.57 ms); not made the default: no clear tokens/s gain in
  the measured run.
- What is left per trip on the dev box is mostly cache-cold native work (the finish kernel streams `u`, `W r`
  and the scale: 22 us warm, about 100 us in place) and the store's amortized derivation, not Python.

## Update 2026-11-22: trip latency floor and gateway per-trip CPU (package T3)

Not reached: the 1-row trip stays at 0.57 ms in three exact Modal runs (12.0-13.0 tokens/s back to back). The
provider side is already native; under Modal's gVisor the TLS lane transport alone costs 0.28 ms per trip (0.03-0.05
ms on a plain VM) and polling makes it worse. Spin knobs added, off by default; details in `deploy/modal/README.md`,
"Where a 1-row trip's 0.57 ms goes". Gateway per-trip CPU measured in place (dev box, `decode_cpu_bench.py`
wrappers): prepare about 340 us (mask take 85, of which the stage copy and amortized derivation; precheck 75;
request 28; noise guard 18 every call, 400 per sampled call; native prepare 30-40) and finish about 120 us (native 20);
one redundant `remaining()` per trip removed. Collapsing these into one native call per side is the remaining work.
## Update 2026-11-22: native pinned float path (package T3)

Per decoded token of one Qwen2.5-0.5B chat co-located: **46-53 ms (from 62-64 in a same-day torch A/B); back to
back 13.4-16.6 tokens/s (12.7), warm chat 13.7-14.9 (11.6)**, every run logprobs bit-identical, text equal, repeat
identical. Details: `deploy/modal/README.md`,
"Native pinned float path".

- **New pinned float reference (decision delegated by the user, 2026-11-22), opt-in.** Model setting
  `float_ops = "native"` (or `PGW_FLOAT_OPS=native`, `TransformerModel.set_float_ops`): RMSNorm, residual adds,
  RoPE, attention over the KV cache and the SiLU gate run in `pgw_fast` (`layer.rs`, `attn.rs`) in a fixed order
  of IEEE operations (no FMA, no libm `exp`; AVX2 and baseline builds give the same bits), one native call
  between consecutive linears. Every output row depends only on its own inputs and the cache prefix: the bits do
  not depend on the ISA, torch's thread count or the rows in a pass (speculative verify rows equal decode rows by
  construction). RMSNorm, attention and SiLU differ from torch in the last bits (RoPE and residual adds are
  torch's bits), so the gateway and its plaintext reference must both select it; the HF closeness tests pass with
  it (`tests/engine/test_native_float.py`, plus the existing suites run under `PGW_FLOAT_OPS=native`). The final
  norm and the LM head are unchanged. Library default stays `torch` (no golden changed); the co-located Modal demo
  now defaults to it (`launch.py --float-ops native`, reference via `plaintext_reference(float_ops=)`) after an
  exact e2e. Privacy unchanged: local gateway math only.
- Per decoded token now (about 50 ms): trips 13 ms, local linears about 10 ms, mask prepare/finish about 8-10 ms,
  LM head 4 ms, float ops and engine Python about 8-10 ms (from about 20). Not reached: 30 tokens/s.
## Update 2026-11-22: native local linears and the LM head on the gateway GPU (package T3)

Goal 30-40 tokens/s for one Qwen2.5-0.5B chat co-located with `--gateway-gpu`; reached **14.5-16.1 tokens/s back
to back** (12.1-13.5 warm chat; the same code with the new paths switched off: 10.9 on that run's host), every run
logprobs bit-identical, text equal and repeat identical. Details: `deploy/modal/README.md`, "Gateway GPU: native
local-linear call, fixed-order LM head".

- Local linears: one GIL-holding native call per linear (`pgw_fast.local_gpu_linear_into`: prepare into pinned
  memory, `cuGraphLaunch` through the driver, spin, range check and rescale, buffers wiped), a lean self-checked
  1-row path in the backend, and measured T4 launch configurations (down 66 -> 46 us). In the running gateway
  220 -> 99 us per call, 15.9 -> 7.1 ms per token (the device floor is about 22 us per call; the rest is the busy
  container). Integer results unchanged: start-up self-check per linear against the CPU reference.
- LM head (`lm_head_device = "cuda"`, opt-in, library default `cpu`; on in the co-located demo with
  `--gateway-gpu`): the existing fixed summation order as one Triton kernel with `mul.rn`/`add.rn` (no FMA, no
  flush to zero), so the logits are the CPU kernel's bit for bit; the plaintext reference needs no change and
  stays on the CPU. Start-up self-check, else the CPU. About 4 -> 1.4 ms per token. The final norm and sampling
  stay on the CPU. HF closeness tests and goldens unchanged (library default).
- Privacy unchanged: the GPU is the gateway's alone (per-UUID visibility, 0600 device nodes); nothing new
  reaches the provider.
- Per decoded token now (about 48 ms): trips 12 ms, the masked trips' gateway work 10 ms, local linears 7 ms,
  LM head 1.4 ms, float ops and Python about 17 ms. Not reached: 30 tokens/s.

## Update 2026-11-22: gateway GPU and pinned float threads (package T2)

Goal 30-40 tokens/s for one Qwen2.5-0.5B chat co-located; reached **13.3-13.6 tokens/s** (warm and back to back,
from 10-13.5), every run logprobs bit-identical and repeat identical. Details: `deploy/modal/README.md`, "Gateway
GPU, pinned float threads, factory cap".

- Exact local linears on a gateway-owned GPU (`local_linears_device = "cuda"`, `pgw.engine.local_gpu`; co-located
  `launch.py --gateway-gpu`, off by default, two T4s, one per party, each process sees only its own). Masked trip
  with `r = 0` on the gateway's device, self-checked per linear against the CPU. Halves the local GEMVs in
  isolation; in the running gateway about 10 ms/token either way, so little net gain for twice the GPU cost.
- **Pinned float formulation (decision delegated by the user, 2026-11-22).** Torch's CPU SDPA gives different bits
  at different thread counts on Modal's AVX-512 hosts, so the reference's bits depended on the host's default
  thread count. The co-located demo now pins both the gateway's process and the e2e's plaintext reference to the
  same torch thread count (`launch.py --float-threads`, default 1; 0 = the old behaviour): the reference's bits
  come from that pinned formulation, and the gateway's `float_threads = 1` compares 1 with 1. Privacy unchanged;
  the e2e compares against the pinned reference (bit-identical in the measured run). Library defaults (and the
  test suite's goldens and HF closeness tests) are unchanged: the pin is a deployment setting.
- Also: q and k rotated in one call (same bits), leaner GPU replay, factory cap knobs measured
  (`PGW_FACTORY_CPU_THREADS=3`, `PGW_FACTORY_NICE=10`: 120 slots/s on 3.9 cores while decoding, 9x the token rate),
  an opt-in stack sampler (`PGW_STACK_SAMPLER`), `benchmarks/local_gpu_bench.py`, `benchmarks/sdpa_threads_bench.py`.
- Per decoded token now (about 60-65 ms): trips 13 ms (24 x 0.52 ms), local linears 10 ms, mask prepare/take/finish
  about 8-10 ms (Python around the native kernels: reservation checks, noise monitor, staging copies), attention,
  norms, RoPE and Python about 20 ms, LM head 4 ms; time to first token 0.5-1.1 s. Not reached: 30 tokens/s.

## Update 2026-11-22: co-located chat throughput on Modal (package T1)

Goal was 30-40 tokens/s for one Qwen2.5-0.5B chat on the co-located Modal placement; reached **10-11 tokens/s
warm chat (12-13.5 tokens/s back to back), from 4.0-4.7**, every run logprobs bit-identical and repeat
identical. Details and the per-token breakdown: `deploy/modal/README.md`, "Co-located throughput".

- Co-located defaults: `attention_down` + TLS lane; the encrypted factory skips the local kinds
  (`PGW_FACTORY_SKIP_LINEAR_KINDS`, gate_up masks only): 138 slots/s on 2.5 cores while chats decode, so
  the factory no longer competes with decode (it was the 4.07 tok/s run's main loss) and keeps up 10x over.
- Local linears of any row count take the native kernel (prompt passes no longer use the two-digit torch
  int8 GEMM, 10-140 ms per call on Modal's AVX2 hosts); decode GEMVs split input features over the pool.
- Pool knobs `PGW_POOL_SPIN_US`, `PGW_POOL_WORKERS`; launcher `--gateway-env`, `--provider-env`,
  `--sustain-s`, loopback warm-chat figure; `gpu_bench.py` CPU targets `c8`/`c16`.
- What is left per decoded token (about 57-62 ms): round trips 12 ms, mask prepare/finish 10.6 ms, local
  GEMVs 17-25 ms (148 MB of int8 weights from DRAM; pool threads scale poorly under gVisor), LM head 3.5-4 ms,
  float ops and Python 12-15 ms; time to first token 520-640 ms. `float_threads = 1` stays off on Modal's
  AVX-512 hosts: torch's CPU SDPA gives different bits at 1 and 14-16 threads there (the self-check is right
  to refuse; the plaintext reference itself depends on the thread count on those hosts). Tried and dropped
  (no gain on Modal): cache warm-ahead of local weights, a 200 us GIL switch interval.
## Update 2026-11-21: encrypted mask factory, gateway side (package M3)

Design and numbers: [research/plans/he-gateway-decrypt.md](research/plans/he-gateway-decrypt.md).

- **Fused NTT decryption** (`PGW_FACTORY_HE_DECRYPT`, default `auto`): the hint product `H S` runs as NTT
  ring products (native `pgw_ntt`, exact by the CRT bound, checked per call) and decryption (subtract, round,
  transpose into the output) is fused into the same native pass, so the correction never leaves native scratch
  memory. `auto` uses it per pass size only when a start-up calibration on public random data found it
  bit-identical to the dense path and faster; any native error falls back to dense. Optional hints in NTT form
  (`PGW_FACTORY_HE_HINT_NTT_GB`, default 0; twice the hint's size in RAM). Measured locally (4 vCPUs with AMX,
  0.5B shapes, decrypt only): 7.6 -> 1.9 ms/slot at 16 groups per pass, 6.1 -> 5.1 at 2 groups (1.5-2.0 with
  hints in NTT form). Kaggle 4-vCPU Xeon without fast int8 (measured): 16.5 -> 8.2 ms/slot at 0.5B and 75 -> 38 at
  7B (16 groups), 5.9 and 27 with hints in NTT form; GPU-provider end-to-end numbers not measured (GPU session cap).
- **Masks of local linears** (`PGW_FACTORY_SKIP_LINEAR_KINDS=qkv,o`, opt-in): for gateways with
  `local_linears = attention` (and no `local_max_rows`) the factory no longer makes qkv/o masks (16% of mask
  bytes at 0.5B). The gateway declares at start-up which linears it sends (`MaskStore.require_linears`);
  segments lacking one are retired (written off, never reserved), so a later configuration change cannot run
  short or reuse a mask. Reservations may now span segments with different linear sets.
## Update 2026-11-21: mask data path at high bandwidth (package X2)

Memo: `research/plans/mask-ram-path-x2.md`. At 1k tokens/s a 7B gateway needs 5.6 GB/s of `W r`, a
70B deployment 27 GB/s, so masks go factory -> locked RAM ring -> gateway, with the encrypted disk
store as the outage reserve. Measured on the 4-vCPU dev box (A/B against the base commit, per full
slot): the factory writes into the ring at 0.85 ms per 7B slot (store: 9.6 ms). The gateway take was
the limit, and is now faster: **7B 172 -> 276 slots/s per core, 0.5B 350 -> 586** (store tier: 89 and
150, unchanged). Ring segments are now read in place under the slab lock
(`RamSegment.read_wr_rows`) instead of each session copying a 1 MiB chunk. The rows are unchanged,
and so are single use, wipe-on-consume and the recycled-slab checks. The remaining floor is the pad
keystream (1.8 ms per 7B slot, 8 keystream bytes per element by SPEC 7.2), not the `W r` bytes
(0.37 ms memcpy). `W r` is already K-bit packed at rest and in RAM, so packing cannot remove anything
more exactly. A pad-format v2 (4 keystream bytes per element at K <= 32) would halve the floor; it
needs a coordinated format change. Also new: an opt-in batched native claim over ring reservations
(`PGW_MASK_RING_BATCHED_CLAIM=1`; off because the native ChaCha20 is slower here), the sustained
producer/consumer bench `benchmarks/mask_ram_path_bench.py`, and an end-to-end test of the encrypted
factory -> ring -> gateway path (bit-identical to the local factory).
## Update 2026-11-21: aggregate throughput with many chats (X4)

Write-up: [research/plans/aggregate-throughput.md](research/plans/aggregate-throughput.md). Cross-session batching
already runs (one trip per linear for all live chats; the service concatenates rows), so aggregate tokens/s flattens
at one gateway CPU core: the plaintext thread does masking, float ops and the LM head for every row in series. Two
plaintext threads in one process are slower (GIL). Changes: `float_threads` defaults to 1 on CPU engines
(self-checked; engine step at 0.5B shapes 214 -> 136 ms with 32 chats, measured on 4 vCPUs); several gateway
processes per host over one mask store scale aggregate throughput (1/2/4 processes: 317/538/923 tokens/s on the dev
box, null service, measured). `gpu_validate.py` gained `--variant`, `--shard-gateways` and per-level CPU cores; GPU
numbers pending (command in the write-up, section 7).
## Update 2026-11-21: TCP+TLS trip lane (package L1, opt-in)

The service's native trip lane now also listens on TCP with TLS 1.3 (`PGW_SERVICE_LANE=tcp`,
`PGW_SERVICE_LANE_PORT`; rustls, the service's certificate, client certificates verified against its client
CA), and the gateway has `lane = "tls"`: decode trips use the native `pgw_fast.TripChannel` (rustls, a
48-byte binary header, one `writev`, the GIL released for the whole trip, spin-then-block wait) instead of
the WebSocket stream lane. Encryption and authentication are unchanged or stricter: the gateway verifies
the server against `service_ca_file` as the `service_url` host and connects only to that host; the
single-use ticket comes from `lane/open` on the regular channel (bearer token, client certificate) and is
bound to the client certificate that asked for it. Same masked bytes, request ids and call schedule; any
call the lane cannot take goes byte-identically to the blocking stream lane. Gateway-level tests show
greedy and seeded-sampled answers bit-identical (`tests/gateway/test_speed_wiring.py`); lane tests in
`tests/service/test_lane_tls.py`. On the service, lane calls without a CUDA graph now run inline on the
event loop when the device is idle (one thread hop instead of a task). Modal: `launch.py --placement
colocated --lane tls` (default off). Measured here per 1-row loopback TLS call (0.5B shapes, stub kernels,
load 30-40): stream lane p50/mean 346/795 us, TCP+TLS lane 219/395 us, transport alone 23/74 us; with two
busy Python threads in the gateway 2157/3100 vs 280/975 us. Details: `deploy/modal/README.md`.
## Update 2026-11-21: gateway CPU per decode token (package L2)

Single chat, decode, 1 row, on the gateway's production decode path (TripEngine, synchronous trips, noise
monitor, `local_linears` qkv + o). All changes are bit-identical (self-checked or tested against the
reference) and nothing sent to the provider changes (bytes, sizes, counts, schedule).

- **Native exact local linear** (`pgw_fast.local_linear_into`, `rust/pgw-fast/src/local.rs`): quantize with
  the TripEngine's constant-time quantizer (zero mask), int16 x int8 GEMV (AVX2 `vpmaddwd`, int32 partial
  sums flushed to int64 every `2^(24-A)` features), range check over every output, the finish kernel's
  rescale. Replaces numpy temporaries + a two-digit `torch._int_mm` (whose OpenMP threads also spun against
  the co-located provider). Works for any K multiple of 8 with A <= 15 (K40A12 now takes the fast path too).
  `PGW_LOCAL_NATIVE=0` restores the old form; the per-shape self-check is unchanged.
- **Native noise fingerprints** (`pgw_fast.noise_dup_keys`): `guard_packed`'s duplicate-row keys in one
  call (`PGW_NOISE_NATIVE=0`: numpy). Same keys bit for bit, same history and alarms.
- **Decode mask prefetch overlap**: with synchronous trips (lane `stream`/`uds`) a single session's 1-row
  decode stages its next 8 rows per (reservation, linear) on the prefetch thread, i.e. while the gateway
  thread waits in `recv`. Same rows, same order, same consumption (`PGW_PREFETCH_DECODE` overrides).
- Small: no logits copy in the sampler without penalties; request ids from `os.urandom(16).hex()` (same
  format); the deferred-check decision memoized per linear.
- Small: no logits copy in the sampler without penalties; the deferred-check decision memoized per
  linear. (Request ids stay `uuid4().hex` on every path: a review dropped an `os.urandom` id for the
  native path, whose missing version nibble would have told real trips from filler-only trips.)

Measured here (4 shared vCPUs at load 20-40 from other jobs, so medians of calling-thread CPU; GPU box
numbers pending, see the L2 report):

| item (1 row) | before | after |
|---|---|---|
| local linear 0.5B qkv 896x1152 / o 896x896 (`local_linear_bench.py`, 1 thread) | 137-175 / 114-255 us | 35 / 29 us |
| local linear 7B qkv 3584x4608 / down 18944x3584 (1 thread) | 2.6-4.6 / 25 ms CPU (5.4-8.2 / 45 ms wall) | 1.2 / 10.9 ms |
| noise `guard_packed` per trip, n = 896 ... 18944 | 17-21 us | 2.6-4.2 us |
| one 0.5B decode trip, gate_up / down (`trip_cpu_bench.py`, gateway thread CPU) | 175-231 / 281-350 us | 137-168 / 205-244 us |
| local linears in the decode loop (`decode_cpu_bench.py`, 0.5B, 4 layers, wall per token) | 12.0 ms | 2.6 ms |
## Update 2026-11-22: strict speculation on a CPU-only gateway (off by default)

Details: [research/plans/strict-speculation-cpu-gateway.md](research/plans/strict-speculation-cpu-gateway.md).
Every linear remote + `float_ops = native` + padded strict speculation with a draft on the gateway CPU.

- Exact: verify rows of R = 2/4/8 passes (with padding rows) equal single-row decoding bit for bit (masked,
  all remote, native ops; also at real 0.5B shapes with both float paths); spec == plain bit for bit for five
  draft sources incl. the pipelined drafter; R rows per pass and L passes on the wire. The GPU refusal is
  about a CUDA target and does not touch this configuration (`float_ops = native` is CPU-only).
- `draft_backend = "native_int8"` (`NativeInt8DraftBackend`, `pgw_fast.local_linear_into`): Qwen2.5-0.5B
  draft 75 ms/token on 1 core, 46-54 on 4 shared vCPUs (torch int8: 390 / 112); memory-bound, ~10-20 ms
  expected on 16 cores.
- `spec_pipeline_steps = m` (`PipelinedDrafter`): the next block is drafted in the background during the
  pass; exactly m draft steps between passes hit or miss (no acceptance timing signal; tested).
- 7B model (`benchmarks/spec_7b_cpu_gateway_model.py`): 25 tokens/s target met with margin by R = 2, C = 2,
  m = 0 at 0.3 ms trips on a 16-core gateway (29-32 tok/s, <= 2 mask slots per token); 0.4 ms needs R = 3;
  0.5 ms only at high acceptance. Plain CPU-only decoding tops out near 22. No 7B acceptance measured.
## Update 2026-11-22: strict speculation at a public mask charge (L3, off by default)

`spec_mask_rows` C (+ `spec_mask_credit` B), research and numbers in
[research/plans/speculation-no-compromise.md](research/plans/speculation-no-compromise.md): strict speculation's
filler rows and dummy passes are padding rows (no mask) and every decode pass charges the public schedule
`prompt + j + min(B, floor((C-1) j))`, so masks per generated token drop from R to about C (or 1 + B/L) while the
provider's view (rows, items, pass count, mask demand, per-call gateway work) stays independent of the drafts.
Exact: bit-identical to the plaintext reference in the new tests (greedy, seeded sampling, penalties, stop strings,
R 2/4, C 2/R, fractional C, credit caps, mask extents ending mid-pass).

- Four distinguishers of the existing padding-row code were found and closed before enabling it: item splits at
  mask-extent boundaries (now one item per padded call, and real rows kept within one extent), unverified dummy
  items (an active provider could learn which passes were dummies; now Freivalds-checked), a gateway turnaround
  that depended on the real/padding split (measured 466 / 529 / 654 us per call for dummy / real / mixed at R = 4;
  now one fixed-structure path in which a dummy call also takes one charged mask row, so every call makes the same store take; a dummy call is now within 0.5% of a 1-real-row call; the per-row take-versus-padding residual (4 real vs 1 real + 3 padding rows) is about 3% per call (~12 us on the 0.5B down linear, so ~0.5 ms per pass) with an in-memory reservation, reveals how many drafts a pass carries, and must be re-measured with the real mask store before production use), and window extension, ahead reservation and part retirement driven by
  mask use (`WindowedReservation(public=True)`: extensions only by `charge_to` on the schedule).
- Simulated on the Kaggle run 28 exact answers (`benchmarks/spec_charge_sim.py`): 1.5B target with the 0.5B draft at
  R = 4, C = 4, B = 160 keeps today's 2.84 tokens per pass at 2.50 masks per token instead of 3.96; the 0.5B with
  prompt lookup at C = 1.5 keeps its 1.08 tokens per pass at 1.48 instead of 3.96. Not measured on a GPU or Modal
  (commands in the plan, section 7; Modal: `launch.py --spec-mask C[/B]`).
- Pre-existing strict-mode timing residuals are documented, not fixed (attention cost grows with context during the
  answer and is flat during the dummy passes; the drafter's catch-up step).

## Update 2026-11-20: chat-speed phase A wired into the gateway

The phase-A packages now run on the gateway's production path (all per-model settings, off by default;
SPEC 9.4a). `local_linears`, `lane`, `spec_rows` + `spec_strict_passes` left `PENDING_SPEED_SETTINGS`;
`float_ops`, `head_mode`, `schedule` and `prefix_cache` stay refused (phase B). New per-model settings:
`local_rtt_ms`, `lane_uds_dir`, `lane_pool_wait_s`, `float_threads`. `/admin/status` reports them under
`speed` (lane state and fallbacks, local-linear decision, start-up RTT). Gateway-level tests show each
setting bit-identical to the plaintext reference on the tiny model in masked mode, including the lane
against the real service UDS lane (`tests/gateway/test_speed_wiring.py`).

- The lane client (WP-A3) and the service lane (WP-A1) were built against different framings
  (`/pgw/v1/lane_open` + `PGWT` vs `/pgw/v1/lane/open` + `PGWL`). `pgw.wire.service_lane` bridges them
  (Python channel; the native `TripChannel` speaks only the A3 framing, so the lane runs at the Python
  channel's cost until one side is changed).
- Strict speculation pads with masked filler rows (the model does not use the backend's mask-free padding
  rows yet), so it costs `spec_rows` times the mask rows per token.
- Modal demo: the customer's `models.json` gets `local_linears = "auto"` and `float_threads = 1` via
  `PGW_GATEWAY_SETUP_MODEL_OPTIONS`; no lane (remote provider) and no strict speculation (no draft model,
  little prompt overlap). Not run on Modal or Kaggle yet.
- `pgw doctor --maskstore-read-mb` sets the `maskstore.read_rate` probe size.

## Update 2026-11-20 (measurement and telemetry for run 26)

Run 26 (after the phase-1 merge) will be measured with corrected accounting; no new speed numbers yet.

- **Decode-only accounting** in `benchmarks/gpu_validate.py`. The run-25 "round trips 0.114 s/token"
  divided whole-request remote time (prefill included) by prompt + generated tokens, so it understated
  decode. The streaming phase now reports `decode_remote_s_per_token`, `decode_local_s_per_token`,
  `decode_remote_ms_per_trip` and `decode_round_trips_per_token` (from the backend's decode counters when
  the gateway has them, else from the stats delta after the first token), and how much of the measured
  s/token they account for. The old fields are kept as `*_incl_prefill`.
- **Per-trip split**: `/admin/status` now has `client_timing` (per rows-per-request class: client RTT,
  service queue and compute, and the rest = network, framing, Python), reported as `decode_trip_split`.
  The compute client wrapper now also times (and breaker-gates) the blocking `compute_sync` path that
  small decode calls use; before, those trips were untimed.
- **Metrics**: `pgw_gateway_round_trip_seconds` uses sub-ms buckets (0.25-25 ms plus a tail).
  `pgw_gateway_network_seconds`, `pgw_gateway_service_seconds` and `pgw_gateway_client_cert_expiry_seconds`
  are implemented but emit only once they are added to the metric contract
  (`pgw.observability.metrics.METRIC_NAMES`) and the gateway API metrics table.
- **Benches**: `benchmarks/net_floor_bench.py` (TCP ping-pong floor; loopback in this 4-vCPU container:
  p50 39 us at 64 B, 75 us at 80 KB, i.e. under 0.01 s/token over 112 trips). `profile_gateway.py`
  warms every layer. Kaggle jobs `pgw-run26-7b` (now `pgw-run27-7b`, 4/8/16 streams, net floor, profile) and
  `pgw-spec-sweep-7b` (k = 1, 2, 3 on cpu and cuda:1, `--spec-max-batch 1`) replace `pgw-spec-7b`
  (`--spec-tokens 8`).

## Update 2026-11-20 (two-party deployment on Modal)

`deploy/modal/` runs a real provider/customer pair as two Modal apps (one command: `launch.py`). The
customer is onboarded through the setup wizard over a public URL: enrollment with CA pin and an
issued client certificate, the model downloaded from Hugging Face and packed by the gateway itself
(fingerprint matches the provider's), encrypted mask generation over the network, finish without a
restart. A masked greedy chat was bit-identical (tokens and logprobs) to the plaintext reference.
Found on the way: a new store waited for a full 4096-token mask segment before the gateway could
serve (over an hour at the 1.5 MB/s the tunnel carries). The factory now makes a small first segment
for an empty store (`PGW_FACTORY_FIRST_SEGMENT_SLOTS`; the wizard sets 256), so the first masks arrive
after about 2 minutes. The service authenticates compute calls with the tenant token; client
certificates are verified and tenant-checked when presented but cannot be required on the port that
also serves enrollment.

## Update 2026-11-20 (deterministic multi-row LM head)

The float32 LM head (`norm(h) @ lm_head^T`) is now `pgw.engine.lm_head.lm_head_logits`: a fixed
summation order (16 partial sums over the hidden dim, fixed tree, no FMA), so a row's logits do not
depend on how many rows share the pass, the thread count or the CPU. Native kernel `pgw_fast.lm_head`
(SSE2/AVX, multi-threaded, streams the head once for all rows); pure-torch fallback with the same
order (bit-identical, tested; slower: about 0.45 s per row at 7B). Both the gateway's masked model and
the plaintext reference use it, so they still agree bit for bit; **the reference logits themselves
moved by ~1e-6 vs torch's old GEMV** (different but fixed order). Because batching is now exact,
`batch_lm_head` defaults on with the native kernel: one pass per step for every sequence's last row and
all rows of speculative verify passes (`RowLogits`); off (lazy per row) with the fallback.
Measured here (4 vCPUs, `benchmarks/lm_head_bench.py`, 152064 x 3584 fp32):

| rows per step | torch per row (old) | torch one GEMM (inexact) | deterministic, one pass |
|---|---|---|---|
| 1 | 0.050 s | 0.051 s | 0.043-0.054 s |
| 4 | 0.18-0.19 s | 0.09-0.11 s | 0.07-0.08 s |
| 8 | 0.39-0.42 s | 0.16-0.18 s | 0.10-0.11 s |
| 16 | 0.75 s | 0.11-0.12 s | 0.17 s |

## Update 2026-11-20 (gateway CPU per call, multi-session throughput)

Measured here (4 shared vCPUs, A/B interleaved, medians of 3; `benchmarks/profile_gateway.py --preset 7b
--ring K32A12C27`, `benchmarks/e2e_local.py`), outputs bit-identical to the plaintext reference:

| metric | before | after |
|---|---|---|
| gateway ms per decode call, 7B, 1 session (in-process CPU service) | 0.95 | 0.61 |
| same, 4 sessions per round trip | 2.71 | 2.01 |
| same, `--null-service` (no CPU matmul evicting caches; closer to a GPU service), 1 / 4 sessions | 0.57 / 1.86 | 0.30 / 1.01 |
| e2e small model, null service, 1 / 4 streams (tokens/s) | 56 / 93 | 94 / 156 |

What changed: `pgw_fast.finish_group` verifies and unmasks the chunks of all sessions whose masks come
from the same segment in one pass over the shared Freivalds vectors (each keeps its own `W r`); check
vectors and `W r` travel as 32-bit words for K=32; `W r` is zeroed inside the kernel; the request payload
replaces the int64 `x_hat`; per-(extent, linear) caches for stream keys, compact check vectors and the
current extent; `MaskReservation.take_checked` (boundary + checks + take under one lock);
`WindowedReservation` no longer rescans old parts; small decode round trips run on the event loop
(`inline_max_rows`, default 8; never for window extensions, a reservation's first pass or the LM head).
LM head (`benchmarks/lm_head_bench.py`, 7B float32): 47 ms per sequence, so 0.19 s per step with 4
sequences; the opt-in `batch_lm_head` (default off, changes logprob bits, not quality) takes 0.11 s.

## Update 2026-11-20: mask factory speed (GPU ChaCha20, native writer, encrypted factory)

Merged, each reviewed (outputs byte-identical, privacy unchanged):
- **GPU ChaCha20 + digit split** (`PGW_FACTORY_GPU_PRNG`, on for CUDA factories after a byte-identity self-test
  against the CPU path; the factory falls back to the CPU otherwise). Runs only on the customer's own factory GPU.
- **Factory quick wins:** native ChaCha on the CPU path, byte-gather digit split, in-place combine, auto chunk size,
  pinned copy-back on CUDA, per-stage stats.
- **Native segment writer** (Rust AES-GCM, same at-rest format): write stage about 30 -> 5 ms per 7B slot (local).
- **Encrypted factory, customer side:** native noise/secret sampling, more groups per pass, requests in flight.
- **Encrypted factory, provider side:** faster maskgen (about 2.1-2.6x on CPU locally), cached row sums, no global lock.

Kaggle T4, Qwen2.5-7B K32A12C27, all 28 layers, generation only (no store writes), 1024 slots:
**122 -> 356 slots/s (about 438k -> 1.28M slots/h) with the GPU PRNG**; rng time 8.7 s -> 0.07 s; GPU tests incl.
byte-identity pass on the T4 (`benchmarks/results/kaggle/factory_gpu_prng_v2`). The first attempt
(`factory_gpu_prng`) correctly fell back to the CPU path because of a cuda vs cuda:0 device comparison, since fixed.

## Update 2026-11-20: phase 2 of the speed plan (Kaggle run 27)

Four packages, each reviewed (no blocking issues): lean wire lane (service frame lane, client
`compute_fast`, zero-copy replies), native `prepare_many` plus single-use mask prefetch on a background
thread, a faster exact LM head (register-blocked; AVX-512 where available), sampler and RoPE, and
optional two-cohort ping-pong (`engine_pingpong`, off by default). Run 27, Qwen2.5-7B on one T4, same steps
as run 26, all outputs bit-identical (text and logprobs):

| metric | run 26 | run 27 |
|---|---|---|
| decode, 1 request | 0.339 s/token | 0.349 s/token (unchanged within noise) |
| time to first token, 40-token prompt | 2.37 s | 2.35 s |
| generated tokens/s, 4 / 8 / 16 streams | 4.16 / 4.70 / 4.61 | **4.80 / 5.72 / 5.74** (+15 / +22 / +25%) |
| gateway CPU per generated token, 4 / 8 / 16 streams | 0.184 / 0.177 / 0.179 s | **0.145 / 0.131 / 0.132 s** |
| decode round trip (remote), 1 request | 1.47 ms | 1.46 ms |

The single-request path is bound by its 112 sequential round trips (0.163 s/token) and did not move;
the gains are in gateway CPU, which is what limits aggregate throughput. Gateway plaintext work per token
on the Kaggle CPU (`benchmarks/profile_token.py`, 28 layers, stub service; this host has AVX2, not
AVX-512): 97.4 -> 82.5 ms at 1 row, 249 -> 227 ms at 4 rows, 967 -> 901 ms at 16 rows; sampling 1.6 ->
0.65 ms. Spinning OpenMP workers (`OMP_WAIT_POLICY=ACTIVE`) cut the layer work 29 -> 20 ms at 1 row but
slowed the LM head; a candidate for scoping to the layer ops only.

Ping-pong A/B (`pgw-pingpong-7b`, 32 tokens, same service and store): 4 streams 4.25 vs 4.29 tok/s
(control, not split), 8 streams 4.98 vs 5.10 (+2%), 16 streams 4.89 vs 5.29 (+8%), bit-identical. It stays
off by default; it is worth enabling for gateways that regularly run 16 or more concurrent sessions.

Raw outputs: `benchmarks/results/kaggle/run27_phase2_7b`, `pingpong_phase2_7b`, `run27_token_profile_cpu`.
(A first run 27 attempt is not recorded: the Rust toolchain check passed without cargo, so it measured
the pure-Python fallback. The check is fixed and a failed native build now stops the job.)

## Update 2026-11-20: phase 1 of the speed and readiness plan (Kaggle run 26)

Plan: `research/plans/speed-readiness-plan-2026-11-20.md` (55-agent sweep, adversarially verified).
Seven packages (wnorm in the manifest, noise-monitor lean path, GQA attention with a load-time
bit-equality check, speculation privacy fix with fixed row counts, service per-session batching,
telemetry, security and ops hardening) plus reviewer fixes. Run 26, Qwen2.5-7B on one T4, all outputs
bit-identical:

| metric | run 25 | run 26 |
|---|---|---|
| decode, 1 request | 0.365 s/token | **0.339 s/token** |
| time to first token, 40-token prompt | 2.89 s | **2.37 s** |
| generated tokens/s, 4 / 8 / 16 streams | 3.0 / - / - | **4.16 / 4.70 / 4.61** |
| decode round trip (remote) | | 1.47 ms (112 per token) |
| decode gateway work per trip | | 0.72 ms |

Decode time per token: 0.165 s round trips, 0.080 s gateway masking, 0.094 s other (LM head, float ops,
Python). Loopback socket floor: 33 us (64 B) to 81 us (80 KB) per exchange.

## Update 2026-11-20: bfloat16 LM head (exact)

Open checkpoints ship in bfloat16, so the float32 LM head's values are all exactly bfloat16. The
gateway now keeps the head in bfloat16 when that is lossless (checked at load) and the deterministic
kernel widens each weight to float32 on load (exact), so logits are bit-identical to the float32 head
while half the bytes are read per token (1.1 GB instead of 2.2 GB for Qwen2.5-7B, and 1.1 GB less
gateway memory). Local measurement, 7B shapes with 4 layers and a stub service (`benchmarks/e2e_local.py
--preset 7b4 --null-service`): 56 ms -> 42 ms per decode step. Kaggle run 23 showed no change (0.436
s/token) because its packs used `--plain-dtype float16`, which rounds the bfloat16 weights so the
head cannot be compacted; the Kaggle jobs now pack with `--plain-dtype bfloat16` (lossless). Run 24 with bfloat16 packs: 0.410 s/token
(from 0.436), TTFT 2.94 s, outputs bit-identical. Run 25 adds the fused Triton GEMV for 1-row calls and the
service wire-overhead cut: **0.365 s/token**, TTFT 2.89 s, 4 streams 3.0 tok/s, service 1-row call p50 1.39 ms
(0.70 ms GPU), round trips 0.114 s/token; outputs bit-identical. (The multi-row tensor-core Triton kernel does
not compile on T4/Triton 3.6; the startup self-check limits the fused kernel to 1 row, as designed.)

## Update 2026-11-19

**Speed (Qwen2.5-7B, one Kaggle T4, default ring K32A12C27, output bit-identical to the plaintext
reference in every run).** Raw outputs are in `benchmarks/results/kaggle/run13_*` to `run19_*` (run 19: all changes together).

| metric | 2026-11-17 | 2026-11-19 | what changed |
|---|---|---|---|
| decode, 1 request | 0.86 s/token | 0.339 s/token | persistent WebSocket channel (uvicorn), K=32 uint32 path, native (Rust) gateway kernels, numpy wipes, batcher fast path |
| time to first token, 40-token prompt | 8.4 s | 3.0-3.3 s | pipelined prefill (gateway and service overlap) |
| one service round trip, 1 row (load test) | 2.5 ms | 1.75 ms (0.80 ms GPU) | uvicorn WebSocket framing, inline small batches |
| decode with speculative decoding, 0.5B draft on a gateway GPU | n/a | 0.43 s/token | exact verification, coupled drafts, adaptive draft length; 2.6x fewer round trips |
| gateway CPU per masked call (Kaggle CPU) | about 2.9 ms | about 1.8 ms | `pgw_fast` kernels: prepare 0.05 ms, finish 0.2 ms; the rest is Python orchestration |

Where a decode token's time goes now: about 112 x 3.5 ms in round trips (of which about 0.9 ms is GPU
work; the rest is service and client Python and WebSocket framing), about 0.2 s gateway work, and
about 0.1 s for the float32 LM head. A draft model on a CPU-only gateway does not pay off yet.

**Concurrency on Kaggle: 2.8 generated tokens/s for 4 streams in run 22 (2.3 in run 21, before the multi-row LM head).** Run 21 breakdown: The 4 sessions
share round trips (7392 trips for 4 x 65 tokens), but each 4-session step takes about 1.7 s: about
0.7 s in round trips (gateway, service and mask factory share the VM's 4 vCPUs), about 0.5 s of gateway
crypto, and about 0.5 s of float32 LM heads computed one row at a time (2.2 GB of weights read per row).

**Fused Triton ring GEMV for one-row calls** (`pgw.core.triton_gemv`, on by default after a device
self-check, `PGW_SERVICE_TRITON_GEMV=false` turns it off). One kernel, int8 weight streamed once, wrapping
uint32 multiply-adds, bit-identical to the digit-GEMM path. Kaggle T4, 7B shapes, K32A12C27, rows=1
(`benchmarks/results/kaggle/triton_gemv/`): service call per layer 2.92 -> 1.92 ms (qkv 0.51 -> 0.33,
o 0.56 -> 0.28, gate_up 1.17 -> 0.81, down 0.68 -> 0.51); device time per layer 1.85 -> 1.26 ms (180-220
GB/s). Several rows stay on the digit-GEMM path on the T4: the CUDA-core kernel is compute-bound there
(4-8x slower) and the int8 `tl.dot` variant does not compile with Triton 3.6 on sm_75 (the self-check
detects it and limits the fused kernel to one row; GPUs where it compiles and passes get up to 16 rows).

**CUDA graphs are off by default** (`PGW_SERVICE_CUDA_GRAPHS=true` turns them on). On torch 2.11 with
a T4 most captures fail inside PyTorch's caching allocator and the service falls back to eager, and
where capture worked the gain was about 6% per layer (run 20). The graph tests skip when capture is
unsupported; exactness is still checked.

**New components.** `rust/pgw-fast` (native hot path, optional, bit-identical, falls back to Python);
PGW Console (`/console/`: operator web UI with sessions, admin/viewer roles, CSRF, strict CSP, usage,
audit, playground); prompt-lookup drafting; WebSocket compute channel with HTTP fallback.

**Fixed.** A shutdown spin in `Generator.wait_closed` that could hang model reload or gateway stop
under load. Speculative verification is now bit-identical (it differed by up to 0.054 in logprobs).

**Readiness.** Unchanged in substance: a technical pilot in topology B is reasonable; a production SLA
pilot still needs SSO for the console, the hardware key path, a GPU soak test, signed images and an
external security review.

## 1. What is built

About 26.8k lines of Python in `src/pgw` (96 modules, mypy strict, ruff clean), 89 test files, and a
2.5k-line Rust port of the core.

| component | state | notes |
|---|---|---|
| `pgw.core` | complete | Ring presets K40A12 (default), K48A20 and K32A8. Exact ring matmul on `cuda_int8` (T4-validated), `cpu_int8` (shared probe-gated int8 digit GEMM), `cpu_f64` and `cpu_f32`. Also masking with range check, Freivalds with soundness bound, the ChaCha20/HKDF mask PRG (known-answer tests), packing and fingerprints. |
| `pgw.modelpack` | complete | HF checkpoint (Qwen2 and Llama families) to an int8 pack with manifest and SHA-256 fingerprint. `verify=full` by default on the gateway. Unsupported RoPE types are refused. |
| `pgw.engine` | complete | Plaintext transformer with KV cache, batching, sampling, logprobs and streaming. Backends: `MaskedBackend` (production), `QuantPlain` (reference), and float (dev). |
| `pgw.factory` | complete | CPU and GPU mask generation, auto Freivalds check count (at least 20 bits), self-check, refill scheduler, partner export (HPKE plus Ed25519 signature plus destination binding), and health/metrics. |
| `pgw.maskstore` | complete (core); providers partial | Encrypted single-use segments, a durable consumption cursor, windowed reservations, crypto-shredding, rollback witnesses (HMAC journal plus external counter), refusal of network filesystems, and KEK rotation. See section 5 for the providers. |
| `pgw.wire` | complete | HTTP/2 plus msgpack, mTLS, retries with a deadline header and retry budget, and byte-budget admission. |
| `pgw.service` | complete | Stateless compute: registry keyed by fingerprint, GPU and CPU kernels, cross-request batching, token or mTLS-identity tenants, quotas, drain, and an internal metrics/probe listener. |
| `pgw.gateway` | complete | OpenAI-compatible `/v1/chat/completions`, `/v1/completions` and `/v1/models`, with SSE, API keys, rate limits and quotas, admin API, audit log, integrity circuit breaker, outbound-noise monitor (fail closed), dashboard, and graceful shutdown. |
| `pgw.observability` | complete | A Prometheus metric contract (`METRIC_NAMES`), the noise monitor and the dashboard. |
| `pgw.cli` | complete | `pgw pack/inspect/factory/maskstore/service/gateway/doctor/bench/config` (SPEC 12). |
| `rust/pgw-core-rs` | complete (reference) | Ring, PRG and Freivalds port, cross-checked against `tests/vectors/core_v1.json`. Optional pyo3 extension. |
| `deploy/` | complete, partly unverified | Five Dockerfiles (digest-pinned, hash-locked requirements), docker-compose demo, Helm charts for gateway, service and factory, systemd units, a VMware OVF/Packer appliance, and AWS/Azure marketplace READMEs. |
| `docs/` | complete | Security whitepaper, threat model, gateway API reference, operations guides (on-prem, customer cloud, air-gapped factory, sizing), compliance (audit readiness, FAQ), and a sizing tool. |
| CI | written | `.github/workflows/ci.yml` (lint, types, tests, audit, images) and `ci/check.sh`. The workflow sits under `pgw/` and has not been copied to the repository root yet. |

### 32-bit ring `K32A12C27` (new, 2026-11-18)

This is a capped ring with per-token adaptive activation bits; it decodes any layer width exactly.
Its quality is identical to `K40A12` on Qwen2.5-0.5B and 1.5B (perplexity 11.714 vs 11.714 on 1.5B,
with 0 tokens needing fewer bits). It cuts 20% from traffic, service GPU work and mask storage. With
encrypted masks it uses q = 2^56: 7B mask-supply traffic drops from 6.7/8.3 MB to 5.8/7.0 MB per
token and the hint from 39.9 to 34.2 GB (T4 run 11). The cost is about 1 ms more gateway CPU per
layer, for 7 Freivalds vectors instead of 3. Not done yet: a 7B quality check (blocked on memory
here) and making it the default.

### Encrypted mask generation (new, 2026-11-18)

`pgw.hemask` lets the mask factory get `W8 r` from the compute service on MLWE-encrypted masks
(`pgw factory run --he-service ...`, demo `--mask-source he`). Storage no longer grows with
token volume. Instead there is a one-time hint of about 9.3 GB for 0.5B and about 45 GB for 7B,
plus a just-in-time buffer, and there is no factory GPU. The privacy assumption is MLWE, with a
single provider.

- Verified: factory output is bit-identical to local generation, a tampered provider is
  rejected, and an end-to-end chat on 0.5B with encrypted masks gives the same answers with 0
  integrity failures.
- The refill daemon refilled the store through the service while the gateway was serving.
- T4: the CUDA `mulmod64` kernel is exact. 7B costs 0.020 s of provider GPU per token.
  Gateway CPU per token is 0.11 s with a gateway GPU and 0.116 s on an AMX CPU, after the
  int8 byte-plane optimisation (it was 1.18 s). The hint is 39.9 GB.
- Not done yet: a lattice-estimator confirmation of the parameters, int8 AMX for gateway
  encrypt and decrypt, and an external crypto review. See
  `docs/operations/encrypted-masks.md` and `research/mask-storage/`.

## 2. Test results (this release run, 4-CPU container, CPU-only torch)

| suite | result |
|---|---|
| `pytest tests` (all, including e2e on real Qwen2.5-0.5B-Instruct) | **1491 passed, 0 failed, 40 skipped** in 7 min 11 s, on the final run after the release fixes. The first run of this release gave 1490 passed. |
| skipped | 35 need CUDA, 3 need the `helm` binary, 2 need the optional Rust extension or `PGW_TEST_RUST=1` |
| `cargo test` (rust/) | **42 passed**, 0 failed |
| `ruff check src tests` | clean (after this release's fixes) |
| `python -m mypy src/pgw` (strict) | clean, 96 files (after this release's fixes) |
| `scripts/e2e_demo.sh` (Qwen2.5-0.5B, CPU) | **pass**: "Paris" in both non-streaming and streaming mode, 384 round trips, 384 Freivalds-checked, 0 integrity failures, 76 MB up / 128 MB down, noise monitor active |

### 2.1 Fixes made in this release pass
- **Demo was broken.** After the security fix that requires an external monotonic counter,
  `scripts/e2e_demo.sh` returned HTTP 500 (`config`) on its first chat request. The demo now
  configures a file counter outside the mask directory.
- **Gateway fails fast.** A gateway whose mask store has no external counter now refuses to start
  with a clear `ConfigError`, instead of starting "ready" and answering every request with 500.
  Regression test: `tests/gateway/test_config_safety.py::test_gateway_refuses_store_without_external_counter_at_startup`.
- **mypy.** Fixed a strict error in `MaskStore.metrics()`: the label tuple type now covers the
  store-wide series that have no labels.
- **ruff.** Fixed the 15 remaining findings: import sorting, line length, and a redundant `int()`.
- **Docs.** Freivalds statements in the whitepaper, threat model T-03 and docs/README now match the
  auto-checks default (c = 3 at width 4096 gives 2^-24, with a floor of 2^-20) instead of the old
  `c = 2` / 2^-16. The docs consistency test is updated to match.
- **PKCS#11 docstring.** It claimed the provider is tested against SoftHSM2 "in the release
  pipeline". No such pipeline exists, so the docstring now says so.
- **Packaging.** Added the optional extras `hsm`, `kms-aws`, `kms-gcp` and `kms-azure` for the
  key-provider SDKs, which `pyproject.toml` did not declare before.

## 3. GPU validation (Kaggle, Tesla T4, K40A12)

Full details are in `benchmarks/RESULTS.md`. Raw outputs are in `benchmarks/results/kaggle/`.

- **Exactness.** Qwen2.5-0.5B and Qwen2.5-7B produce tokens, logprobs and top-2 logprobs that are
  bit-identical to the QuantPlain reference. This holds for non-streaming, streaming and 4 or 8
  concurrent requests, with 0 integrity failures and 0 skipped checks. The 7B pack fits one 16 GB T4
  (12.9 GB used).
- **GPU tests.** `pytest -m gpu` gives 35 passed. The e2e suite with the service on CUDA gives 32 passed.
- **Three GPU bugs found and fixed.**
  1. cuBLASLt rejects a row-major mat2.
  2. `/readyz` probed a 1x8 product, which the T4 rejects.
  3. Dimensions must be padded to multiples of 32.

| metric (7B) | measured | SPEC 14 target | status |
|---|---|---|---|
| service GPU compute, batch 1 | 0.16 s/token (was 0.45) | ≤ 0.5 s | met |
| service GPU compute, 32 rows | 0.018 s/token (0.012 at 128 rows) | ≤ 0.015 at batch 32 | roughly met |
| end-to-end decode, 1 request | 0.86 s/token (was 1.48) | (implied ≤ 0.5) | **missed**: limited by 112 sequential round trips and per-call gateway CPU |
| gateway CPU per token | 0.38 CPU-s (was 1.32; 0.27 at 8 concurrent) | ≤ 10 ms crypto + ≤ 20 ms plaintext | **missed by about 40x** |
| factory, T4 | 169k tokens/h (7B, K40) | ≥ 500k slots/h (8B, K48) | **missed by about 3x** (GEMMs are only 36% of the time) |
| mask storage | 6.95 MB/token | 6.9 MB | met |
| wire | 10.96 MB/token | 11.2 MB | met |
| test suite, 4 CPUs | about 6 to 7.5 min | < 5 min | slightly missed (the e2e on real Qwen is about 4 min of it) |

Only K40A12 was measured end to end on GPU. K32A8 and K48A20 are covered by ring benchmarks and unit
tests. Multi-GPU placement passed its unit test but was not benchmarked. A full 7B re-run with the
final padding fix (bug 3) was not done. No real model shape is affected by that fix.

## 4. Known limitations

**Performance**
- One token is 4 x layers sequential round trips: 112 for 7B and 128 for 8B. On loopback this costs
  about 0.5 s/token. Over a 10–30 ms WAN link (topology A) it costs 1.3–4 s/token unless sessions
  are batched.
- The Python request path is the bottleneck. Masking and unmasking, msgpack, Freivalds and mask
  decryption cost the gateway about 1.3 CPU-s per 7B token. The service through HTTP tops out at
  about 13 token-rows/s on 4 vCPUs while the GPU is about 30% busy.
- The factory spends most of its time in the CSPRNG, digit arithmetic and serialization, not in GEMMs.

**Functional (rejected with HTTP 400)**
- Tools and function calling, structured output (`response_format`), `n > 1`, and multimodal inputs.

**Security**
- Privacy is computational at the PRG level (HKDF-SHA-256 plus ChaCha20 seeds), not information-theoretic.
- Metadata is visible to the service: model, tokens per round, timing and tenant. Row padding
  (`pad_rows_to`) and constant-rate dummy rounds are specified (SPEC 3.4) but not implemented.
- Freivalds covers only the service's product. A wrong `W r` from a partner factory is caught only by
  the factory's own self-check and by the gateway range check for gross errors. An independent
  spot-check (threat model T-29) is planned.
- After an integrity failure the gateway does not rotate to a segment with a fresh `u`. The breaker
  stays open until an admin resets it, which is the default.
- The soundness floor is enforced when a segment is first used, not by a scan of the store at start-up.

**Operations**
- Every production deployment needs an external monotonic counter on storage separate from the mask
  volume. Without one the store and gateway refuse to serve.
- The VMware Packer template does not create the separate counter disk. First boot fails with an
  explanation until an operator attaches it at `/var/lib/pgw-counter`.
- Velero exclusion of the Helm counter PVC relies on a label. CSI VolumeSnapshot policies must
  exclude it separately.
- Segments written before the reliability fixes keep their old 256-slot chunks, which use more gateway
  memory until they are consumed.
- Stale-lease detection assumes a single gateway per store, which is the supported topology.
- The hypercorn mTLS identity shim hooks hypercorn internals, so hypercorn is pinned below 0.19.
- The Helm charts render the service token into the models file. They could use the newer
  `service_token_file` instead.

## 5. What is stubbed or unverified

| item | state |
|---|---|
| `Pkcs11KeyProvider` (HSM, AES-KWP, HMAC) | **Stub-quality.** The code is written, but only construction and error paths are unit-tested. wrap/unwrap/mac have never run against SoftHSM2 or a real HSM. |
| `KmsKeyProvider` (AWS KMS, GCP KMS, Azure Key Vault) | The provider logic is tested with an injected fake client. The real SDK adapters (`_AwsKms`, `_GcpKms`, `_AzureKv`) have never run against a cloud account. |
| `ProviderCounter` (HSM or KMS hardware counter) | No shipped key provider implements `monotonic_counter()` yet, so `counter=provider` cannot be used in practice. |
| `TpmNvCounter` (tpm2-tools) | **Stub-quality** and untested on a TPM. A vSphere vTPM is captured in VM snapshots, so it does not detect a snapshot revert. |
| Metadata padding and dummy rounds (SPEC 3.4) | Specified, not implemented. |
| AWS Marketplace metering hook | Deferred (SPEC 17). Only listing READMEs exist under `deploy/marketplace/`. |
| fbgemm or oneDNN CPU int8 path and a fused GPU kernel | Roadmap (SPEC 16). `cpu_int8` uses torch `_int_mm` digit GEMMs. |
| Docker image builds, `helm lint`, shellcheck, pip-audit, cargo audit | Written and wired into CI, but **not run here**: no Docker daemon, helm or network audit tools. The charts were checked with the repository's Python Helm renderer, and compose with `docker compose config`. |
| CI workflow at the repository root | Not copied. It lives at `pgw/.github/workflows/ci.yml`. CI test jobs install unlocked `.[dev]`; only the image and audit jobs use the hashed locks. |
| Terraform for topology B | Mentioned in SPEC 13, but not in `deploy/`. Only Helm exists. |

## 6. Recommended next steps before a customer pilot

**Must do (blocking)**
1. **Hardware and cloud key path.** Run `Pkcs11KeyProvider` against SoftHSM2 in CI, then against the
   pilot customer's HSM or KMS. Implement and test at least one hardware or cloud-backed
   `ProviderCounter`, or validate the file-counter-on-a-separate-volume design for the customer's
   storage and backup tooling.
2. **Performance work on the request path.** Profile the gateway and service, and move masking,
   unmasking, Freivalds, msgpack and decryption into vectorised or native code (the Rust core is a
   starting point). Fewer per-call copies. Targets: under 50 ms of gateway CPU per 8B token, and under
   0.3 s/token end to end in topology B.
3. **External security review** of the whitepaper, threat model and implementation, focused on mask
   single-use under crash and restore, partner delivery, and the noise monitor.
4. **Build and scan the real images.** Run `docker build` for all five Dockerfiles, `helm lint` and a
   kind install, `pip-audit` and `cargo audit` on the locks, and an SBOM. Then sign the images.
5. **Full 7B or 8B soak on GPU** with the final code: hours of mixed concurrent load, gateway
   restarts, a service rollout during traffic, mask-store exhaustion and refill, and a disk-full test.
   Run it on K40A12 and on K48A20.

**Should do**
6. Topology B packaging: Terraform modules (VPC, PrivateLink or PSC, KMS key, gateway node pool) and
   a customer runbook.
7. Factory throughput: move the CSPRNG and digit split to the GPU to reach 500k or more slots/h on a T4.
   *Implemented, not yet measured on a GPU:* `pgw.factory.gpu_prng` computes the ChaCha20 keystream
   (key still derived on the CPU by HKDF) straight into the int8 digit planes of the GEMM operand, so
   `r` never crosses the host bus. It is on by default on CUDA after a self-test against the CPU
   expander (`PGW_FACTORY_GPU_PRNG=0` turns it off), and the output self-check re-derives its sampled
   rows on the CPU (so the GPU path is refused when `self_check_rows` is 0; a kernel error or a failed
   check switches the generator back to CPU ChaCha20). The Kaggle job `pgw-factory-gpu-prng` runs the CUDA byte-identity tests and the 7B
   K32A12C27 slots/h comparison with the setting at 1 and at 0.
8. Close the integrity gaps: rotate segments after an integrity failure, scan the soundness floor at
   start-up, and spot-check factory `W r` independently (T-29).
9. Metadata hardening: implement `pad_rows_to` and the optional dummy rounds.
10. Move CI to the repository root, run the test jobs against the hashed locks, and add a nightly
    Kaggle or GPU job for `pytest -m gpu` and the e2e suite on CUDA.
11. Bring the test suite back under 5 minutes, for example by marking the real-Qwen e2e as `slow` in
    the fast lane.

**Later (roadmap)**
- Fused GPU ring kernel, SmoothQuant so that K32A8 becomes lossless, a gateway-GPU "self-factory"
  mode, speculative decoding and prefix caching to cut round trips, and tools/structured output in
  the API.

### Phase A wiring check (2026-11-20)
- Commit 450b788 (marked WIP by a forced hook commit) is complete: full suite 3615 passed / 122 skipped, tests/e2e 32 passed.
- Modal e2e (CPU provider, tunnel, 0.5B): exact (first_mismatch null); local_linears auto chose qkv+o on the gateway, 49.5 round trips per token (was ~100). That placement had a 26 ms median trip, so the warm 32-token chat took 47 s; trip latency, not compute, dominates on Modal.

## 2026-11-26 — Round trips per token in the client library: measured, strict speculation opt-in, 7B/32B/72B model

Scope: the shipped path (`from pgw import OpenAI`, `src/pgw/sdk`), CPU here only (no Modal, RunPod or Kaggle).
Measurement: `benchmarks/sdk_trip_profile.py` (real `pgw service` on loopback with the TCP+TLS lane, Qwen2.5-0.5B-Instruct,
the library's default stack, warm streamed greedy chats; trips counted exactly from the runtime's counters, the
provider's compute from the lane replies' `compute_us`).

- **Trips per decoded token: 97 = 4L + 1** (0.5B; 113 at 7B, 257 at 32B, 321 at 72B), all on the lane, 1 row each.
  This is the floor for the library: the gateway holds no weights (no local qkv/o), the exact int8 reference puts a
  nonlinear step between every pair of linears, and the masked head is one trip (a local head would stream the
  bf16 head per token). The 2-per-layer floor needs local linears or fused2; **fused2 cannot be on by default: it is a
  different fixed-point reference** (not bit-identical to the int8 one).
- **Per trip at 0.5B on this box: 3.0-3.1 ms**, of which the CPU provider's kernel path 2.8-2.9 ms (`compute_us`;
  a GPU provider replies in ~0.05-0.2 ms), TLS transport + provider lane I/O ~0.25 ms (stub floor 55 us p50,
  `lane_tls_bench.py`), gateway native work 70-113 us (prepare, finish, p31, float ops). Per token outside the
  trips: sampling over 152k logits + hand-off to the caller 1.2-1.4 ms (the gateway's HTTP path measured 9.2 ms in
  the 7B run); noise-sampled trips (5%) leave the native driver for the inline statistics (~0.2 ms extra each at
  0.5B). Nothing found on the critical path from serialization, per-trip Python hops, TLS records, the batching
  window (lane calls run inline on an idle device) or mask fetch (decrypt-ahead). The adaptive reply spin (400 us
  cap) never hits against a slow CPU provider (CPU waste here; it hits with a GPU provider).
- **Strict speculation in the library, opt-in** (`speculation="lookup" | "draft_head"`, `spec_rows` 2-8, default 3,
  `draft_head_path`; `PGW_SPECULATION`): strict passes (dummy passes after the answer: the provider sees one pass per
  generated token), fixed rows per pass with masked filler rows, the p31 check kept (asked for explicitly: no padded
  call is made), requests of one OpenAI `user` ordered behind the previous one's dummy passes, `close()` runs the
  dummy passes still due. Measured (0.5B, 64 tokens, CPU provider): code-edit prompt **tokens per pass 1.66 / 2.10 /
  2.52 at R = 2 / 3 / 4, trips on the critical path 97 -> 59 / 47 / 39 per token, 3.2 -> 4.7 / 5.75 / 5.9 tok/s**;
  open chat 1.14-1.15 tokens per pass (no gain: 3.08 -> 3.0-3.09 tok/s). Outputs bit-identical in every run.
  Not default: R masks per token, R rows of provider work and wire per trip, and a loss on open chat.
- **Local linears from the factory pack: not proposed.** One remote trip at RTT 0.2 ms is ~0.43 ms; an exact local
  1-row GEMV (stacked digits, 1 thread) is 2.2 / 1.7 / 27 / 12.5 ms per layer for qkv / o / gate_up / down at 7B and
  3.5-97 ms at 32B-72B. Only 0.5B qkv and o (0.13 / 0.12 ms) beat a trip.
- **Model** (`python benchmarks/round_trip_model.py --sdk`; components from the co-located model that reproduces the
  7B run; NIC 25 Gbit/s; H100, 72B on 2): ms per token (tok/s) at RTT 0.1 / 0.2 / 0.5 ms, today (R = 1): **7B 47.6
  (21.0) / 58.9 (17.0) / 92.8 (10.8); 32B 123.5 (8.1) / 149.2 (6.7) / 226.3 (4.4); 72B 163.0 (6.1) / 195.1 (5.1) /
  291.4 (3.4)**. Lookup on copy-heavy text (R = 3): 7B 34.4 / 39.8 / 55.9, 32B 89.5 / 101.8 / 138.5, 72B 123.8 /
  139.0 / 184.9. EAGLE-class draft head (alpha 0.8, R = 3; not trained): 7B 29.6-48.1, 72B 106.5-159.1. **The
  per-trip overhead beyond RTT (lane ~0.13 ms + gateway 0.1-0.13 ms) is larger than a same-data-center RTT**: at 72B
  and RTT 0.2 ms the pass is network 64 + lane 42 + gateway 41 + device 32 + wire 15 ms; lane 0.05 ms and half the
  gateway work would give 149 ms, plus a 0.8 draft head 87 ms (11.5 tok/s).
- Needs a GPU (RunPod later): the lane's fixed cost per trip over TCP+TLS on a real NIC, lookup tokens per pass on
  real 7B+ answers, a trained draft head's alpha at 7B/32B/72B, gateway work per trip at 72B widths with the trip
  split on a many-core gateway, speculation's wire cost at 25 Gbit/s.
- Tests: `tests/sdk/test_speculation.py` (bit-identical greedy / seeded / streamed with lookup and with a draft head;
  provider-metered trips, rows and bytes identical with every draft right vs none; `close()` runs the due dummy
  passes; options), `tests/benchmarks/test_round_trip_model.py`; tests/sdk 99 passed + 1 opt-in skip.


## 2026-11-26: client setup and per-token CPU at 32B-72B (pks-v3, scheme unchanged)

Goal: a smaller client burden (one-time setup, per-token CPU) for 32B-72B, with no scheme or parameter change and
no weaker check. Details, the byte-plane dial and the cache-size analysis: `research/reviews/pks-v3-client-setup-scaling.md`.

- **Hint build kernels** (`pgw_pks`: `a_planes`, `combine_planes`, `switch_hint`, `hint_keyswitch`;
  `pgw.hemask.hint_gemm`): `A`'s 7 byte planes once per linear instead of 8 per 1024-row block, 512-column int8 GEMMs,
  switch fused into the key switch, hand-written NTTs in the key switch. 3.8x less CPU on AMX, bit-identical
  (`tests/hemask/test_hint_build.py`). The int8 path runs only where CPU int8 GEMMs are exact (oneDNN's AVX2 kernel
  saturates: float64 there, as before).
- **`pgw sdk prepare --model ...`**: the setup ahead of the first request, the hint built in the same pass as the
  factory pack, on every core, resumable (partial hint state HMAC-authenticated under the cache's hint key; tampered
  state discarded, changed files rebuilt). Later clients use the prepared store only if its `meta.json` verifies.
  "Big box once, copy the cache": `docs/sdk/configuration.md`.
- **Per-token**: `mask_segment_slots` default 256 -> 1024, pks-v3 segments rounded to whole 256-slot groups (four
  groups per request amortise the upload's public NTTs); reply-key wire form native (3x).
- **Not done (measured, not worth it):** the byte-plane dial (at most ~13 % in a narrow band on AVX2-only hosts);
  hint compaction (48 stored bits vs 47.98 bits of entropy per coefficient pair); dropping the factory pack (the
  factory redraws `v W8` after every rejected segment).

| | 7B | 32B | 72B | how |
|---|---|---|---|---|
| hint build, CPU-s (1 thread, AMX) | 2226 -> 582 | 9618 -> 2319 | 24968 -> 4658 | extrapolated from measured per-row cost at real shapes |
| hint build, wall on this 4-vCPU box | 24 min -> 5.1 min | 1.95 h -> 19 min | 6.4 h -> 34 min | same |
| same with AVX-512 VNNI only / AVX2 only (CPU-s) | 2562 -> 934 / ~6200 -> ~5800 | not measured | not measured | 7B extrapolated, oneDNN ISA capped |
| whole setup, 0.5B end to end | 145 s -> 68 s wall, 383 -> 189 CPU-s | | | measured (`prepare_real_bench.py`) |
| peak RAM | 0.5B: 2.78 -> 2.14 GB; larger: about the same, bound by the f32 head (2.2 / 3.1 / 5 GB) | | | measured 0.5B; extrapolated |
| peak disk = cache (gateway pack + factory pack + hint) | ~13 GB, unchanged | ~45 GB, unchanged | ~91 GB, unchanged | measured 0.5B (2.17 GB both); extrapolated |
| bytes downloaded | ~11 GB, unchanged | ~37 GB | ~80 GB | extrapolated (W8 once + gateway pack) |
| hint on disk | 2.37 GB | 7.38 GB | 10.77 GB | computed (1.5 KB per output row); no lossless gain |
| client CPU per mask slot (1 thread) | 28.4 -> 24.9 ms | 90.8 -> 80.0 ms | 137.4 -> 120.8 ms | measured 0.5B (8.4 -> 7.2 ms), extrapolated per part |
| client cores at 25 tok/s | 0.71 -> 0.62 | 2.27 -> 2.0 | 3.43 -> 3.02 | same; at 32B/72B 1024-slot segments need `mask_store_gb` ~20/28 GB (else 2.23 / 3.38) |

On by default: the native hint kernels and `HintGemm` (where exact), the fused build when `provision` gets
`hint="local"` with `factory_pack=True`, prepared-store pickup (only with a verifying key), native `pack_ksk`, 1024-slot
segments. Opt-in: `resume` (used by `pgw sdk prepare`). Tests: tests/hemask, tests/provision, tests/sdk, tests/service
(1013 passed, 96 skipped; one SDK unit expectation fixed after the run). Open: an exact AVX2 int8 GEMM (~2x there),
a shard-at-a-time public-check packer (the `source="hf"` path keeps 15 / 65 / 145 GB of bf16 checkpoint), a
row-block masked-head derivation, baked 32B/72B pins.

## 2026-11-26: security review before the client pilot (SDK + provider)

Adversarial review of the SDK, the provider endpoints, the console, the images and the RunPod launcher. Every
finding with evidence and status: `research/reviews/security-review-2026-11.md`. No scheme, parameter or output
change.

- **Fixed (high):** S2, the CA pin covered only the first certificate of the CA reply while the whole reply became the
  trust store. An on-path attacker at first enrollment could append its own CA and receive the SDK key. Now the
  reply must hold exactly the pinned certificate (SDK, gateway enrollment codes, RunPod launcher).
- **Fixed (medium):**
  - S1: unauthenticated chunked bodies were buffered without a bound (`pgw.core.http_body.read_limited` on every
    JSON/form route).
  - S3: the request report revealed prefix-cache reuse; it now reports the prompt rows actually sent.
  - S4: with strict speculation, the report's latency revealed the real pass count; it is no longer sent.
  - S5: public `/readyz` showed other tenants' load; the load figures are only on the metrics port now.
- **Fixed (low):**
  - S6: client-certificate CN collisions across tenants are refused.
  - S7: a cached session's CA is re-checked against the key's pin.
  - S8: per-key report rate limit (`accounts_report_records_per_s`, 429; the client keeps the batch).
  - S9: client-side error-code allow-list.
- **Open** *(all closed since: "security review follow-up (S10 to S15, S20)" below)*:
  - S10: provisioning ignores key scope and quotas.
  - S11 (decision): persist the SDK integrity breaker.
  - S12 (decision): stop reporting requests that never reached the provider.
  - S13: bounded bodies on the customer gateway's routes.
  - S14: hash the rustup installer.
  - S15: external review of the Rust `unsafe` code.
  - S20: measure real vs dummy strict-pass timing.
- Tests:
  - New: `tests/sdk/test_security_review.py`, `tests/service/test_security_review.py`, 12 tests.
  - Result: tests/service, sdk, enroll, platform, deploy, console and provision gave 860 passed and 78 skipped.
  - 2 failures are not caused by this change:
    - `test_e2e_cpu.py::test_integrity_failure_raises_and_blocks` needs its module's earlier tests (it fails under
      xdist; it passes in order).
    - `test_console_build.py::test_wheel_contains_the_console` globs `pgw-*.whl` while the distribution is
      `pgw-sdk`.


## 2026-11-26: client disk: the factory pack stays (W8 is used per segment); hf-path duplicate removed

Question: replace the factory pack (int8 `W8`, 6.5 / 31 / 70 GB at 7B / 32B / 72B) with a small stock of secret
check-vector sets drawn at setup, sized against the factory's halt rule (3 failures + margin), plus a re-provision
on exhaustion. Full note: `research/reviews/pks-v3-client-disk.md`.

- **Does not hold; not implemented.** The factory needs `W8` for **every segment**, not just after rejections:
  each segment's own Freivalds vectors `uW = u W8 mod p` come from a fresh `os.urandom` seed per segment
  (SPEC 7.3 / 7.6). That is 3.3 / 11 / 17 MB per 1024-slot segment. The private `v W8` sets (94 / 309 / 486 MB) are
  also redrawn at every worker start and after every aborted segment, transport errors and SIGTERM included,
  and at 72B every segment (1.18 GiB > the 1 GiB in-RAM cache). A pack-sized stock would last ~2 / 2.9 / 4.1 M
  token rows before a full `W8` re-download, so that path would be routine, not rare. A pre-drawn stock would be
  as secret and as sound (same seeds, same derivation, KEK envelope as the store's segments), so no property needs
  weakening; it just does not pay. The levers that would pay are a transposed pks-v3 hint (no client `W8`,
  1.3 / 4.2 / 6.7 GB; a scheme change needing review) and dropping the float head after setup (1.6 / 2.3 / 3.7 GB;
  changes what is verified against the pin). Both are left as decisions.
- **Fixed:** `source="hf"` with a factory pack stored the plain tensors twice (gateway pack copy + factory pack).
  The gateway pack now hard-links plain-only files, as the provider path already did. 0.5B measured 2.17 -> 1.63 GB
  (`du`); 7B / 32B / 72B extrapolated 17.6 / 51.0 / 100.9 -> 13.3 / 44.8 / 90.9 GB. The provider path (SDK
  default) is unchanged at 1.63 / 13.3 / 44.8 / 90.9 GB on disk (`prepare_real_bench.py`'s 2.17 GB counts hard
  links twice).
- Measured too: `W8` order-0 entropy 7.06 bits per weight (lossless compression at most ~12 %, at the price of
  decompressing `W8` every segment): not pursued.
- Tests: `tests/provision` 44 passed / 1 skipped (new `test_public_source_factory_pack_shares_plain_files`);
  `tests/sdk` 103 passed / 1 skipped. tests/hemask and tests/service were not run (the change does not touch them).


## 2026-11-26 — Provider (GPU server) performance audit: memory headroom, bounded transients, H100 measurement plan

Trigger: the Kaggle T4 run of Qwen2.5-7B (`results/kaggle/sdk-7b-validation/run1_7b`) died at provider start-up with
CUDA OOM (a 4.06 GiB int64 copy of the masked LM head for its row sums, on top of 13.2 GiB of doubled weights).
`bfee514` fixed that call site. This audit looked for the same class of problem everywhere in the provider (load,
warm-up, serving, maskgen) and for the per-trip and throughput costs that matter at 72B. Outputs bit-identical
everywhere; the provider sees exactly the same bytes; no tenant data mixes (rows of different tenants share a GEMM
only where they already did, each reply is its own rows); no check weakened. No GPU was used: GPU memory is computed
from shapes and the transient models are checked against allocations measured on CPU (same torch code).

**Verified `bfee514`**: one resident copy of each weight (`w8` is a view of the padded device copy) and chunked row
sums. Still open after it, now fixed (ranked by impact):

| # | issue | evidence | impact before | fix (bit-identical) |
|---|---|---|---|---|
| 1 | eager (prefill-sized) ring matmul unbounded: a call held 24 B per digit row x output column (three stacked `(K/8, b, mp)` int64 temporaries) and staged its whole input | `offset_byte_core`; measured on CPU (`tests/core/test_ringmm_memory.py`: old combine 2-3x the new peak) | one 4096-row call (the default `max_batch_rows`): 7B gate_up K=48 **21 GB**, 72B **33 GB**, the 7B masked head ~84 GB: OOM on any GPU, even at 7B on an H100 | in-place combine from the int32 GEMM result (4 B per digit row + two `(b, mp)` int64); `DeviceExecutor._eager` and `_offset_byte_mm` split rows to `PGW_SERVICE_EAGER_WORK_MB` (512 MiB). Graph pool 0.44 -> 0.15 GiB |
| 2 | pks-v3 device key switch (`pks_gpu.keyswitch`) unblocked: ~(8 + 2G) x 32 KiB per output row at once | shapes in the code; `memplan` | one 4-group item: 72B gate_up ~19 GB, 7B masked head ~78 GB; two concurrent items | blocks of whole reply ciphertexts within `PGW_SERVICE_PKS_GPU_WORK_MB` (512 MiB) |
| 3 | `ComputeKernel.mulmod64` on CUDA (v3 GEMM and per-seed hint build) kept the whole `(m, cols, 8)` result on the device | `_maskgen_planes_cuda` | 5 GB for 7B's head hint (4096 columns), 1.9 GB at 72B gate_up, per concurrent item | results > 256 MiB copied back per column block (`_maskgen_blocks_to_host`) |
| 4 | maskgen constant via `w.sum(1, dtype=torch.int64)`, which casts the whole weight first (the comment said it did not) | torch profiler (`test_weight_rowsum_never_copies_the_weight_to_int64`) | the run1 OOM again (4.1 GiB at 7B's head, 9.3 GiB at 72B's) when maskgen precedes any compute on a linear (warm-up off) | `weight_rowsum` (byte-bounded chunks), shared with the compute path |
| 5 | weight upload made an unpadded device copy next to the padded one | `pad_weight_cuda`, `ComputeKernel` | +1 largest linear at load (1.16 GiB at 72B: the head) | host weights padded on the host (only if needed) and uploaded once |
| 6 | wire-v2 maskgen tail-bit decode: ~13 B of int64 per value at once | `v2_planes_source` | 1.6 GB per 4096-column 72B item | decoded in groups (`TAIL_GROUPS`) |
| 7 | maskgen column loop kept the previous block's planes, partials and result referenced while allocating the next | `maskgen_device` | +1 block (~0.7 GB) per item | `del` per block |
| 8 | Triton timing gate (T4 class) made int64 copies of its two 64 MB weights | `_timing_gate` | 1 GB transient at start-up on a 16 GB T4 | `weight_rowsum` |

Not fixed here (findings, by impact; the measurement plan below quantifies each):

- **pks-v3 provider hint does not fit any cache at 32B+ (throughput, host RAM).** `PKSProvider` keeps the switched
  hint `round(H q1 / 2^56)` as `m x 4096` uint64 per (model, customer seed, linear): **50.5 GB per seed at 7B, 157 GB
  at 32B, 230 GB at 72B**, against `PGW_SERVICE_PKS_CACHE_GB=8`. A decode cycle over all linears evicts every entry, so
  each item rebuilds its hint: a 4096-column GEMM plus the host-side modulus switch (`pks.switch_hint`, numpy, one
  thread: ~2 s per 4096 rows on this box, ~30 s for 72B's gate_up) plus a 2 GB host-to-device upload in the key
  switch. Proposal (needs the H100 numbers first): build and switch the hint on the device per item (the GEMM is
  ~10 ms on an H100 at 72B gate_up, the switch is the `_round_mul_shift` already in `pks_gpu`) and never cache it on
  the host above 7B; keep `y` on the device between the GEMM and the key switch (today device -> host -> device).
- **Maskgen device memory is bounded by count, not bytes** (2 concurrent items per GPU): the default v3 shapes need
  ~1.7-2.4 GB per item at 72B (`memplan`: 4.2 GiB of the 7.3 GiB headroom on one H100 with the masked head). A
  larger item (`max_request_bytes` allows ~11k columns on 7B's head) can still exhaust a small GPU. Proposal: a
  per-device byte budget for maskgen items from the plan (queueing only, never a refusal); not done here because it
  sits in the admission code other teams are changing.
- **Lane fixed cost (~0.13 ms estimated).** The native hot path's CPU work per trip is ~1 us (5 short String
  allocations and 4-5 global mutex acquisitions in admission and stats; lock contention only at many connections). The
  rest is wake-ups and synchronisation: `PGW_SERVICE_LANE_READ_SPIN_US=0` (blocking reads: one socket wake-up per
  trip) and `PGW_SERVICE_LANE_SPIN_US=50` against device times of 0.1-0.2 ms at 32B-72B gate_up/down (the wait falls
  through to `cuStreamSynchronize`). The session sweeps both over a real NIC; candidates for defaults only with numbers.
- **Start-up**: 72B lane graph capture is 321 linears x 8 row counts = 2568 graphs, each self-checked 3 times against
  the eager digit-GEMM path; warm-up captures the bucket graphs the fused GEMV does not take (one per linear on an
  H100, three on a T4); `verify=full` hashes 67 GB. All timed per stage in the session's `memory` phase.
- **Per-request Python (stream channel)**: K=32 rows are unpacked to int64 and then cut to their low 32 bits again
  for the GEMV (4-9 us per trip at 7B-72B widths, measured here); the lane path has no Python per trip.
- **Coalescing**: the Python batcher already merges rows of all tenants per (model, ring, linear) and splits replies by
  row range; the lane's coalescer (`PGW_SERVICE_LANE_COALESCE`, opt-in) merges concurrent same-linear calls into one
  launch (lane graphs exist for 1-8 rows; larger merges take the kernel path). Throughput at 1-64 connections with and
  without it is in the session (`lane_nic_bench.py serve --coalesce`).
- **Tensor parallel 72B** is implemented (output-row shards, `ShardedKernel`, sharded lane graphs and persistent
  workers, byte-identical replies). The plan gives 40.1 GiB peak per H100 at TP=2 (39.6 GiB headroom) and 72.3 GiB
  on one H100 (7.3 GiB headroom). Open: the masked head is not sharded (1.16 GiB extra on the first device); TP gather
  overhead per trip and multi-GPU maskgen balance are measured in the session (`provider_scale_bench --tp 1,2`).

**Memory plan** (`python -m pgw.service.memplan`; peak GiB per device with the masked head, the lane on, pks-v3
maskgen at 1024 columns x 2 concurrent items, 4096-row eager calls; `--json` gives the breakdown):

| model | GPU (torch capacity GiB) | TP | weights | steady | peak load | peak serving | headroom |
|---|---|---|---|---|---|---|---|
| 7B | T4 (14.56) | 1 | 6.58 | 7.49 | 7.68 | 11.43 | 3.13 |
| 7B | L4 / A10G (22.03) | 1 | 6.58 | 7.49 | 7.52 | 11.44 | 10.6 |
| 14B | L4 / A10G (22.03) | 1 | 13.03 | 13.95 | 13.99 | 16.73 | 5.3 |
| 32B | L40S (44.52) | 1 | 29.79 | 30.77 | 30.8 | 34.66 | 9.86 |
| 32B | H100 80GB (79.65) | 1 | 29.79 | 30.77 | 30.8 | 34.66 | 44.99 |
| 72B | H100 80GB (79.65) | 1 | 66.55 | 67.59 | 67.62 | 72.33 | 7.32 |
| 72B | H100 80GB | 2 | 33.86 | 34.81 | 34.84 | 40.1 | 39.55 |
| 72B | H200 (140.4) | 1 | 66.55 | 67.59 | 67.62 | 72.33 | 68.07 |

Before (same settings): `rev=run1` 7B on the T4 needs 14.37 GiB resident + the 4.06 GiB copy at warm-up: the plan
reproduces the run's log from shapes alone (`tests/service/test_memplan.py`). `rev=bfee514`: 7B load fits (8.9 GiB)
but serving does not fit any GPU (one 4096-row K=48 eager call on gate_up 21 GB; the unblocked key switch tens of
GB); 72B on one H100: 69 GiB at load, serving out of memory. Not possible at all: 32B on one 24 GB card, 72B on one
48 GB card. The registry now logs the plan's verdict (and the shortfall with its breakdown) before a pack loads on CUDA
(advisory; the load proceeds).

**H100 measurement plan** (`deploy/runpod/provider_perf_session.py`, ~75 min of one H100 + a CPU client pod; `--dry-run`
prints it): `env`; `memory` (7B/32B/72B at full size with random weights through the service's kernel path: load,
warm-up, lane capture, a 4096-row eager call on the largest linear and the head, the pks-v3 hint build, host switch and
key switch; measured `memory_allocated`/peak and seconds per stage against the plan); `device_time`
(`provider_scale_bench.py`: kernel / graph / persistent service time and the GEMV floor at 7B/32B/72B, 1-8 rows, TP 1
and 2); `gemv_table` (H100 launch table); `lane_local` (UDS/shm lane next to Python-path and maskgen-GEMM load);
`maskgen` (wire v2 fused, 1/4/16 groups, 1/2/4 concurrent, decode next to it); `lane_nic`
(`benchmarks/lane_nic_bench.py serve` on the GPU pod and `client` on the client pod: trip p50/p90/p99, `compute_us`,
TCP RTT -> `lane_fixed_us = trip - compute - rtt` and `transport_us` from a zero-compute stub listener; 1/4/16/64
connections; rerun with `--read-spin-us 2000 --zero-copy`, client `--spin-auto-us 400`, and `--coalesce`). Decisions
it feeds: lane spin/zero-copy/coalesce defaults; whether the pks-v3 hint moves to the device; maskgen byte budget;
72B on one H100 vs TP=2.

Tests (CPU, this box): `tests/core/test_ringmm_memory.py` (measured allocations vs the model, old vs new combine,
chunked == whole, row sums without the int64 copy, padding), `tests/service/test_memplan.py` (run1 reproduction,
targets fit, legacy model vs measured, CLI), `tests/service/test_pks_gpu_memory.py` (blocked key switch == whole == CPU
path), `tests/service/test_kernels_memory.py` (blockwise maskgen result == whole), `tests/service/test_maskgen_gpu.py`
(tail decode in groups), `tests/benchmarks/test_provider_perf_tools.py` (NIC bench over loopback TLS, session dry
run). New settings (all keep outputs identical): `PGW_SERVICE_EAGER_WORK_MB` (512), `PGW_SERVICE_PKS_GPU_WORK_MB`
(512). GPU confirmation of the CUDA-only code paths (one upload in `ComputeKernel`, `_maskgen_planes_cuda`'s
blockwise copy) is in the session's `memory` phase; no Kaggle kernel was spent.

## 2026-11-26: weight-free client checks: no candidate drops `W8` without a loss; nothing implemented

Question: can the client avoid keeping `W8` on disk (6.5 / 31 / 70 GB) without weakening anything, e.g. with the
"transposed pks-v3 hint" (open option 3 of `pks-v3-client-disk.md`)? Full note:
`research/reviews/weight-free-client-checks.md`; numbers, Monte Carlo checks and two attack demos:
`research/reviews/weight_free_numbers.py`.

- **Structural result:** the online p31 check needs `u W8` for provider-unpredictable `u`, tied to the true `W8` by an
  anchor. The anchor can only be `W8`, a long-lived secret derived from it, or a public commitment plus proofs.
- **A0, transposed pks-v3 product used as is: not viable.** An additive shift of the decrypted `u W8` (F1) makes the
  gateway's verdict a function of a private activation: accepts iff `xq_j == 0` (demonstrated). It also has
  9504 ring samples for the head (packet states <= 1848) and the head's worst-case column norm is refused
  (2^14.19; measured 0.5B 2^13.81).
- **A1, A0 plus a secret sketch: viable with a stated loss.** It has a long-lived secret and costs a second hint of
  1.28-6.67 GB. It is dominated by B. **A2, A0 plus a public commitment:** no loss in principle, but it is a new
  proof system (~23-74 MB per opening). Not viable now.
- **B, secret basis `u = B c`, `v = B_v c`: viable with a stated loss.** Soundness is unchanged while `B` is secret
  (2^-31 at k = 5, 2^-42 at k_v = 28). Disk is 0.11 / 0.36 / 0.57 GB, and factory CPU on weights drops from G MAC to
  M MAC. The loss: a one-time leak of `B`, `G` or `k` past `u` breaks integrity for every future segment until a `W8`
  re-stream (demonstrated: 2000/2000 forged acceptances). It changes SPEC 7.3.
- **C, no `W8` at rest, streamed and pin-verified once per 64 segments plus a pre-drawn stock: no security loss.**
  It costs +0.8 / 1.1 / 1.7 % network per segment, a 0.6 / 1.9 / 3.1 GB stock, and a runtime dependency on a `W8`
  source. This is the client-disk note's lever 1; it is a factory decision.
- No candidate passed "no loss", so no prototype was written. The default is unchanged. If disk forces it, the
  recommendation is C; B only as an explicit, documented opt-in with rotation and external review.


## 2026-11-26: client library at many concurrent streams and worker processes (scaling)

Scope: `from pgw import OpenAI` with N = 1-64 concurrent streams per machine and several worker processes per
machine; CPU here (shared 4-vCPU box, provider on the same CPU, load 4-28 from other jobs). Guide and sizing table:
`docs/sdk/scaling.md`; tools: `benchmarks/sdk_concurrency_bench.py` (N streams, `--procs`, bit-identity check,
CPU by thread, lane counters), `benchmarks/sdk_scale_sizing.py` (cores / RAM / disk per model and N).

- **Measured** (every answer bit-identical to the same request alone, 0 errors): tiny 1/4/16/64 streams 45/98/173/211
  tok/s total, client CPU 6.5 -> 2.4 ms/token (batching amortises the per-trip part), rows per trip 2.2 -> 75;
  4 worker processes at 64 streams 296 tok/s. 0.5B 1/4/8 streams 2.1/4.4/4.9 tok/s: the CPU provider saturates
  (268 -> 114 ms provider CPU per token), the client's plaintext thread is 20-27 % busy.
- **Bottlenecks, ranked** (7B-72B extrapolated, H100 provider): (1) mask production: one slot per generated and
  per uncached prompt token, 25 / 80 / 121 ms of client CPU per slot at 7B / 32B / 72B, no batching gain: 16 streams
  at 7B need ~20 factory cores; (2) mask stock admission: a request starts only when the stock covers every running
  request's prompt + max_tokens: the default `mask_store_gb = 8` admits about one 500 + 512-token request at a time
  at 7B (6.2 MB per slot) and none at 32B/72B (19 / 28 MB per slot); (3) the trip lane serves one item per call
  (*fixed since: lane format 2 multi calls, "concurrent answers on the native trip lane" below*):
  with two answers in one step every decode trip falls back to the WebSocket stream channel (tiny: 280/312 lane
  trips at N=1, 1/345 at N=4; per trip 0.22 vs 0.35 ms idle, 0.28 vs 2.2 ms with busy Python threads; not changed
  here: hot-path / lane owners); (4) the gateway's plaintext thread (GIL, one core per process): ~4-5 streams per
  process at 7B-72B decode speeds; (5) provider batching: not a client limit. KV: float32, 115 / 524 / 655 KB per
  token at 7B / 32B / 72B; the prefix cache (1 GiB default) saves masks and the prompt pass on multi-turn chats but
  needs the gateway's own head (`prefix_cache=True`).
- **Multi-process (gunicorn)**: workers with one key and cache dir already shared one store (cursor moves in
  `BEGIN IMMEDIATE` + fsync under an `flock`: single use holds across processes) and one factory (owner lock), but
  each had its own in-memory entitlement ledger, so one worker could admit a request on stock promised to another
  worker's running answer, whose extension then failed mid-answer (reproduced: 1 failed session in the stress
  layout without the shared ledger). Fix: `SharedLedger` (`mask_ledger_dir`; per-process outstanding-entitlement
  files, admission under a cross-process `flock`, dead processes pruned; published values are upper bounds, so it
  is conservative), on in the SDK, off for `pgw gateway` unless set. Recommended pattern: one client per worker
  created after the fork, one factory per machine (no separate mask daemon needed).
- **Bursts**: requests waiting for masks are now served FIFO (only the head polls; a large request is no longer
  starved by smaller later ones); a request that can never fit the stock fails at once (`BadRequestError` naming
  `mask_store_gb`); `mask_admission="overcommit"` (opt-in, `PGW_MASK_ADMISSION`): admit on prompt + one chunk per
  running answer, extensions wait for the factory (`mask_extension_wait_s`, 30 s) then fail closed. 64 tiny streams
  on a 512-slot stock: all served bit-identical, TTFT p95 14.3 s (entitled) / 11.6 s (overcommit).
- **Fairness**: admission FIFO across users, one token per running answer per step; `user` keys the prefix cache
  and the strict-speculation ordering only; no per-user quota in the client (documented). Provider side:
  per-tenant `tenant_max_inflight` / `tenant_rows_per_s`, global `max_inflight_requests`; refusals carry
  Retry-After and the client retries.
- Tests: `tests/maskstore/test_scale_stress.py` (4 processes x 4 threads of incremental sessions with ahead
  extensions, background retirement, gc and a concurrent factory: no slot or mask row twice, consumed total exact,
  no mid-answer failure; ledger unit test), `tests/sdk/test_e2e_scale.py` (bursts larger than the stock with both
  admissions, 3 worker processes x 3 threads: bit-identical, no slot reserved twice in the store's records),
  `tests/sdk/test_unit.py` (FIFO order, never-fits). tests/sdk 108 passed + 1 opt-in skip;
  tests/gateway/test_incremental_masks.py + test_masked.py 35 passed; tests/maskstore: 245
  passed; 2 failures not from this change (`test_ringbuf` / `test_ringshm` crash children: `tests/platform` shadows
  the stdlib `platform` module on their path). Open: multi-item (or parallel single-item) lane calls, a shared
  system-prompt prefix across conversations (needs a privacy review), per-user quotas.


## 2026-11-26 — pks-v3 provider hint recomputed per item on the device (maskgen at 32B+)

Problem (provider audit above): `PKSProvider` cached the switched hint `round(W8 A q1 / 2^56)` as `m x 4096` uint64 per
(model, customer seed, linear): 14.9 / 50.5 / 157.5 / 229.8 GB per seed at 0.5B / 7B / 32B / 72B, against the 8 GB
default cache. A decode cycle over all linears evicted every entry (LRU with cyclic access), so every item rebuilt its
hint. A rebuild was: `A` expanded on the host (numpy `a_mlwe_rows`), a 4096-column GEMM, `H` copied to the host, the
numpy switch on one thread, and the switched hint copied back for the key switch. On top of that, `y` went device ->
host -> device.

**Fix** (`pgw.service.pks_gpu.item`). The scheme, the wire and the replies are unchanged. Each item now runs on the
kernel's device, in row blocks of whole reply ciphertexts (`PGW_SERVICE_PKS_GPU_WORK_MB`):
- Only the customer's public ring rows `a_C` go up (`n/16 x 4096`, 1/16 of `A`). `A`'s 7 byte planes are expanded on
  the device.
- Per row block, four steps run back to back on the device:
  1. the hint rows `W8 A`, using the same int8 plane GEMM and recombination as `ComputeKernel.mulmod64`;
  2. their switch, using the existing 14-bit-digit `_round_mul_shift`;
  3. the item's `y = W8 b`;
  4. the existing device key switch.
- Only the reply rows come back. Nothing is cached.

Policy (`PGW_SERVICE_PKS_DEVICE_HINT`):
- `auto` (the default) uses the device path on CUDA kernels when one seed's hints for the whole model exceed
  `PGW_SERVICE_PKS_CACHE_GB`. The handler passes the model's dims; without them, the linears seen so far count. Below
  the threshold the host cache works as before.
- `device` and `cache` force a path.

Safeguards and related changes:
- Before first use on a device, `item_self_test` must match the host path bit for bit (zero-padded weight, extreme
  rows). Any device error makes that device fall back to the cache path for good.
- Concurrent device items run on their own CUDA streams.
- Tensor-parallel kernels run one item part per shard when the shard bounds are multiples of 32 rows. Otherwise they
  use the cache path.
- The cache path now switches natively when `pgw_pks` is present (`pgw_pks.switch_hint`, multi-threaded, same bits).
- The torch NTT's short butterfly stages (lengths 1-32) had stride-2 operands and took nearly all of its CPU time. They
  now run on a transposed layout with in-place butterflies. The butterflies and the arithmetic are the same, so the
  results are bit-identical (checked against `pks.Ntt`), and every device gets fewer operators and contiguous operands.

The provider learns nothing new: it uses the same `A` derived from the public `a_seed` and the same ciphertexts. There
is no scheme or wire change. The replies are bit-identical, so the customer's checks (self-check, Freivalds) are
unchanged.

**Exactness evidence** (`tests/service/test_pks_device_item.py`, run on CPU torch, the same code that runs on CUDA):
- The device item equals the host path (numpy GEMM, `pks.switch_hint` and the CPU key switch) at two shapes:
  - 64x48: 1 group, unpadded weight;
  - 96x80: 2 groups, zero-padded weight, three 32-row blocks.
- Through `PKSProvider`, the device bytes equal the cache-path bytes, and the customer (`pks_native.Customer`, with its
  own compact hint) decrypts them to `W8 r mod 2^32`. This holds in four cases:
  - 64x48 with 300 rows (2 groups, one of them partial);
  - a 2-shard tensor-parallel kernel;
  - Qwen2.5-0.5B's `o_proj` shape (896x896);
  - Qwen2.5-0.5B's `down_proj` shape (896x4864).
- Further checks:
  - the `A` columns equal `a_mlwe_rows`;
  - the NTTs equal `pks.Ntt`;
  - the native host switch equals numpy;
  - the device self-test passes on CPU;
  - a failure falls back and returns the same bytes;
  - the policy and env parsing behave as specified.

**Memory**, measured with `tests/core/alloc_tracker.py` on CPU: one item's peak tensor bytes stay within
`pks_gpu.item_bytes` and above 40 % of it, and 4 row blocks peak at less than half of 1 block. An item holds the hint
operand's planes (`7 x 4096 x n_pad` B, 0.85 GB at 72B `down`), the upload's planes and one row block. `memplan` now
models this (`--pks-hint`, `--pks-cache-gb`, `--pks-work-mb`). Maskgen items, 2 concurrent, GiB per device:

| model | before | after |
|---|---|---|
| 7B | 3.45 | 2.32 |
| 32B | 3.39 | 2.94 |
| 72B | 4.24 | 3.07 |
| 72B, TP=2 | 4.78 | 3.07 |

Peak serving for 72B on one H100 drops from 72.33 to 71.16 GiB (headroom 7.3 -> 8.5).

**Throughput, ESTIMATES** (`python -m pgw.service.memplan --pks-throughput`). The setting is one H100 serving
1024-column items of every linear in turn; slots per second = 1024 / seconds per cycle. Inputs:
- Device rates are datasheet-based, not measured: int8 GEMM 1.2 POPS (60 % of peak), torch element-wise 2.3 TB/s (70 %
  of HBM3), PCIe 10-20 GB/s, 5 us per eager operator.
- Device traffic was counted per torch operator on CPU (`test_traffic_constants_are_the_counted_ones`): the key switch
  moves 11.7 + 1.66 G MB per output row; the switch moves 488 B and the recombination 140 B per hint value.
- Host rates were measured on this loaded 4-vCPU container: numpy switch 0.40 ms per 4096-value row, `a_mlwe_rows` 56 us
  per feature, ChaCha20 1 GB/s. A server core is perhaps 2x faster, so "before" may be pessimistic by up to ~2x.

| model | hint GB / seed | before: cache miss (numpy switch) | miss, native switch | cache hit | after: device hint |
|---|---|---|---|---|---|
| 0.5B | 14.9 | 5.2 slots/s | 34 | 173 | 182 |
| 7B | 50.5 | 1.5 | 8.8 | 51 | 53 (57 with a 1 GB work budget) |
| 32B | 157.5 | 0.48 | 2.8 | 16.4 | 16.6 |
| 72B | 229.8 | 0.33 (3141 s per cycle, of which the host switch is 2774 s) | 1.8 | 11.2 | 11.1 (12.0 with 1 GB) |

That is ~35x at 7B-72B. The device path also matches a cache hit, which can no longer happen at these sizes: uploading
a cached hint's 32 KB per row costs about as much as recomputing it. So on an H100, caching does not pay even where the
hint fits.

What remains at 72B (92 s per cycle):

| stage | s per cycle |
|---|---|
| torch key switch | 56 |
| hint switch + recombination | 19 |
| host ChaCha20 of the ring rows (GIL held) | 9 |
| GEMMs | 6 |
| uploads | 4 |

The key switch runs 16 + G NTTs of 8192 points per 32 rows over 4 primes as eager int64 operators, about 4.5 KB of
memory traffic per hint value. The next lever is a fused NTT kernel (Triton or CUDA) that does one read and one write
per transform instead of ~90 operator passes. That would make the key switch roughly an order of magnitude cheaper and
leave the GEMMs (~6 s per cycle, ~150 slots/s at 72B). It is not done here because it needs a GPU to validate.

**Per-device maskgen byte budget** (the audit's proposal; `pgw.service.maskgen_budget`, opt-in through
`PGW_SERVICE_MASKGEN_DEVICE_MB`, off by default).
- An item now also waits, first in first out, until its modelled device bytes fit next to the items already running on
  its CUDA device. It is never refused: an item larger than the whole budget runs alone.
- The only handler change is a hook of about 10 lines in `ServiceHandler._maskgen_in_slot`, released together with the
  slot. Nothing else in admission changed.
- pks-v3 items are now 0.6-2.1 GiB whatever the linear. The budget matters for wire v2 items, which grow with their
  columns (11k columns on 7B's head: 7.8 GiB).

**H100 measurement** (`deploy/runpod/provider_perf_session.py`, `maskgen` phase, now ~20 min). After the v2 bench, the
phase runs the largest linears of 7B (gate_up, down, the head), 32B and 72B through `ComputeKernel` + `PKSProvider` on
three paths: cache miss, cache hit, and the device path at 512 / 1024 / 2048 MB work budgets. For each it records
seconds, us per row, peak memory and whether the replies are byte-identical, next to the estimate from
`memplan.pks_item_seconds` (output: `maskgen_pks_v3.json`). The CPU smoke run passes, with identical replies. Decisions
it feeds:
- the `auto` threshold, or using the device path always;
- the default work budget;
- whether the fused NTT comes next.

Tests:
- Full runs, one directory at a time: `tests/service` and `tests/hemask`; also `tests/benchmarks/test_provider_perf_tools.py`.
- New: `tests/service/test_pks_device_item.py` (19 tests, 2 of them `slow`: the 0.5B shapes, ~95 s together here) and
  `tests/service/test_maskgen_budget.py` (5 tests).
- New settings: `PGW_SERVICE_PKS_DEVICE_HINT` (default `auto`) and `PGW_SERVICE_MASKGEN_DEVICE_MB` (default off).
- No GPU or cloud spend.

## 2026-11-26: pack format 2: embeddings / LM head stored in bf16 (the checkpoints' dtype), exact upcast at load

Goal: halve the f32 plain tensors of every pack (gateway pack, factory pack, provider pack) without changing one
computed bit. The public Qwen2.5 checkpoints are bf16. Format 1 widened `embed_tokens` / `lm_head` to f32 at pack
time, which is exact, so storing the bf16 bytes and widening at load gives the same f32 values.

- **Exactness, measured on all four release checkpoints** (`scripts/pin_model_fingerprints.py --compare-format1`;
  Kaggle CPU kernel `pgw-sdk-pins` v2, 26.6 min, `results/kaggle/sdk-pins-format2`; 0.5B / 1.5B also here, same
  output): every checkpoint tensor checked is bf16 (0.5B/1.5B: all 290/338 tensors; 3B/7B: every plain tensor). The
  format-1 pack re-derived each old pin exactly (same bytes as the old packer). Every plain tensor loaded from it equals
  the format-2 pack's bf16 widened to f32, bit for bit (7B: 2 x 545 M elements). Linear tensor entries are identical.
- **Format** (`manifest.version`, fingerprinted): `2` means `quant.plain_dtype: "source"`. Per tensor, the
  checkpoint's dtype is kept if it is bf16 (bytes stored unconverted), else f32 is used (f16/f32 sources are never
  narrowed). The choice is each tensor's `dtype` entry, so the fingerprint covers dtype and bytes. Norms stay f32
  (enforced for format 2). Format 2 is the default of `pack_checkpoint`, `pgw pack` (`--plain-dtype source`), the hf
  provisioning path and the pin script. `plain_dtype="float32"` still writes format 1 byte-identically, so the old
  fingerprints reproduce.
- **Compat decision: old packs still load.** A format-1 pack is internally consistent and is checked against its own
  fingerprint exactly as before. Refusing it would only force re-downloads without adding integrity. What changes is
  the pins. Clients now pin format-2 fingerprints, so a client cache keyed by an old fingerprint is no longer used
  (provisioning makes a new `models/<fp16>` dir; the old one is orphaned and can be deleted to reclaim the space). A
  provider still serving a format-1 pack is refused by the existing pin check. An unknown future version gets
  "uses pack format version N ... re-provision (re-pack)". pgw installs from before format 2 reject format-2 manifests
  with a schema error ("Input should be 1"), so provider and client must both upgrade.
- **Provider:** the service and factory load plain tensors through `load_plain` (f32), or, for the int8 head, in the
  hash-checked stored dtype (`quantize_head` widens bf16 to f64 exactly). Results are unchanged.
  `deploy/runpod/pod_boot.ensure_pack` now rebuilds a volume's pack once when its format is not 2 (otherwise SDK
  clients would refuse it). The Modal deploy packs were not touched (there the provider and its own gateway use one
  volume, so their fingerprints agree).
- **New pins** (`PINNED_MODELS`, docs/sdk): 0.5B `d773a502fb97...`, 1.5B `3d198de49e61...`, 3B `0ec0b8b24b2b...`,
  7B `f7c6fb4f5f8f...` (full values in `src/pgw/sdk/provisioning.py`). The format-1 pins are kept in
  `scripts/pin_model_fingerprints.py:FORMAT1_PINS`, for the reproduction check only.
- **Disk** (`du -sb` of the cache after `provision(source="hf", hint="local", masked_head=True, factory_pack=True)`
  at 0.5B, tied head): format 1 1.629 GB, format 2 1.357 GB. The gateway pack went from 0.558 to 0.285 GB, and its
  plain file is hard-linked into the factory pack. Plain bytes, measured: 0.545 -> 0.272, 0.934 -> 0.467,
  1.245 -> 0.623 and 4.361 -> 2.181 GB at 0.5B / 1.5B / 3B / 7B. Client totals (provider path, layout of review
  section 4): 7B 13.3 -> 11.1, 32B 44.8 -> 41.7, 72B 90.9 -> 85.9 GB (32B/72B extrapolated: -3.11 / -4.98 GB).
- **Tests:**
  - `tests/modelpack/test_source_dtype.py`: format, the per-tensor choice (including an f32-head / bf16-embedding
    checkpoint), bit-identical loads and masked head, fingerprint binding v1 vs v2, rejection of a dtype relabel, a
    re-fingerprinted forgery and a flipped bf16 bit, version / plain_dtype consistency, unknown version.
  - `tests/gateway/test_source_dtype_gateway.py`: the masked gateway, the plaintext reference and the int8-head
    reference give identical tokens and logprobs from format-1 and format-2 packs (tiny untied bf16 model and
    Qwen2.5-0.5B).
  - `tests/provision/test_source_dtype_provision.py`: the provider path serves the bf16 bytes and verifies them in
    full, tampering is rejected, the hf path re-derives the v2 pin and hard-links, and the v1 parameters still
    reproduce the v1 pin.
  - `tests/deploy/test_runpod.py::test_ensure_pack_rebuilds_older_format`.
  - Runs (one dir at a time, `PYTHONPATH=src:.`): modelpack 107 passed; provision 46 passed / 1 skipped; core 560 /
    28 skipped; cli 209 / 3 skipped; enroll 9; deploy/test_runpod 15; gateway masked / masked-head / setup /
    setup-reuse / enroll-client / source-dtype 54 passed (0.5B case run separately: passed); console all but the
    known `test_wheel_contains_the_console`; sdk file by file (a whole-dir run was OOM-killed on the shared box)
    105 passed / 1 skipped. `tests/sdk` needs an absolute PYTHONPATH: otherwise its factory subprocess imports the
    main checkout's pgw, which reads format 1 only.
  - The setup wizard's pack job and the enrollment metadata now default a missing `plain_dtype` to `"source"` (they
    used `"float32"`, which would rebuild format 1 and miss a format-2 service fingerprint).

## 2026-11-26: security review follow-up (S10 to S15, S20)

The open items of `research/reviews/security-review-2026-11.md` are closed. Every one tightens security, and
each has regression tests. No scheme, parameter or output change: answers are bit-identical.

- **S11, persisted integrity halt** (`pgw.sdk.halt`):
  - Halts are kept per (key, model) in the state directory: 0600, and HMAC'd under their own domain label with a
    0600 key file.
  - A restart does not reset them, and other processes of the key pick them up on their next request.
  - A tampered record, a missing MAC or a missing MAC key reads as halted (fail closed), and the message says so.
  - Cleared only by `pgw sdk clear-halt --model M --yes`, which prints a warning, or by
    `reset_integrity_breaker()`. Documented in `docs/sdk/troubleshooting.md`.
- **S12:** requests that failed before their mask session opened are no longer reported. They are counted in
  `status()["local_errors"]` and logged locally (CONTRACT A3b, configuration).
- **S20, measured on CPU** (`benchmarks/sdk_dummy_pass_timing.py`): a TCP relay records the provider's view of the
  trip lane.
  - Steady-state real and dummy passes were already alike. The gap median was 2.5 ms for both, with KS 0.09
    and classifier accuracy 0.54. Trip sizes, the provider's time and the intra-pass gaps also matched.
  - The boundary was distinguishable. A `status.json` write (with fsync) at the end of the answer made the first
    dummy pass's gap 4.5 ms median and 22 ms p90, against 4.9 ms p90 for real passes. "Largest gap = boundary"
    found the real pass count in 13/48 and 15/48 requests, where chance is 2/48.
  - Fix: the write waits until the generator is idle. Also added: a strict-mode pass-gap floor
    `spec_pass_gap_ms`, 8 ms by default.
  - After the fix the boundary attack is at chance (1/48), and the first dummy gap is within the 5 % KS critical
    value: 0.16 and 0.13, against a critical value of 0.21.
  - Cost: about 8 ms per pass (+34 % on the tiny model, 3 to 8 % on real models).
  - Qwen2.5-0.5B was not run: its plain mask stock needs about 10 GB of disk.
- **S10:** provisioning routes apply the SDK key's model scope (403), its revocation, its daily quotas (429) and
  its row bucket (each H chunk costs 256 rows; the client waits out a short `Retry-After`). `/provision/models`
  lists only the key's models.
- **S13:** the customer gateway's own JSON routes use `read_limited`: the setup wizard (unauthenticated sign-in),
  `/admin/keys`, `/admin/models/*/reset`, and the console sign-in and playground.
- **S14:** the provider images download `rustup-init` 1.28.2, pinned and checked with `sha256sum -c` per
  architecture, instead of piping `sh.rustup.rs`.
- **S15:** no code change. The review doc has an "External audit scope" section: `pgw-fast` first (273 unsafe,
  reply path first), then `pgw-lane` (75), with per-file counts.
- Test hygiene:
  - `test_integrity_failure_raises_and_blocks` is self-contained (its own key, cache and baseline). It now also
    covers restart, `clear-halt` and reset.
  - `test_wheel_contains_the_console` globs `pgw_sdk-*.whl`.
- Tests: the new tests are listed in the review doc (sdk +5, service +2, gateway 2 new, deploy). Each directory
  was run on its own with `PYTHONPATH=src:.`, and nothing failed:

  | directory | passed | skipped |
  |---|---|---|
  | sdk (e2e and speculation included) | 115 | 1 |
  | service | 447 | 75 |
  | gateway | 281 | 0 |
  | console | 96 | 0 |
  | provision | 43 | 1 |
  | deploy | 127 | 1 |
  | enroll | 9 | 0 |
  | platform | 32 | 0 |
  | cli | 209 | 3 |

  - `tests/gateway/test_qwen05b.py`, the real 0.5B model, was killed for lack of memory on the shared host,
    twice. The gateway was run in two halves for the same reason.
  - tests/docs still has failures from before this change (`PGW_SPECULATION` is not a settings field, and
    test_numbers). None are in files this change touched. *(Fixed in the integration sweep; test_numbers needs the
    gitignored `results/kaggle` logs of the main checkout.)*


## 2026-11-26: client per-token hot path (library path, 0.5B on CPU, exact)

Scope: what `pgw.OpenAI` does per decoded token outside the GPU. Measured with `benchmarks/sdk_trip_profile.py` (now
also reports client CPU per token, the driver's phase counters and the lane's split of a trip) plus per-thread
timers on the hot functions, the CPU provider on the same 4-vCPU box (load 5-26 from other jobs: walls are noisy,
CPU times and counts are not). A/B: the previous branch head's `pgw_fast` vs this one, interleaved, 2 x 2 runs of 48
tokens; every run gave the same tokens and logprobs (same hash), the same 97 trips per token, the same rows and bytes.

| per decoded token, 0.5B, 97 trips | before | after | how |
|---|---|---|---|
| lane reply spin (400 us cap, no hit against a CPU provider) | 38.8 ms CPU | 0 | spin skipped when the learned reply time exceeds the cap |
| lane, per trip beyond the provider's compute: wait / send + read | 102-150 / 131-133 us | 25-32 / 85-93 us | the spin gave the provider's CPU back; batched TLS reads |
| TLS `recv` calls per trip (7B shapes, strace) | 8.0 | 1.16 | one `recv` of everything the socket holds, records fed to rustls from memory |
| W r chunk decrypts on the token's path | 100 | 38 | a part reserved ahead decrypts every linear's first lease at once |
| lease creation + lookup (Python, incl. those decrypts) | 13-16 ms wall, 5.7-6.6 ms CPU | 4.8-4.9 ms wall, 3.3-3.5 ms CPU | same |
| noise-sampled trips handed back to Python (send + finish) | 4.9 | 0 | held in the driver, sent on the native lane after the inline statistics |
| client CPU, whole process | 87-89 ms | 45-46 ms | |
| native gateway work (prepare + finish + float ops per trip) | 74-128 us | unchanged | qkv/o/gate_up/down prepare ~33/33/33/85 us, finish 16/16/42/16, float 24/2/16/2 |
| inline noise statistics (5 sampled trips) | 2.0 ms CPU | unchanged | 93 / 160 / 225 / 746 us at n = 896 / 4864 / 8192 / 29568 (one core, min) |
| sampling: greedy + logprobs / temperature > 0 / top_p < 1 | 0.35 / +3.9 / +14 ms | 0.35 / +3.9 / +4.4 ms | top-p filter sorts with numpy (bit-identical); `multinomial`'s 152k exponential draws stay |

Changes (each exact: same bytes, request ids, rows, checks, draws; tests added):
- **Adaptive spin** (`rust/pgw-fast/src/lane.rs`): a reply expected after the cap is not spun for; the estimate keeps
  learning from blocking waits, so the spin comes back when replies speed up (a GPU provider under the cap: as before).
- **Batched TLS reads** (same file): the lane client's TLS reader does one `recv` (up to 256 KiB) and hands the
  records to rustls from memory (rustls reads at most 4 KiB per call: 37-60 `recv` per 150-240 KB reply); spin polls
  use `MSG_DONTWAIT`, so the socket's mode no longer toggles per trip. Test: replies up to 1.9 MB arrive intact.
- **Held sampled trips** (`decode.rs`, `decode_native.py`): with inline noise statistics the driver keeps the
  prepared sampled trip (ticket, id, payload), Python runs the statistics before anything of it leaves (an alarm
  stops it, abort zeroes it), the next driver call sends exactly those bytes on the native lane and finishes it there.
  `PGW_DECODE_NATIVE_HOLD=0` restores the hand-back. Tests: bytes, logits and statistics identical across no driver,
  hand-back and hold; no hand-back with the hold; the existing alarm tests pass on the lane.
- **First-lease decrypt-ahead** (`pgw/gateway/masks.py`, with `PGW_TRIP_DECRYPT_AHEAD`, which the library sets): in
  incremental mode (8-slot parts) the leases of a part were created before the next part existed, so the first lease
  of every linear in every part decrypted its chunks inline. Nothing handed out; zeroed unless a lease takes it.
- **Top-p** (`pgw/engine/sampling.py`): numpy's argsort gives the same sorted value sequence; where the mask boundary
  splits equal values, or with NaN, `torch.sort` runs. 15.8 -> 5.5 ms per call on one core (71 ms for `torch.sort`
  alone with four busy intra-op threads here).

Not changed (measured, reasons): the in-process SDK path has no HTTP/JSON/SSE (chunk objects cost ~4 us each); the
noise statistics stay in Python (security code; 6.1 ms per token at 72B widths, now a model term); `multinomial`
(3.9 ms per sampled token on one core, ~10x with busy intra-op threads) is bound to torch's RNG stream; the trip split
over cores does not pay on this 4-vCPU box (STATUS 2026-11-23); lease creation keeps ~90 us of Python per lease
(17 per token at 0.5B, ~40 at 72B with 8-slot parts).

Model (`benchmarks/round_trip_model.py --sdk`): one new measured term, the inline noise statistics (5% of trips,
cost per sampled trip fitted to the numbers above): 1.4 / 4.1 / 6.1 ms per token at 7B / 32B / 72B. 72B at RTT 0.2 ms:
201.2 ms per token (was 195.1). The lane (0.13 ms per trip) and gateway terms are left as they were: the lane gains
here are on loopback next to a CPU provider, not over a NIC to a GPU.

Needs a GPU / same-data-center run: the lane's fixed cost per trip over a real NIC with the spin, batched reads and
held trips (spin hits, `recv` per trip at 72B reply sizes); the gateway's per-trip work at 72B widths on a gateway
with its own cores (trip split); sampling cost per token on the client's CPU with its thread settings.
Tests (one directory at a time; the box was shared and short of memory): tests/engine 1150 passed + 2 that failed
under xdist (an OOM-killed worker, a timing assertion) and pass alone; tests/wire 334; tests/native 343;
tests/service lane TLS + lane server 23; tests/gateway 277 + 3 that failed under xdist (an OOM-killed worker, a
dummy-pass count, a factory child start) and pass alone; tests/sdk 103 passed, 1 opt-in skip (serial);
tests/benchmarks and tests/observability pass; `cargo test -p pgw-fast --lib` 75 passed. Failing, in code this
change does not touch: tests/docs (`PGW_SPECULATION` in docs/sdk/configuration.md, from c9a0e82, is not a setting;
results/kaggle logs missing from the checkout) and two tests/maskstore child-process crash tests (the child's stdlib
`platform` import resolves to another module here).

## 2026-11-26: mask cost per slot on the client (factory CPU, stock bytes; scheme unchanged)

Goal: less client CPU and fewer stored bytes per mask slot (the scaling audit's #1 bottleneck), with single use,
mask secrecy, check soundness, the scheme and its parameters unchanged and outputs bit-identical. Breakdown,
method and the levers not taken: `docs/sdk/scaling.md` ("Mask cost per slot, by component"); tool:
`benchmarks/pks_slot_breakdown.py` (every per-slot and per-segment part at the real shapes of each model).

- **The earlier per-slot figure was incomplete.** 7.2 / 24.9 / 80 / 121 ms (0.5B / 7B / 32B / 72B) counted `r`, the
  upload, the reply keys and the decryption, without the masked LM head. Counting everything the factory does per
  slot (head, the segment's p31 check vectors `u W8`, the every-row 40-bit self-check, the int64 copy of `W r`,
  the writer), the factory cost **14.2 / 66 / 231 / 433 ms** per slot (scaled to the earlier audit's box). The
  largest term from 7B up was `u W8 mod p`: a float64 copy of every `W8` per segment, 5.5 ns per multiply-add
  (38 / 172 / 383 ms per slot at 7B / 32B / 72B).
- **Now 10.6 / 33.5 / 103 / 157 ms** per slot (-25 / -49 / -55 / -64 %); this box under load measured
  17.3 -> 13.0 (0.5B, both measured), 90 -> 46, 347 -> 155, 652 -> 236 ms (after: measured at the real shapes;
  before: the changed parts at measured rates). Factory cores at 16 streams: 15 / 27 / 33 / 38
  (were 20 / 53 / 73 / 104 with everything counted; the old table said 10 / 20 / 25 / 29).
  * `pgw_pks.vecmat_p31` / `vecmat_wrap32` (new `rust/pgw-pks/src/vecmat.rs`): `u W8 mod p` and `u W8 mod 2^32`
    as exact 32-bit lane sums on AVX-512 (0.17 ns per multiply-add, 30x); `ring_vecmat` uses it for up to 4
    vectors or where int8 GEMMs are not fast (the 14 private vectors stay on torch's int8 GEMM on AMX: as fast at
    one thread).
  * `pgw_pks.check_rows32`: the every-row self-check on 32-bit lanes, 16 rows per block, int64 or 32-bit `W r`
    (2.5x; same flags as `verify_rows`).
  * pks-v3 compact output (default; `PGW_PKS_COMPACT=0` off): the decryption's uint32 buffer goes to the
    self-check and the writer as the int32 compact form every segment writer already accepts; no int64 copy
    (2.9 / 9 / 13 ms per slot, 7B measured 23 under memory pressure), the writer packs from 4 bytes.
  * The private self-check vectors are cached as uint32 at K <= 32: at 72B the uint64 set (1.27 GB) overflowed
    the 1 GiB cache, so the last layers' `v W8` were redrawn for every segment (6.6 ms per slot); now 0.63 GB.
- **Bytes per slot: unchanged, at the floor** (1.82 / 6.17 / 19.2 / 28.1 MB): uniform 32-bit masks of every output
  feature; format overhead 0.05 % (`uW`) plus 16 bytes per ~1 MiB chunk. Storing seeds (needs a pass over all of
  `W8` per token at use time) or the provider's reply (4.5 bytes per element, +12.5 %, decryption at use time)
  loses on both counts. Written up, not implemented: a separate head stock for decode tokens only (the masked head
  uses one row per prefill; 22 / 6.5 / 2 / 1.4 % of the stock and 11 / 3 / 1 / 0.7 % of the factory CPU at
  500-token prompts; a store / gateway change), `ring`'s AES-GCM for the writer and the gateway's open (~4x on
  9.7 ms per slot at 7B, but no key-schedule wipe), sharing `u` across segments (a SPEC 7.3 change; no longer
  worth it). The sizing now also counts the gateway's mask open + retirement (1.25 / 3.7 / 11 / 16 ms per slot,
  helper threads) and the stock's write bandwidth (5 GB/s at 7B, 16 streams; twice with `shred_zero_fill`, which
  the guide now says to turn off on SSDs).
- Tests: `tests/factory/test_vecmat_native.py` (new: kernels equal the reference paths over shapes, extremes,
  threads, c up to 17, every `W r` form), `tests/hemask/test_pks.py` (new `test_compact_output_stores_the_same_masks`:
  compact, int64 and local generation store the same masks and check vectors, ring and p31, one- and multi-pass;
  the factory tests compare canonical values), Rust `cargo test -p pgw-pks vecmat`. tests/factory 463 passed /
  9 skipped; tests/hemask 439 passed / 20 skipped; tests/sdk 115 passed / 1 opt-in skip (the factory subset rerun after the last
  kernel-selection change: 96 passed).


## 2026-11-26: client library reliability (failure audit, fixes, soak)

Scope: `from pgw import OpenAI` (src/pgw/sdk) and what it drives, CPU only. Fault injection against a real
`pgw service` on loopback: a TCP proxy that resets / blackholes / refuses connections (`tests/sdk/chaos.py`), SIGKILL
and restart of the provider on the same ports and state (`restart_provider`), SIGKILL / SIGSTOP of the factory, a
tmpfs as the cache (disk full), corrupted cache files, 45-second client certificates. The guarantee for each failure
is in `docs/sdk/reliability.md`; troubleshooting, configuration and the changelog are updated.

Found and fixed (severity order; each has a regression test):
- **close() with a request in flight hung its caller forever** (the loop was stopped under it; "Task was destroyed
  but it is pending"). Requests are tracked and cancelled; callers get `APIConnectionError`; close is bounded.
- **Dropped or cancelled streams kept generating to max_tokens** (sync and async: the engine-side pump referenced
  the stream, so it was never collected; a cancelled async consumer, e.g. an SSE client disconnect, did not stop it).
- **HTTP fallback broken across event loops**: the runtime shared one httpx pool between the engine loop and the
  compute client's private sync loop ("bound to a different event loop"); a key revoked mid-stream surfaced as a
  retryable 503 `compute request failed: RuntimeError`. `PerLoopTransport` (gateway/egress.py): one pool per loop.
- **Mask starvation**: the factory refills only below low water, so a request needing more than the stock (but
  within the target) waited mask_wait_s and failed; one needing more than the target could never run (at 7B with
  defaults, ~1,300 slots: a prompt of ~760 tokens + 512). A demand file (factory `demand_file`) refills at once;
  beyond the target the request fails at once (`payload_too_large`, 400, naming mask_store_gb).
- **Orphaned fleet workers** after a SIGKILLed factory (OOM) kept running. `exit_with_parent` (PDEATHSIG + parent
  watch + hard exit) in the factory child and fleet workers; a hung child (no heartbeat for 300 s) is restarted;
  SIGKILL exits are reported as "killed by SIGKILL (often the out-of-memory killer ...)".
- **An expired mTLS client certificate was a permanent outage** after any reconnect (30-day certificates, no renewal
  in the library). Renewed at 2/3 of the lifetime (live contexts reloaded; other processes pick it up); an expired
  one (idle cache) enrolls again; clock skew is logged as such.
- **Refused credentials never recovered** (provider lost its token store, truncated service token): the next request
  reconnects and enrolls again (at most once a minute); a revoked key still fails clearly.
- **A damaged cache was permanent**: half-written session files, a damaged or truncated cached gateway pack. Session
  TLS material is checked when loaded (the CA against the key's pin); a damaged cached pack is deleted and
  provisioned again once (same pin check). Mask-store damage (database, KEK, rollback counter) stays fatal, never
  repaired automatically; the error names what to delete.
- **Misleading errors**: ENOSPC reported as "connection failed (OSError)"; "one-time mask setup still running" when
  the factory had stopped on its disk reserve or was failing. Now disk-full messages with the path, and the factory's
  disk limit or last error in the request's error; `PGW_FACTORY_MIN_FREE_BYTES` / `PGW_MASKSTORE_MIN_FREE_BYTES`
  reach the factory.
- Flaky tests: the sdk e2e modules depended on test order within a module (xdist spreads them): self-sufficient now;
  `test_lane_coalesces_tenants_without_mixing` raced the lane's own 0.5 s metrics drain (2 of 4 failed under load,
  6 of 6 pass now); `test_head_item_on_the_wire_and_strict_speculation` read `svc.received` twice while a dummy pass
  landed (1 in 3 under load); two maskstore crash tests imported tests/platform as the stdlib `platform`.

Verified, no change needed: a connection reset mid-answer is recovered by sending the same masked trip again (same
bytes; answer bit-identical); a provider SIGKILL mid-answer fails at once with a retryable error and the same client
serves again after the restart; a blackholed path is bounded by the request timeout; a damaged mask segment is never
used (AEAD); factory crash back-off and the "factory failing" fail-fast; integrity failures stay fatal, never retried.

Soak (`benchmarks/sdk_soak.py`, 31 min, tiny model, pks-v3, every connection cut every 5 min): 5,170 answers + 517
dropped + 517 cancelled streams + 517 AsyncOpenAI open/close, 0 errors, 0 mismatches; client RSS 422 MB at 10 min,
420 MB at 31 min (the drift is capped soundness caches), FDs 26-48 without trend (15 after close), threads 45-47;
factory RSS 387-490 MB, FDs 25-26, threads 5-7, no restart; close 0.44 s, no child left. An earlier 33-minute run:
5,160 answers, 0 errors, client RSS 405 -> 438 MB.

Tests: tests/sdk 138 passed + 1 skip (xdist -n 3; new: test_reliability.py 14 end-to-end, test_reliability_unit.py
14, test_session.py +7); gateway + wire + factory + enroll + maskstore 1264 passed (the 2 maskstore failures then
fixed). Open: a stream interrupted after its first token is not resumed (the application retries); a reconnect after
refused credentials stops requests still running on that runtime; a damaged factory pack is refused by the factory
(the error says so) but not downloaded again automatically; the mask store's state database grows ~2 KB per request
until its 30-day audit retention; hang detection is per process (a stalled scheduler that still beats is not
detected); other lane tests read drain_metrics() once and share the drain race in principle; 0.5B not soaked (disk);
no real-NIC packet-loss test; macOS has the parent watch but no PDEATHSIG.


## 2026-11-26 — pks-v3 provider key switch: fused Triton NTTs (`PGW_SERVICE_PKS_GPU_NTT`)

Problem (section above): with the hint on the device, the torch key switch was ~56 of 92 s per 1024-slot cycle at
72B on an H100 (estimate). It runs 16 + G negacyclic NTTs of 8192 points per 32 output rows and prime as ~90 eager
int64 operators each.

**Kernels** (`pgw.service.pks_ntt`, Triton; one program per polynomial, all 13 butterfly stages on chip):
1. `forward` (one launch per prime): gathers the centred operand `A~[b, l, 32 u + j] = hs[32 b + j, 256 l + u]`
   straight from the switched hint rows, reduces it mod p, runs the Cooley-Tukey stages of `pks.Ntt.fwd`
   (bit-reversed output) and stores uint32.
2. `mac_inverse` (one launch per prime): per (reply ciphertext, group) the 16 key products, their sum and the
   Gentleman-Sande stages of `pks.Ntt.inv`.
3. `tail` (one launch): body `round(y q1 / 2^56)`, ModDown, CRT and the 36-bit reply rounding, the same int64
   arithmetic as the torch tail, in one pass over the reply.

Arithmetic, all integer, exact for the four primes (2^23 < p < 2^30):
- twiddle products: Shoup (`w' = floor(w 2^32 / p)`, one `umulhi`, result in `[0, 2p)`, one conditional subtract);
- key products: Montgomery (R = 2^32); the summed factor 2^-32 and 1/8192 are undone by one Shoup scale
  `2^32 / 8192 mod p`;
- input reduction of `hs < 2^48`: 24-bit halves (both below 2p) and a Shoup product by `2^24 mod p`.

Every value is a canonical residue, so the result is the same linear map mod p as `pks.Ntt`: bit-identical. The
scheme, the wire and the replies are unchanged, and the provider sees the same data as before.

**Selection** (`pgw.service.pks_gpu.ntt_mode` / `ntt_impl`, `PGW_SERVICE_PKS_GPU_NTT`):
- `auto` (default): fused on a CUDA device only after `fused_self_test` matched the torch path bit for bit there.
  The self-test uses 96 rows x 2 groups plus 32 rows x 1 group, with edge values: hint rows at 0, q1 - 1 and the
  centring boundary, y at 0 and 2^56 - 1, keys at 0 and p - 1. Any mismatch or error, at the self-test or later,
  falls back to torch for that device for good (same bits). Other devices use torch.
- `torch` / `fused` force one; under `fused`, errors propagate.
- The existing self-tests (`self_test`, `item_self_test`) also run through the chosen path.

**Exactness evidence**
- CPU, Triton interpreter (`tests/service/test_pks_ntt.py`, 21 tests, ~4 min here):
  - forward and MAC + inverse equal `pks.Ntt` on every prime. Inputs are random plus edges: 0, q1 - 1, the
    centring boundary, 2^24 +- 1, whole ciphertexts at the boundary and at q1 - 1, keys all p - 1, alternating zeros.
  - the unit vector `X^k` in each of the first 64 slots, and -1 = q1 - 1 the same way; the round trip inv(fwd(x)) = x.
  - fused key switch = torch key switch = CPU key switch (`pks_native.keyswitch`) for 1 and 2 groups.
  - `pks_gpu.item` with fused NTTs = the host path.
  - the self-test passes and detects a one-bit error.
  - auto / torch / fused selection and the permanent fallback.
  - all three kernels compile to sm_75 and sm_90 cubins without a GPU (`pks_ntt.compile_for`).
- Kaggle T4 (`pgw-pks-ntt-t4`, torch 2.11 + cu130, Triton 3.6; one kernel, 36 min; raw files in
  `results/kaggle/pks-ntt-t4`):
  - The bench's exactness section passed in full:
    - forward and MAC + inverse equal `pks.Ntt` on all 4 primes;
    - fused key switch = torch = CPU on 96 rows x 2 groups with edges;
    - `fused_self_test` passed;
    - 2048 rows x 4 groups: fused = torch.
  - pytest on CUDA: 17 of 19 passed, including `pks_gpu.item` with fused NTTs = host path.
  - The 2 failures are fixed:
    - **One group did not compile.** Triton turns an integer argument equal to 1 into a compile-time constant, and
      the tail kernel called `G.to(...)` on it. The interpreter never runs that front end. The kernels now read
      grid sizes instead of such arguments, and `Tables` refuses constants equal to 1. `auto` would have caught it:
      the first one-group item would have fallen back to torch for good.
    - **A test keyed `cuda` instead of `cuda:0`.**

**T4 speed** (measured, 8 warps; 4 and 16 warps were slower):

| | torch | fused |
|---|---|---|
| forward NTT (us per transform) | 29.8 | 1.07 (includes the gather) |
| inverse NTT (us per transform) | 31.2 (inverse only) | 2.53 (includes the 16 key products) |
| key switch, 4 groups (us per output row) | not measured (see below) | 3.59 |

- The torch key switch was not timed end to end on the T4: that section ran one group first and hit the compile
  error. Its NTTs alone come to >= 75 us per row (4 x (16 x 29.8 + 4 x 31.2) / 32), so fused is >= 20x faster.
- Key-switch seconds per 1024-column item at the largest linears (fused = m x 3.59 us; torch >= m x 75 us):

| model | linear | fused (s) | torch (s) |
|---|---|---|---|
| 7B | gate_up | 0.14 | >= 2.8 |
| 7B | down | 0.013 | >= 0.27 |
| 7B | head | 0.55 | >= 11.4 |
| 32B | gate_up | 0.20 | >= 4.2 |
| 72B | gate_up | 0.21 | >= 4.4 |
| 72B | down | 0.029 | >= 0.62 |

- Whole cycle: 5.5 / 17 / 25 s (7B / 32B / 72B) against >= 116 / 361 / 527 s.
- Whole items (torch vs fused through `PKSProvider`) were not measured. With the one-group bug, the device path's
  self-test failed, and the bench fell back to the host path and ran out of time. The RunPod session measures them.
- The kernels are bound by the SMs, not HBM: the T4 moved ~0.5 MB per row at 3.59 us, ~40 % of its bandwidth.

**H100, ESTIMATES** (`python -m pgw.service.memplan --pks-throughput`, now `--pks-ntt fused` by default):
- The fused model is calibrated on the T4 numbers and scaled by SM count x clock (3.8x): 0.95 us per output row
  for 4 groups. That is ~8x less than the torch model's 7.9 us. Memory traffic would allow 0.22 us.
- Device-hint path, slots/s (before -> after):
  - 7B: 53 -> 119;
  - 32B: 16.6 -> 37;
  - 72B: 11.1 -> 24 (92 -> 43 s per cycle).
- Key switch per cycle at 72B: 56 -> 6.6 s.
- The previous section's ~150 slots/s at 72B assumed only the GEMMs would remain. They do not. What is left at 72B:

| stage | s per cycle |
|---|---|
| hint switch + recombination (eager int64 torch, 628 B per hint value) | 18.6 |
| host ChaCha20 of the ring rows | 8.9 |
| key switch | 6.6 |
| uploads | 4.4 |
| reply copy back | 2.9 |
| GEMMs | 1.3 |

- So NTTs no longer bound mask throughput. Next levers, in order:
  1. a fused hint switch + plane recombination kernel (~36 B instead of 628 B per value);
  2. the ring-row stream off the GIL (or on the device);
  3. in the forward kernel, a coalesced gather (it now reads 32 hint rows per warp access) and all 4 primes per
     program (one hint read instead of 4);
  4. pinned uploads.

**Wiring:**
- `memplan`: `MemConfig.pks_ntt` and `--pks-ntt torch|fused`; `pks_item_seconds(..., ntt=)`;
  `fused_keyswitch_row_seconds`; `FUSED_KS_ROW_BYTES` / `FUSED_KS_ROW_OPS`, calibrated on the T4.
- RunPod `maskgen` phase (now ~25 min): device items at every work budget under `auto`, which records the NTT in
  use, plus one torch-NTT run at 512 MB. Replies must be byte-identical, next to the torch and fused estimates.
- Kaggle job `pgw-pks-ntt-t4` (`benchmarks/kaggle/build.py`, 0.5 MB bundle) and `benchmarks/pks_ntt_bench.py`.
  Rerunning it, ~25 min, would give the missing torch end-to-end and item numbers; it was not rerun, per the
  one-kernel budget.

Tests:
- `tests/service` full: 510 passed, 77 skipped (CUDA), before the T4 fixes.
- After the fixes: `tests/service/test_pks_ntt.py` 21 passed, `tests/service/test_pks_device_item.py` 20 passed, and
  `tests/benchmarks/test_provider_perf_tools.py` 2 passed.
- New setting: `PGW_SERVICE_PKS_GPU_NTT` (default `auto`); `PGW_SERVICE_PKS_NTT_WARPS` (default 8) is for tuning.
- Cloud spend: one Kaggle T4 kernel (36 min).

## 2026-11-26: weight-free client factory (`factory_weights="stream"`, opt-in; option C, scheme unchanged)

Goal: the SDK client's encrypted factory keeps no int8 weights `W8` at rest. Each refill pass streams `W8` from an
untrusted source, checks it against the pinned fingerprint, pre-draws the check material of the next segments and
drops the weights. Option C of `research/reviews/weight-free-client-checks.md`. Default (`"pack"`) unchanged.

- **Security note first** (`research/reviews/weight-free-stock-security.md`): verdict **no weakening** beyond the
  earlier analysis. Each pre-drawn item comes from the same CSPRNG and the same SPEC 7.3 derivation, with the same
  kernels, as an on-demand draw, and is used under the same single-use rules (one segment set per segment, one
  private set from a worker start until its first failed or aborted segment). Two stated changes: more secret
  material at rest (up to `K` future segments, readable only with the KEK, which already decrypts every stored mask)
  and private `v W8` on disk for the first time (an integrity exposure under a condition that already breaks privacy).
- **`pgw.factory.checkstock.CheckStock`**: one item file per segment set (seed + `uW` per linear) or private set (`v`
  seed + `v W8`), AES-256-GCM under a fresh DEK wrapped by the mask store's KEK provider, the header as AAD. The
  index lists each item's header digest and is HMAC'd under the KEK-derived key (domain `pgw/v1/check-stock-index`).
  A generation witness outside the directory catches a rolled-back index. A take records consumption durably
  (index fsync) before the item is read, then deletes it; orphans of a crash or of a dead pass are deleted at open.
  Low-water refill runs in a background thread, one pass at a time across processes (flock), with back-off after a
  failed pass. An empty stock waits up to `check_stock_wait_s`, then fails closed with `CheckStockEmpty`.
- **Factory**: `MaskGenerator(weights="stream")` streams shape-only placeholders. Each segment's seed and `uW` come
  from the stock (the private set is taken first, so a missing one never burns a segment set). `_private_check` /
  `_wipe_private` take and burn private sets exactly where on-demand draws happen. Settings (`PGW_FACTORY_*`):
  `factory_weights`, `weights_source` (`provider` | `hf` | `dir`), `pack_fingerprint`, `check_stock_*`. In stream mode
  the pack is the weight-free `GatewayPack` (verified in full), the pks-v3 hint must exist (built at provisioning),
  and fleet workers share one stock. `pgw factory check-stock [--discard]`.
- **Sources** (`pgw.provision.client.weight_source`): the provider's provisioning endpoint (per-chunk and whole-object
  SHA-256 against the pinned manifest), a full pack directory (each linear `check_hash=True`), or the public
  checkpoint (packed per pass, must reproduce the pin, deleted after). The masked head's `W8` is derived from the
  gateway pack's hash-checked plain head, as before. A pass commits nothing unless every tensor passed.
- **SDK**: `factory_weights` / `factory_weights_source` / `check_stock_segments` (`PGW_FACTORY_WEIGHTS`,
  `PGW_FACTORY_WEIGHTS_SOURCE`, `PGW_CHECK_STOCK_SEGMENTS`). Provisioning keeps no factory pack and builds the hint from
  the streamed weights (it deletes a factory pack left by `"pack"`). The plan sets `max(8, 2 x workers + 2)` private
  sets. `pgw sdk status` shows the stock, its passes and the last pass error. `pgw sdk prepare --factory-weights
  stream`. A request whose model has an empty stock and 3 failed passes fails at once with that reason.
  Docs: `docs/sdk/configuration.md` (option, env, "Keeping no weights"), `docs/sdk/scaling.md` (disk/network table).
- **Measured, Qwen2.5-0.5B-Instruct** (`benchmarks/stock_pass_bench.py`: pinned pack `d773a502fb97`, the real
  provisioning endpoint on loopback, SDK settings: masked head, p31, one check vector, 14 private vectors): one pass
  of 64 segment sets + 8 private sets received **357.9 MB** (357.8 MB of layer `W8`, protocol +0.03 %; the 136 MB head
  `W8` is derived locally). It wrote a **128.6 MB** stock (0.73 MB per segment set, 10.2 MB per private set) in place of
  the 359 MB of `W8`, in 14-22 s wall on this loaded 4-vCPU box. Extrapolated with the same formulas (pks-v3 traffic:
  `hemask_pks_model.py` formulas with the masked head; default 48 segments per pass = 64 minus the low water of 16):

  | | 0.5B (measured) | 7B | 32B | 72B |
  |---|---|---|---|---|
  | int8 `W8` no longer on disk | 0.36 GB | 6.53 GB | 31.2 GB | 70.2 GB |
  | stock on disk (64 segment + 8 private sets) | 0.13 GB | 0.59 GB | 1.94 GB | 3.06 GB |
  | `W8` streamed per pass | 0.36 GB | 6.53 GB | 31.2 GB | 70.2 GB |
  | network vs pks-v3 per 1024-slot segment, default | +0.18 % | +0.98 % | +1.49 % | +2.23 % |
  | same at 64 segments per pass (the review's R = 64) | +0.14 % | +0.73 % | +1.12 % | +1.67 % |

  The first setup streams `W8` twice: once for the hint, once for the first stock.
- **Tests**:
  - `tests/factory/test_check_stock.py` (21). Segments are bit-identical to pack mode and to local generation, ring
    and p31, several seeds stacked. Private sets equal `_private_check`'s draw. A wrong provider product (shift 2^31
    / 2^35) is caught with no `W8` on the client, and both sets are burned. A crash after a take (child process
    killed) and a pass killed midway lead to no reuse and no partial items. Tampered, swapped or header-edited items
    are rejected and burned. A tampered index is discarded; a rolled-back index or a missing witness are refused; a
    foreign KEK cannot use the stock. A bad stream commits nothing. Files are 0600 with no plaintext seed. Further
    tests: low-water background refill, fail-closed exhaustion with the pass error, waiting for a running pass,
    stop, concurrent takers, and the CLI.
  - `tests/provision/test_weight_source.py` (6): the provider, directory and public-checkpoint sources, a tampered
    weight chunk, the derived head, and a real pass over the endpoint whose `uW` equals an on-demand computation.
  - `tests/sdk/test_stream_weights.py` (4): end to end on a real provider (tiny model, pks-v3). Answers are
    bit-identical to the reference. No factory pack, and no file in the cache or state holds a `W8` byte run. The
    stock shows in status. A tampered provider product is caught online, so the p31 vectors come from the stock.
    Also covered: prepare in stream mode deleting an old factory pack, option validation, and the factory env.
  - Runs, one directory at a time, absolute `PYTHONPATH`: tests/factory 484 passed / 9 skipped (CUDA); tests/provision
    52 / 1; tests/sdk file by file 124 passed / 1 opt-in skip; tests/cli 209 / 3 (helm); tests/docs: only the earlier failures (test_numbers logs, README not
    linking sdk/scaling.md, `PGW_MASK_ADMISSION` in the changelog); `PGW_FACTORY_WEIGHTS*` and
    `PGW_CHECK_STOCK_SEGMENTS` added to the documented SDK env list.
- Not done: the first stock is not drawn during provisioning's own `W8` stream, because the check format depends on
  the runtime chosen after provisioning, so setup streams `W8` twice. mlwe in stream mode is not supported (its hint
  is built by the factory). No monotonic counter for the stock beyond the store's own level (security note
  section 2).

## 2026-11-26: mask store AES-256-GCM on AWS-LC (writer, gateway open; format unchanged)

Goal: a faster AES-GCM in the factory's segment writer and the gateway's chunk open, with every security property
kept (key schedule wiped on drop, same ciphertext bytes, same nonces, constant time). Write-up:
`docs/sdk/scaling.md` ("The store's AES-256-GCM"); tool: `benchmarks/maskstore_aead_bench.py`.

- **Root cause: RustCrypto's design, not its configuration.** `aes` 0.9 already ran its VAES AVX-512 backend and
  `polyval` 0.7 its PCLMULQDQ backend (run-time `cpufeatures`; no RUSTFLAGS or `.cargo/config`; `-C target-cpu=native`
  is no faster). `aes-gcm` 0.11 makes two passes (generic `ctr` at 5.5 GB/s, then GHASH as byte-reversed POLYVAL,
  four 128-bit carry-less multiplies at a time, 6 GB/s): 2.8 GB/s on an idle core (the 1.2 GB/s quoted was under
  load). Chunks are already ~1 MiB (64 KiB: same rate), so per-call overhead plays no part.
- **Options, one core, 1 MiB chunks (seal / open GB/s):** RustCrypto 2.8 / 2.5; **AWS-LC via aws-lc-rs 1.19:
  13.8 / 12.9**; OpenSSL 3.0 (`openssl` crate) 12.3 seal (libcrypto would have to be vendored into the wheel);
  graviola 0.4 9.8 / 8.4 (zeroizes, but panics without ADX / AVX2); `ring` excluded (no key-schedule wipe).
- **Chosen: aws-lc-rs** (`LessSafeKey`, non-FIPS, statically linked; a Unix-only dependency of pgw-fast, like the
  writer). Wipe on drop, from the sources (aws-lc-sys 0.47.0 / aws-lc-rs 1.19.0): the round keys and GHASH table
  live in the `EVP_AEAD_CTX` (`OPENSSL_malloc`, `aead/aead_ctx.rs`), whose `LcPtr` drop calls `EVP_AEAD_CTX_free`
  -> `OPENSSL_free`, which `OPENSSL_cleanse`s the whole allocation (`crypto/mem.c`; no memory hooks installed).
  Tests read the freed memory back: `aws_lc_free_cleanses_the_key_schedule` (the free path; a negative control run
  by hand, a plain `free`, leaves the key and fails it) and `dropped_key_leaves_no_key_schedule` (the writer's own
  `Sealer` context after use and drop, recovered by its pointer: all zero; with a built-in control showing that an
  uncleansed context is found). AWS-LC does not clear its per-call stack copy of the GHASH table and `E_K(J0)`, so
  every writer thread and each `chunk_open_into` now zero 16 KiB of stack below their frame after the cipher calls
  (`scrub_stack`). Tag compare: `CRYPTO_memcmp`; a failed open zeroes the output; AES and GHASH use only the
  AES-NI / VAES and carry-less multiply instructions.
- **Also:** the writer's range check is skipped where it cannot fail (uint32 / int32 `W r` at K = 32, the pks-v3
  compact form; every other type and K is still checked in full; test `range_shortcut_is_exact`). It was a separate
  pass over the array, about 1 ms per slot at 7B.
- **Per slot, before -> after** (ms, best of two interleaved runs, shared box under load 3-8), 0.5B / 7B / 32B / 72B:
  writer 1.79 -> 1.19 / 6.14 -> 4.00 / 18.8 -> 11.8 / 29.4 -> 16.7; gateway open + unpack 1.05 -> 0.55 /
  3.52 -> 1.88 / 10.8 -> 5.92 / 16.4 -> 8.89; open + retirement 1.69 -> 1.04 / 5.24 -> 3.45 / 15.4 -> 10.8 /
  24.4 -> 15.6. The writer is now bound by the page-cache `pwrite` (3-5 GB/s) and the packing copy, not the cipher.
- **Not changed, documented:** one fsync per segment at publish (plus the directory's): almost no factory CPU
  (kernel writeback), twice the wall time, and needed for durability before the cursor moves. `O_DIRECT` would need
  aligned chunks (a format change). `shred_zero_fill` stays on by default: 70 % of the retirement CPU (72B: 6.7 vs
  2.0 ms per slot) and twice the disk writes, useful only on disks that overwrite in place; the guide's advice to
  turn it off on SSDs stays. The default writer thread count is unchanged.
- **Compatibility:** `tests/native/aead_golden_rustcrypto.bin` was written by the RustCrypto build. The new build
  writes the same bytes and opens it (`tests/native/test_aead_compat.py`, which passes unchanged against both
  builds); `cryptography` opens it and the new build's chunks, and the new reader opens `cryptography`'s. Rust
  tests seal and open across RustCrypto and AWS-LC both ways over 17 lengths (0 to 1 MiB + 5), plus test case 14
  of the GCM spec. Full store, both directions, checked by hand with the old `.so`: a store written by either build
  reads back identically with the other (native open, `cryptography` open, reserve / take: same digests). Tamper
  tests (bit flips in ciphertext and tag, another chunk index / nonce prefix / header digest / key, truncation, a
  short read) fail closed with the output zeroed, in Rust and Python.
- **Build / release:** aws-lc-sys builds with the `cc` crate (no cmake, Go, NASM or bindgen on Linux / macOS); the
  extension grows from 6.3 to 7.1 MB and links no new shared library. `cargo zigbuild --target
  x86_64-unknown-linux-gnu.2.28` (the release check's manylinux_2_28 path) builds pgw-fast with AWS-LC's assembly
  (4.4 / 7.2 GB/s through Python, as the host build), highest symbol version GLIBC_2.28, needing only libc,
  libpthread and libdl, and the compat and writer tests pass on that `.so`. The full `scripts/release_check.sh`
  was not run: it builds all three extensions from the sdist plus a venv with torch, more than the 2.6-4 GB of
  disk that was free during this work. THIRD_PARTY_NOTICES.txt is regenerated (aws-lc-rs and aws-lc-sys in, the
  RustCrypto AEAD crates out) and `tests/packaging` checks for the new crates.
- Tests: `cargo test -p pgw-fast --lib` 83 passed (11 in writer); tests/native 357 passed; tests/engine/test_trip_native.py
  32 passed; tests/packaging 11 passed; tests/maskstore 245 passed, 2 failed (`test_ringbuf::test_crash_leaves_disk_untouched`,
  `test_ringshm::test_child_process_fills_ring_and_crash_is_reaped`: their child processes fail to start in this
  environment, and they fail the same way with the build from before this change).


## 2026-11-26: concurrent answers on the native trip lane (lane format 2, multi calls; on in the SDK)

Scope: SDK bottleneck 3 of "client library at many concurrent streams" (docs/sdk/scaling.md, item 4). The TCP+TLS
lane carried one item per call, so every trip with two answers (and every prompt item above 8 rows) went to the
WebSocket stream channel. Choice by measurement (`benchmarks/lane_multi_bench.py`, real SDK provider, one step = the
rows of K answers on 0.5B `gate_up`, client CPU / provider CPU / wall per step, idle box):

| K | multi call (a) | K parallel single calls on K connections (b) | K serial calls | stream channel |
|---|---|---|---|---|
| 4 | 0.13 / 2.9 / 4.3 ms | 0.46 / 3.7 / 5.2 ms | 0.40 / 8.6 / 12.3 ms | 0.58 / 3.7 / 5.4 ms |
| 16 | 0.32 / 6.0 / 7.1 ms | 1.48 / 10.6 / 10.0 ms | 1.53 / 32.8 / 47.6 ms | 1.10 / 7.2 / 7.8 ms |

With two busy Python threads (GIL contention) K = 4: multi 17.5 ms wall, parallel 51.6, stream 33.7; K = 16: 45,
66, 690. (b) pays a header, TLS records, a GIL hand-off and an executor dispatch per answer on both sides; (a) pays
them once and is what the stream channel already sends. Chosen: (a).

- **Wire** (`rust/pgw-lane/src/proto.rs`, versioned): `multi` frame = header version 2, kind 4, `linear` = item
  count (<= 64), `rows` = total (<= 256); body = item table (`u32 linear, u16 rows, u16 0`) + the items' `x_hat` in
  order; OK reply = the items' `y_hat` in order (reply header unchanged, version 1). Version-1 frames unchanged; a
  multi frame claiming v1 or a call claiming v2 is a bad header. Negotiation: tickets add `"versions": [1, 2]`,
  `max_items`, `max_multi_rows` next to `"version": 1` (older clients read them as before); a client sends multi
  frames only when offered; a format-1 service (or `PGW_SERVICE_LANE_MULTI=0`) gets one item per call and answers
  a multi frame with `version` (fatal), as before. Fallback to the stream channel with the same request on any
  refusal, error, Python channel or shm connection, unchanged.
- **Provider** (`server.rs`): item table checked against the ticket's tables and the exact body length before any
  body is read; admitted once for the total rows under the connection's tenant (one tenant per connection: no
  mixing); consecutive same-linear rows run as one executor call (at most `max_rows` with CUDA graphs, any count on
  the kernel path as the stream batcher does; a `Fallback` from a longer call re-splits it); one status per frame
  (all or nothing); metering: one trip per frame, rows per call (as the stream channel). The opt-in cross-tenant
  coalescer is untouched.
- **Client** (`pgw-fast` `TripChannel.trip_multi`, `pgw.wire.lane` `LaneConfig.multi` / `PGW_LANE_MULTI`: off in
  the gateway, on in the SDK's `NATIVE_ENV`): several-item requests, and one item above `max_rows` (a prompt), go in
  one multi call; the response is the stream path's, item for item.
- **Bug fixed on the way**: `ServiceLaneOpener.lane_open_sync` ran on the wire client's private loop, but the
  gateway's egress transport is bound to the SDK loop: the first extra lane connection failed (`RuntimeError`) and
  turned the lane off for 30 s. Seen as 0 of 1452 lane trips for 0.5B at N=1 before; now it runs on the loop the
  first `lane_open` used.

Privacy: the provider sees per frame exactly the stream channel's request for the same trip (request id, ring,
deadline, the items' linear, rows and `x_hat` in order) under the same tenant; nothing is added, and the frame
boundaries are the trip boundaries the stream channel shows. Masking, noise statistics, Freivalds / range / p31
checks and held sampled trips run per answer in the backend as before (the lane only moves bytes); a wrong result
inside a multi call fails that answer's checks (test). Strict speculation: rows per pass, passes, provider-metered
trips and rows of a request are unchanged on/off (test); `spec_pass_gap_ms` pacing is above the channel; channel
choice depends on the number of items, which the provider sees anyway, not on real vs dummy passes.

Measured (4-vCPU box shared with other agents, CPU provider on the same box; every answer bit-identical to the same
request alone; before = branch head, after = this change; tiny 64 tokens, 2 interleaved pairs; 0.5B 12 tokens):

| model | N | trips/s before / after | tokens/s | lane trips before / after | client CPU ms/token | provider CPU ms/token |
|---|---|---|---|---|---|---|
| tiny | 1 | 420-439 / 424-429 | 45-47 / 45-46 | 568 / 600 of 600 | 3.6-4.5 / 3.4-4.4 | 6.4-6.9 / 6.7 |
| tiny | 4 | 245-263 / 294-297 | 99-107 / 119-120 | 1 / 633 of 633 | 4.6-5.1 / 4.1 | 2.7-2.9 / 2.1 |
| tiny | 16 | 124-127 / 127-134 | 190-195 / 195-206 | 1 / 666 of 666 | 3.4-3.6 / 3.4-3.7 | 1.1-1.2 / 0.85 |
| 0.5B | 1 | 230 / 272 | 1.9 / 2.3 | 0 / 1452 of 1452 | 165 / 128 | 331 / 303 |
| 0.5B | 4 | 156 / 179 (144 multi off) | 4.1 / 4.7 | 0 / 1837 | 118 / 111 | 163 / 150 |
| 0.5B | 16 | 56 / 61 (51 multi off) | 5.9 / 6.3 | 3 / 1837 | 90 / 89 | 144 / 141 |

0.5B TTFT 1.9 / 1.7 s (N=1), 5.5 / 4.6 s (N=4), 17.0 / 14.4 s (N=16): prompts now ride the lane. At 0.5B the CPU
provider saturates, at tiny-16 the client's plaintext thread does (item 3 of the scaling list): the lane is
no longer the limit. `sdk_trip_profile.py` (0.5B, N=1, decode): same tokens and logprobs (same hash), 97 trips per
token, 302 vs 305 ms per token, lane I/O 84 vs 80 us per trip: the single-stream path is unchanged.
Earlier runs in a busier window gave tiny N=1 16-27 tok/s for the new build and 36-45 for the old; three later
interleaved pairs were equal (44-47 both): box noise, not the change.

Tests: `cargo test -p pgw-lane --lib` 46 (new: frame layout and versions, item table, multi == single calls,
same-linear merge, merge_rows and the Fallback re-split, an item cut at row boundaries, tamper: lengths, unknown
linear, too many items, bad table, reserved bits; all or nothing; admitted once; multi off = version),
`cargo test -p pgw-fast --lib` 75; tests/service/test_lane_multi.py 8 (TLS and UDS multi call == stream path ==
exact product, format-1 service, Python reference encoder `encode_multi`, argument checks, LaneComputeClient
eligibility on/off/format 1/Python channel, ticket parsing); tests/sdk/test_lane_multi.py 4 (concurrent answers
bit-identical alone, reversed and staggered starts, streamed, and with multi off; strict speculation unchanged; a
format-1 provider gets no multi frame; a wrong result in a multi call is caught). tests/wire 334, tests/service 502
(+77 CUDA skips), tests/gateway 283 + 1 OOM-killed xdist worker that passes alone, tests/sdk 124 + 1 opt-in skip
(serial). tests/docs: 20 failures not from this change (results logs missing, README index of sdk/scaling.md,
`PGW_MASK_ADMISSION` in sdk/changelog.md). Needs a GPU run: multi calls against CUDA graphs (calls of at most 8 rows
per same-linear run) and the persistent worker, and the lane's per-trip cost over a real NIC at 72B widths.

## 2026-11-26: SDK end to end on Qwen2.5-7B, GPU provider (Kaggle T4): all checks pass

What ran:
- The provider service on one T4 (`cuda:0`): pack derived from the pinned checkpoint, accounts, provisioning, TCP+TLS lane.
- `from pgw import OpenAI` in the same VM over loopback, with the library's defaults (`pks-v3`, `hint="local"`, masked head on) and an empty cache.
- Harness and job: `benchmarks/sdk_e2e_gpu.py`, `pgw-sdk-e2e-7b`.
- Results: `results/kaggle/sdk-7b-validation/run2_7b/` (`summary.md` and the raw logs of three attempts).

**The T4 numbers are a correctness check, not a measure of speed.** The T4 has no int8 tensor-core GEMV, and 4 vCPUs share the client, its hint build, the mask factory and the provider.

Attempt C (code `e73fec8`) passed everything:
- the provider pack matches the baked pin;
- the provider loads (6.54 GB on the T4, memory plan fits, 47 s);
- enroll and provisioning work;
- the local hint is built;
- bit-identical to the gateway path (text and log-probs, the `test_e2e_product.py` comparison) for greedy, sampled-stream, two 64-token streams and 3 conversation turns;
- no integrity failures (22,352 Freivalds checks);
- the factory is not halted;
- no content in the provider's records;
- the revoked key is refused by both the connected client and a fresh client;
- the GPU unit tests pass (21).

Timings on the T4:

| phase | time |
|---|---|
| provisioning | 182 s (weight stream 6.5 GB in 116 s; artifacts 2.2 GB in 26 s) |
| local pks-v3 hint build (4 vCPUs) | 17.3 min |
| first 64-slot mask segment | 196 s (factory ~1.9 slots/s once segments grow) |
| first answer after `warmup()` | 1,246 s |
| TTFT | 3.9 s |
| steady decode | 8.1 tok/s (2.2 tok/s while a factory segment shares the T4) |

Fixed or found on the way:
- Run 1's start-up CUDA OOM is fixed by `bfee514`.
- Attempt A (`bfee514`): the provider died 66 min in. Host RAM was exhausted by the old pks-v3 host hint cache (`m x 4096` uint64 per seed and linear, 50.5 GB per seed at 7B; host RAM fell to 3.9 GB). `3e347e1` (hint recomputed per item on the device) fixes it; after that, provider RSS stayed <= 2.6 GB and GPU use <= 9.1 GB.
- Open (client reliability): after the provider died, the client's request waited out its whole timeout. The factory's `ConnectError` was not counted as a failure, and the request was not failed. *(Fixed since: "client mask factory stalls" below, `factory_stall_s`.)*
- Attempt B: on a T4 the factory is the bottleneck. A request beyond the stock is correctly refused with `mask_exhausted` after the default 120 s. The harness now waits for the next segment and sends the request again (recorded; options unchanged).
- Harness: provider-exit watchdog and wall budget (kernel under 3 h), per-process RSS, and `provider.log` kept even when the run is cut short.


## 2026-11-26 — pks-v3 device item, round 2: fused hint switch, device ring rows, wire forms on the device, pinned copies

Goal: raise pks-v3 mask throughput on the provider's device path past the fused NTTs (section above: an estimated
24 slots/s at 72B on an H100, 43 s per 1024-slot cycle). The scheme, the wire and the replies are unchanged; the
provider sees the same public data (`a_seed`, the ring rows, `b`, the keys) as before.

**What changed** (`pgw.service.pks_fused`, `pks_gpu`, `pks_ntt`, `pks_provider`):
1. **Fused recombination + hint switch** (`pks_fused.combine`, `PGW_SERVICE_PKS_GPU_SWITCH`). One Triton pass per
   GEMM column block: it reads the 7 int32 plane partials, recombines them mod 2^56 and applies the exact 14-bit-digit
   `round(h q1 / 2^56) mod q1`. Per value that is 28 B read and 8 B written, against ~628 B for the eager int64
   operators. There are no int64 temporaries, and nothing wraps before the final mask: `lo + (c mod 2^56) +
   ((hi mod 2^24) << 32)`. The same kernel, without the switch, recombines `y`.
2. **Ring rows on the device** (`pks_fused.ring_rows`, `PGW_SERVICE_PKS_GPU_RING`). A ChaCha20 kernel (RFC 8439
   block function, zero nonce, 32-bit counter) expands the stream key of `(A_LABEL, linear)`. The host derives that
   key with HKDF, as before, so only 32 bytes go up instead of `n/16 x 4096` uint64. A tensor-parallel item no longer
   expands the stream once per shard. The fallback is the host stream cut at ring-row boundaries (block-counter
   seeks) on a thread pool. OpenSSL's ChaCha20 runs without the GIL, so 4 threads run 4.6x faster than one here. It
   runs in the background while the item queues its uploads (`ring_rows_async`).
3. **4-prime forward NTT** (`pks_ntt.forward4`, `PGW_SERVICE_PKS_GPU_FWD4`). Each program reads the `(32, 256)` hint
   tile once, coalesced, transposes it on chip and runs the four primes' transforms. It is exact, but **slower on the
   T4**: 3.14 against 2.50 us per output row for 4 groups. The transforms are SM-bound, and four on-chip transposes
   cost more than the per-prime kernel's strided gather. So it is **off unless set** (`auto` or `fused`), and the
   RunPod session measures it on the H100.
4. **Pinned, overlapped copies** (`PGW_SERVICE_PKS_PINNED`, default on):
   - The uploads go through pinned staging buffers on a per-thread side stream: the keys as uint32, widened on the
     device, and `b` with its planes. The item's stream waits for them only before the first `y` GEMM, so the copies
     overlap the ring rows, the hint operand's planes and the first row block's hint GEMMs.
   - Each row block's reply goes into pinned buffers by a non-blocking copy. The host drains the previous block while
     the device computes the next.
5. **Wire forms on the device** (`PGW_SERVICE_PKS_GPU_WIRE`). The T4 items showed a stage the model had never
   counted: `PKSProvider.run_item` unpacked the 56-bit upload and packed the 36-bit reply with numpy on one host
   thread. That costs 8.4 and 20.5 ns per value here, about 37 s + 147 s per 1024-slot cycle at 72B (4.4e9 upload
   values, 7.2e9 reply values) on top of the device work. With the T4's rates, the model reproduces the run's whole
   items within 25 % only with this stage included. Both formats are plain bit fields (`pack_bits`):
   - The upload's wire bytes go up as they are, 7 B per value instead of 8, and the device unpacks them column block
     by column block (`32 | 16 << 32 | 8 << 48`).
   - The device packs each row block's reply into the 32-bit field and the 4-bit tail stream. It writes them into
     views of the reply's wire buffer (`PackedReply`), shard offsets included, so 4.5 B per value come back instead
     of 8 and the host only copies.
   - `PackedUpload` keeps the host unpack for the fallback and the cache path.

Every kernel follows the NTT's rule (`pks_gpu.FusedChoice`):
- Under `auto`, a CUDA device uses the kernel only after a start-up self-test matched the torch / host path bit for
  bit there.
- Any mismatch or error falls back to the plain path on that device for good, with the same bits.
- `fused` forces the kernel and lets errors propagate; `torch` disables it.
- The self-tests are `_switch_self_test`, `_ring_self_test`, `_fwd4_self_test` and `_wire_self_test`.
- `item_self_test` now also checks the wire forms end to end against `pack_bits` of the host path.
- `pks_gpu.fused_status(dev)` reports what each device uses.

**Exactness evidence**
- CPU, Triton interpreter (`tests/service/test_pks_fused.py`, 34 tests). Each of these equals its reference bit for bit:
  - **The fused combine:** equals `_combine_planes` (torch and numpy). The switch equals `_round_mul_shift` and
    `pks.switch_hint`. Inputs: partials at +-(2^31 - 1) and -2^31, constants at 0, -1, +-2^63 and 2^56, the rounding
    boundaries `(2k + 1) 2^55 / q1`, `2^56 - 1` (which wraps to 0), shapes that are not block multiples, 1 row, and
    strided output views.
  - **Device ring rows:** equal `pks.public_ring_rows` for 3 seeds and linears. The threaded host stream equals it
    for 1, 64 and 150 ring rows.
  - **`forward4`:** equals `forward` and `pks.Ntt` on edge rows. The key switch with it equals torch for 1 and 2
    groups.
  - **The wire forms:** the device pack and the host block pack equal `pack_bits`, and the device unpack equals
    `unpack_bits`, at the field boundaries.
  - **`pks_gpu.item`:** equals the host path (numpy GEMM, `pks.switch_hint`, CPU key switch) with every kernel
    forced, and with the wire forms in and out at a shard offset.
  - **`PKSProvider`:** with every kernel forced, it returns the cache path's bytes, and the customer decrypts them
    to `W8 r mod 2^32`.
  - **Selection and fallback:** each self-test detects a one-bit error. Selection, env parsing and the permanent
    fallback are tested.
  - **Compilation:** all new kernels compile for sm_75 and sm_90 within 64 KiB of shared memory per block
    (`forward4` uses 32 KiB).
- Kaggle T4 (`pgw-pks-ntt-t4` version 2: torch 2.11 + cu130, Triton 3.6; one kernel, 6.5 min; raw files in
  `results/kaggle/pks-ntt-t4-r2`):
  - **The bench's exactness sections passed in full.**
    - The forward and the MAC + inverse equal `pks.Ntt` on all 4 primes.
    - **The one-group path fixed after the first run now works on CUDA.** For 1 and 2 groups, with and without
      `forward4`, the fused key switch equals both torch and the CPU, including one group of one ciphertext.
    - 2048 rows x 4 groups: fused = fwd4 = torch.
    - The fused combine and switch equal torch, numpy and `pks.switch_hint` at 64x300, 2048x512 and 1x256, and at
      the rounding boundaries.
    - The device ring rows and the threaded host stream equal `public_ring_rows` up to 72B's 29568 inputs.
    - `forward4` equals `forward` and `pks.Ntt`.
    - Every self-test passed, and `auto` chose every kernel there.
  - **Whole items are byte-identical in all three configurations.** The bench ran the largest linears of 7B, 32B and
    72B through `ComputeKernel` + `PKSProvider` with every kernel plain, with the fused NTTs only (the previous
    state), and with every kernel under `auto`.
  - pytest on CUDA: 45 passed, 2 failed, 1 skipped (CPU only). Both failures were test bugs, now fixed:
    - a uint32 xor that CUDA does not implement;
    - a test keyed `cuda` instead of `cuda:0`.
  - The wire forms on the device were added after this run, from what it showed. They are **not GPU-measured**: they
    are tested on CPU (pure torch integer ops, no Triton), self-tested at start-up, and the RunPod session measures
    them.

**T4, measured** (`stages`: 72B's gate_up shape, 2048-row blocks, medians of 5; per unit):

| stage | before | after | |
|---|---|---|---|
| hint switch + recombination (us / row) | 12.53 (torch) | 0.637 (fused) | 19.7x; 36 B per value at ~72 % of the T4's bandwidth |
| `y` recombination (us / row) | 0.987 | 0.177 | 5.6x |
| ring rows (us / input feature) | 1.445 (host, 1 thread) | 0.035 (device ChaCha20) | 41x; 4 host threads: 0.742 |
| upload of `b` (ns / value) | 2.81 (pageable) | 2.21 (pinned) | the host staging copy is included |
| reply copy (us / row, not overlapped) | 2.52 (pageable) | 1.65 (pinned) | the item also overlaps it with the next block |
| key switch, 4 groups (us / row) | 2.50 (fused NTTs) | 2.50 (unchanged) | 3.14 with `forward4`; torch: 97.6 |
| hint GEMM / `y` GEMM (us / row), planes (us / feature) | 13.2 / 3.26 / 2.40 | same | now the T4's largest stage |

Whole items on the T4, seconds:

| linear | every kernel plain | fused NTTs only (before) | this round under `auto` (host wire) |
|---|---|---|---|
| 7B gate_up | 5.25 | 1.70 | 1.20 |
| 7B down | 0.91 | 0.58 | 0.58 |
| 7B head | 20.90 | 6.62 | 4.69 |
| 32B gate_up | 7.74 | 2.57 | 1.91 |
| 32B down | 1.43 | 0.98 | 0.88 |
| 72B gate_up | 8.57 | 3.12 | 2.54 |
| 72B down | 2.08 | 1.40 | 1.27 |

Notes on the whole items:
- The "after" column still packed and unpacked the wire on the host, and it ran `forward4`, which `auto` picked at
  the time.
- With the T4's rates (`memplan.T4_RATES`) the model reproduces both columns within 25 % (before) and 30 % (after),
  per item (`test_memplan_reproduces_the_t4_items`).
- The model attributes ~1.2 s of the 72B gate_up item to the host's pack of its 60M-value reply.

**Model** (`memplan`):
- New option: `MemConfig.pks_kernels` (`--pks-kernels all|ntt`, default `all`).
- New: `--pks-gpu h100|t4` (`T4_RATES`, measured).
- The key switch is recalibrated on this run's direct 1- and 4-group measurements (`T4_R2_S`). The first run
  measured 3.59 us per row for 4 groups on another T4 instance; this run measured 2.50 for the same kernels.
- New stages: `wire_host`, `keys_host`, and the reply keys' upload.
- New kernel costs: the fused combine and the device ChaCha20, each the larger of its traffic and its T4-calibrated
  SM work (an upper bound where the T4 was memory-bound).
- The `all` path models the copies as overlapped. The upload is exposed only beyond the ring rows, the planes and the
  first row block's hint; the packed reply only for the last row block.

H100 at 72B, ESTIMATES, seconds per 1024-slot cycle (`python -m pgw.service.memplan --pks-throughput`):

| stage | before | after |
|---|---|---|
| host wire: upload unpack + reply pack (not modelled before) | 184.5 | 0 (device: 0.04) |
| hint GEMM + switch + recombination | 18.6 | 4.6 (GEMM 3.3, fused switch <= 1.2) |
| host ChaCha20 of the ring rows | 8.9 | 0.04 (device) |
| key switch (recalibrated) | 4.6 (6.6 with the first run's T4) | 4.6 |
| uploads (`b` + keys) | 5.0 | 4.4 (wire bytes, pinned staging; host copy at 10 GB/s assumed) |
| host unpack of the reply keys (not modelled before) | 3.0 | 3.0 |
| reply copy back | 2.9 | 0.2 (4.5 B per value, overlapped but the last block) |
| `y` GEMM + recombination, planes | 1.6 | 1.6 |
| **cycle** | **229.1** (41.6 without the two host stages it never modelled) | **18.4** |

Slots per second (1024 slots per cycle; H100 = estimates, T4 = the model with the T4's measured rates):

| model | H100 before (as modelled before this round) | H100 before (with the host wire work) | H100 after | T4 before | T4 after |
|---|---|---|---|---|---|
| 7B | 122 | 20.9 | 264 | 11.8 | 47.6 |
| 32B | 37.9 | 6.65 | 86.8 | 3.52 | 11.9 |
| 72B | 24.6 | 4.47 | 55.6 | 2.14 | 5.86 |

What is left at 72B (H100 estimate, 18.4 s per cycle):
- the key switch, 4.6 s;
- the hint GEMM + switch, 4.6 s;
- the uploads, 4.4 s: host staging copies, which overlap device work only when 2 items run at once (not modelled);
- the host unpack of the reply keys, 3.0 s. It could move to the device too, with its range check kept there;
- on the T4 the int8 GEMMs dominate (35.7 TOPS measured).

**RunPod** (`deploy/runpod/provider_perf_session.py`, `maskgen` phase, now ~30 min). Device items run:
- under `auto` at every work budget;
- at 512 MB with `fwd4`, `prev` (the fused NTTs only) and `torch`.

Each run records `fused_status`, the replies must be byte-identical, and the estimates `device_fused` / `device_prev`
sit next to them. The CPU smoke run passes.

Tests:
- `tests/service`, one directory at a time: 550 passed, 77 skipped (CUDA). This includes `test_pks_fused.py` (34,
  new), `test_pks_ntt.py` (21) and `test_pks_device_item.py` (20, two memplan assertions updated for the new
  stages and the recalibration).
- `tests/benchmarks/test_provider_perf_tools.py`: 2 passed.
- The bench's CPU smoke run (`benchmarks/pks_ntt_bench.py --cpu`) passes.

New settings:
- `PGW_SERVICE_PKS_GPU_SWITCH`, `PGW_SERVICE_PKS_GPU_RING` and `PGW_SERVICE_PKS_GPU_WIRE`, default `auto`.
- `PGW_SERVICE_PKS_GPU_FWD4`, off by default.
- `PGW_SERVICE_PKS_PINNED`, default on.
- `PGW_SERVICE_PKS_HOST_THREADS`, default `min(8, cpus)`.

Cloud spend: one Kaggle T4 kernel (6.5 min). No other compute.

## 2026-11-26: client mask factory stalls (provider dies during maskgen) and first-use UX

Scope: `src/pgw/sdk` (engine `_open`, factory supervision), docs. Trigger: run 2 attempt A on the 7B T4
(`results/kaggle/sdk-7b-validation/run2_7b/`). The provider died during pks-v3 maskgen. The factory logged
`ServiceUnavailableError: compute service transport error (ConnectError)`, but `consecutive_failures` stayed 0.
The first request, still in its first-fill window, waited out the step's 3 h.

**Root cause.** The supervisor (`FactoryProcess`) counted only *child exits* as failures. A run error inside the
child is retried with back-off (5 s doubling to 10 min) and only lands in `factory-state.json` `last_error`.
- At `bfee514` (attempt A), the first-fill wait (`_first_fill`) ignored that file, so the request waited up to
  `ready_timeout_s` plus `timeout`.
- The later reliability fix (`fac_error is None` in `_open`) went the other way. Any `last_error` ended the
  first-fill wait at once: a single transient blip, or a stale error left in the file by an earlier process.
- A blackholed provider recorded no error for `he_timeout_s` (600 s).
- Nothing measured progress, so the code could not tell "slow" from "stuck".

**Fix (supervisor watch, `pgw.sdk.factory`).** Every `poll_s` the owner's supervisor looks at the child from
outside. Signals:
- the hint build's progress, from the files in `*.building-<pid>-*` or `*.partial`, weighted by rows x columns
  per linear; this gives a percentage and an ETA;
- segment progress: writes to `.part` files and `last_run_at`;
- the child's heartbeat (unchanged);
- the provider's reachability: `GET /readyz` with the session's pinned CA and client certificate, no token. It
  runs only during first fill, after a run error, or while a request waits.

Run errors saved before the current child started are ignored. A **stall** is either of:
- the provider has not answered for `factory_stall_s` (new option, default 120 s, `PGW_FACTORY_STALL_S`) and no
  segment progress was seen in that time;
- run errors for `factory_stall_s` with no segment made, and no retry running against a reachable provider.

What a stall does:
- `_open` fails waiting requests at once, in first fill and in steady state, after one more probe confirms it.
  The provider unreachable gives `APIConnectionError` (`backend_unavailable`). Otherwise it is `NotReadyError`.
  The message names the provider, the probe result, the factory's last error and any first-time setup.
- The child keeps retrying.
- When the provider answers again after an outage, a child that is waiting out its back-off is restarted at once
  ("nudge"). This never happens during a hint build or a running segment, and it does not count as a failure.

A healthy setup is never cut short, however long it takes, up to `ready_timeout_s`.

What the supervisor sees is written to `<factory dir>/supervisor.json` (public values, 0600). `client.status()`
shows it as `factory.health`, and `pgw sdk status` and request errors show:
- `first-time setup in progress: building the one-time pks-v3 hint, 45% done (about 9 min left)`, or the first
  segment and its running time;
- the segment being made and its ETA;
- `provider not answering for N s`;
- `STALLED ...`.

**mask_wait_s scaling (bounded).** In steady state, a request whose masks are minutes away now waits for them
under all of these conditions:
- the factory is visibly making a segment, or made one within `factory_stall_s`;
- the factory is not stalled;
- the segments the request lacks are due, at the last measured rate, before `max(mask_wait_s, timeout_s)` from
  the request's start and before its deadline.

The cap is the request's own `timeout_s` (600 s by default), so the wait stays bounded. Attempt B (needed 42
slots, 14 in stock, 512-slot segments in 266 s) would have waited instead of failing at 120 s. When the request
does fail with `mask_exhausted`, the message says when the factory expects the next masks.

Integrity failures and halts are unchanged: they stay fatal, are never retried and are never nudged away.

**Measured** (tiny model, pks-v3, real provider behind the chaos proxy, SIGKILL plus refused connections,
`factory_stall_s` = 8 s):

| case | result |
|---|---|
| provider dies during first fill: the waiting request fails after | 12.2 s / 13.4 s (two runs) (APIConnectionError naming the provider and the factory's ConnectError) |
| answer after the provider restarts | 7.9 s (1 nudge) / 5.0 s (the child's own retry) after it is ready |
| provider dies in steady state: a request beyond the stock fails after | 10.8 s / 12.9 s (`mask_wait_s` was 600 s) |
| answer after the restart | 1.0 s (the stock covered it) |
| that big request served after the restart | 13.6 s / 10.2 s later (needs the factory: nudge, then segments) |
| slow but healthy first fill (every provider read delayed 0.1 s; stall bound 3 s) | answered after 59.9 s; no stall, no restart |

Detection takes `factory_stall_s` plus up to one probe interval (`min(10 s, stall/6)`) plus one supervisor poll
(3 s). With the defaults that is at most about 133 s, against hours before.

**Docs.**
- `reliability.md`: stall and failing-runs rows, "slow is not stuck", the status lines.
- `troubleshooting.md`: the run-2 first-use timeline, `pgw sdk prepare` before first traffic, the new
  `APIConnectionError` entry, `mask_exhausted` reasons.
- `quickstart.md` and `configuration.md`: `pgw sdk prepare` is recommended before traffic; `mask_wait_s` and
  `factory_stall_s` rows.
- `changelog.md`.

**Tests.**
- `tests/sdk/test_factory_watch.py` (11, no provider): hint progress, percentage and ETA, a dead build ignored;
  a dead provider stalls within the bound, then the recheck and the nudge once it is back; a stale error is not
  counted; a retry against a reachable provider is not a stall; segment progress overrides a failing probe; the
  real probe; option validation; `_open` fails at once on a stall (first fill and steady state); the wait
  extension, and its bound.
- `tests/sdk/test_factory_stall.py` (4, end to end): the three cases above, plus "first-time setup in progress"
  in `pgw sdk status`.
- `tests/sdk/chaos.py`: `delay_s` (a slow but working path).
- tests/sdk: 178 passed, 1 opt-in skip, 10.5 min, sequential, no xdist. That run used a `pgw_fast` built from this tree. The installed site-packages `pgw_fast` lacks the lane-format-2 multi calls, and 3 `test_lane_multi` tests fail with it; this is unrelated to the change (sequential, no xdist).

**Open.**
- A provider that answers `/readyz` but never answers maskgen (no reset, no error) is detected only when the
  factory's request times out (`he_timeout_s`, 600 s) plus `factory_stall_s`.
- A provider that is down while the hint is being built fails waiting requests after `factory_stall_s`. They could
  not be served anyway, and the build continues.
- The nudge is a child restart: at 7B it re-verifies the hint digests (seconds).
- The wait extension uses the last segment's rate (the first 64-slot segment is slower per slot than later ones).
- A real 7B rerun of attempt A's failure was not done (no cloud spend).


## 2026-11-26: integration sweep after the day's merges; RunPod runbook

Scope: every merge of 2026-11-26 together (SDK integration, round trips and speculation, client setup
scaling, security review and follow-ups, provider memory/memplan, the device hint path, the fused NTT and the
round-2 fused kernels, multi-user scaling, mask cost, AES-GCM on AWS-LC, the multi-item lane, reliability, stall
detection, pack format 2, streamed weights, the 7B Kaggle validation). Fixed what was broken; no features. No
cloud spend, no RunPod call.

**Build.**
* `cargo build --release` of pgw-fast, pgw-ntt and pgw-pks, with the `.so` files copied to `src/` as `*.abi3.so`.
* `cargo test --release`: pgw-fast 83 (2 ignored), pgw-ntt 9, pgw-pks 12, pgw-core-rs 36 + 6 (vectors),
  pgw-lane 46. All pass.

**Tests**, one directory at a time, absolute `PYTHONPATH=<worktree>/pgw/src:<worktree>/pgw`, xdist `-n 2`
except sdk, docs, packaging and e2e (serial):

| directory | before | after | notes |
|---|---|---|---|
| docs | 21 failed / 109 passed | 130 passed | 3 real drift fixes (below); 18 `test_numbers` failures are environment-only: they need the gitignored `results/kaggle` logs of the main checkout (copied in for the "after" run) |
| deploy | 128 / 1 skip | 141 / 1 skip | +13 new (`test_runpod_jobs.py`) |
| benchmarks | 1 failed / 43 | 44 | environment-only: `AF_UNIX path too long` under a long `TMPDIR`; passes with a short one |
| engine | 1 failed / 1159 / 15 skip | passes alone | `test_scheduling_pingpong` overlap assertion is timing-dependent under load (another parameter failed on a second loaded run; 8/8 alone) |
| gateway | 1 failed / 283 | passes alone | `test_qwen05b_masked_bit_identical_across_formats`: xdist worker OOM-killed (real 0.5B on the shared 15 GB box) |
| sdk (serial) | 178 / 1 skip | same | |
| service | 558 / 77 skip (CUDA) | same | |
| core 582 / 28 skip; hemask 439 / 20; factory 485 / 9; maskstore 247; native 357; wire 334; cli 209 / 3; modelpack 110 / 1; provision 52 / 1; console 96; observability 71; e2e 34; platform 32; packaging 11; enroll 9; moe 9; vectors 3 / 2; test_skeleton 18 | all pass | | `tests/integration` holds no tests |

**Fixed.**
* Doc drift:
  * `docs/README.md` now indexes `sdk/scaling.md`;
  * the env table in `docs/sdk/configuration.md` lists `PGW_MASK_ADMISSION` and `PGW_FACTORY_STALL_S`;
  * the docs consistency test knows the SDK and factory env switches that are not settings fields
    (`PGW_SPECULATION`, `PGW_MASK_ADMISSION`, `PGW_FACTORY_STALL_S`, `PGW_SDK_FACTORY_HANG_S`, `PGW_PKS_COMPACT`,
    `PGW_VECMAT_NATIVE`, `PGW_WRITER_THREADS`);
  * reliability.md's example URL uses a reserved name;
  * models.md and concepts.md count the masked head's trip: 4L + 1 = 97 / 113 / 145 / 113 per token;
  * pilot-handoff's first-run estimate (11 GB download; ~10 CPU-minutes of setup with AMX; `pgw sdk prepare`).
  * Checked and consistent: option names and defaults against `SDKOptions`, the pins in models.md against
    `PINNED_MODELS`, and pilot-handoff's commands and doctor output against the CLI.
* `setup.py` imported setuptools-rust unconditionally. The provider Dockerfiles'
  `pip install --no-build-isolation .` runs against hash-locked requirements without it, so the image build
  failed at that step. It now installs pure-Python there; the images build `pgw_fast` separately, as before.
  `PGW_REQUIRE_NATIVE=1` still fails without it.
* `provider_perf_session.py --dry-run` created `/workspace/pgw/perf-*` on the launching machine. A dry run now
  writes only into an existing `--out`. One stale `/workspace/pgw/perf-20261126-0701/` from this sweep is still
  on the integration box (removal needs a person: the sandbox refuses `rm` of `/workspace`).
* ruff: the SDK and its tests are clean again (four lint items; no behaviour change).
* STATUS: this file's current-state summary at the top; italic notes where a later section superseded a
  statement (open security items, the one-item lane, the 7B T4 run's client wait, the 2026-11-25 open pins,
  the tests/docs failures).

**Release check.** `scripts/release_check.sh --tests --fallback` passed: sdist, a manylinux_2_28 abi3 wheel with
all three extensions, a fresh-venv install with CPU torch (1.3 GB site-packages, 5.0 MB wheel), the smoke test
native and `PGW_NATIVE=0`, native bit-identity tests against the install (413 passed, 5 skipped), and a
pure-Python wheel without Rust. 5 min.

**mypy** (`src/pgw/sdk` + `src/pgw/service` and what they import, strict): 147 errors at `7b5b3c5` (end of
2026-11-25), 170 now. The 23 new ones are typing-only:
* `**dict` arguments to dataclass constructors: `memplan.py:782`, `wire/service_lane.py:285-286`;
* numpy and torch overloads: `pks_gpu.py`, `factory/kernels.py`, `wire/lane.py`;
* a `Path | None` narrowed through `getattr`: `sdk/engine.py:1008`.

The old baseline was not chased.

**RunPod.** `deploy/runpod/RUNBOOK.md` and `deploy/runpod/jobs.py` (+ `pod_job.py`, `job_setup.sh`,
`sdk_remote_bench.py`):
* sessions driven only through RunPod's API and its HTTPS pod proxy (the session proxy allows no raw TCP, no SSH
  and no pinned TLS to a pod's mapped port);
* public digest-pinned base images;
* a code bundle pinned by SHA-256;
* TTL self-stop, watchdog, `--max-usd`, a $60 `--budget-usd` over all sessions, Secure Cloud only;
* four sessions in priority order: H100 perf + lane over a real NIC; SDK at 7B with speculation and N = 1/4/16;
  32B on one H100; 72B on 2 x H100 or an H200. $38.82 at full TTL with table prices.

Validated without RunPod:
* every dry run;
* the job pod's control server (tokens, pinned bundle, `data`-filter extraction, peer channel);
* a whole two-pod `lane_nic` session on loopback;
* the client SDK bench's CPU smoke.

The unverified RunPod API details are listed in the runbook.

## 2026-11-26: shard-wise streaming packer; baked pins for Qwen2.5-14B / 32B / 72B (first use without the checkpoint)

Goal: SDK first use at 14B-72B without downloading the whole bfloat16 checkpoint (15 / 29 / 65 / 145 GB at 7B / 14B /
32B / 72B, peak disk to match). Two parts: a packer that holds one checkpoint shard at a time, and baked format-2 pins
for the three larger models so that clients never re-derive them.

- **`pgw.modelpack.shardwise.stream_pack_checkpoint`** (new). Plans the whole pack from the safetensors *headers*
  (hub range requests via `get_safetensors_metadata`, or local files; all shape / dtype / index checks of
  `pack_checkpoint` up front), then walks the shards in order: fetch one (`hf_hub_download` into the work dir), pack
  every linear group and plain tensor whose inputs are now all present, copy the tensors still needed by a group
  that continues in a later shard into a spill file, delete the shard. Output files are planned once (same names,
  header bytes and offsets as `pack_checkpoint`) and filled in any order (`st_io.SafetensorsRandomWriter`);
  `out_dir=None` only hashes (pins). Quantization runs over row blocks of at most 64 M elements
  (`quantize_weight_rows` is per row, so the bits are the whole-matrix ones; a unit test checks this across part
  boundaries and a zero row). Resumable: `state.json` (per item: SHA-256 digests, `row_norm_sq_max`; finished
  shards; spill files) is written after the item's bytes are fsynced, under a digest of the whole plan; a changed
  plan starts over. The pack is assembled from the recorded digests, so an edited work dir yields a pack that fails
  full verification (tested). A checkpoint already complete in the local HF cache is read in place. `prefetch=1`
  overlaps the next download (one more shard on disk). `plan_summary()` gives the sizes from the headers alone.
- **Exactness** (`tests/modelpack/test_shardwise.py`, 18 tests): every pack file byte-identical to `pack_checkpoint`
  and the manifest equal except `created_at`; the hash-only manifest is equal too. Cases: the tiny Qwen2 (tied head,
  q/k/v biases) virtually resharded 1 / 3 / 7 ways round-robin (almost every group straddles shards); a bfloat16
  copy with tied and untied head, format 2 and format 1, as real shard files fetched and deleted one by one (never
  two present); the tiny Llama; the hub path with `huggingface_hub` replaced by a local repo (one download at a time,
  same manifest as `pack_checkpoint("org/tiny")` including `source`); the masked head (`MaskedHeadPack` W8, scales,
  head fingerprint) identical tied and untied; **Qwen2.5-0.5B** (resharded like the hub, spill used): written pack
  byte-identical, release pin `d773a502...` and format-1 pin `915e55c1...` reproduced; **Qwen2.5-1.5B** (round-robin
  into 5 virtual shards): byte-identical, pins `3d198de4...` and format-1 `81bb7adc...` reproduced. Also: resume
  after an interruption (write and hash-only), plan change, index mismatch and NaN rejections.
  On Kaggle at **14B**: hash-only, shard-wise write + `verify="full"`, and `pack_checkpoint` on the whole checkpoint
  give the same fingerprint, the same manifest and byte-identical files (`results/kaggle/sdk-pins-14b`).
- **Uses**: `pgw.provision.client._full_from_hf` (the `source="hf"` public check of an unbaked pin, and
  `weight_source(hf_repo=...)` for `factory_weights="stream"` with `factory_weights_source="hf"`) packs shard by
  shard into `models/<fp16>/hf-stream` (resumable), then verifies the pack in full against the pin as before; a pack
  that misses the pin is now deleted instead of left behind. `tests/provision/test_shardwise_provision.py` (3): one
  shard on disk at a time on both paths, factory pack verified, a changed checkpoint refused with nothing installed.
  `scripts/pin_model_fingerprints.py`: shard-wise hash-only by default; `--write-pack` (write + full verify),
  `--compare-packer` (+ `pack_checkpoint`, files compared), `--prefetch`, `--plan`; `--compare-format1` unchanged
  (whole-checkpoint packer). Kaggle jobs `pgw-sdk-pins-{14b,32b,72b}` in `benchmarks/kaggle/build.py`.
- **New pins** (`PINNED_MODELS`, docs/sdk/models.md, CONTRACT.md A3), pack format 2, one Kaggle CPU kernel each
  (4 vCPU, 31 GB), pushed one at a time through the REST API once the other agents' kernels had finished:
  14B `d4afbe77ff07...` @ `cf98f3b3` (hub main), 32B `a4cf6ba9ac2b...` @ `5ede1c97` (the provider image's commit),
  72B `5e8001c02d79...` @ `495f3936` (hub main). 32B / 72B were derived hash-only; their exactness rests on the
  packer tests above and the 14B file-by-file comparison, not on a second packer run at that size.
- **Measured, shard-wise hash-only with one shard prefetched** (Kaggle CPU; 0.5B / 1.5B on this box, cached, read in
  place):

  | | checkpoint | shards | peak scratch disk | spill peak | peak RSS anon (ru_maxrss) | time (download overlapped) |
  |---|---|---|---|---|---|---|
  | 0.5B | 0.99 GB | 1 | 0 (read in place) | 0 | 0.36 GB (2.0) | 27 s |
  | 1.5B | 3.1 GB | 1 | 0 (read in place) | 0 | 1.11 GB (4.6) | 56 s |
  | 14B | 29.5 GB | 8 | 8.1 GB | 0.14 GB | 4.4 GB (8.3) | 12.1 min |
  | 32B | 65.5 GB | 17 | 8.4 GB | 0.57 GB | 4.2 GB (8.0) | 23.9 min |
  | 72B | 145.4 GB | 37 | 8.5 GB | 0.48 GB | 6.2 GB (10.8) | 53.3 min |

  Without prefetch the scratch peak is one shard plus the spill: 4.1 / 4.2 / 4.5 GB (header plan). `ru_maxrss` counts
  the memory-mapped shard pages read; RssAnon is the process's own memory (largest item: the 2.5 GB bfloat16
  `embed_tokens` at 72B). Time is quantization-bound on 4 vCPUs (2-11 min of it is download, overlapped). With the
  pack written (the client's `source="hf"` path, 14B measured): peak 18.9 GB = one shard + the 16.3 GB pack, 13.7 min;
  the whole-checkpoint path needed 29.5 + 16.3 GB. Client disk for an unbaked pin is now about 4 GB plus the int8
  pack (8.7 / 16 / 34 / 75 GB at 7B / 14B / 32B / 72B) instead of the checkpoint plus the pack.
- **Not changed**: `SUPPORTED_MODELS` (the SDK still accepts 0.5B-7B only; 14B-72B are pinned ahead of support, and
  the test now requires `SUPPORTED_MODELS` to be a subset of the pins). The `pgw pack` CLI still uses
  `pack_checkpoint`. Verification against pins is unchanged (full).
- Tests (one dir at a time, absolute `PYTHONPATH`): tests/modelpack 128 passed / 1 skipped (CUDA); tests/provision
  55 / 1; tests/sdk/test_unit.py 47, test_e2e_product + test_prepare 8 / 1 opt-in skip; tests/docs: only the earlier
  failures (test_numbers logs, README index, `PGW_MASK_ADMISSION`).

## 2026-11-26: maskgen watchdog (wedged provider GPU), AVX2 / SSE4.1 check-vector kernels, mypy back to baseline

**Maskgen watchdog.** A provider that answers `/readyz` but finishes no maskgen item (a wedged GPU, a hung kernel)
was noticed by the client only when the factory's maskgen request ran out of its 600 s budget
(`he_timeout_s`); requests waiting for masks waited `mask_wait_s` or longer meanwhile.
* Provider (`pgw.hemask.progress.MaskgenProgress`, counted in `ServiceHandler._maskgen_in_slot`, so the HTTP route
  and the bulk channel alike): every item from admission (waiting for a device slot) through its worker thread
  (running) to its end; a cancelled request's item stays counted until its thread ends, like its slot. Bound:
  `max(PGW_SERVICE_MASKGEN_STALL_S = 30 s, 4x the slowest recent item)`, `PGW_SERVICE_MASKGEN_WARMUP_S = 300 s`
  for the first item of a (device, out, in) shape since start (Triton compilation, device self-tests). Exposure:
  public `/readyz` gets **one coarse bit** `"maskgen": "ok" | "stalled"` (no counts, ages or tenants; the S5 rule);
  `GET /pgw/v1/maskgen/status` (authenticated like maskgen, `Cache-Control: no-store`) gives the caller **its own**
  items only (waiting, running, oldest running age, done, failed, slowest recent, cold) plus the bit; the metrics
  port's `/readyz` gives the operator the totals (`maskgen_progress`, no tenant names). `/readyz` status codes are
  unchanged (a stall does not take the service out of a load balancer).
* Client (`FactoryProcess`, option `maskgen_stall_s` / `PGW_SDK_MASKGEN_STALL_S`, default 30 s, `0` off): while a
  segment is being made (`in_progress`), the supervisor reads the status every `min(5 s, bound / 6)` with the
  session's pinned CA, client certificate and the factory's token; an item of this key running past the same
  adaptive bound (or the provider's bit `stalled` with items of this key waiting) is `phase: "maskgen_stalled"`
  with the reason, item age and bound in `health()` / `supervisor.json` / `pgw sdk status` (`STALLED`). Requests
  waiting for masks then fail at once with the new **`ProviderStalledError`** (subclass of `APIConnectionError`,
  `backend_unavailable`, retryable, `details.reason = "provider_maskgen_stalled"`), after one more status read. It
  clears as soon as the item finishes. Progress, not wall time: the client's own work (encryption, hint build, a
  slow first segment at 32B/72B) is not on the provider's clock; a provider whose items get slower raises the bound
  as they finish. A provider without the endpoint (404/405) turns the watchdog off after one read. Defaults on: no
  false positive in the tests below, and on the GPU an item is ~0.05-1 s against a 30 s / 300 s bound (estimated
  from the T4 throughput at 72B, 30-40 slots/s: about 50 ms per item on average; not measured per item on an H100).
* Tests: `tests/service/test_maskgen_progress.py` (11: fake clock; wedged item, slow-but-finishing items whose
  bound grows, cold first item, queue that does not move, tenant isolation, abandoned/failed items, the handler's
  accounting incl. a cancelled request, the authenticated per-tenant route, public `/readyz` carries only the bit);
  `tests/sdk/test_factory_watch.py` (+7: a fake HTTP provider with a real `MaskgenProgress`: a wedged item is a stall
  after 0.6 s not before, `supervisor.json` has it and no token, recheck clears it; 0.5 s items against a 0.6 s
  bound plus a 1.5 s cold first item: never a stall over 5 s; the provider's bit with waiting items; idle factory
  and a 404 provider: no stall, no more reads; `0` off; the option; `ProviderStalledError` from `Engine._open`);
  `tests/sdk/test_factory_stall.py::test_a_wedged_provider_maskgen_fails_waiting_requests_fast` (the real service,
  pks-v3, every maskgen item blocked through a test-only hook in `tests/sdk/provider_main.py`,
  `PGW_TEST_MASKGEN_HANG_FILE`): 10 answers from the stock, then the refill hangs, the stall is seen at 5 s
  (bound 4 s) and the waiting request fails after 4.2 s with `ProviderStalledError`; items unblocked: served again
  after 1.8 s, bit-identical. Whole file 5 passed (the slow-but-healthy first fill still waits). tests/service
  maskgen/app/handler subset 192 passed.

**Check-vector kernels on CPUs without AVX-512** (`rust/pgw-pks/src/vecmat.rs`: `vecmat_wrap32`, `vecmat_p31`,
`check_rows32`). Before: a hand-written AVX-512F path; AVX2 CPUs ran the generic body compiled with AVX2
(auto-vectorised), all others the SSE2 baseline (no 32-bit lane multiply). Now, picked at run time by `Isa::parse`
(`PGW_PKS_ISA` caps it, as for the other pks kernels): hand-written AVX2 8-column tiles (`x256`), and the generic
body compiled with SSE4.1 where the CPU has it (also for `check_rows32`); aarch64 keeps the generic body with NEON
(`cargo check --target aarch64-unknown-linux-gnu`; not run on an ARM CPU). Integer sums only, so every ISA gives the
same bits: Rust tests over every ISA of the CPU (worst-case lanes `u = p-1`, `w = -128`, 1000 rows, ragged
columns; random shapes), `tests/factory/test_vecmat_native.py` 66 passed and `test_self_check.py` 10 passed with
`PGW_PKS_ISA=scalar`, `avx2` and `auto`; output digests equal in every benchmark. `pks-bench vecmat` (this 4-vCPU
AVX-512 box, one thread, ms for `x @ W8`, best of 3, shared box: +-20 %):

| W8 (m x n), vectors | baseline SSE2 | AVX2 before (auto-vectorised) | AVX2 now | SSE4.1 now | AVX-512 |
|---|---|---|---|---|---|
| 18944 x 3584, c=1: wrap32 / p31 | 44 / 47 | 16.0 / 24.0 | 11.7 / 22.4 | 19.6 / 31.5 | 7.0 / 9.4 |
| 18944 x 3584, c=2 | 45 / 70 | 23.1 / 34.2 | 17.0 / 27.9 | 45 / 78 | 10.8 / 18.6 |
| 18944 x 3584, c=8 | 253 / 442 | 52.9 / 107 | 46.8 / 87.9 | 93 / 149 | 40.0 / 69.2 |
| 3584 x 18944, c=2 | 45 / 78 | 21.1 / 29.2 | 13.3 / 22.8 | - | 8.4 / 14.3 |
| 3584 x 4608, c=4 | 17 / 62 | 10.4 / 16.4 | 7.7 / 13.8 | 9.3 / 17.2 | 3.2 / 6.4 |

AVX2: 1.1-1.6x over the auto-vectorised body (wrap32 1.2-1.6x, p31 1.05-1.45x); SSE4.1: 1.5-3x over the baseline
(e.g. c=8 2.7x / 3.0x); `check_rows32` (1024 rows, 18944 + 3584): baseline 149 ms, SSE4.1 57 ms, AVX2 (unchanged,
auto-vectorised) 44 ms, AVX-512 29 ms. The other native kernels already had AVX2 paths (pgw-pks `simd2`/`vntt`,
pgw-fast, pgw-ntt: checked).

**mypy** (`src/pgw/sdk` + `src/pgw/service` and what they import, strict): 184 at the start of this section (the
23 of the integration sweep plus 14 from the shard-wise packer merge), **147 now = the `7b5b3c5` baseline** (the
only difference is one line-number shift). `mypy src`: 150, the same set as at `7b5b3c5`. Typing-only changes:
`**dict` arguments typed `dict[str, Any]` (`memplan.max_fitting`, `service_lane` ticket), misplaced
`type: ignore`s moved to the line mypy reports (`wire/lane.py`, `pks_gpu.py`), `_tw_t` annotated with the tensor
it always gets, `np.max` for `ndarray.max()`, loop variables renamed where they shadowed differently typed names
(`modelpack/shardwise.py`, two now-unneeded ignores dropped, a `list[dict[str, Any]]` annotation), a local for a
`getattr` narrowing (`sdk/engine.py`). Tests: tests/modelpack/test_shardwise.py 18, tests/service/test_memplan.py 8,
tests/wire 334 passed.

**Open.** The watchdog's bound on a real H100 at 32B/72B (per-item times are estimated, not measured; the RunPod
sessions can log `maskgen_progress` from the metrics port). A wedge of one CUDA stream while others progress is
caught per item; a provider that answers neither the status endpoint nor `/readyz` is the existing
`factory_stall_s` path. The SSE4.1 and NEON paths are not benchmarked on a CPU that lacks AVX2 (only forced here).

## 2026-11-26 — pks-v3 device item, round 3: reply keys on the device, staged uploads (provider maskgen throughput)

Goal: raise provider pks-v3 mask throughput by moving the per-item provider steps still on the host onto the GPU.
The scheme, the wire and the replies are unchanged; the provider sees the same data as before.

**Profile first** (H100 estimate from `memplan --pks-throughput`, host steps measured on this 4-vCPU box). Device
path, seconds per 1024-slot cycle at 72B before this round (18.4 s, 55.6 slots/s):

| stage | s / cycle | where |
|---|---|---|
| key switch (fused NTTs) | 4.62 | device |
| hint GEMM + fused switch | 4.59 | device |
| uploads (`b` wire bytes + keys): one-thread host staging copy, then PCIe | 4.43 | host copy + PCIe |
| reply-key unpack + range check (`unpack_ksk`) | 2.96 | **host**, numpy, one thread |
| `y` GEMM, planes, reply copy, ring rows, reply pack | 1.8 | device / overlapped |

* (1) Key switching and the hint GEMM: **nothing of them is on the host** on the device path
  (`pks_gpu.item`: ring rows, `A` planes, hint GEMM, switch, `y` GEMM and key switch all run on the device since the
  earlier rounds). The cache path (host hint) is used only when a model's hints fit the host cache, which no 0.5B+
  model does under `auto` on CUDA. Nothing to move.
* (2) Uploads: 4.4 s at 72B. The staging copy into pinned memory ran on one thread (7.2-8.0 GB/s measured here)
  and *then* the transfer started, so the two added up.
* (3) Reply-key unpack: 3.46 ns per residue measured here (model: 4.4), 7.3 ms per 4-group item whatever the
  linear: 1.0 / 2.4 / 3.0 s per cycle at 7B / 32B / 72B, **27 % of 7B's cycle**.

**What moved** (`pgw.service.pks_gpu`, `pks_provider`):
1. **Reply keys unpacked and range-checked on the device** (`PGW_SERVICE_PKS_GPU_KEYS`, default `auto`).
   * `run_item` checks the keys' size on the host (as before) and hands their wire bytes on (`PackedKeys`).
   * The device unpacks the `pack_bits` fields (16 + 8-bit byte fields, the 6-bit tail stream of the 30-bit primes)
     straight into the fused kernels' `(4, G, 16, 8192)` uint32 form (`_unpack_ksk_dev`; 47 torch integer
     operators per item, one pass per run of equal-width primes). 1.77 MB per group go up instead of 2.1 MB of
     uint32, and the int64 -> uint32 re-layout of the keys is gone.
   * **The range check is kept**: the device reduces `residue >= p` to one flag, copied to pinned memory on the
     upload stream; the item reads it before its first key switch and refuses the item with the same
     `ProtocolError` as the host unpack. No reply is computed from out-of-range keys or returned.
   * A customer's bad keys are not a device fault: `_run_device` re-raises `ProtocolError` without switching the
     device path off (before, the host unpack raised before the device was involved; tested).
   * Start-up self-test `_keys_self_test` (2 groups, residues at 0, 1, p - 2, p - 1 and every byte boundary; must
     flag `p` and `2^bits - 1` in each prime). Any mismatch or error falls back to the host unpack on that device for
     good (same bits). `item_self_test` now also runs the keys' wire bytes end to end.
2. **Staged uploads** (`staged_copy`, under `PGW_SERVICE_PKS_PINNED`, default on). Every upload of the item (the
   keys' and `b`'s wire bytes, the host fallbacks) is copied into its pinned buffer in 8 MB pieces by the host
   thread pool (numpy copies run without the GIL; 1.7-2.3x one thread: 9.2 against 5.4 GB/s on the T4 VM, 17.3
   against 7.6 GB/s here), and each piece's
   non-blocking transfer is queued as soon as the piece is there, so the host copy and PCIe overlap instead of
   adding up. An error waits for or cancels every copy before it propagates (no copy keeps writing into a released
   buffer). `PGW_SERVICE_PKS_HOST_THREADS=1` gives the previous behaviour.

No secret material is involved: the reply keys are the customer's key-switching ciphertexts and `b` its upload, as
before; nothing new is kept after the item (the host unpack is cached only in the item's `PackedKeys`).

**Estimates** (`python -m pgw.service.memplan --pks-throughput`; `--pks-kernels r2` is the previous round). The
H100 device rates are datasheet estimates, as before; the host copy pool is modelled at 1.7x the previous 10 GB/s
single-thread assumption (17 GB/s; the lower of the two measured scalings), the T4's at its measured 9.2 GB/s.
Slots per second:

| model | H100 before (r2) | H100 after | T4 model before | T4 model after |
|---|---|---|---|---|
| 0.5B | 610 | 1567 | | |
| 7B | 264 | 450 | 47.6 | 49.8 |
| 14B | 159 | 276 | | |
| 32B | 86.8 | 135 | 11.9 | 12.2 |
| 72B | 55.6 | 81.3 | 5.86 | 5.94 |

72B, seconds per cycle: 18.4 -> 12.6 (keys 2.96 -> 0.02 on the device; uploads 4.43 -> 1.49, now bound by the
host copy pool, PCIe close behind). What is left is device compute: key switch 4.6, hint GEMM + switch 4.6, `y`
GEMM 1.2. The uploads would hide fully behind a second concurrent item (2 per device by default), which the
item-by-item model does not count. On the T4 the int8 GEMMs dominate, so the gain there is small.

**Kaggle T4** (`pgw-pks-r3-t4`, torch 2.11 + cu130, Triton 3.6, 4 vCPUs; one kernel, ~3 min of steps; raw files in
`results/kaggle/pks-r3-t4`). Pushed after the other agent's draft-head kernel had finished.
* **Every item byte-identical** across `torch` (every kernel plain, host wire, host keys), `r2` and `fused` (every
  device step under `auto`: `fused_status` = fused NTT, switch, ring rows, **wire**, **keys**; `fwd4` off as by
  default) at 7B gate_up / down / head, 32B gate_up / down, 72B gate_up / down. **This is the first GPU run of
  the device wire pack / unpack** (`PGW_SERVICE_PKS_GPU_WIRE`): its self-test passed and `auto` chose it, and the
  CUDA pytest wire tests passed (device pack and host block pack = `pack_bits`, device unpack = `unpack_bits`,
  the item with wire bytes in and out at a shard offset = the host path).
* Reply keys: device unpack = `unpack_ksk` with edges; every out-of-range residue (p and 2^bits - 1 in each prime)
  flagged where the host refuses; `auto` chose it after the self-test. Per 4-group item: host unpack 10.1 ms
  (11.9 with its uint32 upload) against 1.8 ms on the device (3.2 with the upload of the wire bytes), and the
  device time is asynchronous: the host thread no longer spends it.
* Staged uploads (bytes identical): 3.65 -> 6.7 GB/s at 72B gate_up's 59 MB upload, 3.67 -> 9.15 GB/s at 72B
  down's 212 MB (raw pinned transfer 12.3 GB/s, host copy one thread 5.4, pool 8.5-9.8).
* Whole items, median of 3 (s): 7B gate_up 0.753 -> 0.740, down 0.281 -> 0.276, head 3.16 -> 3.18; 32B gate_up
  1.456 -> 1.460, down 0.649 -> 0.637; 72B gate_up 1.991 -> 2.017, down 1.013 -> 1.014: equal within +-2 %, as
  the T4 model says (GEMM-bound). Against the previous T4 run's items (host wire): 7B gate_up 1.20 -> 0.74,
  7B head 4.69 -> 3.18, 72B gate_up 2.54 -> 2.0: mostly the device wire forms (that run also used the 4-prime
  forward NTT, which `auto` no longer picks), measured for the first time.
* pytest on CUDA (`test_pks_keys_upload`, `test_pks_fused`, `test_pks_ntt`, `test_pks_device_item`, not slow):
  103 passed, 2 skipped, **1 failed: a test bug** (`test_cuda_staged_upload_and_keys` built its uint64 array from
  too few random bytes after its staged-upload assertion had passed); fixed, not rerun on the GPU. The rest of
  that test (device keys on CUDA) is covered by the bench's `round3` and by `test_cuda_item_packed_keys`, which
  passed (including the refusal of an out-of-range key through the pinned flag, with the device path kept).
* The bench's `exact_new` reported `all: false` only because it required every `fused_status` entry to be fused,
  including the 4-prime forward NTT that is off by default; every exactness value in it was true. Fixed in the
  bench.
* Model gap: the T4 model now under-predicts whole items by ~1.6-1.9x on the device-wire path (7B gate_up: model
  0.41 s, measured 0.74 s), where it was within 30 % with the host wire. The before/after ratios above are not
  affected (both columns share the gap), but the absolute H100 figures carry the same uncertainty until the RunPod
  session measures them.

Tests:
* New: `tests/service/test_pks_keys_upload.py` (31 CPU tests, 3 CUDA tests skipped here; on the T4: 2 of 3 CUDA tests
  passed, the third had the test bug above): the device unpack equals
  `unpack_ksk` (and the customer's `pack_ksk` bytes); out-of-range residues flagged for every prime and group; the
  self-test detects a flipped bit, a missing flag and a false flag; `auto` / `torch` / `fused` and the permanent
  fallback; the item with the keys' wire bytes (device and host unpack) equals the host path; an out-of-range key
  is refused before any key switch under `torch`, `fused` and `auto` without falling back; through `PKSProvider`
  the device path returns the cache path's bytes, the customer decrypts them, and bad keys / a wrong size are
  refused on both paths with the device path kept; `staged_copy` order, exactness and error handling; the memplan
  stages.
* `tests/service` (one directory, `-n 2`): 579 passed, 88 skipped (CUDA), 2 failed before the fix of one assertion
  (`fused_status` gained `keys`); the other, `test_lane_persistent.py::test_lane_persistent_is_bit_identical[3]`,
  fails without the native lane library in this worktree (no pks code involved). `tests/benchmarks/
  test_provider_perf_tools.py` passes; `tests/deploy` 140 passed, 1 failed
  (`test_lane_nic_session_two_pods_on_cpu`: the lane NIC bench, same cause); `tests/docs` only the 18
  environment-only `test_numbers` failures known from the integration sweep. `tests/hemask`: 439 passed, 20 skipped.

New setting: `PGW_SERVICE_PKS_GPU_KEYS` (default `auto`). Bench: `benchmarks/pks_ntt_bench.py` gained `--sections`,
`--configs`, `--item-reps`, the `round3` section and the `r2` item configuration; RunPod `maskgen` phase gained `r2`.

Cloud spend: one Kaggle T4 kernel (~10 min with set-up). No other compute.

## 2026-11-26: client KV prefix cache (exact with the masked LM head; opt-in)

Repeated prompt prefixes (a conversation's earlier turns, a shared system prompt, a regenerated answer) are no
longer recomputed when `prefix_cache=True`: the cached positions take no rows on the prompt pass's `4L + 1` trips
and no masks. Design and privacy analysis: `research/reviews/prefix-cache.md`.

* **What changed.** The existing conversation cache (SPEC 9.4b) was refused with the masked LM head, which is the
  SDK default wherever the provider serves it, so the product never reused anything. The masked head is row-exact
  (each logit row is its own row normed alone, quantized per row, an exact integer product): `MaskedHead.rows_exact`
  and `Generator.prefix_reuse_supported` now accept it, and `prefix_cache=True` keeps the masked head (it used to
  turn it off). The SDK's runtime gets a new cache, `pgw.sdk.prefix_cache.ClientPrefixCache` (via
  `ModelRuntime.prefix_cache_factory`): several entries per scope (OpenAI `user`, or the whole client with
  `prefix_cache_scope="client"`), the longest exact token prefix wins, a whole entry is moved and part of one is
  copied, an entry taken by a request that never started (waiting for masks, cancelled) goes back (the gateway's
  cache lost it), entries the new one subsumes are dropped (including the previous turn, whose answer the chat
  template re-tokenizes in its last tokens: 115 of 119 cached positions matched at 0.5B), and a new conversation
  waits up to 0.25 s for a just-finished one of its scope to store the shared system prompt. RAM only (no files),
  `prefix_cache_mb` (1024) LRU budget, `prefix_cache_ttl_s` (600), wiped on integrity failure, noise alarm,
  reload and close; stored only after a clean end with the deferred checks joined. Multi-process sharing: no
  (plaintext KV would need shared memory or disk); route conversations to one worker.
* **Privacy verdict: it leaks something new**, so it stays opt-in with the masked head. The wire changes only in
  the prompt pass carrying `P - m` rows instead of `P`; every row is still masked once. But a short prompt pass
  followed by decode passes over a long context tells the provider *that* a request continued a cached prefix
  (a per-request hit bit, and roughly the context length), which links turns more surely than today's row-count
  arithmetic. Padding the prompt trips to hide hits was analysed and not built: padding rows take no masks, so
  the factory's mask demand (which the provider performs) still shows the aggregate hit volume unless the masks
  are burned (which removes the saving), the client's per-trip time still shrinks, and trips, bandwidth and
  provider work would be unchanged. `"auto"` is unchanged (on only without the masked head, as before); making it
  mean off is recommended as a product decision. `prefix_cache_scope="client"` is opt-in separately (a user could
  time whether another user's prompt shares a prefix).
* **Measured** (`benchmarks/sdk_prefix_cache_bench.py`; CPU provider on this 4-vCPU box, loopback TLS lane,
  masked head, plain masks; one conversation of a system prompt and 6 user turns, then a second conversation with
  the same system prompt and 2 turns; greedy, 16 tokens per answer at 0.5B, 8 at tiny; the mask stock full before
  each turn; every answer's tokens and logprobs bit-identical between the two clients):

  | | 0.5B off | 0.5B on | tiny off | tiny on |
  |---|---|---|---|---|
  | mask slots, 8 requests | 1,492 | **358** (-76 %) | 525 | **206** (-61 %) |
  | prompt-pass rows, 8 requests | 1,372 | **238** (-83 %) | 483 | 164 |
  | TTFT, median of turns 2-8 | 5.94 s | **1.44 s** | 0.108 s | 0.078 s |
  | TTFT, turn 6 (270-token prompt, 251 cached) | 8.35 s | **1.81 s** | | |
  | TTFT, 2nd conversation's 1st turn (91 of 108 cached: the system prompt) | 3.29 s | **1.43 s** | | |

  At 0.5B a continued turn's prompt pass carries 16-20 rows instead of 139-270, and its TTFT stays flat as the
  conversation grows instead of growing with it. The tiny model is trip-bound (TTFT within noise). First turns
  (misses) cost the same (3.7-5.0 s here, noise of the shared CPU). Not measured on a GPU provider or at 7B
  (expected: the prompt-pass share of TTFT, 3.9 s at 7B on the T4, shrinks with the cached fraction; every cached
  token saves one mask slot, 34 ms of factory CPU at 7B).
* **Tests.** New: `tests/engine/test_prefix_cache_masked_head.py` (6: masked head + reuse == recompute == the
  plaintext int8-head reference, greedy and seeded; one head row per pass hit or not; masks and rows only for the
  new positions; a system prompt shared by two conversations; strict speculation with filler rows and with
  padded passes), `tests/sdk/test_prefix_cache.py` (10: longest match, move vs copy, give-back and epoch, claimed
  tokens zeroed, subsumption and the re-tokenized tail, scope isolation and key aliasing, ties and `min_tokens`,
  budget / entries / TTL, the scope wait). `tests/sdk/test_e2e_cpu.py::test_conversation_prefix_reuse_is_exact`
  now runs with the masked head and with the gateway head (turn 2 cached, a second conversation reuses the system
  prompt, another `user` reuses nothing; all bit-identical to the gateway path). `tests/sdk/test_factory_stall.py`
  pins `prefix_cache=False` (it sizes requests against the stock by their whole prompt; a cached repeat needs one
  prompt row and was admitted). Runs: `tests/sdk` 1 opt-in skip and 3 `test_lane_multi` failures (the installed
  `pgw_fast` lacks lane-format-2 multi calls, as in the integration sweep), everything else passed;
  `tests/engine/test_prefix_cache.py`, `test_masked_head*.py`, `tests/gateway/test_prefix_cache.py` passed;
  `tests/docs` only the 18 environment-only `test_numbers` failures.
* **Engine footprint** (another workstream is in the client hot loops): `generate.py` one predicate,
  `masked_head.py` one flag, `runtime.py` a 3-line builder hook; everything else in the new module.

Docs: `docs/sdk/configuration.md` (options and a "Prefix cache" section), `scaling.md`, `security.md`,
`changelog.md`, `CONTRACT.md`, SPEC 9.4b. No external compute.


## 2026-11-26: trained draft heads for strict speculation in the client library (0.5B, 7B; shipped, pinned)

Goal: `speculation="draft_head"` that pays off on ordinary chat, not only on copy-heavy text.

- **7B export bug, fixed** (`benchmarks/feature_head_train.py`, `FeatureHeadDrafter`): the exporter folded the
  input scales (1 / feature std per dimension, 1 / embedding rms) into `W_in` before int8 per-row quantization.
  One scale over both halves (the first Modal export) rounded the feature half to 0 (alpha 0.19); the per-half
  fix of 2026-11-23 still rounded the outlier feature dimensions (the folded columns span 100-1000x in a row).
  Format `pgw-feature-head-v2` stores `W_in` in the trained input space plus `in_scale`; the loader folds it for
  the float kernel and scales the input for the native int8 kernel; v1 files still load. Test
  (`tests/benchmarks/test_feature_head_export.py`, 7B-like feature statistics): exported head vs float head top-1
  agreement 98% (float and native kernels) vs 86% (folded per half) and 48% (one scale).
- **Training on Kaggle** (`benchmarks/feature_head_distill.py`, job builder `benchmarks/kaggle/build_head.py`,
  T4 x 2, one kernel at a time): prompts from UltraChat 200k, Alpaca and Dolly (pinned dataset revisions); the
  target answers them itself (greedy, 25% sampled at T = 0.7, no repetition penalty) and forward hooks capture the
  final-norm feature and the raw arg-max at every generated position (checked equal to a teacher-forced pass);
  features kept as int8 in host RAM; held out first: 179 UltraChat `test_sft` prompts (open chat) and 120 Alpaca
  code prompts. 0.5B: 2.58M rows (66 min). 7B (fp16 split over both T4s): 1.20M rows (generation ran at ~220
  tok/s, so data limited it; 109 min). Results: `benchmarks/results/feature_head/kaggle_{05b,7b}/`.
- **Head + prompt lookup**: `FeatureHeadDrafter(lookup=...)` runs prompt lookup (fixed-work search) on every pass
  and proposes its continuation when it fills the whole chain; the head runs either way. The SDK's
  `draft_head` mode turns it on (n-grams 4..2). `kernel="auto"` keeps the native int8 kernel only when timed
  faster at load (float32 is faster on this box: 0.88 vs 1.3 ms per 7B head step).

Alpha (depth-1 top-1 of the exported head on held-out greedy answers) and tokens per pass (strict chain, R - 1
drafts; replay = exact replay of the target's own answers through the acceptance rule, `feature_head_hybrid_sim.py`):

| model, head | alpha chat / code | tokens per pass, open chat, R = 2 / 3 / 4 | code, R = 2 / 3 / 4 |
|---|---|---|---|
| 0.5B d 256 V 16384 (5.3 MB, shipped), head alone | 0.48 / 0.49 | 1.46 / 1.65 / 1.72 | 1.46 / 1.66 / 1.74 |
| same + lookup (replay) | | 1.52 / 1.78 / 1.91 | 1.50 / 1.74 / 1.86 |
| same + lookup, **end to end through `pgw.OpenAI`** (CPU provider) | | 1.51 / 1.77 / 1.95 | 1.50 / 1.74 / 1.86 |
| 0.5B d 384 V 32768 (14.5 MB), head alone | 0.50 / 0.53 | 1.48 / 1.69 / 1.78 | 1.50 / 1.72 / 1.82 |
| 7B d 384 V 16384 (11.2 MB, shipped), head alone | 0.47 / 0.51 | 1.45 / 1.62 / 1.68 | 1.48 / 1.66 / 1.73 |
| same + lookup (replay) | | 1.46 / 1.66 / 1.74 | 1.48 / 1.69 / 1.77 |
| same + lookup, **end to end on a Kaggle T4** (5 prompts, 48 tokens) | | 1.53 / 1.69 / - | 1.59 / 1.81 / - |
| 7B d 512 V 32768 (23.6 MB), head alone | 0.48 / 0.52 | 1.46 / 1.63 / 1.70 | 1.49 / 1.68 / 1.75 |

End-to-end speed (`benchmarks/sdk_spec_bench.py`, greedy, streamed; every answer's tokens and logprobs identical to
`off`):

- 0.5B on this 4-vCPU box (real pack, CPU provider, 7 chat + 5 code prompts, 96 tokens): open chat 3.22 tok/s off
  -> 3.81 / 4.09 / 4.27 at R = 2 / 3 / 4 (+18 / +27 / +32%); code 3.23 -> 3.82 / 3.96 / 4.11; lookup alone 3.07
  (slower); the code-edit prompt 3.21 -> 4.90 (R = 3; lookup alone 4.99).
- 7B on one Kaggle T4 (provider on cuda, client and plain mask factory on the same 4 vCPUs): 5.38 tok/s off ->
  3.93 (R = 2) / 2.03 (R = 3). On that machine each extra row costs the T4 (no int8 tensor-core GEMV) and the
  factory competes for the CPUs, so it confirms exactness and tokens per pass, not deployment speed.
- Model (`round_trip_model.py --sdk`, now with the measured tokens per pass and R - 1 head steps per pass), 7B,
  H100 provider, RTT 0.2 ms: off 16.6 tok/s, draft head 19.6 / 18.7 / 16.9 at R = 2 / 3 / 4; 32B 6.5 -> 7.7
  (R = 2), 72B 5.0 -> 5.7 (32B/72B with the 7B tau: no head trained there).

*Superseded by the next section: these heads were trained on Alpaca (non-commercial) and Dolly prompts; they are
replaced by `-r2` heads trained without them.* Shipping: the heads are package data (`src/pgw/sdk/draft_heads/*.pgwhead`, 16.5 MB together; in the repo, not
LFS: small, and a pip install from git keeps working), pinned in `pgw.sdk.draft_heads.PINNED_DRAFT_HEADS` (file,
SHA-256, size, target checkpoint = the SDK's model pin). `speculation="draft_head"` without `draft_head_path`
resolves the pinned head: the shipped file (a hash mismatch refuses), else a cache copy, else a mirror
(`PGW_DRAFT_HEAD_URL`), fetched and checked before it is moved into the cache. The head runs on the client's CPU
only; nothing about it is sent to the provider. Docs: `docs/sdk/configuration.md`, "Speculative decoding: when to
enable it".

**Default unchanged (off).** Open chat is faster end to end at 0.5B on this box (+27% at R = 3) and in the 7B model
(+18% at R = 2), but speculation multiplies mask use and per-round traffic by `spec_rows`: where the mask factory
or the link is the limit (7B `pks-v3` on a small machine; the T4 run) it is a loss, and it orders a user's
requests behind the previous answer's dummy rounds. Recommendation: `draft_head` with `spec_rows=2` at 7B and 3 at
0.5B for interactive use when the mask stock keeps up.

Tests: export (4), shipped heads and resolution (7: pins, tampered file, mirror fetch, bad mirror, engine wiring),
head + lookup strict exactness (greedy and seeded; head steps unchanged), gateway config, round-trip model;
`tests/sdk` 165 passed (S4/S20 tests included) with `test_lane_multi.py` deselected (3 failures there that the
base commit has too); `tests/engine/test_tree_spec.py` 134, `tests/gateway/test_speed_wiring.py` 68,
`tests/benchmarks` 49. tests/docs: the `PGW_SPECULATION` failure is fixed; `PGW_MASK_ADMISSION` (not this change)
and the README index test still fail as before.

Open: alpha ~0.48 is the MLP head's level (EAGLE's ~0.8 needs attention over past features); more 7B data needs a
faster generator than HF generate on T4s. The training prompts include Alpaca (CC BY-NC 4.0): retrain without it
(`--mix 0.8,0,0.2`) if that licence matters for the shipped heads. Kaggle: the 7B head kernel started 3 s after
another agent's CPU pin kernel (both checked idle before pushing).

## 2026-11-26: client CPU at many concurrent streams (several-answer trips, sampling; exact)

Scope: item 3 of docs/sdk/scaling.md ("gateway CPU, one core per process"): at tiny-16 the client's plaintext thread
was the limit after lane format 2. Profiled with `benchmarks/sdk_concurrency_bench.py`, `benchmarks/sdk_trip_profile.py`,
py-spy (`--gil`, `--native`) and per-function timers, tiny and 0.5B at N = 1/4/8/16 (4-vCPU box shared with other
jobs, CPU provider on the same box).

**Profile (before).** tiny N=16, per trip (7.5 ms wall, 22 rows): mask prepare 2.2 ms wall / 1.9 ms CPU (the batch
path's Python per item: `_take`, soundness and noise checks, one `trip_prepare_into` per item), lane + provider
3.2 ms, finish 0.5, model step 0.5. The plaintext thread held the GIL only ~18 % of the time: a pass of more than
8 rows ran as split hops, four thread hand-offs per trip (plaintext thread -> event loop -> executor thread for the
lane call -> event loop -> plaintext thread), so the limit was serial latency per trip, not GIL saturation.
0.5B N=8: plaintext thread 48 ms CPU per token (N=1: ~33): per answer and trip ~165 us prepare (W r chunk decrypt and
lease creation ~80, kernels 33-85, Python ~20), ~50 us finish; sampling 6.3 ms per answer per step, all on the
plaintext thread, of which `torch.multinomial` (one exponential draw + `log1p` per vocabulary entry) ~3.9 ms.
N=1 0.5B (native decode driver): prepare / finish / float ops 163 / 96 / 43 us per layer, lane I/O 67 us per trip,
sampling 5.5-6 ms per sampled token (temperature > 0, logprobs).

**Changes** (each exact; the SDK turns them on in `NATIVE_ENV`, the gateway keeps them off unless set):
- **Synchronous several-answer passes** (`PGW_SYNC_MULTI`; `MaskedBackend.sync_trip_ok`, `LaneComputeClient.multi_rows`):
  a decode pass of up to 64 rows that one lane multi call serves runs as one call on the plaintext thread over
  `forward_many`, as passes of <= 8 rows already did. Same calls, order, rows and requests as the split-hop path.
- **Leased several-answer trips** (`PGW_TRIP_NATIVE_MULTI`; `MaskedBackend._prepare_trip_leases`, `_finish_leases`):
  a trip of several one-chunk calls takes one native lease per answer instead of the batch path's Python takes.
  Shapes, finite values and every lease (window `ensure`, exhaustion and soundness checks) are resolved before any row
  is taken; then the Freivalds and noise-sampling draws in call order, one `pgw_fast.trip_prepare_many` (GIL released;
  items on the native pool above 32k elements), the noise monitor's every-call checks (or a sampled call's statistics)
  in call order before the request exists; after the reply one `trip_finish_many` (deferred checks submitted first,
  in order), statuses handled in item order. Same rows, payload bytes, item order, request-id format, checks and
  outputs. On a failure inside the batch, rows already taken for later items stay consumed (burned, never reused; as
  `_native_many`); a lease that cannot serve a call sends the whole trip down the batch path before any row is taken.
- **Parallel sampling** (`PGW_SAMPLE_THREADS`, SDK `auto` = cores, at most 8; `Generator._sample_parallel`): the
  answers of a step whose vocabulary has >= 4096 entries sample on a thread pool (no drafts, no fixed speculation
  rows). Each answer has its own sampler, generator and logits: same tokens, logprobs, alternatives, generator states.
- **Native multinomial** (`pgw_fast.torch_multinomial1`, `rust/pgw-fast/src/sampling.rs`, `sampling.multinomial1`):
  `torch.multinomial(p, 1, g)` reproduced exactly: torch's MT19937 state read from `g.get_state()`, the same
  `random64` draws and `u = (r & (2^53-1)) 2^-53`, `q = (float)(-log1p(-u))`, `argmax(p / q)` (first maximum, NaN
  wins); `log1p` runs only for elements whose bound `p / u` can reach the best exact score (1e-5 margin). The same
  index and generator state byte for byte; checked against `torch.multinomial` once per process before first use
  (`PGW_NATIVE_MULTINOMIAL=0` turns it off; inputs torch rejects go to torch). On by default (exact by test and by the
  start-up check). 152k entries: 3.9 -> 1.6 ms.

Privacy and integrity: what the provider sees per trip is unchanged (same trips, items, rows, bytes and request-id
format; a request alone has the same provider-metered trips and rows with the switches on and off, with and without
strict speculation: test). Masking, Freivalds (ring and p31), range checks, duplicate-row checks and the inline noise
statistics run per answer as before, all before the payload leaves; a wrong result in any item fails the trip.

Measured (`sdk_concurrency_bench.py`, streamed sampled chats, every answer bit-identical to the same request alone;
before = branch head `ffa52f9`, after = this change; two interleaved pairs, ranges):

| model | N | tokens/s before / after | client CPU ms/token before / after | plaintext thread ms/token |
|---|---|---|---|---|
| tiny | 1 | 44-46 / 45 | 3.8-4.1 / 3.9-4.2 | 2.5-2.8 / 2.7-2.8 |
| tiny | 4 | 117-119 / 128-130 | 4.0-4.1 / 2.7-2.8 | 2.8 / 2.2 |
| tiny | 16 | 215-223 / 279-282 | 3.2-3.3 / 1.9-2.1 | 1.8-1.9 / 1.4-1.6 |
| 0.5B | 1 | 2.8 / 2.9 | 65-72 / 69-70 | 44-49 / 46-47 |
| 0.5B | 4 | 5.5-5.7 / 6.2-6.3 | 79-88 / 68-72 | 51-55 / 43-46 |
| 0.5B | 16 | 7.3-8.1 / 9.3-10.1 | 64-74 / 55-59 | 38-42 / 32-35 |

(0.5B: 24 tokens per answer, so client CPU per token includes the prompt pass; the CPU provider on the same 4 vCPUs
saturates.) N=1 per sampled token (`Sampler.sample`, 152k vocabulary, one core, same tokens and logprobs): temperature
0.8 + logprobs 5.5-6.1 -> 3.1-3.3 ms CPU; with top-p 0.9 10.0-10.9 -> 7.5-8.8 (the top-p sort remains); greedy
unchanged (0.3). Same tokens and logprobs hash before/after (`sdk_trip_profile.py` 0.5B N=1 greedy; tiny N=1/4/16 and
0.5B N=4/8 sampled).

Not changed (measured): lease creation (W r chunk decrypt + ~90 us Python, every 16 rows per linear and answer) and
window extensions stay on the plaintext thread for several-answer trips (the single-answer driver has decrypt-ahead);
the top-p filter's full sort (the softmax denominator is summed over the whole sorted array: a partial sort changes
bits); the native decode driver still takes one answer (several answers run the Python per-trip loop with the batched
native calls above); the inline noise statistics stay in Python.

Tests: tests/engine/test_trip_native_multi.py 10 (multi == batch path == plain, payloads byte-identical in order, noise
statistics equal; sync / async / native pool / per-lease fallback; a tampered item fails; a non-finite value takes no
row; argument validation before any row), tests/engine/test_sampling_parallel.py 11 (native multinomial == torch for
6 sizes x 25 states incl. ties, zeros, fresh generators; rejects untouched; `Sampler` with / without native; parallel
== sequential sampling, async and `generate_many`; default off), tests/sdk/test_concurrency_cpu.py 2 (real provider:
answers alone and concurrent identical with the switches on/off, provider trips and rows of a request unchanged,
strict speculation unchanged, the new paths ran); `cargo test -p pgw-fast --lib` 87 (MT19937 reference outputs,
pruned == exhaustive race over 200 cases, rejects, state round trip). Directories, one at a time with the rebuilt
`pgw_fast` in `src/`: tests/native 357; tests/engine 1181 + 15 skips (one failure first: a test spy of
`Generator._consume` with two arguments; the call now passes the third only when it has one); tests/wire 334;
tests/gateway 283 + 1 crashed xdist worker (`test_source_dtype_gateway`, 2 passed alone); tests/service 600 + 80
CUDA skips; tests/sdk 188 + 1 opt-in skip (serial, 12 min); tests/docs: 18 failures in `test_numbers` (research
logs missing from the checkout, as before), the four new variables added to the documented-switch list. mypy `src`:
the same error set as the branch head.

Open: several-answer passes in the native decode driver (one native call per pass for all answers: the remaining
per-trip Python loop); lease creation ahead of need for several-answer trips (decrypt off the plaintext thread);
a GPU-provider run, where the client's per-trip work is the larger share of a trip.

## 2026-11-26: draft heads retrained on commercially usable prompt data only (0.5B, 7B; `-r2`, pinned)

Why: the heads shipped earlier today were distilled on prompts that included Stanford Alpaca (CC BY-NC 4.0,
non-commercial) and Dolly 15k (CC BY-SA 3.0). Their held-out code set was also Alpaca. pgw is a commercial
product, so both heads were retrained without that data. The old files are removed from the package (they remain in
git history only).

**Data** (Hugging Face Hub, pinned revisions; table, attribution and notes in `src/pgw/sdk/draft_heads/README.md`,
which ships next to the heads; each head file also records the list under `training_data`):
* training prompts: UltraChat 200k `train_sft` first user turns, 75% (MIT, `8049631c`); OpenAssistant OASST2 root
  prompts, all languages, 15% (Apache-2.0, `179dd21f`); StarCoder2 Self-OSS-Instruct instructions, code, 10%
  (ODC-By 1.0, `356bb069`);
* held out: chat = 200 UltraChat `test_sft` prompts (unchanged); code = 120 MBPP `full/test` problems (CC BY 4.0,
  `4bb6404f`; evaluation only).

Dolly was dropped as well, not kept. CC BY-SA allows commercial use, but whether trained weights are "adapted
material" is unsettled, and share-alike would then apply to the shipped files. OASST2 (Apache-2.0, human-written)
replaces it. The datasets give prompts only: the target answers them itself, and no dataset text is stored in a
head. UltraChat's turns were written by gpt-3.5-turbo (dataset MIT; recorded as a note).

**Runs** (Kaggle T4 x 2, one kernel at a time, our kernel list checked idle before each push; same sizes as before;
`benchmarks/kaggle/build_head.py`, which now trains only the shipped size and evaluates the old head on the same
held-out answers via `--compare-heads`, from a temporary private dataset):
* 0.5B d256 V16384: 2.80M rows from 9,094 answers, 57 min;
* 7B d384 V16384: 1.38M rows from 4,084 answers, 105 min.

Results: `benchmarks/results/feature_head/kaggle_{05b,7b}_r2/`.

**Old vs new on the same held-out answers.** Alpha is depth-1 top-1. Tokens per pass are head + prompt lookup
(the SDK's `draft_head` mode), from an exact replay at R = 2 / 3 / 4:

| head | alpha chat / code | tokens per pass, chat | code |
|---|---|---|---|
| 0.5B old (removed) | 0.477 / 0.464 | 1.52 / 1.78 / 1.91 | 1.49 / 1.75 / 1.85 |
| 0.5B new, shipped | 0.472 / 0.559 | 1.51 / 1.77 / 1.89 | 1.56 / 1.87 / 2.03 |
| 7B old (removed) | 0.474 / 0.506 | 1.46 / 1.66 / 1.74 | 1.51 / 1.73 / 1.82 |
| 7B new, shipped | 0.467 / 0.635 | 1.45 / 1.64 / 1.72 | 1.61 / 1.94 / 2.12 |

Accepted. Open chat is within 0.01 alpha of the old heads (criterion: 0.03). Code is +0.09 / +0.13 alpha, because
code instructions are now in training. The old 0.5B head's chat numbers match those of the earlier run exactly (same
held-out answers), so the comparison is like for like.

**End to end, 0.5B** (`sdk_spec_bench.py`, CPU provider on this 4-vCPU box, pinned head, 6 chat + 4 MBPP + 2 fixed
prompts, 96 tokens). Every answer's tokens and logprobs are identical to `off` in all configurations.
* Tokens per round: chat 1.52 / 1.82 / 1.95, code 1.62 / 1.92 / 2.15 at R = 2 / 3 / 4.
* Decode speed: chat 3.04 tok/s off -> 3.97 / 4.46 / 4.20; code 3.00 -> 4.26 / 4.76 / 4.48.
* `lookup:3`: -9% (chat), -4% (code).
* The code-edit prompt at R = 3: 2.16 tokens per round, +81%.

The 7B end-to-end T4 row in the docs is still the earlier head's measurement (same size; it was a tokens-per-round
and exactness check, not a speed figure). `round_trip_model.py` uses the new 7B replay tau. Modelled 7B (H100, RTT
0.2 ms): 16.6 -> 19.5 / 18.5 / 16.8 tok/s at R = 2 / 3 / 4.

**Pins** (`PINNED_DRAFT_HEADS`; new file names, so that a mirror or cache holding an old file cannot collide):
* `qwen2.5-0.5b-instruct-d256-v16384-r2.pgwhead`: sha256 `fbb4ed1c0bece068c61cd69a56afe4d0956877ec4de45c925eab11c6096247c2`, 5,289,977 bytes;
* `qwen2.5-7b-instruct-d384-v16384-r2.pgwhead`: sha256 `c3ed147cc58483f8d65ce6e8eaa229f9cd0c9e71119c9a4310c801b0f9b27b56`, 11,191,929 bytes.

The README is added to the package data.

**Tests:** `tests/sdk/test_draft_heads.py` (pins, shipped files load and match the model), `tests/sdk/test_speculation.py`
(strict exactness vs off), `tests/engine/test_tree_spec.py` and `tests/benchmarks`: 196 passed, the same
count as before the change. `tests/docs`: 112 passed; the 18 `test_numbers` failures need research logs that are not in
this worktree (`results/kaggle/...`), and the change does not touch them.

**Kaggle cleanup:**
* deleted: `pgw-spec-7b-t4-src` and the temporary `pgw-head-cmp-tmp`;
* left alone: `pgw-sdk-e2e-7b-src` (another workstream's);
* the private kernels `pgw-head-05b` and `pgw-head-7b` keep their version-1 outputs (the old heads) in their
  private version history.

a GPU-provider run, where the client's per-trip work is the larger share of a trip. *(Since then: the first two
done, next section.)*

## 2026-11-26: client CPU at many concurrent streams, round 2 (several answers in the native decode driver, leases prepared ahead; exact)

Scope: the two open points of the previous section. Profiled first (`benchmarks/sdk_concurrency_bench.py`, which now
also reports generator steps, native driver calls and trip-lease counts; py-spy of the plaintext thread during a
level; `benchmarks/sdk_trip_profile.py`), tiny and 0.5B at N = 1/4/16, 4-vCPU box shared with other jobs, CPU
provider on the same box.

**Profile (before).** Plaintext-thread CPU samples outside the lane wait, per several-answer trip: tiny N=16 lease
creation 28 % (`trip_lease`: `W r` chunks, `ensure`, the ahead reservation's thread start, `TripLease`), noise monitor
26 % (statistics of the sampled answers: with 16 answers about half the trips carry one; their duplicate check in
Python), the Python loop per trip 25 % (generator step and float-op calls of `_layer_native`, `_prepare_trip_leases`,
the lane client's multi plumbing, `_finish_leases`), the batched native prepare 15 %. 0.5B N=16 (3.5 ms plaintext CPU
per trip of 16 answers): leases 25 %, noise 25 %, native prepare 21 %, Python loop 16 %. At N=1 the native driver
already ran every decode pass. The provider limits tokens/s here (0.5B: its kernels hold the GIL, about one core;
tiny: its per-trip latency), so the measure is client CPU per token; tokens/s moves where the client's per-trip time
is on the critical path (tiny, and every model against a GPU provider).

**Changes** (both exact, on in the SDK's `NATIVE_ENV`, off in the gateway unless set):
- **Several answers per pass in the native decode driver** (`PGW_DECODE_NATIVE_MULTI`; `DecodeDriver.begin_multi /
  run_multi / end_multi / current_inputs`, `NativeDecode.eligible_multi / run_multi`): a decode step of several
  answers (each <= 8 rows on its own reservation, the leased several-answer trips on, one lane multi call able to
  carry every row) runs its decoder layers in one native call per stretch of trips, GIL released. Per trip: the
  previous trip's deferred checks joined; every answer's prepare on its own lease (native pool above 32k elements,
  as `trip_prepare_many`; a non-finite value in any answer takes no row of any answer); the duplicate-row checks in
  answer order before anything leaves; one lane format-2 multi call with the answers' items in answer order
  (`TripChannel.trip_multi_raw`); every answer's finish (range check, ring Freivalds synchronous or deferred on the
  native helper, p31 synchronous; unmask, rescale); the float ops of each answer on its own KV cache. Python runs once
  per stretch: every answer's lease resolved before any draw (window `ensure`, exhaustion and soundness checks), the
  Freivalds / noise-sampling draws in trip order and answer order (each once), request ids. A trip with a sampled
  answer ends the call and is held after its prepare: the statistics of its sampled answers run in answer order before
  its payloads leave and before the next trip's draws (an alarm stops it; the pass is aborted and the held payloads
  zeroed). A trip the lane does not serve is sent with the same request (items, payloads, id) on the ordinary
  fallback and finished by `_finish_leases`; a trip with no lease for some answer runs on the ordinary path. Same
  requests, items, rows, payload bytes, checks, statistics and outputs as the per-trip several-answer path. One
  difference, on a failure path only: when the statistics of a sampled answer alarm, the duplicate-row history
  already holds the fingerprints of the later answers of that trip (the per-trip path stops before recording them);
  the trip fails closed either way and nothing of it leaves.
- **Trip leases prepared ahead** (`PGW_TRIP_LEASE_AHEAD`; `MaskReservation.lease_ahead`,
  `WindowedReservation.lease_ahead`, `MaskedBackend._lease_ahead_submit`): for several-answer trips, a new lease starts
  building the next one on a `pgw-lease-ahead` thread (the `W r` chunks decrypted ahead, waited for without the store
  lock; the extent's soundness check, without recording; the native `TripLease`); with the SDK's 8-slot incremental
  parts, the first lease of a part the background extension adds is prepared as the part arrives. Single use and
  accounting: a prepared lease covers rows the session already holds (nothing reserved, taken or counted: admission
  and the ledger unchanged, no `ensure`, no extension), hands out no row until `trip_lease` attaches it at exactly
  the cursor's row under the same soundness policy, and is closed (key zeroized, chunks zeroed) when another path
  takes the rows, on another row or policy, and at close (request end, cancel); the next lease's chunks are decrypted
  ahead only once it is attached (at most as much `W r` plaintext per linear as before). Single-answer trips prepare
  nothing ahead (measured: at N=1 it only added CPU on the other thread).

Privacy and integrity: per trip the provider sees exactly what it saw (the same request: items in answer order, rows,
payload bytes, uuid4-format id, the same lane multi frame); how the generator batches answers into steps is timing,
as before. Masking, Freivalds (ring and p31), range, duplicate-row and noise checks run per answer before anything of
a trip leaves. Secrets: the driver's pass buffers and held payloads are zeroed at the pass end, on abort and on every
error; tickets wiped; prepared leases zeroed when unused.

Measured (`sdk_concurrency_bench.py`, streamed sampled chats, every answer bit-identical to the same request alone;
before = branch head `ee9f0a0` built here, after = this branch; two interleaved pairs, ranges; tiny 256 tokens per
answer, 0.5B 32):

| model | N | tokens/s before / after | client CPU ms/token before / after | plaintext thread ms/token before / after |
|---|---|---|---|---|
| tiny | 1 | 45-47 / 44-47 | 3.4-3.7 / 3.4-3.9 | 2.6-2.7 / 2.5-2.8 |
| tiny | 4 | 123-126 / 140-147 | 2.9 / 1.9-2.1 | 2.3-2.4 / 1.3-1.4 |
| tiny | 16 | 279-290 / 337-360 | 2.0-2.1 / 1.36-1.47 | 1.55-1.64 / 0.86-0.95 |
| 0.5B | 1 | 2.7-2.9 / 2.9-3.0 | 64-67 / 59-66 | 44-45 / 40-46 |
| 0.5B | 4 | 6.4-6.6 / 6.9-7.7 | 62-65 / 49-54 | 41-43 / 28-31 |
| 0.5B | 16 | 10.4-11.0 / 11.9-12.4 | 50-52 / 44-46 | 29-31 / 22-24 |

Each part alone (A/B by switch, same build): at tiny N=16 the driver takes the plaintext thread from 1.36 to 0.81 ms
per token; lease ahead moves 1.5-2 ms of CPU per token at 0.5B N=16 off the plaintext thread onto its own (total
about equal: 3,900 of 9,200 prepared leases were never attached, mostly the next part's leases at request end with
32-token answers), and 83 % of the leases of several-answer trips at tiny N=16 come from ahead. A first version that
also prepared leases for single-answer trips cost 2-5 ms of CPU per token at 0.5B N=1 for nothing: dropped.
`sdk_trip_profile.py` (0.5B, N=1, greedy): the same logprobs hash before / after (`e2231950bdf2ed58`).

Profile (after, tiny N=16): the plaintext thread spends 75 % of its samples (lane wait included) inside the driver's
native call; the Python left: the noise statistics of sampled answers (9 %), lease resolution and window bookkeeping
(11 %: `ensure`, the ahead reservation's thread start), and the masked LM head's trip, still on the per-trip path
(16 %).

Tests: tests/engine/test_decode_native_multi.py 12 (driver == plain bit for bit and == the per-trip several-answer
path: payloads in order, requests (id format, items), noise statistics, every check counter; ring and p31, one and
all layers per call; a wrong result in any answer's item fails, synchronous and deferred; a sampled answer's alarm
stops its trip exactly where the per-trip path stops; a non-finite answer takes no row and sends nothing; native
argument checks and wipes), tests/engine/test_lease_ahead.py 11 (exact and used, ring / p31 / incremental parts, also
racing the plaintext thread; nothing handed out before attach; closed when another path takes the rows, on another
row or policy, at close; a soundness failure ahead is raised by the lease path; reserved slots, extensions and taken
rows unchanged; single-answer trips prepare nothing), tests/sdk/test_concurrency_cpu.py 2 (real provider: answers
alone and concurrent identical with all switches on / off, provider-metered trips and rows of a request alone and
rows of the concurrent runs unchanged, strict speculation unchanged, the driver's several-answer passes ran over lane
multi calls, leases came from ahead); `cargo test -p pgw-fast --lib` 87. Directories, one at a time with the rebuilt
`pgw_fast` in `src/`: tests/engine 1212 + 15 skips; tests/sdk 206 + 1 opt-in skip (serial, 12 min; an earlier full
run had one failure, `test_factory_stall.py::test_provider_dies_in_steady_state`, a timing bound under load: the
file passed twice alone, and the final run passed); tests/maskstore 247; tests/wire 334; tests/native 357; tests/service 600 + 80 CUDA skips;
tests/gateway 282 + 1 crashed xdist worker (`test_source_dtype_gateway`, 2 passed alone, as before); tests/docs: the
18 `test_numbers` failures of before (research logs missing from the checkout), the two new variables added to the
documented-switch list.

Open: the masked LM head's trip of a several-answer step (the per-trip Python path, 16 % of the plaintext thread at
tiny N=16) could join the driver; the noise statistics of sampled answers stay in Python (about one trip in two at
16 answers, each ending a driver call); leases prepared for the next part at request end are wasted work on the
lease-ahead thread (about 40 % of those prepared with 32-token answers); a GPU-provider run, where the client's
per-trip time is the larger share of a trip and these changes show as tokens/s.


## 2026-11-26: client mask CPU per slot at 32B-72B, round 3 (request and reply in place, AVX-512 keystream, buffer reuse; exact)

Goal: less client CPU per mask slot at 32B / 72B shapes, where the factory's CPU caps the users one client machine
serves; no change to the scheme, its parameters, single use, fresh randomness, the check material (p31 `u W8`, the
40-bit private self-check) or wiping. Details, per-part table and what was tried: `docs/sdk/scaling.md`, section
"Round 3: wire, keystream, buffers".

**Profile first.** `benchmarks/pks_slot_breakdown.py` now also times what `RemotePKS` does around the post (numpy
`pack_bits` of the 56-bit upload and the msgpack request; the reply's msgpack decode and copy) and touches memory once
before measuring (this VM charges ~17 us for the first touch of a page, ~20x a normal fault). New
`benchmarks/pks_factory_e2e.py` runs the real chain (`MaskGenerator` + `PKSMaskBackend` + `RemotePKS` + `MaskStore`,
one decoder layer of real shapes, synthetic weights and hint, a stand-in provider whose CPU is subtracted); `pks-bench
stages / freq / ifma` time the IFMA kernels part by part. One core (`taskset`), 1024-slot segments. Before (branch
head `923d078`), ms per slot, 7B / 32B / 72B: `r` 5.8 / 19.6 / 31.1; encryption + reply keys 15.6 / 55.4 / 82.6;
**upload wire form 14.3 / 54.3 / 82.1** and **reply decode 3.5 / 11.0 / 19.0**, both missing from the earlier
per-slot figure (33.5 / 103 / 157 "scaled"); decryption 9.5 / 30.4 / 45.0; self-check 3.1 / 10.2 / 14.9; `u W8`
1.2 / 7.8 / 9.2; writer 3.0 / 8.9 / 13.3; total **56.4 / 199.3 / 299.2**. Underneath: the `r` keystream at 4-8 ns per
element (the `chacha20` crate, a fresh array per chunk) against 1.6 for AVX-512 ChaCha20, and a fresh, page-faulted
buffer for every large array (~0.16 ns per byte on a warm box, much more on first touch here).

**Changes** (default on; each switch restores the old path):
- `r`: `pgw_fast`'s ring stream runs a 16-block AVX-512 ChaCha20 with the ring mask in the store (factory and the
  gateway's mask fetch; same bytes; `PGW_FAST_CHACHA512=0`); the factory fills each chunk's `r` into a zeroed buffer
  of an earlier chunk (`mask_rows_into`, `MaskGenerator`'s `_RBuffers`: one per shape, at most two chunks of the
  widest linear; `PGW_FACTORY_R_REUSE=0`).
- Upload in place: `pgw_pks.encrypt_wire` writes the bodies' bit planes and the reply keys' wire form straight into the
  request `bytes` around the msgpack head and middle (`wire.maskgen_v3_frame`, `RemotePKS.frame`/`send`,
  `Customer.encrypt_wire`): no `b` array, no `pack_bits`, no message copy; the reply keys are computed before the
  upload (independent streams). Same request bytes for the same seeds and id (`PGW_PKS_WIRE_FUSED=0`).
- Reply in place: `wire.decode_maskgen_response_view` walks the message, decodes everything but the `y` payloads with
  the same code and checks, and hands the decryption a read-only view into the response body (anything it does not
  handle takes the copying decode); the native decryption reads any byte buffer.
- Decryption output reused: the segment writer hands back each zeroed output (`PKSMaskBackend.release_output`), the next
  call of that shape decrypts into it (`decrypt_into`; several passes write their rows into one output); one per
  shape, 1 GiB (`PGW_PKS_OUT_REUSE=0`).

**After** (same measurements, ms per slot, 7B / 32B / 72B): `r` 1.6 / 5.2 / 8.1; encryption + keys + upload wire in
one call 17.9 / 55.4 / 83.5 (were 30.4 / 111.3 / 166.7); reply decode 0.0; decryption 7.7 / 25.8 / 36.9; the rest
unchanged; **total 34.8 / 112.0 / 165.8 (-38 / -44 / -45 %)**, i.e. **28.7 / 8.9 / 6.0 slots/s per client core** (were
17.7 / 5.0 / 3.3). End to end (`pks_factory_e2e.py`, decoder layers, two runs each but the first 7B one): 68.1 -> 39.5 / 227-238 ->
127-137 / 379-410 -> 220-240 ms per slot, 25 / 7.3-7.9 / 4.2-4.5 slots/s per core (were 14.7 / 4.2-4.4 / 2.4-2.6).
Gateway open + retirement unchanged (3.9 / 13.1-13.6 / 17.4-17.7).

**Tried, measured no gain on this box, not kept:** pad primes of 36 bits with fully lazy NTT butterflies (8 instead of
10-12 vector operations; bit-identical, tested at the range extremes), radix-4 passes of the inverse NTTs (half the
loads and stores), other L1 block sizes and prefetching for the cross-block NTT levels, an IFMA copy of the self-check
(also L2-tiled). The HE kernels stay as they were (encryption ~15 ns per input value and slot at 4 groups, decryption
~5 ns per output value and slot); without hardware counters on this VM the bound of the NTT loops was not pinned down.

Tests: `cargo test -p pgw-pks` (packed output == packed words for every ISA and thread count, odd offsets; the pad
exact at the range extremes against an i128 schoolbook product; the check kernels), `cargo test -p pgw-fast --lib`
(the vector ChaCha20 == the crate, heads / tails / parallel parts / the last block, every width);
tests/hemask/test_pks.py (frame == encode; `encrypt_wire`'s body == the encoded request of `encrypt` + `pack_ksk`,
same secrets; `PKSMaskBackend` sends the same bytes fused and unfused; `decrypt_into` == `decrypt`; released outputs
reused with the same masks; the generator's stores with and without buffer reuse == local generation, ring and p31,
two segments; the view decode == the copying decode on valid and malformed replies, refusals marked as the provider's);
tests/native/test_store.py (`mask_rows_into`; the vector and portable paths in child processes). Directory runs, one
at a time with the rebuilt `pgw_pks` / `pgw_fast` in `src/`: tests/hemask 485 + 20 skips; tests/factory 485 + 9
CUDA skips; tests/native 373; tests/maskstore 247; tests/service 600 + 80 skips (one `test_lane_persistent` case
failed once while the SDK suite ran alongside; it passed alone and in the full rerun); tests/sdk 206 + 1 opt-in skip;
tests/docs/test_code_consistency.py 24 (the four new switches listed); `cargo test -p pgw-pks` 15, `cargo test -p
pgw-fast --lib` 89.

Open: the reply keys (~3.5 ms per linear and group: a scalar NTT of `t'` per prime and the noise layout loop, ~4 ms
per slot at 72B); the request `bytes` is a fresh allocation per call (httpx takes `bytes`) and the HTTP layer's own
copies and TLS are not in these figures; with the default 8 GB stock, 32B / 72B segments are one 256-slot group per
request, which pays the public `a`'s forward NTTs per group (~6 ns more per input value and slot): a larger
`mask_store_gb`; the NTT kernels' bound (a run with hardware counters on a bare-metal AMX box).

## 2026-11-26: RunPod runbook launch-readiness (blockers fixed, stages for the day's changes; not run on RunPod)

Scope: `deploy/runpod/RUNBOOK.md` and its scripts were written before the day's later merges and have not run
against RunPod. Goal: no session fails midway after paying for its setup, and every change of the day gets a
measurement, within the $60 cap (stop at $55). No RunPod call (no key here), no external compute.

**Blockers fixed.**
1. **`sdk32b` would have died at the client step** after the ~65 GB download and pack: `canonical_model`
   refuses `qwen2.5-32b-instruct` (`SUPPORTED_MODELS` is 0.5B-7B), and the bench's `--fingerprint` did not get
   past that check (since the 32B pin shipped, `jobs.py` no longer even passed it). New, **measurement-only and
   off by default**: `PGW_SDK_PREVIEW_MODELS` (`pgw.sdk.options.preview_models`), a comma list of models that are
   in `PINNED_MODELS` **with a baked fingerprint** and not supported; `canonical_model` and `list_models` then
   accept them. Nothing about verification changes: the provider's pack is checked against the baked pin as for a
   supported model, `model_fingerprints` cannot replace a preview model's pin (`SDKOptions.validate`), and a listed
   name without a baked pin is refused. `SUPPORTED_MODELS` is unchanged (adding 14B-72B is a product decision).
   Documented as a preview in `docs/sdk/models.md` and the env table of `docs/sdk/configuration.md`.
   `sdk_remote_bench.py --preview` sets it for `--model` (recorded in the results' `notes`); `--fingerprint` is
   gone. `jobs.py launch` now checks every sdk step before any pod exists: a baked pin, the provider packing the
   pinned commit (`pod_boot.MODELS`), a head for `draft_head`; `cpu_sdk` warns when the provider's published pack
   fingerprint is not the pin.
2. **`--spec draft_head` without `--draft-head`**: the bench skipped the mode (and `stage_spec` would have crashed
   on `Path(None)`); now the head the library ships pinned (`pgw.sdk.draft_heads`, the `-r2` files) is used by
   default, `--draft-head` / `jobs.py --extra` override it, a missing override file fails at start, a model with no
   shipped head (32B) skips the mode with a note, and the results name the head (`spec_draft_head.draft_head`).
   The smoke run builds a random head of the tiny model's shape (the shipped heads are for the real shapes).
3. **Provider disk**: `pod_boot.ensure_pack` packed with `pgw pack` (`pack_checkpoint`: the whole bfloat16
   checkpoint on the volume, ~100 GB peak at 32B). It now runs the shard-wise packer
   (`pgw.modelpack.shardwise.stream_pack_checkpoint`, one shard + one prefetched, as the service user, resumable
   scratch `packs/.<model>.work`); the same pack (`tests/deploy/test_runpod.py`: same fingerprint and files as
   `pack_checkpoint` on the tiny checkpoint). `"packer": "whole"` in an sdk step falls back.

**Stages added** (short stages in the existing sessions; no new session):
* `perf`: phase `pks_stages` (`benchmarks/pks_ntt_bench.py --sections exact_new,stages,round3`, ~6 min): the device
  item's per-stage seconds before / after scaled to a 1024-slot cycle, device reply keys against the host unpack
  and the staged-upload rate, to refit `memplan`'s H100 `PksRates` and its device-wire terms (the T4 model
  under-predicts device-wire items 1.6-1.9x); `maskgen_pks_v3.json` now also has `memplan`'s per-stage terms
  (`model_stages_s`) and `measured_over_model` per item. The `r2` comparison config was checked against
  `pks_gpu`'s switches: `PGW_SERVICE_PKS_GPU_KEYS=torch` (host unpack) + `PGW_SERVICE_PKS_HOST_THREADS=1` (one-thread
  staging; both read per call) is exactly the round-2 path; correct as it was.
* Maskgen watchdog bound: `device_conc2` in every maskgen phase (two items at once on one device, as the service
  runs them; each item's seconds, `watchdog_bound_s`, `headroom_x` against 30 s): 7B / 32B / 72B in `perf`, 32B in
  `sdk32b`, 72B in `72b` (no service runs at 72B in this plan). Where a provider serves (`sdk7b`, `sdk32b`), the GPU
  pod samples the metrics port's `/readyz` `maskgen_progress` every 2 s while the client's factory runs
  (`maskgen_progress-<model>.{json,jsonl}`: worst oldest running item, slowest item, bound, stalled samples); the
  client side counts `maskgen_stalled` status samples (`maskgen_stalled_samples`, expected 0).
* `sdk7b`: `prefix` stage (`prefix_cache=False` then `True` on a system prompt + 4 turns + a second conversation,
  16 tokens per answer, every answer compared: TTFT per turn, cached tokens, prompt-pass rows, mask slots; ~5 min);
  `native_off` (a second process on the same cache, `conc_16` with `PGW_SYNC_MULTI`, `PGW_TRIP_NATIVE_MULTI`,
  `PGW_DECODE_NATIVE_MULTI`, `PGW_TRIP_LEASE_AHEAD`, `PGW_SAMPLE_THREADS`, `PGW_NATIVE_MULTINOMIAL` = 0; the conc
  stage is greedy, so the sampling pair does not act there; `refs_sha256` compares the answers across the two
  processes; ~5 min). The 7B N = 1/4/16 stages measure the rounds on their defaults as before.
* `factory_e2e` on the client pod of `sdk7b` and `sdk32b`, before the sdk step, i.e. while the GPU pod downloads
  and packs (no GPU time): `benchmarks/pks_factory_e2e.py` on one pinned core at 7B / 32B / 72B layer shapes,
  1024-slot segments with the round-3 switches on and off (`PGW_FAST_CHACHA512`, `PGW_FACTORY_R_REUSE`,
  `PGW_PKS_WIRE_FUSED`, `PGW_PKS_OUT_REUSE`), and 256-slot segments at 32B / 72B; one JSON line per run with the CPU
  model.

**Sizes.** Mask stocks (slot bytes 6.17 / 19.2 MB at 7B / 32B, `docs/sdk/scaling.md`): `sdk7b` 24 GB = ~3,900
slots, `sdk32b` 60 GB = ~3,100 slots: both give 1024-slot segments (32B at the 8 GB default would be 416 slots,
one 256-slot group per request, which `factory_e2e` measures separately); the bench waits for at most 3,000
slots, below both targets. Disks, with the shard-wise packer (scratch ~8.5 GB): GPU volumes 60 / 120 / 200 / 60 ->
**20 / 40 / 80 / 20 GB** (perf and 72b use random weights; 7B pack ~9 GB, 32B ~34 GB); client disks: `sdk7b` 80 ->
70 GB (cache 13.3 + stock 24 + venv and native build), `sdk32b` 150 -> 140 GB (cache 44.8 + stock 60). Storage is
cents either way; smaller volumes are easier to place. The bundle is now ~29 MB (the shipped draft heads), under
`pod_job.py`'s 96 MB limit; RunPod's 100 s proxy limit needs ~3 Mbit/s for it.

**Budget** (`jobs.py plan`, table prices, Secure Cloud, full TTL + watchdog grace):

| # | session | GPU | client | wall (min) | TTL (min) | est. $ | change |
|---|---|---|---|---|---|---|---|
| 1 | perf | 1 x H100 SXM | 16 vCPU | 110 | 135 | 8.38 | +5 min: `pks_stages` |
| 2 | sdk7b | 1 x H100 SXM | 32 vCPU | 105 | 130 | 9.52 | +10 min: `prefix`, `native_off` (`factory_e2e` overlaps the pack) |
| 3 | sdk32b | 1 x H100 SXM | 32 vCPU | 140 | 165 | 12.06 | preview opt-in; smaller disks |
| 4 | 72b | 2 x H100 SXM | - | 70 | 95 | 9.78 | smaller volume |
| | **total** | | | | | **39.74** | was 38.82 |

Stopped when done: ~$31 at the planned wall times, ~$36 with ~15 min of setup per session (the old plan, counted
the same way: ~$30 / ~$35; the runbook had said ~$28). One
retry of the costliest session still fits the cap ($51.80 < $55).

**Verified offline** (this box, 4 vCPUs):
* `jobs.py plan` and `jobs.py launch <s> --dc EU-RO-1 --max-usd 15 --dry-run` for `perf`, `sdk7b`, `sdk32b`, `72b`:
  all "nothing created, nothing spent"; `sdk32b` prints the preview note with the pin `a4cf6ba9ac2b...`, `sdk7b` the
  shipped head.
* `provider_perf_session.py --dry-run`: 8 phases, ~100 min, writes only into the given `--out`.
* `sdk_remote_bench.py --smoke --spec off,lookup,draft_head --stages setup,spec,conc,prefix --conc 1,2`: every
  stage, every mode bit-identical to `off`, `conc_2` bit-identical, prefix 629 -> 254 mask slots and 539 -> 164
  prompt rows, every answer identical. `--smoke --preview --model qwen2.5-32b-instruct` (the tiny pack served under
  the 32B name): through setup, spec and conc, the preview noted. Both are tests now (`tests/deploy/
  test_runpod_jobs.py`, slow).
* `phase_pks_items` on CPU at 0.5B / 72B (tiny shapes): `identical` true with `device_conc2`'s replies included;
  `pks_ntt_bench.py --cpu --sections exact_new,stages,round3`: all exact; `pks_factory_e2e.py` with the round-3
  switches on / off (7B, 256 slots: 37.9 / 52.8 ms per slot, `wire_fused` true / false).
* New tests: `tests/sdk/test_preview_models.py` (7: off by default, a pinned model accepted, an unpinned one and a pin
  without a fingerprint refused, a supported name in the list ignored, no pin replacement, the provisioner's pin);
  `tests/deploy/test_runpod_jobs.py` (+6: the launch-time sdk checks, disk sizes, the maskgen progress sampler
  against a fake metrics port, `factory_e2e` on / off, the preview smoke; the smoke extended);
  `tests/deploy/test_runpod.py` (+1: the boot's shard-wise command = `pgw pack`).
* Directory runs, absolute `PYTHONPATH`, one at a time: tests/sdk (serial) 213 passed, 1 opt-in skip (11 min);
  tests/deploy (`-n 2`) 146 passed, 1 skip, 1 failed: `test_lane_nic_session_two_pods_on_cpu`, timing on this
  4-vCPU box (the CPU-emulated lane bench's 90 s serve window ends during the client's 64-connection level; its log
  shows every earlier level complete; it also fails alone; nothing in the lane path changed);
  tests/docs: only the 18 known environment-only `test_numbers` failures (`PGW_SDK_PREVIEW_MODELS` added to the
  documented-switch list). ruff clean on the changed files. mypy not run (this box's mypy lacks the pydantic plugin).

**Only checkable live** (also in the runbook's "First things to check live"): the RunPod API fields, proxy and
pod-scoped key (unchanged from the integration sweep); the bundle upload at ~29 MB; the shard-wise packer against
the hub on a RunPod volume and its fingerprint equal to the pins; the metrics port sampler on a real service; the
real 32B client setup (37 GB from the provider, local hint, 60 GB stock); every number.


## 2026-11-27: consumer chat, the model side of latency (depth, parallel blocks, pruning, crypto limits); Qwen3 support

Full review: `research/reviews/consumer-model-depth.md` (numbers: `research/reviews/consumer_depth_numbers.py`).

- **Cost of a token = sequential non-linear stops + 1.** Standard blocks: 2 per layer (`2L + 1` fused); parallel
  attention+MLP blocks: 1 (`L + 1`). Within a single stateless provider no crypto or architecture gets below one
  client interaction per sequential non-linear stage (Beaver/FSS need the client online at each stage or a second
  non-colluding server; garbling costs 10^4-10^5 x the wire; FHE ~20 s per 7B token on 8 GPUs, cited, approximate and
  unverified; polynomial activations and softmax-free attention cannot be decoded off additive masks for less than
  the next layer's cost). Early exit leaks per-token depth; padded, it saves nothing. MoE routing must be oblivious,
  which multiplies the wire 4-23x on the expert linears (masks stay routed).
- **Best quality per layer (commercial): Qwen3.** Qwen3-14B, 40 layers, Arena-Hard 86.3 (Qwen3 report) vs Qwen2.5-32B,
  64 layers, 74.5: 81 fused trips instead of 129 at 32B-class chat. Native parallel-block models (Phi-2, StableLM-2,
  Falcon, Pythia, GPT-J) are all weak; LayerNorm/GELU support for them is not worth building.
- **Experiment (measured, Kaggle T4 x 2, `benchmarks/depth_prune_heal.py`, results in
  `benchmarks/results/depth_prune/kaggle_05b_r1/`)**: Qwen2.5-0.5B-Instruct, students healed by self-distillation on
  4.8M permissive tokens (UltraChat prompts with the model's own answers + FineWeb-Edu), 7-8M tokens seen. Original:
  ppl 13.57, MMLU 0.495, HellaSwag 0.522. At 25 trips per pass, **parallel blocks (all 24 layers) beat dropping 12
  layers**: ppl 20.44 vs 25.80, chat top-1 agreement 0.765 vs 0.651, HellaSwag 0.397 vs 0.350; drop 6 layers (37 trips):
  ppl 17.19, agreement 0.774; parallel + drop 6 (19 trips): ppl 25.49. **MMLU falls to chance for every student** at
  this budget: the quality gap is large at 0.5B / 8M tokens; a 1.5B-7B run with ~1B tokens is what a decision needs.
- **20 tok/s up to 32B from a consumer device is not reachable with this protocol family** (derived): 41 trips need
  RTT <= 1.8 ms at 1.7 tokens per pass, and the uniform masked traffic is ~34-59 MB per token at 32B-class / 40 layers
  (~6 Gbit/s down at 20 tok/s). Consumer-direct is ~0.2-2 tok/s at 14B-32B-class. In topology B (RTT 0.3 ms, 25 Gbit/s)
  Qwen3-14B with fused2 models at ~20 tok/s, a parallel conversion at ~27-38. On consumer uplinks fused2's K56 ring is
  slower than K32 (Qwen2.5-7B, RTT 20, 300/30: 0.25 vs 0.31 tok/s): the fused schedule needs a narrower ring there.
- **Engine: Qwen3 dense** (`model_type` qwen3): per-head q/k RMSNorm before RoPE on the client (no trip, no provider
  change), packer + shard-wise packer + manifest checks, exact masked == plain on K32/K40/K48 and fused2
  (`tests/engine/test_qwen3.py`, 13; `tests/modelpack/test_shardwise.py::test_qwen3_byte_identical`). qwen2/llama
  paths unchanged (`qk_norms is None`); the native float ops refuse qwen3 and the native decode drivers step aside.
  Not done: Qwen3 pins / SDK support (pin process; no room on this box for the checkpoints), a native q/k-norm path.
- Tests (absolute `PYTHONPATH`, one directory at a time): tests/engine 1225 passed, 15 skipped; tests/modelpack 129
  passed, 1 skipped; tests/docs only the 18 known environment-only `test_numbers` failures.

