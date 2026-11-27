# RunPod measurement runbook (H100 provider + same-DC client)

For a fresh Claude Code session in this repository whose environment injects the RunPod API key for
`rest.runpod.io` and `api.runpod.io` through its egress proxy. It measures, in priority order, what the
2026-11-26 audits could only estimate on CPU, under **$60 in total by default**, and leaves no pod running.
Written and dry-run 2026-11-26 (integration sweep), made launch-ready against the rest of that day's changes the
same evening (STATUS.md, "RunPod runbook launch-readiness"); nothing here has run against RunPod yet. The pieces
that could not be exercised without RunPod are listed in [Unverified](#unverified-check-these-first).

## How the session reaches the pods

The session's proxy only forwards HTTPS to port 443 by host name (`/root/.ccr/README.md`: no raw TCP, no SSH,
no non-443 ports, no certificate-pinned clients). So `launch.py`'s attested bootstrap (pinned TLS to the pod's
IP and mapped port) and SSH cannot be used from the session. `deploy/runpod/jobs.py` works around that without
weakening anything:

* Pods start from **public base images pinned by digest** (the provider Dockerfiles' bases:
  `nvidia/cuda:13.2.1-base-ubuntu24.04`, `python:3.12-slim-bookworm`); no image registry of ours is needed.
* The start command runs `pod_job.py` (stdlib, passed zlib+base64 in the pod env). It arms the **TTL self-stop
  first**, then serves a control port 8888, which RunPod's proxy publishes as
  `https://<pod id>-8888.proxy.runpod.net` (port 443, a host name: allowed).
* The session uploads the **code bundle** (`pgw/` as git sees it, ~29 MB with the shipped draft heads). The pod
  accepts it only if its
  SHA-256 equals `PGW_JOB_BUNDLE_SHA256`, fixed in the pod spec at creation, and extracts it with tarfile's
  `data` filter. Then `job_setup.sh` installs the hash-locked requirements and builds the three native
  extensions with the pinned Rust toolchain (rustup-init checked by SHA-256), and `jobs.py pod` runs the steps.
* Every control route needs a token. The pod env holds only the token's SHA-256; the raw job token stays in
  `~/.pgw/runpod/jobs.json` (0600). The GPU pod and its client pod exchange small files (the provider URL, the
  CA fingerprint and a **test** SDK key; the lane bench's throwaway tickets) through `/peer/`, gated by a
  separate pair token. Those files never pass through the session.
* The provider's own TLS, accounts and attestation are unchanged. On the GPU pod, `jobs.py pod` starts
  `pod_boot.py` and creates the test org and SDK key over loopback with the same `launch.bootstrap_sdk_key`
  (HMAC attestation, then TLS pinned to the CA).

## 0. Preflight (no spend, ~5 min)

```bash
cd pgw
export PYTHONPATH=$PWD/src:$PWD
export RUNPOD_API_KEY=injected-by-proxy   # placeholder: the proxy replaces the Authorization header; never echo a real key
git status --short                        # commit or stash first: the bundle is the working tree as git sees it
python deploy/runpod/jobs.py plan         # sessions, minutes, $ at table prices (no API call)
for s in perf sdk7b sdk32b 72b; do
  python deploy/runpod/jobs.py launch $s --dc EU-RO-1 --max-usd 15 --dry-run | tail -1   # "nothing created, nothing spent"
done
mkdir -p /tmp/perf-plan && python deploy/runpod/provider_perf_session.py --dry-run --out /tmp/perf-plan   # plan
JOB_SETUP_DRY_RUN=1 bash deploy/runpod/job_setup.sh gpu
python -m pytest tests/deploy/test_runpod.py tests/deploy/test_runpod_jobs.py tests/sdk/test_preview_models.py \
    -q -p no:cacheprovider   # ~3 min
```

`tests/deploy/test_runpod_jobs.py` runs a whole two-pod `lane_nic` session on loopback (both `pod_job.py`
processes, the bundle upload, the peer channel, the lane bench serving and driving), the client SDK bench's CPU
smoke run (speculation off / lookup / draft_head, N = 1 / 2, the prefix cache off / on), its preview smoke (the
tiny pack served as `qwen2.5-32b-instruct` through `--preview`), the `factory_e2e` step on and off, the maskgen
progress sampler and the launch-time checks of the sdk steps. `tests/sdk/test_preview_models.py`: the preview
opt-in accepts a pinned model, refuses an unpinned one, is off by default, and never replaces a pin. If anything
fails here, stop: do not launch.

`speculation="draft_head"` in session 2 uses the 7B head the library ships pinned (`pgw.sdk.draft_heads`,
`qwen2.5-7b-instruct-d384-v16384-r2.pgwhead`, SHA-256 checked at load); no file is needed. `--extra <head file>`
overrides it (shipped as `extras/<name>`); the results' `spec_draft_head.draft_head` names the head used.

## 1. Key check (read-only)

```bash
python deploy/runpod/jobs.py check-key
```

The output gives the balance, `currentSpendPerHr`, the spend limit, every pod with its status and $/h, and the
Secure and Community prices of H100 SXM, H200, H100 PCIe and H100 NVL. Every call is a `GET` or a GraphQL
`query` (no mutation; `tests/deploy/test_runpod_jobs.py::test_check_key_only_reads`). Go on only if:

* it answers at all (HTTP 401 means the key is not injected for these hosts: stop and report);
* the balance is at least $70;
* **no pod is running.** If one is, it is not yours to stop: ask the user;
* `currentSpendPerHr` is 0.

Choose the data center now: one with H100 SXM in Secure Cloud and CPU pods (`cpu5c`), e.g. `US-KS-2`,
`EU-RO-1` or `US-GA-2`. Use the same `--dc` for every session.

## 2. Spend guards (all on by default; do not turn any off)

| guard | where | what it does |
|---|---|---|
| `--max-usd` (required) | `jobs.py launch` | estimate = (GPU $/h + client vCPU $/h) x (TTL + 3 min watchdog grace) + storage; refused above it, before any pod exists |
| `--budget-usd` (default 60) | `jobs.py launch` | sum of the estimates of every session in `~/.pgw/runpod/jobs.json`, this one included; refused above it |
| Secure Cloud only | `jobs.py` pod specs | `cloudType: SECURE`, `interruptible: false` (no Community / spot) |
| TTL self-stop | `pod_job.py` on each pod | armed before anything else; at `PGW_JOB_STOP_AT` it kills the job and stops the pod through RunPod's REST API with the pod-scoped `RUNPOD_API_KEY` RunPod injects (kept out of the job's environment) |
| watchdog | `launch.py --watchdog` (detached, per pod) | **terminates** each pod at TTL + 3 min if it still exists (`~/.pgw/runpod/watchdog-<pod>.log`) |
| stopped pod refused twice | `jobs.py launch` | a session recorded as active cannot be launched again until `jobs.py stop` |
| client pod refused | `jobs.py launch` | if the client pod cannot be created, the GPU pod is terminated at once |
| RunPod spend limit | the account (`check-key` shows it) | the account's own hard limit |

**Stop rule** (whole runbook). Stop launching, run `jobs.py stop --all --terminate` and report when any of
these happens:

1. the spend so far reaches $55 (add up the `~$ so far` from `jobs.py status` across sessions, or compare the
   balance in `check-key` with the balance at the start);
2. a pod's control port is unreachable 20 min after creation (`launch` prints it), or `job_setup.sh` fails
   (`jobs.py log <session> gpu|cpu`). Retry a session at most once, in another data center;
3. the `env` phase or `job_setup.sh` reports `cuda False` on a GPU pod (the host driver is older than CUDA 13);
4. any result says `bit_identical: false`, `bit_identical_to_off: false`, `identical: false` (pks-v3 items) or
   shows integrity errors. That is a correctness finding: stop spending and report it;
5. a session runs past its TTL without `state: done` (the pods stop themselves; fetch what is there).

## 3. Sessions, in priority order

The `--max-usd` values below are the table estimate plus ~10 %. If `launch` refuses because RunPod's live
price is higher, raise `--max-usd` to the printed estimate plus 10 %, as long as `--budget-usd` still holds.

Each session: `launch` (creates the pods, waits for their control ports, uploads the bundle; about 5-15 min),
then `status` every 5-10 min (the pods' step and the cost so far), then `fetch`, then `stop --terminate`. A
session's pods keep running after their last step until you stop them or their TTL ends, so **stop each
session as soon as both pods say `done` or `failed`.**

```bash
python deploy/runpod/jobs.py status <session>        # per pod: RUNPOD status, $/h, ~$ so far, job state, step
python deploy/runpod/jobs.py log <session> gpu       # tail of boot.log + job.log (setup, steps)
python deploy/runpod/jobs.py fetch <session>         # -> benchmarks/results/runpod/<yyyymmdd>-<session>/{gpu,cpu}/
python deploy/runpod/jobs.py stop <session> --terminate
```

### 1. `perf`: provider on one H100, lane over a real NIC (highest priority)

```bash
python deploy/runpod/jobs.py launch perf --dc <DC> --max-usd 10
```

* Pods: 1 x H100 SXM (Secure, 20 GB volume: random weights, no checkpoint), 16-vCPU CPU client pod, TTL 135 min.
  **Wall ~110 min + setup; ~$8.4** (table prices; `launch` prints the live estimate).
* GPU steps: `provider_perf_session.py --phases env,memory,device_time,gemv_table,lane_local,maskgen,pks_stages
  --models qwen2.5-7b,qwen2.5-32b,qwen2.5-72b` (budget 88 min). Then the `lane_nic` phase: the lane bench serves 7B
  and then 72B layer widths on 8444, with a transport-only stub on 8443, for 8 min each.
* Client steps: fetch each lane bench info file from the peer channel and run `lane_nic_bench.py client --conc
  1,4,16,64 --iters 400` against the GPU pod's public IP and RunPod's mapped ports. The throwaway tickets are
  deleted afterwards.
* Outputs (`gpu/results/perf/`): `env` (GPU, driver, GEMV self-check), `memory.json` (measured
  `memory_allocated` and peak per stage against `memplan`, start-up seconds per stage),
  `device_time.json`, `gemv_table.json`, `lane_local_{7b,72b}.json`, `maskgen.json` (wire v2 fused, 1/4/16
  groups, 1/2/4 concurrent), and `maskgen_pks_v3.json`. The last one covers the pks-v3 cache miss, hit and
  device path at 512/1024/2048 MB work budgets, and the fused kernels as `auto` picks them (since round 3: reply
  keys unpacked on the device, `PGW_SERVICE_PKS_GPU_KEYS`, and uploads staged by the host thread pool), against
  `fwd4` (4-prime forward), `r2` (`PGW_SERVICE_PKS_GPU_KEYS=torch` + `PGW_SERVICE_PKS_HOST_THREADS=1`: exactly the
  round-2 path, checked against `pks_gpu`'s switches), `prev` (fused NTT only) and plain `torch`, with
  byte-identity, `memplan`'s estimates (whole item and per stage, `model_stages_s`; `measured_over_model`), and
  `device_conc2`: two items at once on the device, as the service runs them (each item's seconds against the
  maskgen watchdog's 30 s bound: `headroom_x`).
  `pks_stages.json` (`benchmarks/pks_ntt_bench.py --sections exact_new,stages,round3`): the device item's per-stage
  seconds before / after, scaled to a 1024-slot cycle per model (the key is called `t4_cycle` in the bench; it is
  this GPU's), device reply keys against the host unpack, staged-upload GB/s against one thread.
  `cpu/results/lane_nic_{7b,72b}_client.json` gives trip p50/p90/p99, `compute_us`, RTT, `transport_us`,
  `lane_fixed_us` and throughput at 1-64 connections.
* Decisions it feeds:
  * lane spin, zero-copy and coalesce defaults (`lane_fixed_us` against the 0.13 ms estimate);
  * the pks-v3 `auto` threshold and work budget;
  * `memplan`'s H100 `PksRates` and device-wire terms (the T4 model under-predicts device-wire items 1.6-1.9x);
  * round 3 on H100: device reply keys and staged uploads against `r2`;
  * fused kernels on by default on H100 (`fwd4` on or off);
  * the maskgen byte budget (`PGW_SERVICE_MASKGEN_DEVICE_MB`);
  * whether 72B fits one H100 in practice (planned headroom 8.5 GiB) or needs TP 2;
  * the H100 GEMV launch table.

### 2. `sdk7b`: the client library end to end at 7B, speculation off / lookup / draft_head, N = 1/4/16, prefix cache

```bash
python deploy/runpod/jobs.py launch sdk7b --dc <DC> --max-usd 11 [--extra /path/to/other_head.pgwhead]
```

* Pods: 1 x H100 SXM (40 GB volume), 32-vCPU client pod (70 GB disk), TTL 130 min. **Wall ~105 min + setup;
  ~$9.5.**
* GPU: `pod_boot.py` packs Qwen2.5-7B-Instruct at the pinned commit shard by shard (`pgw.modelpack.shardwise`,
  one shard prefetched: the same pack as `pgw pack`, ~8.5 GB of scratch instead of the 15 GB checkpoint;
  `"packer": "whole"` in the step falls back to `pgw pack`; ~10-20 min) and serves it with the RunPod provider tuning: TCP+TLS lane,
  masked head, maskgen v3. The test org and SDK key are created over loopback (attested). While it serves, the
  provider's maskgen progress is read from its metrics port every 2 s (`maskgen_progress-<model>.{json,jsonl}`).
* Client, in order:
  * `factory_e2e` (while the GPU pod still downloads and packs, so it costs no GPU time; ~5 min):
    `benchmarks/pks_factory_e2e.py` on one core of the client pod's server CPU at 7B / 32B / 72B decoder-layer
    shapes, 1024-slot segments with the round-3 switches on and off (`PGW_FAST_CHACHA512`, `PGW_FACTORY_R_REUSE`,
    `PGW_PKS_WIRE_FUSED`, `PGW_PKS_OUT_REUSE` = 0), and 256-slot segments at 32B / 72B (one group per request: what
    the 8 GB default stock gives there). `factory_e2e.jsonl` (CPU model, ms per slot, slots/s per core);
  * `rtt_probe.py` (200 round trips over pinned TLS);
  * `sdk_remote_bench.py --model qwen2.5-7b-instruct --spec off,lookup,draft_head --conc 1,4,16 --tokens 128
    --mask-store-gb 24 --stages setup,spec,conc,prefix` (24 GB: ~3,900 slots of 6.2 MB, 1024-slot segments);
  * stages: `setup`; `spec_off`, `spec_lookup` and `spec_draft_head` (the shipped pinned head), each on an
    open-chat and a copy-heavy code-edit prompt; `conc_1`, `conc_4` and `conc_16` (every client CPU switch of the
    two concurrency rounds on, the SDK default); `prefix`: `prefix_cache=False` then `True` on a system prompt +
    4 turns + a second conversation with the same system prompt, 16 tokens per answer (~5 min);
  * `native_off` (a second process on the same cache and stock, ~5 min): `conc_16` with `PGW_SYNC_MULTI`,
    `PGW_TRIP_NATIVE_MULTI`, `PGW_DECODE_NATIVE_MULTI`, `PGW_TRIP_LEASE_AHEAD`, `PGW_SAMPLE_THREADS` and
    `PGW_NATIVE_MULTINOMIAL` = 0 (the conc stage is greedy, so the two sampling switches do not act there; the
    other four do);
  * pin: the library's baked 7B pin, so the provider's pack is checked as for a customer.
* Outputs (`cpu/results/sdk-qwen2.5-7b-instruct/results.json`):
  * setup: warm-up and first-answer seconds, plus the client CPU they took;
  * per mode and prompt: TTFT, decode tok/s, ms per token p50, trips per decoded token, rows per trip, the
    provider's `compute_us` per trip, the lane trip and the wait beyond compute, and client / factory CPU per
    token, box cores busy;
  * every mode checked bit-identical to `off`;
  * per N: aggregate and per-stream tok/s, TTFT p50/p95, errors, bit-identity with the prompt alone, and
    `refs_sha256` of the answers alone (equal in `sdk-...-native_off/results.json` when the switches change
    nothing but speed);
  * `prefix`: per turn TTFT off / on, cached tokens, prompt-pass rows and mask slots off / on, every answer
    bit-identical, the cache's counters;
  * `status.jsonl` (the client's status every 30 s; `maskgen_stalled_samples` counts the client watchdog's
    stalls: expected 0) and `rtt-*.log`.
  * `gpu/results/maskgen_progress-qwen2.5-7b-instruct.json`: the provider's worst maskgen item ages while the
    client's factory ran (`oldest_running_s`, `slowest_item_s`, `bound_s`, `stalled_samples`; expected 0).
* Decisions it feeds:
  * the pilot's tok/s claim at 7B (docs/sdk/models.md, pilot-handoff);
  * whether `lookup` (or `draft_head`) becomes a recommended setting for copy-heavy workloads (tokens per pass at
    7B);
  * client sizing (docs/sdk/scaling.md: factory CPU per slot at 7B / 32B / 72B on a server core, 1024- against
    256-slot segments, what round 3 bought there; streams per process; what the concurrency rounds bought at
    N = 16 against a GPU provider);
  * whether `prefix_cache=True` is worth recommending for multi-turn use (TTFT and masks at 7B);
  * the `round_trip_model.py --sdk` constants (section 4).

### 3. `sdk32b`: 32B on one H100 (measurement-only preview of the pinned model)

```bash
python deploy/runpod/jobs.py launch sdk32b --dc <DC> --max-usd 14
```

* Pods: 1 x H100 SXM (80 GB volume), 32-vCPU client pod (140 GB disk), TTL 165 min. **Wall ~140 min + setup;
  ~$12.1.**
* GPU:
  * `provider_perf_session.py --phases env,memory,device_time,maskgen --models qwen2.5-32b` (35 min; maskgen
    includes `device_conc2` at 32B);
  * then `pod_boot.py` packs Qwen2.5-32B-Instruct at the pinned commit shard by shard (65 GB downloaded, one shard
    + one prefetched on disk next to the 34 GB pack: peak ~43 GB instead of ~100 GB) and serves it; maskgen
    progress sampled from the metrics port as in session 2.
* Client: `factory_e2e` first (as in session 2, while the provider packs: a second sample of the same CPU type),
  then `sdk_remote_bench.py --preview --spec off,lookup --conc 1,4 --tokens 64 --mask-store-gb 60` (60 GB:
  ~3,100 slots of 19.2 MB, so 1024-slot segments; at the 8 GB default it would be 416 slots, one 256-slot group per
  request).
  * 32B is pinned (`PINNED_MODELS`: `a4cf6ba9ac2b...` at the provider's commit `5ede1c97`) but not in
    `SUPPORTED_MODELS`. `--preview` enables it through `PGW_SDK_PREVIEW_MODELS` (measurement-only, off by default,
    only for models with a baked pin): the provider's pack is verified against the baked pin exactly as for a
    supported model, and `model_fingerprints` cannot replace it. Without the opt-in the client refuses the model
    with `UnsupportedModelError` (after the provider's 65 GB download; `jobs.py launch` checks the step up front).
  * No draft head ships for 32B (draft_head is not in this session's modes).
* Decisions it feeds:
  * the 32B client burden (setup time on 32 vCPUs, factory CPU per slot, disk);
  * 32B tok/s on one H100;
  * whether 32B is offered in the next release;
  * the 32B column of `round_trip_model.py --sdk` and of `docs/sdk/scaling.md`.

### 4. `72b`: 72B on two H100s (TP 1 and 2), or one H200

```bash
python deploy/runpod/jobs.py launch 72b --dc <DC> --max-usd 11
# no 2 x H100 in the DC: one H200 instead (TP 1 only; launch shows the live price, the table assumes $3.99/h)
python deploy/runpod/jobs.py launch 72b --dc <DC> --max-usd 8 --gpu "NVIDIA H200" --gpu-count 1
```

* Pods: 2 x H100 SXM (20 GB volume: random weights), no client pod, TTL 95 min. **Wall ~70 min + setup; ~$9.8.**
  (H200: ~$6.5.)
* GPU: `provider_perf_session.py --phases env,memory,device_time,maskgen --models qwen2.5-72b`. Memory is
  measured at TP 1 on the first GPU; `device_time` runs at TP 1 and 2 when two GPUs are present; maskgen at
  72B widths, including `device_conc2` (two 72B items at once: the per-item seconds the maskgen watchdog bounds;
  no provider service runs at 72B in this plan, so this is the 72B check of the 30 s bound).
* Decisions it feeds:
  * 72B on one H100 (7.3-8.5 GiB planned headroom) or TP 2;
  * the TP 2 gather overhead per trip (`TP2_GATHER_MS`, an estimate today);
  * multi-GPU maskgen balance;
  * the maskgen watchdog's default bound at 72B (`PGW_SDK_MASKGEN_STALL_S` / `PGW_SERVICE_MASKGEN_STALL_S`);
  * the H200 as the single-GPU 72B option.

### 5. Concurrency N = 1 / 4 / 16

Already part of session 2 (`conc_1`, `conc_4`, `conc_16`, after the spec stages; `prefix` and `native_off` come
after them, so a stop loses those first). For a rerun of only the concurrency part, at the end and only if budget
remains:

```bash
python deploy/runpod/jobs.py launch sdk7b --dc <DC> --max-usd 10   # after `stop sdk7b`; same session name
```

Before relaunching, edit `SESSIONS["sdk7b"]["steps"]` in `jobs.py`: drop the `factory_e2e` step and set the sdk
step's `"spec": "off"`, `"stages": "setup,conc"` and `"conc": "1,4,16"` (~60 min, ~$6). If you do, record the
edit in the commit message.

### 6. Proposed, not wired into `jobs.py`: `fused`, GPU validation of the fused schedules (fusion across RMSNorm)

Not launched; no `SESSIONS` entry yet. Design and the CPU prototype: `research/reviews/consumer-round-trips.md`.
Run it inside session 1 (`perf`, same pod, after its phases; ~10 min of extra GPU, ~$0.5) or as its own 1 x H100
session with no client pod (~25 min, ~$1.5):

```bash
# on the GPU pod, repo root, after job_setup.sh
for m in 7b 32b 72b; do
  python benchmarks/fused2_pair_bench.py --model $m --extra-items --rows 1,2,4,8,16 --reps 30 --variants planes,sum \
      --budget-s 120 --json results/fused/pair_$m.json
done
```

* What it measures, byte for byte against the prototype provider (`exact` must be true everywhere; stop and report
  otherwise): the K56 pair op at Qwen2.5-7B / 32B / 72B shapes for the fused2 items (`o+gate_up`, `down+qkv`), the
  **folded-head item** (`down+head`: last down, then the 152064-row LM head) and the **parallel-block item**
  (`[W_o | W_down]`, then `[W_qkv ; W_gate_up]`), at 1-16 rows, against the graphed K32 singles of the same linears.
  72B's `down+head` and parallel items need ~1.2 / ~0.9 GB of int8 weights each: one H100 holds every item.
* Decisions it feeds (the model is `benchmarks/round_trip_model.py --consumer` / `--twenty`):
  * `C_SCHEDULES`' device factor (1.22 measured at 7B, 1 row) at 32B / 72B, and `C_ROWS_DEVICE` (the rows curve: 7B
    measured 1 / 1.62 / 2.83 / 5.37 at 1 / 2 / 4 / 8 rows; a memory-bound kernel would be ~1.0-1.3 to 16 rows);
  * whether the folded head costs the provider less than the separate K32 masked head trip it replaces;
  * the per-pass device time of the parallel-block schedule (same weights as fused2, half the trips).
* Not in scope: end to end over a WAN (the lane, the K56 store and the pks-v3 digit factory are not wired into the
  client library for fused schedules yet; see the review's production list).
### 6. `maskcost` (PROPOSED: not in `jobs.py`, not launched; needs an explicit go-ahead and a `SESSIONS` entry)

Purpose: pin down the consumer-scale mask cost of `research/reviews/mask-cost-consumer.md`. On a T4 the
coalesced items' per-group part measured 3-5x the memplan model, so the H100 gain is 3.5-11x. This session says
where in that range an H100 is, and how many users it serves.

* Pods: 1 x H100 SXM (Secure, 20 GB volume, random weights). Optional: 1 x RTX 4090 Community pod for the client
  kernels on a consumer GPU, ~$0.35 for 30 min. TTL 75 min. **Wall ~45 min + setup; ~$3.0** (H100 at $2.99/h,
  plus ~$0.4 for the 4090 if taken).
* GPU steps, H100:
  1. `PGW_TEST_CUDA=1 pytest tests/hemask/test_pks_torch.py tests/service/test_pks_coalesce.py`;
  2. `benchmarks/pks_client_gpu_bench.py --sections coalesce --items S` for `S` = 1, 4, 16, 64 at the 7B and 32B
     shapes (the bench's five linears). Every reply must be byte-identical to the items one by one and to one
     S-group item;
  3. the same with `PGW_SERVICE_PKS_COALESCE_MS=50` and 2 concurrent callers per device, through
     `PKSProvider.run_item` from N = 8 / 32 threads on the common seed (`pks.crs_a_seed`). Record
     `coalesced_runs`, the groups per run and the seconds per group. This adds a `--threads` option to the bench
     (small; not written yet);
  4. `benchmarks/pks_ntt_bench.py --sections stages` at `G` = 1 / 16 / 64 to split the per-group part (`y` GEMM,
     key switch, reply copies) on an H100.
* 4090 step: `benchmarks/pks_client_gpu_bench.py --sections client` (eager kernels and the fused decryption),
  for a consumer-GPU data point.
* Outputs: `gpu/results/maskcost/{pytest,coalesce_S*,coalesce_threads,stages_G*,client_4090}.json`.
* Decisions it feeds:
  * whether `PGW_SERVICE_PKS_COALESCE_MS` gets a non-zero default on the provider (it is harmless without a
    common seed);
  * whether the per-group path (reply copies, `y` GEMM shapes, tensor-core key switch) is the next kernel work;
  * the users-per-GPU figure quoted for consumer plans;
  * the common seed stays opt-in until the external review answers Q8, whatever this measures.

## 4. Budget

`python deploy/runpod/jobs.py plan` (table prices, Secure Cloud, each session at its full TTL including the
watchdog grace; `launch` recomputes with RunPod's live price):

| # | session | GPU | client | wall (min) | TTL (min) | est. $ | feeds |
|---|---|---|---|---|---|---|---|
| 1 | perf | 1 x H100 SXM | 16 vCPU | 110 | 135 | 8.38 | memplan (incl. per-stage pks-v3 times), pks-v3 device path, round 3 vs `r2`, fused kernels, item concurrency, lane over NIC |
| 2 | sdk7b | 1 x H100 SXM | 32 vCPU | 105 | 130 | 9.52 | 7B TTFT, tok/s, trips/token, client CPU, speculation (shipped head), N = 1/4/16, prefix cache, concurrency rounds on/off, mask CPU per slot 7B-72B, maskgen progress |
| 3 | sdk32b | 1 x H100 SXM | 32 vCPU | 140 | 165 | 12.06 | 32B provider and client end to end (preview opt-in, pin-checked), maskgen progress at 32B |
| 4 | 72b | 2 x H100 SXM | - | 70 | 95 | 9.78 | 72B memory TP 1, device time TP 1/2, maskgen, 72B item time vs the watchdog bound |
| | **total** | | | | | **39.74** | default cap `--budget-usd 60` |

Prices: H100 SXM $2.99/h Secure (the `launch.py` table, 2026-11), CPU $0.04 per vCPU-hour, storage
$0.10/GB-month (volumes now sized for the shard-wise packer: 20 / 40 / 80 / 20 GB; storage is cents either way).
A session stopped as soon as it is done costs about its wall time plus setup, not its TTL: about $31 for all four
at the planned wall times, ~$36 with ~15 min of setup each (before the added stages: ~$30 / ~$35 counted the same
way, $38.82 at full TTL: +$0.9; `factory_e2e` runs while the GPU pod packs and adds nothing). The $60 cap leaves room for one retry of
one session ($39.74 + $12.06 = $51.80); the stop rule stops at $55.

## 5. Results and model updates

`jobs.py fetch <session>` writes `pgw/benchmarks/results/runpod/<yyyymmdd>-<session>/{gpu,cpu}/results/`.
Commit the JSON, `session.log`, `job.log` and `boot.log`. Do not commit service logs over 1 MB or anything
under a `cache/` directory. No file there holds a key: the SDK key lives only in the pods' `peer/` directories,
which are not in the results, and the lane tickets are deleted after use. Check with
`grep -rl "pgw_test_\|pgwa-" benchmarks/results/runpod/`; it must find nothing.

Then update the models from the numbers, each in its own commit, with the source file named in the comment
next to the constant:

* **`src/pgw/service/memplan.py`**
  * `GPUS["h100-80"].total_bytes` (and `h200`): from `env` (`gpus[].total_gib`); `source="measured (RunPod)"`.
  * Transient terms: compare `memory.json`'s `measured` against its `predicted`, per stage (load, warm-up,
    eager call, hint build, key switch). Where a measured peak exceeds the plan, raise the term and add the case
    to `tests/service/test_memplan.py`, as was done for the T4 run1 reproduction.
  * `H100_RATES` (a `PksRates`): `int8_ops_s` from the hint GEMM stage, `hbm_bytes_s` and `int32_ops_s` from the
    device item seconds per row (`maskgen_pks_v3.json`, `device_*` entries against `model_estimate_s`). Fit them
    the way `T4_RATES` was fitted from `results/kaggle/pks-ntt-t4-r2`, rename the rates "H100 SXM (measured)",
    and keep `python -m pgw.service.memplan --pks-throughput` within ~25 % of the measured items
    (`tests/service/test_pks_fused.py` has the T4 version of that check).
  * The device-wire terms (the T4 model under-predicts whole device-wire items 1.6-1.9x, STATUS round 3): put
    `pks_stages.json`'s measured per-stage seconds (`stages.unit_s`, `after_*` columns: ring rows, upload, wire
    unpack, planes, GEMMs, switch + recombination, key switch, reply copy and pack) next to
    `maskgen_pks_v3.json`'s `model_stages_s` for the same item; refit the stage(s) that carry the gap (on the T4
    the candidates are the device unpack / pack and the recombination, not the GEMMs), then check
    `measured_over_model` comes within ~25 % at 7B / 32B / 72B. The `round3` section gives the staged-upload rate
    (`pool_gbps`) for the host copy pool (modelled at 17 GB/s) and the device reply-key unpack time.
  * `r2` against `auto` (`maskgen_pks_v3.json`): what round 3 (device reply keys, staged uploads) buys on an
    H100, measured, to replace the estimate in STATUS (450 / 135 / 81 slots/s at 7B / 32B / 72B).
  * The `auto` threshold of `PGW_SERVICE_PKS_DEVICE_HINT`, `PGW_SERVICE_PKS_GPU_WORK_MB` and
    `PGW_SERVICE_PKS_GPU_FWD4`: change a default only if the session shows it (byte-identical replies are a
    precondition).
* **`benchmarks/round_trip_model.py`** (`--sdk`) and `benchmarks/scaling_model.py`:
  * `SDK_LANE_MS`: from `lane_fixed_us` p50 at 1 connection (`lane_nic_*_client.json`, `derived_1conn`),
    replacing the 0.09 + 0.04 ms estimate;
  * `scaling_model.DEVICE_MS_PER_MB`: from `device_time.json` (graph or persistent 1-row service time per layer
    / layer MB at 7B, 32B and 72B);
  * `TP2_GATHER_MS`: from TP 2 minus TP 1 at 72B;
  * `SDK_GW_TRIP_MS["7B"]`: from `sdk7b` (`decode_ms_per_token` minus trips x (RTT + lane + device)) / trips, or
    from the client's native phase counters if you rerun `benchmarks/sdk_trip_profile.py` there;
  * `SDK_SERVE_MS`: from the same subtraction outside the passes;
  * `SDK_TAU`: from `spec_lookup` / `spec_draft_head` tokens per pass at 7B (`edit` = copy-heavy, `chat` = open
    chat), replacing the 0.5B values.
  * Then check that `python benchmarks/round_trip_model.py --sdk` reproduces the measured 7B ms per token at the
    measured RTT within ~10 %, and update `tests/benchmarks/test_round_trip_model.py` if its pinned outputs move.
* **The maskgen watchdog** (`PGW_SDK_MASKGEN_STALL_S` / `PGW_SERVICE_MASKGEN_STALL_S`, 30 s, at least 4x the
  slowest recent item): from `maskgen_progress-*.json` (32B and 7B through the service: `slowest_item_s`,
  `oldest_running_s`, `stalled_samples` must be 0) and `device_conc2` (72B in process: `item_s`, `headroom_x`).
  Keep 30 s if the slowest 72B item at two per device is under ~7.5 s (4x stays below the floor); otherwise
  raise the floor and say so in STATUS. Any `stalled_samples` or client `maskgen_stalled_samples` on a healthy
  provider is a false positive: a finding.
* **Docs**:
  * `docs/sdk/models.md` "for reference" speeds and resource table;
  * `docs/sdk/scaling.md` sizing: factory ms per slot per server core at 7B / 32B / 72B (`factory_e2e.jsonl`, against
    this box's 39.5 / 127-137 / 220-240 ms), the 256- against 1024-slot segment cost, round 3 on / off; the
    concurrency rounds at N = 16 against a GPU provider (`conc_16` against `native_off`);
  * `docs/sdk/configuration.md` "Prefix cache": the 7B TTFT and masks off / on;
  * pilot-handoff's first-run estimate if the 7B setup differs.
* **`STATUS.md`**: a new dated section with the commands run, the numbers, the decisions taken, and what each
  session cost (from `status`).

## 6. Teardown verification (always, even after a failure)

```bash
python deploy/runpod/jobs.py stop --all --terminate   # every recorded session's pods, and their watchdogs
python deploy/runpod/jobs.py teardown-check          # exit 0 only when no pod is running (lists every pod)
python deploy/runpod/jobs.py check-key               # currentSpendPerHr must be 0; note the final balance
python deploy/runpod/launch.py --status              # pods recorded by the older launcher, if any
```

`teardown-check` lists every pod of the account. Stopped pods (`EXITED`) keep a volume and bill storage:
terminate them too if they are this runbook's (`pgw-job-*`). Report the final balance, and the spend (start
balance minus final balance), in STATUS.

## Unverified: check these first

Exercised here only against a fake API or on loopback (no RunPod call was made while writing this):

* **RunPod REST field names** for the job pods: `allowedCudaVersions`, `ports` with `/http`, `dockerStartCmd`
  for CPU pods, `vcpuCount` and `cpuFlavorIds`. They follow RunPod's REST v1 docs; `runpod_api.py` and the specs
  in `jobs.py` are the one place to fix them. On an HTTP 400 at `create_pod`, read the message and adjust; the
  guards still hold.
* **The GraphQL `myself { clientBalance currentSpendPerHr spendLimit }` query.** If it is refused, `check-key`
  says so and still lists the pods. Then use the RunPod console's balance for the stop rule.
* **RunPod's HTTPS proxy for port 8888** (`https://<pod>-8888.proxy.runpod.net`): first answer after ~1-3 min (CUDA
  image: apt installs python3 first); uploads of the ~29 MB bundle (the draft heads are in it now) within its
  100 s request limit (needs ~3 Mbit/s; if the PUT times out, `jobs.py upload <session>` retries).
* **Pod-scoped `RUNPOD_API_KEY` in the pods** (the TTL self-stop). The watchdog is the second guard; `status`
  shows the live state.
* **`cpu5c` CPU pods with 16/32 vCPUs in the chosen data center.** `--cpu-flavor` changes the flavour; a refused
  client pod terminates the GPU pod at once. The CPU model is recorded in `factory_e2e.jsonl`; the per-core figures
  only compare with this repo's (AVX-512 IFMA box) if it has AVX-512.
* **CUDA 13 hosts** (torch 2.15.0+cu132): `allowedCudaVersions` asks for them; `job_setup.sh` prints
  `cuda True|False`.
* **The shard-wise packer against the hub on the GPU pod** (`pod_boot.SHARDWISE_PACK`): exercised here on a local
  tiny checkpoint (same fingerprint and files as `pgw pack`, `tests/deploy/test_runpod.py`) and on Kaggle at 14B to
  72B (hash-only / 14B written), not on a RunPod volume. First things to look at in `jobs.py log sdk7b gpu`: the
  `packed <model> fingerprint ...` line, which must equal the baked pin (7B `f7c6fb4f...`, 32B `a4cf6ba9...`);
  `cpu_sdk` prints a WARNING when the published fingerprint differs (the client then refuses the provider: a
  finding, stop). `"packer": "whole"` in the sdk step (`PGW_RUNPOD_PACKER`) falls back to `pgw pack` with the old
  disk peak (raise `volume_gb` first: 7B 40 -> 60, 32B 80 -> 140).
* **The provider's metrics port** (`127.0.0.1:9090/readyz`, plain HTTP inside the GPU pod): read by the maskgen
  progress sampler; `maskgen_progress-<model>.json` must show `samples` > 0. If it stays empty, the service's
  metrics listener is not up (the measurement is lost, nothing else is affected).
* **The 32B preview end to end**: the opt-in is unit-tested and smoke-tested on the tiny model served as
  `qwen2.5-32b-instruct`; real 32B client setup (37 GB download from the provider, the local hint build, a
  60 GB stock) has not run anywhere. `setup.warmup_s` in the results is the number to look at first.

### First things to check live (in this order)

1. `check-key` answers and shows no pod; the balance is at least $70.
2. Session 1 `launch`: both control ports answer within 20 min and the bundle PUT returns 200.
3. `jobs.py log perf gpu`: `job_setup.sh` ends with `native ok; torch 2.15.0+cu132 cuda True`.
4. `perf/env`: `gemv_self_check: true`; `maskgen_pks_v3.json`: `identical: true` for every linear (stop rule 4
   otherwise), `fused_status` with `keys` and `wire` `fused`.
5. Session 2: `provider-qwen2.5-7b-instruct.log` has the `packed ... fingerprint f7c6fb4f...` line;
   `sdk-qwen2.5-7b-instruct/results.json` `setup` stage present and every `bit_identical*` true;
   `maskgen_progress-qwen2.5-7b-instruct.json` `samples` > 0 and `stalled_samples` 0.
6. Session 3: `--preview` noted in the results' `notes`; no `UnsupportedModelError` or
   `FingerprintMismatchError` in `sdk-qwen2.5-32b-instruct.log`.
7. After each session: `stop --terminate`, then `teardown-check`.

## Files

| file | role |
|---|---|
| `jobs.py` | session side (`check-key`, `plan`, `launch`, `upload`, `status`, `log`, `fetch`, `stop`, `teardown-check`) and pod side (`pod`) |
| `pod_job.py` | in-pod boot: TTL self-stop, token-gated control port, pinned bundle, peer channel |
| `job_setup.sh` | in-pod install: locked requirements, pgw, pinned Rust, the three native extensions |
| `provider_perf_session.py` | the H100 measurement plan (sessions 1, 3, 4) |
| `sdk_remote_bench.py` | the client library against a remote provider (sessions 2, 3, 5) |
| `pod_boot.py`, `launch.py`, `runpod_api.py`, `rtt_probe.py` | the provider boot, the cost estimate and the watchdog, the API client, the RTT probe (shared) |
