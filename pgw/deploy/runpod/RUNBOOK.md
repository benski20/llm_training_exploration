# RunPod measurement runbook (H100 provider + same-DC client)

For a fresh Claude Code session in this repository whose environment injects the RunPod API key for
`rest.runpod.io` and `api.runpod.io` through its egress proxy. It measures, in priority order, what the
2026-11-26 audits could only estimate on CPU, under **$60 in total by default**, and leaves no pod running.
Written and dry-run 2026-11-26 (integration sweep); nothing here has run against RunPod yet. The pieces that
could not be exercised without RunPod are listed in [Unverified](#unverified-check-these-first).

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
* The session uploads the **code bundle** (`pgw/` as git sees it, ~14 MB). The pod accepts it only if its
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
python deploy/runpod/provider_perf_session.py --dry-run --out /tmp/perf-plan   # the H100 plan, commands, minutes
JOB_SETUP_DRY_RUN=1 bash deploy/runpod/job_setup.sh gpu
python -m pytest tests/deploy/test_runpod.py tests/deploy/test_runpod_jobs.py -q -p no:cacheprovider   # ~2.5 min
```

`tests/deploy/test_runpod_jobs.py` runs a whole two-pod `lane_nic` session on loopback (both `pod_job.py`
processes, the bundle upload, the peer channel, the lane bench serving and driving) and the client SDK bench's
CPU smoke run. If anything fails here, stop: do not launch.

Optional, for `speculation="draft_head"` in session 2: a trained 7B draft head file (the draft-head workstream's
artefact). Without it, that mode is skipped and recorded under `notes` in the results.

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
python deploy/runpod/jobs.py launch perf --dc <DC> --max-usd 9
```

* Pods: 1 x H100 SXM (Secure), 16-vCPU CPU client pod, TTL 130 min. **Wall ~105 min + setup; ~$8.1** (table
  prices; `launch` prints the live estimate).
* GPU steps: `provider_perf_session.py --phases env,memory,device_time,gemv_table,lane_local,maskgen --models
  qwen2.5-7b,qwen2.5-32b,qwen2.5-72b` (budget 85 min). Then the `lane_nic` phase: the lane bench serves 7B
  and then 72B layer widths on 8444, with a transport-only stub on 8443, for 8 min each.
* Client steps: fetch each lane bench info file from the peer channel and run `lane_nic_bench.py client --conc
  1,4,16,64 --iters 400` against the GPU pod's public IP and RunPod's mapped ports. The throwaway tickets are
  deleted afterwards.
* Outputs (`gpu/results/perf/`): `env` (GPU, driver, GEMV self-check), `memory.json` (measured
  `memory_allocated` and peak per stage against `memplan`, start-up seconds per stage),
  `device_time.json`, `gemv_table.json`, `lane_local_{7b,72b}.json`, `maskgen.json` (wire v2 fused, 1/4/16
  groups, 1/2/4 concurrent), and `maskgen_pks_v3.json`. The last one covers the pks-v3 cache miss, hit and
  device path at 512/1024/2048 MB work budgets, and the fused kernels as `auto` picks them, against `fwd4`
  (4-prime forward), `prev` (fused NTT only) and plain `torch`, with byte-identity and `memplan`'s estimates.
  `cpu/results/lane_nic_{7b,72b}_client.json` gives trip p50/p90/p99, `compute_us`, RTT, `transport_us`,
  `lane_fixed_us` and throughput at 1-64 connections.
* Decisions it feeds:
  * lane spin, zero-copy and coalesce defaults (`lane_fixed_us` against the 0.13 ms estimate);
  * the pks-v3 `auto` threshold and work budget;
  * fused kernels on by default on H100 (`fwd4` on or off);
  * the maskgen byte budget (`PGW_SERVICE_MASKGEN_DEVICE_MB`);
  * whether 72B fits one H100 in practice (planned headroom 8.5 GiB) or needs TP 2;
  * the H100 GEMV launch table.

### 2. `sdk7b`: the client library end to end at 7B, speculation off / lookup / draft_head, N = 1/4/16

```bash
python deploy/runpod/jobs.py launch sdk7b --dc <DC> --max-usd 10 [--extra /path/to/draft_head_7b.pt]
```

* Pods: 1 x H100 SXM, 32-vCPU client pod (80 GB disk), TTL 120 min. **Wall ~95 min + setup; ~$8.8.**
* GPU: `pod_boot.py` packs Qwen2.5-7B-Instruct at the pinned commit (download + pack, ~10-20 min) and serves it
  with the RunPod provider tuning: TCP+TLS lane, masked head, maskgen v3. The test org and SDK key are created
  over loopback (attested).
* Client:
  * `rtt_probe.py` (200 round trips over pinned TLS);
  * `sdk_remote_bench.py --model qwen2.5-7b-instruct --spec off,lookup,draft_head --conc 1,4,16 --tokens 128
    --mask-store-gb 24`;
  * stages: `setup`; `spec_off`, `spec_lookup` and `spec_draft_head`, each on an open-chat and a copy-heavy
    code-edit prompt; `conc_1`, `conc_4` and `conc_16`;
  * pin: the library's baked 7B pin, so the provider's pack is checked as for a customer.
* Outputs (`cpu/results/sdk-qwen2.5-7b-instruct/results.json`):
  * setup: warm-up and first-answer seconds, plus the client CPU they took;
  * per mode and prompt: TTFT, decode tok/s, ms per token p50, trips per decoded token, rows per trip, the
    provider's `compute_us` per trip, the lane trip and the wait beyond compute, and client / factory CPU per
    token, box cores busy;
  * every mode checked bit-identical to `off`;
  * per N: aggregate and per-stream tok/s, TTFT p50/p95, errors, bit-identity with the prompt alone.
  * `status.jsonl` (the client's status every 30 s) and `rtt-*.log`.
* Decisions it feeds:
  * the pilot's tok/s claim at 7B (docs/sdk/models.md, pilot-handoff);
  * whether `lookup` (or `draft_head`) becomes a recommended setting for copy-heavy workloads (tokens per pass at
    7B);
  * client sizing (docs/sdk/scaling.md: factory CPU per slot at 7B on real cores, streams per process);
  * the `round_trip_model.py --sdk` constants (section 4).

### 3. `sdk32b`: 32B on one H100

```bash
python deploy/runpod/jobs.py launch sdk32b --dc <DC> --max-usd 13
```

* Pods: 1 x H100 SXM (200 GB volume), 32-vCPU client pod (150 GB disk), TTL 165 min. **Wall ~140 min + setup;
  ~$12.1.**
* GPU:
  * `provider_perf_session.py --phases env,memory,device_time,maskgen --models qwen2.5-32b` (35 min);
  * then `pod_boot.py` packs Qwen2.5-32B-Instruct (~65 GB download) and serves it.
* Client: `sdk_remote_bench.py --spec off,lookup --conc 1,4 --tokens 64 --mask-store-gb 60`.
  * **32B has no baked pin yet.** The client is given the provider's own pack fingerprint (`--fingerprint`,
    recorded in the results' `args`). This is a measurement-only override and checks nothing against a pin.
  * If the shard-wise packer work has baked the 32B pin by then (`PINNED_MODELS` has
    `qwen2.5-32b-instruct`), the pin is used instead, automatically.
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

* Pods: 2 x H100 SXM, no client pod, TTL 95 min. **Wall ~70 min + setup; ~$9.8.** (H200: ~$6.5.)
* GPU: `provider_perf_session.py --phases env,memory,device_time,maskgen --models qwen2.5-72b`. Memory is
  measured at TP 1 on the first GPU; `device_time` runs at TP 1 and 2 when two GPUs are present; maskgen at
  72B widths.
* Decisions it feeds:
  * 72B on one H100 (7.3-8.5 GiB planned headroom) or TP 2;
  * the TP 2 gather overhead per trip (`TP2_GATHER_MS`, an estimate today);
  * multi-GPU maskgen balance;
  * the H200 as the single-GPU 72B option.

### 5. Concurrency N = 1 / 4 / 16

Already part of session 2 (`conc_1`, `conc_4`, `conc_16`, the last stages, so a stop loses them first). For a
rerun of only that part, at the end and only if budget remains:

```bash
python deploy/runpod/jobs.py launch sdk7b --dc <DC> --max-usd 10   # after `stop sdk7b`; same session name
```

Before relaunching, edit `SESSIONS["sdk7b"]["steps"][0]` in `jobs.py` to `"spec": "off"` and
`"conc": "1,4,16"` (~60 min, ~$6). If you do, record the edit in the commit message.

## 4. Budget

`python deploy/runpod/jobs.py plan` (table prices, Secure Cloud, each session at its full TTL including the
watchdog grace; `launch` recomputes with RunPod's live price):

| # | session | GPU | client | wall (min) | TTL (min) | est. $ | feeds |
|---|---|---|---|---|---|---|---|
| 1 | perf | 1 x H100 SXM | 16 vCPU | 105 | 130 | 8.09 | memplan, pks-v3 device path and fused kernels, lane over NIC |
| 2 | sdk7b | 1 x H100 SXM | 32 vCPU | 95 | 120 | 8.83 | 7B TTFT, tok/s, trips/token, client CPU, speculation, N = 1/4/16 |
| 3 | sdk32b | 1 x H100 SXM | 32 vCPU | 140 | 165 | 12.11 | 32B provider and client end to end |
| 4 | 72b | 2 x H100 SXM | - | 70 | 95 | 9.79 | 72B memory TP 1, device time TP 1/2, maskgen |
| | **total** | | | | | **38.82** | default cap `--budget-usd 60` |

Prices: H100 SXM $2.99/h Secure (the `launch.py` table, 2026-11), CPU $0.04 per vCPU-hour, storage
$0.10/GB-month. A session stopped as soon as it is done costs about its wall time, not its TTL (about $28 for all
four). The $60 cap leaves room for one retry of one session; the stop rule stops at $55.

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
* **Docs**:
  * `docs/sdk/models.md` "for reference" speeds and resource table;
  * `docs/sdk/scaling.md` sizing (factory ms per slot on the client pod's cores);
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
  image: apt installs python3 first); uploads of ~14 MB within its 100 s request limit.
* **Pod-scoped `RUNPOD_API_KEY` in the pods** (the TTL self-stop). The watchdog is the second guard; `status`
  shows the live state.
* **`cpu5c` CPU pods with 16/32 vCPUs in the chosen data center.** `--cpu-flavor` changes the flavour; a refused
  client pod terminates the GPU pod at once.
* **CUDA 13 hosts** (torch 2.15.0+cu132): `allowedCudaVersions` asks for them; `job_setup.sh` prints
  `cuda True|False`.

## Files

| file | role |
|---|---|
| `jobs.py` | session side (`check-key`, `plan`, `launch`, `upload`, `status`, `log`, `fetch`, `stop`, `teardown-check`) and pod side (`pod`) |
| `pod_job.py` | in-pod boot: TTL self-stop, token-gated control port, pinned bundle, peer channel |
| `job_setup.sh` | in-pod install: locked requirements, pgw, pinned Rust, the three native extensions |
| `provider_perf_session.py` | the H100 measurement plan (sessions 1, 3, 4) |
| `sdk_remote_bench.py` | the client library against a remote provider (sessions 2, 3, 5) |
| `pod_boot.py`, `launch.py`, `runpod_api.py`, `rtt_probe.py` | the provider boot, the cost estimate and the watchdog, the API client, the RTT probe (shared) |
