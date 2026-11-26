# PGW provider on RunPod

Our GPU compute service (the provider) on a RunPod pod, with SDK keys for the in-process client library
(`docs/sdk/CONTRACT.md`) and two independent stop guards. Nothing here has been run against RunPod yet (no API key at
the time of writing): the launcher is unit-tested against a mocked API and its bootstrap against a real loopback
service (`tests/deploy/test_runpod.py`). Check the first real launch with `--dry-run` and a short TTL.

| file | what |
|---|---|
| `launch.py` | the launcher: cost estimate + hard `--max-usd`, create pod, watchdog, wait, attested bootstrap, SDK key; `--client-pod`, `--status`, `--stop` |
| `pod_boot.py` | runs inside the pod (`pgw-runpod-boot` in the provider images): volume, PKI, packs, service, self-stop at the TTL |
| `runpod_api.py` | minimal RunPod REST/GraphQL client (`RUNPOD_API_KEY`, never printed; one place for the API's field names) |
| `rtt_probe.py` | round trips from the client side to the provider over pinned TLS (run it in the client pod) |
| `jobs.py`, `pod_job.py`, `job_setup.sh` | measurement sessions for a session that reaches RunPod only through its API and HTTPS pod proxy (no image registry, no SSH): [RUNBOOK.md](RUNBOOK.md) |
| `provider_perf_session.py`, `sdk_remote_bench.py` | the H100 provider measurement plan; the client library against a remote provider (TTFT, tok/s, trips/token, client CPU, speculation, concurrency) |

## One-time setup

1. Images. Push a tag `pgw-provider-v<version>`: `.github/workflows/provider-image.yml` (copy it to the repository
   root's `.github/workflows/` while PGW is a subdirectory) builds `ghcr.io/<owner>/pgw-provider:<tag>` (CUDA) and
   `ghcr.io/<owner>/pgw-provider-cpu:<tag>` (CPU), smoke-tests the CPU image and pushes both. Make the packages
   public, or add a registry credential in RunPod (Settings > Container Registry Auth) and pass its id with
   `--registry-auth-id`. Local build: `docker build -f deploy/docker/provider.Dockerfile -t pgw-provider:dev .`
   from `pgw/`.
2. API key: runpod.io > Settings > API Keys (read/write). Export it; the launcher sends it only to RunPod's API in a
   header and never prints or stores it:

```bash
export RUNPOD_API_KEY=...          # never echo it; never pass it on the command line
export PGW_RUNPOD_IMAGE=ghcr.io/<owner>/pgw-provider:pgw-provider-v0.2.0
export PGW_RUNPOD_CLIENT_IMAGE=ghcr.io/<owner>/pgw-provider-cpu:pgw-provider-v0.2.0
cd pgw
```

## Launch (exact commands)

```bash
# 0. what would happen: price, payload (secrets hidden), nothing created or spent
python deploy/runpod/launch.py --dc EU-RO-1 --gpu "NVIDIA L40S" --models qwen2.5-0.5b-instruct \
    --ttl-min 45 --max-usd 2 --dry-run

# 1. the provider: Secure Cloud, one L40S in EU-RO-1, 0.5B pack built on first boot, stops itself after 45 min
python deploy/runpod/launch.py --dc EU-RO-1 --gpu "NVIDIA L40S" --models qwen2.5-0.5b-instruct \
    --ttl-min 45 --max-usd 2
#    prints: provider URL https://<ip>:<port>, CA sha256, SDK key id, SDK key file ~/.pgw/runpod/sdk-key-<pod> (0600)
#    (--print-key also prints the key itself)

# 2. the client side in the SAME data center (CPU pod, idles until its TTL; pgw + rtt_probe.py inside)
python deploy/runpod/launch.py --client-pod --vcpus 8 --ttl-min 45 --max-usd 1
#    in the client pod (RunPod web terminal or SSH):
python /opt/pgw-deploy/runpod/rtt_probe.py "$PGW_PROVIDER_URL" --ca-sha256 "$PGW_PROVIDER_CA_SHA256"
export PGW_API_KEY=pgw_test_...   # paste from the key file (or launch the client pod with --pass-key)

# 3. watch, then stop (GPU released, volume kept) or terminate (pods deleted)
python deploy/runpod/launch.py --status
python deploy/runpod/launch.py --stop
python deploy/runpod/launch.py --stop --terminate
```

The 7B (latency tier): `--gpu "NVIDIA H100 80GB HBM3" --models qwen2.5-7b-instruct --volume-gb 150 --ttl-min 60
--max-usd 5` (first boot downloads and packs the 15 GB checkpoint: about 10-20 min of GPU time). All four SDK
models: `--models qwen2.5-0.5b-instruct,qwen2.5-1.5b-instruct,qwen2.5-3b-instruct,qwen2.5-7b-instruct`. To keep
packs, PKI and keys across pods, use a network volume: `--network-volume pgw-provider-eu-ro-1 --volume-gb 150`
(created in `--dc` if missing; a pod volume disappears with `--terminate`).

## Cost guards

* **Estimate and hard cap before anything is created.** GPU $/h from RunPod's GraphQL price list (Secure or
  Community), else the table in `launch.py` (2026-11 list prices), else refused unless `--price-per-hour` is given.
  The estimate covers the TTL plus the watchdog's grace (3 min) and the volume; above `--max-usd` the launch is
  refused and the message says how many minutes would fit. `--max-usd` is required.
* **Server-side TTL.** The pod gets `PGW_RUNPOD_STOP_AT` (absolute: a restarted container keeps the deadline);
  `pod_boot.py` drains the service and stops the pod through RunPod's REST API with the pod-scoped `RUNPOD_API_KEY`
  that RunPod injects into the pod (used by the boot only; the service's environment never contains it).
* **Client-side watchdog.** A detached `launch.py --watchdog` process (log in `~/.pgw/runpod/watchdog-<pod>.log`)
  stops the pod at the TTL + 3 min if it is still running, retrying with back-off. `--terminate-at-ttl` makes it
  delete the pod instead. It survives the launching shell, not a reboot of your machine: the pod's own timer is the
  first guard.
* `--status` shows the live price ($/h), uptime and the cost so far; `--stop` stops every recorded pod and the
  watchdogs. L4 GPUs are refused (as on Modal: no faster than a T4 for the bandwidth-bound decode GEMV).

## What the pod runs

`pgw-runpod-boot` (root only for set-up; the pack tool and the service run as uid 10001 with `no_new_privs`):

* volume `/workspace/pgw`: `pki/` (service CA and client-issuing CA created once and kept, so SDK keys, which pin
  the CA, survive restarts; a fresh server certificate for the current public IP at every boot), `state/` (tokens,
  `accounts.db`), `packs/` (built once per model from the pinned revision, same table as Modal), `logs/`;
* `pgw service serve` on 8443 (TLS end to end; RunPod only maps the TCP port) with accounts (SDK keys) on, and the
  admin API on only for the launcher's bootstrap token (its SHA-256 is the only thing passed to RunPod);
* the provider tuning (`pod_boot.PROVIDER_TUNING_ENV`), carried over from the Modal runs where it applies to a
  remote client: the native trip lane over TLS 1.3 on 8444 (announced in tickets as RunPod's public port for 8444,
  `PGW_SERVICE_LANE_ADVERTISED_PORT`), decode graphs at the highest stream priority, the masked LM head served
  (`also`), maskgen wire v3 for the SDK's pks-v3 factory (pending external review; v1/v2 stay served). Not carried
  over: the UDS and shared-memory lanes (same host only), UDS zero-copy, and the unproven spin/lock knobs; try them
  per run with `--service-env PGW_SERVICE_LANE_READ_SPIN_US=2000` etc.;
* client provisioning on (`PGW_SERVICE_PROVISION=true`): the SDK downloads the weight-free gateway pack and the
  fingerprint-checked int8 weights for its mask factory from `/pgw/v1/provision/...` (public artifacts);
* the developer console (`/platform`: keys, usage, the content-free request log) is off by default; turn it on with
  `--service-env PGW_PLATFORM_ENABLED=1` and sign in with an organisation admin token
  (`pgw service admin-token create --org ORG` on the pod).

## Bootstrap trust

The launcher makes a random admin token and passes only its SHA-256 to the pod. Once the provider is ready it asks
`POST /pgw/v1/admin/attest` with a fresh nonce over unverified TLS and accepts the CA only if the reply's
`HMAC(SHA-256(token), nonce | ca_sha256)` matches and `/pgw/v1/enroll/ca` serves that CA; from then on TLS is pinned
to it and only then is the token sent (create the organisation and the SDK key). An interceptor never sees the
token and cannot make the launcher pin its own CA. The token, key and CA live in `~/.pgw/runpod/` (0600).

## Latency: put the client next to the GPU

Every generated token makes tens of sequential round trips (about 26 with attention_down local linears on the 7B
stack; more without), so the round-trip time multiplies: 1 ms RTT is about 26 ms per token. Always run the client
library in the provider's data center (`--client-pod` uses the same `--dc`) and check `rtt_probe.py` first:
`rtt_p50_ms` well below 1 ms inside a data center; across regions it is tens of ms and decoding is 10x slower.

## Open items

* The RunPod REST field names (`gpuTypeIds`, `dataCenterIds`, `portMappings`, `cpuFlavorIds`, ...) and the pod
  variables (`RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_<port>`, the pod-scoped `RUNPOD_API_KEY`) follow RunPod's docs but
  were not exercised against the live API; `runpod_api.py` is the one place to adjust.
* The SDK key's URL is the pod's public IP and mapped port: a new pod means new keys unless a stable DNS name points
  at it (`--public-host`) and the network volume keeps the CA; RunPod's mapped port still changes per pod, so stable
  keys also need a fixed port in front (a TCP load balancer or a DNS name per pod).
* The maskgen bulk channel (`PGW_SERVICE_MASKGEN_BULK_PORT`) needs a third mapped port; not wired.
