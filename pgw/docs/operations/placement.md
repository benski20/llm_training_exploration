# Placement: GPU pods where the users are

Each decoded token is a chain of 113 (7B) to 257 (32B) sequential round trips, so the RTT between a user and the GPU
multiplies into every token ([docs/sdk/wan.md](../sdk/wan.md)). The placement controller starts GPU pods in the
RunPod data centres nearest the users' demand, stops them when the demand goes, publishes the endpoints that serve
right now as a **signed directory**, and the client library moves to the nearest healthy endpoint between requests.

Status (2026-11-28): designed, implemented and tested on CPU and in dry runs. **Nothing has run on RunPod** (no API
key in the build environment, no spend). Every part is off unless enabled: the client's `routing` defaults to
`"off"`, the provider's directory route and demand reports need their settings, SDK keys stay format 1 unless the
provider pins a directory key. [RUNBOOK section 3.9](../../deploy/runpod/RUNBOOK.md) is the small live trial.

## Architecture

```
 clients (pgw.OpenAI, routing="signed")                    placement controller (operator; never on a pod)
   | 1. GET /pgw/v1/endpoints/directory  <---- signed ---- DirectoryPublisher (Ed25519 key, version counter)
   |    verify: pinned key, expiry, version                     ^ endpoints changed
   | 2. payload-free TCP probes -> nearest healthy               |
   | 3. trips (masked) to that endpoint                       Registry  <-- register / ready (pods' API)
   | 4. content-free demand report (hourly)                      ^            | leaf cert (pod-issuing CA)
   v                                                             |            v
 enrolled endpoint (accounts, enrollment, aggregates) --->  Controller.plan (forecast, policy, caps)
                                                                 |  create / drain / terminate
                                                                 v
                                     RunPod API: pods in dataCenterIds=[dc], networkVolumeId = pgw-packs-<dc>
                                     pod: verify pack -> own key + CSR -> register -> load + self-tests
                                          -> /readyz -> controller probes like a client -> listed
```

Code: `pgw.placement` (controller side: `catalog`, `predict`, `telemetry`, `forecast`, `policy`, `publisher`,
`certs`, `registry`, `server`, `coldstart`, `packs`), `pgw.sdk.directory` (client side: verification and the
router), `pgw.service.placement_routes` (the provider's routes), `deploy/runpod/placement.py` (RunPod actuator and
CLI), `deploy/runpod/pod_boot.py placement | stage-packs` (pod side), `benchmarks/placement_sim.py` (simulator).

## Demand telemetry and privacy

The controller's only input about users is an aggregate of **content-free demand reports**. A client with
`routing="signed"` and `usage_reports=True` sends, at most once an hour per model, to the endpoint it uses
(`POST /pgw/v1/placement/report`, authenticated like every call):

```json
{"v": 1, "model": "qwen2.5-7b-instruct", "endpoint": "eu-nl", "rtt_bucket": "<=5", "peak_streams": 3}
```

* **Strict schema** (`pgw.placement.telemetry.validate_report`): exactly these fields, the model one the endpoint
  serves, the endpoint a region label, the RTT one of nine coarse buckets (<=1, 2, 5, 10, 20, 40, 80, 160, >160 ms,
  from the payload-free handshake probes), `peak_streams` an integer up to 256. Anything else is refused whole, so
  no free-form field can carry content.
* **Coarse region server-side.** The metro is derived from the connection's source address with an operator table
  of CIDR prefixes (`demand_regions_file`, e.g. a GeoIP database cut at /20 or coarser); the address is used for that
  lookup and dropped. The client never sends a location.
* **Nothing new to the provider.** The provider already sees, on every connection: the client IP, the model, the
  endpoint, every trip's timing (so the RTT: the gap between a reply and the next request) and volume, and how many
  streams run at once (rows per lane call). The report restates those facts in a summable form; it carries no
  prompt, answer, token id, per-request time or identifier beyond the authenticated tenant the endpoint knows.
* **Aggregated before storing.** Per hour and (metro, model, endpoint, RTT bucket): the number of distinct
  installations and the sum of their peak streams. Installations are counted with a per-hour random salt over the
  tenant id, discarded when the hour closes: the stored counts cannot be joined across hours. The open hour is never
  exported.
* **Small-count suppression and retention.** Buckets under `demand_k_min` (default 5) installations fold into
  `metro="other"` and are dropped if still under k; aggregates older than `demand_retention_days` (default 30) are
  deleted. The controller reads them with its own token (`GET /pgw/v1/placement/aggregates`).
* Off unless the operator sets `demand_reports`; a client sends nothing unless it opted into routing and usage
  reports; a 404 turns the client's reports off.

## Policy (`pgw.placement.policy.Controller`)

Input: anticipated peak concurrent streams per (metro, model), the fleet, the spend ledger and the catalog's GPU
availability. Output: desired pods per (data centre, model) and the create / drain actions.

| step | rule |
|---|---|
| candidates | data centres listing the GPU (live catalog); a create refused for capacity holds that data centre out for `unavailable_hold_s` (30 min) and its metros fall back to the next-best one; a data centre without capacity keeps its running pods |
| warm set | `warm[(metro, model)]` pods in each configured metro's nearest available data centre (no cold start for the first users of the day) |
| gain per $ | a data centre opens when the modelled gain (sum over the streams that would move of their tok/s gain) per $/h of the pods it needs is >= `gain_up` (2 stream-tok/s per $/h); one already open stays while >= `gain_keep` (1): hysteresis |
| capacity | pods = ceil(streams / (`POD_STREAMS` x `target_util` 0.75)), at most `max_pods_per_dc`; overflow to the next-nearest serving data centre |
| scale-down | a count falls only after `idle_timeout_s` (30 min) above need, and only pods older than `min_lifetime_s` (15 min); drained pods stay listed as `draining` for two client refreshes, then terminate |
| spend caps | pods are admitted in priority order (warm, running, capacity, gain) while the plan's rate fits `max_usd_per_hour` and one more hour fits `max_usd_per_day` and `max_usd_total` (the ledger); caps also drain running pods |
| anticipation | demand one cold start ahead (`Forecaster`: hour-of-week profile x smoothed trend, from the hourly aggregates) |

Prediction (`pgw.placement.predict`): ms per token = trips x RTT + fixed + link bytes / bandwidth, with trips and
fixed from `benchmarks/round_trip_model.sdk_ms_per_token(model, 0)` (7B: 113 trips, 37.7 ms; 32B: 257, 101.9 ms;
equal to the benchmark within 0.2 %, tested) and bytes per row from `benchmarks/wan_model.py`. RTT from the metro's
access mix and the great-circle distance (docs/sdk/wan.md: access + 0.5 ms peering + km x 1.6 / 100); data-centre
locations are approximate (`pgw.placement.catalog`; `jobs.py datacenters` lists the live ids) and the clients'
measured RTT buckets replace the estimate once traffic exists. `POD_STREAMS` (16 / 5 / 3 at 7B / 32B / 72B) is an
estimate from the H100 pks-v3 maskgen rates; it is the first number the trial must measure.

## Fast cold start

Per data centre the controller keeps a RunPod **network volume** `pgw-packs-<dc>` with the packs, filled once by a
CPU **packer pod** in that data centre (`placement.py stage-packs`; `pod_boot.py stage-packs`: the shard-wise packer
from the pinned revision, one checkpoint shard on disk at a time, then a full verification). A placement pod mounts
it and never downloads or packs. **Nothing on the volume is trusted**: at boot the pod opens each pack with
`verify="full"` (every tensor hashed, the manifest's fingerprint recomputed) against the **pinned** fingerprint
(`PINNED_MODELS`) and stops before registering on any mismatch (tested with one flipped byte); the service verifies
again at load (page cache) and the clients check the fingerprint and every result anyway.

Boot sequence (`pod_boot.placement_steps`, each step timed and reported as `boot_timings`): `verify_pack` ->
`device` -> `keygen` (own P-256 key, tmpfs) -> `register` (leaf for RunPod's view of the pod's address) ->
`tenants` (credential digests) -> `service_ready` (load, the service's own device self-tests and CUDA graph
warm-up, `/readyz`) -> `activate` (the controller connects like a client: TLS against the pinned root only, the
issued leaf exactly, `/readyz`, the served fingerprints; only then is the pod in the next directory).

Estimated cold start (`placement.py coldstart`; `pgw.placement.coldstart.DEFAULTS` are assumptions except the
SHA-256 rate, 1.5 GB/s per core measured here):

| model | packs | image | schedule | pull | mount | download + pack | verify | load | init | self-tests | register | ready | in use (+ refresh/2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7B | pre-staged | cached | 15 | 0 | 5 | - | 17 | 3 | 20 | 30 | 2 | **92 s** | 242 s |
| 7B | pre-staged | pulled | 15 | 75 | 5 | - | 17 | 3 | 20 | 30 | 2 | **167 s** | 317 s |
| 7B | at boot | pulled | 15 | 75 | 0 | 101 + 217 | 4 | 3 | 20 | 30 | 2 | 468 s | 618 s |
| 32B | pre-staged | pulled | 15 | 75 | 5 | - | 69 | 11 | 20 | 30 | 2 | **227 s** | 377 s |
| 32B | at boot | pulled | 15 | 75 | 0 | 437 + 857 | 17 | 11 | 20 | 30 | 2 | 1465 s | 1615 s |

The verify step is bound by the volume's read rate (assumed 500 MB/s). "In use" adds half the client refresh
interval (`routing_refresh_s`, 300 s): clients find a new pod at their next directory refresh (a failure refreshes at
once). Storage: 20 GB per data centre for 7B (8.7 GB pack + headroom + packer scratch), 59 GB for 7B + 32B; eight
data centres at 7B: 160 GB, $11.2 per month.

## Signed endpoint directory

Wire form `{"payload": base64url(P), "sig": base64url(Ed25519(DOMAIN | P))}`, `DOMAIN =
"pgw-endpoint-directory-v1\0"`, `P` = JSON `{format, v: 1, version, issued_at, expires_at, key_id, endpoints:
[{region, url, models, priority, lane_port, dc, state: serving | draining}]}` (`pgw.sdk.directory`). The client
verifies, before using anything in it:

* the signature with the **pinned** key: the SDK key's own (key format 2 inserts the 32-byte public key after the CA
  pin; format-1 keys are unchanged, byte for byte), else the `endpoint_directory_key` the enrollment response sent
  over the CA-pinned TLS channel; when both exist, the key's pin wins;
* `key_id` = SHA-256(public key)[:8]; `issued_at <= now + 5 min`; `now <= expires_at + 5 min`; validity <= 24 h
  (the publisher uses 1 h);
* **no rollback**: `version` >= the highest version this installation ever accepted (persisted 0600 in the
  session's state directory, survives restarts); the same version only with the same bytes;
* the endpoints with the same validation as enrollment endpoints (https host:port only, at most 32), plus closed
  field names; the enrolled URL is always kept as a choice.

Any failure leaves the previous valid directory in place; an expired one is dropped and the client uses the SDK
key's own provider URL only (never less than without routing). The publisher persists its version before signing
(a restarted controller never reuses a version with other content) and holds the signing key alone (file 0600 or
an HSM / KMS signer behind `pgw.placement.publisher.Signer`); pods never see it. Rotation: publish the new key's
directories alongside a key-format-2 rollout; format-1 installations take the new key from their next enrollment.

**What the signature adds.** Every endpoint must still present a certificate chaining to the pinned provider CA, and
the client still checks the pack fingerprint and every result, so a forged list could never read a prompt or pass
a wrong answer. The signature, expiry and version stop *steering*: an attacker on the network, a stale cache or a
compromised pod cannot send clients to a decommissioned or far-away endpoint, or back to an old list.

## Certificates and secrets

| secret | where | who can use it | lifetime |
|---|---|---|---|
| provider **root CA** key (the CA every SDK key pins) | offline (HSM / air-gapped) | signs the pod-issuing CA, yearly | 10 y; `path_length=1` |
| **pod-issuing CA** key | the controller (0600 / KMS) | signs pod leaves after a valid registration | 90 d; `path_length=0` |
| **pod TLS key** | generated on the pod at boot, tmpfs (`/dev/shm`, 0600), never on the shared volume, never sent | that pod's listener | its leaf: 24 h (max 48 h) > the pod's TTL (12 h): no renewal, nothing to revoke |
| **registration token** | the pod's env (visible to the RunPod account only); the controller stores SHA-256 | one registration, bound to pod name + data centre + model, 1 h | single use |
| per-pod secret | returned at registration, stored hashed | that pod's `ready` and tenant-digest calls | the pod's life |
| directory signing key (Ed25519) | the controller (0600 / KMS) | signs directories | rotated with key format 2 |
| client-issuing CA key | the enrolled endpoint only (enrollment) | issues installation certificates | unchanged |
| tenant credentials on pods | **digests only**: the tokens file (SHA-256 of service tokens) and the certificate-subject map, re-read every 30 s | authenticate tenants; a removed pod gets 403 and drains | the pod's life |
| RunPod API key | the controller; pods get RunPod's pod-scoped key for self-stop only (never passed to the service) | create / terminate | operator |

No private key is shared between pods or shipped in an image or volume. The leaf names **the address RunPod's API
reports for the pod**, never one the pod claims, so a pod cannot obtain a certificate for another host. The
controller checks that the pod serves exactly the issued leaf before listing it. A compromised pod gets its own key
for at most a day and public weights; it sees only masked rows, like every provider endpoint, and cannot
impersonate another pod, read another endpoint's traffic, or change the directory.

The root's `path_length=1` differs from the single-endpoint launcher's CA (`launch.py` creates `path_length=0`): a
placement deployment needs its own root and its own SDK keys.

## Mask stock across endpoints

A mask slot is `W8 r mod 2^K` for the pack's int8 weights and the client's secret `r`; the store API has no endpoint
parameter at all (`reserve(fingerprint, ring, slots)`), so the stock is bound to (pack fingerprint, ring, scheme), not
to a host. Tested (`tests/placement/test_stock_endpoints.py`, `tests/sdk/test_routing.py`):

* reservations made before and after a switch, and after a restart onto a third endpoint, draw **disjoint** slots
  from one ledger; when the stock is spent nothing more is handed out (single use is the client's own ledger and
  rollback counter, wherever the trips go);
* another fingerprint or ring gets nothing; the router only chooses endpoints serving the model's fingerprint;
* end to end on two real endpoints (per-pod leaves from one root, shared accounts): every answer is bit-identical to
  the plaintext reference on each endpoint, from the same stock, with zero integrity failures; the Freivalds and
  range checks run on every trip exactly as before (a switch restarts the runtime, it does not skip a check);
* a model never moves during an answer: a switch is decided at the start of a request and only when no request is
  in flight on that model (a streamed answer in progress stays where it began); a failover happens only before the
  first token (retries stop at the first token, docs/sdk/reliability.md).

## Client behaviour (`routing="signed"`, opt-in)

`SDKOptions.routing = "signed"` (`PGW_ROUTING`), `routing_refresh_s = 300`. At the first model start the engine
builds an `EndpointRouter` (pinned key, version store), fetches the directory from the enrolled endpoint (then any
known one), probes with payload-free TCP handshakes, and starts the model on the nearest healthy endpoint serving its
fingerprint (`pgw.sdk.regions.choose`, hysteresis 20 % and 1 ms). Between requests it refreshes and probes in the
background when due and moves an idle model when another endpoint is better by the margin, or when the current one
fails, drains or leaves the directory. On `backend_unavailable` / `unavailable` before the first token it holds the
endpoint out for two minutes, refreshes, and retries on the next. The provisioned artifacts, the masked-head
decision and the schedule are kept across moves (they belong to the fingerprint). The factory's maskgen follows
the model's endpoint. `status()["routing"]` shows the directory version, endpoints, probe RTTs and errors (no
secrets).

## Simulation (`benchmarks/placement_sim.py`)

Synthetic weekday demand in 8 metros (New York 40, San Francisco 30, London 35, Frankfurt 25, Tokyo 20, Chicago 12,
Dallas 10, Stockholm 6 peak concurrent streams; a 09:00 surge, business hours, an evening bump, a 6 % night floor;
noise), 48 h at 1-minute steps; users in each metro split into in-cloud servers / fibre / cable; streams routed to the
nearest serving data centre with capacity (spill to the next); tok/s per stream from `round_trip_model` at that RTT;
H100 at $2.99/h (`launch.GPU_PRICE_H`); a starting pod is billed from its create and serves after the cold start.
`benchmarks/results/placement/sim_7b_2026-11-28.json` (7B, 26-minute surge):

| policy | pods mean / max | $/h | $ per 1M tokens | RTT p50 / p90 / p99 ms | tok/s per user mean / p10 |
|---|---|---|---|---|---|
| single region (US-KS-2, sized for the global peak) | 17 / 17 | 50.8 | 961 | 49.2 / 160.9 / 170.6 | 0.20 / 0.05 |
| static multi-region (each metro's nearest DC, sized for its peak) | 20 / 20 | 59.8 | 108 | 6.3 / 13.2 / 16.3 | 2.09 / 0.65 |
| **dynamic** (warm 1 per metro, gain per $, idle timeout, anticipation) | 12.7 / 17 | **37.9** | **68** | 6.3 / 13.2 / 16.3 | 2.09 / 0.65 |
| dynamic, no anticipation | 12.5 / 17 | 37.3 | 67 | 6.3 / 13.2 / 16.3 | 2.09 / 0.65 |
| dynamic, packs not pre-staged (468 s cold start) | 12.7 / 17 | 38.0 | 69 | 6.3 / 13.2 / 16.3 | 2.09 / 0.65 |

* **Location is the whole latency story**: one region gives every user outside the US interior 0.05-0.3 tok/s at 7B
  (49-170 ms RTT); the nearest data centre gives 1.0-4.3 tok/s by metro (6-16 ms, dominated by the users' own access
  network: cable 10 ms, fibre 2 ms). Per user, by metro: Tokyo 4.3, Stockholm 3.8, Chicago / Dallas 3.1, San Francisco
  2.4, New York 1.5, London / Frankfurt 1.0 (their nearest catalog DCs are 300-500 km away).
* **Dynamic matches static multi-region latency at 37 % less cost** ($37.9 vs $59.8 per hour; 1.6x fewer $ per token
  than static, 14x fewer than single region): pods follow the sun and are shared across metros near the same data
  centre.
* **Anticipation and pre-staging matter only around surges**, because the policy keeps 25 % headroom and neighbouring
  warm pods absorb the spill during a cold start. With a 6-minute surge at 32B (`sim_32b_surge6_2026-11-28.json`),
  tok/s p10 in each metro's first 15 minutes after 09:00 is 0.29 with anticipation, 0.24 without, 0.21 with packs not
  pre-staged (1465 s cold start; the surge's p90 excess RTT 1.8 ms); at 3x the demand
  (`sim_7b_surge6_x3_2026-11-28.json`) 0.65 / 0.65 / 0.54. No policy ever overloaded a pod in these runs.
* At 32B the same geography gives 0.45-1.8 tok/s per user: the WAN, not placement, is the limit (docs/sdk/wan.md).

The demand, the access mix, the data-centre locations and `POD_STREAMS` are assumptions; the ranking (single <<
static ~ dynamic in latency, dynamic cheapest) is robust to them, the absolute figures are not.

## RunPod integration

`deploy/runpod/placement.py` (dry-run tested; `tests/deploy/test_runpod_placement.py`):

* `plan --demand d.json --warm new-york=1,london=1`: the policy's plan, no API call;
* `stage-packs --dcs ... --models ... --dry-run`: one network volume per data centre (`POST /networkvolumes`,
  `dataCenterId`) and a CPU packer pod mounting it, cost-capped (`--max-usd`);
* `apply ... --dry-run`: the exact `POST /pods` bodies: GPU, Secure Cloud, `dataCenterIds: [dc]`,
  `networkVolumeId: pgw-packs-<dc>`, `pgw-runpod-boot placement`, the one-time token (redacted in output), the
  controller's URL and CA, and the pod TTL (each pod stops itself after 12 h even if the controller dies);
* `coldstart`: the estimate above, with this machine's SHA-256 rate;
* `serve`: the live trial's controller (pods' API over TLS + the control loop: aggregates -> forecast -> plan ->
  apply -> directory file). Guards: Secure Cloud only, the policy's caps checked before every create, `--max-pods`,
  a capacity refusal holds the data centre out, drained pods terminate after two refresh intervals, and pods without
  a staged volume are never created.

## Gaps before production

* Not run on RunPod: the cold-start components, `POD_STREAMS`, the network-volume read rate and the catalog's
  availability format are unmeasured.
* Usage metering on placement pods: the pods authenticate from digests but keep no accounts database; per-key quotas
  and usage are enforced and counted at the enrolled endpoint only. Pods must report their metered rows to the
  controller (not built) before keys with quotas are routed to them.
* Leaf renewal is not built (the pod TTL is below the leaf lifetime); pods living longer than 24 h need it.
* The controller is one process: its state (registry, ledger, publisher version) is in memory and files; a restart
  adopts nothing (`RunPodAPI.list_pods` reconciliation by the `pgw-place-` prefix is the next step).
* Client: the directory is fetched from the enrolled endpoint first; a client whose enrolled endpoint is down at
  start cannot enroll (unchanged) but fails over once it holds a directory.
