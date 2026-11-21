# Two-party PGW demo on Modal

A real provider/customer pair, each in its own Modal app, that you onboard through the gateway's
setup wizard in a browser. Model: `Qwen/Qwen2.5-0.5B-Instruct` (pinned revision).

| party | app | hardware | what runs |
|---|---|---|---|
| provider | `pgw-provider` (`provider.py`) | 4 cores, 6 GiB (or one GPU + 2 cores with `--gpu`, see [GPU provider](#gpu-provider)) | `pgw service serve` (TLS, tenant tokens, enrollment, client-cert issuing CA); pack built into the image |
| customer | `pgw-customer` (`customer.py`) | 4 vCPU, 8 GiB, no GPU | `pgw gateway serve` in setup mode; downloads and packs the model itself during the wizard, encrypted mask factory |

```
 browser ──HTTPS (TLS ends at Modal's edge)──▶ customer container: pgw gateway + console + mask factory
                                                     │  TLS end to end (raw TCP tunnel, CA pinned by the
                                                     ▼  enrollment code, client cert issued at enrollment)
                                               provider container: pgw compute service (CPU or GPU)
```

## Use it

Needs the Modal CLI with a configured token (`modal token set ...`) and Python with `modal` installed.

```bash
python pgw/deploy/modal/launch.py            # deploy both apps, start a fresh pair (80 min), print the details
python pgw/deploy/modal/launch.py --show     # print the live details again (not the setup token)
python pgw/deploy/modal/launch.py --stop     # stop both apps now and clear the details
python pgw/deploy/modal/launch.py --e2e --ttl-min 40   # scripted test (below), then stop
python pgw/deploy/modal/launch.py --deploy-only        # build images and deploy, start nothing
python pgw/deploy/modal/launch.py --no-deploy          # start a pair on the already-deployed apps
python pgw/deploy/modal/launch.py --gpu A10G           # provider on a GPU (fixed at deploy time)
python pgw/deploy/modal/launch.py --i6pn               # private network instead of the tunnel (falls back)
python pgw/deploy/modal/launch.py --e2e --gpu A10G --i6pn --max-trip-ms 1   # latency-gated test (below)
python pgw/deploy/modal/launch.py --e2e --gpu T4 --placement colocated      # one container, loopback (below)
```

`launch.py` prints the console URL, the one-time setup token and the enrollment code. The setup token
is printed **once**: right after printing, the launcher replaces it with `(shown)` in the `pgw-live`
Dict, so `--show` cannot print it again; if it is lost, `--stop` and start a new pair. If either
container dies after it has come up (crash, OOM, function timeout), the launcher notices from the
call's status, stops the other one and exits; on its 20-minute start timeout it stops both. Open the URL,
sign in with the setup token and follow the six steps of
[the onboarding guide](../../docs/operations/onboarding.md): paste the enrollment code, choose
`qwen2.5-0.5b-instruct`, accept the default mask store, save the admin token, run the checks (the
one-time encryption hint takes about 2 minutes, then the first 256 tokens of masks about 2 more;
"in progress" for masks is normal and you can finish before they arrive), finish.
Then sign in with the admin token, create an API key under **API keys**, and try the playground or

```bash
curl -s https://<console host>/v1/chat/completions -H "Authorization: Bearer <api key>" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b-instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Cost and lifetime

Every function has an explicit timeout (88 min), `min_containers=0`, `max_containers=1` and no
retries; the pair also stops itself after `--ttl-min` (default 80, max 85). Deployed apps without
a running call cost nothing. At Modal list prices ($0.0473/core/h, $0.008/GiB/h, T4 $0.59/h) the
CPU-only pair costs about $0.49/h, i.e. about $0.65 for the full 80 minutes, plus network egress
($0.04/GiB): the mask factory uploads about 1.2 MB of ciphertext per token of masks, about 5 GB to
reach its 4000-token stock. With a GPU provider the pair costs about $0.99/h (T4), $1.50/h (A10G),
$2.35/h (L40S) or $4.35/h (H100); see [GPU provider](#gpu-provider). The images (CUDA torch
for the provider, CPU torch for the customer, pack built in) are built once and cached.

GPU functions need a payment method on the Modal workspace; without one, deploy with the default
(CPU provider). The service then runs the exact `cpu_f64` backend; results are bit-identical either way.

Measured on 2026-11-20 (CPU provider, customer 4 vCPU): enrollment 0.3 s, model download + pack +
fingerprint check 12 s, encrypted-mask hint 95 s (7.5 GB), first 256-token mask segment 125 s
(2 tokens/s over the tunnel, which carries about 1.5 MB/s), a 41-token prompt + 32 generated tokens
about 50-57 s.

Measured again on 2026-11-20 after phase 2, with the native gateway kernels now built into both images
and both apps pinned to one region (`PGW_MODAL_REGION`, default `us-east`): setup through the first
masks 108 s, the same 41-token prompt + 32 generated tokens **26 s** (bit-identical logprobs), 6.8 ms
per round trip. Without the region pin one run landed the parties about 85 ms apart and the same chat
took 321 s: every token makes about 100 sequential round trips, so the two parties must be close.

## GPU provider

`--gpu T4|A10G|L40S|H100` deploys the provider with that GPU and 2 cores (one for the service's device
worker, one for its network loop); without it the provider runs the exact CPU int8 backend on 4 cores.
The GPU is fixed when the app is deployed (`launch.py` passes it to `modal deploy` as `PGW_MODAL_GPU`,
which also sets the default of `--gpu`), so `--no-deploy` uses whatever was deployed last; the live
details (`--show`) name the GPU and core count the provider actually got, and the printed cost estimate
uses those. Results are bit-identical on every backend. `--gpu` and `PGW_MODAL_GPU` are
case-insensitive and checked the same way before anything is deployed (one GPU only: no `T4:2`; never
L4). `PGW_MODAL_PROVIDER_CPUS` at deploy time overrides the provider's core count.

| GPU | list price (provider, with 2 cores) | when |
|---|---|---|
| none (CPU) | about $0.24/h | default; no payment method needed on the Modal workspace |
| T4 | about $0.73/h | cheapest GPU; decode GEMV device time about 0.33 ms per trip at 7B |
| A10G | about $1.24/h | default choice for speed (about 2x the T4's memory bandwidth) |
| L40S | about $2.09/h | as A10G, more bandwidth |
| H100 | about $4.09/h | latency tier |
| L4 | refused | its 300 GB/s is no better than a T4's 320 GB/s for the bandwidth-bound single-row GEMV, at a higher price |

Prices are Modal list prices as of 2026-11 (GPU plus $0.0473 per core-hour and $0.008 per GiB-hour);
check them before quoting. The customer adds about $0.25/h. GPU functions need a payment method on the
workspace. At 0.5B the provider's device time is a small part of a trip over the tunnel; a GPU pays off
with i6pn.

## Network: public tunnel or i6pn

By default the gateway reaches the provider through the provider's public TCP tunnel
(`modal.forward(..., unencrypted=True)`; TLS end to end). The tunnel adds about 5 ms to every round trip
(6.8 ms per round trip measured on 2026-11-20, CPU provider), which bounds the demo at a few tokens per
second whatever the provider runs.

`--i6pn` (or `PGW_MODAL_I6PN=1`) uses the deployed `serve_i6pn` / `gateway_i6pn` functions instead: both
containers join Modal's private IPv6 network and the enrollment code points the gateway at the
provider's i6pn address (TLS still end to end; the provider's certificate names that address). It is
**off by default**: on 2026-11-20 Modal did not schedule an i6pn CPU container in `us-east` for 25
minutes. The launcher therefore waits at most `--i6pn-schedule-timeout-min` minutes for both functions
to start (each writes a `*_boot` marker to the Dict as its first action); if one has not started, it
cancels the pair and relaunches it on the tunnel, and says so. The window ends at that marker, so it
also covers the image pull and container cold start, not only Modal's scheduling: the default is 6
minutes without a GPU and 10 with one (larger CUDA image, scarcer machines); raise it if a GPU pair
falls back although Modal did schedule it. Both networks carry the
same masked bytes; the provider learns nothing new from either.

The i6pn round-trip time has not been measured yet (*estimate*: 0.1-1 ms). Per the chat-speed plan's
model, i6pn with a T4 provider and exact local linears (below) would give about 50 ms per token
(*estimate*), against about 310 ms per token over the tunnel (*estimate*).

## Low-latency placement: both parties in one container (`--placement colocated`)

Measured on 2026-11-21 over the public tunnel: **17 ms per round trip**, 49 sequential round trips per
token (qkv and o computed on the gateway), so about 0.85 s per token of network wait alone: the demo ran
at about 1 token/s while the T4 did almost nothing. Earlier tunnel placements measured 6.8, 8.1 and 26 ms.
i6pn did not schedule within 10-25 minutes, twice.

`--placement colocated` (or `PGW_MODAL_PLACEMENT=colocated`) runs both parties in **one** container of a
third app, `pgw-colocated` (`colocated.py`), so every round trip crosses loopback TCP instead:

* the provider's compute service runs as Unix user `pgwprov` in `/srv/pgw-provider` (0700), listens on
  `127.0.0.1` only (nothing outside the container can reach it), with its own fresh PKI; the enrollment
  code points the gateway at `https://127.0.0.1:8443` and pins that CA. It has the GPU (`--gpu`).
* the customer's gateway runs as Unix user `pgwcust` in `/srv/pgw` (0700), without the GPU
  (`CUDA_VISIBLE_DEVICES=""`), with its console on a public HTTPS URL as in `customer.py`.
* **TLS end to end** as before (the gateway verifies the pinned CA and presents the client certificate
  issued at enrollment); each party has its own working directory, PKI, state, `HOME` and caches; logs
  are written by the parent process to a root-only directory (0600 files); neither child inherits the
  Modal runtime's `MODAL_*` variables. The only data between the parties is what the protocol sends.
  The same code (`pgw_modal_boot.start_provider_party` / `start_gateway_party`) runs outside Modal with
  `local_rehearsal.py --users` (see the Kaggle check below). If dropping to the party users fails in
  Modal's container runtime (for example GPU device permissions), deploy with
  `PGW_MODAL_COLOCATED_USERS=0`: both parties then run as root (still separate processes, directories,
  PKI and state; weaker isolation; the result says `separate_users: false`).

```bash
python pgw/deploy/modal/launch.py --e2e --gpu T4 --placement colocated --max-trip-ms 2
python pgw/deploy/modal/launch.py --gpu T4 --placement colocated     # leave it up for the browser wizard
```

The launcher deploys `colocated.py` next to the two usual apps and waits up to `--schedule-timeout-min`
(10 min with a GPU) for the container to start; if Modal has not started it by then, it cancels it and
relaunches the pair on the tunnel, as for i6pn, and says so. Everything else (the wizard, `--e2e`,
`--max-trip-ms`, `--stop`) works as for the other placements. Every RTT figure now names the path it was
measured on: `--e2e` prints `trip latency (colocated via loopback, ...)` and `start-up rtt ... ms over
loopback`; the result has `trip_latency.path`, `speed.rtt_path` and `network_path`, the gateway's
`/admin/status` has `speed.rtt_path`, and its start-up log line says `start-up round trip X ms over
loopback | private <ip> | public <host>`. A tunnel run shows `public r500.modal.host` (or whichever relay).

**Cost.** One container with the provider's GPU, 6 cores (2 + 4) and 14 GiB (6 + 8): at Modal list prices
about **$0.99/h with a T4** ($0.59 GPU + 6 x $0.0473 + 14 x $0.008), the same as the two-container T4 pair
($0.73 + $0.25), and about $0.66 for the full 40-minute e2e window; A10G about $1.50/h. With
`local_linears = "auto"` the gateway measures a sub-millisecond start-up round trip and keeps every
linear on the provider (97 round trips per token for Qwen2.5-0.5B; below the 1.6 ms break-even, local qkv
and o cost more than two round trips).

**Expected speed** (*estimate*, to be confirmed by the lead's Modal run): about 0.5-1 ms per round trip
on loopback with TLS and Python on both ends (measured below on CPU stub kernels), plus the T4's device
time, so about 60-100 ms of round trips per token plus 15-30 ms of gateway float ops and LM head: roughly
**8-15 tokens/s instead of about 1**.

**Trust caveat (demo only).** The two parties share one container, one kernel and one host. Separate
users, 0700 directories and no cross-uid ptrace keep the provider's *process* away from the gateway's
files and memory, but this is not the separation of two organisations' machines: a kernel or container
escape on the provider side, or whoever controls the container (the Modal workspace owner, who controls
both apps of the two-container demo too), reaches both parties, and a shared host offers cache and
scheduling side channels. The provider's process can also reach the gateway's listeners on loopback,
with the same credentials as from the internet, and like any process in a Modal container it can reach
the workspace's `pgw-live` Dict (the container's identity authenticates it; the two-container provider
can too). The masked bytes the provider receives are exactly those of the other placements, and nothing
data-dependent is added. Use this placement to show what the protocol costs on a fast link, not as a
deployment model: a real deployment keeps the gateway in the customer's network.

**Other mechanisms considered** (Modal docs and the installed `modal` 1.6.4 source, 2026-11-21):

* `@modal.clustered(size=2)` (formerly `modal.experimental.clustered`) gang-schedules physically
  co-located containers on i6pn, rank 0 / rank 1 of one function (one image, one GPU spec). But Modal
  requires each container to request **all GPUs of its host** (for example `H100:8`), CPU-only clusters
  are not supported, and both ranks would pay for a GPU; for this demo that is at least two full GPU
  hosts (two T4s at $0.59/h each at the very least, if Modal offers T4 cluster hosts at all) and the same
  scheduling risk as i6pn. Not implemented.
* i6pn with two separate functions (`--placement i6pn`): only containers in the same region **and cloud**
  can talk over i6pn, and Modal restricts placement accordingly; cloud pinning is not available on this
  plan, which is the likely cause of the 10-25 minute stalls. `--region` (deploy time; `any` lifts the
  pin, a region group such as `us-east`, never a zone such as `us-east-1`) is available for experiments;
  the scheduling fallback still applies.
* Modal tunnels (`modal.forward`) always go through a relay host: 7-26 ms per round trip measured.
* Sandboxes with i6pn are scheduled independently like functions: no co-location guarantee.

**Per-trip overhead on the data path (checked 2026-11-21).** Every socket on the compute path has
`TCP_NODELAY`: the gateway's WebSocket stream socket and the service's accepted sockets (asyncio sets it
on every TCP transport; `benchmarks/wire_overhead_bench.py --check-nodelay` reads it back: client `[1]`,
service `[1, 1, ...]`), and the native lane sets it explicitly. With a loopback delay proxy
(`benchmarks/delay_proxy.py`, `--delay-ms`; tc netem is unavailable here) a round trip costs the added
delay plus a constant, with no Nagle/delayed-ACK stall (that would show as about 40 ms):

| path (shared 4-vCPU box, 1-row calls at 7B stub shapes, stream channel) | p50 per call |
|---|---|
| plain loopback, no TLS | 0.36 ms |
| loopback, TLS (as on Modal) | 0.68 ms |
| TLS through the proxy, no added delay | 0.83 ms |
| TLS through the proxy, +2 ms | 3.12 ms (2.28 ms over the proxy hop) |
| TLS through the proxy, +17 ms (the tunnel) | 18.43 ms (17.60 ms over the proxy hop) |

The 0.3-0.6 ms above the added delay is the proxy's own thread wake-ups (four threads per connection
on a loaded box), not the client or the service. TLS adds about 0.3 ms per round trip at these
shapes (Python's `ssl` record handling of the ~150 KB 7B `gate_up` reply; less at 0.5B, about 39 KB).

## Trip latency and the `--max-trip-ms` gate

`--e2e` now also reports the gateway's **client-reported round-trip latency** (`trip_latency` in the
result): the compute client's wall time per masked round trip, from the gateway's own
`pgw_gateway_round_trip_seconds` histogram (`/metrics`) and `client_timing` (`/admin/status`), both read on
loopback inside the customer container, as deltas over each chat:

* `warm` (the second chat) and `both_chats`: `trips`, `p50_ms`, `p90_ms`, `mean_ms`, `p50_bucket_ms`
  (the histogram bucket that holds the median), the bucket counts, and for single-row (decode) trips the
  mean `rows_1_mean_ms` split into `rows_1_service_ms` (service queue + compute) and `rows_1_wire_ms`
  (network, framing, Python on both ends).

`--max-trip-ms X` (needs `--e2e`; off by default) gates the run on the warm chat's median: if it is above
`X`, the launcher cancels the pair and starts a fresh one (a new placement), at most `--tries` pairs in all
(default 3). The decision uses the bucket that holds the median, and the gateway's buckets include
0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5 and 10 ms, so a limit on a bucket bound such as 1 ms is decided from the
counts without interpolation. Exit codes: 0 = exact and within the limit; 1 = the masked answer was not
bit-identical (never retried); 2 = still above the limit after the last try, or nothing measured. A
1 ms limit can pass only on i6pn or co-located (the tunnel alone adds about 5 ms): for a limit below 3 ms the launcher
does not relaunch a tunnel pair unless `--placement i6pn` or `colocated` was asked (each relaunch is a full paid run), and says
why. The window counts every trip of the warm chat, all linear kinds and prefill trips included; the
`rows_1_*` split covers single-row (decode) trips only.
The measurement is local to the gateway: nothing new reaches the provider.

## Gateway speed settings (exact local linears, float threads)

The customer gateway's `models.json` is written by the setup wizard; `pgw_modal_boot.gateway_env` presets
its speed settings through `PGW_GATEWAY_SETUP_MODEL_OPTIONS` (`GATEWAY_MODEL_OPTIONS` there). Each keeps
the answer bit-identical to the plaintext reference (the `--e2e` check still compares greedy text and
logprobs) and shows the provider nothing new:

* `local_linears = "auto"` (`launch.py --local-linears auto|attention|off`, default `auto`): exact local
  linears compute `qkv` and `o` on the gateway from the int8 pack (the same integers, so the same
  bits) and send only `gate_up`/`down`, halving the round trips of Qwen2.5-0.5B from 96 to 48 per
  token at a local read of about 1.8 MB per layer. `auto` decides once at start-up from a few mask-free
  dummy round trips: over the tunnel (about 7-26 ms per trip; break-even about 1.6 ms) it picks
  `attention`; on i6pn the measurement decides. The provider then sees only the `gate_up`/`down` items
  (plus those start-up dummy trips: 1-row, uniform random, no mask).
* `float_threads = 1` (and `PGW_FLOAT_THREADS=1`): the gateway's decode float ops on one torch thread,
  self-checked bit-identical at start-up.

Not used here: the trip lane (`lane`; it is a Unix socket for a service on the gateway's host, and the
provider is remote) and strict speculation (`spec_rows` + `spec_strict_passes`). The demo has no draft
model and its chat barely repeats the prompt, so prompt lookup would accept almost nothing; strict mode
shows the provider as many passes either way, while each pass would carry `spec_rows` masked rows
(that many times the mask supply of the encrypted factory).

`launch.py --e2e` prints the settings in effect (`local_linears` decision, start-up RTT, round trips
per token of the warm chat) next to `trip_latency`; the per-trip gate is unchanged (fewer trips per
token, same per-trip latency).

## Configuration choices

* **Service authentication.** The service listens with TLS (server certificate for the tunnel's host
  name, signed by a fresh provider CA whose SHA-256 is in the enrollment code). Client certificates
  are *verified when presented* but not *required* (`PGW_SERVICE_TLS__REQUIRE_CLIENT_CERT=false`,
  `TLS__CA_FILE` = the client-issuing CA): a gateway has no certificate before it enrolls, and
  enrollment shares the main port. Every compute call needs the tenant's bearer token (issued at
  enrollment, only its hash is stored); a presented certificate must map to the same tenant
  (`PGW_SERVICE_MTLS_TENANTS_FILE`, filled at enrollment). Implication: a stolen tenant token alone is
  enough to use the service as that tenant (it is a secret file, 0600, on the gateway). Production
  should use a separate mTLS-required listener for compute once the service supports one, or an
  enrollment endpoint on its own port.
* **Certificate renewal.** The tunnel is raw TCP (TLS end to end to the service), so the service sees
  the gateway's certificate in the handshake and renews it without extra settings. Behind a
  TLS-terminating proxy instead, set `PGW_SERVICE_TRUSTED_CLIENT_CERT_PEM_HEADER` (see
  [onboarding.md](../../docs/operations/onboarding.md#6-for-operators-issuing-an-enrollment-code)).
* **No volumes.** Config, secrets, packs and the mask store live on the customer container's local
  disk. The mask store must never be on network or snapshotted storage (a restored copy could hand out
  masks twice), and the pair is single-use. A restart of the customer container starts the wizard
  from scratch.
* **Console exposure.** `modal.forward(8080)` rather than `@modal.web_server`: a web endpoint scales to
  zero when no request arrives, which would kill the gateway (and its unsaved local state) while it
  downloads the model or builds masks, and keeping it warm needs `min_containers=1`. The tunnel gives
  the same public HTTPS URL for the lifetime of the call.

## Caveats (demo, not a customer deployment)

* The browser-to-gateway hop terminates TLS at Modal's edge, so **Modal can see prompts and answers on
  that hop**, and the gateway (which holds the plaintext) runs on Modal's infrastructure. In a real
  deployment the gateway runs inside the customer's own network; only masked traffic reaches the
  provider. The gateway-to-service hop here is TLS end to end and carries only masked data.
* The enrollment code and service URL are kept in the Modal Dict `pgw-live` so that `launch.py` can
  print them; anyone with access to the Modal workspace can read them while the pair runs. The setup
  token is there only until `launch.py` has printed it once. `--stop` clears them.
* Encrypted mask generation runs over the public Modal tunnel (about 1.5 MB/s here), which supplies
  far fewer masks per hour than a decoding gateway consumes: fine for a demo with a 256-slot first
  segment, not for sustained load. See
  [docs/operations/encrypted-masks.md](../../docs/operations/encrypted-masks.md).
* The two apps run wherever Modal places them in the region, so round-trip times here are not
  representative of a same-zone deployment. `--e2e` reports the measured trip latency of each run and
  `--max-trip-ms` retries bad placements, but the private-network (i6pn) numbers are still unmeasured.
* Both PKIs are generated in the containers for each run and live for two days at most.

## Files

* `provider.py`, `customer.py`: the two Modal apps.
* `colocated.py`: both parties in one container (`--placement colocated`), each as its own Unix user.
* `e2e_driver.py`: the scripted wizard + chats + exactness check of `--e2e` (no Modal imports; used by
  `customer.py`, `colocated.py` and `local_rehearsal.py`).
* `pgw_modal_boot.py`: PKI, service and gateway start-up (no Modal imports; also used locally).
* `wizard_client.py`: drives `/console/api/setup/*` like the browser and checks a masked chat against
  the plaintext reference.
* `trip_gate.py`: the client-reported trip latency (from the gateway's `/metrics` and `/admin/status`)
  and the `--max-trip-ms` decision; no Modal imports.
* `local_rehearsal.py`: the same bootstrap with both parties on one machine (`--device auto` for a GPU
  service, `--users` for one Unix user per party as in `colocated.py`, `--delay-ms` to emulate a slower
  link, `--json` for the end-to-end result).
* `launch.py`: deploy, start, show, stop, the scripted end-to-end test, the GPU switch, the i6pn
  scheduling fallback and the latency gate.

## Verified end to end (2026-11-20)

`launch.py --e2e` on Modal (CPU provider): wizard completed through the public console URL (enrollment
with CA pin and client certificate, Hugging Face download + pack with matching fingerprint
`004f351fefb5`, mask store, admin token, all checks ok, finish switched to normal mode in-process); a
greedy chat completion through the masked path returned the same 32 tokens as the plaintext
reference on the same pack, with bit-identical logprobs (max difference 0.0), and a repeated request
was identical as well.
