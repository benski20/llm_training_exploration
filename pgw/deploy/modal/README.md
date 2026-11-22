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
  `PGW_MODAL_COLOCATED_USERS=0`: both parties then run as root (still separate processes, working
  directories, PKI and state, but one `HOME` and its caches; weaker isolation; the result says `separate_users: false`).

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
`local_linears = "auto"` the start-up round-trip measurement decides as on the other placements (below
the 1.6 ms break-even every linear stays on the provider, 97 round trips per token for Qwen2.5-0.5B;
above it qkv and o run on the gateway, 49). On the Kaggle check below it measured 1.93 ms on a loaded
4-vCPU box and chose qkv + o locally.

**Measured on a Kaggle T4 (2026-11-21, job `pgw-colocated-t4`, 4 vCPU shared by both parties).**
`local_rehearsal.py --device auto --users` runs exactly the co-located code (`start_provider_party` /
`start_gateway_party`, users `pgwprov` / `pgwcust`, loopback TLS), the full wizard and two greedy chats of
Qwen2.5-0.5B (41-token prompt, 32 tokens). The second run adds 17 ms per round trip through the delay
proxy (the Modal tunnel as measured). Both answers were bit-identical to the plaintext reference
(32/32 tokens, max logprob difference 0.0).

| warm chat (32 tokens) | co-located, loopback | same box, +17 ms per trip (tunnel) |
|---|---|---|
| round trip p50 / mean | 1.88 / 2.59 ms | 18.0 / 21.1 ms |
| 1-row trip: service / rest | 0.55 / 1.62 ms | 0.76 / 19.9 ms |
| round trips per token | 49.5 | 49.5 |
| wall time | 13.0 s (**2.45 tokens/s**) | 44.3 s (0.72 tokens/s) |

So the network wait per token drops from about 1.05 s to about 0.13 s (3.4x faster overall on that
box). What is left, about 0.28 s per token, is gateway-side work that the 4 vCPUs share with the
encrypted mask factory, which was still filling its stock during the chats (512- and 2048-slot segments,
about 100 s of CPU GEMM each) and with the service. On Modal the co-located container has 6 cores; giving
the gateway more cores, or letting the factory finish its first ramp before chatting, should help further
(*estimate*; not measured). The lead's Modal run decides the real number.

**Trust caveat (demo only).** The two parties share one container, one kernel and one host. Separate
users, 0700 directories and no cross-uid ptrace keep the provider's *process* away from the gateway's
files and memory, but this is not the separation of two organisations' machines: a kernel or container
escape on the provider side, or whoever controls the container (the Modal workspace owner, who controls
both apps of the two-container demo too), reaches both parties, and a shared host offers cache and
scheduling side channels (without a `hidepid` mount the provider's user can also read the gateway's
world-readable `/proc/<pid>/stat`, i.e. its CPU time over time). The provider's process can also reach the gateway's listeners on loopback,
with the same credentials as from the internet, and like any process in a Modal container it can reach
the workspace's `pgw-live` Dict (the container's identity authenticates it; the two-container provider
can too). The masked bytes the provider receives are exactly those of the other placements, and nothing
data-dependent is added to the protocol. Use this placement to show what the protocol costs on a fast link, not as a
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
On the Kaggle T4 box (job `pgw-colocated-t4`, same bench): TLS loopback p50 0.85 ms, through the proxy
1.07 ms, +17 ms 18.44 ms (17.37 ms over the proxy hop); `TCP_NODELAY` set on the client socket and on
all four accepted service sockets.

### Co-located throughput (2026-11-22): defaults and where a token's time goes

`--placement colocated` now defaults to `--local-linears attention_down` and `--lane tls`, and gives the
gateway party `pgw_modal_boot.colocated_gateway_env()`: `PGW_FACTORY_SKIP_LINEAR_KINDS` for a fixed local
mode (the factory then makes only the masks of linears the gateway sends; gate_up alone for
attention_down), `PGW_POOL_SPIN_US=2000` and `PGW_POOL_WORKERS=4`. `--gateway-env` / `--provider-env`
(PGW_/OMP_/MKL_ keys) add or override settings per run; `--sustain-s N` chats back to back after the
exactness check and reports tokens/s, the mask factory segments finished meanwhile and the cores each
party used. The e2e also sends the warm chat once more on loopback (`warm_loopback_tokens_per_s`, same
answer required) to separate the gateway from Modal's HTTPS edge.

Measured (T4, 14 cores, Qwen2.5-0.5B, 41-token prompt; every run logprobs bit-identical and repeat
identical; factory kept busy with `PGW_FACTORY_TARGET_SLOTS=8000`, 1024-slot segments):

| config | warm chat (edge) | back-to-back chats (loopback) | trips/token | 1-row trip | factory while decoding |
|---|---|---|---|---|---|
| before (lead, attention_down, stream lane, all masks) | 4.07 tok/s | - | 26.25 | 3.18 ms | - |
| attention_down, TLS lane, gate_up masks only | 10.1-11.5 | 13.1-13.5 | 26.25 | 0.54-0.57 ms | 138 slots/s on 2.5 cores |
| attention (qkv, o local), TLS lane, gate_up + down masks | 9.2 | 9.4 | 52.5 | 0.47 ms | 5.7 cores |

Per decoded token (attention_down, in-container, about 57-62 ms): round trips about 12 ms (24 x 0.5 ms),
mask prepare + finish about 10.6 ms, local qkv/o/down GEMVs 17-25 ms (148 MB of int8 weights streamed
from DRAM per token; the native pool helps little under gVisor, where waking a sleeping worker is slow),
LM head + sampling 3.5-4 ms, float ops and Python 12-15 ms. Time to first token 520-640 ms (local linears
of the 41 prompt rows about 210 ms, mask reads about 80 ms, float ops and Python about 150 ms). Modal's
HTTPS edge adds 0.5-0.7 s to the warm chat. Not reached: 30 tokens/s (see STATUS.md).

### Gateway GPU, pinned float threads, factory cap (2026-11-22, package T2)

* `--gateway-gpu` (co-located, needs `--gpu`; default off): the container gets two GPUs of the provider's tier
  (`T4:2`). The provider's process sees only the first and the gateway's only the second (`CUDA_VISIBLE_DEVICES`
  by UUID; the device nodes `/dev/nvidia<minor>` are also chowned 0600 to each party's user, which worked under
  Modal's runtime). The gateway computes its exact local linears there (`local_linears_device = "cuda"`,
  `pgw.engine.local_gpu`): the native prepare kernel with a zero mask writes the K=32 payload into pinned memory,
  a captured CUDA graph `[H2D, ring_gemv, D2H]` computes `x_hat @ W8^T mod 2^32` (the provider's fused ring GEMV),
  the native finish with `W r = 0` range-checks and rescales: the masked trip with `r = 0`, so the same bits;
  every local linear is also self-checked against the CPU reference at start-up (else the CPU is kept). The
  mask factory stays on the CPU (`PGW_HEMASK_ENGINE=cpu_int8`, `PGW_FACTORY_DEVICE=cpu`).
* `--float-threads N` (co-located, default 1; 0 = as before): the pinned float formulation. The gateway's process
  runs torch on N threads (`OMP_NUM_THREADS`) and the e2e's plaintext reference on the same N
  (`wizard_client.plaintext_reference(float_threads=)`), so thread-count-dependent kernels (CPU SDPA:
  `benchmarks/sdpa_threads_bench.py` shows different bits at every thread count on Modal hosts) compute the same
  way in both.
* `--gateway-env OMP_WAIT_POLICY=unset` drops the gateway's PASSIVE default; `GOMP_` keys are accepted.
* `PGW_STACK_SAMPLER=<path>` (gateway, off by default): wall-clock stack sampler (code locations only); the
  co-located e2e returns it as `stack_sample` for the sustain window. Note it over-weights GIL-free code.

Measured (T4, 14 cores, `attention_down`, TLS lane, factory target 8000; every run logprobs bit-identical and
repeat identical; Modal hosts vary by 10-20% between runs, so differences below that are noise):

| config | warm chat | back to back | decode local ms/token | decode trips ms/token | factory cores |
|---|---|---|---|---|---|
| CPU local, factory capped (3 threads, nice 10) | 12.2 | 12.3 | 10.5 | 13.3 | 3.9 (120 slots/s) |
| gateway GPU, factory uncapped | 12.0-12.3 | 12.4-13.2 | 10.6 | 13.2 | 6.9-7.5 |
| gateway GPU, factory capped | 11.0 | 13.8 | 10.2 | 12.6 | 3.7 |
| gateway GPU, capped, float threads pinned to 1 | 13.3-13.4 | 12.1-13.6 | 10.1-10.5 | 13.3-13.4 | 3.3-5.0 |

The GPU halves the local linears in isolation (`benchmarks/local_gpu_bench.py`, 72 decode calls of the 0.5B:
4.8-7.4 ms on a T4 vs 8.4-16.3 ms on the CPU pool, host-dependent, 0 mismatches) but in the running gateway the
local linears take about 10 ms per token either way (the CUDA sync of 72 calls per token under load); the GPU
mainly frees about 2.5 gateway cores. Time to first token is 0.5-1.1 s per chat (about 15% of a 64-token chat).

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

* `lane = "stream"`: the blocking stream lane. Decode trips use the same WebSocket stream endpoint
  through the tunnel (same TLS, same msgpack request bytes) but are written and read on the gateway's
  plaintext thread with the TripEngine and synchronous decode, so no event-loop hand-off per trip. Any
  call it cannot complete is resent byte-identically on the regular stream channel. `/admin/status`
  shows `lane_state = "stream"` and `stream_lane_stats` (calls, fallbacks, opens).

* `lane = "tls"` (`launch.py --placement colocated --lane tls`; opt-in, co-located only): the provider's
  service also serves its native trip lane over TLS 1.3 on `127.0.0.1:8444` (`PGW_SERVICE_LANE=tcp`), and
  the gateway sends decode trips there instead of the WebSocket: a 48-byte binary header instead of
  msgpack, rustls in both processes instead of Python `ssl` and asyncio, the GIL released for the whole trip
  on the gateway (one `writev`, spin-then-block wait), CUDA-graph replay without Python on the provider for
  1-row calls. Same encryption and authentication as the stream lane: TLS 1.3 to `127.0.0.1` verified
  against the CA pinned at enrollment, the ticket from `lane/open` (tenant bearer token checked) and bound
  to the gateway's enrollment client certificate, which the lane connection must present. Same masked
  bytes, request ids and call schedule; any call it cannot take goes byte-identically to the blocking
  stream lane. `/admin/status` shows `lane_state = "open"` and `lane_stats`.

Not used here: the UDS trip lane (`lane = "auto"`; the two parties run as different users with separate
directories). Strict speculation is available but off by default; see the next section.

Per 1-row call over loopback TLS, **measured** on a shared 4-vCPU box (2026-11-21, load average 30-40, so
only the ratios mean much; medians of 3 interleaved runs of 2000-3000 calls; `benchmarks/wire_overhead_bench.py
--tls --sync stream|lane --service-lane tcp` against the real service app with zero-cost stub kernels, and
`benchmarks/lane_tls_bench.py` for the transport alone):

| path | 0.5B shapes p50 / mean | 7B shapes p50 / mean |
|---|---|---|
| blocking stream lane (`lane = "stream"`, today's Modal setting) | 346 / 795 us | 546 / 1019 us |
| TCP+TLS lane (`lane = "tls"`), service kernel path in Python (a CPU provider; multi-row calls on a T4) | 219 / 395 us | 346 / 814 us |
| same two, gateway process with 2 busy Python threads (`--gil-load 2`) | 2157 / 3100 vs 280 / 975 us | |
| TCP+TLS lane transport alone (rustls both ends, a no-Python-work executor: what CUDA-graph replay leaves) | 23 / 74 us | 73 / 67 us (with mTLS) |

The stream lane's cost is Python on both ends (msgpack, WebSocket framing and masking, `ssl` records,
asyncio on the service, and a GIL hand-off at every socket call on the gateway), not TLS: the same TLS 1.3
in Rust costs tens of microseconds. With CUDA graphs (1-row calls on a T4) the provider runs no Python per
trip at all. *Estimate* for the co-located T4 0.5B chat (not measured): the 1-row trip falls from 1.9 ms
(service 0.39 + rest 1.52) to about 0.3-0.5 ms, i.e. about 52 x 2.7 ms = 0.14 s of trips per token becomes
0.02-0.03 s, so about 4.7 -> 9-10 tokens/s with the gateway's 70 ms of CPU per token unchanged.

`launch.py --e2e` prints the settings in effect (`local_linears` decision, start-up RTT, round trips
per token of the warm chat) next to `trip_latency`; the per-trip gate is unchanged (fewer trips per
token, same per-trip latency).

## Strict speculation (`--spec-rows`, `--model`, `--spec-draft`; off by default)

Every token costs one pass of sequential round trips (49 for the 0.5B with local attention linears), so over
the tunnel the trip time sets the chat speed (about 1 token/s at 17 ms per trip). Strict speculation
(`spec_rows` + `spec_strict_passes`) verifies R - 1 draft tokens in the same pass, so a pass yields
`accepted + 1` tokens at the same number of trips. The masked target decides every token: answers stay
bit-identical (the `--e2e` comparison still applies). The provider sees exactly R rows in every decode pass,
and after the answer the gateway runs mask-carrying dummy passes on random data until it has seen as many
passes as tokens generated, at the same per-pass cost. So the provider sees fixed rows and the same pass
schedule as without speculation. Masks are taken the same way whatever the text (R slots per generated
token), so the factory's requests reveal nothing either.

```bash
# 1.5B target on the provider's T4, 0.5B draft on the customer's CPU (about $1.05/h for the pair)
python launch.py --gpu T4 --model qwen2.5-1.5b-instruct --spec-draft --spec-rows 4 --e2e
# the same with the draft (and the mask factory) on a customer T4: faster, about $1.64/h
python launch.py --gpu T4 --customer-gpu T4 --model qwen2.5-1.5b-instruct --spec-draft --spec-rows 4 --e2e
```

* `--spec-rows R` (runtime, 2-8): presets strict speculation with prompt lookup (and the draft with
  `--spec-draft`) in the wizard's `models.json`.
* `--model` (deploy time, `PGW_MODAL_MODEL`): the target the provider packs at image build and the customer
  packs in the wizard (pinned revisions: `qwen2.5-0.5b-instruct` (default), `qwen2.5-1.5b-instruct`,
  `qwen2.5-3b-instruct`). A larger target or the draft gives the customer 16 GiB instead of 8.
* `--spec-draft` (deploy time, `PGW_MODAL_DRAFT=1`): packs Qwen2.5-0.5B-Instruct (public weights, same
  tokenizer) into the customer image as the draft. It runs in plaintext on the gateway only: on its CPU, or on
  its GPU with `--customer-gpu T4`. It is refused with the 0.5B target: a draft as large as the target would
  mean the gateway computes the whole model itself.
* All three apply to `--placement colocated` too: `colocated.py` packs the same target, bakes the draft into
  its image and passes `--spec-rows` to its gateway; the draft runs on the gateway's CPU there (the gateway's
  process sees no GPU). The container gets the provider's 6 GiB plus the gateway's 8 or 16 GiB.

The gateway's target float ops stay on its CPU: speculation is refused for a GPU target device, because a
GPU's float kernels give results that depend on the rows of a pass (Kaggle T4: a CUDA target's greedy answers
changed once drafts were accepted). Only the draft may use a GPU.

**Measured acceptance** (Kaggle T4, `benchmarks/spec_accept_bench.py`, 24 chat prompts, 128 tokens, greedy;
tokens per pass simulated over the target's exact answers; the real `Generator` in strict mode agrees within
a few percent, greedy and sampled at T = 0.7). Raw results are in `benchmarks/results/kaggle/run28_spec_accept`
and `run29_spec_accept_cpu`.

| target, drafter | R = 2 | R = 3 | R = 4 | R = 5 | R = 6 | R = 8 |
|---|---|---|---|---|---|---|
| 0.5B, prompt lookup (today's demo) | 1.06 | 1.08 | 1.09 | 1.09 | 1.09 | 1.10 |
| 1.5B, 0.5B draft (+ lookup) | 1.80 | 2.44 | 2.90 | 3.29 | 3.67 | 4.06 |
| 3B, 0.5B draft (+ lookup) | 1.76 | 2.32 | 2.77 | 3.10 | 3.31 | 3.67 |

Prompt lookup alone barely helps the 0.5B demo (chat answers rarely copy the prompt): about 9% fewer passes
for R times the masks. It is not worth it there. The 0.5B draft agrees with the 1.5B on about 81% of tokens.
The draft costs about 110 ms per pass at R = 4 on a T4 (float16; three draft steps plus catch-up; 142 ms at
R = 5) and about 350 ms on a 4-vCPU CPU (float32).

In the gateway's real configuration (target float ops on the CPU, draft on a T4; Kaggle run29, 4 prompts, 96
tokens) the strict `Generator` at R = 4 gave answers identical to plain decoding, greedy and sampled, at 2.84
and 2.78 tokens per pass, with 111 ms (greedy) and 169 ms (sampled) of draft time per pass.

**Estimated answer speed** (tokens per second while the answer streams). These are estimates, not Modal
measurements. A pass costs `trips x RTT + gateway work + draft`. The trips are 49 for the 0.5B and 57 for the
1.5B (two per layer plus the head). The gateway work is the 170 ms per token the live 0.5B demo spends beyond
its trips, plus the measured extra float work of the 1.5B and of R rows (Kaggle CPU, `spec_cost_bench.py`):
about 200 ms at R = 1 and 290 ms at R = 4.

| configuration | trips at 17 ms (tunnel today) | trips at 2 ms |
|---|---|---|
| 0.5B, no speculation (today) | 1.0 (measured) | 3.7 |
| 1.5B, no speculation | 0.86 | 3.2 |
| 1.5B, R = 4, 0.5B draft on a customer T4 | **2.1** | **5.5** |
| 1.5B, R = 4, 0.5B draft on the customer CPU | 1.7 | 3.7 |
| 1.5B, R = 5, draft on a T4 | 2.2 | 5.3 |

R = 4 is the default choice: R = 5 gains about 5% at 17 ms, loses at 2 ms and takes 25% more masks. The 3B
(73 trips, 2.77 tokens per pass) comes out at about 1.6 tokens/s at 17 ms.

**Costs and limits.**

* Price: provider T4 (2 cores) about $0.73/h; customer 4 vCPU and 16 GiB about $0.32/h (CPU draft) or
  $0.91/h with a T4 for the draft. The pair costs about $1.05/h (CPU draft) or $1.64/h (GPU draft).
* Masks: every decode pass, dummy passes included, takes R slots, so a request reserves
  `prompt + max_tokens + (max_tokens - 1) x (R - 1)` slots up front (`launch.spec_budget`). A slot of the
  1.5B is about 2.1 times the 0.5B's bytes (5.9 MB of encrypted supply against 2.8 MB). Over the tunnel the
  encrypted factory supplied about 2 slots/s for the 0.5B, which suggests about 1 slot/s for the 1.5B. At R = 4
  that is about 0.25 generated tokens per second of supply. So **the mask supply, not the round trips, limits a
  sustained chat**: one 128-token answer with a 50-token prompt reserves 559 slots, about 10 minutes of supply.
  The scripted `--e2e` waits for both chats' slots before chatting (`pgw_modal_boot.e2e_min_masks`).
* `--spec-mask C[/B]` (deploy time, `PGW_MODAL_SPEC_MASK`; off by default; L3,
  `research/plans/speculation-no-compromise.md`): filler rows and dummy passes become padding rows (no mask,
  Freivalds-checked like real rows) and every decode pass charges a public `C` slots, the draft credit capped at
  `B`: the store is asked for `prompt + j + min(B, floor((C - 1) j))` slots by pass j, whatever the text, so the
  factory's requests still reveal nothing. Simulated on the run 28 answers: 1.5B with the draft at `--spec-rows 4
  --spec-mask 4/160` keeps 2.84 tokens per pass at 2.5 slots per token instead of 3.96 (1.6x the chat per slot of
  supply); the 0.5B with lookup at `--spec-mask 1.5` keeps its 1.08 tokens per pass at 1.5 slots per token. Not
  measured on Modal yet.
* The dummy passes run after the answer has streamed: one full pass (drafter included) for every token the
  drafts saved, i.e. about two thirds of the answer's tokens at R = 4. The next request of the same
  conversation (API key) waits for them: about 110 s after a 128-token answer of the 1.5B at 17 ms per trip
  (85 passes of about 1.3 s), about 40 s at 2 ms. Users spend part of that time reading; the total provider
  time per conversation is that of plain decoding at R rows.
* One-time setup grows with the target: the provider packs it in its image; the customer downloads and packs
  it in the wizard (about 1 minute on Kaggle for the 1.5B, 2 for the 3B). The encrypted-mask hint is about
  16 GB for the 1.5B (7.5 GB for the 0.5B).

So, with the trips at 17 ms, the 1.5B and its draft answer about twice as fast as the 0.5B demo does today,
with a larger model. Mask supply over the tunnel limits it to a few chats per session, so it stays opt-in.
On i6pn or a faster link, both the speed-up and the mask supply improve.

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
