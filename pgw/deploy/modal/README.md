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

### Prompt pass (time to first token), 2026-11-22

`PGW_PREFILL_PROFILE=1` (gateway, off by default) adds `speed.prefill_profile` to `/admin/status` (so to the
e2e's `speed`): wall time per part of the last prompt passes (local linears, mask takes, prepare, finish, plaintext
steps, waits for the service, thread hops), gateway-local only. The co-located demo now also sets (all exact; the
library defaults stay off; `--gateway-env KEY=0` turns one off):

* `PGW_PROMPT_PREFETCH=1`: a prompt pass stages the next remote linear's mask rows on the prefetch thread while the
  current trip and the local work in between run (`MaskedBackend.prefetch_prompt`; the same rows in the same order,
  single-use, owned by the reservation; the count is the prompt's row count, which the provider sees anyway).
  Mask takes per 41-row pass: about 100 ms to 20-37 ms.
* `PGW_LOCAL_GPU_PREFILL_ROWS=64` (with `--gateway-gpu`): the T4's fused ring GEMV serves one row only, so prompt
  rows took the eager digit-GEMM path (many small kernels per call). Now one more captured graph per local linear
  on 64-row blocks (`[H2D, two int8 digits, one cuBLASLt int8 GEMM with exact int32 sums, lo + (hi << 8) mod 2^32,
  D2H]`), zero-padded short blocks, self-checked per linear (a 133-row case included). Local linears per pass:
  about 145 ms to 85 ms.
* `PGW_PIPELINE_BLOCKS=1`: a prompt round trip goes out as one item of all its rows instead of four overlapping
  row blocks (a fixed, public split of the public row count: the provider sees one 41-row item per prompt trip
  instead of four of about 10). The four blocks cost about 200 ms of extra gateway hops and the service answers
  small multi-row items no faster.

Measured (T4:2, `--gateway-gpu`, factory capped, 41-token prompt; every run logprobs bit-identical, text equal,
repeat identical; prompt pass = median of the last 6 passes, ms):

| config | prompt pass | local | takes | service wait | warm chat | back to back |
|---|---|---|---|---|---|---|
| before | 880 (541-904) | 145 | 100-121 | 66-460 | 11.7 | 13.8 |
| prompt prefetch | 700 (515-907) | 145 | 32-69 | 67-449 | 13.2 | 13.3 |
| + GPU prompt graphs + one item per trip | 550 (452-761) | 82-89 | 19-74 | 212-523 | 12.6 | 14.0 |
| + GPU prompt graphs, four blocks | 800 (572-852) | 85-96 | 34-74 | 53-431 | 11.8 | 13.2 |
| demo defaults, merged with the native float path, GPU LM head and lean trips | 440 (396-771) | 73-79 | - | 211-580 | 21.6 | 21.7 |

In the merged run every prompt-row local linear took the new graphs (1080 prompt-block calls, 0 eager calls) and
the time to first token was 0.40-0.82 s: bimodal, following the service's answers to the 41-row items (about 220
ms or about 570 ms per pass).

What is left of a prompt pass is mostly the service's answers to multi-row items (200-500 ms for 24 trips,
host-dependent: the provider's T4 has no fused multi-row kernel and the replies take the Python stream channel)
and the gateway's plaintext steps (about 75 ms). Tried and dropped: prompt trips in blocks of at most 8 rows over
the native trip lane (`ineligible:rows` otherwise): the lane took them, but each 7-row block cost about 5 ms on
the provider (720-820 ms per pass), and sending them concurrently broke the lane (one `RuntimeError`, then the
lane was off for the rest of the run).

### Where a 1-row trip's 0.57 ms goes, and lane spin knobs (2026-11-22, package T3)

Co-located T4, `attention_down`, TLS lane, factory capped (`PGW_FACTORY_CPU_THREADS=3`, `PGW_FACTORY_NICE=10`,
target 8000), CPU local linears; every run logprobs bit-identical, text equal, repeat identical:

| run | 1-row trip mean | service (graph) | rest | warm chat | back to back |
|---|---|---|---|---|---|
| baseline | 0.565 ms | 0.198 | 0.367 | 10.8 tok/s | 12.4 |
| all spins (gateway 3 ms, GPU 2 ms, server read 20 ms) | 0.567 | 0.137 | 0.430 | 11.8 | 13.0 |
| GPU spin 2 ms only | 0.569 | 0.183 | 0.386 | 10.5 | 12.0 |

The provider's 1-row path is already native (rust/pgw-lane: rustls, CUDA-graph replay, no Python per call), so
nothing is left to move off the event loop there. The floor is Modal's gVisor network stack: the TLS lane transport
alone (`benchmarks/lane_tls_bench.py`, stub executor, `gpu_bench.py --target c8`, 2 ms gap between trips like
decode) costs **0.28 ms p50 per 1-row trip** (0.23 ms back to back; the same bench on a plain Linux VM: 0.03-0.05
ms); the UDS transport 0.16 ms (0.13 ms with both spins). Busy-polling loopback TCP under gVisor made the TLS
transport slower (0.32-0.36 ms), not faster. Opt-in knobs, all off by default, waiting only (same bytes, same
replies): `PGW_LANE_SPIN_US` (gateway TripChannel reply poll, default 50), `PGW_SERVICE_LANE_SPIN_US` (graph stream
poll, default 50), `PGW_SERVICE_LANE_READ_SPIN_US` (new: the lane server polls a connection for its next request
before blocking; UDS by peek, TLS by non-blocking reads into the session, default 0). `PGW_MODAL_APP_SUFFIX`
(deploy time) suffixes the app and Dict names so parallel experiments do not replace each other's apps.
### Native pinned float path (2026-11-22, package T3)

`--float-ops native` (co-located; now the co-located default, `--float-ops torch` = as before): the gateway runs
the layer float ops (RMSNorm, residual adds, RoPE, attention over the KV cache, SiLU gate) in `pgw_fast`
(`rust/pgw-fast/src/layer.rs`, `attn.rs`) through `PGW_FLOAT_OPS=native`, one native call between consecutive
linears, and the e2e's plaintext reference uses the same path (`wizard_client.plaintext_reference(float_ops=)`).
It is a **new pinned reference**: a fixed order of IEEE operations without FMA or libm, so its bits depend on
neither the ISA (AVX2 and baseline builds agree) nor torch's thread count nor how many rows share a pass; RMSNorm,
attention and SiLU differ from torch's in the last bits (RoPE and the residual adds are torch's bits). The final
norm and the LM head are unchanged. Privacy unchanged (local gateway math only; the provider receives the same).

Measured (T4, `attention_down`, TLS lane, factory capped: `PGW_FACTORY_CPU_THREADS=3`, `PGW_FACTORY_NICE=10`,
target 8000; float threads pinned to 1; both runs logprobs bit-identical, text equal, repeat identical; per
decoded token = (chat time - time to first token) / (tokens - 1), from the gateway's audit log):

| float ops | warm chat | back to back | ms per decoded token (7 chats) |
|---|---|---|---|
| torch (A/B baseline, same day) | 11.6 tok/s | 12.7 | 62-64 (one 68) |
| native (`--float-ops native`) | 14.9 tok/s | 13.4 | 49-53 (one 83: host hiccup) |
| native (the new co-located default) | 13.7 tok/s (16.2 on loopback) | 16.6 | 46-49 |

About 11-16 ms per decoded token saved (Modal's CPU and gVisor make each of the 40-odd torch calls per layer
and the per-step thread scoping more expensive than locally: on the dev box the same change saves 5-6 ms, from
12-13 to 7-8 ms of float ops and engine glue per token in `benchmarks/decode_cpu_bench.py`). Back to back the gain
is smaller because time to first token (0.6-1.2 s, prompt pass) is a third of a 52-token chat.
### Gateway GPU: native local-linear call, fixed-order LM head (2026-11-22, package T3)

* **Why the local linears cost 10-16 ms per token in the running gateway** (4.8 ms in isolation): Python around
  each of the 72 calls per token, which the running container inflates, not the device. With the Python replay
  path a decode call took 220 us in the gateway (`local_linear_timing` below) against 67 us in isolation: about
  15 Python steps in `LocalGpuProducts.forward` (event record, a polling loop, two native calls with their
  argument parsing, buffer slicing and wiping) inside the backend's generic local path (validation, grouping,
  cache lookups, `np.isfinite`, which releases the GIL above 500 elements). `benchmarks/local_gpu_bench.py
  --breakdown` on a T4 (replay + synchronize, median): a one-copy graph 22 us, kernel-only 32 / 58 us (896x1152 /
  4864x896), the full graph 43 / 75 us: about 22 us of every call is launch and completion, the device floor.
* `pgw_fast.local_gpu_linear_into` (default with `--gateway-gpu`; `PGW_LOCAL_GPU_NATIVE=0` = the Python
  replay): one native call does prepare (zero mask) straight into the pinned buffer, `cuGraphLaunch` of the
  captured graph's exec through a `dlopen`ed driver (as the provider's lane), a spin on `cuStreamQuery`
  (`PGW_LOCAL_GPU_SPIN_US`, default 2000, then blocking), the range check and rescale (the finish kernel's
  expressions), and wipes both pinned buffers. It keeps the GIL for the call (`PGW_LOCAL_GPU_HOLD_GIL=0`
  releases it): a call is tens of microseconds and a hand-off costs more. Decode calls (1 row) take a lean
  per-linear callable bound at start-up (`PGW_LOCAL_GPU_ONE_ROW=0` disables it), compared bit for bit with the
  CPU reference on public rows (Gaussian, spike, zero, tiny, non-finite refusal) before use; the regular path
  stays self-checked as before. Prompt passes keep the eager GPU path (about 0.12 s per 41-row prompt).
* Launch configurations of the local graphs' ring GEMV for 1 row are measured per GPU model
  (`pgw.engine.local_gpu.ONE_ROW_CONFIGS`, `benchmarks/local_gemv_sweep.py`; `PGW_LOCAL_GPU_GEMV` overrides,
  `PGW_LOCAL_GPU_TABLE=0` = the provider's heuristic). T4, replay + synchronize: qkv 43 -> 36 us, o 35 us,
  down 66 -> 46 us (no split-K, so no memset node). Any configuration gives the same integers; the start-up
  self-check against the CPU reference still runs per linear.
* **LM head on the gateway GPU, same bits** (`lm_head_device = "cuda"`, on with `--gateway-gpu`;
  `--gateway-env PGW_LM_HEAD_DEVICE=cpu` turns it off; library default `cpu`): the CPU head already has a
  fixed summation order (16 partial sums over 16-column chunks, product rounded then added, no FMA, a fixed
  tree), and each step is a correctly rounded float32 operation, so `pgw.engine.lm_head_gpu` runs the same
  order as one Triton kernel (PTX `mul.rn.f32` / `add.rn.f32`: no contraction, no flush of subnormals; the
  bfloat16 head in a chunked `(D/16, V, 16)` layout; 1-row calls replay a CUDA graph). The logits are the CPU
  kernel's bit for bit, so the plaintext reference needs no change (it stays on the CPU, as do the final
  norm and sampling). Start-up self-check against the CPU kernel (Gaussian, large, subnormal products, zeros,
  signed zeros, infinity), else the CPU; `TRITON_INTERPRET=1` tests the order on any host
  (`tests/engine/test_lm_head_gpu.py`). T4: 1.33 ms vs 2.93 ms on the CPU kernel in isolation, 0 of 20 rows
  differ. The GPU belongs to the gateway (as for the local linears); the provider sees nothing new.
* `/admin/status` speed: `local_linear_timing` (wall time of the local linears, decode vs prompt passes; the
  backend's `local_seconds` also counts the masked trips' gateway work), `local_gpu_stats`, `lm_head` (with
  live `stats`). `PGW_MODAL_APP_SUFFIX` (deploy time) suffixes the app and Dict names for parallel runs.

Measured (T4:2, 14 cores, `attention_down`, TLS lane, factory capped and kept busy, 45 s of back-to-back
52-token chats; every run logprobs bit-identical, text equal, repeat identical; hosts vary 10-20% and more):

| run | back to back (tok/s) | warm chat | ms per decoded token | trips | trips' gateway work | local linears | LM head |
|---|---|---|---|---|---|---|---|
| before (this code with `PGW_LOCAL_GPU_NATIVE=0 PGW_LOCAL_GPU_ONE_ROW=0 PGW_LM_HEAD_DEVICE=cpu PGW_LOCAL_GPU_TABLE=0`) | 10.9 | 9.6 | about 75 | 14.5 | 12.6 | 15.9 (220 us/call) | about 4 (CPU) |
| native call + GPU LM head | 15.3 | 13.5 | about 45 | 12.1 | 10.2 | about 7 (native part 5.5) | about 1.5 |
| + 1-row fast path | 14.5 | 12.1 | about 52 | 12.7 | 11.3 | not measured | about 1.5 |
| + T4 GEMV table, LM-head graph | 16.1 | 13.0 | about 48 | 11.9 | 10.1 | 7.1 (99 us/call) | 1.4 |

The same native call takes 61 us mean (48 us median) in isolation and 99 us in the running gateway; idle gaps
of 100-600 us between calls add only 1-6 us in isolation, so the rest is the busy container (the factory,
spinning pool and lane threads of both parties on the same cores, gVisor), not the device. What remains per
decoded token (about 48 ms): trips 12 ms, the masked trips' gateway work 10 ms, local linears 7 ms, LM head
1.4 ms, float ops and Python about 17 ms; time to first token 0.5-0.55 s. Not reached: 30 tokens/s.

### Lean per-trip path and the UDS lane (2026-11-22, package T3b)

* `PGW_TRIP_FAST=1` (gateway; library default off, `colocated_gateway_env` turns it on; `--gateway-env
  PGW_TRIP_FAST=0` for an A/B): see `MaskedBackend.trip_fast`. Same bytes on the wire, same checks on every row,
  same noise statistics.
* `PGW_NOISE_ASYNC_SAMPLED=1` stays **off** in the demo too: it lets a sampled call's own payload leave before its
  noise statistics finish, so an alarm would block only the next payload. That is weaker than the inline check and
  saves only ~30 µs per trip on average.
* `--lane uds` (co-located): the provider's service serves its native lane on `/srv/pgw-lane/lane.sock`
  (`PGW_SERVICE_LANE=uds`, `PGW_SERVICE_LANE_PEER_UIDS` = the gateway user's uid only); the directory is the
  provider's, 0710 with the gateway user's group, and root widens the socket from 0600 to 0660 for that group once
  the service has bound it. The gateway's `lane = "auto"` takes the UDS ticket of its loopback service; prefill
  calls above 8 rows go to the stream channel (`ineligible:rows`), as on the TLS lane's fallback.

Measured (T4, `attention_down`, native float ops, factory capped `PGW_FACTORY_CPU_THREADS=3`, `PGW_FACTORY_NICE=10`,
`PGW_FACTORY_TARGET_SLOTS=8000`; all logprobs bit-identical, text equal, repeat identical; ms per decoded token =
(chat - TTFT) / 31 for the second and third chat; each run about 7 min, about $0.08):

| run | 1-row trip | warm chat | loopback | ms/decoded token |
|---|---|---|---|---|
| TLS, new paths off | 0.54-0.56 ms | 12.4 tok/s | 13.0 | 54.8, 53.1 |
| TLS, new paths on | 0.56-0.57 ms | 13.1 | 14.3 | 53.6, 51.9 |
| UDS, new paths on | 0.44-0.47 ms | 11.9 | 13.6 | 52.8, 52.9 |

The UDS lane saves about 0.1 ms per trip (2.6 ms per token at 26 trips) but the runs land on different hosts, and
host-to-host variance (10-20%) hides it in tokens/s; `--lane tls` stayed the default then (now `uds`, see below).

### Decode rate, contention and the factory yield (2026-11-22, package S4)

* **Decode-rate metric.** The e2e now also streams the warm chat twice on loopback (`wizard_client.chat_stream`)
  and reports `speed.warm_stream_ttft_s`, `warm_stream_decode_tokens_per_s` = (n-1)/(t_last - t_first) over the
  token chunks, `warm_stream_decode_ms_per_token` and `warm_stream_wall_s` (`streams` has both); the streamed
  answers must equal the first chat's text and logprobs (`stream_identical`, part of `ok`). Existing fields
  unchanged.
* **In-run A/B** (`--gateway-env PGW_MODAL_AB=base,nofactory,factory,pin,unpin,...`, diagnostics): between
  streamed chats of one run, root pauses (`SIGSTOP`/`SIGCONT`) or re-pins the party processes; each phase reports
  decode ms/token, TTFT, the 1-row trip split, local-linear µs per call and the busiest gateway threads. This
  removes the 10-20% host-to-host noise from comparisons.
* **Core pinning is not possible on Modal**: `sched_setaffinity` on any party thread fails with `EPERM` even
  as root (gVisor), so the `pin` phases ran unpinned (36 ms/token either way).
* **The factory was the contention**: with the factory paused, decode went from 35.8-36.3 to 31.8-33.3 ms per
  token in the same container (twice, in alternation), TTFT 0.5-0.9 -> 0.44-0.51 s. The factory's encrypted
  maskgen keeps the provider's GPU busy (provider request time 110 s per 44 s segment, 2.5 requests in flight)
  and moves gigabytes of ciphertext through loopback TLS (5.4-5.9 cores in the factory process while a chat
  decodes).
* `PGW_SERVICE_TRIP_STREAM_PRIORITY=high` (provider; library default `0`): the decode graphs (the lane's
  per-linear graphs, the executor's bucket graphs) are captured on a stream of the GPU's highest priority, so a
  trip's kernels are dispatched ahead of queued maskgen GEMM blocks. Scheduling only: same kernels and bytes.
* `PGW_FACTORY_YIELD_FILE` (+ `_MIN_SLOTS`, `_HOLD_S`, `_MAX_S`; library default unset = off;
  `pgw.core.factory_yield`): the gateway refreshes the file every 50 ms while masked passes run (decode and
  prompt passes) and the mask stock is at least `PGW_FACTORY_YIELD_MIN_SLOTS`; the encrypted factory waits before
  each encrypted batch and each provider request while the file is fresher than 0.25 s (at most 30 s at a
  time). Below the stock threshold it never yields, so its capacity (about 120 slots/s, 4x a 30 tokens/s chat)
  is unchanged; between chats it runs at full speed. Same masks, requests and bytes; the provider already sees
  every decode trip, so the pauses tell it nothing new. The factory log reports `yielded to decode N s`.
* Co-located demo defaults now (all measured exact below): `--lane uds`, `COLOCATED_PROVIDER_ENV`
  (trip stream priority high) and in `colocated_gateway_env` the factory yield at 1000 slots
  (`--lane tls`, `--provider-env PGW_SERVICE_TRIP_STREAM_PRIORITY=0`, `--gateway-env PGW_FACTORY_YIELD_FILE=`
  restore the previous settings).

Measured (T4:2, `--gateway-gpu`, factory capped `PGW_FACTORY_CPU_THREADS=3`, `PGW_FACTORY_NICE=10`, target 8000;
32-token chats, 41-token prompt; every run logprobs bit-identical, text equal, repeat identical, stream identical;
decode = streamed (n-1)/(t_last - t_first), three chats per phase):

| run | phase | decode ms/token | TTFT s | 1-row trip (service / wire) | factory cores |
|---|---|---|---|---|---|
| 1: TLS lane, before (e4198eb+metric) | base | 35.8, 36.3, 35.7 | 0.52-0.78 | 0.44 ms (0.16 / 0.28) | 5.9 |
| | factory paused | 33.3, 32.0, 31.8 / 33.1, 31.8, 31.8 | 0.44-0.51 | - | 0 |
| | pin (EPERM: no change) | 36.1, 35.6, 36.7 | 0.53-0.91 | - | - |
| 2: UDS + trip priority + factory yield | base | 31.1, 30.3, 29.7 / 30.5, 28.6, 30.4 | 0.36-0.61 | 0.29 ms (0.105 / 0.18) | 0.15 |
| | factory paused | 29.4, 30.7, 29.3 | 0.38-0.44 | 0.28 ms (0.105 / 0.17) | 0 |
| 3: new demo defaults, merged with the prompt-pass work (1aa41dc) | base | 31.9, 30.4, 29.7 / 30.7, 30.7, 29.4 | 0.38-0.60 | 0.29 ms (0.106 / 0.18) | 0 |

Run 3 (no extra flags beyond the factory cap): warm stream 29.9 ms per decoded token = 33.4 tokens/s after the first
token, TTFT 0.41 s; warm 32-token chat 24.8 tokens/s wall through Modal's edge (24.8 on loopback).

Tried for the prompt pass, not adopted: the provider's bucket CUDA graphs for the 41-row prompt items
(`--provider-env PGW_SERVICE_CUDA_GRAPHS=true --provider-env "PGW_SERVICE_CUDA_GRAPH_BUCKETS=[32,64,128,256]"`;
run 4, exact): the prompt pass's service wait stayed bimodal at about 200 or 400 ms per 24 trips (pass 375-578 ms,
TTFT 0.39 s warm), so the time is not the eager kernels; the stream channel's handling of multi-row items is the
next suspect (not done here).

Run 2's warm stream: 30.3 ms per decoded token = 33.0 tokens/s after the first token, TTFT 0.46 s (warm chat
wall 24.4 tokens/s through Modal's edge, 24.6 on loopback; run 1: 27.9 tokens/s streamed, 19.9 / 15.7 wall). With
the yield, pausing the factory changes nothing measurable any more (base and paused phases overlap). The factory
still made 73.7 slots/s in the segment that overlapped the e2e's back-to-back chats (22 s yielded), 2.4x the
decode demand during continuous chatting, and full speed otherwise. Local linears on the gateway GPU stay at
about 79 µs per call (factory paused or not): that part is not contention. The lane client's Python per trip is
7 µs (`LaneComputeClient.compute_into` with an instant fake channel), not worth trimming.

### Factory keep-up under back-to-back chats (2026-11-22, package S5)

**Problem (lead's run, S4 defaults).** 62 back-to-back chats (41-token prompt + 52 tokens, about 93 mask slots each,
about 44 slots/s) for 130 s, then HTTP 503 `mask_exhausted`. Three causes, all policy, none capacity: (1) the yield
was all or nothing at 1000 slots, so the factory made nothing while chats decoded above it; (2) the factory refilled
only below its low water (1000; the wizard's default) and re-polled every 30 s; (3) below 1000 slots a 4096-slot
segment (33 s unloaded, longer under decode) could not land before the stock ran out.

**Changes.**

* `pgw.core.factory_yield` (library defaults unchanged: all or nothing): `PGW_FACTORY_YIELD_FULL_SLOTS` turns on
  a **partial yield**: between the low mark and FULL the gateway refreshes the yield file only during the first
  `(stock - low) / (FULL - low)` of each `PGW_FACTORY_YIELD_PERIOD_S` cycle, so the factory's duty cycle grows as
  the stock falls. `PGW_FACTORY_YIELD_HORIZON_S` raises the low mark to the observed consumption rate (stock
  decreases over 30 s) times the horizon, so a store draining fast never yields below the stock a segment needs to
  land. `PGW_FACTORY_YIELD_FULL_HORIZON_S` lowers the full mark to that many seconds of consumption: an occasional
  chat consumes little, so it still gets the full yield (single-chat decode unchanged); only sustained consumption
  brings the duty cycle in. Privacy: the yield timing is a function of the decode activity (which the provider sees
  as trips) and the stock (which follows from the maskgen requests the provider serves and the trips that consume
  the slots); same masks, requests and bytes.
* Co-located demo (`pgw_modal_boot.COLOCATED_FACTORY_KEEP_UP`): band 2500-6000 slots, horizon 45 s, full horizon
  120 s, period 3 s; `PGW_FACTORY_SEGMENT_SLOTS=2048` (lands in about 40 s under load instead of 80+);
  `PGW_FACTORY_LOW_WATER_SLOTS=6144`, `PGW_FACTORY_TARGET_SLOTS=8192` (refill whenever below 6144);
  `PGW_FACTORY_POLL_INTERVAL_S=2`. The factory's threads and nice stay as passed (`--gateway-env
  PGW_FACTORY_CPU_THREADS=3 --gateway-env PGW_FACTORY_NICE=10` in the runs below; an unprivileged process cannot
  lower its nice again, so no adaptive nice).
* `--sustain-s` now streams every chat on loopback (`decode_tokens_per_s_*` per chat and token-weighted
  `decode_tokens_per_s_overall`; `identical`: every chat's text and logprobs equal the first's, whose first 32
  tokens equal the exactness-checked chat; part of `ok`), samples the stock every 2 s (`stock`), and `keep_up`
  gives consumption = (stock at start + slots committed - stock at end) / window, the factory's committed slots per
  second over the window (quantized: one 2048-slot segment is 13.6 slots/s over 150 s), the segment-to-segment rate
  and the stock's least-squares slope. `factory.committed` lists every segment that landed in the window (the old
  `segments` list only those that also started inside it, hence `segments: []` with long segments).
  `--gateway-env PGW_MODAL_SUSTAIN_MIN_STOCK=N` starts the window once N slots are in stock (at most 240 s).

Measured (T4:2, `--gateway-gpu`, factory capped 3 threads / nice 10; 41-token prompt, 52-token answers; every run
logprobs bit-identical, text equal, repeat identical, stream identical, all sustain chats identical):

| run | window | chats | consumption | factory (window / segment-to-segment) | stock start -> end (min) | stock slope | decode tok/s median / token-weighted (min) | warm stream |
|---|---|---|---|---|---|---|---|---|
| lead, S4 defaults | 130 s, then 503 | 62 | about 44 slots/s | 0 segments landed | -> 37 | - | (24.9 overall wall) | 33 |
| 1: partial yield, 2048 segments | 150 s | 74 | 51.7 slots/s | 40.9 / 50.5 | 6547 -> 4921 (4133) | - | 32.7 / - (18.8) | 30.5 |
| 2: same, longer | 302 s | 146 | 51.0 slots/s | 47.6 / 50.6 | 6547 -> 5553 (4081) | -1.1 slots/s | 32.4 / 31.7 (17.0) | 30.3 |
| 3: + full horizon 120 s (demo default) | 242 s | 113 | 49.3 slots/s | 42.5 / 47.4 | 6547 -> 4922 (3766) | -2.2 slots/s | 30.6 / 30.6 (22.4) | 31.6 |

The stock settles into a sawtooth between about 4100 and 6300 slots (a 2048-slot segment lands every 40-45 s; troughs
4342, 4185, 4133, 4081, 4344, 4187, 4240 in run 2), i.e. production equals consumption at about 51 slots/s with the
factory yielding about half of the time; no exhaustion. Over a window that starts at the top of the sawtooth the
committed production reads a little below consumption (the stock starts above its equilibrium); the segment-to-segment
rate and the slope (-1.1 slots/s over 300 s, +422 slots over the second half) show steady state. Decode with the
factory's duty cycle: 30.6-32.4 tokens/s token-weighted (run 3 ran at 29-31 for its middle 100 s on that host); a few chats dip to 17-25 tokens/s (2-3 per 150 s). Factory 2.0-2.1 cores.

Lever 3 (the factory's HE on the gateway's T4) is measured in the next section: opt-in, not the demo default.

### Factory HE on the gateway GPU (2026-11-22, opt-in)

* `PGW_FACTORY_HE_ENCRYPT=gpu_ntt`, `PGW_FACTORY_HE_DECRYPT=gpu_ntt` (each independent; library and demo default off):
  `HEMaskBackend` runs encryption and/or decryption on `pgw.hemask.gpuntt.GpuNTTHE` (A in NTT form on the device,
  hint byte planes cached on the device, fused Triton kernels). Start-up: the engine's float64 self-test, then a
  bit-for-bit cross-check against the backend's own CPU paths on public random data (encryption with fixed public
  test seeds vs `ntt_pre`; decryption of a partial and a full group vs the int8-engine dense path); any mismatch or
  error keeps the CPU paths, and a GPU call that fails is redone on the CPU (CPU from then on). Device work on its
  own lowest-priority stream under one lock; secrets and noise sampled on the CPU as before; every secret-dependent
  device temporary zeroed (as in devmm/gpuntt; a failed GPU encryption also wipes its earlier passes' secrets). A GPU
  decryption pass also waits on the factory yield, like encryption and the provider requests. The provider sees the
  same ciphertexts; every decrypted row still goes through the factory's Freivalds check.
* `PGW_LOCAL_GPU_STREAM_PRIORITY=high` (default 0): the gateway's local-linear graphs and LM head on the GPU's
  highest-priority stream. **Caveat:** stream priorities order work inside one CUDA context; the factory is a
  separate process, so the GPU time-slices the two contexts and the priority does not protect decode from it.
* `PGW_THREAD_NAMES=1` (default off): the factory's stage threads carry kernel thread names; `--sustain-s` now reports
  `sustain.threads.factory` (cores per stage) and the busiest gateway threads; the segment log names the HE engine.

Measured (T4:2, `--gateway-gpu`, factory capped 3 threads / nice 10, sustain window from 6500 slots, 240 s, 41-token
prompt + 52 tokens; both runs logprobs bit-identical, text equal, stream identical, all sustain chats identical;
the self-test and cross-check passed, every segment on `gpu_ntt:cuda`):

| run | sustained decode tok/s median / token-weighted / min | chats < 26 tok/s | consumption vs factory (window / seg-to-seg) | stock min / end | factory cores (stages) | warm stream |
|---|---|---|---|---|---|---|
| S5 run 3 (CPU factory, earlier host) | 30.6 / 30.6 / 22.4 | 2-3 per 150 s | 49.3 vs 42.5 / 47.4 | 3766 / 4922 | 2.0-2.1 | 31.6 |
| A: gpu_ntt both, no yield on decryption | 31.3 / 30.8 / 25.7 | 1 of 113 | 49.7 vs 42.9 / 46.2 | 4028 / 4922 | 0.88 (remote 0.30, decrypt 0.10, encrypt 0.09) | 26.5 (37.7 ms) |
| B: + decryption waits on the yield | 31.6 / 31.3 / 25.5 | 2 of 117 | 51.1 vs 42.6 / 48.4 | 3836 / 5674 | 0.92 (remote 0.36, decrypt 0.16, encrypt 0.05) | 25.3 (39.5 ms) |

In-run A/B of run B (3 streamed 32-token chats per phase while the stock ramps, i.e. the factory unyielded):
factory paused 28.9-29.3 ms/token, local linear 79 µs/call; factory running 30.6-38.2 ms/token, local linear
108-119 µs/call, 1-row trip 0.27 -> 0.30 ms. So with its HE on the GPU the factory costs about 1 gateway core less,
sustained decode is unchanged within host noise (fewer slow chats), keep-up is unchanged (the yield policy and the
provider's maskgen set production, not the gateway's HE), but whenever the factory runs unyielded (stock ramp,
refill below the yield band) its kernels time-slice with the 1-row local-linear graphs (+30-40 µs per call, +2-9 ms
per token); the warm single chat ran during such a ramp in both runs. A same-day CPU-factory A/B run was not made
(spend cap), so the CPU factory's cost in the same unyielded phases is from S4 (36 -> 32 ms/token, older config).
The largest remaining factory stage is the provider-request thread (TLS receive of the replies, 0.3-0.36 cores).

### Larger targets: Qwen2.5-1.5B and 7B co-located (2026-11-22, package L1)

`--model qwen2.5-1.5b-instruct` and `--model qwen2.5-7b-instruct` now run the co-located demo end to end. Fixes and
additions (all exact; library defaults unchanged):

* **Enrollment listed the 0.5B whatever the target** (`create_enrollment_code` used the module's default name), so
  a 1.5B/7B co-located pair failed with `none of the models of this enrollment code is served`. The code now lists
  the deployed target (`PGW_MODAL_MODEL`).
* **7B sizing** (`colocated.py`, co-located only; `launch.py` refuses the 7B elsewhere and deploys only
  `colocated.py` for it, no tunnel fallback): 16 GiB provider + 48 GiB gateway (a request, not a limit); Modal's
  default ephemeral disk (at least 512 GiB) holds the checkpoint, both packs, the 34 GB hint and the mask store.
  `PGW_MODAL_DEPLOY_COLOCATED_ONLY=1` does the same for any target (faster deploys for experiments).
* **Factory keep-up per target** (`pgw_modal_boot.factory_keep_up`): the S5 slot counts divided by 2 (1.5B) or 4
  (3B, 7B), since a slot is 2-6x the 0.5B's bytes (segments land in about the same time).
* **Bulk maskgen channel on in the demo** (`--he-bulk`, default on co-located, `--no-he-bulk` = HTTP only): the
  provider serves `PGW_SERVICE_MASKGEN_BULK_PORT=8445` on 127.0.0.1 and the factory uses it
  (`PGW_FACTORY_HE_BULK_PORT`, split 2). The service now has **one** `ByteBudget` (thread-safe) shared by the compute
  routes, `POST /pgw/v1/maskgen` (which had none) and the bulk channel (reserved before a body is read, released after
  the reply; a body that does not fit gets `unavailable` and the connection is closed). Library default stays off.
* **Encryption hint on the gateway's GPU** (`PGW_HINT_DEVICE=cuda`, library default off; on with `--gateway-gpu`):
  `pgw.hemask.hint_gpu` computes `to_planes(W8 A mod 2^64)` with int8 byte-plane GEMMs (exact int32 sums, wrapping
  int64 recombination), self-checked against the CPU path on the pack's linear shapes first. Same files bit for bit
  (tested). 7B: 213 s on a T4 vs about 20 min on 16 vCPUs (the CPU-only run below); 1.5B: 74 s.
* The e2e reports every factory segment of the run (`e2e.factory_all`: alone before the chats, and during them).

Measured (Modal, `--region any`, every run logprobs bit-identical, text equal, repeat identical, stream identical,
all sustain chats identical; 41-token prompt, 32/52-token answers; bulk channel on; factory threads 6-8, nice 10):

| run | placement | decode tok/s (warm stream / sustain token-weighted) | ms/token | TTFT s | trips/token | 1-row trip (service / wire) ms |
|---|---|---|---|---|---|---|
| 1.5B, 2xT4, `--gateway-gpu`, attention_down | gateway GPU | 20.0 / 19.0 | 50.0 | 0.81 | 28 | 0.51 (0.28 / 0.23) |
| 7B, 2xT4, `--gateway-gpu`, attention_down | gateway GPU | 10.0 / 9.8 | 99.7 | 1.49 | 28 | 1.24 (0.99 / 0.25) |
| 1.5B, 1xT4, CPU gateway, `--local-linears off`, 2 factory workers | CPU-only customer | 9.5 / 9.7 | 104.8 | 1.72 | 112 | 0.34 (0.14 / 0.20) |
| 7B, 1xT4, CPU gateway, `--local-linears off` | CPU-only customer | 5.0 / 4.7 | 198.6 | 3.1 | 112 | 0.97 (0.57 / 0.40) |

Per decoded token (gateway GPU runs): 1.5B = trips 14.2 ms + local qkv/o/down 11.9 ms (84 calls, 142 us each) +
LM head 2.2 ms + mask prepare/finish, float ops and Python 21.7 ms; 7B = trips 34.8 ms (the provider T4's gate_up
GEMV is 0.99 ms of each) + local 31.4 ms (374 us per call: 2.7 GB of int8 per token on the gateway T4) + LM head
4.6 ms + the rest 29 ms. Both T4s are bandwidth-bound: about 3.8 GB per token on each side.

Mask factory (encrypted, CPU decryption, bulk channel):

| run | slot (bytes) | alone (no chats) | during back-to-back chats | consumption | stock |
|---|---|---|---|---|---|
| 1.5B, gate_up only | 2.0 MB | 118-142 slots/s (1024-slot segments) | 42-58 slots/s (yielding 14-30 s per segment), segment-to-segment 43.2 | 43.6 slots/s | 4499 -> 2321 over 122 s (band 1250-3000) |
| 7B, gate_up only | 4.2 MB | 39-46.5 slots/s (512-slot segments) | 18.1-18.7 slots/s (yielding 23-27 s of 28 s), segment-to-segment 18.5 | 18.6 slots/s | 1939 -> 1165 over 124 s |
| 1.5B, all linears, 2 workers | 2.6 MB | about 57 slots/s (two workers' overlapping segments, 22.9-54.5 each) | 8-30 slots/s (yielding 30-35 s per segment) | 22.8 slots/s | 3475 -> 1664 over 125 s |
| 7B, all linears, 1 worker | 5.6 MB | 17.9-18.2 slots/s (512-slot segments; provider maskgen 37 s and the every-row check 27 s per segment, overlapped) | 9.7-13.4 slots/s while yielding | 9.1 slots/s | 2451 -> 1611 over 93 s |

During chats the factory is held back by the keep-up yield (S5) while the stock is above its band, not by capacity:
alone it makes 2.4x (7B gate_up) to 3x (1.5B) the consumption. The earlier Kaggle 7B figure over HTTP was 6.36 slots/s
for every linear; the bulk channel was not A/B-tested against HTTP on Modal here (no budget left), so the gain is
not isolated from the host and skip-kind differences.

**CPU-only customer gateway** (the two `--local-linears off` rows: no gateway GPU, every linear on the provider, LM head
and float ops on the gateway's CPU, `--gpu T4` for the provider only; container about $1.4/h for the 1.5B, $1.8/h for the
7B). Per decoded token of the 7B: 112 trips x 0.97 ms = 109 ms (the provider's 1-row GEMV 0.57 ms, wire 0.40 ms) and
about 90 ms of gateway CPU, i.e. about 0.6-0.8 ms per trip at 7B vector sizes (mask prepare/finish over 3584-37888
outputs, the native float ops) plus the bfloat16 LM head (1.09 GB read per token on the CPU). The warm chat ran while
the factory used 15 of 16 gateway cores. 1.5B: 112 x 0.34 ms = 38 ms of trips and 67 ms of gateway CPU. The fleet
(`PGW_FACTORY_WORKERS=2`) ran in the 1.5B run (two workers' segments interleaved, about 57 slots/s alone).

What the customer stores, 7B (CPU-only gateway as run): checkpoint download 15.2 GB (deletable after packing), int8
pack 7.6 GB (used by the hint build and the Freivalds keys; not by decode with `local_linears = "off"`), bfloat16
embedding and LM head 2.2 GB, encryption hint 34.2 GB (24.6 KB per output row of every linear: rank 16 x 256 x 6
bytes), masks 5.56 MB per slot (only `W r` mod 2^32 is stored; `r` already comes from a per-segment seed), so the
demo's 2048-slot target is 11.4 GB. The audit log also shows 25-33 reserved slots burned per chat (reserved for
`max_tokens`, not used): 24-31% of the supply in these chats.

### CPU-only 7B gateway, strict speculation, H100 provider: the decisive run (prepared 2026-11-22)

The launcher is ready for one co-located measurement of the configuration of
`research/plans/strict-speculation-cpu-gateway.md` (section 7): Qwen2.5-7B on the provider's H100, a CPU-only,
weight-free customer gateway (every linear remote, native float ops, the native per-trip path), the Qwen2.5-0.5B draft
on the gateway's cores (`draft_backend = "native_int8"`), strict speculation (R = 2, pipelined m = 0), the
one-vector Freivalds check over F_p (`p31`), and, in the same container, the same gateway without speculation as the A/B. Everything new is opt-in (launcher flags;
library defaults unchanged).

**New launcher options** (co-located only unless noted):

| option | what | where it goes |
|---|---|---|
| `--gpu H100` (existing) + no `--gateway-gpu` | the provider's GPU; the gateway's process sees none | deploy |
| `--gateway-cpus N`, `--provider-cpus N`, `--gateway-mem-gib N`, `--provider-mem-gib N` | container sizing (requests) | deploy (`PGW_MODAL_GATEWAY_CPUS` ...) |
| `--draft-backend native_int8` | the draft's CPU kernels (`GatewayModelConfig.draft_backend`) | wizard `models.json` |
| `--spec-pipeline-steps M` | pipelined drafter (`spec_pipeline_steps`; prompt lookup off, as the config requires) | wizard `models.json` |
| `--spec-mask C[/B]` (existing, now also at run time) | padded strict speculation, public charge of C slots per pass | wizard `models.json` |
| `--trip-native` | `PGW_TRIP_NATIVE=1` next to the demo's `PGW_TRIP_FAST=1` | gateway environment |
| `--decode-native` (with `--trip-native`) | `PGW_DECODE_NATIVE=1`: the native decode driver (a decode pass's decoder layers in one GIL-free native call; STATUS 2026-11-22 "native decode driver") | gateway environment |
| `--prime-check` | `PGW_FACTORY_CHECK_FIELD=p31` + `PGW_GATEWAY_PRIME_CHECK=true`; **refused with `--spec-mask`** (padding rows refuse prime extents, fail closed) | factory + gateway environment |
| `--ab-plain S` | after the speculative phase: rewrite `models.json` without the speculation keys, restart the gateway process (same pack, mask store, keys, service, enrollment; normal mode from the finished wizard), 1 warm-up + 3 streamed chats, then `S` seconds back to back | `colocated.py` |
| `--gateway-cache`, `--prepare-cache` | the gateway side's cache Volume (below) | deploy / one-off T4 container |
| `--allow-self-draft` | the 0.5B target drafting for itself (plumbing tests only) | deploy |

**What the e2e now reports** for both phases (`e2e.speed`, `e2e.sustain`, `e2e.plain`, `e2e.plain.sustain`; a summary
is printed at the end): streamed decode tok/s (warm chats; per chat and token-weighted over the sustain window), TTFT,
exactness (`logprobs_bit_identical`, `text_equal`, `stream_identical`, sustain `identical`, plain-phase `identical`, all
part of `ok`), trips per token, the 1-row trip split (service / wire), **gateway process CPU per trip**
(`sustain.derived.gateway_cpu_ms_per_trip`: the gateway's cores x seconds / trips, factory excluded), **mask slots per
generated token** (`derived.slots_per_token`: consumption x window / tokens, prompts and burned reservations included),
**factory slots/s against consumption** (`derived.factory_vs_consumption`: window, commit-to-commit, stock slope), and for
speculation **tokens per verify pass**, draft rows per pass, the pipelined drafter's hit rate, draft ms per background
step and the foreground wait (`stream_spec`, `sustain.spec`; from new gateway-local counters `spec_stats` /
`drafter_stats` in `/admin/status`, never sent to the provider). With strict speculation each chat starts after the
previous answer's dummy passes have drained (`spec_drain_s`), so TTFT is not inflated by the queue. Also new:
`--ab`-style `PGW_MODAL_AB` phases now run with speculation too.

**Gateway cache (`--gateway-cache`): why it holds no customer secret.** A Modal Volume (`pgw-gateway-cache`,
`PGW_MODAL_CACHE_VOLUME`) mounted at `/cache` (root, 0700) in the co-located container, read and written only by the
root parent process; neither party's process sees it. It holds two public artefacts per target:

* the model pack: open weights, quantized deterministically. The wizard uses a restored pack only with
  `PGW_SETUP_REUSE_PACK=1` (new, library default off) and only if `ModelPack.open(verify="full")` (every tensor, the
  tokenizer and the linears hashed) gives the fingerprint of the enrollment; otherwise it deletes it and downloads and
  packs as before (tested: `tests/gateway/test_setup_pack_reuse.py`, also a tampered tensor).
* the encrypted-mask hint `H = W8 A` (SPEC 7.9): `W8` is the int8 weights the provider holds anyway and `A` the public
  matrix expanded from a seed. Decision-MLWE keeps `b = A s + e + Delta r` hiding `r` with `A` public; the secrets `s`
  and `e` are fresh per 256-slot group, sampled at encryption time and never part of the hint. So `H` is a function of
  public data only, and reusing `A` across runs is reusing a public matrix (as every LWE scheme does). What must hold
  is that the **customer side chose the seed** (a provider-chosen `A` could expose the masks, SPEC 7.9): it is drawn
  with `new_seed()` by the gateway's own hint build or by `prepare_cache`, a T4 container in which no provider process
  runs.

Not done, on purpose: building the hint on the provider's H100 and handing it over. Secret-free is not enough there:
the customer would decrypt with a provider-supplied `H`, and a provider could perturb rows of `H` so that whether a
group's decryption passes the factory's every-row check depends on bits of that group's `s` (a selective-failure
attack: it learns those bits from which segments land), unless the customer re-derives `H` to check it, which costs
the build it would save. The cache's integrity rests on the Modal workspace (who can write the Volume), the same trust
the demo already places in whoever controls both containers; a real deployment keeps it on customer storage.

`launch.py --prepare-cache` (with the run's deploy flags plus `--gateway-cache`) fills the cache once on a T4 container
(8 cores; the pack copied from the image, the hint's GEMMs on the T4 with `PGW_HINT_DEVICE=cuda`: the CPU path's bits,
self-checked first; 213 s for the 7B measured in L1). A co-located run with `--gateway-cache` then copies both into
the gateway's data directory before the gateway starts (`cache_restore` in the result) and saves what the cache lacks
after the run (`cache_save`; `--gateway-env PGW_MODAL_CACHE_SAVE=0` skips that). Never cached: the mask store, the
factory's state, keys, tokens, certificates.

**Mask factory sizing (estimate).** At 7B with every linear remote a slot is 5.56 MB on disk, and the encrypted
factory moves about 11.5 MB of ciphertext per slot through the provider (2.8 GB of `b` up and 3.1 GB of `y` down per
512-slot segment: `sum(n_in) = 831k` and `sum(m_out) = 1.39M` over the 28 layers at 53 and 35 bits per value). The
T4 run's 37 s of provider maskgen per 512 slots is therefore mostly host work (TLS, unpacking), not the GEMM (about
4.7e13 int8 operations per segment: under 1 s on an H100), so the H100's device time per segment is estimated at 1-3 s
and the provider side's limit becomes its cores: give it 8 (`--provider-cpus 8`, bulk channel split 2). The gateway's
side per worker was 27 s of every-row check per 512 slots (overlapped with maskgen; 18 slots/s per worker), so four
workers (`PGW_FACTORY_WORKERS=4`, 4 threads each) are estimated at 60-75 slots/s if 16 gateway cores are free for them.
Demand: strict speculation charges C = 2 slots per pass whatever the acceptance; at about 20-25 passes per second plus
41-slot prompts that is about 50-65 slots/s. So the run (a) pre-fills a stock of 9000 slots (50 GB; the sustain
window waits for it, `PGW_MODAL_SUSTAIN_MIN_STOCK`, at most 240 s) so a 150 s window cannot run dry, and (b) runs the
factory unyielded at nice 10 (`PGW_FACTORY_YIELD_FILE=` empty), so the window shows the factory's real production next
to the consumption instead of a policy-limited one (decode then shares the cores with it; nice keeps decode first).

**p31 with speculation: masked fillers, not padding.** Padded calls (`spec_mask_rows`) refuse p31 extents (fail
closed), but strict speculation without `--spec-mask` sends its filler rows as ordinary masked rows, which take the
TripEngine / native lease paths that accept p31. At R = 2 the two modes cost the same: 2 slots per pass whatever the
acceptance (C = 2 charges `prompt + 2j` by pass j; masked fillers take R = 2 rows every pass), and the provider sees R
rows per pass and as many passes as tokens either way. So the run uses `--spec-rows 2 --prime-check` without
`--spec-mask` (validated exact on the T4 below, every segment `p31`); with R > 2 and a credit cap, padding (and so the
ring check) is the cheaper choice in slots.

**The command** (one run; validated end to end at 0.5B on a T4 below; the 7B numbers are the question it answers):

```bash
cd pgw/deploy/modal
# deploy flags shared by both steps (deploy time: GPU, sizes, draft, cache)
FLAGS="--placement colocated --gpu H100 --model qwen2.5-7b-instruct --spec-rows 2 --spec-draft --gateway-cache \
  --gateway-cpus 32 --provider-cpus 8 --gateway-mem-gib 80 --provider-mem-gib 24"
# 1) once: the gateway side's pack + hint into the cache Volume (T4 container, ~6-10 min, about $0.15-0.25)
PGW_MODAL_DEPLOY_COLOCATED_ONLY=1 python launch.py $FLAGS --prepare-cache
# 2) the measurement (redeploys the same image from cache in seconds; a finished e2e stops the app)
PGW_MODAL_DEPLOY_COLOCATED_ONLY=1 python launch.py $FLAGS --e2e --ttl-min 35 \
  --local-linears off --float-ops native --trip-native \
  --prime-check --draft-backend native_int8 --spec-pipeline-steps 0 \
  --sustain-s 150 --ab-plain 120 \
  --gateway-env PGW_MODAL_SUSTAIN_MIN_STOCK=9000 \
  --gateway-env PGW_FACTORY_TARGET_SLOTS=12288 --gateway-env PGW_FACTORY_LOW_WATER_SLOTS=11264 \
  --gateway-env PGW_FACTORY_SEGMENT_SLOTS=1024 --gateway-env PGW_FACTORY_WORKERS=4 \
  --gateway-env PGW_FACTORY_CPU_THREADS=4 --gateway-env PGW_FACTORY_NICE=10 \
  --gateway-env PGW_FACTORY_YIELD_FILE= --gateway-env PGW_POOL_WORKERS=8 --gateway-env PGW_THREAD_NAMES=1
```

Step 2's defaults already give `--lane uds` (since 2026-11-22 with the gateway's native UDS channel, `lane_native`,
and with `--trip-native` the per-trip native driver `PGW_TRIP_DRIVER=1`, which needs that channel; `--lane-python`
restores the Python channel and leaves the driver off), `--he-bulk` (split 2), `--float-threads 1`, the trip stream priority and
`PGW_TRIP_FAST=1`; `PGW_POOL_WORKERS=8` gives the native pool (trip kernels, float ops, the draft) 8 workers instead of
the demo's 4 (the native LM head runs its own threads, up to 16). Alternative arm if R = 2 falls short (trips above
0.4 ms): `--spec-rows 3 --spec-mask 3/96` without `--prime-check` (ring check; 1.9 instead of 3 slots per token).

**Cost estimate** (Modal list prices, `launch.py` prints the same before deploying): H100 $3.95/h + 40 cores x
$0.0473 + 104 GiB x $0.008 = **about $6.7/h**; the run is expected to take 22-28 min (container and provider start
4-5 min, cache restore 1-2 min, wizard with the reused pack and hint about 2 min, first masks, chats and the CPU
plaintext reference 3-4 min, stock pre-fill up to 4 min, 150 s speculative window, gateway restart about 2 min, plain
chats, a second stock wait (same `PGW_MODAL_SUSTAIN_MIN_STOCK`, at most 4 min, usually less with the factory unyielded)
and a 120 s window), so **$2.5-3.5, at most $3.9 at the 35 min TTL**; step 1 adds about $0.2 once (T4 $0.59/h
+ 8 cores + 32 GiB for about 10 min). Without the cache the gateway builds the hint on its own CPU (about 20 min, +$2.2).

**Expected outcome (model, not measured).** Without `--masked-head` (below) this run keeps the gateway's bfloat16 LM
head on its CPU: 1.09 GB read per pass, about 15-30 ms on 16 threads, on top of 112 trips. With trips at 0.3-0.35 ms (H100 GEMV well under the T4's
0.57 ms service time) and about 0.15 ms of native gateway CPU per trip, a pass costs about 55-85 ms: plain decoding
about 12-18 tok/s, R = 2 at about 1.6-1.85 tokens per pass about 20-30 tok/s. So 25 tok/s is plausible but not
certain with the CPU head; the head is the first lever if it falls short.

**Plumbing validation (T4, Qwen2.5-0.5B target with the 0.5B as its own draft, `--allow-self-draft`; 2026-11-22,
`PGW_MODAL_APP_SUFFIX=-prep`).** `--prepare-cache` on the T4: 68 s wall (pack 0.92 GB copied, hint 7.47 GB built on the
T4 in 36 s). Then two e2e runs with the 7B command's options scaled down (`--gpu T4 --gateway-cpus 8 --local-linears off
--trip-native --prime-check --spec-rows 2 --spec-draft --draft-backend native_int8 --spec-pipeline-steps 0 --gateway-cache
--sustain-s 30|40 --ab-plain 30|40`), 216 s and 235 s from spawn to result. Both: `ok`, logprobs bit-identical, text
equal, stream identical, every speculative and plain sustain chat identical, plain-phase chats identical; run 2's
factory log shows every segment `p31` (none `ring`); cache restore 9.2 s then 2.7 s for 8.4 GB (0.9-3.4 GB/s), the wizard
reused the restored pack (full verification) and found the hint (no build); gateway restart for the plain phase 10.5 s.

| run 2 | decode tok/s (token-weighted / median / min) | TTFT median | tokens per verify pass | drafter hit rate | trips/token | gateway CPU per trip | slots/token | consumption vs factory (window) |
|---|---|---|---|---|---|---|---|---|
| speculative, 40 s | 11.7 / 12.4 / 10.2 | 0.85 s | 1.96 | 0.48 | 97.9 | 5.7 ms | 3.3 | 16.0 vs 48.7 slots/s |
| plain, 40 s | 15.8 / 16.0 / 14.9 | 0.91 s | - | - | 97.9 | 1.35 ms | 2.1 | 26.2 vs 0 (yielding, stock 5.8k) |

These numbers say nothing about the 7B: a self-draft costs as much as the target (41 ms per background draft step on
the 8 shared cores, which is the 4.4 ms of extra gateway CPU per trip), so speculation is slower here by construction;
they show that every report is filled in. Trips per token are equal in both phases by design (strict passes: the
provider sees one pass per generated token). Slots per token include the 41-slot prompt of each 64-token chat. The
speculative phase's 1-row trip split is empty (every decode item has 2 rows); the plain phase's: 0.24 ms (service
0.07, wire 0.18). Spend for the prep and both runs: about $0.2.

### 1-row trip fixed cost: lane latency knobs (opt-in, 2026-11-23; CPU-measured, not yet on a GPU)

Per 1-row trip at 7B the H100 run spent about 0.46 ms, of which the device is 0.04-0.09 ms. The knobs below
attack the fixed part on both sides of the co-located UDS lane. All default off; the bytes, their order, the
tickets, the peer-uid check and what the provider sees are unchanged (the same 48-byte request header, 40-byte
reply header and `x_hat` / `y_hat` bodies, one call at a time per connection).

| knob | side | what |
|---|---|---|
| `PGW_LANE_SPIN_AUTO_US=N` | gateway | adaptive reply spin: per (linear, rows) the native channel learns the reply time (moving average) and busy-polls 1.25x that + 20 us, between the fixed `PGW_LANE_SPIN_US` and `N`; so gate_up (about 190 us of device time) no longer falls off a 50 us spin into a blocking wake-up. UDS polls are `recv(MSG_DONTWAIT)` (no `fcntl` toggling per trip). |
| `PGW_SERVICE_LANE_READ_SPIN_US=N` (existing) | provider | the connection thread polls for the next request this long before blocking (the gateway sends the next trip about 0.1-0.2 ms after a reply at 7B): no scheduler wake-up per request. One provider core per active connection while it spins. |
| `PGW_SERVICE_LANE_ZERO_COPY=1` | provider | OK replies leave straight from the CUDA graph's pinned `h_out` (non-blocking `writev` under the device lock; what the socket does not take at once is copied and sent after the lock, so a slow peer cannot hold the GPU): one reply copy less (151 KB for 7B gate_up). |
| `PGW_SERVICE_LANE_OWN_LOCK=1` | provider | the lane's graph replays take their own per-device lock instead of `DeviceExecutor`'s: a 1-row lane trip never queues behind a Python-path call (stream-channel batches, prefill) on the same GPU. The lane's graphs have their own stream, pinned and device buffers and graph pool (no shared memory); captures hold both locks. Not with the persistent worker. |
| `PGW_SERVICE_LANE_SHM=1` + `PGW_LANE_SHM=1` | both | shared-memory transport of a UDS connection (`rust/pgw-lane/src/shm.rs`): after the ticket `open`, the gateway asks with a `shm` frame; the service creates a sealed `memfd` (no path, sealed against shrink/grow) and passes it back over the authenticated socket (`SCM_RIGHTS`), so only these two processes ever hold it (the socket's uid check and single-use ticket are its access control). Each call: header + body written into the segment, a sequence word, spin then park on the socket as a doorbell (one byte, only when the other side is parked). The service copies the header and body out once and validates them as socket frames. A service without it answers `fallback` and the connection stays on the socket. |

Reporting: the service's `queue_us` in the reply header is now the device wait only (lock, or batch), not the
body read; `TripChannel.stats()` sums trips, spin hits, blocking waits, wait / trip time and the service's
`queue_us` / `compute_us`; `LaneComputeClient.channel_stats()` and `decode_cpu_bench.py --lane-proc` report them.

**Measured here** (shared 4-vCPU box, 3-4 cores of other load: spin-based results are unreliable; interleaved
runs, medians of 4-5; `benchmarks/lane_tls_bench.py --preset 7b`, the native listener with the native bench stub
`--stub-us`; "overhead" = trip minus the service's `compute_us`):

| scenario | variant | overhead p50 / mean (us) | client blocked |
|---|---|---|---|
| transport floor (stub 0, back to back) | UDS, defaults | 12.3 / 19-20 | 0.10 |
| | UDS, zero-copy | 11.9 / 18.0 | 0.10 |
| | UDS, server read spin 2 ms | 44 / 96 | 0.60 |
| | shm | 14.5-14.9 / 38-70 | 0.00-0.10 |
| H100-like device times (44/40/191/79 us), 140 us gateway work between trips | UDS, defaults | 28.7 / 40.1 | 0.50 |
| | UDS, adaptive spin (cap 400) | 21.1 / 74.5 | 0.30 |
| | + server read spin 2 ms | 25.1 / 249 | 0.30 |
| | shm (+ adaptive, server spin 2 ms) | 21.9 / 259 | 0.40 |

On the gateway's real native decode path (`decode_cpu_bench.py --preset 7b --layers 2 --tokens 24 --local off
--float-ops native --lane-proc`, `PGW_TRIP_FAST/NATIVE/DRIVER=1`, `PGW_DECODE_NATIVE=1`, the Python lane stub in a
child process; 2 interleaved pairs): the driver's `lane_us_per_trip` 104.6 / 99.0 us with the fixed 50 us spin vs
86.5 / 84.1 us with `--spin-auto-us 400` (channel wait per trip 40 -> 21-28 us, blocking waits 14 % -> 9-11 % of
trips); outputs identical; wall per token unchanged within noise (8 trips per token at 2 layers).

So on this box: zero-copy is a small sure gain, the adaptive spin cuts the median and the blocking waits, and
everything that adds a second spinning thread (server read spin, shm) loses in the mean, because the spinners
are preempted by the box's other load (p99 2-2.6 ms = scheduler time slices). shm is **not** a win over UDS here
(its floor is 2-3 us above UDS); it can only pay off where syscalls and wake-ups are expensive (gVisor) and cores
are dedicated. Decide on the H100 micro-benchmark below (same container as a co-located pair: gVisor, 4 cores).

**H100 micro-benchmark** (`benchmarks/lane_gpu_trip_bench.py`; about 2 min of H100 with 4 cores, about $0.10-0.15;
the image rebuilds the native extension from this checkout):

```bash
cd pgw
PGW_MODAL_APP_SUFFIX=-lt modal run deploy/modal/gpu_bench.py --target h100_c4 --timeout-min 5 \
  --out lane_gpu_trip.json \
  --cmd "python benchmarks/lane_gpu_trip_bench.py --preset 7b --iters 200 --repeat 2 --load none,python --json /tmp/lane_gpu.json"
```

It times real 1-row trips (native channel over the UDS lane, the native listener, captured and self-checked
graphs, bytes compared with the kernel path) for `uds` (today), `uds_tuned` (adaptive spin 400, server read
spin 2 ms, zero-copy) and `shm`, idle and with a Python-path load thread on the same GPU, the latter with the
shared and with the lane's own lock. Read `summary`: `overhead_p50` / `overhead_mean` per variant (target <= ~150
us all-in, today about 0.37 ms), `queue_us` (device-lock wait: should drop to about 0 with `own_lock` under
`python` load), `mismatches` (must be 0).

**7B e2e flags** (the CPU-only 7B command above plus, for the variant the micro-benchmark picks):

```bash
# UDS, tuned (no shm)
  --decode-native \
  --gateway-env PGW_LANE_SPIN_AUTO_US=400 \
  --provider-env PGW_SERVICE_LANE_READ_SPIN_US=2000 --provider-env PGW_SERVICE_LANE_ZERO_COPY=1 \
  --provider-env PGW_SERVICE_LANE_OWN_LOCK=1
# shm (only if lane_gpu_trip_bench shows it ahead of uds_tuned): add
  --gateway-env PGW_LANE_SHM=1 --provider-env PGW_SERVICE_LANE_SHM=1
```

`--provider-cpus 8` stays enough: a spinning connection thread per gateway lane connection (the decode driver
uses one; `max_channels` 4 at most) plus the factory's bulk channel. The e2e's lane split (`rows_1_service_ms` /
`rows_1_wire_ms`) and `/admin/status` show the effect; per-channel counters via `LaneComputeClient.channel_stats()`.

### Qwen2.5-32B co-located: CPU-only gateway, H100 provider (prepared 2026-11-22, not run)

`--model qwen2.5-32b-instruct` (pinned revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`) is ready for a short
plain-decoding e2e: every linear remote, H100 80GB provider, CPU-only gateway, the MLWE mask factory with a cached
hint. Nothing was run on Modal; the numbers below are estimates scaled from the 7B H100 run (plain 11 tok/s, 112
trips per token, 0.26 ms per trip, 45 GB cache restore in 12 s). Validated locally: tests and `--dry-run`.

**Sizes** (`pgw_modal_boot.model_sizes`; from the checkpoint's config: 64 layers, hidden 5120, intermediate 27648,
8 KV heads, so 4.65M output rows over the linears, 3.35x the 7B's 1.39M):

| artefact | 32B | 7B (for scale) |
|---|---|---|
| checkpoint download (bf16, 17 shards) | 65.5 GB | 15.2 GB |
| pack: int8 linears + float32 embedding and LM head | 31.2 + 6.2 = **37.5 GB** | 6.5 + 4.4 = 10.9 GB |
| encryption hint (24,576 bytes per output row) | **114.4 GB** | 34.2 GB |
| mask slot (`W r mod 2^32`, every linear remote) | **18.6 MB** | 5.56 MB |
| trips per token (4 per layer) | 256 | 112 |

(The 7B's pack is 10.9 GB with float32 plain tensors, which is what the 45 GB restore of its H100 run held with the
34.2 GB hint; the 32B hint is 114 GB, not about 78 GB.)

**What changed for it** (all opt-in by `--model`; library and demo defaults unchanged):

* **The provider's pack comes from a provider-side Volume** (`pgw-provider-packs`, `PGW_MODAL_PROVIDER_PACK_VOLUME`),
  not the image: baking 37.5 GB (a 65.5 GB download) into the image would rebuild it on every source change.
  `prepare_provider_pack` (a CPU container: 16 cores, 48 GiB; mounts only that Volume) downloads the pinned revision
  inside Modal and packs it with the image build's own command (`pgw.cli pack`, same shard size and float32 plain
  tensors, so the same fingerprint). In a run, the root parent copies it from the Volume (mounted 0700, root only)
  into `/srv/pgw-provider/pack`, which becomes the provider user's with the rest of its directory; the gateway never
  reads it.
* **The gateway side's pack and hint come from the gateway cache** (`--gateway-cache`, required for the 32B).
  `prepare_cache` (the T4 container, now 16 cores / 48 GiB / 150 min for the 32B; mounts only the gateway cache)
  downloads and packs the checkpoint itself, as the gateway's wizard would, commits the pack, then builds `H = W8 A`
  on the T4 with a fresh customer-side seed (`hint_gpu`, self-checked; the CPU path's bits). Neither container runs a
  provider process or mounts the other side's Volume; the hint is still never built by the provider (README "Gateway
  cache": selective-failure attack). Resumable: a cached pack or hint is kept.
* `launch.py --prepare-cache` runs both containers in parallel and then a **preflight** (free: reads the two
  manifests and lists the hint directory through the Modal API, no container) that the provider's pack, the cached
  pack and a hint carry the same fingerprint. Every 32B run repeats the preflight before deploying and refuses
  without it; in the container, `require_cached_setup` refuses to start the parties if the restore brought no pack or
  no hint (the wizard would otherwise download 65.5 GB and build the hint on the gateway CPU: about 65-70 min on
  16 vCPUs, beyond the TTL).
* **Sizing** (`COLOCATED_MEM_GIB`, same table in `launch.py` and `colocated.py`): provider 48 GiB (31.2 GB of int8
  linears staged), gateway 160 GiB: the factory streams the whole 114 GB hint once per hint pass and derives the
  Freivalds keys from the pack per segment, so the hint and pack (152 GB) stay in the page cache instead of being
  re-read from disk by every worker; plus the float32 embedding and head (6.2 GB), the workers' buffers and, during
  the e2e's plaintext reference, a resident int8 copy of the linears (31 GB). Requests, not limits; if Modal refuses
  208 GiB, `--gateway-mem-gib 112` still runs (hint pages re-read from disk; slower factory, enough for this run).
  Disk: about 220 GB in the run (two pack copies, the hint, a 1536-slot store), 217 GB in `prepare_cache`; the
  default ephemeral disk (>= 512 GiB) holds both. `--gpu` must be `H100` for the 32B (refused otherwise).
* **Factory keep-up**: `FACTORY_SLOT_DIVISOR` 16 (segments of 128 slots = 2.4 GB, refill below 384 to 512, yield
  156-375). The short run below overrides it (256-slot segments, target 1536, refill below 640).
* **Plaintext reference on 16 threads for the integer products** (`--gateway-env PGW_MODAL_REF_INT_THREADS=16`,
  default off): the e2e's reference runs the 32B on the root process's CPU with the gateway's pinned 1 float thread;
  single-threaded int8 GEMMs over 31 GB per token would take an estimated 10-15 min. Only `exact_int_matmul` (int8
  digit GEMMs, int32 accumulation: the same integers in any order) gets the threads; quantization, float ops and the
  head stay on the pinned count, so the reference's bits are unchanged (tested bit-identical on the tiny pack).
  Estimated 2-4 min (loading and transposing 31 GB of linears included).
* **`--mask-source {factory,gpu-service}`** (default `factory`): a hook for the customer-side GPU mask service built
  in a separate package. `gpu-service` is refused by the launcher (`GPU_MASK_SERVICE_READY = False`) and in the
  container (`pgw_modal_boot.mask_source_env` raises) until that package wires it; it then reaches the gateway as
  `PGW_MODAL_MASK_SOURCE`, the preflight and `require_cached_setup` stop asking for a hint (the pack is still
  needed), and the same short run below works with `--mask-source gpu-service` added (no hint build at all; where
  its GPU lives, e.g. a second GPU in the co-located container, is that package's choice and changes the price).
* `--dry-run` (any target): validates the arguments, prints the deploy environment, the run's gateway/provider
  environment, the sizes and the keep-up, and the cost per hour, then exits without any Modal call.

**Commands** (from `pgw/deploy/modal`; the same deploy flags for both steps):

```bash
FLAGS="--placement colocated --gpu H100 --model qwen2.5-32b-instruct --gateway-cache --gateway-cpus 32 --provider-cpus 8"
# 1) once: provider pack (CPU container) + gateway cache pack and hint (T4 container), in parallel, then the preflight
python launch.py $FLAGS --prepare-cache
# 2) the short e2e: plain decoding, exactness, 60 s back to back (redeploys the small image in seconds)
python launch.py $FLAGS --e2e --ttl-min 30 \
  --local-linears off --float-ops native --trip-native --prime-check --sustain-s 60 \
  --gateway-env PGW_MODAL_SUSTAIN_MIN_STOCK=640 \
  --gateway-env PGW_FACTORY_TARGET_SLOTS=1536 --gateway-env PGW_FACTORY_LOW_WATER_SLOTS=640 \
  --gateway-env PGW_FACTORY_SEGMENT_SLOTS=256 --gateway-env PGW_FACTORY_WORKERS=4 \
  --gateway-env PGW_FACTORY_CPU_THREADS=4 --gateway-env PGW_FACTORY_NICE=10 \
  --gateway-env PGW_POOL_WORKERS=8 --gateway-env PGW_THREAD_NAMES=1 --gateway-env PGW_MODAL_REF_INT_THREADS=16
```

Append `--dry-run` to either to see the plan for free. Mask budget of step 2: the wizard waits for one 256-slot
segment; the five 32-token chats before the reference take about 5 x 73 = 365 slots plus burned reservations; the
60 s window, at an estimated 3.5-5 tok/s with 64-token chats (prompt 41), takes 3-5 chats = 315-525 slots. So the
factory fills to 1536 (28.6 GB) during the chats and the reference, does not refill above 640, and the window starts
at once with >= 640 in stock (`PGW_MODAL_SUSTAIN_MIN_STOCK`; it waits at most 240 s otherwise). The window then
measures decoding without the factory competing for cores (the keep-up at 32B is not what this run measures).

**Cost and time (Modal list prices, estimates).**

| step | container | price | expected time | expected cost | cap |
|---|---|---|---|---|---|
| 1 `prepare_cache` | T4 + 16 cores + 48 GiB | $1.73/h | 25-45 min: download 65.5 GB 2-11, pack 5-10, copy 37.5 GB into the Volume 1-3, hint on the T4 about 12 (213 s x 3.35), copy 114 GB 2-8 | $0.7-1.3 | 150 min, $4.3 |
| 1 `prepare_provider_pack` (parallel) | 16 cores + 48 GiB | $1.14/h | 10-25 min | $0.2-0.5 | 150 min, $2.9 |
| 2 short e2e | H100 + 40 cores + 208 GiB | $7.51/h | 12-20 min: start 2-4, Volume copies 1-4 (37.5 + 152 GB at the 7B's 0.9-3.75 GB/s), service start with 31 GB to the H100 1-2, wizard with a full pack verification 1-2, first segment about 1, chats 1, reference 2-4, window 1, shutdown 1 | **$1.5-2.5** | TTL 30 min, $3.75 |

Total about **$2.5-4.3**, at most $11 if every container ran to its cap. Storage: the gateway cache grows by 152 GB
and the provider Volume holds 37.5 GB; check Modal's current Volume storage price, and when the decision is made
remove them (`modal volume rm pgw-gateway-cache qwen2.5-32b-instruct -r`, `modal volume delete pgw-provider-packs`).

**Expected outcome (estimate, not measured).** Short context (about 100 tokens), so attention is small. Per token:
256 trips x 0.3-0.35 ms (the H100 GEMVs are 1.4-2x the 7B's bytes) = 77-90 ms, gateway CPU about 0.4-0.45 ms per trip
(the 7B's per-trip cost scaled by the 1.45x vector sizes) = 100-115 ms, the CPU LM head over 1.56 GB of bfloat16 25-40
ms: 200-245 ms, i.e. **plain about 3.5-5 tok/s** (the 7B's 11 tok/s scaled by 2.3-3x). Speculation, the masked head and
the gateway attention work in `research/plans/scaling-32b-70b.md` are what would take it toward 25-30.

#### Qwen2.5-32B with the full fast stack (2026-11-23, dry-run tested, not run)

The 7B 28 tok/s stack (shared-memory lane, native decode driver, masked head, p31, native float ops) at 32B, plus the
32B-width gateway knobs measured locally (`STATUS.md`, 2026-11-23 "32B widths"): `PGW_LANE_WAIT_PREFETCH=1` (the trip's
`W r` and check vectors prefetched while the lane waits), the trip split over 12 warm, pinned pool workers
(`PGW_TRIP_SPLIT`, `PGW_POOL_WORKERS=12`, `PGW_POOL_ACTIVE_SPIN_US=20000`: workers poll only while a decode call is
active; `PGW_POOL_CPUS=4-15`: their own cores, so they do not share cores with the 4 factory workers and the
service). Every knob is timing only (same bits; `tests/native/test_scale32_knobs.py`); the split was not
shown to pay on the 4-vCPU dev box (oversubscribed there), so for an A/B drop `PGW_TRIP_SPLIT` / `PGW_FUSED2_SPLIT`
and `PGW_POOL_ACTIVE_SPIN_US` from the command and compare `decode_native` per-trip times. Expected (model,
`python benchmarks/scaling_model.py --colocated`): about 12-13 tok/s on one H100 (lane 23 ms + device 25 ms + gateway
18-27 ms + 9 ms serving per token): the question this run answers is whether the per-trip gateway work at 32B widths
is the measured-and-calibrated 0.07-0.11 ms and the lane stays at 0.09 ms per trip, not 25 tok/s.

Standard path (needs the 2026-11-22 `--prepare-cache` of this target; from `pgw/deploy/modal`):

```bash
python launch.py --placement colocated --gpu H100 --model qwen2.5-32b-instruct --gateway-cache \
  --gateway-cpus 32 --provider-cpus 12 --e2e --ttl-min 20 --sustain-s 60 \
  --local-linears off --float-ops native --trip-native --decode-native --masked-head --prime-check \
  --gateway-env PGW_LANE_SHM=1 --gateway-env PGW_LANE_SPIN_AUTO_US=400 --gateway-env PGW_TRIP_DECRYPT_AHEAD=1 \
  --gateway-env PGW_LANE_WAIT_PREFETCH=1 --gateway-env PGW_TRIP_SPLIT=12 --gateway-env PGW_TRIP_SPLIT_MIN_ELEMS=4096 \
  --gateway-env PGW_POOL_WORKERS=12 --gateway-env PGW_POOL_ACTIVE_SPIN_US=20000 --gateway-env PGW_POOL_CPUS=4-15 \
  --provider-env PGW_SERVICE_LANE_SHM=1 --provider-env PGW_SERVICE_LANE_ZERO_COPY=1 \
  --provider-env PGW_SERVICE_LANE_OWN_LOCK=1 \
  --gateway-env PGW_MODAL_SUSTAIN_MIN_STOCK=640 --gateway-env PGW_FACTORY_TARGET_SLOTS=1536 \
  --gateway-env PGW_FACTORY_LOW_WATER_SLOTS=640 --gateway-env PGW_FACTORY_SEGMENT_SLOTS=256 \
  --gateway-env PGW_FACTORY_WORKERS=4 --gateway-env PGW_FACTORY_CPU_THREADS=4 --gateway-env PGW_FACTORY_NICE=10 \
  --gateway-env PGW_THREAD_NAMES=1 --gateway-env PGW_MODAL_REF_INT_THREADS=16 --gateway-env PGW_MODAL_CACHE_SAVE=0
```

fused2 (two trips per layer, the `planes` pair op): first add the public fused2 tables to the gateway cache once
(`--prepare-cache --schedule fused2`: `prepare_cache` resumes, keeps the cached pack and hint, and derives the tables on
its 16 cores, about 10-15 min, +31 GB in the cache), then:

```bash
python launch.py --placement colocated --gpu H100 --model qwen2.5-32b-instruct --gateway-cache \
  --gateway-cpus 32 --provider-cpus 12 --schedule fused2 --prepare-cache            # once: + fused2 tables
python launch.py --placement colocated --gpu H100 --model qwen2.5-32b-instruct --gateway-cache \
  --gateway-cpus 32 --provider-cpus 12 --provider-mem-gib 96 --schedule fused2 --fused2-factory plain \
  --e2e --ttl-min 30 --sustain-s 60 \
  --gateway-env PGW_LANE_SHM=1 --gateway-env PGW_LANE_SPIN_AUTO_US=400 --gateway-env PGW_LANE_WAIT_PREFETCH=1 \
  --gateway-env PGW_FUSED2_SPLIT=12 --gateway-env PGW_POOL_WORKERS=12 --gateway-env PGW_POOL_ACTIVE_SPIN_US=20000 \
  --gateway-env PGW_POOL_CPUS=4-15 \
  --provider-env PGW_SERVICE_LANE_SHM=1 --provider-env PGW_SERVICE_LANE_ZERO_COPY=1 \
  --provider-env PGW_SERVICE_LANE_OWN_LOCK=1 --provider-env PGW_SERVICE_FUSED2_VARIANT=planes \
  --gateway-env PGW_MODAL_SUSTAIN_MIN_STOCK=256 --gateway-env PGW_FACTORY_TARGET_SLOTS=768 \
  --gateway-env PGW_FACTORY_LOW_WATER_SLOTS=256 --gateway-env PGW_FACTORY_SEGMENT_SLOTS=64 \
  --gateway-env PGW_FACTORY_WORKERS=2 --gateway-env PGW_FACTORY_CPU_THREADS=8 --gateway-env PGW_FACTORY_NICE=10 \
  --gateway-env PGW_THREAD_NAMES=1 --gateway-env PGW_MODAL_REF_INT_THREADS=16 --gateway-env PGW_MODAL_CACHE_SAVE=0
```

What the 32B combinations needed (checked here, `--dry-run` of both commands; nothing ran on Modal):

* **fused2 tables** (new): the public fused2 tables' derivation does one dense SVD per `o` / `down` for `sigma` (about
  27 s each at 32B widths on 4 vCPUs single-threaded; the gateway's `OMP_NUM_THREADS=1`): about an hour per party,
  and the gateway, each factory worker, the provider and the e2e's reference each derived them. Now: the provider
  and the factory derive without `sigma` (`Fused2Weights(..., sigma=False)`: their pair op, fingerprint and masks
  use `w8` / `hmul` only), and `--prepare-cache --schedule fused2` derives the full tables once into the cache
  (`<model>/fused2/fused2-m8.pt`), restored next to the pack; the gateway (`fused2_weights_path`, base fingerprint
  checked, now an allowed wizard preset), the factory (`PGW_FACTORY_FUSED2_WEIGHTS_PATH`) and the plaintext reference
  load that one file, so `sigma` is the same bits everywhere. Without the cached file the run still works but
  derives on the gateway at start (beyond a 30-min TTL at 32B).
* **fused2 has no masked head yet** (refused with `--masked-head`): the gateway streams the 1.56 GB bfloat16 head per
  token (about 50-80 ms): the fused2 32B run measures exactness, pair-trip timings and the planes op at 32B, not
  speed (expected 5-7 tok/s). Combining fused2 with the masked head (one more K32 trip and reservation) is the
  next code step for fused2 at 32B.
* **provider memory**: with `--schedule fused2` the service holds the K32 kernels (31.2 GB) and the fused2 pair-op
  weights (31.2 GB) on the H100: about 63 GB of 80 (fits; the planes graphs and buffers come on top). Host RAM for
  the provider's own derivation (31 GB of `w8` plus float64 temporaries): `--provider-mem-gib 96`.
* **masked head at 32B**: the provider derives the 0.78 GB int8 head; the gateway factory extends the cached hint
  with the head's `W8 A` at start (3.2e12 int8 MACs on CPU, estimated 2-4 min; not cached by `--prepare-cache`).
  `PGW_MODAL_CACHE_SAVE=0`: `save_cache` would copy the extension's hard-linked base files (114 GB) into the cache
  at shutdown; the pack and base hint are already cached.
* **pool and factory cores**: 32 gateway cores = the driver thread + 12 pinned split workers (CPUs 4-15) + the
  factory (4 x 4 threads, nice 10; fused2: 2 x 8) + attention/float threads; the warm workers poll only while a
  decode call runs (`PGW_POOL_ACTIVE_SPIN_US`), so the factory keeps its cores between passes.
* **fused2 masks**: `--fused2-factory plain` (customer-side plain CPU factory; no fused2 hint, which at 32B would be a
  K40 digit hint of several hundred GB built on first start); at 32B about 4-5 slots/s on 16 cores (7B: 5.3 slots/s
  on 4 vCPUs, x4.8 the products per slot), so the window starts from stock (256-768 slots of about 33 MB).

**Cost (Modal list prices; `launch.py` prints the same).** Container: H100 + 12 + 32 cores + 208 GiB = $7.70/h
(fused2 with `--provider-mem-gib 96`: about $8.1/h).

| command | expected time | expected cost | cap (TTL) |
|---|---|---|---|
| standard, full stack | 14-20 min: start + Volume copies 3-8, service start 1-2, wizard + pack verification 1-2, head hint extension 2-4, first segment 1, chats 1-2, reference 2-4, window 1, shutdown 1 | **$1.8-2.6** | 20 min, $2.57 |
| `--prepare-cache --schedule fused2` (once, resumes) | 10-15 min on the T4 container (tables only) | $0.3-0.45 | 150 min |
| fused2 | 20-28 min: as above + tables load 1, provider fused2 derivation without SVDs 2-4, factory stock 3-4, slower chats (float head) | **$2.7-3.8** | 30 min, $4.04 |

### Masked LM head (`--masked-head`, opt-in; 2026-11-22, package W3)

`--masked-head` (co-located only) takes the LM head off the gateway's CPU: the head becomes one more masked linear
`"<L>.lm_head"` per pass (`pgw.engine.masked_head`), computed by the provider like every linear (range check and
Freivalds on every row; the provider sees one more masked call of public shape per pass). What it sets:

| where | setting | effect |
|---|---|---|
| provider | `PGW_SERVICE_MASKED_HEAD=also` | each pack is also served under `SHA-256("pgw/v1/masked-lm-head\0" \|\| fp)`, the layers' kernels shared, plus the int8 head the provider derives from its own hash-checked pack |
| gateway factory | `PGW_FACTORY_MASKED_HEAD=true` | segments for the derived fingerprint, with the head's masks and check vectors (ring or `p31`); the encrypted factory's hint is the base hint (hard-linked) plus the head's `W8 A` with the same seed |
| gateway model | `head_mode = "provider_int8"` (wizard preset) | derives the fingerprint from its verified pack (a service that does not serve it is refused at connect), never loads the float head, sends the head as one masked call per pass (R rows with strict speculation) |
| e2e reference | `plaintext_reference(..., masked_head=True)` | the same int8 head on the plaintext backend: a new pinned reference, compared bit for bit |

Memory: the provider holds `vocab x hidden` more int8 (0.55 GB at 7B, 0.14 GB at 0.5B); the gateway drops the float
head (at 7B 1.09 GB bfloat16 in RAM and per pass; at 0.5B the head is the tied embedding table, which stays). Masks:
+`4 x vocab` bytes per slot (+0.61 MB: +11% at 7B, +50% at 0.5B); hint +3.7 GB at 7B. No sizing flag changes are
needed (`--gateway-mem-gib` can drop by about 2 GB at 7B).

Quality of the int8 head against the float head (`benchmarks/masked_head_quality.py`, Qwen2.5-0.5B-Instruct, K32A12C27,
exact int8 layers on `PlainBackend`, 12 chat prompts teacher-forced, 504 positions): top-1 agreement 99.60% (2 of 504
positions differ), |delta logprob| of the float head's top-1 token mean 0.012, max 0.060, max over the whole vocabulary
0.37. The 0.5B's head is the tied embedding table; the 7B's untied head is not measured here.

Rings: the head's `in_features` is the hidden size, so its bounds are the qkv linear's: K32A12C27 ring check 21 bits
with 7 vectors, `p31` 31 bits with one (`2B = 2^28 < p`); K32A8 ring check 28 bits (`p31` 31); K40A12 / K48A20 ring
check only (`p31` needs K32). All exact in tests; the T4 runs used K32A12C27 + `p31`.

**Validation (T4, Qwen2.5-0.5B, `PGW_MODAL_APP_SUFFIX=-mh`, same flags both runs: `--gpu T4 --gateway-cpus 8
--local-linears off --trip-native --prime-check --spec-rows 2 --sustain-s 25 --ab-plain 25 --gateway-cache`).**
Masked head: `ok`, logprobs bit-identical to the int8-head reference (max |delta| 0.0), text equal, stream identical,
every speculative and plain sustain chat identical, every segment `p31`, no integrity failure, 285 s spawn to stop.
Without it (A/B): 196 s.

| plain phase, 25 s | decode tok/s | trips/token | gateway CPU per trip | gateway CPU per token | 1-row trip |
|---|---|---|---|---|---|
| float head on the gateway | 18.0 | 97.9 | 1.333 ms | 130 ms | 0.208 ms |
| `--masked-head` | 15.2 | 98.9 | 0.826 ms | 82 ms | 0.256 ms |

Gateway process CPU per token falls by 49 ms (38%): the native float head (up to 16 threads over a 0.54 GB float32
tied table) is gone. Decode is slower at 0.5B on the T4: the head trip returns 608 KB per row and costs more on the
provider than the 0.5B's whole float head costs the gateway, and the factory makes 50% more mask bytes. At 7B the
balance is the other way (the head is 11% of the mask bytes and 1.09 GB of bfloat16 per pass on a CPU-only gateway,
about 60 ms per token measured on the H100 run); the H100 rerun with `--masked-head` is the measurement that counts.

### fused2, 2 provider trips per layer: one exact 7B e2e (`fused2_e2e.py`, opt-in; prepared 2026-11-23, not run)

A separate small app (`pgw-fused2-e2e`), not `launch.py`: one container, the provider's fused2 pair op on the GPU
(`python -m pgw.engine.fused2_remote serve`: public `w8` + residual multipliers only, masked ring elements only, a
0700-directory Unix socket) and the gateway on the CPU (`benchmarks/fused2_e2e.py`: builds the migrated fused2
reference from the public checkpoint at start-up, makes its own masks with an in-process plain factory, masks
every trip, range-checks and Freivalds-checks both stages over F_(2^61-1), greedy-decodes, then replays the first
calls on the plain reference and requires bit-identical logits). Trips per token: `2L + 1` = 57 at 7B (today 113).

```bash
PGW_F2_MODEL=qwen2.5-7b-instruct PGW_F2_GPU=H100 modal run pgw/deploy/modal/fused2_e2e.py \
    --tokens 32 --ring K56A12C53 --migrate-bits 8 --replay-calls 3
```

Defaults: 24 vCPU, 96 GiB, H100, 40-minute timeout. The image build (CPU, once) downloads the pinned 7B
checkpoint (15 GB), packs it and builds the native kernels. Start-up in the container: folding + spectral norms +
check vectors (CPU, about 3-6 min), masks for prompt + 32 slots (about 1-2 min), decode, replay of prefill + 2
decode steps (CPU, a few minutes): about 10-15 minutes, $1.2-1.8 on an H100. Writes `fused2_e2e_result.json`
locally (`exact`, `trips_per_token`, `bytes_per_token`, `trip_ms_p50/p90`, `provider_compute_ms_p50`,
`trip_ms_sum_per_token`, `decode_tok_s`, `act_bits_hist`, `slots_per_token`, `factory_macs_per_slot`).

What this run does **not** use (fused2 has no path there yet): the production gateway runtime and console, the mask
store and the encrypted (HE/PKS) factory (K56 needs an HE modulus above 2^64; here the gateway makes its own masks
from public weights, so the gateway holds the int8 weights: not weight-free), the binary trip lane and its UDS/shm
transports (`PGW_LANE_SHM`, spin, zero-copy: the harness has its own framed Unix socket), `--trip-native` /
`--decode-native` (per-linear native kernels, no pair op: the harness's per-trip gateway work is Python/numpy),
`--prime-check` (p31 cannot hold fused2's bounds; fused2 checks over p61 in numpy limb arithmetic) and
`--masked-head` (fused2 keeps the LM head on the gateway; folding it into the last trip is not prototyped). So
read `trips_per_token` and `trip_ms_sum_per_token` against today's 113 trips, not `decode_tok_s` against the
28 tok/s of the native path: on this harness the gateway's Python per-trip work (about 2 ms per trip at 0.5B on a
loaded CPU core, more at 7B) dominates.

### fused2 in the co-located demo (`launch.py --schedule fused2`, opt-in; 2026-11-23, dry-run tested, not run)

The production stack with the fused2 schedule (`research/plans/fused2-two-trips-per-layer.md` section 12): the
provider serves the fused2 lane entry derived from its own pack (`PGW_SERVICE_FUSED2_FROM_PACK=8`,
`PGW_SERVICE_FUSED2_RING=K56A12C53`; the Triton pair op, graphed on CUDA) and, for the encrypted factory, maskgen
for its fused2 linears (`PGW_SERVICE_FUSED2_MASKGEN=1`); the wizard presets `schedule = "fused2"`, `ring =
K56A12C53`, `local_linears = "off"` and the lane (`uds` -> `auto`, or `tls`); the gateway's factory makes fused2
pair-mask segments (`PGW_FACTORY_SCHEDULE=fused2`: by default the encrypted MLWE factory as 4 x 14-bit digits at
K40, its own fused2 hint built on first start and kept by `--gateway-cache`; `--fused2-factory plain`: the
customer-side plain CPU factory, no provider work); the gateway runs the native fused2 driver; the e2e's
plaintext reference is the fused2 reference on the same public weights. Exact 7B command (H100 provider, 32-core
CPU-only gateway, shared-memory lane; about the 28 tok/s run's cost, $5-6/h, plus the one-time fused2 hint build
in the first run):

```bash
python pgw/deploy/modal/launch.py --placement colocated --gpu H100 --model qwen2.5-7b-instruct --gateway-cache \
    --gateway-cpus 32 --provider-cpus 12 --schedule fused2 --e2e --sustain-s 120 \
    --gateway-env PGW_LANE_SHM=1 --gateway-env PGW_LANE_SPIN_AUTO_US=400 \
    --provider-env PGW_SERVICE_LANE_SHM=1 --provider-env PGW_SERVICE_LANE_ZERO_COPY=1 \
    --provider-env PGW_SERVICE_LANE_OWN_LOCK=1 --provider-env PGW_SERVICE_FUSED2_VARIANT=planes
```

Add `--dry-run` to print the plan without any Modal call. Refused with `--schedule fused2`: `--lane stream`,
local linears, `--spec-rows` / drafts, `--masked-head` (the head stays on the gateway), `--prime-check` (fused2
checks over p61), `--float-ops torch`.

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
* **No volumes** by default. Config, secrets, packs and the mask store live on the customer container's local
  disk (`--gateway-cache` adds one Volume for public artefacts only: the pack and the hint, never masks or secrets;
  see "CPU-only 7B gateway, strict speculation, H100 provider"). The mask store must never be on network or snapshotted storage (a restored copy could hand out
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
