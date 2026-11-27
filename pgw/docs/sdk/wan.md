# WAN: a consumer device to our hosted GPUs

The client library (`from pgw import OpenAI`) holds every plaintext; our GPUs compute masked linear layers. Each
decoded token is a chain of sequential round trips (113 at 7B today, 57 with fused2, 257 / 129 at 32B), each
carrying that layer's masked activations both ways. Over a wide-area network three things set the speed, in this
order: **bandwidth** (the bytes every trip moves), **RTT x trips**, and **per-trip overhead beyond one RTT**. This
page measures all three on an emulated WAN, says what was fixed, what decides who can reach a given speed, and how
the client picks the nearest GPU.

## Summary

* **Per-trip overhead beyond one RTT is small and now has no idle penalty on the provider side.** The lane stack
  (TLS 1.3, frames, both sides' copies and wake-ups) adds **0.04-0.10 ms** per trip over bare TCP of the same bytes
  for decode-sized trips and 0.14-0.29 ms for the 150-220 KB `gate_up` replies, at every RTT from 1 to 50 ms
  (measured, below). The one effect that cost whole RTTs was TCP's restart after idle: the first trips after a
  pause of more than ~200 ms (between two chats) took **2-4 extra RTTs** (81 ms instead of 21 ms at RTT 20 for one
  7B `gate_up` trip). `net.ipv4.tcp_slow_start_after_idle = 0` on the **provider** removes it completely (20.8 ms;
  measured); the provider now logs a warning when its namespace has `1`. BBR on the provider also removes it but
  costs +1.4-2.4 ms on every trip (pacing), so it is not recommended.
* **Bandwidth binds first for a consumer.** A 7B token moves 3.3 MB up and 6.2 MB down (9.5 MB; 32B: 30 MB) on
  the trips alone, plus ~7 MB (32B: ~22 MB) of pks-v3 mask replies unless the stock was filled while idle. On a
  1 Gbit/s link the trips' bytes alone take 76 ms per 7B token, as much as 38 trips' worth of RTT at 2 ms.
* **20 tok/s at 7B / 32B from a consumer device is not reachable with today's wire format.** At 4 tokens per pass
  with fused2 (a 200 ms pass budget), 7B needs RTT <= 3.2 ms *and* ~2.5 Gbit/s of combined bandwidth (both at their
  limit cannot hold together: in practice 10G fibre at RTT <= ~1.5 ms); 32B needs RTT <= 1.2 ms *and* ~10 Gbit/s.
  Best realistic consumer cases (`benchmarks/wan_model.py`, measured overheads): 7B fused2 at 4 tokens per pass on
  10G fibre in the same metro ~27 tok/s, on 1G fibre ~7 tok/s, on cable / 5G < 1 tok/s; 32B at most ~10 tok/s on
  10G fibre. A metro-edge deployment (GPUs within ~50 km of the users) is necessary for any interactive speed at
  7B, but not sufficient for 20 tok/s: the bytes per token must shrink first (see [What would change
  this](#what-would-change-this)).
* **QUIC: not adopted.** Trips are strictly sequential single request/reply exchanges, so QUIC's stream
  independence has nothing to unblock; its loss recovery for a lost tail packet (PTO) costs about what TCP's tail
  loss probe does; 0-RTT would only shorten connection setup (off the critical path) and replays early data. The
  harness shows TCP's per-trip cost under 0.1-1 % loss (below); no measured gain justifies a second transport.
* **Region routing designed, with the provider side** (`pgw.sdk.regions`, `endpoints` in the enrollment
  response): payload-free TCP-handshake probes, nearest healthy endpoint with hysteresis, failover order; the mask
  stock moves with the client when the endpoint serves the same pack fingerprint and ring.
* **Keep-alive on the lane** (fixed): idle lane connections had no keep-alive, so a NAT or firewall that forgets
  idle flows (carrier-grade NAT: often 30 s to a few minutes) turned the next trip into a reset or a silent stall
  until the trip timeout. Lane sockets (native and Python client, the provider's listener) now use keep-alive (idle
  15 s, 5 s x 3 probes) and `TCP_USER_TIMEOUT`, so idle lanes survive NATs and a dead path fails over in seconds
  (`PGW_LANE_KEEPALIVE_S`, 0 = off). Probes are empty ACKs: no payload, nothing new for the provider.

## The harness

`benchmarks/wan_bench.py` (needs root, `ip` and `/dev/net/tun`; this container has no `tc netem`):

```text
netns pgwwc (client) -- tun 10.77.0.1 <-> pgw-wanemu <-> tun 10.77.0.2 -- netns pgwws (provider)
```

`rust/tools/wanemu` (`pgw-wanemu`) forwards every IP packet between two TUN devices with, per direction, a
bottleneck rate with a drop-tail queue, a propagation delay, jitter (FIFO, no reordering) and Bernoulli or
Gilbert-Elliott loss; MTU 1500. Both kernels run real TCP end to end (slow start, delayed ACKs, cwnd validation,
RACK/TLP loss recovery), unlike a TCP-terminating proxy (`delay_proxy.py`), whose loopback legs hide all of it.
Each namespace sets its own congestion control (default `cubic`, what consumer OSes and most clouds run; this box
defaults to BBR, which must not leak into a measurement) and `tcp_slow_start_after_idle`, per side
(`--server-cc`, `--server-ssai`: the provider's kernel is ours to configure, the client's is not).

```bash
python benchmarks/wan_bench.py lane --rtts 0,1,2,5,20,50 --rows 1,4           # transport, no model
python benchmarks/wan_bench.py lane --rtts 5,20,50 --server-ssai 0            # a provider-side fix
python benchmarks/wan_bench.py lane --rtts 5,20 --loss 0,0.001,0.01           # loss
python benchmarks/wan_bench.py sdk  --rtts 0,1,2,5,20,50 --tokens 16          # pgw.OpenAI end to end, 0.5B
python benchmarks/wan_bench.py sdk  --rtts 20 --rate-mbit 100 --up-mbit 20    # a consumer access link
python benchmarks/wan_report.py benchmarks/results/wan/*.json                 # tables
python benchmarks/wan_model.py                                                # "who can get 20 tok/s"
```

* `lane`: the real native TLS listener (`pgw_fast.LaneService`) with its benchmark executor (answers at once: no
  compute) serving every linear's request / reply sizes at 0.5B / 7B / 32B and the masked head, driven by the
  gateway's native `TripChannel`; interleaved with a bare TCP request/reply of the same sizes on its own
  connection (`raw`). `lane - raw` is what the lane stack adds; `raw - RTT` is the kernel + emulator; one decode
  pass (every trip of a token in order) gives the per-token view; the idle test times the first 7B `gate_up` trip
  after a gap on a warm connection.
* `sdk`: the SDK tests' provider (`tests/sdk/provider_main.py`: TLS, TCP+TLS lane, masked head, real accounts,
  enrollment; CPU kernels) in the provider namespace, `pgw.OpenAI` in the client namespace running
  `sdk_trip_profile.py`'s measurement; `excess_ms_per_trip = ms/trip(RTT) - ms/trip(0) - RTT` cancels the CPU
  provider's compute. Every RTT must give the same tokens and log-probabilities (checked).

Calibration: the emulator delivers a packet 40-80 us after its due time on average on this shared 4-vCPU box
(`stats`: `late_us_mean`), so `raw - RTT` (0.1-0.6 ms) is mostly the emulator and this box's TUN path, not what a
real network adds; `lane - raw` does not depend on it.

## Results (this box, 2026-11-27)

### Per trip vs RTT (`benchmarks/results/wan/lane_unlimited.json`, 1 row, p50 ms)

| RTT | 7B qkv | 7B gate_up (148 KB down) | 7B down (74 KB up) | 7B head (594 KB down) | 32B gate_up (216 KB) | lane - raw, decode trips | lane - raw, 150-220 KB replies |
|---|---|---|---|---|---|---|---|
| 0 | 0.164 | 0.585 | 0.350 | 2.42 | 1.04 | 0.04-0.13 | 0.08 |
| 1 | 1.211 | 1.500 | 1.343 | 3.16 | 2.00 | 0.04-0.13 | 0.14-0.26 |
| 2 | 2.282 | 2.529 | 2.376 | 3.21 | 2.69 | 0.04-0.16 | 0.09-0.19 |
| 5 | 5.379 | 5.860 | 5.422 | 12.1 | 5.78 | 0.05-0.16 | 0.19-0.24 |
| 20 | 20.38 | 20.70 | 20.48 | 22.1 | 20.93 | 0.07-0.10 | 0.14-0.27 |
| 50 | 50.42 | 50.88 | 50.59 | 52.4 | 51.09 | 0.06-0.14 | 0.21-0.29 |

One decode pass (all trips of a token, no compute; ms per trip and excess over RTT):

| RTT | 0.5B (97 trips) | 7B (113) | 32B (257) |
|---|---|---|---|
| 0 | 0.29 | 0.60 | 0.88 |
| 1 | 1.21 (+0.21) | 1.29 (+0.29) | 1.45 (+0.45) |
| 2 | 2.36 (+0.36) | 2.37 (+0.37) | 2.55 (+0.55) |
| 5 | 5.27 (+0.27) | 5.37 (+0.37) | 5.51 (+0.51) |
| 20 | 20.5 (+0.51) | 20.6 (+0.64) | 20.9 (+0.90) |
| 50 | 51.4 (+1.38) | 51.2 (+1.24) | 50.8 (+0.76) |

Where the excess goes, per trip at RTT 1-5 ms: ~0.05-0.1 ms the lane stack (TLS records, frames, both sides'
copies; up to 0.3 ms for 150-220 KB replies, 0.2-0.7 ms for the 594 KB head reply: AES-GCM and copies at ~1 GB/s
on this box), ~0.15-0.3 ms the emulator and this box's TUN path (not present on a real link), and the rest the
masked head's 594 KB reply: it needs a congestion window of ~410 segments once per pass while the other trips keep
it at ~100-150, so at RTT >= 20 ms it sometimes takes two flights (raw head p50 41 ms at RTT 20). On a real link the
serialisation of the bytes adds on top (1 Gbit/s: 1.2 ms for a 148 KB reply, 4.9 ms for the head's 594 KB).

Toward the 0.1-0.2 ms per-trip target: the lane's own part already meets it for every trip except the large
replies; what remains is the bytes (the next section), not framing or syscalls.

### TCP restart after idle (first 7B gate_up trip after a gap, p50 ms)

| RTT | gap | default (both `ssai=1`) | provider `ssai=0` | provider BBR | both `ssai=0` |
|---|---|---|---|---|---|
| 5 | 300 ms | 5.83 | 5.75 | 6.05 | 5.72 |
| 5 | 1 s | 16.3 | **5.85** | 6.08 | 5.72 |
| 5 | 3 s | 20.9 | **5.79** | 6.27 | 5.87 |
| 20 | 300 ms | 20.7 | 20.8 | 22.4 | 20.7 |
| 20 | 1 s | 61.0 | **20.8** | 22.3 | 20.9 |
| 20 | 3 s | 81.1 | **20.9** | 25.2 | 20.8 |
| 50 | 1 s | 151 | **50.9** | 56.0 | 50.9 |
| 50 | 3 s | 201 | **51.1** | 56.0 | 50.8 |

Per decode pass, BBR on the provider costs +1.4-2.4 ms per trip at RTT 20-50 (7B pass 2459 vs 2318 ms at RTT 20):
its pacing spreads every large reply. `ssai=0` changes nothing in steady state (same pass times within noise).

What this means: within one answer the trips never pause long enough for the restart (it needs a gap longer than
the RTO, >= 200 ms), but the **first pass of every new request after a think-time pause** (a chat turn) pays it,
on every lane connection; at RTT 20 the prompt pass's first large trips lose ~60 ms. The provider sends the large
replies, so the provider's setting is what matters: set `net.ipv4.tcp_slow_start_after_idle=0` for the provider's
network namespace (`docker run --sysctl net.ipv4.tcp_slow_start_after_idle=0`; Kubernetes `securityContext.sysctls`
once the kubelet allows that sysctl; `sysctl -w` on a host). `pgw service` logs a warning at start when the TCP lane
runs with `1` (`pgw.service.lane.wan_tcp_check`, read-only: the service never changes a kernel setting). The
client's own uploads (a 7B `down` input is 74 KB) still restart on a consumer OS; nothing in the application can
change that per socket.

### Loss (`lane_loss.json`)

LOSS_TABLE

### End to end, the client library (`sdk_*.json`, Qwen2.5-0.5B, CPU provider)

SDK_TABLE

## Per-trip WAN overhead: what was checked

| item | finding | action |
|---|---|---|
| Nagle | `TCP_NODELAY` on both ends of the lane (native client, provider listener) and the stream channel | none needed |
| delayed ACKs | requests and replies piggyback ACKs; no small-write-then-wait pattern (one `writev` per request and per reply) | none needed (measured `lane - raw` <= 0.1 ms) |
| TLS record sizing | rustls 16 KB records; the last record of a reply is decrypted when its last byte arrives: ~5 us | none |
| slow start after idle / cwnd validation | 2-4 extra RTTs on the first large trips after > RTO idle | provider: `tcp_slow_start_after_idle=0` (measured fix), startup warning; BBR rejected (measured) |
| cwnd for the 594 KB masked-head reply | sometimes 2 flights at RTT >= 20 ms | none in transport; see "What would change this" |
| initial window | 10 segments on every OS; a new connection's first large reply takes `log2(size / 14 KB)` RTTs | connections are opened at warm-up (the RTT probe), not on a request's critical path |
| idle connections behind NAT | no keep-alive on the lane: a forgotten NAT mapping meant a reset or a stall until the trip timeout | **fixed**: keep-alive 15 s / 5 s x 3 + `TCP_USER_TIMEOUT` on every lane socket |
| head-of-line blocking under loss | trips are sequential: one message in flight per connection, so there is no other stream to block | none (see QUIC) |
| connection setup | TCP 1 RTT + TLS 1.3 1 RTT + lane open 1 RTT, plus `lane/open` on the HTTP channel (1 RTT): measured 3-3.5 RTT (`lane_connect_ms` 75 ms at RTT 20) | at warm-up; tickets single-use, valid 30 s for opening, connections stay open |
| ticket refresh | a ticket only opens connections; open connections never need one. After a lane failure the client uses the stream channel for `retry_s` (30 s), then opens a new ticket | none |
| WebSocket fallback | one RTT per trip like the lane (same TCP behaviour, keep-alive pings every 20 s already); 0.1-2 ms more CPU per trip | none |

## QUIC / UDP lane: verdict

Not adopted, by the criteria of the task (a real gain in the harness, same authentication, TLS 1.3, framing and
privacy):

* **No head-of-line blocking to remove.** A decode pass is a chain: trip i+1 needs trip i's answer, and one lane
  connection carries one trip at a time. QUIC's independent streams only help when several messages are in flight;
  here there is one.
* **Loss recovery is not faster.** The costly loss is the last packet of a request or reply. Linux TCP recovers it
  with RACK-TLP (probe after ~2 SRTT); QUIC with its probe timeout (SRTT + 4 RTTVAR + the peer's max_ack_delay,
  25 ms by default), i.e. no better at the RTTs that matter. The harness's loss runs (above) show what TCP costs.
* **0-RTT only shortens setup**, which is already off the critical path (warm-up), and 0-RTT data is replayable:
  a trip must never be accepted twice.
* **Cost**: a second transport with its own TLS integration, connection migration, UDP blocked on some networks
  (corporate, some carriers), and a second audit surface.

The case to revisit: connection migration for a mobile client switching networks mid-answer (Wi-Fi to cellular).
Today such a switch resets the lane and the trip is retried on a new connection (the same masked bytes,
docs/sdk/reliability.md).

## Prompt pass over a WAN

Bytes per trip (K32: 4 bytes per element; a prompt trip carries every prompt row at once):

| model | linear | up per row | down per row | 500-row prompt trip |
|---|---|---|---|---|
| 0.5B | qkv / o / gate_up / down | 3.5 / 3.5 / 3.5 / 19 KB | 4.5 / 3.5 / 38 / 3.5 KB | gate_up: 1.75 MB up, 19 MB down |
| 7B | qkv / o / gate_up / down | 14 / 14 / 14 / 74 KB | 18 / 14 / 148 / 14 KB | gate_up: 7 MB up, 74 MB down |
| 32B | qkv / o / gate_up / down | 20 / 20 / 20 / 108 KB | 28 / 20 / 216 / 20 KB | gate_up: 10 MB up, 108 MB down |
| all | masked head (last row only) | 3.5-20 KB | 594 KB | once per prompt |

A whole prompt pass moves `rows x 9.5 MB` at 7B (500 tokens: 1.6 GB up, 3.1 GB down): 38 s at 1 Gbit/s, 6 min at
100 Mbit/s. The prompt pass is **bandwidth-bound** on every consumer link, and the conversation prefix cache
(`prefix_cache=True`: reused rows send nothing) is the lever that matters, not transport tuning.

Chunking / pipelining: a prompt pass costs `trips x RTT + bytes / bandwidth`. Splitting the rows into chunks
pipelined over two lane connections can overlap one chunk's transfer with the other's RTT, at best
`max(trips x RTT, bytes / bandwidth)` instead of their sum: at most 2x, and only where the two terms are equal
(7B at 1 Gbit/s and RTT 20 ms: ~30-token prompts). Longer prompts are bandwidth-bound (nothing to overlap),
shorter ones RTT-bound (more chunks add trips). Not worth a scheduler change today. TCP itself is not the limit: a
single connection reaches 1 Gbit/s at RTT 20 ms within ~8 RTTs of slow start (~150 ms of a 38 s pass) with the
default receive-buffer auto-tuning (Linux `tcp_rmem` max 6 MB covers 2.4 Gbit/s at 20 ms).

## Bandwidth: the binding constraint

| model | trips | up per token | down per token (incl. head) | pks-v3 mask reply per token | 20 tok/s needs (trips only) |
|---|---|---|---|---|---|
| 0.5B | 97 | 0.73 MB | 1.82 MB | ~2 MB | 0.4 Gbit/s |
| 7B | 113 (57 fused2) | 3.3 MB (fused2 K40: 4.2) | 6.2 MB (7.6) | ~7 MB | 1.5 Gbit/s (1.9) |
| 32B | 257 (129 fused2) | 11 MB (13.8) | 19.2 MB (23.9) | ~22 MB | 4.8 Gbit/s (6.0) |

Every byte is a uniform ring element (that is what hides the activations), so it cannot be compressed. With
speculation the bytes scale with the rows verified per pass, not with the tokens accepted.

## Location: who can get 20 tok/s

Consumer RTT to a GPU = access network + distance + the GPU site's peering:

* access network, idle, wired: fibre (FTTH) ~1-3 ms to the ISP's metro edge; cable (DOCSIS 3.1) ~5-15 ms
  (request-grant scheduling; low-latency DOCSIS ~1-5 ms where deployed); DSL ~10-25 ms; 5G ~10-30 ms (fixed
  wireless at the low end, mobile in motion at the high end and with 50+ ms tails); 4G ~30-60 ms. Home Wi-Fi adds
  1-5 ms median and tens of ms at p99. Under load (a concurrent upload, the mask factory's traffic) bufferbloat adds
  tens to hundreds of ms on links without AQM (FCC Measuring Broadband America: working latency well above idle).
* distance: ~1 ms of RTT per 100 km of fibre path (light in glass covers ~200 km per ms one way), routes 1.5-2x the
  straight line: a GPU 500 km away adds ~7-10 ms, a continent away 40-80 ms, an ocean 70-150 ms.
* GPU site: cloud regions sit in a few metros per continent (RunPod lists ~30 data centres worldwide:
  US-KS-2, US-TX-3, US-GA-1/2, US-DE-1, CA-MTL-3, EU-RO-1, EU-CZ-1, EU-SE-1, EU-NL-1, ... - take the current list
  from the catalog API, `jobs.py datacenters`); most users are 300-1500 km from the nearest one, i.e. 5-25 ms
  RTT on fibre. A metro edge (colocation in each large city, at the internet exchange where the access ISPs peer)
  brings a fibre user to 1-3 ms.

`python benchmarks/wan_model.py` (o = 0.15 ms per trip, H100 device time, mask stock prefilled; decode tok/s):

| access (RTT to the GPU) | 7B today, no spec | 7B fused2, draft head R=3 | 7B fused2, 4 tokens / pass | 32B fused2, 4 tokens / pass |
|---|---|---|---|---|
| co-located, same DC (0.2 ms, 25 Gbit) | 20.7 | 43.6 | 88.6 | 29.7 |
| fibre 10G, wired, same metro (1.5 ms) | 5.0 | 12.7 | 27.1 | 10.2 |
| fibre 1G, wired, same metro (2 ms) | 3.1 | 4.0 | 6.7 | 2.2 |
| fibre 1G over Wi-Fi 6, same metro (4 ms) | 1.8 | 2.8 | 4.9 | 1.6 |
| cable 1G/50, same metro (12 ms) | 0.5 | 0.6 | 0.9 | 0.3 |
| 5G 300/50 (25 ms) | 0.3 | 0.4 | 0.7 | 0.2 |
| fibre 1G, ~1000 km away (18 ms) | 0.5 | 1.2 | 2.6 | 1.0 |

With the mask factory's pks-v3 replies on the same link (`--stock live`) every row drops further (7B: +86 ms per
token at 1 Gbit/s). So:

* **20 tok/s at 7B**: only a client in the GPU's data centre, or a wired 10 Gbit/s fibre user within ~1-2 ms of a
  metro-edge GPU with 4 tokens per pass and fused2 (27 tok/s modelled). No mainstream consumer link (1 Gbit/s
  or less) gets there at any distance.
* **20 tok/s at 32B**: only co-located clients (in our own data centre, e.g. a customer's server next to the GPU).
* **A metro-edge deployment** is required for interactive 7B on fibre (5-27 tok/s at 1-2 ms vs 1-3 tok/s from a
  regional cloud 500-1000 km away) and pointless for cable / mobile users, whose access network alone exceeds the
  per-trip budget.

### What would change this

Not in this work's scope (the scheme and its checks are unchanged here), listed because they, not transport
tuning, decide the target: fewer bytes per token (rows that the provider does not need to return; a narrower ring
where the error budget allows; the masked head's 594 KB per pass replaced by a local head on clients with a GPU),
fewer trips per token (fused2, deeper fusion), and more tokens per pass (speculation; bytes grow with rows
verified, so acceptance per row matters more over a WAN than co-located).

## Region routing

**Provider side.** `ServiceSettings.endpoints` (`PGW_SERVICE_ENDPOINTS`, JSON) lists the endpoints an enrolled
client may use: `{region, url: "https://<host>:<port>", models: [pack fingerprints] (empty = all), priority}`. The SDK
enrollment response carries it as `endpoints` (absent = this service only); the setting is validated with the
client's own parser at start. Requirements on a multi-region deployment: every endpoint presents a certificate from
the **same provider CA** the SDK key pins; every endpoint accepts the installation's credentials: the client
certificate (issued at enrollment by the shared client-issuing CA, so verifiable anywhere without state) and the
service token (a shared, replicated token store or signed tokens), with usage reports merged centrally; each
endpoint serves its own TCP lane (tickets from its own `lane/open`).

**Client side** (`pgw.sdk.regions`; enrollment stores the validated list in `enrollment.json`):

* `parse_endpoints`: https only, host and port, no path / query / credentials, at most 32, the enrolled URL always
  included; a malformed list is refused whole (enrollment then keeps the enrolled URL only).
* `probe`: N TCP handshakes per endpoint (connect, close; **no byte of payload**: no TLS ClientHello, so no SNI,
  token, certificate, model name or prompt), interleaved across endpoints, timeout 1 s. Run at start and between
  requests (never during one), e.g. every 10 minutes while idle. An endpoint learns only that this address opened
  a connection - what it learns anyway when used.
* `choose`: the lowest median RTT among endpoints that answered every probe and serve the model's fingerprint;
  the current endpoint is kept unless another is faster by 20 % and 1 ms (no flapping). The chosen endpoint is then
  used exactly as the enrolled one today: pinned CA, client certificate, token, readiness check.
* `failover_order`: on a connection failure, the healthy endpoints by RTT, then the unanswered ones, never the
  failed one; a request already past its first token is not moved (docs/sdk/reliability.md: retries stop at the
  first token).

**Masks when the region changes.** A mask slot is `W8 r mod 2^K` for the pack's int8 weights and the client's
secret `r`; pks-v3 masks are produced homomorphically under the client's key and checked by the client. The stock
is bound to (pack fingerprint, ring, scheme), not to a host: an endpoint that serves the same fingerprint and ring
uses the same stock and the same verified hint (`stock_compatible`); single use is enforced by the client's own
ledger, wherever the trips go. An endpoint serving another fingerprint (another revision or quantisation) needs
its own stock and is chosen only for that model. The provider of one region learns nothing about trips sent to
another: each sees only its own trips' public fields, as today.

Status: the module, the setting, the enrollment field and their tests ship; switching the engine's service URL
from the probe result is the next step (opt-in), once a multi-region deployment exists to test it against.

## RunPod `wan` session

`deploy/runpod/RUNBOOK.md` section 6: one H100 provider (default US-KS-2) and one CPU client pod per data centre
(same DC, US-TX-3, US-GA-2, EU-RO-1 by default; `--clients name=DC[:tokens],...`), measured one client at a time,
nearest first: TCP / TLS RTT, lane trips over the real path (stub and GPU listener, 7B / 32B widths), then the
client library at 0.5B and 7B (masks from the local factory pack: the same trips). ~$16.6 at table prices (TTL 175
min), ~$11 on an L40S; `jobs.py plan` totals $56.38 for all five sessions under the default $60 cap.
`python deploy/runpod/jobs.py datacenters` lists RunPod's data centres (read-only catalog call) to choose others.
