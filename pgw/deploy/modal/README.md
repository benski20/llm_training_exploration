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
1 ms limit can pass only on i6pn (the tunnel alone adds about 5 ms): for a limit below 3 ms the launcher
does not relaunch a tunnel pair unless `--i6pn` was asked (each relaunch is a full paid run), and says
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
provider is remote). Strict speculation is available but off by default; see the next section.

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
  The scripted `--e2e` waits for both chats' slots before chatting (`pgw_modal_boot.e2e_min_masks`). Filler rows
  without masks (padding) would cut this to about R / (tokens per pass) slots per token. They are not used:
  with the encrypted factory on the provider, the slots taken would then reveal how many passes the answer
  took.
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
* `pgw_modal_boot.py`: PKI, service and gateway start-up (no Modal imports; also used locally).
* `wizard_client.py`: drives `/console/api/setup/*` like the browser and checks a masked chat against
  the plaintext reference.
* `trip_gate.py`: the client-reported trip latency (from the gateway's `/metrics` and `/admin/status`)
  and the `--max-trip-ms` decision; no Modal imports.
* `local_rehearsal.py`: the same bootstrap with both parties on one machine (CPU service).
* `launch.py`: deploy, start, show, stop, the scripted end-to-end test, the GPU switch, the i6pn
  scheduling fallback and the latency gate.

## Verified end to end (2026-11-20)

`launch.py --e2e` on Modal (CPU provider): wizard completed through the public console URL (enrollment
with CA pin and client certificate, Hugging Face download + pack with matching fingerprint
`004f351fefb5`, mask store, admin token, all checks ok, finish switched to normal mode in-process); a
greedy chat completion through the masked path returned the same 32 tokens as the plaintext
reference on the same pack, with bit-identical logprobs (max difference 0.0), and a repeated request
was identical as well.
