# Two-party PGW demo on Modal

A real provider/customer pair, each in its own Modal app, that you onboard through the gateway's
setup wizard in a browser. Model: `Qwen/Qwen2.5-0.5B-Instruct` (pinned revision).

| party | app | hardware | what runs |
|---|---|---|---|
| provider | `pgw-provider` (`provider.py`) | 4 vCPU, 6 GiB (or 1 x T4 + 1 vCPU with `PGW_MODAL_GPU=T4` at deploy time) | `pgw service serve` (TLS, tenant tokens, enrollment, client-cert issuing CA); pack built into the image |
| customer | `pgw-customer` (`customer.py`) | 4 vCPU, 8 GiB, no GPU | `pgw gateway serve` in setup mode; downloads and packs the model itself during the wizard, encrypted mask factory |

```
 browser ──HTTPS (TLS ends at Modal's edge)──▶ customer container: pgw gateway + console + mask factory
                                                     │  TLS end to end (raw TCP tunnel, CA pinned by the
                                                     ▼  enrollment code, client cert issued at enrollment)
                                               provider container: pgw compute service (CPU or T4)
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
reach its 4000-token stock. With a T4 provider the pair costs about $0.85/h. The images (CUDA torch
for the provider, CPU torch for the customer, pack built in) are built once and cached.

GPU functions need a payment method on the Modal workspace; without one, deploy with the default
(CPU provider). The service then runs the exact `cpu_f64` backend; results are bit-identical either way.

Measured on 2026-11-20 (CPU provider, customer 4 vCPU): enrollment 0.3 s, model download + pack +
fingerprint check 12 s, encrypted-mask hint 95 s (7.5 GB), first 256-token mask segment 125 s
(2 tokens/s over the tunnel, which carries about 1.5 MB/s), a 41-token prompt + 32 generated tokens
about 50-57 s.

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
* There is no co-located latency benchmark yet: the two apps run wherever Modal places them, so
  round-trip times here are not representative of a same-zone deployment.
* Both PKIs are generated in the containers for each run and live for two days at most.

## Files

* `provider.py`, `customer.py`: the two Modal apps.
* `pgw_modal_boot.py`: PKI, service and gateway start-up (no Modal imports; also used locally).
* `wizard_client.py`: drives `/console/api/setup/*` like the browser and checks a masked chat against
  the plaintext reference.
* `local_rehearsal.py`: the same bootstrap with both parties on one machine (CPU service).
* `launch.py`: deploy, start, show, stop, and the scripted end-to-end test.

## Verified end to end (2026-11-20)

`launch.py --e2e` on Modal (CPU provider): wizard completed through the public console URL (enrollment
with CA pin and client certificate, Hugging Face download + pack with matching fingerprint
`004f351fefb5`, mask store, admin token, all checks ok, finish switched to normal mode in-process); a
greedy chat completion through the masked path returned the same 32 tokens as the plaintext
reference on the same pack, with bit-identical logprobs (max difference 0.0), and a repeated request
was identical as well.
