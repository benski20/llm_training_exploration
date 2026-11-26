# Security review of the SDK and the provider before the client pilot (2026-11)

Date: 2026-11-26. Reviewer: an internal adversarial review (not an external audit). Branch
`claude/charming-carson-qxlk5v`.

Scope: the SDK (`src/pgw/sdk`, with the parts of `pgw.enroll`, `pgw.gateway` and `pgw.provision` it runs), the
provider (`src/pgw/service`: accounts, admin API, `/pgw/v1/sdk/*`, provisioning, enrollment, lane, `/readyz`), the
developer console (`src/pgw/platform`), the Docker images (`deploy/docker`), the RunPod launcher (`deploy/runpod`),
lockfiles and the Rust crates (spot checks only).

Threat model, as in `SPEC.md` and `research/reviews/pks-v3-*.md`:

* The provider is the adversary for data privacy and result integrity.
* Network attackers and other tenants are adversaries for everything.
* Hiding weights from customers is out of scope.

Method: I read the code paths above. I wrote a proof of concept against the code before the fix where one was
feasible (the S1 and S2 scripts were run against `HEAD` and against the fixed tree). Each fix has a regression test.
The pks-v3 scheme, its parameters and the model outputs are unchanged.

## Summary

| # | sev | finding | status |
|---|---|---|---|
| S1 | medium | Provider JSON routes buffer a chunked body of any size before checking it. Unauthenticated memory DoS on `/pgw/v1/sdk/enroll`, `/pgw/v1/enroll`, `/pgw/v1/enroll/renew`, `/pgw/v1/admin/attest` and `/platform/login` | **fixed** |
| S2 | **high** | CA pin bypass: only the *first* certificate of the CA reply is checked, but the whole reply becomes the trust store. An on-path attacker at first enrollment appends its own CA, receives the SDK key and service token, and stays trusted in `service-ca.pem` afterwards. The same applies to gateway enrollment codes and the RunPod launcher | **fixed** |
| S3 | medium | The request report sends the full prompt length, including tokens reused from the conversation prefix cache, which never reach the provider. This tells the provider how long a prefix matched an earlier turn (default path) | **fixed** |
| S4 | medium | With strict speculation, the report's `latency_ms` measures the answer, which returns before the dummy passes. That is the number of real passes, which gives the drafts' acceptance, a function of the text. Strict mode hides this (opt-in path) | **fixed** |
| S5 | medium | Unauthenticated `/readyz` on the public port returns `inflight`, `queued_rows` and lane connection counts. Polled, these show when other tenants send work and how many rows | **fixed** |
| S6 | low | Client-certificate CN `pgw-gw.<tenant>.<name>`, cut at 64 characters, can collide across tenants whose ids contain dots or are long (code enrollment). Re-mapping the CN moves the other tenant's mTLS identity | **fixed** |
| S7 | low | A cached SDK session whose `service-ca.pem` is missing falls back to the system trust store. The cached CA is not re-checked against the key's pin | **fixed** |
| S8 | low | `/pgw/v1/sdk/report` has no rate limit (256 records per call). One installation can grow the request log and usage rows without bound | **fixed** |
| S9 | low | Report `error_code` is not allow-listed on the client (the server's regex is the only guard) | **fixed** |
| S10 | low | Provisioning routes (`/pgw/v1/provision/...`) ignore SDK key model scope and quotas. Artifacts are public, but hint jobs cost provider GPU/CPU | open |
| S11 | low | The SDK's integrity breaker and failure counter are in memory: a process restart resets them | open (decision) |
| S12 | low | Requests that fail before reaching the provider (for example, context length) are still reported, with their error class | open (decision) |
| S13 | low | Customer-side gateway routes (`gateway/setup.py`, `gateway/app.py`, `console/routes.py`) also buffer bodies without a bound | open |
| S14 | low | Provider images fetch the rustup installer over TLS with no hash (the toolchain version is pinned) | open |
| S15 | info | Rust: about 490 `unsafe` uses across five crates. I spot-checked the network-facing copies only | recommend audit |
| S16 | info | Smaller observations: unauthenticated `/admin/attest`, `pack_path` in enrollment replies, load error text in `/readyz` | no change |
| S17 | info | Supply chain: hash-pinned lockfiles, digest-pinned base images, no pickle, `torch.load(weights_only=True)`, no eval/exec | no issue |
| S18 | info | Crypto hygiene: CSPRNG, constant-time comparisons, nonces, HMAC domains | no issue |
| S19 | info | Selective failure and retries: integrity failures are fatal, never retried and data-independent (p31 check) | no issue |
| S20 | info | Strict speculation: whether real and dummy passes can be told apart by inter-pass timing has not been measured | open (measure) |

PoC scripts (run against `HEAD` and against the fix):

* S1: 256 MiB buffered before the 413 → 1 MiB.
* S2: before the fix, enrollment succeeds, the SDK key is sent and the CA file trusts 2 certificates. After the fix,
  `IntegrityError` and nothing is sent.

## 1. What the provider observes

| observable | depends on | verdict |
|---|---|---|
| rows per trip, trips per request | prompt length (prefill rows), answer length L (decode passes) | inherent, documented (SPEC) |
| strict speculation | fixed `spec_rows` per pass, pass count = L (dummy passes after the answer) | holds on the wire; the report leaked the real pass count (**S4**, fixed); inter-pass timing not measured (**S20**) |
| prefix cache (`prefix_cache="auto"`, default without masked head) | prefill rows = new tokens only | wire view fine; the report leaked the reused length (**S3**, fixed) |
| masked LM head | one more masked linear per token; logits stay masked, sampling is local | no data-dependent observable found |
| message sizes | row counts × fixed widths | same as rows |
| errors / retries | only `unavailable` / `backend_unavailable` / `deadline_exceeded` before the first token are retried; integrity failures never | the provider chooses when to fail, so nothing data-dependent |
| mask demand | public schedule (prompt + chunk; `spec_mask_rows` public charge) | no issue (pks-v3 review) |
| content-free report | ts, model, status, error code, token counts, latency, ttft, stream, checks_ok | now limited to wire-observable values (S3, S4, S9); pre-flight errors still reported (**S12**) |
| `/readyz` | other tenants' load | other-tenant channel (**S5**, fixed) |

The report is content-free: the client sends allow-listed fields, the server parses them by allow-list again, and no
text field remains (`model` matches `_MODEL_RE`, `error_code` is a public code). After S3, S4 and S9 it also
carries nothing the masked traffic does not already show. The exceptions are the end-to-end latency without
speculation (the same as the time from the first to the last trip, plus client CPU) and the reports of requests that
never reached the provider (S12).

## 2. Findings

### S1 (medium, fixed): unbounded request bodies on JSON routes

**Evidence.** These routes check `Content-Length` and then call `await request.body()`:

* `service/accounts.py` `_read_json` (`/pgw/v1/sdk/enroll`, `/pgw/v1/sdk/report`, all admin routes,
  `/pgw/v1/admin/attest`);
* `enroll/service.py` `post_enroll`, `post_renew`;
* `service/provision.py` `hint_create`, `service/lane.py` `lane_open`;
* `platform/app.py` `form()` (`/platform/login`).

A chunked request has no `Content-Length`, so the check passes, the whole body is buffered, and only then is it
compared with the 64 KiB limit. Several of these routes need no credential: `/pgw/v1/sdk/enroll` reads the body
before checking the key, and `/admin/attest` and `/platform/login` take no credential at all.

PoC: `tests/service/test_security_review.py::test_unauthenticated_json_routes_do_not_buffer_chunked_bodies`. A
scratch script measured 256 MiB buffered before the reply with the old code.

**Fix.** `pgw.core.http_body.read_limited` reads the stream and stops at the limit. It is used by every route
above.

### S2 (high, fixed): CA pin covers only the first certificate of the reply

**Evidence.** `enroll/client.py` `_fetch_pinned_ca`:

* It checks `ca_fingerprint(pem)`, which is the SHA-256 of the *first* certificate (`load_pem_x509_certificates(..)[0]`).
* It returns the whole text. `sdk/session.py` `_enroll` and `enroll/client.py` then build
  `ssl.create_default_context(cadata=ca_pem)`, which trusts *every* certificate in it, and write the text to
  `service-ca.pem`.

Attack: an on-path attacker at the first use of an SDK key, or of a gateway enrollment code, serves `genuine CA +
attacker CA`. The pin matches. The attacker's server certificate, signed by the attacker CA, is accepted. The
attacker then gets the following:

* the SDK key (sent as the bearer of `/pgw/v1/sdk/enroll`);
* the service token and the client certificate;
* a persistent man-in-the-middle position on every later connection: compute, lane, provisioning and reports.

`deploy/runpod/launch.py` `attested_ca` has the same pattern, and the admin token is sent over that context.

PoC: `tests/sdk/test_security_review.py::test_appended_attacker_ca_is_refused_before_the_key_is_sent`. A scratch
script against `HEAD` printed `ENROLLED; key sent: True; certs trusted in CA file: 2`.

**Fix.**

* `pgw.enroll.service.pinned_ca` requires exactly one certificate and returns its canonical PEM. The provider only
  ever serves one (`load_ca_pem`).
* `_fetch_pinned_ca` and `attested_ca` use it, so only the pinned certificate is trusted or written.
* A reply with more than one certificate is an integrity error, raised before anything secret is sent.

### S3 (medium, fixed): the report revealed prefix-cache reuse

**Evidence.**

* `sdk/engine.py` reported `prompt_tokens = len(sess.prompt_ids)`.
* With `prefix_cache` (default `"auto"` without the masked head), only `len(ids) - prefix_reused` prompt rows
  reach the provider.
* The difference is exactly how many tokens of this prompt equal the cached conversation. That links requests into
  conversations and shows where an edited history diverged, which the masked traffic does not show.

**Fix.** `pgw.sdk.reports.build_record` reports the prompt rows actually sent (`prompt_tokens - cached_tokens`).
The OpenAI `usage` returned to the caller is unchanged.

Test: `test_report_counts_only_prompt_rows_sent_to_the_provider`.

### S4 (medium, fixed): the report's latency undid strict speculation

**Evidence.** The design of strict speculation:

* every pass has `spec_rows` rows;
* after the answer, dummy passes run until the provider has seen L passes;
* the answer is returned to the caller *before* those dummy passes (`engine/generate.py`: "a strict-mode result
  returned before its background dummy passes have run").

`latency_ms` measured to the end of the answer. Divided by the per-pass time (visible on the wire), it gives the
number of real passes, and L divided by that gives the acceptance rate. With `speculation="lookup"`, that rate shows
how much the answer copies the prompt.

**Fix.** `latency_ms` is `null` when `speculation != "off"`. `ttft_ms` is kept: the first token always comes from
the prefill pass.

Test: `test_report_has_no_latency_with_strict_speculation`.

### S5 (medium, fixed): `/readyz` showed other tenants' load

**Evidence.** `service/app.py` `readiness()` returned `inflight`, `queued_rows` and `lane.connections` on the main
listener without authentication. A tenant or a network observer polling it sees the prefill bursts (prompt lengths)
and decode cadence of the other tenants.

**Fix.** `readiness(public=True)` on the main port leaves these out. The metrics port (`create_metrics_app`,
loopback by default in the images) keeps them.

Test: `test_public_readyz_has_no_load_figures`.

### S6 (low, fixed): client-certificate CN collisions across tenants

**Evidence.** `enroll/service.py` `issue_credentials` builds `cn = f"pgw-gw.{tenant}.{name}"[:64]` and then calls
`authenticator.add_subject(cn, tenant)`, which overwrites any existing mapping. Two cases collide:

* tenant `a` with name `b.c`, and tenant `a.b` with name `c`;
* two long tenant ids with a common 57-character prefix.

A colliding enrollment moves the other certificate's identity. mTLS-only calls then authenticate as the wrong
tenant. SDK tenants (`sdk.<16 hex>`) cannot collide with each other; code-enrolled tenants can.

**Fix.** Enrollment is refused (403) before anything is issued when the CN already belongs to another tenant.

Test: `test_colliding_client_certificate_name_is_refused`.

### S7 (low, fixed): cached session without its CA

**Evidence.** In `sdk/session.py`:

* `load_session` set `ca_file=None` when `service-ca.pem` was missing.
* `Session.http_client` then used `ssl.create_default_context()`, the system trust store, as did the gateway
  clients.
* The cached CA was never re-checked against the key's pin.

**Fix.** `load_session` requires, for a pinned key, a CA file holding exactly the pinned certificate. Otherwise the
library enrolls again, which re-fetches the CA and checks it against the pin.

Test: `test_cached_session_with_a_replaced_or_missing_ca_is_not_used`.

### S8 (low, fixed): no rate limit on reports

**Evidence.** `Accounts.record_report` accepted any number of 256-record calls. Each call writes rows to the
`requests` table and the usage rows.

**Fix.**

* Per-key token bucket `accounts_report_records_per_s` (default 20/s, burst 4096); over it the reply is 429.
* The client `Reporter` keeps the batch on 429. Before, it dropped every 4xx.

Tests: `test_report_route_is_rate_limited_per_key`, `test_reporter_keeps_records_on_429`.

### S9 (low, fixed): client-side error-code allow-list

**Evidence.** `engine.chat_events` reported `str(getattr(exc, "code", ""))` of any exception. Only the server's
regex `[a-z0-9_.-]{1,64}` bounded it.

**Fix.** `build_record` maps anything other than an `ErrorCode` value, `cancelled` or `deadline_exceeded` to
`internal`.

Test: `test_report_error_codes_are_public_codes_only`.

### S10 (low, open): provisioning ignores SDK key scope and quotas

`service/provision.py` authenticates like compute but never calls `Accounts.admit`. A key scoped to model A can:

* download the public artifacts of model B (public; not a confidentiality issue);
* start hint jobs, two per tenant, which cost provider CPU/GPU.

Suggested fix: check the key's model scope in `fp_of` (`Accounts._model(fp).name in st.allowed`), and count hint
work against the row bucket.

### S11 (low, open, decision): integrity breaker not persisted

The SDK's in-process gateway opens its breaker on an integrity failure until `reset_integrity_breaker()`. The state
and `integrity_failures` live in memory and in `status.json`, which is written but not read back, so a restart clears
them.

Each failure is data-independent (S19), so this does not leak data. It only lets a provider retry tampering across
restarts: about 2^-31 escape probability per attempt with the p31 check. The factory side already persists its halt
(F3).

Decision: persist the breaker per key and model, mirroring `factory clear-halt`, or document a restart as a reset.

### S12 (low, open, decision): reports of requests that never reached the provider

A request refused locally still produces a record with status `error` and its code. Examples are a context-length
error (prompt + `max_tokens` too long) and a mask shortage. The record tells the provider that a request happened
and, for the context case, that the prompt was long, although it never saw the request.

Decision: either do not report requests that opened no session, or keep them for the console's error view and
document it.

### S13 (low, open): customer-side unbounded bodies

`gateway/setup.py:783`, `gateway/app.py:882/943` and `console/routes.py:285/412` use `request.body()` /
`request.json()` without a bound. They are the customer gateway's local surfaces, usually loopback or behind the
customer's own auth, so the exposure is lower. The same `read_limited` helper applies.

### S14 (low, open): rustup installer without a hash

`deploy/docker/provider*.Dockerfile` run `curl https://sh.rustup.rs | sh` (into a file, then `sh`). TLS protects it,
but its hash is not checked. The toolchain version is pinned, and `cargo fetch --locked` honours `Cargo.lock`.
Suggested fix: a digest-pinned `rust:<ver>` build stage, or the checksum of `rustup-init`.

### S15 (info): Rust `unsafe`

Counts: `pgw-fast` 274, `pgw-pks` 106, `pgw-lane` 75, `pgw-ntt` 35, `pgw-core-rs` 1.

I checked the paths that copy peer-controlled bytes:

* `pgw-lane` `cuda.rs` `run_key_into` / `run_sharded` and `persistent.rs`: `body.len() == in_len` is checked
  before each `copy_nonoverlapping`.
* Tickets: `server.rs` `ticket_eq` is constant time, and tickets are single use.
* Customer side, `pgw-pks` `decrypt_item`: lengths are re-checked (review F17).

This is not an audit. An external review of `pgw-fast`, which parses provider replies on the client, is
recommended before general availability.

### S16 (info): smaller observations, no change

* `POST /pgw/v1/admin/attest` is unauthenticated. It returns `HMAC(SHA-256(bootstrap token), nonce | ca_sha256)`
  for an attacker-chosen nonce, bound to the real CA fingerprint. It cannot attest another CA. It reveals the number
  of bootstrap tokens.
* `pod_spec` puts the admin token's SHA-256 in the pod env, so RunPod can attest. That is acceptable, because RunPod
  hosts the pod.
* The enrollment reply lists the provider's `pack_path` (the SDK drops it). `/readyz` shows a load error string.
* Admin API: every org route checks the admin token's org scope, and key revocation checks the key's org. SQL is
  parameterised everywhere; the only f-strings name fixed column tuples.
* Console: CSP `default-src 'self'`, CSRF token on every form, `SameSite=Strict` cookie, same-origin check, all
  values `html.escape`d, CSV formula-injection guard.

### S17 (info): supply chain

* `requirements/*.txt`: all pins `==` with `--hash` (`pip install --require-hashes`).
* Base images are digest-pinned. Runtime images run as uid 10001: `USER` in most, `setpriv` with no caps and
  `no_new_privs` in the provider images.
* Exposed ports: 8443 service and 8444 lane. Metrics stay on 127.0.0.1.
* The wheel ships only `pgw` plus static assets. No private keys are in `src/` or `deploy/`.
* `torch.load` everywhere uses `weights_only=True`. `np.load` uses the default `allow_pickle=False` (two calls
  pass it explicitly). There is no `pickle`, `eval`, `exec`, `yaml.load` or `shell=True` in `src/`.

### S18 (info): crypto hygiene

* Secrets come from `os.urandom` or `secrets`: SDK keys, tokens, tickets, DEKs, nonce prefixes, segment seeds and
  pks-v3 seeds. `random` is used only for retry jitter, demo fixtures and the noise monitor's sampling (which is
  seeded from `os.urandom`).
* Constant-time comparisons: tenant tokens (all entries scanned), SDK key digests, admin tokens, the SDK key
  checksum, CSRF, hint-job tenant, lane tickets, MACs (store meta/rows, journal, hint `meta.json`/partial). The
  remaining `!=` compare public SHA-256 digests of public artifacts.
* Nonces:
  * Mask segments use AES-GCM with a fresh random DEK per segment, a random 4-byte prefix and a chunk counter.
  * The KEK wrap uses a random 12-byte nonce.
  * pks-v3 runs ChaCha20 under distinct keys (F14).
* HMAC domains are separate: `pgw/v1/meta|`, `pgw/v1/segment`, `pgw/v1/journal\0`, `pgw-pks-hint-meta\0`,
  `pgw-pks-hint-partial\0`, `pgw-admin-attest-v1`, `pgw-sdk-key-v1`, `pgw-sdk-key-secret-v1`.
* Secrets in logs and reprs: `SDKKey.__repr__` and `Session` hide secrets, tokens are `SecretStr`, and key-parse
  errors never contain the key (`from None`). Audit records carry token ids only.

### S19 (info): selective failure and retries

* The SDK gateway runs `freivalds_rate=1.0`.
* The prime-field (p31) check is on by default with the native stack: `_prime_check` is auto, or explicit with
  strict speculation. It detects any wrong mask or product with probability 1 - 2^-31, whatever the data.
* `CoreIntegrityError` is never retried (`engine._chat_events`). It opens the breaker and is counted in
  `integrity_failures`.
* Retries happen only before the first token and only for `unavailable`, `backend_unavailable` and
  `deadline_exceeded`. They open a new session with fresh masks, so the same masked values are never resent.
* The factory counts provider integrity failures persistently and halts after 3 (F3).

I found no path where whether a failure happens depends on the data.

### S20 (open, measure): timing of strict-mode dummy passes

The row count and the pass count of dummy passes match real ones, and with `spec_mask_rows` the gateway's work per
call depends only on the row count. Real passes also run sampling and, with `draft_head`, a draft step; dummy passes
run `_dummy_samples`. If their client CPU differs, the inter-pass gap may separate the answer's passes from the
dummy tail, and so reveal the real pass count again.

Suggested follow-up: measure the gap distributions of real and dummy passes on the reference client (CPU only, no
cloud). Pad the dummy passes' client work if they differ.

## 3. Files changed

* `src/pgw/core/http_body.py` (new): `read_limited`.
* `src/pgw/service/accounts.py`:
  * `_read_json` is bounded;
  * per-key report bucket.
* `src/pgw/service/config.py`: `accounts_report_records_per_s`.
* `src/pgw/service/app.py`: public `/readyz` view.
* `src/pgw/service/provision.py`, `src/pgw/service/lane.py`, `src/pgw/platform/app.py`: bounded bodies.
* `src/pgw/enroll/service.py`:
  * `pinned_ca`;
  * bounded bodies;
  * CN collision refusal.
* `src/pgw/enroll/client.py`: only the pinned certificate is trusted.
* `src/pgw/sdk/session.py`: cached CA re-checked against the pin.
* `src/pgw/sdk/reports.py`:
  * `build_record`;
  * keep the batch on 429.
* `src/pgw/sdk/engine.py`: reports built by `build_record`.
* `deploy/runpod/launch.py`: `attested_ca` uses `pinned_ca`.
* `docs/sdk/CONTRACT.md` A3b.
* Tests: `tests/sdk/test_security_review.py` (7), `tests/service/test_security_review.py` (5).

Outputs are bit-identical: no model, mask, scheme or check path was touched.
