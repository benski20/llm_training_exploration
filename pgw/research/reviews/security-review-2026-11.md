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
| S10 | low | Provisioning routes (`/pgw/v1/provision/...`) ignore SDK key model scope and quotas. Artifacts are public, but hint jobs cost provider GPU/CPU | **fixed** |
| S11 | low | The SDK's integrity breaker and failure counter are in memory: a process restart resets them | **fixed** (persisted) |
| S12 | low | Requests that fail before reaching the provider (for example, context length) are still reported, with their error class | **fixed** (not reported) |
| S13 | low | Customer-side gateway routes (`gateway/setup.py`, `gateway/app.py`, `console/routes.py`) also buffer bodies without a bound | **fixed** |
| S14 | low | Provider images fetch the rustup installer over TLS with no hash (the toolchain version is pinned) | **fixed** |
| S15 | info | Rust: about 490 `unsafe` uses across five crates. I spot-checked the network-facing copies only | recommend audit (scope in section 4) |
| S16 | info | Smaller observations: unauthenticated `/admin/attest`, `pack_path` in enrollment replies, load error text in `/readyz` | no change |
| S17 | info | Supply chain: hash-pinned lockfiles, digest-pinned base images, no pickle, `torch.load(weights_only=True)`, no eval/exec | no issue |
| S18 | info | Crypto hygiene: CSPRNG, constant-time comparisons, nonces, HMAC domains | no issue |
| S19 | info | Selective failure and retries: integrity failures are fatal, never retried and data-independent (p31 check) | no issue |
| S20 | info | Strict speculation: whether real and dummy passes can be told apart by inter-pass timing has not been measured | measured; **fixed** (end-of-answer status write moved after the dummy passes); residual in section 2 |

PoC scripts (run against `HEAD` and against the fix):

* S1: 256 MiB buffered before the 413 → 1 MiB.
* S2: before the fix, enrollment succeeds, the SDK key is sent and the CA file trusts 2 certificates. After the fix,
  `IntegrityError` and nothing is sent.

## 1. What the provider observes

| observable | depends on | verdict |
|---|---|---|
| rows per trip, trips per request | prompt length (prefill rows), answer length L (decode passes) | inherent, documented (SPEC) |
| strict speculation | fixed `spec_rows` per pass, pass count = L (dummy passes after the answer) | holds on the wire; the report leaked the real pass count (**S4**, fixed); inter-pass timing measured: a status write at the end of the answer marked the boundary (**S20**, fixed) |
| prefix cache (`prefix_cache="auto"`, default without masked head) | prefill rows = new tokens only | wire view fine; the report leaked the reused length (**S3**, fixed) |
| masked LM head | one more masked linear per token; logits stay masked, sampling is local | no data-dependent observable found |
| message sizes | row counts × fixed widths | same as rows |
| errors / retries | only `unavailable` / `backend_unavailable` / `deadline_exceeded` before the first token are retried; integrity failures never | the provider chooses when to fail, so nothing data-dependent |
| mask demand | public schedule (prompt + chunk; `spec_mask_rows` public charge) | no issue (pks-v3 review) |
| content-free report | ts, model, status, error code, token counts, latency, ttft, stream, checks_ok | now limited to wire-observable values (S3, S4, S9); requests that never reached the provider are not reported (**S12**, fixed) |
| `/readyz` | other tenants' load | other-tenant channel (**S5**, fixed) |

The report is content-free: the client sends allow-listed fields, the server parses them by allow-list again, and no
text field remains (`model` matches `_MODEL_RE`, `error_code` is a public code). After S3, S4 and S9 it also
carries nothing the masked traffic does not already show. The exception is the end-to-end latency without
speculation (the same as the time from the first to the last trip, plus client CPU). Requests that never reached the
provider are no longer reported (S12).

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

### S10 (low, fixed): provisioning ignored SDK key scope and quotas

`service/provision.py` authenticates like compute but never calls `Accounts.admit`. A key scoped to model A can:

* download the public artifacts of model B (public; not a confidentiality issue);
* start hint jobs, two per tenant, which cost provider CPU/GPU.

Suggested fix: check the key's model scope in `fp_of` (`Accounts._model(fp).name in st.allowed`), and count hint
work against the row bucket.

**Fix (follow-up, same day).** `service/provision.py` applies `Accounts.admit` to every provisioning route, with the
model named by the base fingerprint's catalog entry (provisioning addresses base packs, which the compute registry
may not list):

* `/models` lists only the key's models; `manifest`, `object` and `hint` creation of another model are refused
  (403, `permission_denied`);
* a revoked or expired key, and a key or organisation over its daily token quota, are refused (401 / 429);
* each H chunk (256 weight rows) is charged to the key's row bucket. Over it the reply is 429 with `Retry-After`;
  `pgw.provision.session.ProvisionSession` waits that out (up to 60 s per reply, without spending a retry). A
  daily quota's `Retry-After` (until midnight) is not waited out.

Tests: `tests/service/test_security_review.py::test_provisioning_enforces_the_sdk_key_scope_quota_and_row_rate`,
`::test_provisioning_client_waits_out_the_row_rate_but_not_a_daily_quota`.

### S11 (low, fixed): integrity breaker not persisted

The SDK's in-process gateway opens its breaker on an integrity failure until `reset_integrity_breaker()`. The state
and `integrity_failures` live in memory and in `status.json`, which is written but not read back, so a restart clears
them.

Each failure is data-independent (S19), so this does not leak data. It only lets a provider retry tampering across
restarts: about 2^-31 escape probability per attempt with the p31 check. The factory side already persists its halt
(F3).

Decision: persist the breaker per key and model, mirroring `factory clear-halt`, or document a restart as a reset.

**Fix (decision: persist).** `pgw.sdk.halt`:

* One record per (key, model): `<state dir>/sessions/<key id>/integrity/<model>.json`, 0600, written atomically.
  It holds `halted`, the failure count, timestamps and a short event log (no secrets, no content).
* HMAC-SHA256 under `<state dir>/sessions/<key id>/integrity.key` (32 random bytes, 0600) with its own domain
  label `pgw-sdk-integrity-halt-v1\0`, over the canonical JSON of the record, the key id and the model.
* Fail closed: a record with a missing or wrong MAC, unknown fields, another key's or model's name, invalid JSON,
  or whose MAC key is gone reads as halted, and the request's `IntegrityError` says the record cannot be verified.
  No record means no failure was ever recorded.
* Every integrity failure the breaker sees (`ModelRuntime.safety_listeners`, so failed checks of background
  strict-mode dummy passes count too) is persisted before the breaker's listener returns. A new runtime trips its
  breaker from the record before it starts, and every request re-reads the record (cached by mtime, size and
  inode), so a halt reaches the other processes of the same key and state directory on their next request.
* Clearing is explicit: `pgw sdk clear-halt --model M [--key-id K] --yes [--note ...]` prints a warning (an honest
  provider never fails a check; running processes stay blocked until they restart) and keeps a `cleared` event,
  or `OpenAI.reset_integrity_breaker(model)` in code. Documented in `docs/sdk/troubleshooting.md`.

Tests: `tests/sdk/test_security_review.py::test_integrity_halt_is_persisted_private_and_mac_protected`,
`::test_engine_trips_the_breaker_from_the_persisted_halt`, `::test_sdk_clear_halt_cli`, and end to end against the
real provider `tests/sdk/test_e2e_cpu.py::test_integrity_failure_raises_and_blocks` (restart stays blocked, the CLI
clears, `reset_integrity_breaker` clears; now self-contained: its own key, cache and baseline answer).

### S12 (low, fixed): reports of requests that never reached the provider

A request refused locally still produces a record with status `error` and its code. Examples are a context-length
error (prompt + `max_tokens` too long) and a mask shortage. The record tells the provider that a request happened
and, for the context case, that the prompt was long, although it never saw the request.

Decision: either do not report requests that opened no session, or keep them for the console's error view and
document it.

**Fix (decision: do not report).** `Engine.chat_events` reports a request only once its mask session opened
(`meta["opened"]`, set after `_open`; its trips follow at once). A request that failed before that is counted by
error code in `client.status()["local_errors"]` (and per model in `status.json`) and logged at INFO without content.
Documented in `docs/sdk/CONTRACT.md` A3b and `docs/sdk/configuration.md` (`usage_reports`).

Test: `tests/sdk/test_security_review.py::test_requests_that_never_reached_the_provider_are_not_reported`.

### S13 (low, fixed): customer-side unbounded bodies

`gateway/setup.py:783`, `gateway/app.py:882/943` and `console/routes.py:285/412` use `request.body()` /
`request.json()` without a bound. They are the customer gateway's local surfaces, usually loopback or behind the
customer's own auth, so the exposure is lower. The same `read_limited` helper applies.

**Fix.** `read_limited` on every one of them:

* the setup wizard's `_json` (64 KiB; its sign-in takes no credential, so this was the exposed one);
* the gateway's `POST /admin/keys` and `/admin/models/{name}/reset` (64 KiB; the app-wide `_BodyLimit` middleware
  allowed 4 MiB);
* the console's sign-in (16 KiB) and playground (`max_body_bytes`).

Tests: `tests/gateway/test_security_review.py` (2).

### S14 (low, fixed): rustup installer without a hash

`deploy/docker/provider*.Dockerfile` run `curl https://sh.rustup.rs | sh` (into a file, then `sh`). TLS protects it,
but its hash is not checked. The toolchain version is pinned, and `cargo fetch --locked` honours `Cargo.lock`.
Suggested fix: a digest-pinned `rust:<ver>` build stage, or the checksum of `rustup-init`.

**Fix.** Both provider Dockerfiles download `rustup-init` 1.28.2 from
`static.rust-lang.org/rustup/archive/<version>/<triple>/rustup-init` and check it with `sha256sum -c` before
running it. The sums (`RUSTUP_INIT_SHA256_X86_64`, `RUSTUP_INIT_SHA256_AARCH64`) were taken from the published
`rustup-init.sha256` files and match the downloaded binaries. Another architecture fails the build. rustup then
checks every toolchain component against the channel manifest's sha256. The images were not rebuilt here (no
change to anything after the installer). Test: `tests/deploy/test_provider_image.py` (pinned version, sums per
architecture, check before run, no `sh.rustup.rs`).

### S15 (info): Rust `unsafe`

Counts: `pgw-fast` 274, `pgw-pks` 106, `pgw-lane` 75, `pgw-ntt` 35, `pgw-core-rs` 1.

I checked the paths that copy peer-controlled bytes:

* `pgw-lane` `cuda.rs` `run_key_into` / `run_sharded` and `persistent.rs`: `body.len() == in_len` is checked
  before each `copy_nonoverlapping`.
* Tickets: `server.rs` `ticket_eq` is constant time, and tickets are single use.
* Customer side, `pgw-pks` `decrypt_item`: lengths are re-checked (review F17).

This is not an audit. An external review of `pgw-fast`, which parses provider replies on the client, is
recommended before general availability. Section 4 lists the scope.

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

### S20 (measured, fixed): timing of strict-mode dummy passes

The row count and the pass count of dummy passes match real ones, and with `spec_mask_rows` the gateway's work per
call depends only on the row count. Real passes also run sampling and, with `draft_head`, a draft step; dummy passes
run `_dummy_samples`. If their client CPU differs, the inter-pass gap may separate the answer's passes from the
dummy tail, and so reveal the real pass count again.

**Method.** `benchmarks/sdk_dummy_pass_timing.py`:

* The client library runs with `speculation="lookup"` (`spec_rows=3`) against the real `pgw service` of the SDK
  tests (TLS, TCP+TLS trip lane, masked LM head).
* A TCP relay in front of the lane records what the provider's network sees: when each request arrives, its size,
  and when the reply leaves. Only these numbers are used as the provider's features.
* In the client, each pass is tagged real, dummy or prefill (`Generator._plan_fixed`). The tags are used only to
  label the passes.
* Per decode pass, the features are:
  * the gap from the last reply of the previous pass to the first request of this one;
  * the mean gap between the trips of the pass;
  * the provider's time per trip;
  * the pass duration;
  * request and reply bytes.
* For each feature the script reports:
  * the Kolmogorov-Smirnov distance D, real vs dummy, with its 5 % critical value;
  * the balanced accuracy of the best single-threshold classifier (0.5 means indistinguishable);
  * the same for the first dummy pass of each request, the boundary;
  * an attack on the S4 quantity, which takes the request's largest gap as the boundary. Chance is about 1/24.
* As a control, D between the early and the late half of each kind's passes. Time and position effects, such as
  load from other processes on the host, show up there too.
* Setup: tiny model, CPU, 4 vCPUs shared with other jobs (load average 14 to 20). 48 requests of 32 tokens over
  four prompts (repeat, summarise, poem, list), non-streaming and streaming. The library's real pass counts were
  8 to 12 of 32.

**Result before the fix: distinguishable at the boundary.**

| | non-stream | stream |
|---|---|---|
| steady-state gap, real / dummy median | 2.54 / 2.54 ms | 2.55 / 2.44 ms |
| steady-state gap, D (critical) / accuracy | 0.089 (0.083) / 0.544 | 0.091 (0.083) / 0.545 |
| intra-pass gap D, provider time D, duration D | 0.030, 0.038, 0.054 | 0.047, 0.049, 0.040 |
| **first dummy gap**, median / p90 (real: 2.5 / 4.9 ms) | **4.47 / 22.5 ms** | **6.36 / 23.6 ms** |
| first dummy vs real gap, D (critical) / accuracy | **0.452 (0.207) / 0.726** | **0.479 (0.207) / 0.739** |
| largest gap = the real/dummy boundary | **13/48** | **15/48** |

Earlier runs of the same configuration found the boundary in 11/40, 11/40, 9/40 and 12/40 requests. Request and
reply sizes are equal per trip, and they come in the same order in real and dummy passes.

**Cause.** At the end of the answer, `Engine._chat_events` wrote `status.json` on the client's event loop: JSON,
a file lock and `fsync`, at most every 2 s. The write fell between the last real pass and the first dummy pass. A
diagnosis run without the write found the boundary in 2/40 requests. A smaller residual remained: the application
thread wakes up with its answer and finishes it (GIL), and the generator's consumer runs its end-of-request work.

**Fix.**

1. With strict speculation, the status write at the end of a request waits until the generator is idle
   (`Engine._status_after_passes`). It does not poll the loop: a periodic wake-up during the dummy passes would
   be a difference of its own. Its time then depends only on the public pass count.
2. A pass-gap floor `spec_pass_gap_ms`, 8 ms by default with speculation, 0 to turn it off
   (`Engine._pace_strict_passes`). A forward pass starts no earlier than the floor after the previous pass ended,
   so client work shorter than the floor is not visible. Rows, trips and outputs are unchanged.

**Result after the fix.**

| | after 1 only, non-stream / stream | after 1 + 2 (default), non-stream / stream |
|---|---|---|
| largest gap = boundary | 2/48 / 1/48 | **1/48 / 1/48** |
| first dummy vs real gap, D (critical) / accuracy | 0.249 (0.207) / 0.625; 0.238 / 0.619 | **0.162 (0.207) / 0.581; 0.133 / 0.567** |
| steady-state gap, real vs dummy D / accuracy | 0.168 / 0.584; 0.145 / 0.572 | 0.119 / 0.560; 0.133 / 0.567 |
| control D, dummy early vs late (critical 0.103) | 0.138; 0.109 | 0.132; 0.086 |
| pass period, median | 24.7 / 23.6 ms | 31.5 / 32.0 ms |

With the floor:

* The boundary attack is at chance.
* The first dummy pass is within the 5 % critical value in these two runs. Across four earlier floor runs, D was
  0.176, 0.161, 0.222 and 0.357, so one of the four was above the critical value.
* The remaining real-vs-dummy D of the steady-state gap (about 0.12) is the same size as the control between
  early and late dummy passes. On this loaded host, that is the size of time and position effects.

Cost: the floor minus the usual gap, per pass. On the tiny model that is about 8 ms on about 24 ms (+34 %). On real
models, a pass of `4 x layers + 1` round trips takes about 100 to 300 ms or more, so the cost is 3 to 8 %.

Limits:

* Only the tiny model was measured. A plain-mask run of Qwen2.5-0.5B needs 1.8 MB per slot, about 10 GB of mask
  stock for a useful run, and the shared disk had 2 GB free. The client work that scales with the model, such as
  sampling 3 rows over the vocabulary, is the same for real and dummy passes (`_dummy_samples`). The effects
  found here (a file write, the application waking up) do not depend on the model.
* An application that does heavy CPU work right after an answer still delays the next pass by more than the
  floor in the same Python process. A larger `spec_pass_gap_ms` covers more of it.
* Observation, not caused by real or dummy passes: in about 1 pass in 15, the head call spans two mask extents.
  It then goes as a 2-item request over HTTPS instead of the lane, so the lane shows 8 trips instead of 9. These
  passes fall at fixed pass indices of a request, every 3 passes from index 20 here. Those indices depend on the
  prompt length and the mask reservation's chunking, not on the drafts. Follow-up: carry 2-extent calls on the
  lane.

Tests:

* `tests/sdk/test_security_review.py::test_strict_passes_start_no_earlier_than_the_gap_floor`;
* the strict-speculation e2e tests (`tests/sdk/test_speculation.py`) run with the floor. Answers stay bit-identical
  and the provider-metered trips and rows stay equal.

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

Follow-up (S10 to S14, S20), same rules:

* `src/pgw/sdk/halt.py` (new), `src/pgw/sdk/options.py` (`spec_pass_gap_ms`), `src/pgw/sdk/engine.py`
  (persisted halt, `local_errors`, no report before a session, status write after the dummy passes, pass-gap
  floor), `src/pgw/sdk/cli.py` (`pgw sdk clear-halt`, halts in `status`);
* `src/pgw/service/provision.py`, `src/pgw/service/accounts.py` (`admit(model_name=)`, `allowed_models`),
  `src/pgw/provision/session.py` (waits out a short 429);
* `src/pgw/gateway/setup.py`, `src/pgw/gateway/app.py`, `src/pgw/console/routes.py` (bounded bodies);
* `deploy/docker/provider.Dockerfile`, `deploy/docker/provider-cpu.Dockerfile` (verified rustup-init);
* `benchmarks/sdk_dummy_pass_timing.py` (new; S20 measurement);
* docs: `docs/sdk/troubleshooting.md`, `docs/sdk/CONTRACT.md` A3b, `docs/sdk/configuration.md`;
* tests: `tests/sdk/test_security_review.py` (+5), `tests/service/test_security_review.py` (+2),
  `tests/gateway/test_security_review.py` (new, 2), `tests/deploy/test_provider_image.py`,
  `tests/sdk/test_e2e_cpu.py` (integrity test self-contained), `tests/console/test_console_build.py` (wheel name
  `pgw_sdk-*.whl`), `tests/sdk/test_unit.py` (a fake in-flight request marks its session open).

## 4. External audit scope (S15)

Priority order for an external review of the Rust `unsafe` code. Counts are lines containing `unsafe` (blocks,
`unsafe fn`, `unsafe impl` and a few comments), from `grep -rw unsafe` on this branch.

1. **`rust/pgw-fast`** (273; client side, parses and unmasks provider replies, so a malicious provider controls
   the input):
   * reply path first: `src/trip.rs` 17, `src/trip/split.rs` 14, `src/tripnative.rs` 25, `src/lane.rs` 4
     (lane client framing, `from_le_bytes` length fields), `src/finish.rs` 29 (unmask, range and Freivalds checks),
     `src/prime.rs` 8 (p31 check), `src/p61.rs` 2, `src/decode_f2.rs` 6, `src/bitpack.rs` 10 (maskgen wire
     unpacking), `src/hemask.rs` 11, `src/store.rs` 14 (decrypted mask rows), `src/lane_service.rs` 2;
   * local compute, lower priority (inputs are the client's own): `src/lmhead.rs` 49, `src/attn.rs` 31,
     `src/localgpu.rs` 21, `src/prepare.rs` 12, `src/local.rs` 7, `src/writer.rs` 6, `src/layer.rs` 2,
     `src/trip/tests.rs` 3 (tests).
2. **`rust/pgw-lane`** (75; provider side, parses gateway requests from tenants): `src/shm.rs` 31 (shared-memory
   transport between users), `src/cuda.rs` 25 (driver FFI, `copy_nonoverlapping` of request bodies),
   `src/persistent.rs` 8, `src/server.rs` 6 (framing, tickets), `src/coalesce.rs` 3, `src/lib.rs` 2.
3. Then `rust/pgw-pks` (101; customer-side pks-v3 decryption, F17 re-checks lengths) and `rust/pgw-ntt` (35).
   `rust/pgw-core-rs` has none.

For each: every raw-pointer copy and slice construction must be bounded by a length checked against the buffer it
reads, after any peer-supplied length field is parsed; FFI and shared-memory regions must be validated before
use; `unsafe impl Send/Sync` must be justified. Fuzz targets for the lane framing (`pgw-fast/src/lane.rs`,
`pgw-lane/src/server.rs`, `src/shm.rs`) and the reply decoders (`trip.rs`, `finish.rs`, `bitpack.rs`) are the most
useful single addition.
