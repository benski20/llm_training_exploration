# Reliability: what happens when something fails

This page states what the client library (`from pgw import OpenAI`) does on each kind of failure, what your
application sees, and what recovers by itself. Error codes are those of [Troubleshooting](troubleshooting.md#errors-during-requests).
Every statement here is covered by a test in `tests/sdk/test_reliability.py` (end to end, against a real service
behind a fault-injection proxy) or `tests/sdk/test_reliability_unit.py`, unless marked otherwise.

## Rules that never change

* **A failed integrity check is final.** A provider result that fails its check (Freivalds, prime-field or range
  check, the outbound-noise monitor) raises `IntegrityError` and blocks the model until you call
  `reset_integrity_breaker()`. It is never retried, not by the library and not by a reconnect, so a provider
  cannot probe by answering wrongly and watching for a retry.
* **A mask is used once.** A request that stops for any reason (error, timeout, cancellation, `close()`, a dropped
  stream) burns the masks it reserved and did not use. A transport retry re-sends the *same* masked bytes of the
  same round trip (the provider learns nothing new from a copy); it never takes a new mask for old data.
* **Answers are bit-identical.** Every answer that completes is the answer of the plaintext reference, whatever
  failed and recovered on the way (the end-to-end tests compare every recovered answer bit for bit).
* **Retries stop at the first token.** Before the first token of an answer, transient failures
  (`backend_unavailable`, `unavailable`, `deadline_exceeded`) are retried `max_retries` times (default 2) with
  back-off. After the first token, the request fails: a streamed answer is never silently restarted.
* **Errors carry no secrets.** Messages and `details` never contain prompts, answers, keys, tokens or masks; the
  paths they name are your cache directories.

## Network and provider

| what happens | what you see | recovery |
|---|---|---|
| a connection is reset mid-answer (network blip, load balancer) | usually nothing: the round trip is sent again on a new connection (lane, then the stream channel, then HTTP); else a retryable error | automatic; the next request opens new connections |
| the provider is killed or restarts mid-answer | `APIConnectionError` or `NotReadyError` (`backend_unavailable` / `unavailable`, `retryable=True`) at once | the same client serves again as soon as the provider is back (new lane ticket, new connections); no restart of your process |
| the provider is down when a request starts | retried `max_retries` times with back-off, then `NotReadyError` | as above |
| the network path hangs (packets dropped, no reset) | `APITimeoutError` when the request's `timeout` (or the client's) runs out | the next request reconnects |
| the provider refuses the session's credentials (its token store was lost, the certificate mapping is gone) | that request fails with `AuthenticationError` / `PermissionDeniedError` | the next request reconnects and enrolls again with your key (at most once a minute); no action needed |
| your key is revoked, also mid-answer | `AuthenticationError` (not retryable): the provider closes the key's connections | none: use a valid key |

## Certificates and clocks

* The mTLS client certificate the provider issues (30 days by default) is **renewed at 2/3 of its lifetime**
  while the client runs and loaded into the live connections; other processes sharing the cache pick the renewed
  pair up within 10 minutes. A renewal that fails is retried with back-off; requests keep working while the current
  certificate is valid.
* A certificate that has **expired** (a cache left idle, a machine that slept) is replaced by enrolling again:
  at the next start, or while running (the models then reconnect).
* **Clock skew.** The provider checks certificates with its own clock, so a client clock that is minutes off does
  not matter. A clock off by more than a certificate's lifetime makes fresh certificates look invalid: the library
  logs `the clock is wrong (check it, e.g. 'date -u', and enable NTP time sync)` and falls back to token
  authentication where the provider allows it. Server certificate checks follow your clock (standard TLS).

## The background mask factory

| what happens | what you see | recovery |
|---|---|---|
| the factory process crashes | nothing while masks are in stock | restarted with exponential back-off (3 s doubling to 5 min); after 3 failures in a row, a request that waits for masks fails at once with `NotReadyError` naming the factory's error |
| it is killed by the out-of-memory killer (SIGKILL) | as above; `pgw sdk status` shows `killed by SIGKILL (often the out-of-memory killer: lower mask_workers ...)` | restarted; its worker processes exit with it (never orphans) |
| it hangs (stopped, deadlocked) | as above | no heartbeat for 5 minutes (`PGW_SDK_FACTORY_HANG_S`): killed and restarted |
| your process dies, however it dies (SIGKILL included) | - | the factory and its workers stop within seconds (SIGTERM, then a hard exit after 20 s) |
| the factory halts after provider integrity failures | `NotReadyError` naming the halt once the stock is used up | operator action (`pgw factory clear-halt` after investigating), never automatic |
| a request needs more masks than are in stock | it waits up to `mask_wait_s` (default 120 s) while the factory refills now (the factory normally refills only below its low-water mark) | automatic |
| a request needs more masks than the stock ever holds (prompt plus `max_tokens` above `mask_target_slots`, set by `mask_store_gb`) | `BadRequestError` (`payload_too_large`, 400) at once, naming the setting | lower `max_tokens`, shorten the prompt or raise `mask_store_gb` |

Several processes using the same key and cache share one factory: if its owner exits, another process takes over
within a few seconds.

## Disk

* The factory never fills the disk: it stops producing when less than 2 GiB would remain free on the cache volume
  (`PGW_FACTORY_MIN_FREE_BYTES`), and the mask store keeps 1 GiB (`PGW_MASKSTORE_MIN_FREE_BYTES`). A request that
  then waits for masks fails with `NotReadyError` saying the disk is almost full and where.
* A full disk anywhere else (enrollment, provisioning, the mask store's database) gives an `InternalServerError`
  saying the disk holding the cache is full. Nothing half-written is used: provisioning resumes from its verified
  partial download (`pack.partial`), a half-written session file is detected and enrolled again. Once space is free
  the same client recovers (no restart needed).

## Damaged or partial cache

| file | what happens |
|---|---|
| session files (`enrollment.json`, the service token, `client.crt`, `client.key`, `service-ca.pem`) | checked when loaded (the CA against your key's pin, the certificate against its key); a damaged session is enrolled again |
| a cached model pack (default provisioner) | verified byte for byte against the pinned fingerprint at every start; a damaged copy is deleted and downloaded again once (checked the same way); a pack you supplied yourself (`LocalPackProvisioner`) is never touched |
| an interrupted download | resumed; bytes that failed verification are never resumed from |
| a mask segment | authenticated encryption: a damaged segment is never used as a mask |
| `status.json`, `factory-state.json` | rebuilt (the factory keeps a corrupt state file aside) |
| the mask store's database, key (`masks.kek`) or rollback counter (`masks.counter`) | **fatal, never repaired automatically**: the counter is what stops an old copy of the store from serving used masks again. The error names what to delete to start over (the stored masks are discarded) |

## Timeouts and cancellation

* `create(timeout=...)` bounds the whole request, waiting for masks and a model that is still starting included.
  The one-time setup of a model is not cancelled by a short timeout: the next request continues it.
* A stream stops generating (and burns its unused masks) when you `close()` it, leave its `with` block, stop
  iterating and drop it, or cancel the task that iterates it. A web framework that cancels its handler when the
  HTTP client disconnects therefore stops the generation too.
* Cancelling an `await client.chat.completions.create(...)` cancels the request.

## Shutdown

* `close()` (or the end of a `with` block, or interpreter exit through `atexit`) runs the dummy passes still due
  (strict speculation), then stops requests still running: their callers get `APIConnectionError` ("the request was
  stopped because the pgw client was closed") instead of waiting. It returns within about its timeout (30 s; 10 s
  at interpreter exit) and stops the factory, its workers and every thread of the library.
* Signals: Python ends a process on SIGTERM without running `atexit` unless the application handles SIGTERM (web
  servers such as uvicorn and gunicorn do, then exit normally). Without that, the factory still exits with the
  process, unused masks of running requests are never reused, but dummy passes still due (strict speculation only)
  are not run. Handle SIGTERM (`signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))`) or call `close()` to get
  them.

## Soak (2026-11-26, `benchmarks/sdk_soak.py`)

31 minutes on a 4-vCPU box: tiny model, default mask scheme (pks-v3, masked head off), real service on loopback
behind the fault-injection proxy, every connection cut every 5 minutes. Per round: 4 sequential requests (greedy,
one in three streamed), 1 seeded sampled request, 4 concurrent requests from threads, 1 stream dropped after 3
chunks, 1 `AsyncOpenAI` opened and closed with one request and one stream whose consumer task is cancelled.

* 517 rounds: 5,170 answers, 517 dropped and 517 cancelled streams, 517 async clients, 6 faults; **0 errors, 0
  answers differing from the first one**.
* Client process: RSS 422 MB at 10 min, 406 / 412 / 414 / 420 MB at 15 / 20 / 25 / 31 min (the first minute
  includes warm-up at 547 MB); open file descriptors 26-48 with no trend (15 after `close()`); OS threads 45-47.
  The slow RSS drift is the masked backend's soundness caches, which are capped (65,536 entries each).
* Mask factory process: RSS 387-490 MB, 25-26 file descriptors, 5-7 threads, never restarted.
* Mask store on disk: segments 2-6 MB (bounded by the stock target); its state database grows by about 2 KB per
  request (closed reservation records, kept 30 days: `PGW_MASKSTORE_AUDIT_RETENTION_DAYS`).
* `close()` 0.44 s; no child process left afterwards.
