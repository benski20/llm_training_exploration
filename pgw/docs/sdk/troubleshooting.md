# Troubleshooting

Start with `pgw doctor`. It checks Python, the installed packages, the native extensions and exact
arithmetic on this machine, and prints one line per check. Attach `pgw doctor --json` to any support
request; it contains no keys, prompts or answers.

## Installation

### pip downloads gigabytes of `nvidia-*` packages

You installed the default PyTorch build for Linux, which bundles CUDA. PGW needs only the CPU build.
Reinstall in a fresh virtual environment with the CPU index:

```bash
pip install pgw-sdk --extra-index-url https://download.pytorch.org/whl/cpu
```

If PyTorch is already installed with CUDA in a shared environment, PGW works with it; it only costs disk.

### pip tries to compile Rust, or fails with "can't find Rust compiler"

pip found no prebuilt wheel for your platform and fell back to the source distribution. Prebuilt wheels
exist for Linux x86_64 and aarch64 (glibc 2.28 or newer) and macOS on Apple silicon, for Python 3.11 to
3.13. Check:

* `python --version` is 3.11, 3.12 or 3.13, and is CPython;
* `pip --version` is recent (`pip install --upgrade pip`); old pip versions do not recognise current wheel tags;
* on Linux, `ldd --version` reports glibc 2.28 or newer (Alpine/musl is not supported).

Without a Rust toolchain, a source install still completes and runs on the pure-Python path (see the next
section). To build the native extensions from source, install Rust 1.89 or newer from rustup and reinstall.

### Native extensions not loaded

`pgw doctor` shows `[WARN] native ... pure-Python fallback`. Results are identical, but generation is much
slower. Causes, in order of likelihood:

* the package was installed from source without Rust (previous section);
* `PGW_NATIVE=0` is set in the environment;
* the wheel is for a different platform than the interpreter (for example an x86_64 Python under Rosetta on an
  Apple silicon Mac: use a native arm64 Python).

### `ImportError` or unexpected `pgw` module

An unrelated package called `pgw` exists on PyPI and uses the same import name. Remove it:
`pip uninstall pgw`, then `pip install --force-reinstall pgw-sdk` (with the CPU index on Linux).

## First start

### The first request takes minutes

Expected once per machine and model: enrollment, model preparation and mask setup run before the first
answer ([Quickstart](quickstart.md#4-first-call)): at 7B on 4 cores, a one-time download of about 6.5 GB of public
weights and about 20-30 minutes of CPU for the mask setup (on a 4-vCPU Kaggle VM shared with a T4 provider: a 17-minute
hint build, then about 3 minutes for the first 64 mask slots, 21 minutes in all). `pgw sdk status` shows where it is,
for example `first-time setup in progress: building the one-time pks-v3 hint, 45% done (about 9 min left)`. Later
starts reuse the cache. If every start is slow, the cache directory is not persistent (for example a container
without a volume); see [Configuration](configuration.md#cache-directory).

**Run `pgw sdk prepare --model <model>` before the first traffic** (at deploy time, or in the image build with the
cache directory on a volume): it does the downloads and the one-time setup on every core, ahead of time, so the
first request only waits for the first mask segment, if at all.

A request with a short `timeout` during that first preparation fails with a timeout but does not stop the
preparation; the next request continues where it is.

### `integrity_check_failed` before the first answer, mentioning a fingerprint

The service serves a model build other than the one this library pins (`FingerprintMismatchError`). Nothing was
used. Tell us; do not override the pin with `model_fingerprints` unless you have checked the build yourself.

### Enrollment fails with a certificate or TLS error

The service's certificate authority is pinned by your key. A TLS error usually means the key belongs to a
different environment than the service you reach, or a proxy intercepts TLS. PGW does not accept an
intercepting proxy's certificate by design: allow direct HTTPS to the provider address, or contact us.

### `unauthenticated` (401)

The key is mistyped, truncated, or revoked. Check that `PGW_API_KEY` holds the complete key, including the
`pgw_` prefix, without quotes or line breaks. Ask us for a new key if needed.

### `model_not_found` (404)

The model id is misspelled or not enabled for your key. Valid ids are in [Supported models](models.md).

## Speed

### Tokens arrive slowly

Per-token time is dominated by round trips to our GPU ([Concepts](concepts.md#why-your-application-should-run-near-our-gpu)).

1. Confirm your process runs in the same data center as our GPU (a requirement, not a tuning hint). A machine in another region adds seconds
   per token.
2. Give the process more vCPUs; check that nothing else saturates the CPU (`top`).
3. Reuse one client per process; concurrent requests on one client are batched.
4. Confirm the native extensions are loaded (`pgw doctor`).

### Generation pauses after a burst of traffic

The background mask generator fell behind and the library waits for masks (`mask_exhausted`, or a pause
before generation continues). It catches up at a rate set by the vCPUs available; give the process more
vCPUs, or lower `max_tokens` for bulk jobs. While the factory is visibly making the masks a request needs, the
request waits for them up to its `timeout` (not only `mask_wait_s`); a `mask_exhausted` error says when the
factory expects more masks.

### `APIConnectionError` while waiting for masks: "the mask factory has made no progress"

The mask factory could not reach our service (or its runs kept failing) for `factory_stall_s` (120 s). The message
names the provider's health-check result and the factory's last error. The factory keeps retrying and resumes by
itself when the service is back; retry the request later. `pgw sdk status` shows `STALLED` with the reason until
then.

### `ProviderStalledError` while waiting for masks: "its mask generation is stalled"

Our service answers its health check, but a mask generation item of your key has been running there for longer
than `maskgen_stall_s` (30 s, and at least 4x the slowest recent item) without finishing: the service's GPU is
wedged or its mask generation hangs. `ProviderStalledError` is an `APIConnectionError` (code
`backend_unavailable`, retryable; `details["reason"]` is `provider_maskgen_stalled`, with the item's age and the
bound). Nothing on your side needs to change: report it to us (the service has to be restarted), and retry later;
the factory resumes by itself once the service makes progress again. `pgw sdk status` shows `STALLED` with the
reason, and `supervisor.json` in the factory's directory has `"phase": "maskgen_stalled"`. Before 2026-11-26 this
was noticed only after the 600 s budget of the factory's request. A service so slow that one item legitimately
takes longer (not a GPU service) needs a larger `maskgen_stall_s`; `0` turns the watchdog off.

## Errors during requests

What the library does by itself on each failure (reconnects, re-enrollment, factory restarts, cache repair) is in
[Reliability](reliability.md).

| error code | meaning | what to do |
|---|---|---|
| `rate_limited` (429) | your key's rate or quota limit | wait for the `Retry-After` time |
| `unavailable`, `backend_unavailable` (503) | service busy or restarting, or the connection was lost mid-answer | retry with exponential back-off; the same client reconnects |
| `deadline_exceeded` (504) | a request timed out | retry with back-off |
| `mask_exhausted` (503) | masks not ready for `prompt + max_tokens`; the message says why (first-time setup still running, with its progress; factory failing; disk almost full; when the next masks are due) | retry shortly, or lower `max_tokens`; act on the reason given. For first use: `pgw sdk prepare` ahead of time |
| `payload_too_large` (400) | `prompt + max_tokens` needs more masks than the stock ever holds | lower `max_tokens` or shorten the prompt, or raise `mask_store_gb` |
| `bad_shape` (400) | prompt too long for the model context | shorten the prompt |
| `integrity_check_failed` (502) | a result from our GPU failed verification | **do not retry automatically**; report it. The model stays blocked, across restarts, until you clear it (below) |
| `mask_reuse` (500) | the library stopped an attempted mask reuse | **do not retry**; report it at once |

### A model stays blocked after `integrity_check_failed`

After any result of the provider fails verification, the library blocks that model for your API key: every
request raises `IntegrityError` (code `integrity_check_failed`) and nothing is sent. The block is stored in the
state directory (`PGW_STATE_DIR`, `sessions/<key id>/integrity/<model>.json`, mode 0600, protected by an HMAC
whose key is `sessions/<key id>/integrity.key`), so **restarting the application does not lift it**, and it
applies to every process that uses the same key and state directory. `pgw sdk status` shows it as
`HALTED (persisted)`. If that file was edited, or the key file next to it is missing, the library cannot verify
it and treats the model as blocked (fail closed); the message says so.

An honest provider never fails a check. Report it to us before you continue. Clear the block only after the
cause is understood (for example, after we confirm a faulty deployment was replaced):

```bash
pgw sdk clear-halt --model qwen2.5-7b-instruct --yes --note "provider incident 2026-11-30 resolved"
```

The command prints a warning, clears the block for every key under the state directory (`--key-id` limits it to
one key, `--state-dir` picks the directory) and keeps a record of the clear. Processes that are already running
keep the model blocked until they restart. From code, `client.reset_integrity_breaker(model)` does the same for
the client's key.

| `unauthenticated` (401), `permission_denied` (403) during requests | the key was revoked, or the service no longer accepts this installation's credentials | if the key is valid, the next request enrolls again by itself; else use a valid key |
| `integrity_check_failed` (502) | a result from our GPU failed verification | **do not retry automatically**; report it |
| `mask_reuse` (500) | the library stopped an attempted mask reuse | **do not retry**; report it at once |

### "the request was stopped because the pgw client was closed"

`close()` was called (or the interpreter exited) while the request ran in another thread or task. Close the client
only after its requests are done, or treat this as a cancellation.

### "the disk holding the pgw cache is full"

Free space on the volume of the cache directory (`pgw sdk status` shows it). The client recovers by itself once
there is room. The mask factory keeps 2 GiB free on that volume and stops producing below it; requests waiting for
masks then say so.

### "the mask store of this API key cannot be opened safely"

Its database, key or rollback counter is damaged, or was changed outside the library. The library never repairs
this by itself, because the rollback counter is what stops an old copy of the store from serving used masks again.
If you know the cause (a full disk, a crash, a partial copy of the cache), stop every process using the cache and
delete the two paths the message names; the stored masks are discarded, enrollment and models are kept.

### The mask factory "killed by SIGKILL"

Usually the out-of-memory killer. Give the process more memory or lower `mask_workers`; the factory is restarted by
itself in the meantime.

### "the clock is wrong"

This machine's clock is far off. Enable time synchronisation (NTP); check with `date -u`.

## Reporting an issue

Send us:

* `pgw doctor --json` output and `pip freeze` output;
* the time (with time zone), the model id, the error code and message, and the request id if one is shown;
* what you expected and what happened.

Never send your SDK key, the cache directory, prompts or answers. If a prompt is needed to reproduce a
problem, write a synthetic one that shows it.
