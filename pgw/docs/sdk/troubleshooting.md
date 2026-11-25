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
weights and about 20-30 minutes of CPU for the mask setup. `pgw sdk status` shows where it is. Later starts reuse
the cache. If every start is slow, the cache directory is not persistent (for example a container without a
volume); see [Configuration](configuration.md#cache-directory).

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
vCPUs, or lower `max_tokens` for bulk jobs.

## Errors during requests

| error code | meaning | what to do |
|---|---|---|
| `rate_limited` (429) | your key's rate or quota limit | wait for the `Retry-After` time |
| `unavailable`, `backend_unavailable` (503) | service busy or restarting | retry with exponential back-off |
| `deadline_exceeded` (504) | a request timed out | retry with back-off |
| `mask_exhausted` (503) | masks not ready for `prompt + max_tokens` | retry shortly, or lower `max_tokens` |
| `bad_shape` (400) | prompt too long for the model context | shorten the prompt |
| `integrity_check_failed` (502) | a result from our GPU failed verification | **do not retry automatically**; report it |
| `mask_reuse` (500) | the library stopped an attempted mask reuse | **do not retry**; report it at once |

## Reporting an issue

Send us:

* `pgw doctor --json` output and `pip freeze` output;
* the time (with time zone), the model id, the error code and message, and the request id if one is shown;
* what you expected and what happened.

Never send your SDK key, the cache directory, prompts or answers. If a prompt is needed to reproduce a
problem, write a synthetic one that shows it.
