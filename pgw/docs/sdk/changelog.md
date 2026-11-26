# Changelog

All notable changes to the PGW client library (`pgw-sdk`). Versions follow semantic versioning; while the
major version is 0, minor versions may change interfaces, and such changes are listed here.

## Unreleased: prefix cache

* `prefix_cache=True` now works with the masked output layer (it used to turn the masked layer off) and keeps
  several cached prompts per `user`: the longest exact token prefix wins, so a system prompt shared by several
  conversations, a regenerated answer or an edited message reuse what they share. Answers and logprobs stay
  bit-identical. Opt-in with the masked output layer (`"auto"` is unchanged); see
  [Configuration](configuration.md#prefix-cache) for what the service can then infer.
* New options `prefix_cache_mb` (1024), `prefix_cache_ttl_s` (600) and `prefix_cache_scope` (`"user"`, or
  `"client"` to share across users). `client.status()` shows the cache's counters.

## Unreleased

* Worker processes with the same key and cache directory now share one admission ledger for the mask stock, so a
  request admitted in one worker never takes the masks a running answer of another worker needs.
* Requests waiting for masks are served in arrival order; a request that can never fit the stock fails at once.
* `mask_admission="overcommit"` (opt-in, `PGW_MASK_ADMISSION`).
* New page: [Scaling](scaling.md) (sizing per model and concurrency, worker processes, fairness).

## Unreleased: factory stalls and first use

* A provider that dies or cannot be reached while the mask factory makes masks no longer leaves requests waiting
  out their budget (a first request waited hours in the Kaggle 7B run): once the factory has made no progress for
  `factory_stall_s` (new option, default 120 s, `PGW_FACTORY_STALL_S`), waiting requests fail with
  `APIConnectionError` naming the provider and the factory's last error. The factory keeps retrying and, when the
  provider answers again, resumes at once instead of after its back-off. A long first-time setup that makes
  progress is still waited for.
* `pgw sdk status`, `client.status()` and request errors say `first-time setup in progress` with the share of the
  one-time hint built and an estimate of the time left, the segment being made, and whether the provider answers.
* A request that needs more masks than are in stock waits for the segment the factory is making, up to its
  `timeout`, when that segment is due in time (was: `mask_exhausted` after `mask_wait_s`).
* The docs recommend `pgw sdk prepare` before the first traffic.

## Unreleased: reliability

What happens on each failure is now written down in [Reliability](reliability.md), with a test for each.

* `close()` stops requests still running (their callers get `APIConnectionError` instead of waiting forever); a
  stream that is dropped, or whose consuming task is cancelled (an SSE client that disconnected), stops generating.
* The client certificate is renewed while the client runs; an expired one, damaged session files and a damaged
  cached model pack are replaced by themselves; a refused session enrolls again on the next request.
* The mask factory's processes never outlive the application (also after SIGKILL); a hung factory is restarted;
  a request waiting for masks makes the factory refill at once; a request that can never fit the mask stock fails
  at once (`payload_too_large`, 400).
* A key revoked mid-answer is reported as `AuthenticationError` (was a retryable 503); a full disk is reported as
  such (was a connection error).

## 0.1.0 (2026-11-25): pilot release

First release of the client library for pilot customers.

* `from pgw import OpenAI, AsyncOpenAI`: drop-in replacements for the OpenAI chat-completions client, with
  streaming and async support. The gateway runs in your process on CPU; our GPU service sees only masked data.
* Models: Qwen2.5-Instruct 0.5B, 1.5B, 3B and 7B.
* SDK keys (`pgw_...`, also from `PGW_API_KEY`) with automatic enrollment, pinned service certificate
  authority and a local cache.
* Mask schemes: `pks-v3` (default, **pending external review**), `mlwe` and `plain`.
* Integrity check (Freivalds and range check) of every result; self-check of all provider-assisted mask material.
* Prebuilt wheels with native extensions (`pgw_fast`, `pgw_ntt`, `pgw_pks`) for Linux x86_64 and aarch64
  (manylinux 2.28) and macOS arm64, one wheel per platform for CPython 3.11 to 3.13 (stable ABI); a
  pure-Python fallback when no wheel matches.
* `pgw doctor` reports which native extensions are loaded; `pgw sdk status` and `pgw sdk doctor` show the
  enrolled keys, prepared models, mask stock and last integrity check (no secrets).
* Fingerprint pins for every supported model in the library; the one-time mask setup is built locally by
  default (`hint="local"`; `hint="provider"` is opt-in). Per-request `timeout` in `create()`.
* Content-free request reports (`usage_reports`, on by default) for the organisation's console.
* Distribution name `pgw-sdk` (import name `pgw`); proprietary license; type information included (`py.typed`).

Known limitations: no tool calling, structured output, `n > 1` or image inputs; request metadata (model,
token counts, timing) is visible to the service; no external security audit has been completed yet; the process
using the library must run in the same data center as the GPU service.
