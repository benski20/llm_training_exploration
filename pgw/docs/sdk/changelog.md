# Changelog

All notable changes to the PGW client library (`pgw-sdk`). Versions follow semantic versioning; while the
major version is 0, minor versions may change interfaces, and such changes are listed here.

## Unreleased

* Worker processes with the same key and cache directory now share one admission ledger for the mask stock, so a
  request admitted in one worker never takes the masks a running answer of another worker needs.
* Requests waiting for masks are served in arrival order; a request that can never fit the stock fails at once.
* `mask_admission="overcommit"` (opt-in, `PGW_MASK_ADMISSION`).
* New page: [Scaling](scaling.md) (sizing per model and concurrency, worker processes, fairness).

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
