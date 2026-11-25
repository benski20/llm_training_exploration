# Changelog

All notable changes to the PGW client library (`pgw-sdk`). Versions follow semantic versioning; while the
major version is 0, minor versions may change interfaces, and such changes are listed here.

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
* `pgw doctor` reports which native extensions are loaded.
* Distribution name `pgw-sdk` (import name `pgw`); proprietary license; type information included (`py.typed`).

Known limitations: no tool calling, structured output, `n > 1` or image inputs; request metadata (model,
token counts, timing) is visible to the service; no external security audit has been completed yet.
