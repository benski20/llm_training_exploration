# Supported models

Release 0.1 serves the Qwen2.5 Instruct family, an open-weight model family published by Alibaba Cloud
under the Apache 2.0 license (the 3B model has its own Qwen research license; check it against your use).

| model id (`model=`) | upstream checkpoint | parameters | layers | round trips per token |
|---|---|---|---|---|
| `qwen2.5-0.5b-instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5 B | 24 | 96 |
| `qwen2.5-1.5b-instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | 28 | 112 |
| `qwen2.5-3b-instruct` | `Qwen/Qwen2.5-3B-Instruct` | 3 B | 36 | 144 |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | 7 B | 28 | 112 |

Your key lists the models it may use; a model outside that list fails with `model_not_found`.

## What "the same answer" means

The service runs each model with int8 weights and a fixed activation quantization (the pinned reference).
PGW's output is bit-identical to that reference run in plaintext: the same tokens and the same
log-probabilities. It is not bit-identical to the original bfloat16 checkpoint; quantization changes
perplexity only slightly (measured on Qwen2.5-0.5B: 10.93 against 10.92 with float activations; see the
[security whitepaper](../security/whitepaper.md#31-notation)).

## Context and limits

* Prompt plus completion must fit the model context; `max_tokens` is clamped to the context left.
* `max_tokens` above 8192 is rejected.
* One completion per request (`n = 1`).
* Text only. No tool calling or structured output in this release.

## Resources per model

Your process needs CPU for the parts of the model that run locally and for background mask generation,
memory for the model's small plaintext parts and working state, and disk for the cache. The figures
below are planning estimates for a pilot, not measurements; the pilot will confirm them.

| model | vCPUs (estimate) | memory (estimate) | free disk for the cache (estimate) |
|---|---|---|---|
| `qwen2.5-0.5b-instruct` | 4 | 8 GB | 5 GB |
| `qwen2.5-1.5b-instruct` | 4-8 | 8 GB | 8 GB |
| `qwen2.5-3b-instruct` | 8 | 16 GB | 10 GB |
| `qwen2.5-7b-instruct` | 16 | 32 GB | 20 GB |

More vCPUs raise tokens per second until the round trips to our GPU dominate. For reference, in our lab
(measured, 2026-11-23) a CPU-only gateway with 48 cores placed next to an H100 decoded Qwen2.5-7B at about
28 tokens per second over back-to-back chats (22 tokens per second for a single warm chat), with mask
generation keeping up. A small pilot machine will be slower.

## Planned

Larger Qwen2.5 models (14B, 32B) and other open-weight families are planned. Tell us which models matter
to you.
