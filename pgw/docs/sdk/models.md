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

| model | vCPUs (estimate) | memory (estimate) | free disk for the cache (estimate) | first-use download |
|---|---|---|---|---|
| `qwen2.5-0.5b-instruct` | 4 | 8 GB | 10 GB | about 1 GB |
| `qwen2.5-1.5b-instruct` | 4-8 | 8 GB | 13 GB | about 3 GB |
| `qwen2.5-3b-instruct` | 8 | 16 GB | 17 GB | about 6 GB |
| `qwen2.5-7b-instruct` | 16 | 32 GB | 26 GB | about 11 GB (6.5 GB of it the int8 weights) |

The disk figures include the mask stock budget (`mask_store_gb`, 8 GB per model by default), the public int8
weights kept for background mask generation, and the one-time mask setup (about 2.4 GB at 7B). The mask setup is
built on your machine on first use (`hint="local"`): about 20-30 minutes of CPU at 7B on 4 cores, once.

Fingerprints: the library pins every model above. 0.5B and 1.5B are pinned by their pack fingerprint; 3B and 7B
by the public checkpoint's commit, and on first use the library re-derives the model from that public checkpoint
to check our service's fingerprint (a one-time download of the checkpoint from Hugging Face: about 6 GB at 3B,
15 GB at 7B, in place of the int8 weights from us), until a later release bakes their fingerprints in too.

More vCPUs raise tokens per second until the round trips to our GPU dominate. For reference, in our lab
(measured, 2026-11-23) a CPU-only gateway with 48 cores placed next to an H100 decoded Qwen2.5-7B at about
28 tokens per second over back-to-back chats (22 tokens per second for a single warm chat), with mask
generation keeping up. A small pilot machine will be slower.

## Planned

Larger Qwen2.5 models (14B, 32B) and other open-weight families are planned. Tell us which models matter
to you.
