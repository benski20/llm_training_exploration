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

Fingerprints: the library pins every model above by its pack fingerprint (the int8 pack of the public checkpoint
at a fixed commit), so a service serving anything else is refused before any download:

| model id | pinned checkpoint commit | pack fingerprint |
|---|---|---|
| `qwen2.5-0.5b-instruct` | `7ae55760...` | `915e55c19364542c4fd8ab5c8f7154806e2511048755e0d10426374b5c8fe9c7` |
| `qwen2.5-1.5b-instruct` | `989aa798...` | `81bb7adcea1524171297c0004c0efb1edad9f71dd0535462c9b32ab03ac99348` |
| `qwen2.5-3b-instruct` | `aa8e7253...` | `a8e7bac934d82f6435fe9385bd1eb3cfdf1a68f4ff415829e49a3dd4f23bcd23` |
| `qwen2.5-7b-instruct` | `a09a3545...` | `d3db0ca858981a66dd262143d7f3d5b0120069068e0a98a806e1eea3f12653fa` |

You can re-derive any of them from the public checkpoint with `scripts/pin_model_fingerprints.py`
(`pgw pack <repo> --revision <commit> --name <model>` with the default parameters).

More vCPUs raise tokens per second until the round trips to our GPU dominate. For reference, in our lab
(measured, 2026-11-23) a CPU-only gateway with 48 cores placed next to an H100 decoded Qwen2.5-7B at about
28 tokens per second over back-to-back chats (22 tokens per second for a single warm chat), with mask
generation keeping up. A small pilot machine will be slower.

## Planned

Larger Qwen2.5 models (14B, 32B) and other open-weight families are planned. Tell us which models matter
to you.
