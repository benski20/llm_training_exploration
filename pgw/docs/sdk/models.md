# Supported models

Release 0.1 serves the Qwen2.5 Instruct family, an open-weight model family published by Alibaba Cloud
under the Apache 2.0 license (the 3B model has its own Qwen research license; check it against your use).

| model id (`model=`) | upstream checkpoint | parameters | layers | round trips per token |
|---|---|---|---|---|
| `qwen2.5-0.5b-instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5 B | 24 | 97 |
| `qwen2.5-1.5b-instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | 28 | 113 |
| `qwen2.5-3b-instruct` | `Qwen/Qwen2.5-3B-Instruct` | 3 B | 36 | 145 |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | 7 B | 28 | 113 |

Round trips per token are four per layer plus one for the masked LM head (when the service serves it, as ours
does); strict speculation (`speculation=`) can yield several tokens per pass of these trips.

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
built on your machine on first use (`hint="local"`): at 7B about 10-16 CPU-minutes with AMX or AVX-512 VNNI
(1.5-2 CPU-hours on AVX2-only CPUs), once; `pgw sdk prepare` builds it ahead of time
([Configuration](configuration.md#preparing-a-model-ahead-of-time)).

Fingerprints: the library pins every model above by its pack fingerprint (the int8 pack of the public checkpoint
at a fixed commit), so a service serving anything else is refused before any download:

| model id | pinned checkpoint commit | pack fingerprint |
|---|---|---|
| `qwen2.5-0.5b-instruct` | `7ae55760...` | `d773a502fb976d45070332cbd4222ee30bc971d14c7f9c56a317805c230c0b44` |
| `qwen2.5-1.5b-instruct` | `989aa798...` | `3d198de49e61c32b43150c237c0fad0f1ec062f0a355df75d64706bbac8d70d9` |
| `qwen2.5-3b-instruct` | `aa8e7253...` | `0ec0b8b24b2bc59632d9530e1bfd749b082c101f82c5ff819f8e8c5ff4d8d752` |
| `qwen2.5-7b-instruct` | `a09a3545...` | `f7c6fb4f5f8fbc5e4149972c07afdc749de8e5f9adc604d30edb7eb94b834b67` |

You can re-derive any of them from the public checkpoint with `scripts/pin_model_fingerprints.py`
(`pgw pack <repo> --revision <commit> --name <model>` with the default parameters).

The larger Qwen2.5 Instruct models are pinned the same way ahead of their support (they are not yet accepted as
`model=`; see [Planned](#planned)), so their first use will check the service against a baked fingerprint and never
needs the checkpoint:

| model id | pinned checkpoint commit | pack fingerprint |
|---|---|---|
| `qwen2.5-14b-instruct` | `cf98f3b3...` | `d4afbe77ff07ee3e0b9fc2f1c2074930e2f8acfd358b73d3e4cdc5329c34c0b4` |
| `qwen2.5-32b-instruct` | `5ede1c97...` | `a4cf6ba9ac2b654fdd8a14b850a792772d7f3f918d671b0b6a124b01020dd372` |
| `qwen2.5-72b-instruct` | `495f3936...` | `5e8001c02d7903a66f56d01f99890d56bff3bc5373eeaa451b80ceb0b98ab02f` |

These were derived with the shard-wise packer (one checkpoint shard of about 4 GB on disk at a time, hash only),
which produces exactly the same pack as `pgw pack`: `scripts/pin_model_fingerprints.py <model>` needs about 4.5 GB of
scratch disk and a few GB of memory even at 72B (`--write-pack` also writes the pack, 16 / 34 / 75 GB).
A model pinned without a fingerprint is checked on first use by packing the public checkpoint on your machine the
same way, shard by shard: about one shard plus the int8 pack in disk, never the whole bfloat16 checkpoint.

More vCPUs raise tokens per second until the round trips to our GPU dominate. For reference, in our lab
(measured, 2026-11-23) a CPU-only gateway with 48 cores placed next to an H100 decoded Qwen2.5-7B at about
28 tokens per second over back-to-back chats (22 tokens per second for a single warm chat), with mask
generation keeping up. A small pilot machine will be slower.

## Planned

Larger Qwen2.5 models (14B, 32B, 72B; their fingerprints are already pinned, above) and other open-weight
families are planned. Tell us which models matter to you.

The engine and the packer support the Qwen3 dense architecture (`model_type` `qwen3`; Apache 2.0) since 2026-11-27,
ahead of any pin. Qwen3 gives more chat quality per layer than Qwen2.5, and every layer costs round trips: Qwen3-14B
(40 layers) scores above Qwen2.5-32B (64 layers) on Arena-Hard in the Qwen3 report, at 161 round trips per token
instead of 257. Qwen3 models are not accepted as `model=` until they are pinned
([research note](../../research/reviews/consumer-model-depth.md)).

For measurements only, a pinned model that is not supported yet can be enabled with
`PGW_SDK_PREVIEW_MODELS=qwen2.5-32b-instruct` (a comma list; off by default). This is a **preview, not support**:
sizing, defaults and these docs are not checked for those models. It relaxes no check: the provider's pack must
match the baked pin above, `model_fingerprints` cannot replace a preview model's pin, and a model without a baked
pin is refused even when listed.
