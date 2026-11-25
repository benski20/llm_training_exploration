# Configuration reference

The client library works with no configuration beyond the key. This page lists what you can change.

## Client options

`pgw.OpenAI(...)` and `pgw.AsyncOpenAI(...)` accept the following options. Anything not listed keeps the
behaviour of the OpenAI client it replaces.

| option | default | meaning |
|---|---|---|
| `api_key` | `PGW_API_KEY` from the environment | your SDK key (`pgw_<environment>_...`). It carries the provider address, the pinned certificate authority of our service and a revocable credential |
| `cache_dir` | a per-user cache directory | where enrollment credentials and model data are kept ([below](#cache-directory)) |
| `mask_scheme` | `"pks-v3"` | how one-time masks are generated: `"pks-v3"`, `"mlwe"` or `"plain"` ([Concepts](concepts.md#mask-schemes)) |
| `hint` | prepared with our help, verified in your process | `"local"` builds the one-time mask setup entirely on your machine (slower first start, no data from us) |

There is no `base_url`: the provider address comes from the key. Use one client per process and share it
between threads or tasks; it is safe for concurrent use and batches concurrent requests.

## Request parameters (`chat.completions.create`)

| parameter | supported values | notes |
|---|---|---|
| `model` | `qwen2.5-0.5b-instruct`, `qwen2.5-1.5b-instruct`, `qwen2.5-3b-instruct`, `qwen2.5-7b-instruct` | see [Supported models](models.md) |
| `messages` | roles `system`, `developer`, `user`, `assistant`; text content (a string or a list of text parts) | images and other part types are rejected |
| `max_tokens` / `max_completion_tokens` | 1 to 8192 | clamped to the context left after the prompt |
| `temperature` | 0 to 2 (default 1.0) | 0 is greedy decoding and is reproducible |
| `top_p` | (0, 1] (default 1.0) | |
| `stop` | a string or up to 16 strings | |
| `seed` | integer | reproducible sampling |
| `presence_penalty`, `frequency_penalty` | -2 to 2 (default 0) | |
| `stream` | `true` or `false` | streaming yields OpenAI-style chunks |
| `stream_options` | `{"include_usage": true}` | a final chunk with token counts |
| `logprobs`, `top_logprobs` | `true`/`false`, 0 to 20 | |
| `n` | 1 | values above 1 are rejected |
| `user` | string | accepted and ignored |

Not supported in this release: tool or function calling, structured output (`response_format` with a JSON
schema), `n` above 1, and image or audio inputs. Such requests fail with a clear error instead of being
silently ignored.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `PGW_API_KEY` | none | your SDK key, used when `api_key` is not passed |
| `PGW_NATIVE` | `1` | `0` disables all native extensions (pure-Python path: same results, much slower). For diagnosis only |
| `PGW_NTT` | `1` | `0` disables only the native NTT products of mask generation |
| `PGW_PKS_NATIVE` | `1` | `0` disables only the native `pks-v3` kernels |
| `PGW_REQUIRE_NATIVE` | `0` | build time only: `1` makes a source build fail when the native extensions cannot be compiled |

## Cache directory

The cache directory holds:

* **Enrollment credentials**: your client certificate and private key, created on your machine, and the
  service token. Files are created with owner-only permissions (0600). Treat the directory like an SSH key
  directory: do not share it, do not bake it into container images, do not copy it between machines.
* **Model data**: tokenizer, chat template, the small plaintext parts of the model and the model fingerprint.
  All of it is public and is verified against the model fingerprint.
* **Mask setup**: the one-time material for mask generation (about 2 GB at 7B with `pks-v3`).

Plan the disk per model with [Supported models](models.md#resources-per-model). Deleting the directory is
safe: the next start enrolls again and prepares the model again. In containers, mount the cache directory
on a persistent volume so that a restart does not repeat the first-start work.

## Logging

The library logs through Python's `logging` module under the `pgw` logger name. It does not log prompts,
answers, keys or credentials by design. Configure it like any other library logger:

```python
import logging
logging.getLogger("pgw").setLevel(logging.INFO)
```

## Command line

The `pgw` command installed with the package is for diagnosis:

```bash
pgw version          # library and protocol version
pgw doctor           # environment, native extensions, exact arithmetic self-test
pgw doctor --json    # the same as JSON, for support requests
```
