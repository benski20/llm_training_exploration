# Configuration reference

The client library works with no configuration beyond the key. This page lists what you can change.

## Client options

`pgw.OpenAI(...)` and `pgw.AsyncOpenAI(...)` accept the following options. Anything not listed keeps the
behaviour of the OpenAI client it replaces.

| option | default | meaning |
|---|---|---|
| `api_key` | `PGW_API_KEY` from the environment | your SDK key (`pgw_<environment>_...`). It carries the provider address, the pinned certificate authority of our service and a revocable credential |
| `timeout` | 600 s | budget of one request in seconds (queueing and the wait for masks included). `create(timeout=...)` sets it for one request; the first request of a model additionally waits up to `ready_timeout_s` for the one-time preparation unless you pass `timeout` to `create` |
| `max_retries` | 2 | retries after a transport failure, only before the first token |
| `cache_dir` | `PGW_CACHE_DIR`, else `~/.cache/pgw` | where enrollment credentials, model data and the mask stock are kept ([below](#cache-directory)) |
| `state_dir` | `PGW_STATE_DIR`, else `~/.local/state/pgw` | the mask store's key and rollback counter (keep it next to the cache, on a persistent volume) |
| `mask_scheme` | `"pks-v3"` | how one-time masks are generated: `"pks-v3"`, `"mlwe"` or `"plain"` ([Concepts](concepts.md#mask-schemes)) |
| `hint` | `"local"` | how the one-time `pks-v3` setup is built: `"local"` builds it on your machine from the public weights (at 7B about 10-16 CPU-minutes with AMX or AVX-512 VNNI, 1.5-2 CPU-hours with AVX2 only; once, then cached; `pgw sdk prepare` does it ahead of time, [below](#preparing-a-model-ahead-of-time)); `"provider"` lets our GPU compute the expensive part for seeds your process chose, and your process accepts it only after a check (faster on few cores, but it downloads about 44 GB at 7B) |
| `mask_workers` | a quarter of the CPUs (1 to 8) | background mask-generation processes per model (low priority) |
| `mask_store_gb` | 8 | disk budget of the mask stock per model |
| `masked_head` | `"auto"` | the model's output layer as one more masked step on our GPU when we serve it (`"on"`, `"off"`) |
| `prefix_cache` | `"auto"` | exact reuse of the previous turn of a conversation (`extra_body={"conversation_id": ...}`) when the output layer runs in your process |
| `lane` | `"auto"` | our fast connection for the per-token round trips when the service offers it (`"stream"`, `"off"`) |
| `model_fingerprints` | `{}` | extra or replacement fingerprint pins `{model: fingerprint}`. The library pins every supported model itself and refuses a service that serves anything else; set this only for a custom model build you checked yourself |
| `usage_reports` | `True` | after each request, send the service a record without content (model, token counts, latency, status, whether every check passed) for your organisation's console. The service sees the same counts and timing in the masked traffic anyway. Requests that fail before reaching the service (for example no masks yet, or a prompt beyond the context) are not reported; they are counted in `client.status()["local_errors"]` only |
| `speculation` | `PGW_SPECULATION`, else `"off"` | strict speculative decoding (opt-in): each round of round trips verifies `spec_rows` rows, so one round can yield up to `spec_rows` tokens. Answers are bit-identical; the service sees the same number of rounds and rows whatever the drafts (after the answer, your process runs the remaining rounds on random data, and the next request of the same `user` waits for them). `"lookup"` drafts from the prompt and the conversation (no model; helps when the answer copies text: code edits, rewrites, summaries, retrieval; about 2x tokens per round on a code edit, little on open chat), `"draft_head"` uses a trained draft-head file (`draft_head_path`). Costs `spec_rows` masks per token and `spec_rows` times the per-round traffic |
| `spec_rows` | 3 | rows per round with `speculation` (2 to 8) |
| `draft_head_path` | none | the draft-head file for `speculation="draft_head"` |
| `spec_pass_gap_ms` | 8 | with `speculation`: the least time between the end of one round and the start of the next (0 = off). It hides from the service the small timing differences between the answer's rounds and the random rounds after it; it costs a few ms per round |
| `log_level` | `PGW_SDK_LOG`, else silent | structured (JSON) logs on stderr at this level |

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
| `user` | string | accepted and ignored (with `speculation`, requests of one `user` are ordered behind the previous one's remaining rounds) |

Not supported in this release: tool or function calling, structured output (`response_format` with a JSON
schema), `n` above 1, and image or audio inputs. Such requests fail with a clear error instead of being
silently ignored.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `PGW_API_KEY` | none | your SDK key, used when `api_key` is not passed |
| `PGW_CACHE_DIR` | `~/.cache/pgw` | cache directory (`cache_dir`) |
| `PGW_STATE_DIR` | `~/.local/state/pgw` | state directory (`state_dir`) |
| `PGW_HINT` | `local` | `hint` |
| `PGW_MASK_SCHEME` | `pks-v3` | `mask_scheme` |
| `PGW_MASK_WORKERS` | a quarter of the CPUs | `mask_workers` |
| `PGW_SPECULATION` | `off` | `speculation` |
| `PGW_MASK_STORE_GB` | `8` | `mask_store_gb` |
| `PGW_SDK_LOG` | unset | `log_level` |
| `PGW_NATIVE` | `1` | `0` disables all native extensions (pure-Python path: same results, much slower). For diagnosis only |
| `PGW_NTT` | `1` | `0` disables only the native NTT products of mask generation |
| `PGW_PKS_NATIVE` | `1` | `0` disables only the native `pks-v3` kernels |
| `PGW_REQUIRE_NATIVE` | `0` | build time only: `1` makes a source build fail when the native extensions cannot be compiled |

## Cache directory

The cache directory holds:

* **Enrollment credentials**: your client certificate and private key, created on your machine, and the
  service token. Files are created with owner-only permissions (0600). Treat the directory like an SSH key
  directory: do not share it, do not bake it into container images, do not copy it between machines.
* **Model data for your process**: tokenizer, chat template and the small plaintext parts of the model
  (embeddings, norms, per-row scales). No weights of the model's layers: those stay on our GPU. All of it is
  public and is checked against the model fingerprint the library pins (about 0.6 GB at 0.5B, 4.4 GB at 7B).
* **The public int8 weights, for mask generation only**: the background mask processes need them for their
  private check values and to build the one-time setup; they are downloaded once from us and checked against
  the same fingerprint (about 6.5 GB at 7B). The process that handles your requests never loads them.
* **Mask setup**: the one-time material for mask generation (about 2.4 GB at 7B with `pks-v3`).
* **Mask stock**: masks prepared ahead of use, up to `mask_store_gb` (8 GB) per model.

First use of a model, once per machine: enrollment (seconds), the model data and the weights (about 11 GB at
7B, mostly the 6.5 GB of int8 weights), then the mask setup (with the default `hint="local"`, at 7B about 10 CPU-minutes
on a CPU with AMX and 16 with AVX-512 VNNI, about 5 minutes on a 4-vCPU AMX machine; on CPUs with AVX2 only about
1.5-2 CPU-hours). Later starts reuse everything and take seconds.

### Preparing a model ahead of time

To keep the first request from waiting for this, prepare the model beforehand, for example while the
application is being deployed:

```bash
export PGW_API_KEY=pgw_live_...
pgw sdk prepare --model qwen2.5-7b-instruct      # add --cache-dir / --state-dir if your client sets them
```

It downloads the model data and the weights, checks them against the pinned fingerprint and builds the mask
setup in the same pass, on every core, with progress lines. If it is interrupted, run it again: it continues
where it stopped (what was already built is re-checked, never trusted blindly). A client started later with
the same cache directory uses the prepared setup at once. Use the same `masked_head` setting as your clients
(`--masked-head`, default `auto`).

**Set up on a big machine once, then copy.** The mask setup depends only on public data and on values your
machine draws, so you can prepare a model on a large machine you control and copy the result to your
application hosts: copy `<cache>/models/<directory of the model>/` and `<cache>/keys/pks-hint.key` to the same
places under the target's cache directory (nothing else: the enrollment credentials in the cache stay on their
machine). The key authenticates the setup's metadata (HMAC-SHA256 over its seeds, its noise bounds and the
SHA-256 of every file); the target uses the copied setup only if the metadata verifies under that key, and
re-hashes every file before use. Copy the key over a channel you trust and keep it readable by the client's
user only. Without the key the copied setup is ignored and rebuilt.

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
pgw version                # library and protocol version
pgw doctor                 # environment, native extensions, exact arithmetic self-test
pgw doctor --json          # the same as JSON, for support requests
pgw sdk status             # enrolled keys, prepared models, mask stock, last integrity check (no network)
pgw sdk doctor             # the key's format, the cache and state directories, the mask workers
pgw sdk doctor --online    # also contact the service and check its certificate authority against the key
```

None of them prints your key or your service token.
