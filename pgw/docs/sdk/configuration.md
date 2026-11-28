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
| `factory_weights` | `PGW_FACTORY_WEIGHTS`, else `"pack"` | where the background mask processes get their check values from. `"pack"` keeps the public int8 weights in the cache (6.5 / 31 / 70 GB at 7B / 32B / 72B). `"stream"` (opt-in, `mask_scheme="pks-v3"`) keeps no int8 weights: whenever a small encrypted stock of check values runs low, the weights are streamed once more, checked against the pinned fingerprint and dropped. Trade: about 1-2 % more network and a stock of 0.6 / 1.9 / 3.1 GB instead of the weights ([below](#keeping-no-weights-factory_weightsstream)) |
| `factory_weights_source` | `PGW_FACTORY_WEIGHTS_SOURCE`, else `"provider"` | with `factory_weights="stream"`: stream the weights from our provisioning endpoint (`"provider"`) or pack the pinned public checkpoint from Hugging Face on each pass (`"hf"`; packed one checkpoint shard at a time, so it needs about one shard, 4 GB, plus the pack in temporary disk: 8.7 / 16 / 34 / 75 GB at 7B / 14B / 32B / 72B, not the bfloat16 checkpoint). Neither is trusted: every tensor is checked |
| `check_stock_segments` | `PGW_CHECK_STOCK_SEGMENTS`, else 64 | with `factory_weights="stream"`: segments of check values one weight pass prepares (one segment is 1024 mask slots); the next pass starts at a quarter |
| `mask_workers` | a quarter of the CPUs (1 to 8) | background mask-generation processes per model (low priority) |
| `mask_store_gb` | 8 | disk budget of the mask stock per model. A request starts only when the stock covers the prompt + `max_tokens` of every running request: at 7B the default holds about one such request at a time; size it for your concurrency ([Scaling](scaling.md)) |
| `mask_admission` | `PGW_MASK_ADMISSION`, else `"entitled"` | `"overcommit"` (opt-in): a request starts when the stock covers its prompt and one chunk per running answer; answers that find the stock short wait for the factory (up to 30 s) instead of new requests queueing ([Scaling](scaling.md#when-the-stock-runs-short-bursts)) |
| `mask_wait_s` | 120 s | how long a request waits for masks when the stock is short. Longer during a model's first fill on a machine while the setup makes progress (up to `ready_timeout_s`), and while the factory is visibly making the segment the request needs and expects it within the request's `timeout` |
| `factory_stall_s` | `PGW_FACTORY_STALL_S`, else 120 s | how long the background mask factory may make no progress because our service cannot be reached (or its runs keep failing) before requests waiting for masks fail with `APIConnectionError` naming the reason ([Reliability](reliability.md#the-background-mask-factory)); the factory keeps retrying and resumes by itself |
| `maskgen_stall_s` | `PGW_SDK_MASKGEN_STALL_S`, else 30 s; `0` turns it off | the maskgen watchdog: while the factory makes masks, how long one of its items may run on our service without finishing (at least 4x the slowest recent item; 300 s for the first item of a shape after the service starts) before the service's mask generation counts as stalled and requests waiting for masks fail with `ProviderStalledError` ([Reliability](reliability.md#the-background-mask-factory)). Your own CPU work (encryption, the one-time setup) never counts |
| `masked_head` | `"auto"` | the model's output layer as one more masked step on our GPU when we serve it (`"on"`, `"off"`) |
| `prefix_cache` | `"auto"` | exact reuse of prompt prefixes your process has already computed: the earlier turns of a conversation, a system prompt shared by several conversations, a regenerated answer. Cached tokens take no round-trip rows and no masks; answers and logprobs are bit-identical. `True`: on (with or without the masked output layer; opt-in, see [below](#prefix-cache)); `"auto"`: on only when the output layer runs in your process (`masked_head="off"` or not served); `False`: off |
| `prefix_cache_mb` | 1024 | memory budget of the cached KV states per model and process (MiB, RAM only; least recently used are wiped first) |
| `prefix_cache_ttl_s` | 600 | a cached KV state unused this long is wiped |
| `prefix_cache_scope` | `"user"` | which requests may reuse each other's prefixes: the same OpenAI `user` (`"user"`), or every request of this client (`"client"`, opt-in: a system prompt shared by all users is computed once) |
| `lane` | `"auto"` | our fast connection for the per-token round trips when the service offers it (`"stream"`, `"off"`) |
| `model_fingerprints` | `{}` | extra or replacement fingerprint pins `{model: fingerprint}`. The library pins every supported model itself and refuses a service that serves anything else; set this only for a custom model build you checked yourself |
| `usage_reports` | `True` | after each request, send the service a record without content (model, token counts, latency, status, whether every check passed) for your organisation's console. The service sees the same counts and timing in the masked traffic anyway. Requests that fail before reaching the service (for example no masks yet, or a prompt beyond the context) are not reported; they are counted in `client.status()["local_errors"]` only |
| `speculation` | `PGW_SPECULATION`, else `"off"` | strict speculative decoding (opt-in): each round of round trips verifies `spec_rows` rows, so one round can yield up to `spec_rows` tokens. Answers are bit-identical; the service sees the same number of rounds and rows whatever the drafts (after the answer, your process runs the remaining rounds on random data, and the next request of the same `user` waits for them). `"draft_head"` drafts with a small trained head that ships with the library for `qwen2.5-0.5b-instruct` and `qwen2.5-7b-instruct` (checked against its pinned SHA-256; it runs on your CPU only and is never sent anywhere) plus prompt lookup: about 1.5-1.9 tokens per round on open chat and code. `"lookup"` drafts from the prompt and the conversation only (no model; about 2x tokens per round on a code edit, little on open chat). Costs `spec_rows` masks per token and `spec_rows` times the per-round traffic. When to enable it: [below](#speculative-decoding-when-to-enable-it) |
| `spec_rows` | 3 | rows per round with `speculation` (2 to 8) |
| `draft_head_path` | the shipped head | your own draft-head file for `speculation="draft_head"` (`benchmarks/feature_head_distill.py` trains one) |
| `spec_pass_gap_ms` | 8 | with `speculation`: the least time between the end of one round and the start of the next (0 = off). It hides from the service the small timing differences between the answer's rounds and the random rounds after it; it costs a few ms per round |
| `schedule` | `PGW_SCHEDULE`, else `"standard"` | how the per-token round trips are grouped: `"standard"` (`4 x layers + 1` round trips per token), `"fused"` (opt-in: `2 x layers` round trips per token, values 1.75x larger; for high-latency links such as consumer devices on the internet) or `"auto"` (measures the link when the model starts and picks one). `"fused"` gives slightly different answers (a different fixed reference, not bit-identical to `"standard"`), so the choice is fixed per model for the life of the client: [below](#schedule-standard-or-fused) |
| `schedule_auto_rtt_ms` | 3 | `"auto"`: the least measured round-trip time (ms) at which `"fused"` may be picked |
| `schedule_auto_margin` | 0.1 | `"auto"`: `"fused"` must be estimated at least this much faster per token (fraction) |
| `routing` | `PGW_ROUTING`, else `"off"` | `"signed"` (opt-in): use the provider's signed endpoint directory (verified with the key pinned in the SDK key or at enrollment: signature, expiry, no rollback) and run each model on the nearest healthy endpoint serving its fingerprint; a model moves only between requests and fails over only before the first token; the mask stock is shared across endpoints of one fingerprint. `"off"`: the SDK key's provider URL only. [docs/sdk/wan.md](wan.md#region-routing), [placement](../operations/placement.md) |
| `routing_refresh_s` | 300 | `"signed"`: seconds between directory refreshes and probe rounds (a failure refreshes at once) |
| `log_level` | `PGW_SDK_LOG`, else silent | structured (JSON) logs on stderr at this level |

## Prefix cache

With `prefix_cache=True` the client keeps the plaintext KV state of finished requests in its own RAM (never on
disk, never sent anywhere) and a later request whose prompt starts with the same tokens runs only the rest of its
prompt. In a multi-turn chat every turn after the first then costs masks and round-trip rows only for the new
message and the answer, instead of the whole conversation; a shared system prompt is computed once per `user`
(or once per client with `prefix_cache_scope="client"`). `usage.prompt_tokens_details.cached_tokens` shows how
many prompt tokens came from the cache, and `client.status()["models"][model]["prefix_cache"]` the counters.

* **Exact.** Answers and logprobs are bit-identical to recomputing the prompt (greedy, seeded sampling,
  speculation), with the masked output layer and without.
* **What it costs in privacy.** Our service then sees a short first round of trips for a request that continued
  an earlier one (only the new tokens' rows), so it can tell *that* a request reused a cached prefix and roughly
  how long its context is, and link a conversation's turns more surely than from token counts alone. It never
  sees more content: every row it gets is still masked with a fresh single-use mask. This is why the cache is
  opt-in with the masked output layer (the default where served). The analysis:
  `research/reviews/prefix-cache.md`. With `prefix_cache_scope="client"` an application user can also tell from
  the response time whether another user's prompt started with the same tokens: keep `"user"` if your users are
  untrusted.
* **Memory.** About 25 KB per cached token at 0.5B and 115 KB at 7B (1 GiB holds about 43,000 / 9,300 tokens).
* **Several worker processes** each keep their own cache: route a conversation (or a `user`) to the same worker
  for hits.
* Everything cached is wiped on an integrity failure, a model reload and when the client closes.

There is no `base_url`: the provider address comes from the key. Use one client per process and share it
between threads or tasks; it is safe for concurrent use and batches concurrent requests. For many concurrent
users and several worker processes, see [Scaling](scaling.md).

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
| `PGW_FACTORY_WEIGHTS` | `pack` | `factory_weights` |
| `PGW_FACTORY_WEIGHTS_SOURCE` | `provider` | `factory_weights_source` |
| `PGW_CHECK_STOCK_SEGMENTS` | `64` | `check_stock_segments` |
| `PGW_MASK_WORKERS` | a quarter of the CPUs | `mask_workers` |
| `PGW_SPECULATION` | `off` | `speculation` |
| `PGW_SCHEDULE` | `standard` | `schedule` |
| `PGW_ROUTING` | `off` | `routing` |
| `PGW_SDK_LINK_RTT_MS` | unset | `schedule="auto"`: use this round-trip time (ms) instead of measuring the link (tests, or a link you know) |
| `PGW_MASK_ADMISSION` | `entitled` | `mask_admission` |
| `PGW_FACTORY_STALL_S` | `120` | `factory_stall_s` |
| `PGW_SDK_MASKGEN_STALL_S` | `30` | `maskgen_stall_s` (`0`: the maskgen watchdog is off) |
| `PGW_SDK_PREVIEW_MODELS` | unset (off) | measurement-only preview: a comma list of pinned models outside the supported list (e.g. `qwen2.5-32b-instruct`) that the library then accepts; still verified against their baked pins ([Models](models.md#planned)) |

| `PGW_DRAFT_HEAD_URL` | unset | a mirror to fetch the pinned draft heads from when the installed package does not carry them (`<url>/<file>`; checked against the pinned SHA-256 before use) |
| `PGW_MASK_STORE_GB` | `8` | `mask_store_gb` |
| `PGW_SDK_LOG` | unset | `log_level` |
| `PGW_FACTORY_MIN_FREE_BYTES` | `2147483648` (2 GiB) | free space the mask factory leaves on the cache volume; it stops producing below it ([Reliability](reliability.md#disk)) |
| `PGW_MASKSTORE_MIN_FREE_BYTES` | `1073741824` (1 GiB) | free space the mask store keeps when it writes a segment |
| `PGW_SDK_FACTORY_HANG_S` | `300` | a mask factory process without a heartbeat for this long is killed and restarted |
| `PGW_NATIVE` | `1` | `0` disables all native extensions (pure-Python path: same results, much slower). For diagnosis only |
| `PGW_NTT` | `1` | `0` disables only the native NTT products of mask generation |
| `PGW_PKS_NATIVE` | `1` | `0` disables only the native `pks-v3` kernels |
| `PGW_PKS_ISA` | `auto` | caps the instruction set of the native `pks-v3` and check-vector kernels (`avx512ifma`, `avx512`, `avx2`, `scalar`); the best one the CPU has is picked by default. Every setting gives the same bits. For diagnosis and benchmarks only |
| `PGW_REQUIRE_NATIVE` | `0` | build time only: `1` makes a source build fail when the native extensions cannot be compiled |

## Schedule: standard or fused

Every generated token needs a fixed number of sequential round trips to the service. On a link with a high
round-trip time (RTT), such as a laptop or phone on a home network, those round trips are most of the time per
token. `schedule="fused"` halves them:

| | round trips per token | 0.5B / 1.5B / 3B / 7B | bytes per value | answers |
|---|---|---|---|---|
| `"standard"` (default) | `4 x layers + 1` | 97 / 113 / 145 / 113 | 4 | the standard fixed reference |
| `"fused"` (opt-in) | `2 x layers` | 48 / 56 / 72 / 56 | 7 | the fused fixed reference |

How: each round trip carries two linear layers of the model (the service applies them one after the other to
masked values), the normalisation between them is folded into the second layer's public weights, the model's
output layer rides the last round trip, and the first layer's first linear runs on your machine (its input is the
token you just sampled, so nothing could be sent ahead of it). Privacy and checks are unchanged: the service sees
only one-time-masked values of public shapes, each mask is used once, and every value the service returns is
range- and Freivalds-checked (over the prime `2^61 - 1`) before it is used; a wrong value stops the answer with an
`IntegrityError`, as on the standard schedule.

**Answers are not bit-identical to `"standard"`.** The fused schedule rounds differently (a fixed-point residual
stream and folded int8 weights), so it is its own fixed reference: the same seed and prompt give the same tokens
and logprobs on every run of `"fused"`, but not the same as `"standard"`. Measured quality on Qwen2.5-0.5B-Instruct
(4088 wikitext tokens): perplexity 19.90 vs 19.85, the same top-1 token at 97.4 % of positions
(`benchmarks/results/fused2/quality_qwen05b_headfold_i8.json`); 7B in `STATUS.md`. Because of this, the schedule is
decided **once per model when the client starts it** and never changes while the client runs: not during an
answer, not between the turns of a conversation, not after a reconnect. Pin `"standard"` or `"fused"` when every
run of your application must use the same reference; `"auto"` may choose differently on another day or network.

**Costs.** Values on the wire are 7 bytes instead of 4: the same number of values per token, so 1.74x the bytes
per token (Qwen2.5-0.5B: 1.27 MB up and 3.18 MB down per token instead of 0.73 / 1.82 MB; 7B: 5.8 / 10.8 MB instead
of 3.3 / 6.2 MB). On a consumer link the bytes often cost more than the round trips (7B at 100 Mbit/s up: about
0.47 s per token for the upload alone), so `"fused"` helps most where the RTT is high and the bandwidth is not the
limit; `"auto"` weighs both. Mask generation
for the fused schedule costs more client and service CPU per token (the `pks-v3` factory makes each 56-bit mask from
3-4 ordinary 32-bit products), and its first use builds a second one-time mask hint. The service must offer it:
`client.status()["models"][model]` shows `schedule` and, with `"auto"`, the decision.

**`"auto"`.** When the model starts, the client measures the link to the service (the median round-trip time of a
few empty authenticated probes, then the upload and download time of 512 KiB each way; no content) and estimates
the time per token of both schedules from public shapes only: `round trips x RTT + bytes up / upload rate + bytes
down / download rate + client work per round trip`. It picks `"fused"` when the service offers it, the RTT is at
least `schedule_auto_rtt_ms` (3 ms) and the fused estimate is at least `schedule_auto_margin` (10 %) lower;
otherwise `"standard"`. On a slow uplink the larger values can cost more than the saved round trips, which is why
bandwidth counts. The decision with its inputs (`link.rtt_ms`, `link.up_mbps`, `link.down_mbps`, both estimates,
the reason) is in `client.status()["models"][model]["schedule_decision"]`. A service without the fused schedule
or the probe gives `"standard"` with the reason recorded; `schedule="fused"` against such a service fails with
`NotFoundError` (never a silent switch of reference).

Not combined with `"fused"` yet: `prefix_cache=True`, `factory_weights="stream"`, `lane` other than `"auto"`,
`masked_head="off"`. Strict `speculation` works on both schedules (every verify pass has `spec_rows` rows; filler
rows are masked rows on the fused schedule, so it costs `spec_rows` masks per pass).

## Speculative decoding: when to enable it

Speculation never changes an answer and never changes what the service sees (rounds, rows, bytes); it changes how
fast the answer reaches you and what it costs you. Measured on 2026-11-26 with the shipped heads (greedy decoding,
held-out prompts the heads were not trained on: open chat from UltraChat, code from MBPP):

| | open chat, tokens per round | code, tokens per round | decode speed vs off |
|---|---|---|---|
| `qwen2.5-0.5b-instruct`, `draft_head`, `spec_rows` 2 / 3 / 4 (end to end, CPU service) | 1.52 / 1.82 / 1.95 | 1.62 / 1.92 / 2.15 | chat +31 / +47 / +38 %, code +42 / +59 / +50 % |
| `qwen2.5-0.5b-instruct`, `lookup`, `spec_rows` 3 (same run) | 1.18 | 1.20 | -9 % (chat), -4 % (code) |
| `qwen2.5-7b-instruct`, `draft_head`, `spec_rows` 2 / 3 / 4 (replay of the model's own answers) | 1.45 / 1.64 / 1.72 | 1.61 / 1.94 / 2.12 | modelled at 0.2 ms round-trip time on a GPU service: +17 / +11 / +1 % (open chat) |
| `qwen2.5-7b-instruct`, `draft_head`, `spec_rows` 2 / 3 (end to end on one 4-vCPU T4 machine: service, your process and its mask factory together; earlier head of the same size, 5 prompts) | 1.53 / 1.69 | 1.59 / 1.81 | -27 / -62 %: on that machine every extra row costs the T4 (no int8 tensor-core GEMV) and the factory shares the 4 CPUs; it measures tokens per round, not a deployment |
| a code edit (the answer copies the prompt), 0.5B, `spec_rows` 3: `draft_head` / `lookup` | 2.16 / 1.98 | | +81 % / +58 % |

Enable `speculation="draft_head"` when:

- one conversation at a time matters more than throughput (an interactive chat, an agent waiting on each answer);
- the mask stock keeps up with `spec_rows` times the tokens: each round takes `spec_rows` masks whatever is accepted,
  so the background factory must produce `spec_rows` times as fast as without speculation. With `pks-v3` at 7B on
  a small machine the factory, not decoding, is usually the limit; check `client.status()` (`masks_available`) under
  your load first;
- the next request of the same `user` can wait for the previous answer's remaining rounds (they run after the answer
  streams, up to `(1 - 1 / tokens per round)` of its length).

Use `spec_rows=2` at 7B (the best measured trade on a fast service: the extra rows cost provider work and wire on
every round) and 3 at 0.5B. Use `"lookup"` only when answers mostly copy the prompt (code edits, rewrites,
retrieval); on open chat it is slower than off. Speculation stays off by default: it multiplies mask use and
per-round traffic by `spec_rows`, which makes it a loss when the mask factory or the link is the bottleneck, and it
orders requests of one `user` behind each other.

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
  the same fingerprint (about 6.5 GB at 7B). The process that handles your requests never loads them. With
  `factory_weights="stream"` they are not kept ([below](#keeping-no-weights-factory_weightsstream)).
* **Mask setup**: the one-time material for mask generation (about 2.4 GB at 7B with `pks-v3`).
* **Mask stock**: masks prepared ahead of use, up to `mask_store_gb` (8 GB) per model.

First use of a model, once per machine: enrollment (seconds), the model data and the weights (about 11 GB at
7B, mostly the 6.5 GB of int8 weights), then the mask setup (with the default `hint="local"`, at 7B about 10 CPU-minutes
on a CPU with AMX and 16 with AVX-512 VNNI, about 5 minutes on a 4-vCPU AMX machine; on CPUs with AVX2 only about
1.5-2 CPU-hours). Later starts reuse everything and take seconds.

### Preparing a model ahead of time

**Recommended before the first traffic.** Without it, the first request of a fresh machine waits for the whole
setup (21 minutes at 7B on a 4-vCPU VM in our Kaggle run), and requests arriving during the first minutes of mask
production may be refused with `mask_exhausted`. Prepare the model beforehand, for example while the
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

### Keeping no weights (`factory_weights="stream"`)

Opt-in, for hosts whose disk cannot hold the int8 weights. The mask processes need the weights only for check
values: per segment of masks, the segment's secret check vector times the weights, and per worker start (and after
every aborted segment) a set of private check values. With `"stream"` they prepare these ahead of time in passes:

1. The weights are streamed from `factory_weights_source` (our provisioning endpoint, or the public checkpoint).
   The source is not trusted: every tensor is checked against the SHA-256 in the manifest that the library's
   pinned fingerprint covers, and nothing from a pass is kept unless every tensor passed.
2. During the pass the processes draw fresh secret seeds from the operating system's random generator, exactly as
   they would on demand, and compute the check values of the next `check_stock_segments` segments and a few
   private sets with the same code. Masks, check vectors and answers are bit-identical to `"pack"`.
3. The check values go into an encrypted stock under the mask store's key (`<state>/.../factory/.../check-stock`,
   owner-only files, authenticated index). Each item is used once; its use is recorded on disk before it is read,
   so a crash or restart never hands one out twice. Tampered items are refused. The weights are dropped.
4. When the stock falls to a quarter, the next pass starts in the background.

The first setup streams the weights twice (once for the mask setup, once for the first stock). Requests always
run with checked material: if the stock is empty and no pass can reach a source, masks stop being made, and
requests wait for masks and then fail with a clear error naming the failing pass. `pgw sdk status` shows the
stock, its passes and the last error; `pgw sdk prepare --factory-weights stream` sets a model up this way and
removes a factory pack left by an earlier `"pack"` setup. Disk and network per model:
[Scaling](scaling.md#keeping-no-weights-on-the-client). Security note:
`research/reviews/weight-free-stock-security.md` (nothing weakens; the private check values now sit on disk inside
the mask store's encryption).

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
