# Client-side KV prefix cache (2026-11-26)

Status: implemented behind `prefix_cache=True` (opt-in with the masked LM head; the earlier `"auto"` default is
unchanged). Code: `src/pgw/sdk/prefix_cache.py` (new), a 4-line predicate change in
`pgw.engine.generate.Generator.prefix_reuse_supported` plus a `rows_exact` flag on `pgw.engine.masked_head.MaskedHead`,
a 3-line builder hook in `pgw.gateway.runtime.ModelRuntime`, options in `pgw.sdk.options`, wiring in
`pgw.sdk.engine`. Tests: `tests/engine/test_prefix_cache_masked_head.py`, `tests/sdk/test_prefix_cache.py`,
`tests/sdk/test_e2e_cpu.py::test_conversation_prefix_reuse_is_exact`. Benchmark: `benchmarks/sdk_prefix_cache_bench.py`.

## Problem

Every prompt token costs one row on each of the `4L + 1` round trips of the prompt pass (rows of a pass go in
multi-row items) and one single-use mask slot, which the factory produces at 11 / 34 / 103 / 157 ms of client CPU
per slot at 0.5B / 7B / 32B / 72B ([scaling](../../docs/sdk/scaling.md)). A chat client resends the whole
conversation every turn, and most applications prepend the same system prompt to every conversation, so most
prompt tokens of a busy client are tokens the model has already processed in this process. TTFT at 7B on a T4
was 3.9 s.

A conversation prefix cache already existed (SPEC 9.4b, `pgw.engine.prefix_cache.ConversationPrefixCache`), but:

1. it was refused with the masked LM head (`prefix_reuse_supported` required the default head), and the masked
   head is the SDK default whenever the provider serves it, so in the default product it was off;
2. it keeps one entry per conversation key and a request *takes* it: a system prompt shared by several
   conversations, a regenerated answer or two concurrent conversations without a `conversation_id` thrash it
   (a mismatching prompt wipes the entry).

## Design

### What is cached

The plaintext KV cache (keys after RoPE, values; float32 on the native path) of every position that went through
the model for a finished request: its prompt and its generated tokens but the last (never fed back). Per token:
24.6 KB at 0.5B, 115 KB at 7B, 524 KB at 32B, 655 KB at 72B (`2 x layers x kv_heads x head_dim x 4`), plus 8 bytes
for the token id. Logits are not cached: a request always runs at least its last prompt token (the prompt pass
needs one new row for the first token's logits).

Where: client process RAM only. Nothing is written to disk, nothing is sent anywhere (the KV and the tokens never
leave the process; the provider only ever sees masked rows of the positions that are *not* cached). There are
no cache files, so there is nothing to protect at rest and `pgw sdk clear` has nothing to wipe; process exit
frees it. As for the live KV caches of running requests: deploy without swap or with encrypted swap.

### Keying

* **Exact token prefix.** A lookup compares the request's token ids (after the chat template) with each stored
  entry's token ids and takes the longest common prefix, capped at `len(prompt) - 1`. No hashing of blocks, no
  approximate matching: the reused positions are exactly the positions whose tokens are equal.
* **Model, pack and ring configuration.** The cache object belongs to one `ModelRuntime`, which is bound to one
  pack fingerprint (the masked-head fingerprint when the head is masked), ring, head mode and float path. The
  runtime builds it when its engine is finished and closes (wipes) it on reload and stop; a new runtime (new
  provisioning, new fingerprint, other ring) starts with an empty cache. The engine re-checks the KV layout
  (layers, kv heads, head dim, dtype, device) of every hit.
* **Scope.** The OpenAI `user` of the request (`prefix_cache_scope="user"`, default) or the whole client
  (`"client"`, opt-in). Requests only see entries of their own scope. Scope strings are hashed to a fixed-length
  digest in the key, so no `user` or `conversation_id` value can alias another scope. The `conversation_id`
  (`extra_body` or `X-PGW-Conversation-Id`) is no longer needed for hits; it narrows only the settle wait (a new
  turn waits up to 30 s for the commit of a finished turn of the same conversation).

### Lookup, move, copy

* A request scans its scope's entries (at most `max_entries`, 1024) and picks the longest match (ties: the most
  recently used).
* **Move** when the whole entry is used (the next turn of a conversation): the request takes the entry's KV
  cache over without a copy and its own cache, which extends it, replaces it when the request ends.
* **Copy** when only part of an entry is used (a shared system prompt, a regenerated answer, an edited earlier
  message): the first `m` positions are copied under the cache lock (no eviction can zero the source meanwhile)
  and the entry stays for others. Cost: one memory copy of `m` positions (25 MB for 1,000 tokens at 0.5B).
* A moved entry whose request never started (it waited for masks and timed out, was cancelled in the queue, or its
  reservation failed) goes back into the cache unless the cache was wiped meanwhile (epoch check). Once the
  engine claims it, its token array is zeroed at once.
* **Store** only after a clean end (`stop` / `length`, not cancelled, strict-mode dummy passes done) and after the
  session's deferred integrity checks joined without a failure (unchanged from SPEC 9.4b). Entries of the same
  scope whose tokens are a prefix of the new entry are dropped (they can never win a lookup again); a new entry
  that is a prefix of a stored one is not stored.

### Eviction and memory budget

`prefix_cache_mb` (default 1024 MiB per model and process) bounds the stored entries (KV plus tokens; the KV is
compacted to its committed length on store). Least recently used entries are evicted first; an entry larger than
the budget is never stored. `prefix_cache_ttl_s` (600 s) expires unused entries (a sweeper thread and every
lookup). At most `prefix_cache_max_entries` (gateway default, 1024) entries. Every eviction, expiry, refusal,
mismatch and wipe zeroes the KV tensors and the token array before dropping them. A request's own copy is its
live KV cache (bounded by the context length and `max_concurrent`) and is not counted in the budget.
1 GiB holds about 43,000 tokens at 0.5B, 9,300 at 7B, 2,000 at 32B and 1,600 at 72B.

### Exactness (bit-identical outputs)

Reuse is exact when every K/V row and every logit row depends only on its own token and the cache prefix it
attends to, never on how many rows share a pass (the earlier turn computed its generated tokens one row per pass;
the recompute would run them inside one prompt block). SPEC 9.4b proves this for the native float path
(`float_ops = "native"`, the SDK's setting: fixed-order per-row RMSNorm, RoPE, attention, SiLU) with row-exact
integer linears (per-row activation scales, exact ring products). The masked LM head is one more such linear:
each logit row is its own hidden row normed alone (`rms_norm` of the last row, `norm_rows` for verify rows),
quantized per row and multiplied exactly by the per-row int8 head; a prompt pass and a decode pass both send one
row (verify rows: `R` rows, each normed alone). `MaskedHead.rows_exact = True` states this and
`prefix_reuse_supported` accepts a head mode only when it declares it. torch float kernels stay refused.

Sampling determinism: the sampler is built per request from its parameters and its whole prompt (as before);
with bit-identical logits it makes the same draws. Seeded sampling and logprobs are covered by the tests.

### Masks, reservations, admission

The reused positions take no row and no mask: the reservation is `(prompt - reused) + max_tokens` (strict
speculation: `mask_rows_needed(prompt, max_tokens, reused)`; padded: the public charge schedule from
`prompt - reused`). The entitlement ledger and the stock check use the same smaller figure, so a cached request
also starts sooner when the stock is short. Single use is unchanged: every row that is sent still takes a fresh
slot. The masked head's slot rows are part of every slot (unchanged).

### Speculation (strict mode, dummy passes)

Unchanged apart from the shorter prompt pass: one prompt pass, then exactly one pass per generated token with
`spec_rows` rows, dummy passes after the answer; an entry is stored only after the dummy passes (`_owes`). The
prompt-lookup drafter sees the full context (the reused prefix is handed to it as context). The head's item per
pass has the same row count with and without a hit (tested with filler rows and with padded passes).

### Integrity

Unchanged and fail-closed. Reused rows were checked (Freivalds rate 1.0, p31 in the SDK) when they were
computed, and an entry exists only if its request's deferred checks all passed. Any integrity failure or noise
alarm of the model wipes every entry and bumps an epoch, so requests running at that moment can neither store
their cache nor give a taken entry back. New rows of a hit are checked exactly like any row.

### Multi-process sharing: not done

Each client process has its own cache. Sharing would need the plaintext KV in shared memory or on disk (a new
at-rest secret, 115 KB per token at 7B, needing encryption, key handling and crash-safe wiping) or a broker
process on the hot path. The gain is small when a load balancer routes a conversation to the same worker
(sticky routing on the conversation or user, which the docs now recommend); the shared system prompt is paid once
per process. Mask admission is already shared across processes (`SharedLedger`); the cache is not.

## Privacy analysis

What the provider sees for a request today (masked head, no cache): a prompt pass of `4L + 1` trips whose items
carry `P` rows (the head: 1 row), then one pass per generated token (`R` rows each with strict speculation), the
timing of every trip and of the client's work between trips, the factory's mask generation over time
(segment-granular), and the content-free usage report (`prompt_tokens` minus `cached_tokens` since S3).

With a hit of `m` tokens, the only change on the wire is that the prompt pass carries `P - m` rows instead of `P`.
Decode passes, their rows, the pass count and the dummy passes are identical. Every sent row is still masked with
a fresh single-use mask; no token, activation, KV or mask information changes.

What the provider can infer that it could not before:

1. **A per-request hit bit.** A continued turn's prompt pass is short (the new user message and template
   tokens), shorter than any fresh prompt of an application that uses a system prompt, and it is followed by
   decode passes whose client-side turnaround grows with the real context the client attends over (`P`, not
   `P - m`). So the provider can tell, with good confidence, that a request reused a cached prefix, and roughly
   how long its context is.
2. **Turn linking without arithmetic.** Today the provider links the turns of a conversation from row counts
   (`P(n+1) = P(n) + L(n) + new message + template`), which gets ambiguous with many concurrent conversations of
   one key. The hit bit marks a continuation per request, independent of other traffic (timing usually narrows
   it to one earlier request).
3. With `prefix_cache_scope="client"`, a first turn that hits shows that it shares a prefix with an earlier
   request of *another* user of the same key (in practice: the application's system prompt).

What it shows less: the full prompt length `P` of a continued turn is no longer sent as a row count (only
estimable from timing).

**Verdict: the cache leaks something new** (1 and 2; 3 with client scope). It is small next to what row counts
and timing already show (prompt lengths, answer lengths, turn timing, linkability by arithmetic), but it is not
nothing. So: with the masked LM head (the default product) the cache stays **opt-in** (`prefix_cache=True`).
`"auto"` keeps its earlier meaning (on only when the masked head is not used); that pre-existing default
enables the same leak for clients whose provider does not serve the masked head. It is left unchanged here (it is
documented in SPEC 9.4b and changing it is a product decision); the recommendation is to make `"auto"` mean off
in a later release so the default never shows the hit bit. Cross-user sharing (`"client"`) is separately opt-in:
besides 3, it lets one application user measure (by TTFT) whether another user's prompt started with the same
tokens, the usual prompt-cache timing side channel inside the application.

### Hiding hits by padding prompt trips: considered, not built

Padding would send the `m` cached positions as padding rows (uniform random ring elements, no mask; the
mechanism exists for strict speculation) so the prompt pass keeps `P` rows. It does not hide hits well enough to
make the cache safe by default, and it removes most of the gain:

* **Mask demand.** Padding rows take no mask, so over time the factory's mask generation (which the provider
  performs) falls short of the rows it sees by the cached volume: the aggregate hit rate stays visible. Burning
  `m` masks per hit to close that gap gives back the main saving (masks are the dominant client cost).
* **Timing.** The client's work per trip still shrinks with a hit (the padding rows need no attention, norm or
  unmasking), so the per-trip turnaround shows the hit unless the client also wastes the time.
* **Gain left.** Trips, bandwidth and provider work per trip are unchanged; TTFT barely improves.
* **Cost.** Padding rows inside a prompt pass (positions after the real rows, the head on the last *real* row,
  KV positions beyond the context) is a sizeable change of the engine's hot path.

So the honest options are "cache (opt-in, documented leak)" or "no cache"; a padded mode would be neither safe by
default nor much faster. Documented in `docs/sdk/configuration.md` and `docs/sdk/security.md`.

## Measurements

See STATUS.md (2026-11-26, client KV prefix cache) for the numbers measured with
`benchmarks/sdk_prefix_cache_bench.py` (tiny and 0.5B, CPU provider on the same 4-vCPU box, plain masks).
