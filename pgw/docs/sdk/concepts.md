# Concepts

This page explains what the PGW client library does inside your process, what our service sees, how
results are verified, and why your application should run close to our GPU.

## The short version

* The model is a public open-weight model (Qwen2.5-Instruct). Its weights are not secret.
* Your prompts, the model's intermediate values and its answers are secret. They exist in readable form
  only inside your process.
* The expensive part of inference, multiplying activations by large weight matrices, runs on our GPU. Before
  any activation leaves your process, the library adds a fresh random mask that is used once and never
  again. To our GPU the data is uniformly random noise. The library removes the mask from the result.
* Every result that comes back is checked before it is used. A wrong result is detected and the request
  fails; it is never silently accepted.
* The answer is the same, bit for bit, as running the same quantized model yourself.

## What stays private

| Data | Readable by you | Readable by us |
|---|---|---|
| Prompts, system prompts, conversation history | yes | no |
| Generated text, token ids, logits, log-probabilities | yes | no |
| Intermediate activations of the model | yes | no (masked) |
| Your one-time masks and their secret keys | yes (generated in your process) | no |
| Which model you use, how many tokens, when | yes | **yes** (needed to run and bill the service) |
| Request timing and sizes on the network | yes | **yes** |

The second group is metadata. PGW does not hide it in this release: we see that a request happened, for
which model, its length in tokens and its timing. We do not see what it says.

## What runs where

```
your process (CPU)                                          our service (GPU)
---------------------------------------------------------   -----------------------------------
your app  -- OpenAI-style call -->  pgw client library
                                    tokenizer, chat template
                                    embeddings, attention, norms,
                                    sampling (plaintext, local)
                                    linear layers:  x + r  ------------>  W (x + r)   (int8 GEMM)
                                                    subtract W r  <-----
                                                    check result (Freivalds)
                                    background mask generation:
                                    encrypted masks  ------------------->  products on ciphertexts
                                    decrypt, self-check  <---------------
```

In your process:

* the tokenizer and chat template, the embedding lookup, attention, normalisation and sampling, all on CPU
  and in plaintext;
* masking and unmasking of every activation sent to us, and the integrity check of every result;
* background threads that prepare one-time masks ahead of use;
* a cache directory with your enrollment credentials and the non-secret model data
  ([Configuration](configuration.md#cache-directory)).

Your process does not hold the large weight matrices of the model, needs no GPU and keeps no large mask store.

On our side: the int8 matrix products on masked data, and help with mask generation on encrypted data.
We store no prompts or answers because we never receive them.

## Verification

* **Results.** Each product our GPU returns is checked with Freivalds' algorithm against secret random
  vectors held in your process, plus a range check. With the default settings a wrong result passes one check with probability at most
  2^-20 per call, and a single detected failure stops the request with `integrity_check_failed`. See the
  [security whitepaper](../security/whitepaper.md#5-integrity-the-freivalds-bound) for the bound.
* **Masks made with our help.** When our service helps generate masks, every row it contributes is
  self-checked in your process with at least 40 bits of soundness against errors we could choose, before
  any mask is used. The one-time setup material we build for you is derived from seeds your process chose
  and is verified the same way (or built locally if you prefer).
* **Model data.** Every downloaded file is checked against the model fingerprint pinned in the protocol.
* **Service identity.** The service certificate authority is pinned by your key; a different server is
  refused even if it has a publicly valid certificate.

## Exactness

The model runs with int8 weights and a fixed quantization of activations. Masking does not change any bit of
the result: tokens and log-probabilities equal those of the same quantized model run in plaintext (measured,
max logit difference 0.0 on Qwen2.5-0.5B; see the [docs index](../README.md#key-facts-at-a-glance)).
Greedy decoding (`temperature=0`) is therefore reproducible across runs.

## Why your application should run near our GPU

Generating one token needs a chain of round trips between your process and our GPU: four per transformer
layer, one after the other, so 112 round trips per token for Qwen2.5-7B. Each round trip costs at least
the network round-trip time (RTT). The RTT therefore multiplies:

| where your process runs | RTT (estimate) | added per token, Qwen2.5-7B (estimate) |
|---|---|---|
| same data center as our GPU | 0.1-0.3 ms | +0.01 to +0.035 s |
| same cloud region, other zone | 0.5-1 ms | +0.05 to +0.11 s |
| other region on the same continent | 10-30 ms | +1.1 to +3.4 s |

These are estimates from the [sizing guide](../operations/sizing.md#3-latency); the
pilot measurement ([pilot hand-off](pilot-handoff.md)) gives your real numbers. Concurrent requests share round trips (they are batched),
so throughput scales better than single-request latency.

In practice: run the process that imports `pgw` in the same data center as our GPU. Your users can be
anywhere; the call from your users to your application crosses the long distance once per request, not
112 times per token.

### For the pilot: a CPU pod next to our GPU on RunPod

For the pilot our GPU service runs on RunPod in one data center, which we name when we send your key. You
run your application, or a small test application, on a RunPod **CPU pod in that same data center**, in
your own RunPod account. Nothing else is required: the pod reaches our service over HTTPS at the address
in your key. The [pilot hand-off](pilot-handoff.md) has the five-minute setup. If your application must
stay elsewhere for now, run the [FastAPI example](../../examples/README.md) on the pod as a private
endpoint and call it from your application; only that one hop crosses the longer distance.

## Mask schemes

How masks are generated is selected with the client option `mask_scheme`
([Configuration](configuration.md#client-options)):

| scheme | how masks are made | setup in your cache | status |
|---|---|---|---|
| `pks-v3` (default for the client library) | encrypted with a compact ring-LWE scheme; our GPU computes on ciphertexts; your process decrypts and self-checks | about 2 GB at 7B | **pending external review** (see [Security model](security.md)) |
| `mlwe` | encrypted with module-LWE, the scheme of our server deployments | much larger one-time setup (tens of GB at 7B) | default of our server-side deployments; no external audit yet |
| `plain` | computed without our help, no encryption assumption | the model's int8 weights (several GB at 7B) | slowest; most CPU and disk |
