# Tree speculation under strict mode, and cheap drafts for 32B/72B targets

Date: 2026-11-22. Scope: raise the tokens per verify pass (tau) of strict speculation from 1.85-2.84 towards
3-5 without weakening the strict-mode privacy property, because a 32B/72B pass costs 2L+1 to 4L+1 sequential
round trips (L = 64-80 layers). Local CPU work only (4 shared vCPUs at load 30-40 from other jobs; Qwen2.5
0.5B/1.5B/3B checkpoints and the Kaggle run 28 exact answers; no Modal, no Kaggle).

Lead input during this work: on the H100 7B run the Qwen2.5-0.5B int8 draft took ~60 ms per step on a
32-core Modal gateway (~0.49 GB read per step; memory-bandwidth bound), the pipelined drafter hit 0.355 and
gave 1.19 tokens per pass at R = 2, so speculation was slower than plain (7.0 vs 11.0 tok/s). So: drafts must
read a few MB per step. This note therefore ranks drafts by tau *and* bytes read per draft step.

## 0. Summary

| item | result |
|---|---|
| tree verification (engine, opt-in `spec_tree`) | **implemented, exact**: a public fixed topology of R nodes verified in one pass with tree attention (torch and native float paths, no kernel change); outputs bit-identical to plain decoding for greedy and seeded sampling (the target's own sample decides every token); R rows per pass and the strict L-pass schedule unchanged; fixed gateway work per pass (section 1). 123 new test cases pass, plus the existing speculation/padding/CPU-gateway suites. |
| does a tree help the 0.5B draft? | **no, at R <= 8**: alpha_top1 = 0.81 (1.5B) / 0.78 (3B), so the optimal static tree of <= 8 nodes is the chain (R = 4: chain 2.84 vs best fork 2.51; R = 8: 3.93 vs 3.88). Trees pay only at alpha <= ~0.7 with R >= 8 (section 6). |
| cheap drafts (lead's priority: a few MB per step) | gateway **does** hold the target's final hidden state in plaintext (only linears leave it). **`FeatureHeadDrafter`** (EAGLE-style head on the target feature, engine opt-in, exact, fixed work): **5.5 MB per draft step at 1.5B, 9.7 MB at 32B, 13 MB at 72B** (int8), vs 490 MB for the 0.5B draft. Prompt lookup: ~0 bytes but tau 1.08. |
| head acceptance, measured | **not achievable with local data**: ~6.6k training windows (24 chat answers in 2-fold CV + 0.5B/3B answers + 3.3k wikitext tokens; the 1.5B prefill ran at ~7 tokens/s on this loaded box) overfit a 5.5 M-weight head (train top-1 0.94, held-out depth-1 top-1 0.13). Real heads need EAGLE-scale data (a one-off GPU job); tau vs alpha is tabulated instead (section 6). |
| mask cost | a tree costs what a chain of the same R costs; tau vs R vs slots/token in section 4: R = 4 2.84 tau at 2.50 slots/token (C = 4, B = 160); R = 8 3.93 at 4.08 (C = 8, B = 320) vs 7.91 uncapped. |
| 32B / 72B model | the 0.5B draft makes 32B slower than plain (8 vs 14.5 tok/s at 0.3 ms trips, 2L+1); **25-30+ tok/s at 32B needs 0.3-0.5 ms trips, a 2L+1 schedule, per-row trip work on spare cores (one extra row costs ~15 ms of gateway CPU at 32B) and a few-MB head at alpha ~0.8, R = 8 (29-36 tok/s)**; 72B tops out at ~26 (section 5). |

## 1. Tree verification in one pass (engine, opt-in)

`Generator(spec_rows=R, spec_strict=True, draft=..., spec_tree=[(parent, rank), ...])` (default `None`: the
chain). The tree shape is a **public, fixed topology** of R nodes: node 0 is the pending token, draft node i is
the `rank`-th choice of the draft at its parent (rank 0: the draft's own pick, coupled to the request's
sampler for sampled requests; rank r: the next most likely tokens). Nodes are listed so that every parent
comes first; any prefix of the node list is a tree.

Model side (`SequenceInput.tree_parents`, `tree_base`; `TransformerModel._tree_attention` (torch float path),
`_tree_attend_native` (native float path); `KVCache.layer_storage`, `KVCache.keep_path`):

* the tree's rows are ordinary rows for every linear (row-wise exact integer products on the masked
  backend; padding rows still trail), so nothing changes for the provider or the backend;
* attention: the new rows are split into chains (runs where each row's parent is the previous row; a function
  of the topology only). For a chain whose first node has depth d, the K/V of its ancestors are copied into
  cache positions `base .. base + d - 1`, and the chain is attended at `base + d` with RoPE at
  `base + depth`: exactly the computation one-row decoding of that root-to-node path performs (the native
  kernel computes each row on its own over `[0, pos]`; the torch path attends row by row like the serial
  tail). After each layer the nodes are written back in node order (`base + i`);
* after acceptance `KVCache.keep_path(base, path, R - 1)` moves the accepted nodes to `base, base + 1, ...`
  with exactly R - 1 position copies (self-copies for the unused ones), so its cost does not depend on the
  accepted length;
* `tree_base < length` lets a draft model grow a tree over several steps (rows of later steps attend to the
  right nodes of earlier steps).

Generator side: the drafter fills one token per node (`DraftJob.tree`); a credit-limited pass (`spec_mask_rows`)
sends the longest node-order prefix as real rows and the rest as padding; the walk samples the target's token
at the root with the request's own sampler, keeps the child whose token equals that sample, samples again at
that child, and so on; the remaining rows are sampled with a throw-away fork (R samples per pass, as before).

Drafter side (`BuiltinDrafter`, draft model, no lookup): one catch-up step (the unseen tokens: accepted path +
pending token, **padded to R rows** with filler rows that are dropped again, so the step's row count is R
whatever was accepted; this also closes the chain drafter's documented "catch-up rows" residual for tree
mode), then one step per topology level that feeds that level's internal nodes as tree rows; every internal
node makes the same picks (a top-k, plus one coupled draw for its rank-0 child when sampled). After the pass
the tree rows are dropped from the draft cache.

### Exactness

The output is the target's own sample at every position, drawn with the request's sampler in the same
order as plain decoding; a draft only decides whether the logits row of the next position is already
available. So outputs are **bit-identical** to plain decoding for greedy and seeded sampling (stronger than
the distribution-level exactness of SpecInfer-style multi-draft rejection sampling, which would change the
RNG consumption and hence the seeded outputs). The acceptance rule "sampled token is one of the node's
children" accepts with probability p_target(children set); with the coupled rank-0 child plus the draft's
top alternatives this is the natural multi-candidate generalisation of the coupled chain.

Tests (`tests/engine/test_tree_spec.py`):

* `test_tree_rows_equal_path_decoding`: every node row of a tree pass equals one-row decoding of its path, bit
  for bit, torch and native float paths, plain and masked (every linear remote) backends, three topologies;
  after `keep_path` decoding continues exactly as plain decoding of the kept path;
* `test_tree_continued_over_passes`: a tree grown over two passes (`tree_base`) is exact;
* `test_strict_tree_speculation_is_exact`: the generator in strict mode with tree speculation equals plain
  decoding (tokens, text, finish reason, logprobs), greedy and seeded, topologies fork4 / wide4 / tree6 /
  tree8, masked filler rows, padded rows with C = R and C = 2 (credit-limited prefixes), a same-model and a
  different-architecture draft, torch and native float paths (native target + native int8 draft); the
  provider sees R rows per linear on every decode pass and exactly L passes; the drafter runs 1 + levels
  steps per pass with R catch-up rows and fixed rows per level;
* `test_tree_config_validation`.

### What the provider sees (privacy)

Unchanged from strict chain speculation: R rows per linear per decode pass (one wire item per linear in pad
mode), exactly L passes (dummy passes after the answer run the same tree code path on filler tokens), uniform
masked or padding payloads, the public mask charge. The topology is public configuration, like R. Gateway
work per pass is a function of the topology and R: the chain split, the ancestor copies, the R - 1 path
copies, R samples, the drafter's 1 + levels steps with fixed row counts. Residuals (same class as before, not
new): attention cost grows with the context during the answer and is flat during dummy passes (L3 section
2.7); per-row cost of a store take versus a padding row (L3 section 2.4) now also tracks how many tree nodes
were real in a credit-limited pass.

## 2. Does a tree raise tau with the 0.5B draft? No, not at R <= 8

`benchmarks/tree_spec_sim.py` replays the 24 exact 1.5B greedy answers of Kaggle run 28 (3,000 tokens; the
same set as L3) through the tree acceptance rule. On the answer path a node's token is the draft's
rank-th choice given the true prefix, so one teacher-forced pass of the draft per answer gives, per position,
the rank of the target's token; the replay is then exact for greedy decoding (up to float differences between
the fp32 HF draft used here and the engine's int8 draft). It reproduces L3's chain numbers exactly (R = 2, 4, 5,
8: 1.78, 2.84, 3.21, 3.93), which validates it.

Rank of the 1.5B's token under the 0.5B draft: rank 0 81.2%, rank 1 10.2%, rank 2 3.5%, rank 3 1.4%.

| R | chain | best tree tried | tree shapes (tau) |
|---|---|---|---|
| 3 | **2.40** | 1.87 | root + 2 alternatives at the root |
| 4 | **2.84** | 2.51 | chain 2 + root alternative 2.51; 3 root alternatives 1.91 |
| 5 | **3.21** | 2.96 | chain 3 + root alternative |
| 6 | **3.55** | 3.33 | chain 4 + root alternative 3.33; chain 3 + two alternatives 3.04 |
| 8 | **3.93** | 3.88 | chain 6 + root alternative 3.88; chain 5 + 2 alternatives 3.74; chain 4 + 3 alternatives 3.49 |

With a draft this well aligned (alpha_top1 = 0.81) the next chain node (value alpha^d, 0.81^7 = 0.23 at depth 7)
is always worth more than an alternative (0.10 at the root, less deeper), so the optimal static tree of up to
8 nodes **is the chain** (the best-first construction in `best_tree` returns the chain for every R <= 8). Trees
pay when the top-1 acceptance is lower (weaker drafts, feature heads, sampling at temperature) or when R is
much larger than 8. The engine support is still what makes those cases possible; for the 0.5B draft model,
keep the chain.

Prompt lookup as an extra root branch (rows taken from the chain): R = 4 with one lookup row 2.33 (chain alone
2.84), lookup-only R = 3: 1.08 tokens per pass on these answers. Lookup adds nothing on top of a draft model
here; alone it gives ~1.1 (as L3 found for the 0.5B).

## 3. Drafts that read a few MB per step

### Does the gateway hold the target's hidden states in plaintext? Yes

With every linear remote, the gateway still computes everything between the linears in plaintext: the
embedding lookup, RMSNorms, RoPE, attention over its own KV cache, SiLU gates and the residual adds
(`TransformerModel._layer_native` / `_layer_standard`: only `LinearCall`s leave the gateway, masked; their
outputs are unmasked and verified on the gateway). So after a verify pass the gateway holds the residual
stream of every row, including the final hidden state (and with the final norm, the LM head's input) of the
node that produced the pending token. `SequenceInput.capture_hidden` already exposes it; the generator now
hands it to a drafter that asks for it (`uses_features`). The target's embedding table (the head's token
input, and for tied models the LM head) is on the gateway already for the embedding lookup.

### Engine: `FeatureHeadDrafter` (opt-in)

EAGLE-style head, autoregressive over its own predicted features:
`z = W_in [f ; E[x]] + b`, `z = z + W_2 silu(W_1 z)`, `f' = W_out z`, logits over a reduced vocabulary V
(`U = E_V W_out` precomputed, so the step never touches the full LM head). Per pass: one batched step per
topology level (the root at level 0, then the internal nodes of each depth), the same picks for every
node, no catch-up step (the target's feature replaces the draft's own context). Chain or tree (`spec_tree`),
strict mode, padding rows: `test_feature_head_drafter_strict_exact` (bit-identical to plain decoding,
greedy/seeded, chain/tree6/wide4, torch/native; R rows, L passes, 1 + levels head steps per pass, a target
feature on every job). The generator's extra work is fixed: the last layer's hidden rows are captured on
every pass and one row (the accepted node's) is normed (O(1) indexing by a private index).

Bytes per draft step (weights read; int8): H = 1536, d = 384, V = 8192: W_in 1.18 M + MLP 0.59 M + W_out
0.59 M + U 3.15 M = **5.5 MB** (1.5B); 32B (H = 5120): **9.7 MB**; 72B (H = 8192): **13.1 MB** — 40-90x less
than the 0.5B draft's 490 MB. Rows per step are almost free (memory-bound), so trees cost little extra
draft time: a tree of depth D costs D steps whatever its width.

### Measured head acceptance (prototype, data-limited)

`benchmarks/draft_head_train.py` (self-distillation on the 1.5B's own features and greedy labels; teacher-forced
bf16 HF forward; 2-fold cross-validation by prompt on the 24 exact chat answers, plus the 0.5B / 3B answers to
the training-fold prompts and 3.3k wikitext tokens): 6.6k training windows. d = 384, V = 8192, 10 epochs: train
top-1 at depth 1 0.94, **held-out 0.13**: the head memorises its training text. A smaller, regularised head
(d = 128, V = 4096, weight decay 0.1, 3 epochs; 0.6 M + 0.5 M weights) reaches train 0.41 and held-out 0.10. The 1.5B prefill ran at ~7
tokens/s on this box (load 35-50 from other jobs; two runs were OOM-killed and the scratch directory was
wiped twice), so no more data was feasible. This says nothing about what a properly trained head reaches; it
says the head must be trained offline on EAGLE-scale data (tens of millions of target-feature tokens; public
weights, one GPU job) before its tau can be measured. The engine path (`FeatureHeadDrafter`) takes the trained
weights as they are.

Bigger target, same draft (0.5B -> 3B, the 24 run-28 3B answers): rank 0 77.8%, rank 1 10.0%; chain
tau 1.74 / 2.30 / 2.73 / 3.04 / 3.24 / 3.58 at R = 2 / 3 / 4 / 5 / 6 / 8; again no tree of <= 8 nodes beats the
chain (R = 8: chain 3.58, chain 6 + root alternative 3.55).

## 4. Mask cost: tau vs R vs slots per generated token

Strict mode with padding rows (`spec_mask_rows` = C, credit cap B): the store hands out
`prompt + j + min(B, floor((C - 1) j))` slots by decode pass j, L passes per answer of L tokens, so slots per token
are public: about `1 + min(B, (C - 1) L) / L`. Real rows of a pass are the pending token plus the longest
node-order prefix of the tree (or chain) that the credit allows. A tree costs exactly what a chain of the same R
costs (same rows; the credit buys nodes in node order). Exact greedy answers, chain = best tree here
(`tree_spec_sim.py`; "uncapped" is C = R without B, which costs what masked filler rows cost):

| R | tau 1.5B (0.5B draft), C = R uncapped | slots/token | tau, C = R, B = 160 | slots/token | tau, C = R, B = 320 | slots/token | tau 3B, uncapped / B = 160 |
|---|---|---|---|---|---|---|---|
| 1 (plain) | 1.00 | 1.00 | | | | | 1.00 |
| 2 | 1.78 | 1.98 | 1.78 | 1.98 | 1.78 | 1.98 | 1.74 / 1.74 |
| 3 | 2.40 | 2.97 | 2.40 | 2.39 | 2.40 | 2.97 | 2.30 / 2.30 |
| 4 | 2.84 | 3.96 | 2.84 | 2.50 | 2.84 | 3.59 | 2.73 / 2.73 |
| 5 | 3.21 | 4.95 | 3.00 | 2.56 | 3.21 | 3.78 | 3.04 / 2.72 |
| 6 | 3.55 | 5.93 | 2.94 | 2.61 | 3.55 | 3.93 | 3.24 / 2.54 |
| 8 | 3.93 | 7.91 | 2.52 | 2.67 | 3.93 | 4.08 | 3.58 / 2.28 |

Reading: with these ~125-token answers B = 320 keeps the uncapped tau up to R = 8 at about 4 slots per token
(half of uncapped R = 8), B = 160 is right up to R = 4 (2.5 slots per token) and starves larger R (the
credit runs out after B / (R - 1) passes). Slots per token grow with tau roughly as `1 + (R - 1) x passes with
credit / L`; a feature head does not change this accounting (its rows are rows like any others). For long
answers the cap B front-loads the speed-up into the first B / (C - 1) passes.

## 5. 32B / 72B: tokens per second at given per-pass costs

`benchmarks/spec_big_target_model.py`: pass time = trips x trip + gateway CPU(R) + draft critical path, tokens/s
= tau / pass time.

* Trips per pass: 2L+1 (fused schedule) or 4L+1 (qkv, o, gate_up, down + masked LM head); L = 64 (Qwen2.5-32B),
  80 (Qwen2.5-72B).
* Gateway CPU per pass, scaled from the measured 7B main-core cost (strict-speculation-cpu-gateway.md section 5:
  4.7 + 6.9 R ms with the r keystream prefetched; a 16-core server core 1.5x faster) by masked elements per row
  (input + output widths of the four linears x layers: 32B 3.34x, 72B 5.06x the 7B) plus 0.035 ms of
  orchestration per trip: **32B: 25.8 / 41.1 / 71.8 / 133 ms at R = 1 / 2 / 4 / 8**, 72B 39 / 62 / 109 / 201 ms,
  all on the main core; "overlap 0.7" puts 70% of the extra rows' work on spare cores (32B: 26 / 30 / 40 / 58
  ms). Extra rows are therefore *not* free at 32B/72B: one more row costs as much gateway CPU as ~50 trips at
  0.3 ms unless it runs on other cores.
* Draft critical path: steps per pass x bytes per step / 8 GB/s (the lead's Modal measurement: 0.49 GB in
  ~60 ms), drafting sequentially between passes (pipelining can hide it but lowers acceptance, section 3 of
  the CPU-gateway note).
* tau: measured on the 1.5B (0.5B draft; feature head) and, for the draft model, the 3B (section 2). Transfer
  assumptions are stated per row below.

tau assumptions (`benchmarks/results/tree_spec/big_target_taus.json`): the 0.5B draft's alpha falls from 0.81
(1.5B) to 0.78 (3B), extrapolated to ~0.70 at 32-72B (tau 2.4 at R = 4; an assumption, nothing measured at
that size here); the feature head at alpha 0.80 (EAGLE-class heads report this range with large training
sets; *not* reproduced here, section 3) or 0.70; tau from the independence model of section 6 x 0.95 (the
model overestimates the measured chain by ~5%). Tokens/s (`benchmarks/results/tree_spec/big_target_model.json`):

| target | trip | schedule | extra-row work on spare cores | plain | lookup R=3 | 0.5B chain R=4 | head chain R=4 (a .70) | head chain R=8 (a .80) | head tree R=12 (a .70) |
|---|---|---|---|---|---|---|---|---|---|
| 32B | 0.3 ms | 2L+1 | no | 14.5 | 10.8 | 8.0 | 20.2 | 21.3 | 13.9 |
| 32B | 0.3 ms | 2L+1 | 70% | 14.5 | 13.8 | 9.0 | **27.7** | **35.9** | 26.5 |
| 32B | 0.3 ms | 4L+1 | 70% | 8.9 | 8.9 | 7.8 | 18.5 | 25.8 | 19.9 |
| 32B | 0.5 ms | 2L+1 | 70% | 10.5 | 10.4 | 8.2 | 21.4 | **29.1** | 22.1 |
| 32B | 1.0 ms | 2L+1 | 70% | 6.3 | 6.4 | 6.7 | 13.6 | 19.7 | 15.6 |
| 72B | 0.3 ms | 2L+1 | no | 10.8 | 7.8 | 6.9 | 14.4 | 15.0 | 9.6 |
| 72B | 0.3 ms | 2L+1 | 70% | 10.8 | 10.1 | 8.1 | 20.4 | **26.3** | 19.1 |
| 72B | 0.5 ms | 2L+1 | 70% | 8.0 | 7.8 | 7.3 | 16.0 | 21.6 | 16.2 |
| 72B | 0.3 ms | 4L+1 | 70% | 6.8 | 6.7 | 6.8 | 14.0 | 19.4 | 14.7 |

Reading:

1. **A 0.5B draft model makes 32B/72B slower than plain decoding** on a CPU gateway (8 vs 14.5 tok/s): three
   ~61 ms draft steps per pass (490 MB each at 8 GB/s) cost more than the 2.4 tokens per pass save. This is the
   lead's 7B observation, and it gets worse at R = 8.
2. **Rows are not free on the gateway at 32B/72B**: one extra row of masking work is ~15 ms (32B) / 23 ms (72B)
   on the main core, i.e. as much as 50-80 trips at 0.3 ms. Without spreading the per-row work over spare
   cores (the r keystream, `W r` copies, prepare/finish are per row and independent) even a good head barely
   beats plain decoding (21 vs 14.5). Parallelising the per-row trip work is a precondition for any R > 2.
3. **25-30 tok/s at 32B needs all of**: 0.3-0.5 ms trips, a fused 2L+1 schedule, per-row work on spare cores,
   and a few-MB draft at alpha ~0.8 with R = 8 (29-36 tok/s), or alpha ~0.7 with R = 4 (21-28). 72B tops out at
   ~26 tok/s under the same assumptions.
4. Trees (R = 12) only win where alpha is ~0.7 *and* rows are cheap; with today's per-row cost a chain of 4-8
   is better. The engine caps `spec_rows` at 8; raising it is a one-line change once rows are parallelised.
5. Mask supply (section 4) scales with R as well: R = 8 at B = 320 is ~4 slots per token.

## 6. tau needed vs alpha (independence model)

`tree_spec_sim.py --alpha-grid`: tau of the best chain and the best static tree when the target's token is the
draft's top choice with probability alpha and misses keep the measured rank shape of the 0.5B draft (P(rank 1 |
miss) = 0.54, rank 2 0.19, ...). The model is ~5% optimistic against the measured chain at alpha 0.81.

| alpha | R = 4 chain / tree | R = 8 chain / tree | R = 12 chain / tree | R = 16 chain / tree |
|---|---|---|---|---|
| 0.50 | 1.88 / 2.02 | 1.99 / 2.51 | 2.00 / 2.78 | 2.00 / 2.98 |
| 0.60 | 2.18 / 2.18 | 2.46 / 2.78 | 2.49 / 3.09 | 2.50 / 3.31 |
| 0.70 | 2.53 / 2.53 | 3.14 / 3.22 | 3.29 / 3.61 | 3.32 / 3.88 |
| 0.80 | 2.95 / 2.95 | 4.16 / 4.16 | 4.66 / 4.68 | 4.86 / 5.01 |
| 0.85 | 3.19 / 3.19 | 4.85 / 4.85 | 5.72 / 5.72 | 6.17 / 6.17 |
| 0.90 | 3.44 / 3.44 | 5.70 / 5.70 | 7.18 / 7.18 | 8.15 / 8.15 |

tau = 3-5 needs alpha >= 0.8 with R = 8-12, or alpha ~0.7 with a tree of 12-16 rows. Trees matter only for
alpha <= ~0.7 and R >= 8; below R = 8 the chain is (near) optimal at any alpha >= 0.6.

## 7. Privacy summary (what the provider sees, residuals)

* Per decode pass: exactly R rows per remote linear (one wire item per linear in pad mode), uniform payloads,
  Freivalds-checked; exactly L passes per answer of L tokens (dummy passes after the answer run the tree path on
  filler tokens); mask demand on the public per-pass charge. The topology is public configuration (like R);
  which nodes were real, which were accepted and the tokens are never visible.
* Gateway timing, new code: tree attention work is a function of the topology; `keep_path` makes R - 1 copies;
  R samples per pass; the tree drafter's catch-up step is padded to R rows and its level steps have fixed row
  counts; the feature head runs 1 + levels steps with fixed rows; feature capture is the same every pass.
* Residuals (pre-existing, unchanged): attention cost grows with the context during the answer and is flat
  during dummy passes; take-vs-padding row cost (L3 2.4) now also tracks how many tree nodes were real in a
  credit-limited pass; the chain `BuiltinDrafter`'s catch-up rows (closed in tree mode only).

## 8. What is left

* A feature head trained on real data (EAGLE-scale: ~100k chat turns of target features, a one-off GPU job on
  public weights; the head ships as a public artefact), then a strict-mode tau measurement with
  `FeatureHeadDrafter` (weights from `draft_head_train.py` load directly).
* Parallelise the per-row trip work over spare cores (precondition for R > 2 at 32B/72B), then raise the
  `spec_rows` cap above 8 for trees.
* Gateway settings for `spec_tree` / the head drafter (engine opt-in only so far; defaults off).
* Sampled requests: trees should matter more (lower coupled alpha); not measured (greedy answers only).
