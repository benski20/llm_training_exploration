# L3: strict speculation without compromising privacy or multiplying mask cost

Date: 2026-11-22. Scope: make strict speculation (`spec_rows` R + `spec_strict_passes`) usable: today it is exact and
hides the acceptance from the provider, but every decode pass (dummy passes included) takes R masks, so it costs R
masks per generated token, and on Modal the mask supply, not the round trips, limits a sustained chat
(`deploy/modal/README.md`: 1.5B at R = 4 is about 0.25 generated tokens per second of encrypted-factory supply).
Rules: outputs bit-identical to the plaintext reference (greedy and seeded sampling); the provider sees only masked
data whose sizes, counts and timing schedule depend on public information; Freivalds and range checks on every real
row; new modes default off. CPU only (4 shared vCPUs, load 25-30 during this work); GPU runs are the lead's.

Implementation in this branch: `Generator(spec_mask_rows=C, spec_mask_credit=B)` (engine), the fixed-structure
padding path in `MaskedBackend`, `SequenceInput.pad_rows` / `fixed_item` in `TransformerModel`, the public mode of
`WindowedReservation`, gateway settings `spec_mask_rows` / `spec_mask_credit` (SPEC 9.4a), Modal `--spec-mask`.
Tests: `tests/engine/test_spec_no_compromise.py` (new), `test_padding_rows.py`, `test_speculative.py`,
`tests/deploy/test_modal_{boot,launch}.py`. Benches: `benchmarks/spec_charge_sim.py`,
`benchmarks/spec_pad_timing_bench.py`, `benchmarks/spec_no_compromise_bench.py`.

## 0. Summary

| idea | verdict | why |
|---|---|---|
| (a) dummy rows without factory masks | **accepted, implemented** (`spec_mask_rows`) | A padding row is the low K bits of a fresh ChaCha20 stream, exactly what a mask's `r` is, so `x_hat` has the same distribution as a real row's. But four other distinguishers had to be closed first (below): item splits at mask-extent boundaries, unverified dummy items, a gateway turnaround that depended on the real/padding split, and on-demand mask-window extension. |
| (b) public mask consumption | **accepted, implemented**: a public per-pass charge `prompt + j + min(B, floor((C - 1) j))` | The store is asked for slots only on that schedule (pass index j, public constants C and B), never on use; parts are not retired before the request ends. Mask demand is then a function of the prompt length and the pass count, which strict mode makes equal to the generated length L. |
| (b') constant-rate factory | rejected for now | Decouples demand from use completely, but stock exhaustion and "factory paused, store full" events still leak aggregate use, it must be provisioned for the peak, and it needs factory changes (another group's area). The per-pass charge gets most of the saving without touching the factory. |
| (c) hiding the dummy passes | partly: the answer streams first (as before); dummies cannot be made faster or overlapped without a leak | Dummy passes must look like real passes on the wire and in time. They fit in the user's reading time whenever a pass takes less than about 0.3 s at 2.84 tokens per pass (section 4). Merging them into other sessions' passes already happens (continuous batching) and saves provider trips, not latency. |
| (d) better drafting for small targets | no exact gain found for the 0.5B; the 1.5B with a 0.5B draft is the case that pays | Prompt lookup tops out at 1.13 tokens per pass for the 0.5B (n down to 2); a 2-candidate tree is no better (upper bound). A 0.5B draft gives the 1.5B 2.84 tokens per pass at R = 4. A confidence-gated draft length (spend credit only on likely drafts) is the next lever. |

**Result (1.5B target, 0.5B draft, 24 chat prompts, greedy answers of up to 128 tokens; simulation of the
generator's rules on exact target answers, `spec_charge_sim.py`):** R = 4 with C = 4, B = 160 gives **the same 2.84
tokens per pass as today's R = 4 at 2.50 masks per generated token instead of 3.96 (-37%)**; R = 5, C = 5, B = 160
gives 3.00 tokens per pass at 2.56 masks per token (today's R = 5: 3.21 at 4.95). For answers longer than these
(L = 512), the same B costs about 1 + 160/L = 1.3 masks per token, but the speed-up then covers only the first
B / (C - 1), about 53, passes (about 150 tokens) and falls back toward 1 token per pass after that. On the 0.5B (prompt lookup only) C = 1.5 keeps
all of today's small gain (1.08 tokens per pass at R = 4) at 1.48 masks per token instead of 3.96.

## 1. Background: what strict speculation did before this change

`Generator(spec_rows=R, spec_strict=True)`: every decode pass carries exactly R rows per sequence (pending token,
drafts, random filler tokens), and after the answer, dummy passes on random tokens run until the provider has seen
exactly L passes (L = generated tokens, which it sees anyway without speculation). Filler rows and dummy passes are
ordinary masked rows, so the request takes `prompt + 1 + R (L - 1)` masks whatever the text: R per token. The
backend already had padding rows (`PAD_ROWS`: uniform random payload rows, no mask; `DUMMY`: all-padding calls), and
the generator had a pad mode that drafted from "spare" reservation rows, but no model enabled it (STATUS: "the model
does not use the backend's mask-free padding rows yet"), and as written it would have leaked (below).

## 2. (a) Padding rows instead of masked filler rows: every distinguisher checked

What the provider receives per decode pass: one `ComputeRequest` per remote linear group with a random request id,
the fingerprint, ring name, deadline, and one item per call: linear key, row count, `rows x n x K/8` payload bytes.
It returns `y_hat = W x_hat mod 2^K` per item. It also sees when each request arrives (the gateway's turnaround),
and, indirectly, the mask factory's demand (encrypted maskgen requests to the provider). Checked one by one:

1. **Payload distribution.** A real row is `x_hat = xq + r mod 2^K` with `r` the low K bits of a ChaCha20 stream
   (`SeedExpander.ring_uniform_rows`). On the capped ring K32A12C27 the cap bounds the *output* (`|W8 xq| <= 2^27`,
   enforced by choosing each token's activation bits); the mask is uniform over all of `Z_{2^32}`, so `x_hat` is
   uniform over `Z_{2^32}^n` for any `xq` (one-time pad, up to the PRF). A padding row now uses `r` from a ChaCha20
   stream under a fresh 32-byte OS-random key, read by the same function (`MaskedBackend._random_r`), with zero
   activations: the same distribution as a real row, not just "uniform". (Before: `os.urandom & mask`, also uniform,
   but 5x slower than a store mask for wide linears: a timing difference, see 4.)
2. **Row counts and item layout.** Rows per item are R (public). **Leak found:** a padded call whose real rows
   spanned two mask extents (segment ends, window boundaries) went out as two items `[a, b + pad]`, and `a` is the
   rows left in the extent, i.e. a function of the masks used so far, which with padding depends on the drafts.
   Fixed twice over: the backend sends every padded call as one item (`groups` / fixed path), and the generator
   keeps a pass's real rows inside one extent (`rows_until_boundary`), so one take and one set of check vectors.
   Pipelined blocks (`_split_blocks`) now split a padded call by its wire rows (real + padding), so block sizes do
   not depend on the split either.
3. **Verification and aborts (active provider).** Padding rows are Freivalds-checked against the item's check
   vectors (they need no mask: `u . y_hat == (uW) . x_hat` holds for any `x_hat`). **Leak found:** dummy calls (all
   padding) were *not* checked. A provider that corrupts one reply learns from whether the session aborts whether that
   pass was a dummy, i.e. a bound on the number of real passes P, hence the acceptance. Fixed: dummy items are
   checked against the reservation's current vectors at the normal `freivalds_rate`, through the same code. Range
   checks need `W r`, so they run on real rows only; a corruption that passes Freivalds (probability <= 2^-s,
   s >= `min_soundness_bits`) might then trip the range check on a real row and not on a padding row: a
   distinguishing advantage of at most 2^-s per corrupted reply, and each detected corruption trips the breaker.
4. **Gateway turnaround (timing).** **Leak found:** the padded path prepared real rows through the normal path and
   padding rows through a separate filler kernel, and finished them separately, so the work per call depended on
   the split. Measured per remote linear call of the 0.5B (prepare + finish, median, 1 thread, this box at load
   25-30; the masks come from a reservation that expands `r` with the store's ChaCha20 reader and copies
   precomputed `W r`, i.e. a take's work without disk or ring I/O):

   | R = 4, per call | R real rows | 1 real + 3 padding | dummy (all padding) |
   |---|---|---|---|
   | gate_up, existing padded path | 529 us | **654 us** | **466 us** |
   | down, existing padded path | 481 us | **591 us** | **401 us** |
   | gate_up, fixed path (this branch) | 627 us | 630 us | 609 us |
   | down, fixed path (this branch) | 445 us | 443 us | 442 us |

   (Existing path: `spec_pad_timing_bench.py`, 30 interleaved reps. Fixed path: `_fixed_item` + `_finish_fixed`
   alone, 300 interleaved reps; the whole-call bench with the in-process provider gives the same picture within
   its +-30 us noise: down 466 / 484 / 431, gate_up 561 / 581 / 526.) With 48 remote calls per pass the old
   100-200 us differences were a 5-10 ms per-pass signal, enough to tell dummy passes from real ones over a long
   answer. **Fixed** by a fixed-structure path for padded and dummy calls (`MaskedBackend._prepare_fixed` /
   `_finish_fixed`): one `trip_prepare_into` over all R rows (padding rows: zero activations, fresh ChaCha `r`),
   one Freivalds pass over all R rows, one finish of the real rows and one of the padding rows (together R rows,
   the padding verdict dropped); padding rows also copy and wipe a `W r`-sized row like a take. A dummy call
   takes ONE mask row from the store (zero activations, result discarded; review fix): the public schedule always
   leaves that slot (every pass charges at least one and dummy passes come after the real ones), so it costs the
   factory nothing, and every call, real or dummy, makes the same store take (the earlier one-row ballast left a
   dummy call 18 us / 3% cheaper per call, about 0.9 ms per pass, enough to count real passes over many
   answers). In pad mode every decode call of the session takes this path, 1-row passes included, so real and
   dummy passes run the same code. Residual: the per-row cost of a store take versus a padding row (1 real + 3
   padding vs 4 real: padding rows about 3% slower per call here, ~12 us of ~420 us on the 0.5B down linear; dummy vs 1 real + 3 padding within 0.5%), which tracks how many drafts a pass carries; a real store's take
   (decrypting `W r` from a segment or copying it out of the mask ring) may cost more per real row than this
   bench's reservation, so it must be re-measured with the real store (the GPU runs in section 7 report
   per-trip timings). The path needs the native trip kernels
   (`supports_fixed_pad`); without them pad mode is refused.
5. **Mask store behaviour.** `WindowedReservation` extended on demand (`ensure`) and reserved the next window in the
   background once few rows were left, and retired exhausted parts at step starts: all three track the masks
   *used*. With an in-memory mask ring a retired part frees slabs, which the factory refills: the timing of the
   factory's requests would follow use. **Fixed** by the public mode (section 3).
6. **Noise monitor.** Padding payloads go through `guard_packed` like real payloads (uniform either way).
7. **Not changed by this work (pre-existing strict-mode residuals, also with masked fillers):** the gateway's
   attention cost grows with the context during the answer (it advances by the accepted tokens per pass) and stays
   flat during the dummy passes (the cache is rolled back), so per-pass gateway time has a slope during the answer
   and none after; and the drafter's catch-up step feeds a data-dependent number of rows (documented in
   `BuiltinDrafter`). Both are timing signals of the order of attention cost per token (about 5 us per token of
   context per pass for the 0.5B). A constant-cost attention (attend over the reserved length with masking) would
   close the first; listed in section 8.

Exactness: real rows are prepared and finished by the same kernels (`trip_prepare_into` / `trip_finish_into`) with
the same masks, scales and bounds as the normal path (both are row-wise), padding rows come after the real rows and
the verify rows are computed with `serial_tail` (row j attends only to rows before it), so the real rows' logits do
not depend on what the padding rows hold. Tested bit for bit against the plaintext reference, greedy and seeded
sampling, penalties and stop strings, R in {2, 4}, C in {2, R}, four draft sources
(`test_padded_strict_is_exact_and_fixed`: 48 cases), fractional C and credit caps, a mask extent ending mid-pass.

## 3. (b) Mask consumption that depends only on public quantities

Padding alone makes the store hand out `prompt + (real rows)` masks, where real rows = passes + drafts sent: a
function of the drafts. Since a request's reservation is burned when it closes, consumption from the store is what
the reservation *asked for*, not what it used; the leak is in when and how much it asks. Options:

* **B1, public per-pass charge (implemented).** Before decode pass j (dummy passes included) the session holds
  `target(j) = prompt + j + min(B, floor((C - 1) j))` slots; `charge_to(target)` extends a windowed reservation by
  whole windows (`max(window, target - reserved)`), never on demand, and `ensure()` fails closed if a step would
  need more (it cannot: the generator limits drafts to `target - used - 1`). With a fixed up-front reservation
  (`mask_window_tokens <= 0`) the request reserves `target(max_tokens - 1)` at once. Either way, what is asked for,
  when, and what is burned are functions of (prompt length, pass index, C, B), and strict mode makes the pass count
  L. Parts are retired only at close in this mode (they hold at most the request's own charge).
  `C` may be fractional (exact `Fraction` arithmetic); `B` caps the total draft credit, which front-loads the budget
  into the answer's passes: dummy passes come after the answer and need no credit.
* **B2, constant-rate factory.** Rejected for now (section 0).
* **B3, one lump per request.** Is B1 with windows off.

Masks per generated token (store demand), tokens per real pass (tau), real masked rows per token (private), from
`spec_charge_sim.py` on the Kaggle run 28 exact greedy answers (24 prompts, up to 128 tokens; the 1.5B's draft
matches recomputed here with the 0.5B pack on the CPU, teacher-forced; "filler" = today):

1.5B target, 0.5B draft (+ lookup):

| config | tau | store masks / token | real rows / token | dummy share |
|---|---|---|---|---|
| no speculation | 1.00 | 1.00 | 1.00 | 0 |
| filler R = 2 (today) | 1.78 | 1.98 | 1.98 | 0.44 |
| filler R = 4 (today) | 2.84 | 3.96 | 3.96 | 0.65 |
| filler R = 5 (today) | 3.21 | 4.95 | 4.95 | 0.69 |
| filler R = 8 (today) | 3.94 | 7.91 | 7.91 | 0.75 |
| pad R = 4, C = 2 | 1.78 | 1.98 | 1.09 | 0.44 |
| pad R = 4, C = 3 | 2.40 | 2.97 | 1.21 | 0.58 |
| pad R = 4, C = 4, B = 64 | 1.62 | 1.67 | 1.19 | 0.38 |
| pad R = 4, C = 3, B = 96 | 2.22 | 1.91 | 1.19 | 0.55 |
| pad R = 4, C = 4, B = 128 | 2.62 | 2.24 | 1.33 | 0.62 |
| **pad R = 4, C = 4, B = 160** | **2.84** | **2.50** | 1.35 | 0.65 |
| pad R = 5, C = 5, B = 160 | 3.00 | 2.56 | 1.46 | 0.67 |
| pad R = 8, C = 8, B = 128 | 2.12 | 2.35 | 1.56 | 0.53 |

0.5B target, prompt lookup (n 4..3):

| config | tau | store masks / token | real rows / token |
|---|---|---|---|
| filler R = 4 / R = 8 (today) | 1.08 / 1.10 | 3.96 / 7.92 | same |
| pad R = 4, C = 1.25 | 1.07 | 1.23 | 1.06 |
| pad R = 4, C = 1.5 | 1.08 | 1.48 | 1.08 |
| pad R = 8, C = 1.5 | 1.09 | 1.48 | 1.16 |

Reading: a draft model always proposes R - 1 tokens, so without a cap the credit is spent evenly and tau(C) equals
today's tau(R = C): C alone buys nothing over a smaller R. The cap is what pays: the answer's passes come first, so
a high C with a cap B gives the answer today's tokens per pass while the total charge stays near `L + B`. B = 160 is
about right for 100-token answers; answers that outgrow it fall back to tau ~1 after about B / (C - 1) passes. With
prompt lookup (mostly misses) C barely matters: 1.25-1.5 keeps the whole gain.

Remaining gap: the real rows per token (1.35 at the recommended point) against 2.50 charged: the credit charged
during the dummy passes is burned. That is the price of a schedule that cannot know when the answer ends.

## 4. (c) Latency: what the user sees, and the dummy passes

* The answer streams as before: user-visible speed is tau / pass time, unchanged by padding (the same R rows of
  gateway work per pass as masked fillers; the fixed path costs within noise of the normal path, section 2.4).
* The dummy passes run after the answer: (L - P) passes, P = L / tau, i.e. a share `1 - 1/tau` of the answer's
  passes (0.65 at tau = 2.84). A follow-up turn of the same conversation waits for them.
* Making dummies cheaper or faster: **rejected**. They must look like real passes in time as well as on the wire;
  a sleep standing in for compute would have a different time distribution.
* Starting the follow-up's prefill while dummies still run: **rejected**. The provider would see the new turn
  overlap the old sequence's passes and learn that the user replied before L passes, a bound on P.
* Merging dummies with other sessions' passes: already done by continuous batching (a session owing dummies is
  batched with others), which saves trips for the provider but not latency for the owner. Side effect to know:
  with `spec_max_batch = 1` a step of two sequences runs 1-row passes, so a draining session stops the other
  session's speculation while it drains (public, because the step size is public; throughput, not privacy).
* Hiding in reading time: a person reading the answer at 4-6 tokens per second needs about 0.2 s per token. The
  drain takes `(1 - 1/tau)` pass times per token, so it is over before the person has read the answer whenever the
  pass time is under about 0.2 / (1 - 1/tau) = 0.3 s at tau = 2.84. Modal co-located today is about 0.2 s per pass
  (0.5B); a 1.5B pass at 2 ms trips was estimated at 0.3-0.5 s. Programmatic clients that fire the next request at
  once do wait (about 0.65 x L pass times).

CPU measurements (masked path, in-process provider, the 0.5B, lookup-friendly prompts): section 6.

## 5. (d) Acceptance

* Prompt lookup tuning on the 0.5B's exact answers (R = 4 / 8, filler; same with pad C = 1.5 within 0.01):
  n 4..3: 1.08 / 1.10; n 4..2: 1.13 / 1.14; n 6..2: same as 4..2. min_n = 2 is worth about 4% for the 0.5B
  (it is a setting already: `spec_lookup_min_n`).
* Multi-candidate tree (two lookup continuations sharing the R - 1 draft rows; upper bound with perfect branch
  choice): 1.12 / 1.14 at R = 4 / 8: no better than the chain. Tree attention would need a non-chain mask in
  `serial_tail` and KV compaction; not worth it for lookup.
* For a target with a smaller sibling (1.5B with the 0.5B) the draft model is what pays: 2.84 tokens per pass at
  R = 4. Its credit is spent on drafts that are rejected about 19% of the time per token, compounding along the
  chain; a draft length gated by the draft's own confidence (private: it changes only which of the R rows are real)
  would raise accepted tokens per charged mask. Not implemented here (follow-up), exactness is unaffected because
  every emitted token is still the target's own sample.
* Cross-request n-gram caches: only per conversation (the prompt already holds the history). A cache across users of
  one gateway would let one user's answer speed depend on another's text (a user-visible side channel inside the
  customer); not done.
* For the 0.5B itself no exact source found gives more than about 1.13 tokens per pass. The levers there are
  elsewhere (local linears, R1).

## 6. Measurements on this CPU box

`benchmarks/spec_no_compromise_bench.py`, measured 2026-11-22: the real `Generator` in strict mode over
`MaskedBackend` with an in-process provider (`LocalComputeClient`: the provider's ring GEMMs on the same CPU),
Qwen2.5-0.5B pack, K32A12C27, `local_linears = attention`, prompt lookup (n 4..3), greedy, two short chat prompts
(haiku, planets), 32 tokens each. The box was shared with other agents (load 25-30, a 4 GB memory cgroup that
killed several runs), so each configuration ran in its own process and tokens/s varied by 30% between identical
runs (plain: 2.35 and 3.15 tok/s). Counts are exact; tokens/s are indicative only.

| config | tokens / real pass | store masks / token | real masked rows / token | provider passes | answer tok/s | drain after the answer |
|---|---|---|---|---|---|---|
| plain | 1.00 | 1.00 | 0.97 | 64 | 2.35-3.15 | 0 |
| filler R = 2 (today) | 1.10 | 1.94 | 1.94 | 64 | 1.90 | 3.1 s |
| pad R = 2, C = 1.5 | 1.10 | 1.44 | 1.02 | 64 | 2.07 | 3.5 s |
| filler R = 4 (today) | 1.16 | 3.88 | 3.88 | 64 | 2.22 | 4.8 s |
| pad R = 4, C = 1.5 | 1.14 | 1.44 | 1.02 | 64 | 1.56 (\*) | 5.2 s |
| filler R = 8 (today) | 1.21 | 7.75 | 7.75 | 64 | 1.60 | 6.8 s |
| pad R = 8, C = 1.5 | 1.16 | 1.44 | 1.09 | 64 | 1.93-2.20 | 4.2-5.3 s |

Answers were identical to plain decoding in every run checked against the plain tokens (the pad R = 4 run's
reference process was OOM-killed, so its identity is covered by the tests instead). (\*) a single run on the
loaded box. On this CPU the provider's GEMMs run next to the gateway, so R rows cost real CPU time and lookup-only
speculation does not pay on the 0.5B (as section 5 says); the counts are what carries over: same provider passes
and rows per pass as today, the masks per token down from R to 1.44 (C = 1.5 over 31 decode passes), and the real
masked rows the drafts used at 1.02-1.09 per token. The tokens per real pass of pad R = 4 and 8 are a little below
filler (1.14 / 1.16 vs 1.16 / 1.21): the credit (0.5 per pass) occasionally cuts a lookup proposal short.

GPU-provider estimate for user-visible speed (not measured): a pass costs `trips x trip time + gateway work`; the
padded pass has the same trips and the same gateway work per call as a filler pass of the same R (section 2.4), so
answer tokens/s are today's strict-speculation numbers (Modal README table: 1.5B, R = 4, 0.5B draft: about 2.1 tok/s
at 17 ms trips, 5.5 at 2 ms) at 2.5 instead of 3.96 masks per token; on the mask-supply-bound Modal tunnel (about
1 slot/s for the 1.5B) the sustained chat rate rises from about 0.25 to about 0.4 generated tokens per second.

## 7. GPU measurement to run

Not run here (no Modal or Kaggle from this branch). Exact commands for the lead, from the repo root, after merge:

```bash
# 1. Baseline: today's strict speculation, 1.5B target, 0.5B draft on the customer CPU, co-located T4 (about 25 min)
python pgw/deploy/modal/launch.py --gpu T4 --placement colocated --model qwen2.5-1.5b-instruct --spec-draft \
    --spec-rows 4 --e2e --ttl-min 40
# 2. The same with the public mask charge C = 4, credit cap B = 160 (deploy time; about 25 min)
python pgw/deploy/modal/launch.py --gpu T4 --placement colocated --model qwen2.5-1.5b-instruct --spec-draft \
    --spec-rows 4 --spec-mask 4/160 --e2e --ttl-min 40
# 3. 0.5B demo with prompt lookup: filler R = 4 vs pad C = 1.5 (about 15 min each)
python pgw/deploy/modal/launch.py --gpu T4 --placement colocated --spec-rows 4 --e2e --ttl-min 30
python pgw/deploy/modal/launch.py --gpu T4 --placement colocated --spec-rows 4 --spec-mask 1.5 --e2e --ttl-min 30
```

Expected (estimates): identical answers to the plaintext reference in all four (the e2e comparison); tokens per
pass within a few percent of each other in 1 vs 2 and 3 vs 4; masks taken per generated token (the e2e's
`masks_consumed` over the chat's tokens, and the store's reserved slots) 3.96 -> about 2.5 (1.5B) and 3.96 -> about
1.5 (0.5B); per-trip latency unchanged (the fixed path costs within noise of the normal path); drain time after the
answer unchanged (same pass count). Watch `pad_rows` and `dummy_calls` in `/admin/status` backend stats and
`integrity_failures` = 0. The e2e's `e2e_min_masks` still waits for R slots per token (conservative).

## 8. What is left

* GPU/Modal measurement (section 7).
* A confidence-gated draft length to spend the credit on likely drafts (section 5).
* Constant-cost attention for strict mode (attend over the reserved length) to remove the context-slope timing
  signal that strict speculation has with or without padding (section 2.7).
* B2 (constant-rate factory) if the mask supply stays the binding constraint after B1; it needs a factory change.
* `e2e_min_masks` in the Modal driver could use the charge schedule instead of R per token.
* Pad mode needs the native trip kernels; without them `spec_mask_rows` is refused (masked fillers still work).

## Sources

* Internal: `deploy/modal/README.md` (strict speculation section, mask supply), `research/plans/fewer-round-trips.md`
  (section 4), Kaggle run 28/29 results in `benchmarks/results/kaggle/run28_spec_accept`, `run29_spec_accept_cpu`.
* Leviathan et al., Fast Inference from Transformers via Speculative Decoding (ICML'23) — https://arxiv.org/abs/2211.17192
* Saxena, Prompt Lookup Decoding (2023) — https://github.com/apoorvumang/prompt-lookup-decoding
