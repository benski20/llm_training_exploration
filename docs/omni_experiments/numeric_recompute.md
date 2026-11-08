# Independent recompute tier for numeric answers (DROP, FinanceBench, ProcessBench)

Script: `kaggle/omni/numeric_recompute.py`. Kernel: `omni-numrec` (2x T4, fp16). Raw outputs: `results/kaggle/omni-numrec/`.
Summary with the reader combination: `results/omni/numeric_recompute.json`.

## Why

HaluBench DROP and FinanceBench were the verifier's weakest grounded domains. Every reader size (1.7B-14B) and NLI
scored AUROC 0.5-0.65 there. The hand-written "numbers appear in passage" checker scored 0.48 / 0.54. The answers are
computed metrics, and the wrong ones are plausible near-misses. Reading the reply does not detect them, but
re-deriving the answer does.

## Method

1. **Write programs.** Qwen3-8B (non-thinking, temperature 0.7, top-p 0.8) sees the passage and the question but
   never the stated answer. It writes a short Python program: copy the numbers from the passage, compute, and assign
   `answer`. There are k=5 samples per item.
2. **Run them in a sandbox.**
   - An AST whitelist: no import but `math`; no dunder names, attributes or strings; no eval / exec / open / getattr
     / type / ...
   - Restricted builtins, with `print` made a no-op.
   - A separate `python -I -S` subprocess with an empty environment, a temp cwd, and rlimits (CPU 5 s, 1 GB address
     space, no file writes), plus a 10 s timeout.
   - `unshare -n` was unavailable on Kaggle, so network isolation rests on the no-import whitelist.
   - Outcomes of the 6,400 programs: 97% ran; 54 rejected; 2 timeouts.
3. **Compare with the stated answer.**
   - Numbers must agree within the stated answer's own rounding: half a unit of its last decimal, so "$2135.00"
     against 2134 disagrees.
   - Scale variants are also accepted: ×100 for a fraction vs a percent, and 1e3 / 1e6 / 1e9 for units.
   - Sign is ignored, and accounting "(12)" counts as negative.
   - Text answers are compared by token F1 ≥ 0.5 or containment.
   - For math, the stated answer is the final `\boxed{}`, else the last number, compared at 1e-4 relative. Symbolic
     or multi-part answers are not compared and get a neutral score.
   - **The tolerance was chosen after the run.** The kernel run used a looser tolerance as well: max(half a unit,
     0.05% relative). It cost FinanceBench about 0.05-0.11 AUROC, because HaluBench's wrong FinanceBench answers are
     often off by one unit (2134 vs 2135). The stored recomputed answers were re-scored offline with rounding only.
     Rounding only was the best rule on the corpus rows, and the fresh rows confirm it. The script now uses it.
     Both reports are in the JSON.
4. **Score.**
   - `disagree`: the fraction of valid samples that disagree with the stated answer.
   - `maj`: the majority verdict weighted by the majority's share (self-consistency).
   - `flag`: majority disagrees and at least 60% of the samples agree on one value.

Corpus rows were rebuilt with `omni_corpus.halubench(200)` / `processbench(240)` and joined to corpus ids by md5 (all
880 matched). The fresh sample has 200 balanced rows per domain from HaluBench that are not in the corpus (seed 11).

## Results (AUROC; flag precision / recall; k=5 unless noted)

| set / domain | n | reader q8b (LODO) | recompute k=1 | recompute k=5 (`maj`) | flag rate | P | R | reader + recompute (5-fold LR) |
|---|---|---|---|---|---|---|---|---|
| corpus DROP | 200 | 0.56 | 0.78 | **0.82** | 0.44 | 0.83 | 0.73 | 0.83 |
| corpus FinanceBench | 200 | 0.63 | 0.85 | **0.88** | 0.60 | 0.79 | 0.94 | 0.91 |
| fresh DROP | 200 | n/a | 0.75 | **0.79** | 0.44 | 0.81 | 0.70 | n/a |
| fresh FinanceBench | 200 | n/a | 0.81 | **0.88** | 0.55 | 0.81 | 0.88 | n/a |
| ProcessBench GSM8K | 240 | 0.91 | 0.93 | 0.94 | 0.48 | 0.94 | 0.91 | 0.96 (rank average 0.97) |
| ProcessBench MATH (85% coverage) | 240 | 0.93 | 0.87 | 0.88 | 0.38 | 0.83 | 0.63 | 0.97 |

- **Numeric-answer subsets** (DROP 150 / 134 rows; FinanceBench 189 / 196 rows): 0.80 / 0.78 (DROP) and 0.90 / 0.87
  (FinanceBench).
- **Sampling:** k=3 is within about 0.01 of k=5. k=1 loses 0.03-0.06.
- **Combining with the reader** (`results/omni/expert_scores_q8b.npz`, LODO reply score; corpus rows only):
  - On DROP and FinanceBench, the zero-label rank average does not beat recompute alone (0.78 / 0.86), because the
    reader is near chance there.
  - A 5-fold in-domain logistic regression adds about 0.01-0.02.
  - On math the two are complementary: 0.97 on both, against 0.88-0.94 for either alone.

### Cost

- **Measured:** 41.8 T4-seconds per check at k=5, with 2 T4s held (vLLM 0.11.0, TP=2, fp16). That is about $4 per
  1,000 checks at $0.35 per T4-hour.
  - Prompts average 626 tokens, and each sample generates 152 tokens.
  - On T4, vLLM falls back to FlexAttention, which is slow. The HF `generate` smoke test was faster (23 T4-s per check
    at k=3).
  - An A10 / L4 with FlashAttention, or k=3, should cut this several-fold.
- **Sandbox:** about 0.08 CPU-seconds per program.
- **Comparison:** about 20x the 8B reader ($0.19 per 1,000 on 2x T4). So this is an escalation tier for numeric questions, not a default.

### Caveats

- **Tolerance chosen after the run.** It was set on the corpus rows after the run (see Method), and the smoke test
  also saw 16 of the fresh rows. The fresh FinanceBench result (0.88) agrees with the corpus result.
- **DROP misses.**
  - Span and multi-answer questions, where the program picks a different but valid span.
  - Questions where the recompute shares the stated answer's misreading.
- **FinanceBench false flags.**
  - A misread table row or year.
  - The model's sign or definition conventions (for example, a change computed as a negative).
  - Free-text answers compared against a "yes" or a sentence.

## Verdict

The recompute tier fixes the gap:
- DROP: 0.56 → 0.79-0.82.
- FinanceBench: 0.63 → 0.88 on both the corpus and fresh rows.
- The flag runs at about 80% precision and catches 70% (DROP) to 88-94% (FinanceBench) of wrong answers.

Ship it as an escalation tier for questions with numeric answers over documents. Run it at k=3 to halve the cost, and
on faster GPUs than T4. On math, combine it with the reader: 0.97.
