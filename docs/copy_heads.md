# Copying heads as a grounding signal (`experiments/copy_heads.py`)

Prompted by "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026,
https://rikiyat.github.io/blog/weight-space-programming/). That post finds that a frontier model can locate a
network's copying (induction) heads from its weights. It also finds that a sycophant shows itself in where one
head looks, not in its weights.

The idea here: a reply token copied from the evidence comes from where the reader's copying heads point. When
those heads point at a *different* evidence token, the evidence says something else at that spot, such as a
swapped number or name.

## Method

- **Copying heads, found once with no labels.** Score each head on a random token sequence repeated twice: its
  mean attention from each token of the second copy to the token after that token's first occurrence. In
  Qwen3-4B the top 16 heads score 0.54 to 0.98 (results/copy_heads_qwen3_4b.json.heads.json).
- **Per check, one forward pass** (eager attention, the same input the grounding check reads). Before each reply
  token, take the copying heads' attention over the evidence and split it:
  - `match`: attention on evidence positions holding that same token;
  - `conflict`: attention on evidence positions holding a different token.
- **Per answer features:**
  - the strongest conflict;
  - the share of word tokens with no copy support (`uncopied`);
  - a plain lexical baseline: the share of the answer's words that are absent from the evidence.
- **Benchmark:** the same 300 HaluBench answers as `results/halubench_compare_qwen3_4b_seed1.json` (seed 1;
  50 per source, half hallucinated). CPU, about 7 s an answer for this research version with eager attention.

## Results (AUROC for "hallucinated", results/copy_heads_qwen3_4b.json)

| score | all | DROP | FinanceBench | RAGTruth | CovidQA | HaluEval | PubMedQA |
|---|---|---|---|---|---|---|---|
| grounding check (as served) | 0.724 | 0.671 | 0.563 | 0.921 | 0.749 | 0.897 | 0.591 |
| MiniCheck | 0.635 | 0.454 | 0.523 | 0.653 | 0.885 | 0.730 | 0.814 |
| copying-head conflict (max), no training | 0.672 | 0.603 | 0.677 | 0.552 | 0.816 | 0.958 | 0.493 |
| copying-head conflict (mean), no training | 0.614 | 0.578 | **0.749** | 0.144 | 0.746 | 0.862 | 0.592 |
| lexical baseline (words absent from evidence) | 0.609 | 0.514 | 0.480 | 0.776 | 0.800 | 0.903 | 0.444 |
| grounding only, 5-fold cross-validated | 0.717 | 0.678 | 0.529 | 0.918 | 0.719 | 0.891 | 0.587 |
| **grounding + copying heads + lexical, 5-fold cross-validated** | **0.753** | 0.656 | 0.654 | 0.955 | 0.950 | 0.997 | 0.552 |

- **Combined with the grounding check, it helps.** Adding three copying-head and lexical features lifts
  cross-validated AUROC by 3.0 to 4.6 points across five fold shuffles. Paired bootstrap 95% interval:
  +0.5 to +9.2 points.
- **Biggest gains:**
  - FinanceBench, numbers in filings, the weakest source so far: 0.53 to 0.65, and 0.75 for mean conflict alone
    with no training;
  - CovidQA: 0.72 to 0.95.
- **No help on DROP or PubMedQA.**
  - DROP answers are computed (counts and differences), so no copy source exists.
  - PubMedQA answers are yes/no/maybe judgements.
- **The copying-head features catch different errors than the probe.** Alone they reach 0.67, below the grounding
  check, and the early 0.78 on the first 90 answers did not hold.
- **The copying-head signal beats the lexical baseline** (0.67 against 0.61). Where the heads point carries more
  than whether a word appears in the evidence at all.

## Caveats

- 50 answers per source: per-source numbers are noisy.
- The combination's weights were fitted on labels from this sample (cross-validated). A served version needs them
  fixed from other data (the calibration step can carry them).
- Eager attention costs time. A served version would read only the copying heads' attention, through a hook on
  those layers.

## Next

- Serve it: read the 16 copying heads' attention to the evidence in the grounding check's existing pass, and add
  `conflict` and `uncopied` to the sentence score.
- Re-test on the seed-0 sample and on RAGTruth's full test set (the Kaggle kernel).
