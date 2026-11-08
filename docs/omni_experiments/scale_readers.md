# Reader scaling on GPU: Qwen3 1.7B, 4B, 8B and 14B (4-bit) on the full corpus

**Question.** How does omni-verifier accuracy grow with the frozen reader, on the full corpus? Before this run, 4B had been
measured on a 1,800-reply subset only, because CPU extraction took 5-8 s a reply.

**Setup.**
- Kaggle kernel, 2× T4 16 GB (`kaggle/omni/scale_readers.py`, run with `kaggle/kgrun.py`).
- Corpus: built in the kernel with `omni_corpus.py build`. 5,107 replies; its md5 matches the repo's `corpus.jsonl`.
- Features: `omni_corpus.py extract`, i.e. `vt.omni.Reader.features`, unchanged.
  - The kernel only changes how the model loads: fp16, sdpa attention, `device_map="auto"`.
  - This is a monkeypatch of `from_pretrained` inside the kernel script. No repo file was edited.
  - 1.7B and 4B ran one worker per T4. 8B spans both T4s.
- Evaluation: `experiments/omni_final.py` (LODO over 21 domains). Experts: reply head, worst unit, judge, and exact arithmetic.
  - There is no evidence/NLI expert. The comparison row is therefore the repo's `final_q17_noev`, 0.728; with evidence it is 0.727.
- Heads: `vt-omni train --parts mean judge`.

**fp16 is safe for Qwen3 on T4.**
- **No overflow.** Across all 5,107 replies there were no non-finite layer outputs and no non-finite features.
  - The largest residual activation: 14.9k (1.7B), 16.4k (4B), 22.7k (8B). The fp16 limit is 65.5k.
- **fp16 matches fp32** (smoke test: reference replies plus the 6 longest):
  - mean-state cosine ≥ 0.99999 at every layer;
  - |Δ judge log-odds| ≤ 0.12;
  - |Δ mean log-prob| ≤ 0.014.
- **GPU matches CPU.** The GPU fp16 features match the repo's CPU bf16 features within bf16 noise.
- **The 1.7B result reproduces exactly.** The GPU 1.7B run gives 0.728 zero-label, the same as the repo's CPU run.

## Results (held-out domains, AUROC for "wrong")

Results files: `results/omni/final_q4b_full.json`, `final_q8b.json` and `final_q14b_nf4.json`. The 1.7B numbers are the repo's CPU run
(`final_q17_noev`); the GPU rerun is within ±0.01 of it, and its outputs are in `results/kaggle/omni-scale-readers/`.

| format | reader | zero labels | reader's own judge | 25 labels | 50 labels | 100 labels | guarantee, 100 uniform labels: false alarms / caught |
|---|---|---|---|---|---|---|---|
| **all 21** | 1.7B | 0.728 | 0.639 | 0.736 | 0.744 | 0.742 | 5.2% / 28% |
| | 4B | 0.778 | 0.719 | 0.793 | 0.798 | 0.793 | 4.9% / 40% |
| | **8B** | **0.805** | **0.779** | **0.814** | **0.816** | **0.815** | **4.9% / 47%** |
| | 14B (nf4) | 0.820 | 0.804 | 0.838 | 0.842 | 0.840 | 4.7% / 49% |
| reasoning (4) | 1.7B | 0.849 | 0.741 | 0.847 | 0.850 | 0.852 | 5.3% / 45% |
| | 4B | 0.885 | 0.825 | 0.885 | 0.886 | 0.886 | 4.2% / 56% |
| | 8B | 0.906 | 0.884 | 0.905 | 0.904 | 0.904 | 5.7% / 59% |
| | 14B (nf4) | 0.906 | 0.886 | 0.904 | 0.905 | 0.904 | 5.4% / 60% |
| long-form (5) | 1.7B | 0.751 | 0.637 | 0.765 | 0.770 | 0.727 | 5.5% / 37% |
| | 4B | 0.794 | 0.678 | 0.810 | 0.815 | 0.737 | 4.9% / 41% |
| | 8B | 0.839 | 0.801 | 0.837 | 0.836 | 0.796 | 4.7% / 55% |
| | 14B (nf4) | 0.855 | 0.838 | 0.849 | 0.852 | 0.799 | 4.6% / 60% |
| RAG (10) | 1.7B | 0.691 | 0.618 | 0.706 | 0.718 | 0.727 | 5.0% / 19% |
| | 4B | 0.746 | 0.695 | 0.767 | 0.772 | 0.783 | 5.2% / 38% |
| | 8B | 0.767 | 0.732 | 0.783 | 0.788 | 0.798 | 4.6% / 44% |
| | 14B (nf4) | 0.790 | 0.764 | 0.829 | 0.833 | 0.842 | 4.4% / 47% |
| closed-book chat (2) | 1.7B | 0.613 | 0.538 | 0.591 | 0.594 | 0.615 | 5.2% / 11% |
| | 4B | 0.679 | 0.729 | 0.695 | 0.707 | 0.711 | 5.2% / 12% |
| | 8B | 0.707 | 0.750 | 0.731 | 0.730 | 0.740 | 4.7% / 20% |
| | 14B (nf4) | 0.714 | 0.754 | 0.720 | 0.738 | 0.743 | 4.2% / 22% |

**How to read the table.**
- The label columns are k balanced labels, with the expert weights refitted (omni_final's adaptation).
- The guarantee targets ≤ 10% false alarms at 95% confidence, at 20% prevalence. It covers 14 domains with enough replies.
  - Its false alarms stay at about 5% for every reader.
  - The share of draws over 12% is 3.6-5.7% overall. For 8B on reasoning it is 11%.

**What scaling buys.**
- **The judge improves the most.** 0.64 → 0.72 → 0.78, so at 8B the reader's untrained judge alone beats the whole 1.7B verifier.
- **Recall at the guarantee rises most of all.** At the same ≤ 10% promise, 8B catches 47% of wrong replies against 28% for 1.7B.
  - RAG: 44% against 19%.
- **Per domain, 4B → 8B gains most on knowledge-heavy and grounded domains:**
  - FELM world knowledge 0.61 → 0.68 (0.53 at 1.7B);
  - FELM writing 0.79 → 0.89;
  - CovidQA 0.84 → 0.89;
  - PubMedQA 0.77 → 0.86;
  - TruthfulQA 0.73 → 0.77.
- **It still fails on numbers in passages.** DROP stays 0.47-0.56 and FinanceBench 0.60-0.65 at every size.
- **On closed-book chat, the summed combination is worse than the judge alone** (0.71 against 0.75 at 8B). Giving the
  judge more weight when there are no documents is a cheap follow-up.
- **The full corpus lifts 4B less than expected.** 0.778 on all 5,107 replies, against 0.779 on the 1,800-reply subset.
  - The doc's estimate of about 0.80 was optimistic. On the full corpus, 4B beats 1.7B by +0.05, not +0.08.

## Speed (T4, fp16, batch 1, median reply ≈ 350 tokens, up to 2,048)

| reader | s / reply (forward + features) | T4-seconds / reply | full corpus | ≈ $ / 1,000 checks at $0.35 per T4-hour |
|---|---|---|---|---|
| 1.7B | 0.29 | 0.29 | 14 min on 2 workers | 0.03 |
| 4B | 0.72 | 0.72 | 33 min on 2 workers | 0.07 |
| 8B (2 T4s, layers split) | 0.97 | 1.93 | 85 min | 0.19 (≈ 0.09 on one 24 GB GPU) |
| 14B nf4 4-bit (2 T4s) | 1.59 | 3.18 | 140 min | 0.31 |

- Timing is the wall time inside `Reader.features` per reply.
- 1.7B and 4B ran two workers at once, sharing the CPU.
- 8B in fp16 does not fit one T4. On a single L4 or A10 (24 GB) it would cost about one GPU-second a reply.

**Qwen3-14B in nf4 4-bit (bitsandbytes, fp16 compute), across both T4s** (`results/omni/final_q14b_nf4.json`;
head in `results/kaggle/omni-scale-q14b/heads/omni_qwen3_14b_nf4.npz`).
- **Two T4s are required.** On one T4 it ran out of memory on the longest replies' logits.
- **Zero labels: 0.820** (+0.015 over 8B).
- **With labels it gains more:** 0.842 at 50 labels, and RAG 0.833 at 50 labels, 0.842 at 100.
- **Reasoning is saturated** at 0.906.
- **The cost is 1.6× that of 8B.** The returns are now diminishing.

## Recommendation

| tier | reader | zero labels | 100 labels + guarantee recall | price (T4) |
|---|---|---|---|---|
| screening (every reply) | 1.7B | 0.73 | 28% | ≈ $0.03 / 1k |
| **default** | **4B** | 0.78 | 40% | ≈ $0.07 / 1k |
| accuracy (RAG, long-form, knowledge, high stakes) | 8B | 0.81 | 47% | ≈ $0.09 / 1k on one 24 GB GPU, ≈ $0.19 on T4s |
| (not worth a tier) | 14B nf4 | 0.82 | 49% | ≈ $0.31 / 1k on T4s |

- **Make 4B the default.** It costs 2.5× the 1.7B compute and adds +0.05 zero-label and about 1.5× the recall at a held
  false-alarm rate.
- **Make 8B the premium tier.** Over 4B it adds about another +0.03, and recall on long-form rises from 41% to 55%.
- **Do not make 14B a tier.** 4-bit 14B adds +0.015 zero-label over 8B, or +0.025 at 50 labels, for about 1.6× the cost.
  Keep it in mind for RAG customers who label 50 or more replies.
- **Use 1.7B only where cost dominates,** or for math and reasoning, where it is already at 0.85.

Heads: `heads/omni_qwen3_4b.npz` (now trained on the full 5,107 replies) and `heads/omni_qwen3_8b.npz`.
