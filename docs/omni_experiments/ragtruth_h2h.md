# The omni verifier on the official RAGTruth benchmark, head to head

**Question.** On RAGTruth's full official test split, scored exactly like the published detectors, does the omni
verifier's frozen reader plus linear heads match the fine-tuned detectors?

**Answer.** It ties the fine-tuned leaders, LettuceDetect-large (79.2) and Llama-2-13B (78.7). It does not reach
RAG-HAT (83.9).
- **Qwen3-8B reader:** response F1 **79.3 [77.3, 81.1]**, AUROC **0.926** [0.915, 0.935]. Its frozen reader read only 45% of the training answers.
- **Qwen3-4B reader:** response F1 **79.1 [77.1, 80.9]**, AUROC 0.921. It read all of the training answers.
- **NLI:** on its own it is far weaker (68 F1) and adds almost nothing on top of the reader.

## Protocol

| | |
|---|---|
| data | `wandb/RAGTruth-processed`: train 15,090 answers, **test 2,700 (900 QA / 900 Summary / 900 Data2txt, 943 hallucinated)**, all scored, none missing |
| label | an answer is hallucinated if it has any human-labelled span (response level, as published) |
| threshold | maximises F1 on 5-fold out-of-fold scores of **training** answers. The test set never chose a threshold, a C, a feature set or the headline |
| headline rule | the expert or stack with the highest training out-of-fold F1, fixed before looking at test |
| CIs | 95% bootstrap over test answers (2,000 resamples for F1, 1,000 for AUROC) |
| hardware | Kaggle 2× T4, fp16. The 8B is split over both GPUs; the 4B runs one worker per GPU. Up to 3,072 tokens |
| code | `kaggle/omni/ragtruth_h2h.py` (kernel `ragtruth-h2h`); raw outputs in `results/kaggle/ragtruth-h2h/`; summary in `results/omni/ragtruth_h2h.json` |

Each answer gets one forward pass through the product's reader, `vt.omni.Reader.features`, unchanged. The reader
reads the documents, the question, the reply, and the judge question appended after the reply. In the same pass,
hooks read the MLP neurons (down_proj inputs) of three middle layers, averaged per sentence. This is the feature
from `experiments/ragtruth_sota.py`, without its second, question-only pass. The NLI expert is
`vt.omni.NLIScorer`: HHEM-2.1-open at the pinned revision plus MiniCheck-RoBERTa-L, run on GPU 1.

The experts, every one fitted on training answers only:
- **omni reply head:** PCA-256 of [mean residual, judge state] at the 3 middle layers, plus output signals and the judge log-odds, fed to a class-balanced logistic regression.
- **omni worst unit:** a sentence-level logistic regression on PCA-256 of the sentence's mean state; the answer's score is its worst sentence.
- **neurons:** a sentence-level logistic regression on 36,864 (8B) or 29,184 (4B) neurons; the answer's score is its worst sentence.
- **HHEM, MiniCheck, and the reader's judge:** no training.
- **stacks:** a class-balanced logistic regression on the out-of-fold expert scores plus a one-hot task indicator.

**Training size.** The 8B extraction was cut at 7 h, so the 8B read **6,800 of 15,090** training answers, a random
45%. The 4B read all 15,090.

## Result: response-level F1 on the 2,700 test answers

| detector | training | F1 [95% CI] | P | R | AUROC [95% CI] |
|---|---|---|---|---|---|
| RAG-HAT (reported) | fine-tuned | 83.9 | | | |
| LettuceDetect-large | fine-tuned ModernBERT, 15k | 79.2 | | | |
| **omni + neurons + NLI, Qwen3-8B (headline)** | frozen reader + logistic, 6.8k | **79.3 [77.3, 81.1]** | 76.8 | 81.9 | **0.926 [0.915, 0.935]** |
| **omni + neurons + NLI, Qwen3-4B (headline)** | frozen reader + logistic, 15.1k | **79.1 [77.1, 80.9]** | 75.2 | 83.5 | 0.921 [0.911, 0.931] |
| Llama-2-13B | fine-tuned, 15k | 78.7 | | | |
| GPT-4-turbo | prompted | 63.4 | | | |

**By task** (headline stack, global threshold):

| reader | QA F1 (P / R) | Summary F1 (P / R) | Data2txt F1 (P / R) |
|---|---|---|---|
| Qwen3-8B | 66.7 (62.2 / 71.9) | 65.8 (69.8 / 62.3) | 87.1 (83.1 / 91.5) |
| Qwen3-4B | 66.5 (61.4 / 72.5) | 66.5 (65.4 / 67.6) | 87.0 (82.5 / 92.1) |

Per-task AUROC (8B) is 0.915 for QA, 0.891 for Summary and 0.903 for Data2txt. Thresholds chosen per task on
training answers do not help: 78.7 for the 8B and 77.0 for the 4B.

### Every expert alone and stacked (test F1 [CI] / AUROC; training-OOF F1 in brackets)

| expert | Qwen3-8B (6.8k train) | Qwen3-4B (15.1k train) |
|---|---|---|
| reader's own judge (no training) | 66.9 [64.6, 69.1] / 0.807 (70.0) | 62.6 / 0.786 (68.0) |
| HHEM-2.1-open (no training) | 68.7 [66.4, 70.7] / 0.823 (74.7) | 69.3 / 0.823 (73.1) |
| MiniCheck-RoBERTa-L (no training) | 61.2 / 0.740 (67.6) | 60.8 / 0.740 (67.2) |
| NLI mean (the product's expert) | 68.4 / 0.822 (72.5) | 67.9 / 0.822 (71.4) |
| omni reply head | 76.6 [74.4, 78.5] / 0.907 (78.9) | 75.8 / 0.899 (78.2) |
| omni worst unit | 76.1 [74.1, 78.1] / 0.905 (79.1) | 75.2 / 0.895 (77.4) |
| MLP-neuron sentence probe | 77.0 [74.7, 79.0] / 0.887 (79.0) | 74.3 / 0.883 (77.8) |
| stack: omni (reply + unit + judge) | 78.9 [76.9, 80.7] / 0.920 (80.6) | 76.9 / 0.910 (79.2) |
| stack: omni + NLI | 78.9 / 0.921 (80.6) | 77.5 / 0.912 (79.6) |
| stack: neurons + NLI | 78.0 / 0.908 (79.9) | 77.1 / 0.907 (79.6) |
| stack: omni + neurons | 79.4 [77.4, 81.4] / 0.924 (81.3) | 78.9 / 0.920 (80.6) |
| **stack: all (omni + neurons + NLI)** | **79.3 [77.3, 81.1] / 0.926 (81.5)** | **79.1 / 0.921 (80.8)** |

**Which features are stronger?** On held-out training answers, the omni features and the MLP-neuron features are
equal: 80.6 against 79.9 with NLI, and 78.9-79.1 against 79.0 alone. On test the omni features have the higher
AUROC (0.907 against 0.887) and the neurons have the higher F1 at 8B. The two families are complementary: stacking
both adds about 1 point over either alone, and the training-OOF rule picks that stack for both readers.

## Speed (T4, batch 1, mean 1,082 tokens per answer, p95 1,869)

| component | wall s / answer | T4-seconds / answer |
|---|---|---|
| Qwen3-8B reader (2 T4s) | 2.56 (median 2.26) | 5.1 |
| Qwen3-4B reader (1 T4) | 1.28 (median 1.02) | 1.3 |
| NLI: HHEM + MiniCheck (1 T4) | 1.50 | 1.5 |
| **total, 8B / 4B stack** | | **6.6 / 2.8** |

The 8B timing includes contention: the NLI worker shared GPU 1 during the whole 8B run. On the omni corpus the
8B ran at 0.97 s per answer without contention, on inputs whose median is about 350 tokens.

The NLI is the slow part for what it adds. HHEM reads up to 12,000 characters per answer, and MiniCheck runs up to
4 sentences × the document's chunks. Dropping NLI costs about 0.1 F1 at 8B: the omni + neurons stack scores 79.4.
That makes the 4B reader without NLI the cheap configuration: about 1.3 T4-seconds per answer for about 79 F1.

## Verdict

- **A tie with the fine-tuned leaders, not a win.** 79.3 [77.3, 81.1] is statistically indistinguishable from LettuceDetect-large (79.2) and Llama-2-13B (78.7), and clearly below RAG-HAT (83.9).
- **The CPU learning curve held.** That earlier run extrapolated 79-81 from the full training set (docs/ragtruth_sota.md), and 79 is what this run measured.
- **AUROC 0.92-0.93.** A threshold-free number of that size was not published for the fine-tuned detectors in the papers cited here.
- **Summary is the weak task.** Summaries reach about 66 F1, with recall around 62-68%.
- **More reader stopped helping on this benchmark.** 8B on 45% of the training answers equals 4B on all of them. Under this protocol the 4B is the better buy.
- **NLI is weak here.** HHEM and MiniCheck alone score 61-69, and stacked on the reader they add ≤ 0.6 F1. That is the opposite of the omni corpus result, where NLI dominated RAG after adaptation. With 15k in-domain labels, the reader's own states carry the signal.

**Limits.**
- **The 8B did not use the full training set.** Under the omni corpus's learning curve, all 15k answers might add about 1 F1. That is unmeasured.
- **Several configurations were scored on test.** The headline was fixed by the training-OOF rule, and every configuration's CI overlaps the headline's.
- **The training set is more hallucinated than the test set.** 44.5% of training answers are hallucinated against 34.9% of test answers. A threshold chosen on training answers therefore carries that shift. The published detectors face the same shift.
- **NLI scores are incomplete.** 1,190 training answers got no NLI score before the 8B time cut. The 4B stacks fill them with the training mean. Every test answer has an NLI score.
- **The shipped universal head is not used.** It was trained on a corpus that includes RAGTruth test answers, so every head here is refitted on RAGTruth training answers only.
