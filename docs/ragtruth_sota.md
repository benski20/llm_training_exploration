# Toward state of the art on RAGTruth from a frozen reader

RAGTruth (ACL 2024) is the standard benchmark for hallucinations in answers written from documents: QA, news
summaries, data-to-text, answers from six LLMs, hallucinated spans labeled by people. The published detectors that
lead it are fine-tuned on its ~15,000 training answers:

| detector | response-level F1 | how |
|---|---|---|
| LettuceDetect large | 79.2 | fine-tuned ModernBERT ([paper](https://arxiv.org/html/2502.17125)) |
| Llama-2-13B | 78.7 | fine-tuned on RAGTruth ([paper](https://aclanthology.org/2024.acl-long.585.pdf)) |
| GPT-4-turbo | 63.4 | prompted |

## The method

No detector is fine-tuned and no LLM is asked anything. A frozen reader (Qwen3-4B) reads each answer once with
its documents in context and once with the question alone; per sentence, the mean of its MLP neurons (down_proj
inputs, every 4th layer, 87,552 values) and two evidence signals (how much the documents raise the words written;
where they make the reader prefer another word) go into a class-balanced logistic regression trained on
RAGTruth's sentence labels. An answer is flagged when any sentence's probability passes a threshold chosen on
held-out training answers. (`experiments/ragtruth_sota.py`)

## Result (CPU: 800 training answers, 600 random test answers, `results/ragtruth_sota_qwen3_4b.json`)

| features | response F1 [95% CI] | precision | recall | response AUROC | sentence AUROC |
|---|---|---|---|---|---|
| neurons | **70.4** [65.9, 74.8] | 64.1 | 78.0 | 0.856 | 0.904 |
| neurons + evidence signals | 70.1 [65.6, 75.0] | 64.0 | 77.5 | 0.856 | 0.904 |
| evidence signals only | 58.0 [53.4, 62.1] | 44.1 | 84.6 | 0.629 | 0.699 |

By task (neurons): data-to-text 84.2, QA 57.4, summaries 46.6 (recall 42%: summary hallucinations are the
subtle ones). Above prompted GPT-4-turbo by 7 points, 9 below the fine-tuned leaders, from 5% of their training
data, a frozen 4B model and a linear layer. The evidence signals add nothing once the neurons are there.

## Learning curve (same test answers, threshold chosen on them: the ceiling at each size)

| training answers | response AUROC | F1 |
|---|---|---|
| 100 | 0.791 | 66.5 |
| 200 | 0.815 | 69.4 |
| 400 | 0.833 | 71.2 |
| 800 | 0.856 | 72.8 |

About +2 F1 and +0.02 AUROC per doubling, not yet flattening. Four more doublings, RAGTruth's full training set,
would reach 79-81 if the trend holds: an extrapolation, to be measured. That run, on the full 2,700-answer test
set so it is directly comparable with the published numbers, is `kaggle/ragtruth_sota` (a free T4, ~3 hours; it
needs a Kaggle API token). Why it would matter: a frozen reader and a linear probe cost a fraction of a fine-tuned
13B detector, the reader can verify any model's answers, and the same reader can serve as the chat model.

Caveats: 600 test answers are a random subset (natural 38% hallucinated), not the full test set; the training
subset is 800 answers; the threshold in the learning curve is chosen on the test set (its "F1" is a ceiling, the
main table's is not).
