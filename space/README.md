---
title: VeriChat
emoji: 🔎
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# VeriChat

VeriChat is a chatbot that marks every name, number and date in its replies with how likely the
claim is to be right. The estimate comes from the model's own hidden states, not from asking the
model. When you give it documents, it also says whether each claim came from the documents or from
the model's memory, and quotes the sentence it used. A second tab checks an answer written by any
AI.

Prepared, not published. To build the Space directory, run `bash space/build.sh`. It copies `vt/`
and the trained heads into `space/`.

## How it works

- **Claim scores.** A linear probe reads the hidden state the model has already computed when it
  writes a claim, so scoring adds no extra pass. Green means likely right (P ≥ 0.7), amber means
  uncertain, and red means doubtful. A doubtful claim can be marked or withheld before it is written.
- **Sources.** With the model's internal decisions fixed (normalisation scales, attention patterns,
  gate values), the network is linear. One extra pass then splits each claim's score exactly between
  the documents, the question and the model's memory.
- **Checking another AI's answer.** A frozen reader model with trained probes scores each token of
  the answer.

## Measured performance

Full tables with confidence intervals are in `VISION.md` in the source repository.

| what | result | baseline |
|---|---|---|
| claim correctness, Qwen2.5-7B, 5 formats | AUROC 0.856 | token probability 0.760 |
| errors in the most confident 25% of claims, same model | 7% | 18% |
| reasoning model (DeepSeek-R1-Distill-Qwen-7B) | AUROC 0.811 | 0.651 |
| hallucinated responses on RAGTruth, 3B reader | AUROC 0.864, F1 0.71 | log-probability 0.42 |

## Limits

- **Scores rank, they do not prove.** Even "likely right" claims are sometimes wrong.
- **New domains need examples.** On domains the checker never saw it drops to AUROC 0.61, and to
  chance on financial filings. `vt adapt` refits it from a few hundred labelled answers.
- **Claim detection is a heuristic** (capitalised words and numbers). About 20% of short replies
  contain none, and then the reply's first token is judged instead.
- **Source tracing explains one forward pass exactly.** It does not say what would happen if a
  document were removed. It roughly doubles the cost of a reply that comes with documents.
- **Calibration shifts across domains.** Recalibrate with a few hundred labels where it matters.
- **Moderation screens toxic language only.** A toxicity classifier (unitary/toxic-bert) blocks
  abusive messages and replies. Polite requests for harmful content are left to the chat model's own
  refusals.
- **Do not rely on it for medical, legal or financial decisions.**
