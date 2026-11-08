# Closed-book chat: retrieve, then ground (Qwen3-8B reader, Kaggle 2× T4)

**Question.** Closed-book chat is the omni verifier's weakest format: 0.73 held-out AUROC with the Qwen3-8B reader.
The recommended fix was "retrieve, then ground": look the reply's facts up, then score the reply *with* the passages
as documents. That lets the reader and the NLI cross-encoders check claims instead of recalling facts.

**Verdict: no.** With live Wikipedia evidence, nothing beats the closed-book score:
- the reader with documents scores 0.728 against 0.734;
- NLI on the retrieved passages scores 0.537, near chance;
- the best fair combination scores 0.735.

The evidence seldom contains the fact that decides the reply (about 1 in 6 by spot-check). When it does not, the
passages pull the reader's judge and the NLI scores toward "topic is related" and away from "fact is right". Retrieval
also adds about 7 s and ~8 Wikipedia requests per reply. Closed-book chat needs evidence of a different kind (see the
last section), not this route.

## Setup (`kaggle/omni/closedbook_retrieval.py`, `results/omni/closedbook_retrieval.json`)

**Replies.** All 1,009 closed-book replies of the four held-out domains in the corpus (`experiments/omni_corpus.py`,
md5 matches the repo):

| domain | replies | what they are |
|---|---|---|
| TruthfulQA | 400 | short answers, half of them misconceptions |
| HaluEval general | 300 | real ChatGPT replies to user queries |
| FELM world knowledge | 184 | long-form ChatGPT replies |
| FELM science | 125 | long-form ChatGPT replies |

**Queries.**
- Short replies (under 25 words): the question plus the reply, then the question alone.
- Long replies: the question, plus up to 3 (HaluEval) or 4 (FELM) of the reply's sentences that carry names or numbers.

**Retrieval.** `vt.retrieval.Wikipedia(full_text=True, k=2, pages=2, passage_words=100)`, live, from the Kaggle kernel.
- Pacing: one request in flight, 0.35 s apart, with backoff on HTTP 429/503.
- Kaggle was not rate-limited. 8,090 requests drew 8 HTTP 500s and 2 timeouts; 0.6% of replies got no evidence.
- The passages from all queries are pooled. Up to 6 are kept: each query's best passage, then BM25 against question + reply.

**Scoring.**
- **Reader.** `vt.omni.Reader` (Qwen3-8B, fp16 across both T4s) reads each reply twice:
  - closed book, exactly as in the corpus (its judge log-odds reproduce `results/omni/expert_scores_q8b.npz` exactly);
  - with the passages as documents (`vt.omni.make_item`, the RAG framing).
- **Head.** The shipped head `heads/omni_qwen3_8b.npz` scores both reads (reply head, worst unit, judge × 2). Each
  domain is standardised by its own replies (the zero-label protocol).
- **NLI.** `NLIScorer` scores the reply against the passages (HHEM-2.1-open + MiniCheck, first 4 sentences, as
  shipped). MiniCheck was also run over every sentence.

**Fair baseline (a).** The leave-one-domain-out closed-book expert scores (`expert_scores_q8b.npz`, z(r) + z(u) + 2 z(j)).
- The shipped head was trained on all 21 domains, these four included. On these replies it is in-sample: 0.771 closed book against 0.734 held out.
- So every result that uses the shipped head with documents is optimistic by up to about 0.04.

## Results: AUROC for "wrong", zero labels from the domain

| | TruthfulQA | HaluEval gen. | FELM wk | FELM sci. | **mean** |
|---|---|---|---|---|---|
| (a) closed book, held-out heads (the shipped number) | 0.784 | 0.671 | 0.673 | 0.808 | **0.734** |
| (a′) closed book, shipped head (in-sample, leak check) | 0.824 | 0.703 | 0.714 | 0.843 | 0.771 |
| (b) reader with retrieved documents (r + u + 2j; shipped head, so optimistic) | 0.781 | 0.625 | 0.695 | 0.812 | 0.728 |
| (b) reader's own judge with documents (no training) | 0.740 | 0.627 | 0.639 | 0.766 | 0.693 |
| — the same judge closed book | 0.782 | 0.719 | 0.638 | 0.745 | 0.721 |
| (c) NLI on retrieved documents (shipped scorer) | 0.497 | 0.456 | 0.565 | 0.629 | 0.537 |
| (c) NLI, MiniCheck over every sentence | 0.497 | 0.442 | 0.567 | 0.623 | 0.532 |
| (d) (a) + NLI | 0.760 | 0.649 | 0.667 | 0.811 | 0.722 |
| (d) (a) + 2 × judge with documents | 0.794 | 0.665 | 0.676 | 0.804 | **0.735** |
| (d) (a) + 2 × judge with documents + NLI | 0.773 | 0.650 | 0.672 | 0.801 | 0.724 |
| (d) reader with documents + NLI | 0.750 | 0.607 | 0.680 | 0.802 | 0.710 |
| (d) learned weights, leave one of the 4 out: (a) experts + judge with documents + NLI | 0.797 | 0.657 | 0.672 | 0.796 | 0.731 |
| (d) learned weights, (a) experts only (a control for reweighting alone) | 0.787 | 0.660 | 0.665 | 0.795 | 0.727 |
| (d) learned weights, all experts incl. the in-sample head with documents | 0.838 | 0.669 | 0.704 | 0.817 | 0.757* |

\* uses the shipped head on these domains' own replies: not a held-out number. Its gain over (a) is no larger than
the leak itself (0.771 − 0.734).

Learned weights put about 0.34 on the closed-book judge, 0.14 on the judge with documents and −0.02 to −0.04 on NLI.

**What the table says:**
- **Documents make the reader's judge worse on chat.** HaluEval general falls 0.72 → 0.63 and TruthfulQA 0.78 → 0.74.
  FELM science rises slightly (0.745 → 0.766). The head with documents is below the same head closed book
  on every domain (0.728 against 0.771, both in-sample).
- **NLI against retrieved evidence is chance on chat**: 0.44-0.50. It reaches 0.57-0.63 on FELM, where the long replies restate facts a page contains.
  - Adding NLI to the closed-book score lowers it everywhere except FELM science (+0.003).
- **No combination that avoids the leak beats the shipped closed-book score by more than 0.01** (0.735 against 0.734).
  - Per domain, TruthfulQA gains +0.010 from adding the judge with documents. That is within the ±0.05 confidence interval.

## Retrieval diagnostics

| | TruthfulQA | HaluEval gen. | FELM wk | FELM sci. |
|---|---|---|---|---|
| passages kept (mean) | 3.7 | 5.7 | 4.9 | 5.6 |
| Wikipedia requests per reply | 3.9 | 11.3 | 9.0 | 12.0 |
| retrieval seconds per reply (mean, paced) | 2.4 | 6.8 | 6.0 | 7.7 |
| evidence contains the correct answer's content words (TruthfulQA, automatic) | 46% of words; all words in 20% | | | |
| **spot-check: evidence states the deciding fact** (10 random wrong replies; yes / partial / no) | 2 / 1 / 7 | 0 / 1 / 9 | 2 / 3 / 5 | 0 / 2 / 8 |

Spot-check reading of `results/kaggle/omni-cb-retrieval/evidence.json`, 40 wrong replies:
- **TruthfulQA.** The deciding fact was found for the pyramids ("archaeologists now believe ... skilled workers") and
  the Bermuda Triangle (Lloyd's, the Coast Guard).
  - Most misconception questions retrieve pages that only share a word: "food cravings" → Chrissy Teigen's *Cravings*
    cookbooks; "lemmings" → *Grizzy & the Lemmings*; "blood type CEOs" → *True Blood*.
  - Querying with the reply also pulls in pages that discuss the misconception itself. "Inside job" retrieves
    *9/11 conspiracy theories*, which NLI reads as support.
- **HaluEval general.** Nine of ten wrong replies have no retrievable fact. They answer tasks with missing input,
  creative prompts, arithmetic ("compute the regression coefficient") or a refusal. This domain's "hallucination" is
  mostly not a world-knowledge error, so retrieval cannot help it.
- **FELM world knowledge.**
  - Found: Saturn's density (refutes "Jupiter is less dense than Saturn") and the years with a Friday the 13th.
  - Victoria Falls: the retrieved passage *supports* the wrong claim ("considered to be the largest waterfall in the world").
  - "Famous graduates of the University of Leicester" retrieved the University of Cambridge.
- **FELM science.** Technical claims (EPR line counts, a misapplied Newton's law, a named researcher's awards) are not
  stated in any passage. The pages are on topic; the passages are not.

**In all, about 4 of 40 wrong replies (10%) got the deciding fact, and about 7 more (18%) got the right page without the
fact.** Full-text BM25 over two pages per query cannot do better on open-domain chat. The search is a title or
keyword search, and many errors are near-misses that only a precise passage refutes.

## Cost per reply (measured, unbatched)

| step | seconds per reply | where |
|---|---|---|
| retrieval (mean of 4 domains; paced at 0.35 s per request) | 5.0 | network, CPU |
| reader, closed book (the shipped check) | 0.18 | both T4s |
| reader with documents (~800 more tokens) | 1.06 | both T4s |
| NLI (HHEM + MiniCheck, GPU) | 0.91 | one T4 |

- **Added per reply:** about 7 s of wall time and ≈ 3 T4-seconds, roughly 10× the closed-book check.
  - At $0.35 per T4-hour, the GPU part is about $0.29 per 1,000 checks on top of the shipped one.
  - Each reply also costs about 8 Wikipedia API calls. At this pace a single client manages about 500 replies an hour.
- **The whole run:** 1,009 replies in 85 minutes of kernel time.

## What would be worth trying instead

- **For TruthfulQA-style misconceptions.** The useful source is a curated misconceptions index ("List of common
  misconceptions" was retrieved for only a few questions), or a dense retriever over passages, not title search.
- **For long-form FELM.** The retrieved pages are usually right. Ranking passages *per claim* with a cross-encoder
  (the `rerank` hook in `vt.retrieval`) and scoring each claim against its own best passage, not the whole reply, is
  the cheap next step. NLI rose to 0.57-0.63 here even with crude passages.
- **For HaluEval general.** Retrieval is the wrong tool: most of its wrong replies are task failures, which the
  closed-book judge already catches best (0.72).
- **Biggest closed-book lever measured so far.** The 8B reader's own judge, weighted up. The learned weights here
  confirm it: the judge carries about 0.34 of the weight.
