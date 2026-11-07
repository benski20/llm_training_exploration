# Commercial landscape and the accuracy bar for an LLM-output verifier

*Research note, 2026-11-07. Scope: who sells or publishes hallucination, grounding and faithfulness checkers, how
accurate they are, what they cost, how fast they are, and where our approach (a frozen small open reader with a
linear probe and per-domain calibrated thresholds, see `docs/product.md`) can win.*

**How reliable each fact is.** Every number has a tag:
**[P]** read from the primary source (leaderboard data, model card, official docs or pricing page);
**[V]** what the vendor says about itself, not reproduced;
**[S]** from a secondary or aggregator source (blog, price tracker, analyst note), not checked against the primary;
**[?]** sources conflict or the number is unclear.
Many 2026 facts below come from fast-moving launch coverage. Re-check prices before quoting them to a customer.

---

## 0. Summary for the team

1. **The market changed in September-October 2026.** A new category of non-generative "decision models" launched:
   TypeSafe Jev, Bespoke Labs Nimble-v4, Cloudflare Clef, Perplexity pplx-decider, Liquid AI d1, Nace.AI Drex,
   Microsoft Decision-1 and AWS Strands Decider. They read the input once and return calibrated probabilities from a
   head, with no generation. That is close to our "two forward passes, no generation" pitch. They now hold the
   top five places on LLM-AggreFact (78.5-79.5 balanced accuracy, above Bespoke-MiniCheck-7B at 77.4). Several are
   sold at **$0.042 per million input tokens, about $0.03 per 1,000 checks of 700 tokens**, with latency of
   0.1-0.5 s. Three are open weights under Apache-2.0 (Clef, Clef-Flash, pplx-decider). **Price, speed and plain
   accuracy on a standard RAG grounding check are now commodities.** Our planned $0.50-1.00 per 1,000 checks is
   15-30 times their API price, and our accuracy is lower.
2. **The accuracy bar.**
   - LLM-AggreFact: **≥77.4 balanced accuracy to be credible, ≥79.5 to claim the state of the art**.
     Small-model tier: FactCG-DeBERTa (0.4B) 75.6, MiniCheck-Flan-T5-L (0.8B) 75.0. Our reader, Qwen3-4B, prompted
     as a judge in think mode scores 73.8.
   - RAGTruth, response-level F1: **79 (LettuceDetect-large, fine-tuned) to 84 (RAG-HAT)**. Clef self-reports 79.4.
     On the RAGTruth subset of AggreFact, the top decision models reach 87-90.5 balanced accuracy.
   - HaluBench: **about 87% accuracy** (Lynx-70B 87.4, GPT-4o 86.5).
   - **We have not measured LLM-AggreFact at all.** Our HaluBench AUROC without labels is 0.62-0.72 and our
     supervised RAGTruth F1 is 70.4. On today's evidence we are not "extremely accurate" on generic grounding.
3. **What nobody sells well**, where our design (white-box, cheap readout, per-domain calibration) fits:
   - **A stated, verified false-alarm rate per customer domain from a few hundred labels.** The decision-model
     vendors claim calibration, but independent checks find overconfidence (Jev ECE 0.074; Laya ECE 0.47 before a
     refit) and big drops on new domains (Clef's new-domain score falls to the low 40s). Nobody sells "at most X%
     false alarms on *your* traffic with 95% confidence, plus a model card".
   - **On-prem, small, open, explainable verifiers with sentence-level evidence.** The open decision models need
     about 52 GB of VRAM (27B), and Nimble's weights are non-commercial.
   - **Closed-book chat, long-form and reasoning outputs with no evidence attached.** Every product in the table
     needs a context to check against. Probes on internals are the only cheap signal here (entity probes reach
     AUC 0.87-0.90 on long-form).
4. **Recommended position:** *"the calibration and assurance layer for LLM verification"*. We sell guarantees,
   not raw scores. Offer per-domain, label-efficient, confidence-bounded false-alarm rates and audit-grade model
   cards, on-prem, over any scorer: our probe, an open decision model, or both combined. Add white-box closed-book
   detection where the customer runs open models. **Do not compete on price per check for generic RAG grounding.**
5. **Minimal credible evidence package** (§6):
   - full LLM-AggreFact and RAGTruth results (response F1 and span F1) and HaluBench, against the open decision
     models, MiniCheck-7B, HHEM and Granite Guardian;
   - a multi-domain test of the false-alarm guarantee, reporting how often the rate is violated against the number
     of labels;
   - measured GPU p50/p95 latency and cost;
   - one design-partner model card.

---

## 1. The accuracy bar: current SOTA numbers

### 1.1 LLM-AggreFact (balanced accuracy, 11 human-labeled grounding datasets)

The data was parsed from the live leaderboard ([llm-aggrefact.github.io](https://llm-aggrefact.github.io/)) on
2026-11-07 **[P]**. It has 47 entries. The metric is balanced accuracy over CNN, XSum, MediaSum, MeetingBank, WiCE,
REVEAL, ClaimVerify, FactCheck-GPT, ExpertQA, LFQA and RAGTruth. Note: **the board has no 2025-2026 frontier chat
LLMs** (no GPT-5.x, Claude 4.x or Gemini 2.5/3), so "beats GPT-4o" is an old comparison.

| rank | model | maker, type | size | avg BAcc | RAGTruth subset | weights |
|---|---|---|---|---|---|---|
| 1 | nimble-v4-max | Bespoke Labs, decision model | 27B | **79.5** | **90.5** | CC BY-NC (commercial use via API only) [S] |
| 2 | Clef | Cloudflare, decision model | 27B | 79.0 | 87.3 | Apache-2.0 [P] |
| 3 | Decider v1.2 (pplx-decider) | Perplexity, decision model | 26-27B | 78.7 | 89.0 | Apache-2.0 [P] |
| 3 | Jev 1.13 | TypeSafe, decision model | undisclosed | 78.7 | 87.0 | closed, API only |
| 5 | d1-ultra | Liquid AI, decision model | undisclosed | 78.5 | 89.2 | closed, API |
| 6 | nimble-v4 | Bespoke Labs | 9B | 77.8 | 88.9 | |
| 7 | **Bespoke-MiniCheck-7B** | Bespoke Labs, fine-tuned classifier | 7B | **77.4** | 84.0 | CC BY-NC |
| 8 | Claude-3.5 Sonnet | Anthropic, prompted | - | 77.2 | 86.1 | |
| 9 | Clef-Flash | Cloudflare | 9B | 76.6 | 83.2 | Apache-2.0 |
| 10 | Granite Guardian 3.3 | IBM | 8B | 76.5 | 82.2 | Apache-2.0 |
| 10 | Drex v1.5 | Nace.AI | 9B | 76.5 | 86.1 | Open RAIL-M |
| 10 | Mistral-Large 2 | Mistral | 123B | 76.5 | 85.9 | |
| | gpt-4o-2024-05-13 | OpenAI | - | 75.9 | 84.3 | |
| | **FactCG-DeBERTa-L** | academic | 0.4B | **75.6** | 82.3 | open |
| | MiniCheck-Flan-T5-L | academic | 0.8B | 75.0 | 78.0 | open |
| | gpt-4o-mini | OpenAI | - | 74.0 | 81.6 | |
| | MiniCheck-RoBERTa-L (our HaluBench baseline) | academic | 0.4B | 73.5 | 77.2 | open |
| | HHEM-2.1-open | Vectara | 0.1B | 71.8 | 69.9 | Apache-2.0 |
| | AlignScore | academic | 0.4B | 70.5 | 71.7 | |
| | Llama-3.1-8B-Instruct | prompted | 8B | 70.3 | 73.6 | |

Not on the board, but relevant:
- **HalluGuard-4B** (Qwen3-4B fine-tuned, think mode): 75.7 in the arXiv v1 and 77.1 in ACL Findings 2026 [?].
  **Its base, Qwen3-4B, prompted in think mode: 73.8**
  ([arXiv 2510.00880](https://arxiv.org/abs/2510.00880), [ACL 2026](https://aclanthology.org/2026.findings-acl.835/)).
  Our reader is that same model. A probe on Qwen3-4B has to beat about 74 just to beat simply prompting the reader,
  and prompting costs generation.
- **Jev scored by Braintrust** on a 1,086-claim AggreFact sample: 80.2% overall, 92.3% on RAGTruth, 53.3% on
  ExpertQA; median latency 0.17-0.18 s; $0.054-0.072 per 1,000 judgments [S]
  ([Braintrust](https://www.braintrust.dev/evals/jev-vs-gpt-eval-judges)).

**Ceiling note.** Every system scores 59-61 on ExpertQA and 61-70 on CNN. Label noise caps these sets, so the
practical ceiling of the board is probably low 80s. A gap of 1-2 points between the top models is within noise.

### 1.2 RAGTruth (response-level F1 unless noted)

| system | F1 | notes |
|---|---|---|
| RAG-HAT (Llama-3-8B, DPO) | **83.9** | best fine-tuned LLM, cited in the LettuceDetect paper [P-cited] |
| unnamed 2026 model | 81.8 | ResearchGate snippet, unverified [?] |
| Clef (27B decision model) | 79.4 | self-reported, "RAGTruth hallucination F1" from Cloudflare's Decision Index run [V] ([HF card](https://huggingface.co/cloudflare/clef)) |
| LettuceDetect-large (ModernBERT 395M, token classifier) | 79.2 | [P] ([arXiv 2502.17125](https://arxiv.org/abs/2502.17125)) |
| fine-tuned Llama-2-13B (RAGTruth paper) | 78.7 | |
| Jev 1.x | 76.5 | in Cloudflare's table [V] |
| LettuceDetect-base (149M) | 76.1-76.8 [?] | |
| TinyLettuce (Ettin 68M) | 75.0 | [S] ([HF blog](https://hf-2025.pages.dev/blog/adaamko/tinylettuce)) |
| **ours: supervised probe on frozen Qwen3-4B, 800 training answers** | **70.4** [65.9, 74.8] | `docs/ragtruth_sota.md` |
| Luna (Galileo, DeBERTa) | 65.4 | |
| prompted GPT-4-turbo | 63.4 | |
| Clef-Flash | 35.6 | [V] |

Span-level F1 is much lower across the field (mid-50s to about 60 for the best encoders). Few vendors report it,
so **sentence/span localization is a place to stand out**.

White-box detection on RAGTruth is **disputed**:
- [LUMINA](https://arxiv.org/abs/2509.21875) (ICLR 2026) and ReDeEP report AUROC 0.73-0.81.
- The [PARALLAX preprint](https://arxiv.org/abs/2605.17028) (May 2026) finds most of 22 activation methods near
  chance under controlled conditions. The exceptions are supervised upper-layer probes (SAPLMA, DRIFT). It also
  shows that several corpora leak the answer into the prompt.
- Our label-free 0.77-0.82 AUROC on RAGTruth is in the competitive band. We should run PARALLAX-style artifact
  checks before a buyer's ML team does.

### 1.3 HaluBench (Patronus; 6 sources: HaluEval, RAGTruth, FinanceBench, DROP, CovidQA, PubMedQA)

| system | metric | value |
|---|---|---|
| Lynx-70B (Patronus) | accuracy | **87.4%** [V] ([docs](https://docs.patronus.ai/docs/research_and_differentiators/Lynx)) |
| GPT-4o | accuracy | 86.5% [V] |
| Lynx v2.0 8B | accuracy | +2.2 over Claude-3.5-Sonnet [V] |
| Cleanlab TLM (on gpt-4o-mini) | AUROC per source | FinanceBench 0.85, PubMedQA 0.885, CovidQA 0.943, DROP 0.886 [V, Cleanlab-linked study, [arXiv 2503.21157](https://arxiv.org/abs/2503.21157)] |
| LLM judge (gpt-4o-mini), same study | AUROC | 0.74-0.84 |
| HHEM-2.1 | AUROC | 0.45-0.73 |
| **ours, label-free** | **AUROC overall** | **0.62 / 0.72 / 0.64** over three samples; FinanceBench and DROP near chance |

**Gap:** TLM and strong LLM judges reach 0.85-0.94 AUROC on the exact sources where we are weakest (FinanceBench,
PubMedQA, DROP). They get there by sampling and self-consistency at LLM cost, about 10-100 times ours.

---

## 2. Vendor-by-vendor landscape

### 2.1 The new decision-model category (Sept-Oct 2026): the most direct threat

How they work: a post-trained backbone (mostly Qwen3.x-27B or 9B) reads a state and typed questions and returns
probabilities from a head in one forward pass (yes/no "Noul", Choice or Score). Groundedness is one question type.

| model | maker | license | price | latency | groundedness evidence |
|---|---|---|---|---|---|
| Jev 1.13/1.14 | TypeSafe (launched Sep 15 2026; $40M from DCVC [S]) | closed, API and OpenRouter | $0.042/M input, output free [S, consistent across sources] | 0.17-0.62 s hosted [S] | AggreFact 78.7 [P]; 80.2 on the Braintrust sample [S] |
| nimble-v4 / v4-max | Bespoke Labs (maker of MiniCheck) | CC BY-NC weights; API | $0.04/M input [S] | n/a | AggreFact 77.8 / **79.5** [P] |
| Clef / Clef-Flash | Cloudflare | Apache-2.0 | $0.24/M on Workers AI [S]; self-host | 209 ms / 39 ms median on H200 [V] | AggreFact 79.0 / 76.6 [P]; RAGTruth F1 79.4 [V] |
| pplx-decider v1.2 | Perplexity | Apache-2.0, about 52 GB of weights | self-host | **96 ms median on GB200** [V] | AggreFact 78.7 [P] |
| d1-ultra; d1-3B; d1-omni-600M | Liquid AI | ultra closed; small ones "LFM Open" | $0.042/M (ultra) [P] | d1-3B 16 ms [S] | AggreFact 78.5 (ultra) [P] |
| Drex v1.5 | Nace.AI | Open RAIL-M | API (OpenRouter) | 0.65 s at 8-32k tokens [V] | AggreFact 76.5 [P] |
| Microsoft-Decision-1 (9B), AWS Strands Decider 2B, OpenAI Decisions API (preview), H2O-Lightning-4B, Laya (421M) | various | mixed | $0.042/M (MS) [S] | 0.46 s (MS), <150 ms (AWS) [V] | not on AggreFact |

Independent counter-evidence ([survey](https://redreamality.com/blog/open-decision-models-2026-survey-benchmarks/)) [S]:
- Jev is overconfident (mean confidence 0.81 against 0.74 accuracy, ECE 0.074).
- Reordering options changes 7-11% of answers.
- One embedded instruction flips 12% of Jev's groundedness verdicts (prompt injection, per Braintrust).
- Answers drift between versions.
- Clef's new-domain score drops to the low 40s.
- Distilled models' low ECE is measured against the teacher, not the truth.

**These are exactly the weaknesses that per-domain calibration with held-out guarantees addresses.**

Sources: [OpenRouter on Jev](https://openrouter.ai/blog/insights/what-is-jev/),
[metricnexus](https://metricnexus.ai/blog/jev-benchmarks-pricing),
[Clef card](https://huggingface.co/cloudflare/clef), [pplx-decider card](https://huggingface.co/perplexity-ai/pplx-decider-v1.2),
[Liquid d1-ultra](https://www.liquid.ai/blog/d1-ultra), [Drex card](https://huggingface.co/nace-ai/drex-v1.5),
[infrabase Nimble pricing](https://infrabase.ai/models/nimble-v4-max).

### 2.2 Specialist grounding and hallucination models

| product | method | accuracy | latency | price | domain adaptation | deployment |
|---|---|---|---|---|---|---|
| **Vectara HHEM-2.1-open** | T5-base cross-encoder, premise/hypothesis score | AggreFact 71.8, RAGTruth subset 69.9 [P] | about 1.5 s per 2k tokens on CPU, <600 MB RAM [P]; 0.34 s median in the Lund thesis [S] | free (Apache-2.0) | none built in | anywhere |
| **Vectara HHEM-2.3** (commercial) | longer-context backbone, 11 languages | TofuEval BAcc 78-86%; RAGTruth-Summary recall ≥0.758 against ≤0.377 for 2.1 [V] ([blog](https://www.vectara.com/blog/hallucination-detection-commercial-vs-open-source-a-deep-dive)) | "faster" [V] | bundled with the Vectara platform: SaaS from $100K/yr, VPC $250K, on-prem $500K [S] | none public | SaaS/VPC/on-prem |
| **MiniCheck** (RoBERTa/DeBERTa/Flan-T5) and **Bespoke-MiniCheck-7B** | claim-vs-document classifier trained on synthetic data | 73.1-75.0 (small), 77.4 (7B) [P] | small ones fast on CPU; 7B needs a GPU | small ones open; 7B CC BY-NC with a self-serve API (price not published) | none | open / API |
| **LettuceDetect / TinyLettuce** (KRLabs) | ModernBERT/Ettin token classifier, 8k context | RAGTruth F1 79.2 / 76 / 75 [P] | ms-scale on GPU | MIT, free | retrain on your spans; v2 adds typed spans and agent/tool output | open |
| **Patronus Lynx** (8B/70B), **Glider** (3.8B judge) | fine-tuned Llama judges with chain-of-thought | HaluBench 87.4 (70B) [V] | seconds (generative) | API: $10-20 per 1,000 evaluator calls [S, unverified] | custom judges via rubric | open weights, API. Patronus raised a $50M Series B and moved toward "digital world models" [S] |
| **Galileo Luna / Luna-2** (now **Cisco/Splunk**, deal announced Apr 2026, completed [P]) | small fine-tuned evaluators with LoRA adapters per metric; ChainPoll (LLM polling) for the premium path | "0.95 F1" against GPT-4o's 0.94 on unnamed tasks [V]; Luna-1 RAGTruth F1 65.4 | **152 ms avg; 10-20 metrics in <200 ms on L4** [V] | **$0.02-0.15 per M tokens [?]** (inside the platform) | "correct a handful of verdicts" in Luna Studio [V] | managed (Splunk); no on-prem stated |
| **IBM Granite Guardian 3.3** (8B) | generative risk judge with groundedness, context relevance and answer relevance modes | AggreFact 76.5, #1 on REVEAL [P] | 8B generation (yes/no plus optional think) | Apache-2.0, free | prompt-defined risks | open, watsonx |
| **Llama Guard family** | safety classifier | **not a grounding model**: no hallucination category | fast | open | - | - |
| **HalluGuard-4B** (academic) | Qwen3-4B reasoning fine-tune | AggreFact 75.7-77.1 [?] | think-mode generation | Apache-2.0 promised | - | open |

### 2.3 Cloud-platform grounding checks

| service | method | accuracy | price | notes |
|---|---|---|---|---|
| **AWS Bedrock Guardrails: contextual grounding** | model scores grounding and relevance against a user threshold | none published | **$0.10 per 1,000 text units** (1 unit ≤ 1,000 chars of source + query + response) [S] | about $0.30 per 1,000 checks of 2.8k chars; Bedrock-only |
| **AWS Automated Reasoning checks** (GA Aug 2025) | NL policy → formal logic (SMT); checks answers against rules | "up to 99%" on correct responses, under narrow conditions [V] | $0.17 per policy per 1,000 text units [S] | verifies rules and policies, not documents; heavy policy authoring |
| **Azure AI Content Safety groundedness detection** | fine-tuned model; non-reasoning (fast) and reasoning (explains) modes; optional *correction* (rewrite) | none published | per 1,000 text records (1 record = 1,000 chars); public page shows "$-"; about $0.38/1k records per a third party [?]; 5k free per month | [docs](https://learn.microsoft.com/en-gb/azure/ai-services/content-safety/concepts/groundedness) |
| **Google Vertex check grounding API** | support score plus per-claim citations | none published | **$0.00075 per 1,000 answer characters** [P] ([pricing](https://cloud.google.com/generative-ai-app-builder/pricing)) | about $0.38 per 1,000 checks of 500-char answers; cheapest of the clouds per answer |

### 2.4 Uncertainty, judges and evaluation frameworks

| product | method | numbers | cost / latency |
|---|---|---|---|
| **Cleanlab TLM** | samples several responses, self-reflection and consistency into a trust score; works closed-book | best HaluBench AUROCs (0.85-0.94) [V, Cleanlab-linked] | pay per token, rates shown only in the account [S]; several LLM calls; the default preset "is not latency/cost-optimized" [P] |
| **Ragas / DeepEval / TruLens / Arize Phoenix** faithfulness | LLM claim decomposition plus an LLM NLI judge, open-source | Ragas on GPT-5.2: AUROC 0.708 on RAGTruth (Lund thesis) [S] | median **18.5 s** (Ragas on gpt-4o-mini) [S]; cost is that of the LLM. **Arize is being acquired by Dynatrace for about $915M** [S] |
| **LLM-as-judge** (G-Eval style) | prompted frontier model | G-Eval with GPT-5.2: AUROC 0.852 on RAGTruth [S] | gpt-4o-mini at 4.5k tokens: **$0.74 per 1,000** [S]; 2.5 s median |
| **Contextual AI LMUnit** | natural-language unit tests scored 1-5 (Llama-3.1-70B / Qwen2.5-72B, MIT) | SOTA on FLASK/BigGenBench, RewardBench 93.5 [V]; not a groundedness benchmark | 70B generation |
| **Guardrails AI** | validator hub, wraps other detectors | - | **acquired by Harvey, Sep 2026** [S] |
| **Lakera** | prompt-injection and security firewall, not hallucination | - | **acquired by Check Point, about $300M, 2025** [S] |
| **Haize Labs** | red-teaming and judge-scaling (Verdict) | none found | services and enterprise |

### 2.5 Interpretability companies (probes as a service)

- **Goodfire "inside-out monitors"** (launched Oct 8 2026; distributed through Baseten) are activation probes on
  the serving model, priced at **$185 per 1M exchanges ($0.185 per 1,000)**, with <2% added latency and four
  probes running at once. The launch targets hacking, CBRN misuse and reward hacking, *not hallucination*. They
  reported a 93% catch rate at 5.5% false positives for malicious hacking sessions [S]
  ([Forkast](https://forkast.news/goodfires-inside-out-monitors-show-interpretability-is-becoming-agent-security-infrastructure/)).
  Their research (RLFR) uses token- and span-level hallucination probes and reports 58% fewer hallucinations
  [P-research]. **This is the closest analogue to our approach and shows buyers will accept probe-based monitors.
  It also means Goodfire could enter hallucination detection at any time.**
- **Obeso, Arditi, Nanda et al.** ([arXiv 2509.03531](https://arxiv.org/abs/2509.03531)): linear probes on the
  generator flag hallucinated entities in long-form text at AUC 0.87, and LoRA probes at 0.90, on Llama-3.3-70B.
  Semantic entropy gets 0.60-0.71. Recall at 10% false positives is about 0.7. This is the evidence that
  **closed-book, long-form detection from internals works when you own the generator**.
- **Transluce:** no hallucination product found (research and datasets only).
- **Google DeepMind** reportedly runs similar internal probes in Gemini (not sold).

---

## 3. Price and latency to beat (normalized per 1,000 checks)

These figures assume a typical RAG check of about 700 tokens (about 2,800 characters, answer about 500 characters).
That is close to our measured HaluBench mean of 650 tokens. Unit conversions are ours.

| offering | $ per 1,000 checks | p50 latency | on-prem |
|---|---|---|---|
| **our compute cost (estimate, L4)** | **≈$0.04** | ≈0.15 s on GPU (estimate); 1.4-5.9 s on CPU (measured) | yes |
| Galileo Luna-2 (platform needed) | $0.014-0.10 | 152 ms | no |
| **Jev / d1-ultra / Microsoft-Decision-1 / Nimble API** ($0.042/M) | **≈$0.03** (×N if one question per sentence) | 0.1-0.5 s | Jev/d1-ultra no; Nimble no (NC weights) |
| Clef on Workers AI ($0.24/M) | ≈$0.17 | 0.21 s | yes (Apache-2.0, about 54 GB of VRAM) |
| pplx-decider self-hosted | GPU cost, about $0.2 per 1k decisions (JevBench estimate) [S] | 0.1 s (GB200) | yes |
| Goodfire monitors | $0.185 | <2% overhead | via Baseten |
| AWS contextual grounding | ≈$0.30 | n/a | no |
| Vertex check grounding | ≈$0.38 | n/a | no |
| Azure groundedness | ≈$1.1 [?] | n/a | containers for some features |
| gpt-4o-mini judge, one call | ≈$0.12 (700 tokens); $0.74 (4.5k tokens) | 2.5 s | no |
| Claim-decomposed judge (Ragas-style) | $0.5-10 | 5-20 s | no |
| Patronus evaluator API | $10-20 [S] | seconds | some |
| Vectara HHEM-2.3 | $100K+/yr platform | n/a | $500K on-prem |
| HHEM-2.1-open, MiniCheck small, LettuceDetect | self-host, ≈$0.005-0.02 | 10-300 ms on GPU, about 1-2 s on CPU | yes |

**What follows:**
- The floor for *hosted generic grounding* is now **about $0.03 per 1,000 checks at 100-500 ms**.
- Small encoders (LettuceDetect, MiniCheck, FactCG) are about 10 times cheaper than a 4B reader and score about
  75 on AggreFact.
- At 4B parameters with two passes, our compute (≈$0.04 per 1,000) matches the API price of a 27B decision model.
  **We cannot win on cost per check. At best we tie, and only on GPU.**
- Our CPU latency (1.4-5.9 s) misses the commonly cited guardrail budget of about 100-200 ms for synchronous
  chat. GPU serving, a single pass or a smaller reader is needed for an inline product. An async/audit product
  can live with seconds.

---

## 4. What enterprise buyers require

Practitioner and vendor sources agree on the following, but there is no rigorous survey [S]
([tianpan.co](https://tianpan.co/blog/2025-10-22-llm-guardrails-production-guide),
[premai](https://blog.premai.io/production-llm-guardrails-nemo-guardrails-ai-llama-guard-compared)):

- **Latency:**
  - Inline guardrails need **under 100-150 ms p50 and under 200 ms p99**, or must run in parallel with streaming.
  - A synchronous LLM judge adds at least 500 ms, which is why judges get pushed to async or offline eval.
  - Galileo markets "10-20 metrics in <200 ms" for exactly this reason.
- **False positives come first:**
  - Blocked legitimate answers erode trust faster than an occasional miss.
  - False alarms compound across stacked guards: five guards at 90% accuracy each flag about 41% of good traffic.
  - Buyers tune thresholds per domain and want the false-positive rate measured on their own labeled set.
    **That is exactly our calibrated-threshold promise, but nobody turns it into a contractual number.**
- **Audit trails and explanations:**
  - Per-sentence verdicts with the evidence span, versioned models (decision-model answers drift between versions),
    and stored scores.
  - Regulated buyers (EU AI Act high-risk, finance model-risk management under SR 11-7, healthcare) want a
    **model card with held-out error rates**.
- **Deployment:**
  - VPC or on-prem for legal, finance and health.
  - Vectara charges a 5× premium for on-prem ($500K against $100K), which shows on-prem is a high-value tier.
  - Closed decision models (Jev, d1-ultra, Nimble) cannot serve this. The open 27B ones need an 80 GB-class GPU.
- **Robustness:** prompt-injection resistance (12% of Jev's groundedness verdicts flipped by one embedded line),
  multilingual support (HHEM-2.3 has 11 languages; decision models are weak on CJK), and long context (32k-128k).
- **Integration:** it should plug into existing observability (Splunk/Galileo, Dynatrace/Arize, Datadog,
  Braintrust, Langfuse). Most budget is bought **through platforms**, and the platforms are being acquired.

### Market signals

- AI guardrails platforms: about **$3.1B in 2026, about $7.3B by 2030 (24% CAGR)** [S, analyst report;
  [Research and Markets](https://www.researchandmarkets.com/reports/6231463/ai-guardrails-platform-market-report)].
  Hallucination-detection-only estimates vary wildly and are not credible.
- **Consolidation** (2025-2026):
  - Cisco/Splunk bought Galileo.
  - Dynatrace is buying Arize (about $915M).
  - Harvey bought Guardrails AI.
  - Check Point bought Lakera (about $300M).
  - Galileo raised a $45M Series B in Feb 2026 just before its sale; Patronus raised a $50M Series B.
  - Exit pattern: **evaluation and guardrail startups are bought by observability, security and vertical
    application companies**. They are bought for distribution and trust, not for state-of-the-art accuracy.
- **Commoditization:** within three weeks of Jev, OpenAI, AWS, Cloudflare, Perplexity, Liquid and Microsoft all
  shipped decision models. Hyperscalers are giving away the generic grounding score.

---

## 5. Gaps nobody fills well, and our best wedge

| gap | who comes closest | why it is open | fit for us |
|---|---|---|---|
| **A stated false-alarm rate per customer domain, from 100-300 labels, with confidence bounds** | Galileo (tune with "a handful" of corrections, no guarantee); academic conformal work ([ACL 2026 multiple testing](https://aclanthology.org/2026.findings-acl.1705.pdf), [arXiv 2512.15068](https://arxiv.org/abs/2512.15068)) | vendors sell scores and claim global calibration; independent checks show miscalibration and domain shift | **strong**: we have it working (10.7% measured against a 10% target, violated in 8% of splits; it holds from 100 labels on) |
| **On-prem small verifier with sentence-level evidence** | HHEM-2.1, MiniCheck and LettuceDetect (small but about 72-79 accuracy); Clef and pplx-decider (accurate but 27B) | no open, commercially licensed model under 10B at ≥78 on AggreFact | medium: we are 4B, but our accuracy is unproven on AggreFact |
| **Closed-book chat (no evidence)** | Cleanlab TLM (sampling, expensive); entity probes (research) | every grounding product needs a context | **strong if we read the customer's own generator** (white-box, as in Goodfire and Obeso et al.); weak with an external reader (our closed-book result: 1-2 of 13 caught) |
| **Long reasoning traces** (math, agent steps) | process reward models (research), LLM judges | decision models score final answers; contexts are long | medium: probes on internals transfer to MATH errors (Obeso et al.), unproven by us |
| **Numbers in financial and medical documents** | TLM (0.85 on FinanceBench) | encoders and our probe are near chance on FinanceBench | weak today; arithmetic re-computation is our partial answer |
| **Signals that resist prompt injection** | none | text classifiers read the injected text | possible: internal signals plus a calibrated threshold; untested |
| **Model-agnostic assurance and a model-card layer** | Galileo/Splunk (inside the platform) | buyers will run several scorers | **strong**: our `calibrate`/`evaluate` pipeline is scorer-agnostic in principle |

**Recommended positioning: "the guarantee layer", not "a cheaper detector".**

1. **Core promise.** "At most X% of your good answers flagged, with 95% confidence, measured on your traffic from
   about 300 labels. Re-certified when your model or documents change." The deliverable is a model card. Price
   it per domain certification plus a self-hosted licence. Do not compete per check against $0.03 per 1,000.
2. **Scorer-agnostic.**
   - Combine our probe features with an **open decision model's probability** (Clef-Flash 9B or pplx-decider,
     both Apache-2.0) and/or MiniCheck as extra features under one calibrated threshold.
   - This turns the commoditized models from threats into inputs.
   - Raw AUROC then comes from the best open scorer. Our value is calibration, localization and assurance.
3. **White-box mode for open-model shops.** When the customer serves Llama/Qwen/Mistral themselves, probes on
   *their* generator's activations cost almost nothing per reply. This is the Goodfire pricing logic at
   $0.185 per 1,000. It is also the only cheap route to closed-book and long-form detection.
4. **Regulated on-prem first.** Legal, finance and health buyers pay on-prem premiums (Vectara's 5×), need audit
   trails, and cannot use the closed APIs.

**What not to claim:** "most accurate", "beats MiniCheck-7B", or "real-time" (until measured on GPU). Our
HaluBench claim stays "at least level with MiniCheck-RoBERTa", and that model is 3.9 points below the board's
7B models.

---

## 6. A minimal credible product and evidence package

### 6.1 Benchmarks (public, reproducible, with confidence intervals)

| benchmark | protocol | bar for "credible" | bar for "SOTA" |
|---|---|---|---|
| **LLM-AggreFact**, full test set, 11 datasets | balanced accuracy; threshold from the dev split (leaderboard protocol); submit to the board | ≥77.4 (MiniCheck-7B); at 4B, ≥75.6 (FactCG) at minimum and above Qwen3-4B prompted (73.8) | ≥79.5 |
| **RAGTruth** test set (2,700) | response-level F1 *and* span-level F1, per task (QA/summary/data2text) | ≥79 response F1 (LettuceDetect) with in-domain training | ≥84 (RAG-HAT) |
| **HaluBench** full set (about 15k) | accuracy and AUROC per source | ≥85% acc / ≥0.85 AUROC | ≥87.4% (Lynx-70B) |
| **Artifact controls** | PARALLAX-style text-only baseline; leaked-answer check | probe must clearly beat a text-similarity baseline | - |
| **Closed-book / long-form** (LongFact entities, FActScore bios) | AUC and recall at 10% FPR; white-box on the generator | ≥0.85 AUC | ≥0.90 |

**Head-to-head set**, on the same machine with measured latency and cost:
- MiniCheck-RoBERTa/Flan-T5 and Bespoke-MiniCheck-7B (via API);
- HHEM-2.1-open; LettuceDetect-large; Granite Guardian 3.3;
- **Clef-Flash and pplx-decider (open)**; **Jev / Nimble (APIs, cheap)**;
- a gpt-4o-mini judge.

### 6.2 The guarantee evidence (our differentiator; nobody else publishes this)

- **At least 5 domains** (RAGTruth QA, RAGTruth summary, a medical source, a financial source, plus one
  design-partner domain). For each one: label budgets of 50/100/200/300, 500+ random splits, and the **rate at
  which the 95% bound is violated** (target ≤5%) together with recall at each budget.
- **The same protocol applied to the competitors' scores.** Show that their global thresholds miss the target
  false-alarm rate per domain, and that our calibration layer fixes *their* scores too. This shows
  the layer is worth paying for whichever scorer wins.
- **Drift test:** a threshold certified on generator A, then replies from generator B. Show detection of a broken
  guarantee and re-certification.

### 6.3 Product minimum

- `POST /v1/verify` returns per-sentence verdicts with the evidence span, a calibrated probability and the
  certified false-alarm rate. The pieces already exist in `vt/verify_service.py`.
- **Measured GPU latency** p50/p95 on L4 and T4 at 700 and 4k tokens, with a target of **<200 ms p50**. If it
  misses, offer an "async audit" tier and a smaller or single-pass reader for inline use.
- Self-hosted container (CPU and CUDA), signed model versions, an immutable score log for audit, and the model
  card per domain.
- Integrations: an OpenTelemetry export, plus plugins for Langfuse/Braintrust/Arize Phoenix/Splunk, because
  budgets are spent inside those tools.
- Pricing to test:
  - per-domain certification fee (a few thousand dollars, model card included);
  - an annual on-prem licence (five figures; compare Vectara's $500K on-prem tier as the ceiling);
  - a metered hosted API *near the decision-model floor* ($0.05-0.10 per 1,000 checks) as the entry point, not
    the $0.50-1.00 in `docs/product.md`.

### 6.4 Decision points

- **If our probe reaches about 77+ on LLM-AggreFact at 4B** (or a probe combined with a decision model reaches 80+),
  sell accuracy *and* guarantees.
- **If it stays below about 75**, use an open decision model or MiniCheck as the main scorer. Sell calibration,
  localization, white-box closed-book detection and on-prem assurance. Keep the probe as an added feature,
  especially where we own the generator.

---

## 7. Open uncertainties

- Many 2026 facts about decision models (sizes, prices, the "Qwen3.8" base) come from aggregators within weeks of
  launch. The AggreFact numbers are from the leaderboard itself [P], but the leaderboard does not say who submitted
  them or how thresholds were set for decision models.
- Prices for Bespoke MiniCheck, Cleanlab TLM, Patronus and Azure groundedness are not published or not verified.
  Vectara's tiers come from a third party.
- No independent HaluBench or RAGTruth numbers exist for the decision models or HHEM-2.3.
- Buyer latency and false-positive tolerances come from practitioner blogs, not surveys. Design-partner interviews
  should confirm them.
- Market size figures come from analyst reports of unknown quality. Treat them only as a sign of direction.
