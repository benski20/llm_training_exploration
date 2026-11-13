# Calibrated AI-text detection

**The product.** A detector that says whether a text was written by an AI model. Its false-accusation rate is
a promise the customer sets and can audit: send 150 of your own human-written texts, choose the rate (say 5%), and
the detector picks its thresholds so that at most that share of human texts like yours is flagged, with 95%
confidence. No AI samples are needed from you. It runs on a CPU: two small open models (Qwen3-0.6B base and
Qwen3-0.6B, Apache-2.0) read the first 256 tokens of each text.

Why this angle: false accusations, of students above all, are the main complaint against AI detectors. Existing
tools publish an accuracy figure measured on their own data; none lets a customer set the false-accusation rate
for their own kind of writing and get a bound on it.

## How it decides

Two scores per text:

| score | what it reads | trained? |
|---|---|---|
| **probe** | the observer's mean hidden state at layers 7/14/21/28 plus seven statistics (log-perplexity, cross-perplexity, entropy, ...) | logistic regression on 5,280 RAID texts (8 domains, 11 generators, plain and paraphrased): `heads/aitext_probe_qwen3_0.6b.npz`, 51 KB |
| **Binoculars** (Hans et al., 2024) | performer log-perplexity / observer-performer cross-entropy | no (zero-shot) |

A text is flagged when **either** score passes its threshold. Each threshold is the one-sided Clopper-Pearson
threshold over the customer's human texts at half the false-alarm budget (2.5%) and half the miss confidence
(97.5%), so by the union bound the pair keeps "at most 5% flagged, with 95% confidence". With fewer than 146
human texts no threshold can make that promise, and calibration refuses rather than guess.

## Measured (RAID, results/ai_text/)

Every number is on a writing domain the probe never saw in training (leave one domain out, 8 domains). The promise
is checked over 50 random calibrations per domain: 150 human texts calibrate, the domain's other human texts and
its AI texts test.

| detector, at the 5% / 95% promise | measured false alarms | promise broken in | AI caught | paraphrased AI caught |
|---|---|---|---|---|
| Binoculars alone | 2.0% | 6.2% of calibrations | 51% | 72% |
| probe alone | 1.9% | 2.8% | **82%** | 35% |
| **probe + Binoculars (shipped)** | **1.3%** | **1.3%** | 77% | **74%** |

* Per domain, the shipped pair catches 86-93% of AI text in abstracts, books, news and recipes; 76-80% in
  Reddit and Wikipedia; about 50% in poetry and product reviews. A promise is kept everywhere: no domain broke it
  in more than 4% of calibrations, against the 5% allowed.
* **Unseen generators.** With both the test domain and the test generator's whole family held out of training
  (OpenAI, Meta, Mistral, MosaicML, Cohere), the probe's AUROC is 0.962 on average (0.92 Cohere to 0.996 Meta)
  and it catches 79% at the promise; Binoculars 0.785 and 51% (`raid_generalize.json`, `unseen_family`).
  This was measured for the probe trained without paraphrases; the shipped pair's unseen-family figure is not
  yet measured.
* **Newer generators, new genres (HAP-E).** The shipped detector, trained on RAID only, on HAP-E
  (`hape_modern.json`): continuations of human texts by GPT-4o, GPT-4o-mini and Llama-3 8B/70B (base and
  instruct), in six genres, three of them (fiction, spoken, TV/movie scripts) absent from RAID. Each genre
  calibrated on 150 of its own human texts, tested on its other 100 and 300 AI texts, 50 times:

  | generator | probe AUROC | Binoculars AUROC | caught at the 5% promise |
  |---|---|---|---|
  | GPT-4o (2024-08-06) | **0.974** | 0.591 | 65% |
  | GPT-4o-mini | **0.979** | 0.706 | 76% |
  | Llama-3-70B-Instruct | 0.984 | 0.976 | 96% |
  | Llama-3-8B-Instruct | 0.984 | 0.979 | 94% |
  | Llama-3-70B (base) | 0.862 | 0.888 | 44% |
  | Llama-3-8B (base) | 0.921 | 0.962 | 69% |

  Measured false alarms 1.5% (promise 5%), the promise broken in 3.3% of 300 calibrations (news: 10%, 5 of 50;
  every other genre at most 4%); 74% of AI text caught (59% academic to 95% spoken). GPT-4o is where the trained
  probe earns its place: Binoculars barely separates it (0.59). Base models continuing a human text are the
  hardest case.
* **A threshold does not carry between kinds of writing.** Set at 1% false alarms on seven domains, the probe
  flagged 29% of human recipes (`raid_qwen3_0.6b.json`, `transfer_1pct`). This is why calibration on the
  customer's own writing is the product, not an option.
* **Hosted cost, measured** (`service_bench.json`, `experiments/aitext_service_bench.py`): the real API server
  over HTTP on 4 vCPUs (Intel Xeon, 2.3 GHz): 2.19 s a request (median; p90 2.39 s), 2.04 s a text in batches
  of 8, 1,764 texts an hour, peak memory 7.9 GB, model load 6 s, calibration on 150 texts 6 minutes. At the
  on-demand price of a 4-vCPU, 16 GB AWS m7i.xlarge (us-east-1, $0.2016 an hour) that is **$0.11 per 1,000
  texts**, about $0.0006 per 1,000 words read. Spot capacity ($0.075-0.094 an hour) halves it; a GPU would
  cut it further.

## Price (proposed, not yet validated with a buyer)

Published API prices of incumbents, from third-party listings (check each vendor before relying on them):
GPTZero about $135 for 1M words ($0.135 per 1,000 words); Pangram about $0.05 per 100 words
($0.50 per 1,000 words); Originality.ai's API from about $137 a month; Copyleaks by quote.

| plan | price | basis |
|---|---|---|
| API | $1 per 1,000 texts checked (up to 256 tokens read each), calibrations free | ~90% gross margin on the measured $0.11 |
| Institution | $500 a month: 1M checks, private calibrations per course or department, monthly audit report of measured false alarms on new human texts | |
| Self-hosted | annual licence; runs on the customer's own CPU, no text leaves their network | for schools and publishers that cannot send student or author text out |

The distinctive item is the audit: because the promise is a statement about human texts, a customer can test it
on their own new human writing at any time, and the report says whether it held.

## Limits, stated

* **The human texts in both tests are mostly from before ChatGPT.** Human writing today is partly AI-edited;
  whether that writing counts as human is the customer's policy, and the calibration set should reflect it.
  GPT-4o and Llama 3 are tested (HAP-E); Claude, Gemini and reasoning models are not.
* **Only the first 256 tokens are read.** A long document mixing human and AI passages is judged on its start.
* **Only the paraphrase attack was tested.** RAID's other attacks (homoglyphs, misspellings, synonym swaps,
  whitespace) and humanizer tools were not.
* **Poetry and reviews**: at 5% the pair catches only about half of AI text.
* **Short texts**: the measurements are on texts of 40+ words, read up to 256 tokens. Texts under 64 tokens get a
  note with their verdict.
* **The promise covers human texts like the calibration set.** If a customer calibrates on essays and checks
  emails, the bound does not apply. It is a statement about the share of human texts flagged, not about any one
  text.

## Use

```bash
pip install ".[verifier]"
vt-aitext calibrate --human my_human_texts.jsonl --out calibration.json      # 150+ texts; {"text": ...} per line
vt-aitext check --calibration calibration.json --file to_check.txt
vt-aitext serve --calibration-dir calibrations/ --port 8000                  # API keys: VERIFY_API_KEYS="acme:key1"
```

API (same server hardening, API keys, per-key rate limits and `/metrics` as the grounding verifier):

    POST /v1/aitext/calibrate  {"human_texts": [...150+...], "fpr"?: 0.05, "confidence"?: 0.95}
                               -> the calibration, stored for this API key
    POST /v1/aitext            {"text": "..."}
                               -> {"verdict": "ai" | "not flagged", "fired": ["probe" | "binoculars"],
                                   "probe", "binoculars", "tokens", "promise", "ms"}
    POST /v1/aitext/batch      {"items": [{"text": ...}, ...up to 16]}

Reproduce: `experiments/ai_text_detect.py` (sample RAID, read, evaluate) and `experiments/ai_text_generalize.py`
(unseen generator families, paraphrase, the union).
