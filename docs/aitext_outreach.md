# AI-text detector: getting from product to customer

What is built and measured is in [aitext.md](aitext.md). What is missing is evidence that someone will pay. This
page is the plan for getting it: who to ask, what to show, what to ask, and what counts as validated.

## One-pager (for buyers)

**AI-text detection where you set the false-accusation rate, and can check it.**

Every AI detector publishes an accuracy figure measured on its own data. None tells you how often it will accuse
*your* students, authors or applicants. Ours does:

1. **Calibrate** on 150 texts you know are human: last year's essays, past submissions, archived articles.
2. **Choose the cap**: for example, at most 5% of human texts like these flagged, with 95% confidence.
3. **Check** new texts: each gets "AI" or "not flagged".
4. **Audit** whenever you like: run new human writing through it, and the report says whether the cap held.

Measured on public benchmarks the detector never trained on (RAID; HAP-E):
* 1.3-1.5% of human texts flagged when 5% was promised;
* 74% of GPT-4o and Llama-3 text caught, and 74% of paraphrased AI text;
* runs on an ordinary 4-core server: about 2 seconds a text, no GPU, and it can run inside your own network.

Limits, said up front: it reads the first 256 tokens of a text; it is weaker on poetry, short product reviews and
text continued by base models; Claude, Gemini and reasoning models are not yet tested.

## Who to ask first

| segment | why they might pay | who to reach |
|---|---|---|
| Universities and schools | false accusations are their public problem; most cannot send student work to an outside API | academic-integrity office, teaching and learning centre, learning-technology lead |
| Publishers, magazines, contests | must screen submissions; a wrongly rejected author is a reputational cost | managing editor, submissions or contest manager |
| Hiring and freelance platforms | screen applications and samples at volume; need a defensible error rate | trust-and-safety lead, product manager for applications |

Start with ten conversations, three or more per segment.

## Interview (20 minutes)

Do not pitch first. Learn how they decide today.

1. How do you deal with AI-written submissions now? Which tool, if any?
2. When did a detector last flag someone who had written their own work? What happened?
3. Who decides to buy a tool like this, and from which budget?
4. What do you pay now, and for how many checks a month?
5. *(show the one-pager)* Would a cap you set yourself, with an audit, change what you use? Why, or why not?
6. Can student, author or applicant text leave your network? Would self-hosting matter?
7. Price: at what monthly price would this be too cheap to trust, a bargain, getting expensive, too expensive
   for your volume?
8. Would you run a free 30-day pilot on your own texts? Who would need to approve it?

Write down the exact words for questions 2, 5 and 7.

## Pilot (30 days, free)

The customer sends 150+ human texts (calibration) and, if they can, a separate set of new human texts plus any AI
texts they have. We run:

```bash
vt-aitext calibrate --human calibration.jsonl --out customer.json --fpr 0.05
vt-aitext audit --calibration customer.json --human new_human.jsonl          # did the cap hold on their writing?
vt-aitext check --calibration customer.json --file ai_examples.txt           # how much of their AI text is caught
```

or give them an API key on `vt-aitext serve --calibration-dir ...`. The pilot report is their audit (flagged
share, interval, consistent or exceeded) plus the share of their AI examples caught.

## What counts as validated

* **Problem:** at least 5 of 10 interviewees name a false accusation that cost them something (question 2).
* **Price:** the median "getting expensive" answer (question 7) is at or above the proposed price for that
  segment ($500 a month institution plan; $1 per 1,000 checks API).
* **Customer:** one signed pilot, then one paid conversion or written intent to buy at a stated price.

Until those three are met, the prices in [aitext.md](aitext.md) are proposals.
