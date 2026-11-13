# Grounding guard: getting from product to customer

What is built and measured is in [product.md](product.md) ("What ships now"). What is missing is a customer: no
number so far is on real traffic, and the prices are proposals. This page is the plan: who to ask, what to show,
what to ask, how a pilot runs, and what counts as validated.

## One-pager (for buyers)

**Check every answer your AI writes from your documents, at a false-alarm rate you set and can audit.**

RAG assistants and document Q&A tools state things their documents do not say. Detectors exist (most open,
some free); none tells you how often they will wrongly block a good answer *on your traffic*. Ours does:

1. **Calibrate** on 60-300 answers you know are right (reviewed tickets, approved replies). No examples of
   hallucinations are needed.
2. **Choose the cap**: for example, at most 5% of good answers like these flagged, with 95% confidence.
3. **Drop it in**: change your OpenAI-compatible base URL to the guard; each answer comes back marked supported
   or unsupported, with the unsupported words highlighted, or is blocked, your choice.
4. **Audit** any time: send new known-good answers, and the report says whether the cap held.

Measured on public benchmarks:
* LettuceDetect's default cut wrongly flags 8-14% of good answers per RAGTruth task; calibrated to "at most 10%",
  2.4-5.5%, while catching 43-62% of hallucinations;
* end to end through the API: 0 of 40 new good answers flagged at a 5% cap; 7 of 20 labelled hallucinations
  caught, each with the right words highlighted;
* runs on a 4-core CPU, about 1-2.5 s a check, as one container with networking off: answers and documents never
  leave your network.

Limits, said up front: at a strict cap it catches a third to a half of hallucinations, not all; medical papers and
numbers in financial filings are its weakest ground; it checks against the documents you pass, not the world;
streaming replies are checked only when complete.

## Who to ask first

| segment | why they might pay | who to reach |
|---|---|---|
| Teams shipping support or "ask the docs" assistants | a wrong answer to a customer costs a ticket or a refund; a guard that over-blocks costs the feature | head of support automation, AI product manager |
| Legal, compliance, insurance, healthcare document Q&A | an unsupported statement is a liability; data cannot leave their network | knowledge-management lead, risk or compliance officer, CTO |
| RAG platforms and LLM observability vendors | want a cheap grounding signal with a defensible error rate to embed | product lead for evaluation or guardrails |

Start with ten conversations, three or more per segment.

## Interview (20 minutes)

Do not pitch first. Learn how they decide today.

1. How do you find out today that your assistant said something your documents do not support?
2. When did a wrong answer last reach a customer or a decision? What did it cost?
3. Do you check answers now (LLM judge, a detector, human review, sampling)? What does it cost a month?
4. If a guard blocked good answers, what share would be too many? Who would complain first?
5. *(show the one-pager)* Would a cap you set yourself, with an audit, change what you use? Why, or why not?
6. Can answers and documents leave your network? Would self-hosting matter?
7. Price: at what price per 1,000 answers checked would this be too cheap to trust, a bargain, getting
   expensive, too expensive for your volume?
8. Would you run a free 30-day pilot on your own answers? Who would need to approve it?

Write down the exact words for questions 2, 4 and 7.

## Pilot (30 days, free)

The customer sends 100-300 answers they know are supported, with the question and the documents each was written
from; if they can, a second set of new known-good answers and any answers they know were wrong.

```bash
docker run -d -p 8000:8000 -e VERIFY_API_KEYS="pilot:$KEY" -v pilot-cal:/data verify      # Dockerfile.verify
curl -H "Authorization: Bearer $KEY" localhost:8000/v1/verify/calibrate -d '{"replies": [...good...], "fpr": 0.05}'
curl -H "Authorization: Bearer $KEY" localhost:8000/v1/verify/audit -d '{"replies": [...new good...]}'
curl -H "Authorization: Bearer $KEY" localhost:8000/v1/verify/batch -d '{"items": [...known wrong...]}'
```

or, in their app, the guard in front of their model provider (`vt/guard_proxy.py`). The pilot report: their audit
(flagged share, interval, consistent or exceeded), the share of their known-wrong answers caught with the words
highlighted, and seconds per check on their hardware.

## What counts as validated

* **Problem:** at least 5 of 10 interviewees name a wrong answer that cost them something (question 2), and say
  how many wrong blocks they would tolerate (question 4).
* **Price:** the median "getting expensive" answer (question 7) is at or above the proposed price ($0.50-1.00 per
  1,000 checks hosted; an annual licence self-hosted).
* **Customer:** one signed pilot whose audit comes back "consistent" on their own answers, then one paid
  conversion or written intent to buy at a stated price.
