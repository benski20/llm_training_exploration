# Business-document hallucination test set (`business_test.jsonl`)

**Evaluation only.** Never use this file for training, tuning thresholds, prompt selection or
few-shot examples. It is a held-out set for showing how the detector does on the kind of
documents business buyers actually have.

## What it is

60 fictional business documents, each with one realistic user question and 4 AI-assistant answers
(240 records). Each document has 2 faithful answers and 2 hallucinated answers, so exactly half the
records are label 1. Each hallucinated answer contains exactly one subtle error, and the two
hallucinated answers for a document have different error types. The rest of each hallucinated answer
is correct and fluent. The error falls in the first sentence 31 times, a middle sentence 32 times and
the last sentence 57 times. Answer order within a document (`a`-`d`) is shuffled.

Documents are 193-326 words (mean 255). They include plain-text tables, prices, dates, durations,
eligibility conditions, exceptions and cross-references. Many questions require combining two
clauses or doing a small calculation, such as prorations, deductibles, date windows, business-hour
SLAs, tiered fees or sequential discounts.

Record schema (one JSON object per line):

```
{"id": "biz_<doc#>_<a|b|c|d>", "domain": ..., "question": ..., "documents": [<document text>],
 "response": ..., "label": 0|1, "error_type": null|<type>,
 "spans": [[start, end], ...], "explanation": null|<one sentence>}
```

`spans` are character offsets into `response`. `response[start:end]` is the erroneous text. Most
records have one span. Three records (`biz_16`, `biz_24` and `biz_47`) have a second span covering a
figure that follows directly from the same single error.

## Counts

Documents by domain (each document has 4 records):

| domain | docs |
|---|---|
| hr_leave_benefits (parental leave, PTO, medical plans, bereavement, tuition) | 5 |
| saas_contract_sla (uptime credits, renewal/termination, support SLA, DPA, order form) | 5 |
| api_documentation (rate limits, webhooks, OAuth, deprecation) | 4 |
| bank_fee_schedule | 4 |
| financial_report (quarterly excerpts, cash flow, guidance, KPIs) | 4 |
| insurance_coverage (travel, home, pet, dental) | 4 |
| it_security_policy | 4 |
| product_specs_pricing | 4 |
| refund_returns | 4 |
| travel_expense | 4 |
| lease_agreement (commercial, residential, vehicle) | 3 |
| medication_leaflet (fictional drugs) | 3 |
| meeting_minutes | 3 |
| onboarding_guide | 3 |
| shipping_terms | 3 |
| support_kb | 3 |

Hallucinated answers by error type (12 each, 120 total): `number_change`,
`date_or_time_arithmetic`, `calculation_error`, `entity_swap`, `negation_flip`,
`dropped_condition`, `unsupported_addition`, `conflated_clauses`, `wrong_unit_or_currency`,
`overgeneralization`.

## How it was made

- Everything was written by an AI assistant (Claude). Nothing was downloaded or copied, and no
  model or API generated any text. All companies, products, people and drugs are fictional. The
  medication leaflets are illustrations, not medical advice.
- For each hallucinated answer, the author recorded the exact erroneous substring. A script then
  computed the offsets and checked that each substring occurs exactly once in its response.
- Labels were self-checked. Every hallucinated answer was re-read against its document to confirm
  that the error is real and is the only error. Every faithful answer was re-read to confirm that
  each claim is supported. A faithful answer can add a harmless hedge, or say correctly that the
  document does not cover something. Arithmetic and date calculations were checked with code. The
  labels have not been reviewed by a human, so treat them as high-quality but not infallible.

## Validation

```
python experiments/validate_business_test.py
```

The validator checks the schema, span bounds, that faithful records have no spans or error type,
the 2 + 2 structure per document, distinct error types per document, document length (150-700
words), answer length (1-6 sentences), exact 50/50 label balance and at least 8 records per error
type. It then prints the counts.
