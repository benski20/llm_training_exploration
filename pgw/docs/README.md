# PGW documentation

PGW (Private Gateway) runs private LLM inference on public open-weight models. Our compute service
does the heavy matrix products on one-time-masked data. Your gateway keeps all plaintext. The
normative technical specification is [SPEC.md](../SPEC.md). These documents explain it for security
reviewers, operators, developers and compliance teams.

## Documents

| audience | document | contents |
|---|---|---|
| Security architects, auditors | [Security whitepaper](security/whitepaper.md) | scheme, proof sketch of perfect secrecy per one-time use, Freivalds integrity bound, what is and is not protected, collusion and supply-chain risks |
| Security architects | [Threat model (STRIDE)](security/threat-model.md) | assets, trust boundaries, data flows, STRIDE tables, residual risks |
| Customer administrators | [Getting started with an enrollment code](operations/onboarding.md) | plain-language setup: one enrollment code and the console setup wizard; for operators: issuing enrollment codes |
| Operators | [On-prem gateway](operations/install-on-prem.md) | topology A: install, configure, verify, operate |
| Operators | [Gateway in your cloud account](operations/install-customer-cloud.md) | topology B: private link, Helm, KMS, cloud storage |
| Operators | [Mask factory air-gapped from us](operations/air-gapped-factory.md) | offline factory, partner delivery with HPKE, media handling |
| Operators, architects | [Sizing](operations/sizing.md) | bandwidth, mask storage, factory throughput, latency, with measured numbers |
| Operators, security | [Console single sign-on](operations/console-sso.md) | OpenID Connect sign-in for the operator console: settings, role mapping, Entra ID, Okta and Keycloak examples |
| Operators, security | [Encrypted mask generation](operations/encrypted-masks.md) | no factory GPU and no large mask store: the service computes W8.r on MLWE-encrypted masks |
| Operators, security | [Operator console](operations/console.md) | enabling the web console, roles and tokens, sessions/CSRF/CSP, network exposure, playground quotas, audit data, troubleshooting |
| Developers | [Gateway API reference](api/gateway-api.md) | OpenAI-compatible endpoints, streaming, errors, limits, admin API |
| Operators, frontend | [Console API contract](api/console-api.md) | Web console sign-in, roles, usage/audit/overview endpoints, security headers |
| Legal, compliance | [Compliance FAQ (draft, needs legal review)](compliance/faq.md) | GDPR, HIPAA, FERPA and other questions |
| Auditors, security teams | [Audit-readiness checklist](compliance/audit-readiness.md) | controls, evidence, owners |
| Everyone | [Glossary](glossary.md) | terms and symbols |

Client library (`pip install pgw-sdk`, `from pgw import OpenAI`: the gateway runs in the application's process):

| audience | document | contents |
|---|---|---|
| Developers | [Quickstart](sdk/quickstart.md) | install, key, first call, streaming, async |
| Developers, architects | [Concepts](sdk/concepts.md) | what stays private, verification, what runs where, why the application runs near the GPU |
| Developers | [Configuration reference](sdk/configuration.md) | client options, request parameters, environment variables, cache directory |
| Developers | [Supported models](sdk/models.md) | Qwen2.5-Instruct 0.5B-7B, limits, resources per model |
| Developers, operators | [Troubleshooting](sdk/troubleshooting.md) | installation, first start, speed, error codes, reporting issues |
| Security reviewers | [Security model summary](sdk/security.md) | what is and is not protected, review status (pks-v3 pending external review) |
| Pilot customers | [Pilot hand-off](sdk/pilot-handoff.md) | one page: what you get, five-minute setup on RunPod, what to test, privacy promise |
| Everyone | [Changelog](sdk/changelog.md) | client library releases |
| Maintainers | [Build contract](sdk/CONTRACT.md) | internal: interfaces between the client library's components |
| Designers, maintainers | [Design system](design/README.md) | the shared console and platform design system, reference screenshots |

Tools and data:

* [`measurements.json`](measurements.json) is the single source of every quoted number, with its
  provenance (Kaggle T4 runs, analytic formulas, estimates).
* [`tools/pgw_sizing.py`](tools/pgw_sizing.py) is a sizing calculator (standard library only):
  `python docs/tools/pgw_sizing.py --model llama-3-8b --ring K48A20 --tokens-per-day 1000000`.

## Conventions

* Numbers are labelled **measured** (observed in a run), **analytic** (exact from model shapes and
  ring), **derived** (arithmetic on measured values) or **estimate** (based on assumptions; validate
  before relying on it). MB means 10^6 bytes, and TB means 10^12 bytes.
* "Planned" marks features that are not in the v1.0 build.
* The documents are checked by `pytest tests/docs`. The checks cover CLI commands, environment
  variables, error codes, endpoints and metric names against the code, the numbers against
  `measurements.json` and the sizing calculator, and links.

## Key facts at a glance

| | value | basis |
|---|---|---|
| Exactness | masked equals quantized plaintext: identical tokens, max logit difference 0.0 (Qwen2.5-0.5B) | measured |
| Server cost, 8B, T4 | 0.45 s/token batch 1; 0.014 s/token amortized at batch 32 | measured |
| Wire per token, 8B, K48A20 | 13.4 MB (5.1 MB up, 8.3 MB down) | analytic |
| Masks per token | 8.3 MB at K48A20 and 6.9 MB at K40A12 for 8B; 1.5 MB at K40A12 for Qwen2.5-0.5B | analytic |
| Mask generation, 8B, K48A20, T4 | 0.0066 s/slot (550k slots/h) | measured |
| CPU int8 GEMM, 4 vCPU (fbgemm) | 0.30 TOPS | measured |
| Round trips per token | 128 round trips for 8B, 320 round trips for 70B | analytic |
| Integrity, K40A12, width 4096, c = 3 (auto default) | false accept ≤ 2^−24 per call (floor: 2^−20) | analytic |
