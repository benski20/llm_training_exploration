# PGW threat model (STRIDE)

Version 1.0 (protocol v1). Companion to the [security whitepaper](whitepaper.md), which proves the
cryptographic properties this model relies on. SPEC section 3 is the normative summary.

Method: we list assets, trust boundaries and data flows, then go through each element with STRIDE
(**S**poofing, **T**ampering, **R**epudiation, **I**nformation disclosure, **D**enial of service,
**E**levation of privilege). Each threat has an id (`T-xx`), the controls that exist in the v1.0 build,
and the residual risk. "Planned" marks controls that are not in v1.0.

## 1. Assets

| id | asset | confidentiality | integrity | availability |
|---|---|---|---|---|
| A1 | Prompts, completions, activations, logits | **critical** | high | medium |
| A2 | Mask seeds, `W r` rows, Freivalds vectors `u`, `uᵀW` | **critical** (seeds and `u`) | **critical** | high (no masks, no inference) |
| A3 | Mask store state (cursors, generation counter, journal) | low | **critical** (rollback leads to reuse) | high |
| A4 | KEK / HSM / KMS keys, recipient X25519 private key | **critical** | high | high |
| A5 | Model pack (public weights) and its fingerprint | public | high | high |
| A6 | Gateway API keys, admin token, service tenant tokens, TLS keys | high | high | medium |
| A7 | Audit log, metrics | medium (no content by design) | high | medium |
| A8 | PGW software and container images | public or source-available | **critical** | high |

## 2. Trust boundaries and data flows

```
 TB1: customer boundary                                       TB2: our cloud
 ┌───────────────────────────────────────────────┐            ┌─────────────────────┐
 │ apps ─(F1 OpenAI API, TLS)─▶ gateway           │─(F4 x̂)────▶│ compute service     │
 │                              │  ▲              │◀─(F5 ŷ)────│ (untrusted)         │
 │              (F2 reserve/take)│  │(F3 segments) │  HTTP/2    └─────────────────────┘
 │                              ▼  │              │  mTLS
 │                           mask store ◀── factory (customer)                       │
 └───────────────────────────────────────────────┘
              ▲ F6 HPKE-sealed segments (file transfer)
 TB3: independent partner factory (optional)
 F7: model pack distribution (all parties, verified by fingerprint)
 F8: admin/ops: /admin/*, /metrics, /dashboard, CLI
```

| flow | from → to | content | protection |
|---|---|---|---|
| F1 | applications → gateway | prompts, completions | TLS (optionally mTLS), API keys `pgw-<id>-<secret>` (stored hashed), scopes, rate limits |
| F2 | gateway ↔ mask store | decrypted masks (in process memory) | same host or process; durable reservation before decryption |
| F3 | factory → store | encrypted segments | AES-256-GCM per chunk, DEK wrapped by KEK |
| F4/F5 | gateway ↔ service | `x̂`, `ŷ` (uniform ring elements), shapes | one-time masks (content); TLS 1.2+/mTLS plus a tenant token (channel) |
| F6 | partner → customer | segments | HPKE (RFC 9180) sealed DEK, AEAD on every chunk, verified on import |
| F7 | build → all | int8 weights, manifest | SHA-256 fingerprint compared by every party |
| F8 | operators → gateway | admin actions | admin token or `admin` scope; audit log |

## 3. STRIDE by element

Severity is the product of impact and likelihood after controls: **H**igh, **M**edium, **L**ow.

### 3.1 Compute service (untrusted, TB2)

| id | STRIDE | threat | controls (v1.0) | residual |
|---|---|---|---|---|
| T-01 | I | Service operator reads customer data from payloads | one-time masks: the view is uniform and independent of the data (whitepaper Theorem 2); scales never sent | **L** (PRG assumption, whitepaper 4.4) |
| T-02 | I | Service keeps payloads and analyses them later, including with future compute | same as T-01; the stored data is uniform noise; seeds never leave TB1 | L |
| T-03 | T | Service returns wrong products to corrupt outputs | range check plus `c` Freivalds checks on every response; accepted with probability ≤ `2^(−c·h)`; circuit breaker | L (K40A12/K48A20 with the auto `c`, floor 20 bits); K32A8 wide layers cannot reach the floor and are refused |
| T-04 | I | Selective-failure probing via crafted errors | Freivalds rejection independent of the plaintext; advantage ≤ `2^(−c·h)` per probe; breaker cuts off | L |
| T-05 | I | Metadata: prompt and generation lengths, timing, volume | accepted (SPEC 3.4); batching; padding and dummy rounds **planned** | **M** (accepted, disclosed) |
| T-06 | S | Service impersonated by a third party (DNS hijack, rogue endpoint) | TLS with pinned CA (`service_ca_file`), `require_service_tls`, egress allowlist; even an impersonator sees only masked data, and wrong answers fail Freivalds | L |
| T-07 | D | Service refuses or delays, or a breaker storm | retries with backoff, `unavailable` errors, multiple endpoints, SLA | M (availability is not cryptographic) |
| T-08 | T | Service serves a different model (fingerprint mismatch) | gateway checks `/pgw/v1/models` fingerprint and ring at startup and per request (`fp` field); Freivalds fails for wrong weights | L |
| T-09 | R | Service disputes billing or work done | per-tenant counters on the service; gateway audit log with `masks_consumed`, `request_id` | L |
| T-10 | E | Tenant A affects tenant B on a shared service | stateless service, per-tenant quotas; rows are uniform, so mixing is harmless; tenant tokens or certificates | L |

### 3.2 Gateway (TB1)

| id | STRIDE | threat | controls (v1.0) | residual |
|---|---|---|---|---|
| T-11 | S | Unauthorised application uses the API | API keys hashed with SHA-256, scopes (`inference`, `admin`), per-model restriction, anonymous access refused in production | L |
| T-12 | S | Stolen API key | per-key rate, token and concurrency limits, daily quotas, `DELETE /admin/keys/{id}`, audit `auth.failed` events | M (depends on customer key hygiene) |
| T-13 | T | Bug sends unmasked activations | outbound-noise monitor (chi-square, top-bit, correlation with `xq`), fail-closed; tests assert masked equals plain | L |
| T-14 | T | Mask reuse through a bug, crash or restored backup | durable reservation before decryption; rows never returned; rollback detection; duplicate-row detector; `mask_reuse` stops traffic | L |
| T-15 | R | User denies having sent a request | audit log (`inference` events: key id, model, tokens, status; no content) | L |
| T-16 | I | Prompts in logs, traces, metrics or crash dumps | logging rules (no content); audit field allowlist; metrics labels never carry prompt data; disable core dumps (operator guide) | L |
| T-17 | I | Gateway host compromise | **out of scope** (SPEC 1): host hardening, read-only container, non-root, seccomp, customer EDR | depends on the customer |
| T-18 | D | Request floods or long `max_tokens` exhaust masks | per-key RPM/TPM, `max_tokens_limit`, `max_body_bytes`, queue limits, `mask_exhausted` fail-fast, low-water alerts | L |
| T-19 | E | Admin endpoints reached by an application key | separate admin token or `admin` scope; constant-time comparison; optional `metrics_require_admin`; NetworkPolicy | L |
| T-20 | T | Malicious configuration (dev modes, sampled Freivalds, plain http) | production policy refuses `plain`/`float`, `freivalds_rate < 1`, anonymous access and non-loopback http unless `allow_dev_modes`; `/admin/status` shows `production_policy` | L |

### 3.3 Mask store (TB1)

| id | STRIDE | threat | controls (v1.0) | residual |
|---|---|---|---|---|
| T-21 | I | Disk or backup theft exposes masks | AES-256-GCM per chunk, per-segment DEK wrapped by KEK (file 0600, env, PKCS#11 or KMS); consumed segments shredded (wrapped DEK overwritten first); full-disk encryption, because overwrites are not reliable on copy-on-write file systems and SSDs | L (if the KEK is stored separately) |
| T-22 | T | Rollback of `state.sqlite` to replay slots (backup restore, VM or volume snapshot revert, malicious admin) | HMAC journal keyed from the KEK detects a restored database alone; a monotonic counter off the mask volume (TPM, HSM, provider, or file counter on separate storage) detects a whole-volume revert; `verify_integrity()` at open refuses to serve; runbooks forbid restoring the mask volume | L with a counter off the mask volume; **M** with journal only: restoring state *and* journal from the same snapshot needs no key and is not detected (default `counter = none`); **H** with a provider that cannot MAC and no counter (no witness at all) |
| T-23 | T | Tampered or truncated segment file | AEAD with header hash in the AAD; chunk index in nonce and AAD; fingerprint and ring checked | L |
| T-24 | D | Store exhaustion or disk full | low-water alerts (`pgw_masks_low_water`), factory refill scheduler, `mask_exhausted` | M (capacity planning; see [sizing](../operations/sizing.md)) |
| T-25 | R | Dispute about which masks were consumed | reservation records with `session_id`, consumed and burned counters, audit retention | L |
| T-26 | E | Another local process reads the KEK | key file mode checks, dedicated user, HSM or KMS options | L to M |

### 3.4 Mask factory (customer TB1, or partner TB3)

| id | STRIDE | threat | controls (v1.0) | residual |
|---|---|---|---|---|
| T-27 | I | **Factory colludes with the service operator** (shares seeds) | customer-run factory; partner contractually independent; HPKE delivery; operator-run factory forbidden | **M** for partner deployments (organisational); L for customer-run |
| T-28 | I | Backdoored factory build with predictable seeds | open review of seed generation; air-gapped factory; independent Rust implementation with known-answer vectors; customer entropy mixing **planned** | M (supply chain) |
| T-29 | T | Factory produces wrong `W r` (bug or malice) | Freivalds checks verify the **service**, not the factory; a wrong `W r` shifts `yq` and is caught by the range check with high probability for random corruption, but a small deliberate offset can pass; segments are AEAD-protected in transit and at rest; the factory is trusted for mask integrity (SPEC 3.1); spot-check of `W r` against the pack weights by a second party **planned** | M (integrity only, no privacy impact) |
| T-30 | S | Fake partner delivers segments | HPKE to the customer's key only proves confidentiality, not origin; segments carry `factory_id`; import verifies fingerprint, ring and every AEAD tag; deliver over an authenticated channel (signed transfer, SFTP with known host keys) | L to M |
| T-31 | D | Factory down, so stores run dry | inventory between `low_water_slots` and `target_slots`; alerting; buffer days of masks (sizing guide) | M |
| T-32 | I | Partner keeps copies of exported segments | contract, audit, export-then-delete default; segments sealed to the customer and useless to the service without collusion | M (organisational, same as T-27) |

### 3.5 Network (F1, F4/F5, F6)

| id | STRIDE | threat | controls (v1.0) | residual |
|---|---|---|---|---|
| T-33 | I | Eavesdropping | TLS; content is also masked on F4/F5 | L |
| T-34 | T | Man-in-the-middle alters `ŷ` | TLS or mTLS; Freivalds catches alteration end to end | L |
| T-35 | I | Traffic analysis (volume, timing) | private links (PrivateLink/PSC); padding **planned** | M (accepted) |
| T-36 | D | Link failure, high RTT | retries; same-region deployment; batching | M |

### 3.6 Model pack (F7)

| id | STRIDE | threat | controls (v1.0) | residual |
|---|---|---|---|---|
| T-37 | T | Substituted weights or tokenizer | SHA-256 fingerprint over every file; `verify="full"` at install and service start; cross-party comparison; reproducible packing from the public checkpoint | L |
| T-38 | S | Pack claims a different upstream model | manifest records source repository and commit; rebuild and compare | L |

### 3.7 Operations and supply chain (F8, A8)

| id | STRIDE | threat | controls (v1.0) | residual |
|---|---|---|---|---|
| T-39 | T | Malicious or compromised PGW release (gateway or factory) | signed images, SBOM, pinned dependencies, read-only non-root containers, egress allowlist plus the customer's firewall; source review; external audit **planned** | **M** (largest residual risk; see whitepaper section 9) |
| T-40 | E | Operator abuses admin access | admin actions in audit log (`admin.*` events); least privilege; separate admin token | L to M |
| T-41 | R | Missing evidence for auditors | audit log, metrics retention, change records ([audit checklist](../compliance/audit-readiness.md)) | L |
| T-42 | I | Secrets in env, CLI arguments or images | `SecretStr`, secrets from files or env only, never on the command line; Helm values reference existing Secrets | L |

## 4. Highest residual risks and owners

| rank | risk | ids | owner | treatment |
|---|---|---|---|---|
| 1 | Malicious or compromised PGW build | T-39, T-28 | vendor security, customer security | signed reproducible builds, external audit, customer egress policy |
| 2 | Factory and service collusion (partner topology) | T-27, T-32 | vendor legal, partner, customer | customer-run factory for regulated data; contracts; audits; dual sourcing (roadmap) |
| 3 | Metadata leakage | T-05, T-35 | product | disclose; padding and dummy rounds (roadmap) |
| 4 | Availability of the service and the mask supply | T-07, T-24, T-31 | operations | SLA, multi-endpoint, buffer sizing, alerts |
| 5 | Rollback (snapshot revert) without a monotonic counter off the mask volume | T-22 | customer operations | configure a TPM, HSM, provider or separate-volume file counter; never restore the mask volume |
| 6 | Integrity margin at K32A8 | T-03 | customer configuration | K32A12C27 default (7 check vectors, ≥ 20 bits); `checks ≥ 4` for K32A8 |

## 5. Assumptions and out of scope

* The gateway host and the customer's internal network are trusted (SPEC 1).
* The OS CSPRNG, AES-GCM, HKDF-SHA-256, ChaCha20, X25519 and SHA-256 are secure.
* The factory and the service operator do not collude.
* Out of scope: physical attacks on customer hardware, insider misuse of legitimately obtained
  completions, model-level risks (jailbreaks, hallucination, bias). Those are the same as for any
  locally run open-weight model.

## 6. Review cadence

Review this model on every protocol version change, every new deployment topology, after any security
incident, and at least yearly. Record each review in the change log of the audit evidence
([audit checklist](../compliance/audit-readiness.md), item G-05).
