# Audit-readiness checklist

For customer security teams, internal audit and external auditors (SOC 2, ISO 27001, HIPAA
assessments). Each item has an id, the control, the evidence to collect, and who owns it:
**C** customer, **V** vendor (us, the compute service operator), **P** partner factory.
Mark each item *done*, *n/a* (with a reason) or *open*. Map the ids to your control framework.
The [threat model](../security/threat-model.md) ids (`T-xx`) show which risk each control treats.

## G. Governance

| id | control | evidence | owner | threats |
|---|---|---|---|---|
| G-01 | Deployment topology documented (A on-prem, B customer cloud, C partner factory) | architecture diagram with trust boundaries | C | all |
| G-02 | DPIA or risk assessment completed using the whitepaper and threat model | signed DPIA | C | T-05, T-27 |
| G-03 | Contracts in place: DPA/BAA with the vendor; independence agreement with the partner factory | executed contracts | C, V, P | T-27, T-32 |
| G-04 | Compliance FAQ reviewed by counsel for your jurisdiction | legal memo | C | — |
| G-05 | Threat model reviewed in the last 12 months and after changes | review record | C, V | all |
| G-06 | Roles assigned: gateway admin, key custodian, factory operator, incident contact | RACI | C | T-40 |

## S. Software supply chain

| id | control | evidence | owner | threats |
|---|---|---|---|---|
| S-01 | Release provenance: version and image digest of gateway, factory and service recorded | `pgw version --json`, image digests | C, V | T-39 |
| S-02 | SBOM obtained and scanned for known vulnerabilities | SBOM, scan report | C, V | T-39 |
| S-03 | Image signatures verified before deployment | verification log | C | T-39 |
| S-04 | Cross-implementation check: `pytest tests/vectors` and `cargo test` in `rust/` pass for the deployed version (known-answer vectors) | CI log | V | T-28 |
| S-05 | Optional source review or rebuild of gateway and factory by the customer | review report | C | T-39, T-28 |
| S-06 | External security audit of protocol and implementation (planned, SPEC 16) | audit report | V | all |

## M. Masks and keys

| id | control | evidence | owner | threats |
|---|---|---|---|---|
| M-01 | Mask factory is customer-run, or a partner contractually independent of the vendor; never the vendor | contract, network diagram | C, P | T-27 |
| M-02 | Factory host has no network route to the vendor (or no network at all) | firewall rules, `--network none` config | C, P | T-27, T-28 |
| M-03 | KEK stored outside the mask volume; HSM or KMS provider for production | `PGW_MASKSTORE_KEY_PROVIDER`, key policy | C | T-21, T-26 |
| M-04 | Segment transfer log for partner or offline deliveries (who, when, segment ids), media wiped | transfer log | C, P | T-30 |
| M-05 | Recipient public key fingerprint confirmed out of band with the partner | confirmation record | C, P | T-30 |
| M-06 | Rollback protection: HMAC journal on, plus a monotonic counter off the mask volume (TPM, HSM, provider, or a file counter on separate storage) configured, or the journal-only risk (undetected whole-volume snapshot revert) accepted in writing | `pgw maskstore status`, risk acceptance | C | T-22 |
| M-07 | Mask store excluded from restore-capable backups | backup policy | C | T-14, T-22 |
| M-08 | `pgw doctor --maskstore` clean at install and after upgrades | doctor output | C | T-14 |
| M-09 | KEK rotation procedure tested (`pgw maskstore rotate-key`) | test record | C | T-21 |
| M-10 | Consumed segments shredded (`shred_consumed` on) | config, disk usage trend | C | T-21 |
| M-11 | Freivalds `checks` appropriate for the ring (≥ 4 for K32A8) | factory config | C | T-03 |

## N. Network and access

| id | control | evidence | owner | threats |
|---|---|---|---|---|
| N-01 | Gateway egress allowed only to the compute service origin(s), at the firewall **and** in `egress_allowlist` | firewall export, `/admin/status` egress list | C | T-39 |
| N-02 | TLS to the service with our CA pinned (`service_ca_file`); `require_service_tls` on | config | C | T-06, T-33 |
| N-03 | mTLS client certificate or tenant token issued per gateway; stored as a secret | secret inventory | C, V | T-06 |
| N-04 | Gateway API served over TLS; admin endpoints and `/metrics` reachable from the ops network only | ingress or NetworkPolicy | C | T-19 |
| N-05 | API keys per application with least scope, model allow-list and limits; an owner for each key | `GET /admin/keys` export | C | T-11, T-12 |
| N-06 | Admin token held in a secret manager; rotation procedure | secret manager record | C | T-19, T-40 |
| N-07 | Private connectivity for topology B (PrivateLink, PSC or Private Endpoint) | network config | C, V | T-35 |

## O. Operations and monitoring

| id | control | evidence | owner | threats |
|---|---|---|---|---|
| O-01 | Production policy on: `/admin/status` shows `production_policy: true`; no `allow_dev_modes` | status snapshot | C | T-20 |
| O-02 | Every compute response verified (`freivalds_rate = 1.0`) | config, status snapshot | C | T-03 |
| O-03 | Outbound-noise monitor active and fail-closed (`noise_monitor_required`, `noise_monitor_fail_closed`) | config, status snapshot | C | T-13, T-14 |
| O-04 | Alerts: `pgw_gateway_integrity_failures_total` increase, `pgw_gateway_noise_alarm`, `pgw_masks_low_water`, `/readyz` failing | alert rules, test page | C | T-03, T-13, T-24 |
| O-05 | Audit log (`audit_log_file`) enabled, shipped to the SIEM, retained per policy | SIEM query | C | T-15, T-41 |
| O-06 | Logs checked for absence of content (sample review: no prompts, completions or payload bytes) | review record | C, V | T-16 |
| O-07 | Core dumps disabled for gateway processes | systemd unit or pod spec | C | T-16 |
| O-08 | Fingerprints match across gateway, service and segments (pinned in `models.json`) | `pgw inspect --verify full`, status | C | T-08, T-37 |
| O-09 | Capacity: mask buffer ≥ the defined days of traffic; sizing reviewed quarterly | sizing worksheet ([sizing.md](../operations/sizing.md)) | C | T-24, T-31 |
| O-10 | Vendor service: stateless operation, no payload logging, retention of metadata documented | vendor attestation, SOC report | V | T-02 |

## I. Incident response

| id | control | evidence | owner | threats |
|---|---|---|---|---|
| I-01 | Runbook for integrity failure (breaker open): collect `request_id`, notify vendor, investigate, then admin reset | runbook | C, V | T-03 |
| I-02 | Runbook for noise alarm and `mask_reuse`: stop traffic, preserve the store and logs, security incident | runbook | C | T-13, T-14 |
| I-03 | Runbook for suspected factory compromise or collusion: stop imports, rotate the recipient key, regenerate masks in-house, assess past exposure | runbook | C, P | T-27, T-28 |
| I-04 | Runbook for gateway host compromise (out of PGW's protection scope) | runbook | C | T-17 |
| I-05 | Tabletop exercise run in the last 12 months | exercise report | C | all |

## Evidence collection commands

```bash
pgw version --json
pgw doctor --role gateway --settings --pack /var/lib/pgw/packs/<pack> --maskstore /var/lib/pgw/masks --json
pgw inspect /var/lib/pgw/packs/<pack> --verify full --json
pgw maskstore status --json
pgw gateway keys list
curl -s -H "Authorization: Bearer $ADMIN" https://pgw.internal:8080/admin/status
curl -s -H "Authorization: Bearer $ADMIN" https://pgw.internal:8080/admin/masks
```

None of these outputs contain prompts, completions, masks, seeds or key material. They can be
attached to audit evidence as they are.
