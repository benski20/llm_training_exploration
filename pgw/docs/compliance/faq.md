# Compliance FAQ (draft)

> **DRAFT: REQUIRES LEGAL REVIEW.** This FAQ was written by engineering to explain the technical
> facts that legal and compliance teams need. It is **not legal advice** and has **not** been reviewed
> by counsel. Every answer below is marked *[needs legal review]*. Regulatory conclusions depend on the
> customer's specific deployment, contracts, jurisdiction and data. Customers must obtain their own
> legal assessment. Do not quote this document in contracts or marketing until legal has approved it.

Technical basis: [security whitepaper](../security/whitepaper.md), [threat model](../security/threat-model.md).
The questions below rely on three technical facts:

1. **What we receive.** Our compute service receives only one-time-masked vectors. Each is
   uniformly random and statistically independent of the customer's data (whitepaper, Theorem 2).
   We never receive prompts, completions, documents, tokens, activation scales, masks or seeds.
2. **Metadata.** We do receive operational metadata: tenant id, model, number of tokens per round
   (from which prompt and output lengths can be inferred), timing and volumes.
3. **Customer boundary.** All plaintext processing happens on the customer-operated gateway, in the
   customer's data center or cloud account.

---

## General data protection

### Q1. Does the compute service operator process personal data? *[needs legal review]*

Technical position: the masked vectors are, to us, indistinguishable from random noise. They cannot
be linked to a person by us, or by anyone without the customer's mask seeds, which we never hold.
The seeds are held by the customer and, in the partner topology, by the independent mask factory.
Legal should take that party into account.
Whether such data counts as "personal data" (GDPR Art. 4(1)), or as anonymous or pseudonymous data
in our hands, is a legal question. It turns on the "means reasonably likely to be used" test
(Recital 26) and on recent case law about relative versus absolute identifiability. Legal should
assess both the payloads and the metadata (tenant id, request timing), and the metadata is at least
business contact data of the customer.

### Q2. Are we a processor under GDPR Art. 28? *[needs legal review]*

Engineering expectation: we should sign a DPA anyway. Even if the payloads are not personal data
for us, the metadata and account data are, and customers' procurement processes will require a DPA.
The DPA should state exactly which categories we receive (section 7 of the whitepaper) and that
plaintext content is technically excluded.

### Q3. International transfers (GDPR Chapter V, Schrems II)? *[needs legal review]*

The service can run in a region chosen by the customer (topology B: same region, private link).
Technically, the payloads that cross a border are one-time padded. Legal should assess whether this
counts as a "supplementary measure" in the sense of the EDPB Recommendations 01/2020 (encryption
where the importer has no key). The customer, not us, holds the masks. Each mask is used once, and
fully consumed mask segments are shredded: the wrapped key is overwritten first, then the file. On
copy-on-write file systems and SSDs, complete erasure relies on full-disk encryption and KEK
rotation.

### Q4. Data minimisation, storage limitation and erasure requests (Art. 5, 17)? *[needs legal review]*

The service is stateless. It keeps no payloads, no KV cache, no sessions and no payload bytes in logs
(SPEC 10, 11.2). The gateway's audit log has a fixed field allowlist with no content. Prompts and
completions exist only in the customer's applications and, transiently, in gateway memory. Erasure
requests are therefore handled by the customer's own systems. We hold only account and billing
metadata.

### Q5. DPIA (Art. 35): what do we provide? *[needs legal review]*

We provide the whitepaper, the STRIDE threat model, the architecture diagram, the data-flow table
(threat model section 2), and the [audit-readiness checklist](audit-readiness.md) as inputs to the
customer's DPIA. The DPIA should cover the residual risks: metadata, the supply chain of the gateway
software, and partner collusion if a partner factory is used.

### Q6. Security of processing (Art. 32)? *[needs legal review]*

Technical measures: one-time masking of all data sent to us, TLS and mTLS, integrity verification
(Freivalds), encryption at rest of masks (AES-256-GCM with HSM or KMS options), fail-closed policies,
audit logging, rollback protection, and key rotation. Organisational measures (partner independence,
access control, incident response) are listed in the audit checklist.

## Healthcare (HIPAA, United States)

### Q7. Is PGW's compute service a Business Associate? *[needs legal review]*

Technical facts: the service never receives PHI in readable form. It receives masked data that
cannot be decrypted without keys it does not have. HHS guidance on cloud service providers says a CSP
that stores or processes **encrypted** ePHI is still a business associate even without the key
("no-view" services). Whether one-time masking with customer-held masks falls under that guidance
is the central question for counsel. Engineering's recommendation is to **plan to sign a BAA** and
to treat the "conduit exception" as unlikely to apply.

### Q8. HIPAA Security Rule safeguards? *[needs legal review]*

Mapping to 45 CFR 164.312: access control (API keys, scopes, admin separation), audit controls
(audit log, metrics), integrity (Freivalds checks, AEAD on stored masks, model fingerprints),
person or entity authentication (mTLS, tenant tokens), and transmission security (TLS plus one-time
masking). The gateway is a customer system and falls under the customer's own HIPAA program.

### Q9. Breach notification if our service is compromised? *[needs legal review]*

Technically, a full compromise of our service exposes only uniform masked vectors and metadata
(whitepaper section 7). Whether such an incident is a "breach of unsecured PHI" depends on whether
the masking qualifies as rendering PHI "unusable, unreadable, or indecipherable" under the HHS
guidance (which names NIST-approved encryption). Counsel should assess this. The masking relies on
ChaCha20 and HKDF-SHA-256, plus the non-collusion of the factory.

## Education (FERPA, United States)

### Q10. Can a school or university send education records through PGW? *[needs legal review]*

Technical position: records are processed in plaintext only on the institution's gateway. We receive
masked vectors and metadata. Legal should assess whether we are a "school official" with a
"legitimate educational interest" under the outsourcing provisions of 34 CFR 99.31(a)(1)(i)(B),
or whether no disclosure of personally identifiable information from education records occurs at
all. State student-privacy laws (for example California SOPIPA) may impose additional contract terms.

### Q11. Directory information and metadata? *[needs legal review]*

Request metadata (timing, lengths, tenant) could, combined with other information, reveal patterns of
use. Institutions with strict requirements can deploy the gateway on-prem and batch traffic. Padding
and dummy traffic are planned features, not in v1.0.

## Other frameworks

### Q12. SOC 2 / ISO 27001? *[needs legal review]*

PGW is not yet certified. The [audit-readiness checklist](audit-readiness.md) maps PGW controls to
evidence an auditor will request. Certification of our operation of the compute service is on the
roadmap. Customers include the gateway in the scope of their own certifications.

### Q13. EU AI Act? *[needs legal review]*

PGW is infrastructure for running open-weight general-purpose models. The customer's use case
determines any risk classification. The model provider's obligations (GPAI) stay with the model
publisher. Legal should assess whether operating the compute service creates any provider or deployer
obligations for us.

### Q14. Sector rules on outsourcing (finance: EBA guidelines, DORA)? *[needs legal review]*

The compute service is an ICT third-party service. Masking reduces the confidentiality impact, but it
does not remove availability dependencies (exit strategy, SLA, concentration risk). The gateway can be
pointed at another compatible endpoint, and the protocol is openly specified (SPEC). That supports
exit plans.

## Contract checklist for legal (draft)

- [ ] DPA describing the categories received (masked payloads, metadata, account data)
- [ ] BAA template for US healthcare customers
- [ ] Partner factory agreement: independence, non-collusion, deletion and audit rights (see [air-gapped-factory.md](../operations/air-gapped-factory.md#42-partner-generate-and-export))
- [ ] Sub-processor list (cloud providers hosting the compute service)
- [ ] Statement on metadata use: billing and capacity only; retention periods
- [ ] Incident notification terms covering integrity alerts reported by customer gateways
- [ ] Wording review: avoid "unbreakable" or "guaranteed compliant"; describe the guarantee and its assumptions accurately (computational at the PRG level, non-collusion, metadata not hidden)
