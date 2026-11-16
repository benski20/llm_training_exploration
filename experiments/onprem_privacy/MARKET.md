# Market check: private / on-prem AI (sources checked through Nov 2026)

These are web-sourced findings. Figures from aggregators conflict in places; see the caveats at the end.

## K-12: keeping data out of a third-party cloud is not the main blocker

US districts already buy cloud AI under standard data privacy contracts:

- The SDPC registry holds 130k+ signed data privacy agreements, covering 12k+ schools and districts and 6.6k vendors.
- Vendors receive student data under FERPA's "school official" exception, with protections set by contract.

Scale and pricing of what districts already use:

- MagicSchool: 2M+ educators, $45M Series B.
- SchoolAI: 1M+ classrooms, $25M Series A.
- Khanmigo: about $10-15 per student per year.
- Gemini for Education: free in every Workspace for Education plan since Jul 2025.
- Microsoft 365 Copilot academic: $18 per user per month. Copilot Chat is free with A1/A3/A5 licences.

The barriers surveys actually report:

- Budget is the top challenge (CoSN 2026).
- Many districts have no policy: only 45% of principals report one (RAND).
- Teacher training is lacking.
- Districts lack capacity to vet vendors.
- Privacy is a concern for 45-60% of district leaders, but it is about trust in vendors and breaches (PowerSchool: about 62M students), not about where the model runs.

Budgets are tight: 63% of districts spend 5% or less of their budget on technology, and ESSER funding has run out.

## Segments where the cloud really is a blocker

- **Defense and classified (IL5, IL6, IL7, Top Secret):**
  - Ask Sage: $49M from the Army plus $10M from CDAO.
  - Google: up to $200M from CDAO, with its air-gapped Distributed Cloud authorized at IL6.
  - The DoD signed IL6/IL7 AI deals with eight vendors in May 2026.
- **EU and sovereign:** Gartner forecasts sovereign cloud IaaS at $80B in 2026, up 36%.
- **Banking:** 59% of banks delayed an AI project over security or privacy (a vendor-run survey).
- **Healthcare:** no on-prem mandate found. HIPAA business associate agreements with cloud providers are common.

## Competitors by layer

- **Confidential inference:**
  - Opaque ($55M raised; $300M valuation, Feb 2026)
  - Anjuna ($58-67M)
  - Fortanix ($122M)
  - Edgeless Systems
  - Tinfoil (YC)
  - Phala (H100 TEE at $3.08 per GPU-hour)
- **PII gateways:**
  - Skyflow (about $100M)
  - Nightfall ($60M)
  - Lakera (acquired by Check Point for about $300M)
  - Private AI
  - Tonic
  - Protecto
- **FHE:**
  - Zama ($1B+ valuation, now blockchain-focused)
  - Duality ($50M+)
  - Enveil (government sales, NIAP/CSfC certified)
  - Niobium and Cornami (FHE silicon)
  - None sells a commodity per-query LLM product.
- **On-prem hardware:**
  - Dell AI Factory: 3k-5k+ customers (self-reported).
  - HPE Private Cloud AI.

## Confidential GPU economics

- Azure confidential H100 (NCC40ads v5) lists at the same price as the standard H100: $6.98 per hour, East US 2.
- Measured overhead: 4-8%, and 12-21% in one preprint. Llama-8B time to first token is about 19% slower.
- Multi-GPU scaling is limited because RDMA and GPUDirect are unavailable in confidential mode.

Caveats: several funding totals come from aggregators and conflict. MagicSchool's current numbers, Tinfoil and Privatemode pricing, and HPE customer counts were not verified. Full source URLs are in the session transcript.
