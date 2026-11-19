# Glossary

Terms are in alphabetical order. The normative definitions are in [SPEC.md](../SPEC.md).

**A (activation bits)**: Bit width of quantized activations. `qmax = 2^(A−1) − 1`. A12 in the default
ring K40A12.

**Admin token**: Secret bearer token for the gateway's `/admin/*` endpoints (`PGW_GATEWAY_ADMIN_TOKEN`).
Separate from application API keys.

**AEAD**: Authenticated encryption with associated data. PGW uses AES-256-GCM for mask segment chunks.

**API key**: Application credential for the gateway, `pgw-<key_id>-<secret>`. Only a SHA-256 hash is
stored. Carries scopes, a model allow-list and limits.

**Audit log**: JSON-lines record of who did what and when on the gateway (requests, admin actions,
auth failures). It has a fixed field allowlist and never contains content.

**Balanced digits**: Representation of a ring element as base-256 digits in `[−128, 127]`. It lets
`K/8` int8 GEMMs compute an exact product modulo `2^K`.

**Burned slot**: A reserved mask slot that a request did not use (for example after an early stop).
It is never reused and is counted in `pgw_masks_burned_slots_total`.

**Circuit breaker**: Gateway safety switch that blocks masked traffic to a service endpoint after an
integrity failure, until a cool-down or an admin reset.

**Collusion**: Sharing of mask seeds (or masks) between the mask factory and the compute service
operator. It breaks privacy. PGW's main organisational assumption is that it does not happen.

**Compute service**: Our stateless server (`pgw.service`). It computes `ŷ = W8·x̂ mod 2^K` on masked
inputs and is treated as untrusted.

**DEK / KEK**: Data-encryption key (random, one per segment) and key-encryption key (customer-held;
file, env, PKCS#11 HSM or KMS) that wraps it.

**Egress allowlist**: The outbound origins the gateway may contact: the configured service URLs plus
`PGW_GATEWAY_EGRESS_ALLOWLIST`.

**Fail-closed**: On a detected anomaly, stop serving rather than continue in a degraded or unsafe mode.

**Fingerprint**: SHA-256 over the canonical model-pack manifest, which includes the hash of every file.
It binds gateway, factory, service and mask segments to the same weights. The short form is the first
12 hex characters (`system_fingerprint`).

**Freivalds check**: Probabilistic verification of a matrix product with secret vectors `u`:
`u·ŷ ≡ (uᵀW8)·x̂ (mod 2^K)`. Together with the range check, it bounds the false-accept probability by
`2^(−c·h)`.

**Gateway**: The customer-side component (`pgw.gateway` and `pgw.engine`). It serves the
OpenAI-compatible API and does all plaintext work. It masks, unmasks and verifies.

**HPKE**: Hybrid Public Key Encryption (RFC 9180). A partner factory uses it to seal segment data
keys to the customer's X25519 public key.

**K (ring bits)**: Arithmetic is in `Z_{2^K}`. Presets are K32, K40 and K48.

**Linear group**: One masked matrix product per layer: `qkv`, `o`, `gate_up` or `down`. There are
`4·L` round trips per forward pass.

**Low water**: Mask inventory threshold (`low_water_slots`). Below it, an alert fires and the factory
refills toward `target_slots`.

**Mask (`r`)**: Uniform random vector in `Z_{2^K}^n`, added to a quantized activation before it is
sent. Each mask is used exactly once.

**Mask factory**: The component (`pgw.factory`) that generates seeds, `W8·r` and the Freivalds data
offline. It is run by the customer or by an independent partner, never by the service operator.

**Mask reuse**: Using one mask for two rows. This leaks the plaintext difference and is a
catastrophic failure. It is prevented and detected by several independent mechanisms.

**Mask store**: The customer-side encrypted store of single-use mask segments (`pgw.maskstore`), with
durable reservations and rollback detection.

**Metadata**: Information the service sees besides payloads: model, rows per round (token counts),
timing, tenant. It is not hidden in v1.0.

**Model pack**: Directory with int8 weights, scales, plain tensors, tokenizer and a manifest with the
fingerprint. Built with `pgw pack`.

**mTLS**: Mutual TLS. The gateway presents a client certificate to the compute service.

**Noise monitor (outbound)**: Gateway component that statistically tests sampled outbound payloads for
uniformity and duplicate rows. It alarms, and by default blocks traffic, if something unmasked or
reused appears to be sent.

**One-time pad**: Encryption by adding a uniformly random key of the same size that is used once. It
gives perfect secrecy.

**Perfect secrecy**: The ciphertext distribution is independent of the plaintext (Shannon). PGW
achieves it per call with ideal masks; the deployed system is computationally secure at the level
of its seed expander (ChaCha20).

**Prefill**: Processing the prompt. All prompt tokens of a linear group are sent in one round trip.

**PRG / seed expander**: Deterministic generator that expands a 32-byte seed into masks
(HKDF-SHA-256 and ChaCha20, SPEC 7.3).

**Range check**: Gateway check that every decoded output satisfies `|yq| ≤ qmax·127·n`. It is needed
for the Freivalds bound.

**Reservation**: Durable claim of mask slots for one request, committed before any mask is decrypted.

**Ring preset**: Named ring `K..A..[C..]`: K32A12C27 (default; 32-bit with a per-call output cap of 2^27
and adaptive activation bits, measured lossless up to 7B), K40A12 (previous default), K48A20 (maximum
quality), K32A8 (cheapest, measured lossy).

**Rollback protection**: Detection of a restored, older mask-store state (which could cause reuse),
via a generation counter mirrored to an HMAC journal (detects a restored database) and a monotonic
counter off the mask volume (also detects a whole-volume snapshot revert).

**Round trip**: One gateway-to-service request and response for one linear group (all rows of that
group in the current step).

**Segment**: Encrypted mask file (`*.pgwm`) holding `S` slots for one fingerprint and ring, with its
seed, `uᵀW8` and `W8·r` rows.

**Slot**: One token position's worth of masks: one row for each of the `4·L` linear groups.

**Soundness bits (`c·h`)**: `−log2` of the per-call false-accept bound of the integrity checks, with
`h = K − ⌈log2(4·qmax·127·n)⌉`.

**STRIDE**: Threat-modelling mnemonic: Spoofing, Tampering, Repudiation, Information disclosure,
Denial of service, Elevation of privilege.

**Tenant token**: Bearer token identifying a customer gateway to the compute service. It is stored
server-side as a SHA-256 hash.

**Topology A / B / C**: Deployment options: on-prem gateway; gateway in the customer's cloud account
over a private link; independent partner mask factory.

**W8, sw**: int8 weight matrix, symmetric per output row, and its per-row scales. Both are public.

**x̂, ŷ (x_hat, y_hat)**: Masked input `x̂ = xq + r mod 2^K` and the service's product
`ŷ = W8·x̂ mod 2^K`. These are the only content the service sees.
