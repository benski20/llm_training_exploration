# Security model summary

This page summarises the security of the PGW client library for security reviewers. The full analysis is
in the [security whitepaper](../security/whitepaper.md) and the [threat model](../security/threat-model.md).
It also states plainly what has not been reviewed yet.

## Parties and trust

| party | trusted with | assumption |
|---|---|---|
| Your process and machine | everything (plaintext prompts, answers, masks, keys) | you control it; if it is compromised, PGW cannot protect the data |
| Our compute service and GPUs | masked activations, encrypted mask material, usage metadata | **untrusted**: may be curious or actively malicious |
| The network | TLS-encrypted traffic | untrusted |

The model weights are public. Their secrecy is not a goal.

## What is protected

* **Confidentiality of content.** Every activation sent to our GPU is masked with a fresh uniformly random
  value used exactly once. For each such use the masked value is statistically independent of your data
  (one-time pad; whitepaper section 4). Masks are never reused: the library refuses reuse and treats an
  attempt as a security incident (`mask_reuse`).
* **Confidentiality of the masks.** With the encrypted schemes (`pks-v3`, `mlwe`), our service helps
  generate masks while seeing them only encrypted under lattice-based encryption keyed in your process.
  Here confidentiality is computational: it rests on the hardness of ring or module learning-with-errors
  (post-quantum lattice assumptions), not on the one-time pad alone. With `mask_scheme="plain"` no help from
  us is used and no such assumption is needed.
* **Integrity.** Each result from our GPU is checked in your process with Freivalds' algorithm and a range
  check before use; with the default settings a wrong result passes one check with probability at most
  2^-20 (floor; usually better), and a detected failure fails the request (`integrity_check_failed`).
  Mask material we help produce is self-checked with at least 40 bits of soundness against errors of our
  choosing before use.
* **Authenticity of the service.** The certificate authority of our service is pinned by your key; TLS to
  any other server fails.
* **Your credentials.** The client private key is generated on your machine and never leaves it. The SDK
  key is revocable by us at your request.

## What is not protected

* **Metadata.** We see which model you use, token counts, request timing and traffic volume, and we keep
  usage records for billing and capacity. Padding and dummy traffic are planned, not in this release.
* **Prefix cache hits** (opt-in, `prefix_cache=True`): a request that reuses a cached prompt prefix sends only
  its new tokens' rows, so we can tell that it continued an earlier request and roughly how long its context is.
  No content is revealed ([configuration](configuration.md#prefix-cache)).
* **A compromised client machine.** Anyone controlling the machine that runs your process can read
  everything your process can.
* **Availability.** We can refuse or delay service. We cannot make you accept a wrong answer.
* **Outputs you send elsewhere.** Once your application passes a prompt or answer to another system, PGW's
  guarantees end there.

## Review status, stated plainly

* **No external security audit of PGW has been completed yet.** The design, proofs and code have had
  internal reviews, including adversarial reviews of the mask schemes; findings and fixes are tracked in our
  release status.
* **`pks-v3`, the default mask scheme of the client library, is pending external review**: an independent
  run of the lattice estimator on its parameters and a cryptographer's sign-off. Its internal adversarial
  review is done and the findings on the default path have been addressed. The library uses `pks-v3` only with an explicit
  acknowledgement of this status, which the client library gives on your behalf for the pilot.
* If your policy requires only reviewed constructions, select `mask_scheme="mlwe"` (larger one-time setup,
  same protocol and checks otherwise) or `mask_scheme="plain"` (no lattice assumption; more CPU and disk).
  Note that `mlwe` has not been externally audited either.
* The one-time pad argument and the Freivalds bound are standard results; their application here is set out
  in the whitepaper for your reviewers.

We will update this page and the [changelog](changelog.md) when the external reviews conclude.

## Operational recommendations

* Keep the SDK key in a secret manager or environment variable, never in code or images. Ask us to revoke a
  key you suspect is exposed; a new key is issued at once.
* Protect the [cache directory](configuration.md#cache-directory) like an SSH key directory, and do not
  share it between machines.
* Install from the package source we name, check the file hashes we send you, and use a dedicated virtual
  environment.
* Treat `integrity_check_failed` and `mask_reuse` errors as security events: do not retry automatically;
  report them to us with the time and request id.
