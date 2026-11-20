# Getting started: set up your gateway with an enrollment code

This guide is for the person who sets up the PGW gateway at your organisation. You do not need to
know how the cryptography works. You need:

* a Linux machine (or VM) for the gateway, with Python and the `pgw` package installed
  (see [install-on-prem.md](install-on-prem.md#2-install-the-software)), at least 16 GB of memory
  and enough disk for the model and the masks (the wizard tells you how much);
* outbound HTTPS from that machine to your provider's compute service, and to `huggingface.co`
  while the model downloads;
* the **enrollment code** your provider sent you. It starts with `PGW1-`.

Setup takes about ten minutes of your time. Downloading and preparing the model takes longer:
from a few minutes for small models to an hour for a 7B model on a slow connection. You can
leave the browser open while it runs.

## 1. Start the gateway in setup mode

Start the gateway without any model configured. It recognises that it is new and opens the
setup wizard instead of the normal service:

```bash
PGW_GATEWAY_CONFIG_DIR=/etc/pgw PGW_GATEWAY_DATA_DIR=/var/lib/pgw pgw gateway serve
```

The gateway prints a box like this:

```text
PGW gateway setup mode: no model is configured yet.
Open http://localhost:8080/console/ and enter this one-time setup token:

    pgw-setup-...
```

Open that address in a browser on the same machine (or through an SSH tunnel), and enter the
setup token. The token proves that you have access to the gateway machine. Nobody who merely
reaches the port can change the configuration. A new token is printed each time the gateway
starts, and the wizard remembers how far you got.

If you need to run the wizard again later, start the gateway with `PGW_GATEWAY_SETUP_MODE=1`.

## 2. Follow the six steps

**Connect to your provider.** Paste the enrollment code. Spaces and line breaks do not matter.
If you mistype a character, the wizard says so before it contacts anybody. The gateway then
checks that it is really talking to your provider: the code contains a fingerprint of your
provider's certificate authority, and the gateway refuses to continue if the service presents a
different one (for example because a proxy intercepts the connection). A code works once and
expires (usually after 24 hours); ask your provider for a new one if it was used or expired.

**Choose a model.** Pick one of the models your provider runs. The gateway downloads the public
model weights and prepares them. When it finishes, it compares the result with what the provider
runs and refuses to continue if they differ, so both sides compute with exactly the same model.

**Protect the masks.** Every request is hidden with one-time random masks that the gateway makes
in advance. The wizard shows how much disk this needs. For a 7B model the one-time encryption hint
alone is about 34 GB (see [encrypted-masks.md](encrypted-masks.md)), plus a few GB per thousand
tokens of ready-made masks. Choose a folder on a local disk. The wizard creates an encryption key
file that only the gateway can read and keeps it apart from the masks. Cloud key services (AWS
KMS, Google Cloud KMS, Azure Key Vault) appear as "coming soon" and cannot be selected yet.

Never restore the mask folder from a backup or snapshot, and never put it on a network share.
Restored masks could be used twice. The gateway detects that and refuses to serve.

**Admin access.** The wizard creates an admin token and shows it **once**. Store it in your
password manager; you sign in to the console with it after setup. Optionally, enter your
company's single sign-on details (OpenID Connect). The fields are explained in
[console-sso.md](console-sso.md).

**Check everything.** The gateway checks the connection to your provider, the model, the mask
store and its safety features, and starts making masks in the background. You see a checklist in
plain language:

| check | what it means |
|---|---|
| Compute service reachable over pinned TLS | the provider answers, over an encrypted connection whose identity was fixed at enrollment |
| Model verified | your copy of the model is identical to the provider's |
| Mask store protected | masks are encrypted and protected against being restored from a snapshot |
| Masks available | masks are ready; the background factory keeps refilling them |
| Every result checked (Freivalds) | each answer from the provider is verified before use |
| Outbound noise monitor | traffic stops if outgoing data ever stops looking random |
| Network advice | which outbound connection to allow in your firewall |

"In progress" for masks is normal for the first few minutes. Problems are shown in red with
what to do.

**Finish.** The wizard writes the configuration and switches the gateway to normal operation
without a restart. Sign in to the console with the admin token, create an API key under
**API keys**, and point your applications at the gateway.

## 3. After setup

The wizard writes these files (the paths below assume the example above):

| file | contents |
|---|---|
| `/etc/pgw/gateway.env` | the gateway settings; load them when you run the gateway as a service |
| `/etc/pgw/factory.env` | settings of the background mask factory |
| `/etc/pgw/models.json` | your model, pinned to the provider's fingerprint |
| `/etc/pgw/secrets/` | service token, admin token, encryption key, client key (mode 0600) |
| `/etc/pgw/service-ca.pem` | your provider's certificate authority |
| `/var/lib/pgw/` | the model, the masks and the gateway's usage records |

When the gateway restarts, it reads `/etc/pgw/gateway.env` automatically and starts the mask
factory itself. To run it under systemd, use `EnvironmentFile=/etc/pgw/gateway.env` and the
command `pgw gateway serve`.

Keep the gateway's own port private or put it behind your TLS proxy: the gateway sees prompts in
plaintext (see [install-on-prem.md](install-on-prem.md#8-firewall-and-egress)). Allow outbound
HTTPS only to your provider's compute service.

The wizard never offers development shortcuts: no unencrypted connections to remote services, no
disabled checks and no development modes.

## 4. Without the wizard (command line)

Administrators who prefer the command line can enroll directly and then follow
[install-on-prem.md](install-on-prem.md):

```bash
pgw gateway enroll PGW1-AEXD4-... --config-dir /etc/pgw
```

The command checks the code, pins the provider's certificate authority, generates the gateway's
private key locally (it never leaves the machine) and writes the token (mode 0600), the CA, the
client certificate and a `models.json`. It refuses to overwrite an earlier enrollment unless you
add `--force`. It then prints the remaining commands: build the model pack with `pgw pack`,
initialise the mask store with `pgw maskstore init` and start `pgw gateway serve`.

## 5. Troubleshooting

| message | what to do |
|---|---|
| "has a typo (its checksum does not match)" | copy and paste the code again instead of typing it |
| "did not accept this enrollment code" | the code was used, revoked or has expired: ask your provider for a new one |
| "certificate authority does not match" | something between you and the provider intercepts TLS (often a corporate proxy); exempt the provider's address from TLS inspection, then use a new code |
| "is not the one the service runs" | the downloaded model differs from the provider's; ask your provider which revision they run |
| "not a setup option" | the wizard only accepts production settings; development options cannot be set here |
| setup token not accepted | copy the token printed by the most recent start of the gateway |

## 6. For operators: issuing an enrollment code

This section is for the compute provider's operators.

Enable enrollment on the compute service:

```bash
PGW_SERVICE_TOKENS_FILE=/etc/pgw/tenants.txt
PGW_SERVICE_ENROLLMENT_FILE=/var/lib/pgw/enrollment.json
PGW_SERVICE_PUBLIC_URL=https://compute.example.net:8443
PGW_SERVICE_ENROLLMENT_CA_FILE=/etc/pgw/tls/ca.pem
```

`PGW_SERVICE_ENROLLMENT_CA_FILE` is the CA that issued the service's TLS certificate; its SHA-256
fingerprint goes into every code. To also issue short-lived mTLS client certificates for the
gateways' locally generated keys, set `PGW_SERVICE_CLIENT_ISSUING_CA_CERT_FILE` and
`PGW_SERVICE_CLIENT_ISSUING_CA_KEY_FILE` (lifetime `PGW_SERVICE_CLIENT_CERT_VALIDITY_DAYS`,
default 30). With `PGW_SERVICE_MTLS_TENANTS_FILE` set, the certificate's name is mapped to the
tenant automatically. Gateways enroll before they have a client certificate, so the enrollment
listener must not require one.

Create, list and revoke codes:

```bash
pgw service enroll create --tenant acme --ttl 24h --models qwen7b --note "Acme pilot"
pgw service enroll list
pgw service enroll revoke 3f2a9c1b7e44
```

`create` prints the code once. Only a hash of its secret is stored. Send it to the customer's
administrator over a trusted channel (for example a password-manager share or a phone call for
the last groups), not in a ticket or a public chat. Each code works once, until its TTL expires
(at most 30 days), and only for the listed models (all served models if `--models` is omitted).

On `POST /pgw/v1/enroll` the service checks the code in constant time, marks it used atomically,
creates a fresh tenant token (only its hash is appended to the tokens file, exactly as
`pgw service token create` does) and returns it together with the CA and the model list. Attempts
are limited per client IP (`PGW_SERVICE_ENROLLMENT_RATE_LIMIT_PER_MINUTE`, default 10). Every
attempt is recorded in `PGW_SERVICE_ENROLLMENT_AUDIT_FILE` (default: next to the enrollment file)
without secrets, tokens or certificates. Unknown, used, revoked and expired codes all get the same
answer; the audit record has the reason.
