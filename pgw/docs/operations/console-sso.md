# Console single sign-on (OpenID Connect)

The operator console (`/console/`) signs people in with the gateway admin token or a viewer token
by default. You can add single sign-on (SSO) through your identity provider (IdP) with OpenID
Connect. Users then click **Sign in with SSO** and land in the console with the `admin` or `viewer`
role that their groups or app roles grant. SSO is optional and off by default. Air-gapped
installations without an IdP keep using tokens; nothing in the gateway calls an IdP until SSO is
configured.

The API contract (endpoints, status codes, audit events) is in
[console-api.md](../api/console-api.md#single-sign-on-openid-connect).

## How it works

1. The browser opens `GET /console/api/oidc/login`. The gateway creates a pending sign-in with a
   random `state`, `nonce` and PKCE `code_verifier`, keeps it in memory for 10 minutes (single use),
   binds it to the browser with a short-lived `pgw_oidc` cookie and redirects to the IdP
   (authorization code flow, `code_challenge_method=S256`).
2. The user authenticates at the IdP (MFA, conditional access and so on stay the IdP's job). The
   IdP redirects back to `.../console/api/oidc/callback` with a code.
3. The gateway redeems the code at the IdP's token endpoint with the client secret and the
   `code_verifier`, then verifies the ID token: RS256 or ES256 signature against the IdP's
   published keys (JWKS), issuer, audience, `azp`, expiry, issue time and nonce.
4. The gateway reads the role claim (default `groups`) and maps it to `admin` or `viewer`. Users
   with neither value get `403`. It then creates the same console session as a token sign-in
   (HttpOnly `pgw_console` cookie, CSRF token, 8 h idle and 24 h absolute expiry) and records
   `console.login` with `method: "oidc"`, the subject and the email in the audit log.

The gateway is a **confidential client**: it holds a client secret, read from a file. Access
tokens, refresh tokens and ID tokens are used once and never stored or logged. Signing out of the
console ends the console session only; the IdP session is unaffected.

## Settings

All settings are environment variables of the gateway (or fields of its settings file). SSO is on
when both `PGW_GATEWAY_CONSOLE_OIDC_ISSUER` and `PGW_GATEWAY_CONSOLE_OIDC_CLIENT_ID` are set; the
gateway then refuses to start if the rest of the SSO configuration is incomplete. List values are
JSON arrays.

| setting | default | meaning |
|---|---|---|
| `PGW_GATEWAY_CONSOLE_OIDC_ISSUER` | unset | issuer URL, exactly as the IdP's discovery document states it (https) |
| `PGW_GATEWAY_CONSOLE_OIDC_CLIENT_ID` | unset | client (application) id registered at the IdP |
| `PGW_GATEWAY_CONSOLE_OIDC_CLIENT_SECRET_FILE` | unset (required with SSO) | file holding the client secret; never put the secret itself in the environment |
| `PGW_GATEWAY_CONSOLE_OIDC_REDIRECT_URL` | unset (required with SSO) | the redirect URI registered at the IdP; must end with `/console/api/oidc/callback` |
| `PGW_GATEWAY_CONSOLE_OIDC_SCOPES` | `openid email profile` | space-separated scopes; must include `openid` |
| `PGW_GATEWAY_CONSOLE_OIDC_ROLE_CLAIM` | `groups` | ID token claim with the user's groups or roles; a dotted path reads a nested claim (`realm_access.roles`) |
| `PGW_GATEWAY_CONSOLE_OIDC_ADMIN_VALUES` | `[]` | claim values that grant `admin` |
| `PGW_GATEWAY_CONSOLE_OIDC_VIEWER_VALUES` | `[]` | claim values that grant `viewer` (read-only); at least one of the two lists must be non-empty |
| `PGW_GATEWAY_CONSOLE_OIDC_ALLOWED_DOMAINS` | `[]` (any) | optional allowlist of email domains (exact match; subdomains must be listed) |
| `PGW_GATEWAY_CONSOLE_OIDC_CA_FILE` | unset (system/certifi roots) | PEM CA bundle for the IdP's TLS certificate (internal CAs) |
| `PGW_GATEWAY_CONSOLE_OIDC_LABEL` | `SSO` | name on the sign-in button: "Sign in with <label>" |
| `PGW_GATEWAY_CONSOLE_TOKEN_LOGIN_ENABLED` | `true` | set `false` to make SSO mandatory for the console |

Notes:

* **Role mapping.** A user whose role claim contains an admin value is an admin; otherwise a user
  with a viewer value is a viewer; anybody else is refused with `403` and a
  `console.login_failed` audit record (`reason: oidc_no_role`, with subject and email). Group names
  are matched exactly (case-sensitive).
* **Domain allowlist.** When set, the ID token must carry an `email` claim in one of the listed
  domains, and `email_verified` must not be `false`. The role mapping remains the authorization
  control; the allowlist is defense in depth.
* **Mandatory SSO.** With `PGW_GATEWAY_CONSOLE_TOKEN_LOGIN_ENABLED=false` the sign-in page shows only
  the SSO button and `POST /console/api/login` answers `403`. `PGW_GATEWAY_ADMIN_TOKEN` keeps
  working as a Bearer token for scripts on `/admin/*`, which is also your break-glass access when
  the IdP is down. Keep that token in your secret manager.
* **SSO without an admin token.** SSO works without `PGW_GATEWAY_ADMIN_TOKEN`; then only SSO users
  can sign in to the console and `/admin/*` has no Bearer token.

## Network, TLS and clocks

* The gateway makes outbound HTTPS requests to the IdP: the discovery document
  (`<issuer>/.well-known/openid-configuration`, cached for an hour), the JWKS (cached for an hour,
  refetched when a token names an unknown key, at most every 30 s) and the token endpoint (once per
  sign-in). Allow these from the gateway's network. The IdP endpoints must be https. The standard
  `HTTPS_PROXY`/`NO_PROXY` environment variables apply.
* For an IdP with an internal CA, set `PGW_GATEWAY_CONSOLE_OIDC_CA_FILE`. TLS verification cannot be
  turned off.
* The browser must reach both the gateway and the IdP; the IdP never connects to the gateway.
* Keep the gateway clock synchronized (NTP). ID tokens are accepted with 60 s of clock skew.
* Users must open the console on the host of `PGW_GATEWAY_CONSOLE_OIDC_REDIRECT_URL`. A sign-in
  started on another host name fails with "started in another browser" because the binding cookie
  is per host.
* Pending sign-ins and sessions live in gateway memory. With several gateway replicas behind a load
  balancer, enable session affinity (sticky sessions) for `/console/`, or sign-ins will fail
  intermittently.

## Register the gateway at your IdP

Every IdP needs the same four things:

1. A **web** application (confidential client) using the authorization code flow, with PKCE
   allowed. Implicit and hybrid flows stay off.
2. The redirect URI, for example `https://gw.example.com/console/api/oidc/callback`.
3. A client secret, written to a file readable only by the gateway user (mode 0400).
4. A groups or roles claim in the **ID token** (not only in the access token or userinfo), with
   values you map to admin and viewer.

The examples below use the host name `gw.example.com` for the gateway and the secret file
`/etc/pgw/oidc-client-secret`. They show settings only; follow your IdP's documentation for the
registration steps.

### Microsoft Entra ID

Register an app (Web platform, redirect URI as above), create a client secret and define two app
roles (for example `PGW.Admin` and `PGW.Viewer`) assigned to groups in the Enterprise application.
App roles arrive in the `roles` claim. Use the tenant id (a GUID) in the issuer, not `common` or
`organizations`, because the issuer must match the discovery document exactly.

```bash
PGW_GATEWAY_CONSOLE_OIDC_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
PGW_GATEWAY_CONSOLE_OIDC_CLIENT_ID=<application-client-id>
PGW_GATEWAY_CONSOLE_OIDC_CLIENT_SECRET_FILE=/etc/pgw/oidc-client-secret
PGW_GATEWAY_CONSOLE_OIDC_REDIRECT_URL=https://gw.example.com/console/api/oidc/callback
PGW_GATEWAY_CONSOLE_OIDC_ROLE_CLAIM=roles
PGW_GATEWAY_CONSOLE_OIDC_ADMIN_VALUES='["PGW.Admin"]'
PGW_GATEWAY_CONSOLE_OIDC_VIEWER_VALUES='["PGW.Viewer"]'
PGW_GATEWAY_CONSOLE_OIDC_LABEL="Microsoft"
```

Prefer app roles over the `groups` claim: Entra puts group object ids (GUIDs) there and drops the
claim for users in many groups (group overage). If you use the domain allowlist, add the optional
`email` claim to the ID token; Entra sends no `email_verified`, so rely on the role mapping for
authorization.

### Okta

Create an OIDC Web application (grant type: authorization code; PKCE is accepted for web apps) and
add a `groups` claim to the ID token of your authorization server (filter, for example, groups
starting with `pgw-`). The example uses the `default` custom authorization server; for the org
authorization server the issuer is just the org URL.

```bash
PGW_GATEWAY_CONSOLE_OIDC_ISSUER=https://<your-org>.okta.com/oauth2/default
PGW_GATEWAY_CONSOLE_OIDC_CLIENT_ID=<client-id>
PGW_GATEWAY_CONSOLE_OIDC_CLIENT_SECRET_FILE=/etc/pgw/oidc-client-secret
PGW_GATEWAY_CONSOLE_OIDC_REDIRECT_URL=https://gw.example.com/console/api/oidc/callback
PGW_GATEWAY_CONSOLE_OIDC_SCOPES="openid email profile groups"
PGW_GATEWAY_CONSOLE_OIDC_ADMIN_VALUES='["pgw-admins"]'
PGW_GATEWAY_CONSOLE_OIDC_VIEWER_VALUES='["pgw-viewers"]'
PGW_GATEWAY_CONSOLE_OIDC_ALLOWED_DOMAINS='["example.com"]'
PGW_GATEWAY_CONSOLE_OIDC_LABEL="Okta"
```

### Keycloak

Create a confidential OpenID Connect client (client authentication on, standard flow on, direct
access grants off) and set "Proof Key for Code Exchange Code Challenge Method" to `S256`. Add a
"Group Membership" mapper with token claim name `groups`, "Full group path" off and "Add to ID
token" on. For an internal CA, give the gateway the CA bundle.

```bash
PGW_GATEWAY_CONSOLE_OIDC_ISSUER=https://sso.example.internal/realms/pgw
PGW_GATEWAY_CONSOLE_OIDC_CLIENT_ID=pgw-console
PGW_GATEWAY_CONSOLE_OIDC_CLIENT_SECRET_FILE=/etc/pgw/oidc-client-secret
PGW_GATEWAY_CONSOLE_OIDC_REDIRECT_URL=https://gw.example.com/console/api/oidc/callback
PGW_GATEWAY_CONSOLE_OIDC_ADMIN_VALUES='["pgw-admins"]'
PGW_GATEWAY_CONSOLE_OIDC_VIEWER_VALUES='["pgw-viewers"]'
PGW_GATEWAY_CONSOLE_OIDC_CA_FILE=/etc/pgw/internal-ca.pem
PGW_GATEWAY_CONSOLE_OIDC_LABEL="Keycloak"
PGW_GATEWAY_CONSOLE_TOKEN_LOGIN_ENABLED=false
```

To map realm roles instead of groups, enable "Add to ID token" on the realm roles mapper and set
`PGW_GATEWAY_CONSOLE_OIDC_ROLE_CLAIM=realm_access.roles`.

## Security properties

* Authorization code flow with PKCE (`S256`), `state` and `nonce`. The pending sign-in is single use,
  expires after 10 minutes and is bound to the browser that started it (`pgw_oidc` cookie,
  SameSite=Lax), so a sign-in link cannot be replayed or planted in another browser (login CSRF).
* The `iss` parameter of the authorization response is checked when the IdP sends it (RFC 9207,
  mix-up defense), and the discovery document must name the configured issuer.
* ID token signatures: RS256 (RSA keys of at least 2048 bits) and ES256 only. `alg: none`, HMAC
  (`HS256` and friends, including the public-key-as-HMAC-secret confusion), and tokens that try to
  bring their own key (`jwk`, `jku`, `x5u`) are refused. Verification uses the `cryptography`
  package; no JOSE library is added to the gateway.
* Claims: exact issuer, audience contains the client id, `azp` checked, `exp`/`iat`/`nbf` with 60 s
  leeway, `iat` at most 10 minutes old, nonce compared in constant time.
* The callback counts against the console login rate limit (10 attempts per minute per client
  address); sign-in starts are limited to 30 per minute per address.
* SSO sessions are ordinary console sessions: in memory, HttpOnly, SameSite=Strict, `Secure` except
  for plain-HTTP loopback, CSRF token required on writes, viewers read-only.
* The audit log records the subject and email of SSO sign-ins and refusals, never tokens or codes.

## Troubleshooting

Refused sign-ins show an error page and leave a `console.login_failed` audit record whose `reason`
names the cause.

| reason | likely cause |
|---|---|
| `oidc_bad_state` | the sign-in link was reused, took longer than 10 minutes, or the gateway restarted |
| `oidc_bad_binding` | the sign-in started on another host name or in another browser, or cookies are blocked |
| `oidc_provider_error` | the IdP refused the sign-in (for example the user is not assigned to the application) |
| `oidc_token_error` | wrong client secret or redirect URI, or the code expired at the IdP |
| `oidc_bad_issuer` | `PGW_GATEWAY_CONSOLE_OIDC_ISSUER` differs from the token's `iss` (check trailing slashes and tenant ids) |
| `oidc_bad_audience` | the token was issued for another client id |
| `oidc_expired` | the gateway clock is off; check NTP |
| `oidc_bad_alg`, `oidc_bad_signature`, `oidc_unknown_key` | the IdP signs with an unsupported algorithm (configure RS256 or ES256), or a forged token |
| `oidc_no_role` | the role claim is missing from the ID token or holds no mapped value |
| `oidc_domain_not_allowed` | the email is missing, unverified or outside `PGW_GATEWAY_CONSOLE_OIDC_ALLOWED_DOMAINS` |
| `oidc_unreachable`, `oidc_bad_discovery` | the gateway cannot reach the IdP (firewall, proxy, CA), or the discovery document does not match the issuer |

A `503` on **Sign in with SSO** means the discovery document could not be fetched; the gateway log
names the failing step. Token sign-in and Bearer access keep working while the IdP is unavailable.
