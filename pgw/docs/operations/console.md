# Operator console

The PGW Console is a web app for gateway operators. It shows model health, mask inventory, usage
and the audit log, manages API keys, and has a chat playground. The gateway serves it at
`/console/` from the same process and port as the API. It is a static single-page app with no
external resources (no CDN, fonts or analytics). Its data API is described in the
[console API contract](../api/console-api.md).

## Enabling it

The console is on by default. It needs an admin token, because sign-in checks tokens against it.

| setting | default | meaning |
|---|---|---|
| `PGW_GATEWAY_CONSOLE_ENABLED` | `true` | serve `/console/`; `false` removes every `/console/` path (the `/admin/*` API still works with the Bearer admin token) |
| `PGW_GATEWAY_ADMIN_TOKEN` | none | admin token (at least 16 characters); without one, console sign-in answers `503` |
| `PGW_GATEWAY_ADMIN_TOKEN_FILE` | none | read the admin token from a file instead (a mounted Docker or Kubernetes secret); read at start-up |
| `PGW_GATEWAY_CONSOLE_VIEWER_TOKENS_FILE` | none | read-only tokens, one per line, `#` comments allowed; re-read when the file changes |
| `PGW_GATEWAY_AUDIT_LOG_FILE` | none | the audit log; the usage charts and the audit page read it |

The built app ships inside the `pgw` package (`pgw/console/static/`) and therefore in the gateway
image. If it is missing, `/console/` shows a short "not built" page and the rest of the gateway
works normally. `pgw doctor --role gateway` warns when no admin token is set and fails when the
viewer tokens file does not exist (the gateway refuses to start in that case).

Deployment assets:

* **Helm** (`deploy/helm/pgw-gateway`): `console.enabled` (default `true`) and
  `console.viewerTokensKey`, the key in `secrets.existingSecret` that holds the viewer tokens. The
  chart mounts it as a file under `/etc/pgw/console/`. `consoleIngress` publishes only `/console`
  and `/admin` on a separate Ingress (see [Network exposure](#network-exposure)).
* **docker compose demo** (`deploy/compose`): the `init` service writes a random admin token to the
  `gw-secrets` volume and prints it once. The gateway reads it with `PGW_GATEWAY_ADMIN_TOKEN_FILE`.
  Open `http://localhost:8080/console/`.
* **systemd** (`deploy/systemd`): the unit already loads the admin token as a credential;
  `gateway.env.example` shows the console settings.

Example: viewer tokens for the Helm chart.

```bash
openssl rand -hex 24 > viewers.txt          # one token per line; add a line per viewer
kubectl create secret generic pgw-gateway-secrets \
  --from-literal=admin-token="$(openssl rand -hex 32)" \
  --from-literal=kek="$(openssl rand -base64 32)" \
  --from-file=service-token=./service-token \
  --from-file=console-viewers=./viewers.txt
helm upgrade --install gw deploy/helm/pgw-gateway --reuse-values \
  --set console.viewerTokensKey=console-viewers
```

## Roles and tokens

| role | signs in with | can |
|---|---|---|
| admin | the gateway admin token | everything: create and revoke API keys, reload models, reset the safety breaker, use the playground |
| viewer | a token from the viewer tokens file | read every page; no buttons that change state, no playground |

The console has no user accounts. Everyone with the admin token is the same admin. Hand out
viewer tokens instead of the admin token to people who only need to look. Give each viewer their
own token so you can revoke one person by deleting their line from the file; the gateway notices
the change without a restart. Existing sessions of a removed token stay valid until they expire
(at most 24 hours) or the gateway restarts.

Rotate the admin token by changing the secret and restarting the gateway. A restart ends all
console sessions, because sessions live in memory.

The gateway keeps only SHA-256 digests of the tokens in memory and compares them in constant time.
The audit log records each sign-in with an actor id such as `console-admin:3f2a9c1e`. That id is
random per session and contains no token material. Admin actions taken in the console carry it in
`key_id`, so you can tell console actions from scripts that use the Bearer token (`admin-token`).

## Sessions, CSRF and CSP

* **Sessions.** Sign-in sets the `pgw_console` cookie: `HttpOnly`, `SameSite=Strict`, `Path=/`,
  and `Secure` except for plain HTTP between loopback addresses. A session expires after 8 hours
  without use or 24 hours after sign-in, whichever comes first. At most 1000 sessions exist at once.
  Sessions are not shared between gateway processes and do not survive a restart.
* **CSRF.** Every state-changing request made with the cookie must carry the session's CSRF token
  in the `X-PGW-CSRF` header. The app does this itself. `SameSite=Strict` is a second layer.
* **Login rate limit.** 10 attempts per minute per client address. Failed attempts are audited as
  `console.login_failed`.
* **Content-Security-Policy.** Every `/console/` response carries
  `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self';
  frame-ancestors 'none'; base-uri 'none'; form-action 'self'`, plus `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer` and `X-Content-Type-Options: nosniff`. The app uses no inline
  scripts or styles. API responses carry `Cache-Control: no-store`.
* **Same origin.** The app calls only the gateway that served it (`/console/api/*` and `/admin/*`).
  Do not add the console's origin to `PGW_GATEWAY_CORS_ORIGINS`; it is not needed.

Scripts keep working unchanged: every `/admin/*` endpoint still accepts
`Authorization: Bearer <admin token>`.

## Network exposure

The console is exactly as reachable as `/admin/*`: same listener, same port. Treat it as an
administrative interface.

Recommendations:

1. **Keep it internal.** Publish `/console` and `/admin` only on an internal network: a VPN, a
   corporate network, or an internal load balancer. Applications only need `/v1`.
2. **Split the ingress.** In Kubernetes, route only `/v1` through the ingress that applications use,
   and enable `consoleIngress` on an internal ingress class with its own host name:

   ```yaml
   ingress:
     enabled: true
     className: nginx
     hosts:
       - host: pgw.example.internal
         paths: [{path: /v1, pathType: Prefix}]
   consoleIngress:
     enabled: true
     className: nginx-internal
     host: pgw-console.example.internal
     tls: [{secretName: pgw-console-tls, hosts: [pgw-console.example.internal]}]
   networkPolicy:
     ingressFrom:
       - podSelector: {}
       - namespaceSelector:
           matchLabels: {kubernetes.io/metadata.name: ingress-nginx}
   ```

   The chart refuses a `consoleIngress` without TLS. `helm install` prints a note when the API
   ingress routes `/`, because that also publishes `/admin` and `/console`.
3. **Use HTTPS.** Terminate TLS at the ingress or on the gateway (`tls.enabled` in the chart). The
   session cookie is `Secure`, so browsers drop it over plain HTTP except on `localhost`.
4. **One origin, no path rewriting.** Serve the console at `/console/` on its own host name. The app
   is built for the `/console/` base path and calls `/admin/*` on the same host. A prefix rewrite
   (for example `/pgw/console`) breaks it.
5. **Without an ingress**, use a port-forward and open `http://localhost:8080/console/`:

   ```bash
   kubectl -n pgw port-forward svc/gw-pgw-gateway 8080:8080
   ```

6. **Turn it off** with `PGW_GATEWAY_CONSOLE_ENABLED=false` (`console.enabled=false`) where nobody
   uses it. The probe and metrics listener (`PGW_GATEWAY_OPS_PORT`) never serves the console.

Behind a proxy, the gateway sees the proxy's address as the client. The login rate limit then
applies to all users of that proxy together. uvicorn trusts `X-Forwarded-For` only from the
addresses in its `FORWARDED_ALLOW_IPS` environment variable (default: loopback); set it to the
ingress controller's addresses if you want per-user limits and client addresses in the audit log.

## Playground, keys and quotas

The playground sends chat requests through the normal inference path. They run as the built-in key
`console-playground`, never as a stored API key. That key:

* is available to admin sessions only;
* has the gateway's default limits: `PGW_GATEWAY_REQUESTS_PER_MINUTE`,
  `PGW_GATEWAY_TOKENS_PER_MINUTE` and `PGW_GATEWAY_DAILY_TOKEN_QUOTA` (daily usage is persisted in
  `PGW_GATEWAY_QUOTA_STATE_FILE` like any key);
* runs at most 4 requests at once;
* consumes masks like any request (one slot per prompt or generated token), so keep `Max tokens`
  small on a gateway with a small mask store;
* is audited with `key_id` `console-playground` and appears in the usage charts under that name.

Prompts and completions are never written to the audit log, by the playground or otherwise.

API keys created in the console are the same keys as `pgw gateway keys create` makes, stored in
`PGW_GATEWAY_API_KEYS_FILE`. The secret is shown once in the dialog. The gateway keeps only its
hash. Revoking a key takes effect immediately.

## Audit and usage data

| page | source | notes |
|---|---|---|
| Overview, usage charts | the audit log (`PGW_GATEWAY_AUDIT_LOG_FILE`) | `inference` events of the last 1 h, 24 h or 7 d; cached for 5 s |
| Audit log | the audit log | newest first, filter by key, model, event and status |
| Models, Masks, Security | live gateway state (`/admin/status`, `/admin/masks`) | counters reset when a model restarts |
| API keys | the key file plus quota state | tokens today from the quota tracker |

The audit log is append-only JSON lines (mode 0600) with an allowlist of fields: timestamps, event
names, key ids, model, status, token and mask counts, durations. It never contains prompts,
completions or secrets. Each record is also written to the gateway's log stream.

Retention: the gateway never deletes or rotates the audit file. Choose the retention your policy
requires and rotate with a tool that keeps the file in place (logrotate `copytruncate`), because the
gateway keeps the file open. The console reads only the current file, and at most the last 128 MiB
of it per request. Older data is not shown (responses then say `truncated`); archive rotated files
for audits. Without an audit log file, the usage charts show zeros and the audit page is empty.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| "PGW Console is not built" page | the installed package lacks `pgw/console/static/`; install a release build or rebuild the console (`pgw/console/README.md`) |
| sign-in answers 503 "console login is disabled" | no admin token: set `PGW_GATEWAY_ADMIN_TOKEN` or `PGW_GATEWAY_ADMIN_TOKEN_FILE` |
| 404 on `/console/` | `PGW_GATEWAY_CONSOLE_ENABLED=false`, or a proxy rewrites or does not forward `/console` |
| sign-in succeeds but the next page asks to sign in again | the cookie was dropped: plain HTTP to a host other than `localhost`; use HTTPS |
| "That token was not accepted" | wrong token, or a viewer token that is not in the viewer file (check for stray spaces; one token per line) |
| "Too many sign-in attempts" | 10 attempts per minute per client address; behind a proxy all users share one address (see [Network exposure](#network-exposure)) |
| gateway does not start, "cannot read console_viewer_tokens_file" | the file is missing or unreadable for the gateway user; fix the path or the mount |
| buttons missing, "Read-only" badge | signed in with a viewer token; sign in with the admin token |
| writes fail with 403 "missing or invalid CSRF token" | the session was replaced (for example a sign-in in another tab); reload the page |
| usage charts empty | no `PGW_GATEWAY_AUDIT_LOG_FILE`, or the file was rotated recently |
| blank page, CSP errors in the browser console | a proxy injects scripts or styles, or rewrites paths; serve `/console/` unchanged |
| playground answers 429 | `console-playground` hit the default rate limit or daily quota |

`pgw config show gateway` prints the effective settings (secrets redacted). Sign-ins, sign-outs and
failed attempts are in the audit log as `console.login`, `console.logout` and
`console.login_failed`.
