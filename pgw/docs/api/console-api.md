# PGW Console: API contract (v1)

The console is a static single-page app served by the gateway at `/console/` (built from
`pgw/console/`, output committed to `src/pgw/console/static/`). It talks only to the gateway it is
served from. No external network resources: no CDNs, web fonts or analytics.

## Authentication

| method | path | body / response |
|---|---|---|
| POST | `/console/api/login` | `{"token": "<admin or viewer token>"}` -> `200 {"role", "csrf"}` (role is `admin` or `viewer`), sets cookie `pgw_console` (HttpOnly, SameSite=Strict, Secure unless plain-HTTP loopback, Path=/, 8 h idle expiry) |
| POST | `/console/api/logout` | -> `204`, clears the session |
| GET | `/console/api/me` | -> `200 {"role", "csrf", "expires_at", "version"}` or `401` |

- The console is served when `PGW_GATEWAY_CONSOLE_ENABLED` is true (the default). When false, nothing
  under `/console/` exists and session cookies are ignored; the `/admin/*` data endpoints below still
  work with the Bearer admin token.
- Sessions live in gateway memory (lost on restart, not shared between replicas): 8 h idle expiry,
  24 h absolute expiry, at most 1000 at once (least recently used dropped). The session id is a random
  256-bit value; the gateway stores only its SHA-256. `expires_at` is Unix seconds (the earlier of
  idle and absolute expiry; every authenticated request refreshes the idle timer). The cookie carries
  `Max-Age` of the absolute lifetime. `version` is the gateway package version (`pgw.__version__`).
- `Secure` is omitted only when the request is plain HTTP, the `Host` is a loopback name or address
  and the client address is loopback (local development). Behind a TLS-terminating proxy, run the
  gateway with the proxy's forwarded scheme (uvicorn `--proxy-headers`) or keep the default
  (non-loopback `Host`), so the cookie stays `Secure`.
- Login body must be JSON. Attempts are limited to 10 per minute per client address (all attempts
  count; `429` with `Retry-After` above that). Audit events: `console.login` (`reason` = role),
  `console.login_failed`, `console.logout`; the actor id is `console-<role>:<8 hex>` (a random
  per-session id, never token or cookie material). Admin actions made through the console carry that
  actor id in `key_id`.
- If no admin token is configured, `POST /console/api/login` and `GET /console/api/me` answer `503`
  (`unavailable`, "console login is disabled") even when viewer tokens exist.
- `POST /console/api/logout` needs `X-PGW-CSRF` when a live session cookie is sent (viewers may sign
  out); without a live session it answers `204` anyway and clears the cookie.
- Cookie-authenticated errors: `401` no or expired session, `403` (`permission_denied`) for a missing
  or wrong CSRF token or a viewer write.

- Admin tokens: the gateway's existing `PGW_GATEWAY_ADMIN_TOKEN`. Viewer tokens: optional
  `PGW_GATEWAY_CONSOLE_VIEWER_TOKENS_FILE` (one token per line; SHA-256 stored in memory only).
- Every console request that changes state (POST, PUT, DELETE) must send header `X-PGW-CSRF: <csrf>`.
- The admin endpoints below accept either `Authorization: Bearer <admin token>` (unchanged, for scripts)
  or a console session cookie, plus the CSRF header for writes. Viewers may only use GET.
- Login attempts are rate limited (per client IP) and audited; failures return `401 {"error": {...}}`.

## Data endpoints (all JSON)

| method | path | role | notes |
|---|---|---|---|
| GET | `/admin/status` | viewer | existing: models, backend counters, sessions, breaker, client, noise monitor |
| GET | `/admin/masks` | viewer | existing: mask inventory per model |
| GET | `/admin/keys` | viewer | existing: API keys (never the secret) |
| POST | `/admin/keys` | admin | existing: create; the secret is shown once |
| DELETE | `/admin/keys/{key_id}` | admin | existing: revoke |
| POST | `/admin/models/{name}/reload` | admin | existing |
| POST | `/admin/models/{name}/reset` | admin | existing |
| GET | `/admin/overview` | viewer | NEW: `{"version", "uptime_s", "models": [{"name", "state", "ring", "ready", "reason", "masks_available", "masks_runway_s"}], "requests_24h", "tokens_24h", "errors_24h", "integrity_failures_total", "freivalds_checked_total", "stream_requests_total", "privacy": {"masked": true, "ring", "freivalds_rate", "noise_monitor": {...}}}` |
| GET | `/admin/usage?window=24h` (1h, 24h or 7d) | viewer | NEW: `{"window", "bucket_s", "series": [{"t": <unix s>, "requests", "prompt_tokens", "completion_tokens", "errors", "p50_ms", "p95_ms", "ttft_p50_ms", "masks_consumed"}], "by_model": {...}, "by_key": {...}}` from the audit log |
| GET | `/admin/audit?limit=100&before=<ts>&key_id=&model=&status=` | viewer | NEW: newest-first audit events (the JSONL records; never prompts or completions) `{"events": [...], "next_before": <ts or null>}` |
| POST | `/console/api/playground/chat` | admin | NEW: same body as `/v1/chat/completions` (incl. `stream`); runs as the built-in key `console-playground` so usage is attributed and quota-limited; SSE when streaming |

Errors use the gateway's existing shape `{"error": {"message", "type", "code", "param"}}`.

### Computed fields (what the numbers mean)

- `/admin/overview`: `requests_24h`, `tokens_24h` (prompt + completion) and `errors_24h` come from the
  same audit-log aggregation as `/admin/usage?window=24h` (`inference` events). An error is an
  `inference` event with `status >= 400` (including 4xx such as rate limits). `masks_runway_s` is
  `masks_available / (masks consumed by that model in the last 24 h / 86400)`, `null` when nothing was
  consumed or the model is not masked (`masks_available` is `null` for dev-mode models).
  `integrity_failures_total` and `freivalds_checked_total` are the masked backends' counters since the
  model last started (process-local; the Prometheus counter `pgw_gateway_integrity_failures_total` is
  the durable alerting signal). `stream_requests_total` counts streaming API requests since gateway
  start-up. `privacy.masked` is true when every model runs in `masked` mode; `privacy.ring` is the
  masked models' ring (comma-separated if they differ); `privacy.freivalds_rate` is the lowest rate of
  the masked models; `privacy.noise_monitor` is `{"enabled", "fail_closed", "alarm", "models": {name:
  {...status}}}`. Each model entry also has `mode`; the response also has `production_policy`.
- `/admin/usage`: buckets are 60 s (`1h`), 900 s (`24h`) and 3600 s (`7d`); `t` is the bucket start
  (aligned to the bucket size), every bucket of the window is present (empty ones have zero counts and
  `null` percentiles). `p50_ms`/`p95_ms` are nearest-rank percentiles of the request `duration_ms`
  (for streams: until the stream closed), `ttft_p50_ms` of `ttft_ms`. `by_model` and `by_key` map a
  model name / `key_id` to the same fields without `t` (`by_key` entries add `name`, the key's name
  when known). Extra fields: `totals` (the whole window), `source` (`"audit_log"`, or `null` when no
  `PGW_GATEWAY_AUDIT_LOG_FILE` is configured: then all counts are zero) and `truncated` (the read
  bound of 128 MiB from the end of the file was hit, so older events in the window are missing).
  Only the current audit file is read (not rotated ones). Results are cached for 5 s.
- `/admin/audit`: `limit` is 1-1000 (default 100). `before` is exclusive and takes an audit `ts`
  string (as returned in `next_before`) or Unix seconds. Filters (exact match): `key_id`, `model`,
  `event` (e.g. `inference`, `admin.key_created`, `console.login`) and `status` (`200`, `2xx`, `4xx`,
  `5xx`, `ok` = below 400, `error` = 400 and above). Events sharing the timestamp of the last returned
  event are all included, so a page can exceed `limit` slightly but paging never skips events. Each
  event has only `ts`, `event` and the audit allowlist fields (`pgw.gateway.audit.AUDIT_FIELDS`).
  Extra field `truncated`: one call reads at most 128 MiB; if that bound was hit, `next_before`
  continues below the oldest line examined (the page may then hold fewer than `limit` events).
- `/console/api/playground/chat` accepts the admin session (with `X-PGW-CSRF`) or the Bearer admin
  token; viewers and API keys get `403`. It runs as `console-playground` with the gateway default
  limits (`PGW_GATEWAY_REQUESTS_PER_MINUTE`, `PGW_GATEWAY_TOKENS_PER_MINUTE`,
  `PGW_GATEWAY_DAILY_TOKEN_QUOTA`) and at most 4 concurrent requests; its audit records have
  `key_id: "console-playground"` and `endpoint: "chat"`.

## Security headers on `/console/*`

`Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:;
connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`, `Cache-Control: no-store` on API
responses, long-lived immutable caching for hashed assets.

`Cache-Control: no-store` is also set on `/admin/*` responses. Static files: `/console/assets/*` get
`public, max-age=31536000, immutable` (so asset file names must contain a content hash),
`index.html` and other top-level files `no-cache`. `GET /console` redirects (`308`) to `/console/`.
A path under `/console/` that is not a file falls back to `index.html` (client-side routes) unless it
is under `assets/` or its last segment contains a dot (then `404`); unknown `/console/api/*` paths
are JSON `404`s. Paths with `..`, hidden (dot) segments or backslashes never resolve outside the
static directory. If the console has not been built (`src/pgw/console/static/index.html` missing),
`/console/` and client routes answer `200` with a short page saying so.
