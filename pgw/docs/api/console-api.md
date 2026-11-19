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

## Security headers on `/console/*`

`Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:;
connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`, `Cache-Control: no-store` on API
responses, long-lived immutable caching for hashed assets.
