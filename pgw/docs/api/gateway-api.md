# Gateway API reference (OpenAI-compatible)

The PGW gateway serves an OpenAI-compatible HTTP API inside your network. Existing OpenAI client
libraries work unchanged: point their base URL at the gateway and use a PGW API key. Every request
is executed by the masked protocol. Prompts and completions never leave your boundary in plaintext.

Base URL: `https://<gateway-host>:<port>` (default port 8080). All bodies are JSON (UTF-8). The
maximum request body is `PGW_GATEWAY_MAX_BODY_BYTES` (default 4 MiB).

## 1. Authentication

| caller | header | notes |
|---|---|---|
| Applications | `Authorization: Bearer pgw-<key_id>-<secret>` | `key_id` is 12 hex characters; only a SHA-256 hash of the secret is stored. Keys carry scopes (`inference`, `admin`), an optional model allow-list, and limits |
| Operators | `Authorization: Bearer <admin token>` | `PGW_GATEWAY_ADMIN_TOKEN`, or an API key with the `admin` scope |

Keys are created with `pgw gateway keys create` or `POST /admin/keys`, and the plaintext is shown
**once**. A key restricted to some models gets `model_not_found` (404) for others, so other model
names are not disclosed to it.

Anonymous access exists only for development (`allow_anonymous` together with `allow_dev_modes`)
and is refused under the production policy.

## 2. Endpoints

| method | path | auth | purpose |
|---|---|---|---|
| POST | `/v1/chat/completions` | key (`inference`) | chat completion, streaming or not |
| POST | `/v1/completions` | key (`inference`) | text completion, streaming or not |
| GET | `/v1/models` | key | models this key may use |
| GET | `/v1/models/{name}` | key | one model card |
| GET | `/admin/status` | admin | models, readiness, fingerprints, breaker, noise monitor, policy |
| GET | `/admin/masks` | admin | mask store inventory per model (no secrets) |
| GET | `/admin/keys` | admin | key records (no hashes) and tokens used today |
| POST | `/admin/keys` | admin | create a key; returns it once |
| DELETE | `/admin/keys/{key_id}` | admin | revoke a key |
| POST | `/admin/models/{name}/reload` | admin | reload a model (pack, service check) |
| POST | `/admin/models/{name}/reset` | admin | reset the integrity circuit breaker and/or the noise alarm |
| GET | `/healthz` | none | process alive: `{"status":"ok"}` |
| GET | `/readyz` | none | 200 when every model is ready, else 503 with reasons |
| GET | `/metrics` | none or admin | Prometheus text (`metrics_require_admin`) |
| GET | `/dashboard` | none (page) / admin (data) | self-contained status page |

Every inference response carries an `x-request-id` header. The same id appears in the audit log.

## 3. `POST /v1/chat/completions`

### Request

| field | type | default | notes |
|---|---|---|---|
| `model` | string | required | a name from `/v1/models` |
| `messages` | array | required | roles `system`, `developer`, `user`, `assistant`, `tool`; `content` is a string or a list of `{"type":"text","text":...}` parts. Other part types (images) are rejected |
| `max_completion_tokens` / `max_tokens` | int ≥ 1 | `PGW_GATEWAY_DEFAULT_MAX_TOKENS` (512) | clamped to `max_tokens_limit` (8192) and to the context left |
| `temperature` | 0–2 | 1.0 | 0 means greedy |
| `top_p` | (0, 1] | 1.0 | |
| `stop` | string or list of up to 16 strings | none | |
| `seed` | int | none | reproducible sampling |
| `presence_penalty`, `frequency_penalty` | −2 to 2 | 0 | |
| `stream` | bool | false | Server-Sent Events |
| `stream_options.include_usage` | bool | false | final chunk with `usage` |
| `logprobs`, `top_logprobs` | bool, 0–20 | false | |
| `n` | int | 1 | **only 1** is supported |
| `user` | string | none | accepted and ignored (not logged) |
| `top_k`, `min_p`, `repetition_penalty`, `ignore_eos`, `stop_token_ids` | | | vLLM-style extensions |

Unknown fields are ignored. Tools and function calling, response formats and multimodal inputs are
not supported in v1.

Each request reserves `prompt_tokens + max_tokens` mask slots before any compute. If the store has
fewer, the request fails immediately with `mask_exhausted` (503). Unused slots are burned when the
request ends, so keep `max_tokens` realistic.

### Example

```bash
curl -s https://pgw.internal:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen7b","messages":[{"role":"user","content":"Summarise the attached note."}],"max_tokens":128}'
```

```json
{
  "id": "chatcmpl-6f1c0e4b2a9d4c33b8e1a2f0",
  "object": "chat.completion",
  "created": 1763337600,
  "model": "qwen7b",
  "system_fingerprint": "3fa9c2d1e07b",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."},
               "logprobs": null, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 31, "completion_tokens": 54, "total_tokens": 85}
}
```

`system_fingerprint` is the first 12 hex characters of the model pack's SHA-256 fingerprint. Use it
to prove which exact weights produced an answer. `finish_reason` is `stop`, `length`, `error` or
`cancelled`.

Python (official OpenAI SDK):

```python
from openai import OpenAI
client = OpenAI(base_url="https://pgw.internal:8080/v1", api_key="pgw-<key_id>-<secret>")
r = client.chat.completions.create(model="qwen7b", messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

### Streaming

With `"stream": true` the response is `text/event-stream`. Each event is a line
`data: {chat.completion.chunk}`, followed by a blank line. The first chunk carries
`delta.role = "assistant"`. Later chunks carry `delta.content`, and the last content chunk carries
`finish_reason`. With `include_usage`, a final chunk has `"choices": []` and `usage`. The stream ends
with `data: [DONE]`. An error during a stream is sent as one `data: {"error": {...}}` event, followed
by `data: [DONE]`.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],...}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}],...}

data: [DONE]
```

Closing the connection cancels generation and burns the remaining reserved slots.

## 4. `POST /v1/completions`

Same sampling fields as chat, plus:

| field | type | notes |
|---|---|---|
| `prompt` | string or list of strings | up to `max_prompts_per_request` (16); streaming needs a single prompt |
| `echo` | bool | prepend the prompt to `text` |
| `logprobs` | int 0–5 | legacy completions format |
| `suffix` | string | not supported (error) |

Response: `object: "text_completion"`, with `choices[i].text`, `finish_reason` and `usage`.

## 5. Models

`GET /v1/models` returns `{"object":"list","data":[{"id":"qwen7b","object":"model","created":...,"owned_by":"pgw"}]}`.

## 6. Errors

All errors use the OpenAI shape. `code` is a stable PGW error code (SPEC 6.6), and `type` follows
OpenAI conventions.

```json
{"error": {"message": "mask store has 120 slots, request needs 640", "type": "server_error", "code": "mask_exhausted", "param": null}}
```

| code | HTTP | `type` | when | client action |
|---|---|---|---|---|
| `protocol` | 400 | `invalid_request_error` | malformed JSON or invalid field (`param` names it) | fix the request |
| `bad_shape` | 400 | `invalid_request_error` | prompt too long for the context | shorten the prompt or `max_tokens` |
| `ring_overflow` | 400 | `invalid_request_error` | value not representable (should not occur with valid packs) | report |
| `unauthenticated` | 401 | `authentication_error` | missing, malformed, unknown or revoked key | check the key |
| `permission_denied` | 403 | `permission_error` | key lacks the scope | use a key with the right scope |
| `model_not_found` | 404 | `not_found_error` | unknown model, or not allowed for this key | check `/v1/models` |
| `fingerprint_mismatch` | 409 | `invalid_request_error` | service or masks do not match the model pack | operator action |
| `payload_too_large` | 413 | `invalid_request_error` | body above `max_body_bytes` | send less |
| `version_mismatch` | 426 | `invalid_request_error` | protocol version not supported by the service | operator action |
| `rate_limited` | 429 | `rate_limit_error` (`insufficient_quota` for the daily quota) | per-key RPM, TPM, concurrency or daily quota | wait for `Retry-After` |
| `internal` | 500 | `server_error` | bug (details are not exposed) | retry later; report the `x-request-id` |
| `config` | 500 | `server_error` | invalid configuration | operator action |
| `mask_reuse` | 500 | `server_error` | attempted mask reuse: traffic stopped | **never retry**; security incident |
| `mask_store` | 500 | `server_error` | store I/O, decryption or format failure | operator action |
| `integrity_check_failed` | 502 | `server_error` | compute result failed Freivalds or the range check, or the outbound-noise alarm blocks masked traffic (fail-closed) | do not retry; operator investigates and resets |
| `mask_exhausted` | 503 | `server_error` | not enough masks for `prompt + max_tokens` | retry after refill, or lower `max_tokens` |
| `backend_unavailable` | 503 | `server_error` | compute backend not available | retry with backoff |
| `unavailable` | 503 | `server_error` | queue full, draining, service unreachable, integrity circuit breaker open | retry with backoff, honour `Retry-After` |
| `deadline_exceeded` | 504 | `server_error` | queue or service deadline passed | retry with backoff |

`Retry-After` (seconds) is set on 429 and 503 responses when the gateway knows a wait time.

## 7. Limits

| limit | setting (`PGW_GATEWAY_*`) | default |
|---|---|---|
| requests per minute per key | `REQUESTS_PER_MINUTE` (per key: `--rpm`) | 600 |
| tokens per minute per key (prompt + completion) | `TOKENS_PER_MINUTE` (per key: `--tpm`) | 200,000 |
| tokens per UTC day per key | `DAILY_TOKEN_QUOTA` (per key: `--daily-tokens`) | unlimited |
| prompt tokens | `MAX_PROMPT_TOKENS` | 32,768 (and the model context) |
| `max_tokens` cap | `MAX_TOKENS_LIMIT` | 8192 |
| prompts per `/v1/completions` request | `MAX_PROMPTS_PER_REQUEST` | 16 |
| body size | `MAX_BODY_BYTES` | 4 MiB |
| concurrent sequences per model | model `max_concurrent_sessions` | 32 |
| queued requests per model | model `max_queue` / `queue_timeout_s` | 256 / 30 s |

## 8. Admin API

All admin calls need the admin token or an `admin`-scoped key. Every call writes an audit event.

### `POST /admin/keys`

```json
{"name": "billing-app", "scopes": ["inference"], "models": ["qwen7b"],
 "requests_per_minute": 120, "tokens_per_minute": 100000, "daily_token_quota": 2000000,
 "max_concurrent_requests": 4}
```

Response `201`: `{"key": "pgw-<key_id>-<secret>", "record": {...}}`. The key is shown only here.

### `DELETE /admin/keys/{key_id}`

Returns `{"revoked": "<key_id>"}`. Revocation takes effect immediately.

### `POST /admin/models/{name}/reset`

Body (optional, both default `true`): `{"breaker": true, "noise": true}`. Clears the integrity circuit
breaker and/or the outbound-noise alarm. Do this **only** after the cause has been investigated
(see the operator guide).

### `GET /admin/status`

Returns uptime, `production_policy` (true unless dev modes are allowed), and per model: `mode`,
`ready`, `fingerprint`, `ring`, `sessions`, `breaker`, `freivalds_rate`, service info,
`masks_available` and noise-monitor state. It also lists the effective egress allowlist and the
audit-log path. `telemetry` is always `"disabled"`: the gateway sends nothing to us except compute
requests.

## 9. Metrics

`/metrics` exposes Prometheus metrics from the contract list (`pgw.observability.metrics.METRIC_NAMES`).
Labels never contain prompt data. Gateway series:

| metric | type | meaning |
|---|---|---|
| `pgw_gateway_requests_total` | counter | requests by model, endpoint, status |
| `pgw_gateway_tokens_total` | counter | prompt and completion tokens |
| `pgw_gateway_ttft_seconds` | histogram | time to first token |
| `pgw_gateway_tpot_seconds` | histogram | time per output token |
| `pgw_gateway_round_trip_seconds` | histogram | compute round trips |
| `pgw_gateway_active_sessions` | gauge | generating sequences |
| `pgw_gateway_integrity_failures_total` | counter | Freivalds or range-check failures (**alert on any**) |
| `pgw_gateway_noise_alarm` | gauge | 1 when the outbound-noise monitor alarms (**alert**) |
| `pgw_masks_available_slots` | gauge | slots in the store per model and ring |
| `pgw_masks_low_water` | gauge | 1 below low water (**alert**) |
| `pgw_masks_consumed_slots_total` | counter | slots used |
| `pgw_masks_burned_slots_total` | counter | reserved but unused slots |

## 10. Compatibility notes

* `n > 1`, `best_of`, tools and function calling, `response_format`, audio and images: not supported
  (400 `protocol`, or the field is ignored where OpenAI treats it as optional).
* Embeddings and fine-tuning endpoints are not provided.
* Token counts use the model's own tokenizer, from the model pack.
* Rate-limit headers in the `x-ratelimit-*` style are not sent. Use `Retry-After`.
