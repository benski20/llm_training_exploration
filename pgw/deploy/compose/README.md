# Local demo with docker compose

Runs the whole pipeline on one CPU machine: **pack -> factory -> service -> gateway**.

```bash
cd pgw/deploy/compose
cp .env.example .env              # optional: model, ring, number of mask slots, port
docker compose up --build -d
docker compose logs init          # prints the demo API key once (demo only)
KEY=pgw-...                       # from the log line "demo API key"
curl -s localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" \
     -H 'content-type: application/json' \
     -d '{"model":"demo","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

| service | image | what it does |
|---|---|---|
| `init` | gateway | one-shot, idempotent: `pgw pack` the model (downloads it once), `pgw maskstore init` (KEK), demo TLS certificates, a tenant token (`pgw service token create`), an API key (`pgw gateway keys create`), the gateway's models template |
| `factory` | factory | `pgw factory generate` masks into the encrypted store; **no network** |
| `factory-refill` | factory | optional (`--profile refill`): `pgw factory run` keeps the store topped up |
| `service` | service-cpu | stateless masked compute over TLS; sees only the read-only pack and its token hashes; internal network only |
| `gateway` | gateway | OpenAI-compatible API on `127.0.0.1:8080`; renders its models file from the template at start-up |

Masks are single use: every prompt or generated token consumes one slot (about 1.5 MB per slot for
Qwen2.5-0.5B at K40A12, so the default 2048 slots take about 3 GB). When the store runs low, start
the refill profile (`docker compose --profile refill up -d`) or rerun `docker compose run --rm factory`.

Inspect the stack:

```bash
docker compose exec gateway pgw doctor --role gateway --maskstore /var/lib/pgw/masks --key-file /var/lib/pgw/kek/kek
docker compose exec gateway pgw maskstore status --root /var/lib/pgw/masks --key-file /var/lib/pgw/kek/kek
curl -s localhost:8080/metrics | grep pgw_masks_available_slots
```

Everything generated here (CA, tokens, KEK, API key) is for the demo only. Production deployments
use the Helm charts or the VM image with secrets from your secret manager.

`tests/cli/test_deploy_compose.py` runs this exact flow (the init script, the factory command, and the
service and gateway with the environment from this compose file) without Docker on every CI run.
