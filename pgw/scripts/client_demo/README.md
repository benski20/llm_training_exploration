# Client demo

A web page that runs the real PGW stack (factory, mask store, compute service, gateway) on one
machine and shows, side by side:

- **What your team sees:** a chat with the model through the gateway's OpenAI-compatible API.
- **What our cloud service receives:** the actual bytes on the wire, captured by a byte-level relay
  between gateway and service. You see the values, their distribution over the whole ring and the
  layer names.
- **Verification:** round trips, Freivalds-checked replies, caught tampering and one-time masks left.
- **Attack button:** the relay flips one bit in one service reply. The gateway rejects the answer and
  trips its integrity breaker. **Reset** clears the breaker through the admin API.

```bash
python scripts/client_demo/run.py            # Qwen2.5-0.5B-Instruct on CPU, UI on http://127.0.0.1:8080/
python scripts/client_demo/run.py --model Qwen/Qwen2.5-1.5B-Instruct --name qwen2.5-1.5b-instruct --slots 600
```

State lives in `~/.pgw-client-demo` (pack, masks, demo-only keys, logs). On a 4-vCPU CPU-only host,
0.5B runs at about 1.3 tokens/s, because each token is 96 sequential masked round trips. A GPU
service and speculative decoding are the planned fixes. All keys and tokens are demo-only. Gateway
and service use plain HTTP on loopback; production uses TLS/mTLS.
