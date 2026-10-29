"""python experiments/pool_stress.py new|old  --  Can a connection-pooling client (like Modal's aiohttp proxy) lose requests to vt's HTTP server?"""
import asyncio, json, sys, threading, time
sys.path.insert(0, ".")
import numpy as np, torch, aiohttp
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from vt.correctness import Probe
from vt.verichat import VeriChat
from vt import serve
from vt.verichat_server import make_server
mode = sys.argv[1]
if mode == "old":                                   # the behaviour before the fix
    from http.server import BaseHTTPRequestHandler
    def send_response(self, code, message=None):
        self.request.settimeout(self.write_timeout_s)
        BaseHTTPRequestHandler.send_response(self, code, message)
    serve.HardenedHandler.send_response = send_response
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"); torch.manual_seed(0)
m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2)).eval()
p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
bot = VeriChat(m, tok, p, layer=1)
srv = make_server(bot, port=0, rate_per_minute=0); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}/chat"

async def main(n=200, conc=8):
    res = {"ok": 0, "err": {}}
    conn = aiohttp.TCPConnector(limit=32, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=60)) as s:
        sem = asyncio.Semaphore(conc)
        async def one(i):
            async with sem:
                await asyncio.sleep((i % 7) * 0.05)
                try:
                    async with s.post(url, json={"messages": [{"role": "user", "content": f"Hi {i}"}], "mode": "flag"}) as r:
                        await r.read(); res["ok"] += r.status == 200
                        if r.status != 200: res["err"][r.status] = res["err"].get(r.status, 0) + 1
                except Exception as e:
                    k = type(e).__name__; res["err"][k] = res["err"].get(k, 0) + 1
        t = time.time(); await asyncio.gather(*[one(i) for i in range(n)])
        res["seconds"] = round(time.time() - t, 1)
    print(mode, json.dumps(res), flush=True)
asyncio.run(main())
srv.shutdown()
