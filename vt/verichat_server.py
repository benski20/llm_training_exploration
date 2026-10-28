"""HTTP server and web page for VeriChat (vt/verichat.py).

    python -m vt.verichat_server --model Qwen/Qwen2.5-7B-Instruct --head <claims head .pt> --port 8080

  GET  /         the chat page: conversation, optional documents, every claim coloured by P(correct),
                 hover for its source and citation
  GET  /health   status and the head's layer
  POST /chat     {"messages": [{"role", "content"}...], "documents": ["..."], "mode": "hedge"|"stop"|"flag"}
  POST /judge    {"messages": [...], "response": "...", "documents": [...]}: score a reply from any model with the
                 general detector (vt/detector.Detector; only with --detector); {"items": [...]} for up to
                 16 at once, batched into padded forward passes

POSTs are rate-limited per client (--rate-per-minute); with VERICHAT_API_KEY set they need
"Authorization: Bearer <key>". Generation runs one request at a time; extra connections get 503.
"""
import argparse
import collections
import hmac
import os
import threading
import time

from .serve import HardenedHandler, bounded_server

MAX_MESSAGES, MAX_CHARS, MAX_DOCS, MAX_DOC_CHARS, MAX_BATCH = 16, 4000, 4, 12000, 16

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VeriChat</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--card:#ffffff;--line:#e4e3df;--good:#1f7a4d;--goodbg:#e3f3ea;
--mid:#8a5a00;--midbg:#fbefd6;--bad:#a8231c;--badbg:#f8e0de;--accent:#2f5bd3}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe7;--muted:#9a9a93;--card:#1f1f1d;--line:#33332f;
--good:#6fd3a0;--goodbg:#173527;--mid:#f0c060;--midbg:#3a2e12;--bad:#ff8a80;--badbg:#3d1c1a;--accent:#8aa8ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 120px}h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.user{border-left:3px solid var(--accent)}.bot .text{white-space:pre-wrap}
.c{border-radius:4px;padding:0 2px;cursor:help}.c.g{background:var(--goodbg);color:var(--good)}
.c.m{background:var(--midbg);color:var(--mid)}.c.b{background:var(--badbg);color:var(--bad)}
.meta{color:var(--muted);font-size:13px;margin-top:6px}.claims{font-size:13px;margin-top:8px}
.claims div{padding:2px 0;border-top:1px dashed var(--line)}textarea{width:100%;background:var(--card);color:var(--fg);
border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit}details{margin:8px 0}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}button{background:var(--accent);color:#fff;border:0;
border-radius:8px;padding:8px 16px;font:inherit;cursor:pointer}button:disabled{opacity:.5}select{font:inherit;padding:6px;border-radius:6px}
</style></head><body><main>
<h1>VeriChat</h1>
<p class="sub">Every name, number and date in a reply is scored by the model's own internals. Green: likely right.
Amber: uncertain. Red: doubtful. With documents, each claim shows whether it came from them (with the sentence) or from memory.</p>
<div id="log"></div>
<details><summary>Documents (optional)</summary><textarea id="docs" rows="5" placeholder="Paste source text. Separate documents with a blank line containing ---"></textarea></details>
<textarea id="msg" rows="3" placeholder="Ask anything"></textarea>
<div class="row"><select id="mode"><option value="hedge">mark doubtful claims</option><option value="stop">withhold doubtful answers</option>
<option value="flag">only score</option></select><button id="send">Send</button><span id="busy" class="meta"></span></div>
<details id="jp" hidden><summary>Check an answer from any AI</summary>
<textarea id="jq" rows="2" placeholder="The question or prompt it answered"></textarea>
<textarea id="ja" rows="5" placeholder="Its answer (documents above, if any, are used as the source)"></textarea>
<div class="row"><button id="jsend">Check</button><span id="jbusy" class="meta"></span></div><div id="jout"></div></details>
</main><script>
const log=document.getElementById('log'),msgs=[];
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function render(r){let t=r.text,html='',pos=0;
 for(const c of r.claims){if(!c.text)continue;const k=(c.start!==undefined&&c.start>=pos)?c.start:t.indexOf(c.text,pos);if(k<0)continue;
  const cls=c.verdict==='likely right'?'g':(c.verdict==='uncertain'?'m':'b');
  let tip='P(correct) '+c.score.toFixed(2);if(c.source)tip+=' | from '+c.source+(c.citation?': "'+c.citation.sentence+'"':'');
  html+=esc(t.slice(pos,k))+'<span class="c '+cls+'" title="'+esc(tip)+'">'+esc(t.slice(k,k+c.text.length))+'</span>';pos=k+c.text.length}
 html+=esc(t.slice(pos));html=html.split('[unsure:] ').join('').split('[unsure:]').join('');
 const list=r.claims.filter(c=>c.text).map(c=>'<div><b>'+esc(c.text)+'</b> — '+c.verdict+' ('+c.score.toFixed(2)+')'+
  (c.source?' · from '+esc(c.source)+(c.citation?': “'+esc(c.citation.sentence)+'”':''):'')+'</div>').join('');
 return '<div class="text">'+html+'</div>'+(list?'<div class="claims">'+list+'</div>':'')+
  '<div class="meta">'+(r.withheld?'Answer withheld: the model doubted it. ':'')+
  (r.retrieval&&r.retrieval.retrieved?'The model doubted its first answer ('+esc(r.retrieval.closed_book_text)+'), so it looked it up. ':'')+
  'mode '+r.mode+'</div>'}
function add(cls,inner){const d=document.createElement('div');d.className='card '+cls;d.innerHTML=inner;log.appendChild(d);d.scrollIntoView()}
fetch('/health').then(r=>r.json()).then(h=>{if(h.detector)document.getElementById('jp').hidden=false});
function renderJudge(r,a){let html='',pos=0;for(const s of r.spans){html+=esc(a.slice(pos,s.start))+'<span class="c b" title="'+
 esc('P(hallucinated) '+s.score.toFixed(2))+'">'+esc(a.slice(s.start,s.end))+'</span>';pos=s.end}html+=esc(a.slice(pos));
 return '<div class="card"><div class="text">'+html+'</div><div class="meta">'+(r.flagged?'Likely contains a hallucination':
 'No hallucination detected')+' (P = '+r.p_hallucinated.toFixed(2)+'). Red: the spans the reader doubts most.</div></div>'}
document.getElementById('jsend').onclick=async()=>{const q=document.getElementById('jq').value.trim(),a=document.getElementById('ja').value.trim();
 if(!q||!a)return;const docs=document.getElementById('docs').value.split(/\n\s*---\s*\n/).map(s=>s.trim()).filter(Boolean);
 const b=document.getElementById('jsend');b.disabled=true;document.getElementById('jbusy').textContent='reading…';
 try{const res=await fetch('/judge',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({messages:[{role:'user',content:q}],response:a,documents:docs})});const r=await res.json();
  document.getElementById('jout').innerHTML=r.error?esc('Error: '+r.error):renderJudge(r,a)}
 catch(e){document.getElementById('jout').textContent='Error: '+e}b.disabled=false;document.getElementById('jbusy').textContent=''};
document.getElementById('send').onclick=async()=>{const m=document.getElementById('msg').value.trim();if(!m)return;
 const docs=document.getElementById('docs').value.split(/\n\s*---\s*\n/).map(s=>s.trim()).filter(Boolean);
 msgs.push({role:'user',content:m});add('user',esc(m));document.getElementById('msg').value='';
 const b=document.getElementById('send');b.disabled=true;document.getElementById('busy').textContent='thinking…';
 try{const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({messages:msgs,documents:docs,mode:document.getElementById('mode').value})});
  const r=await res.json();if(r.error){add('bot',esc('Error: '+r.error))}else{msgs.push({role:'assistant',content:r.text});add('bot',render(r))}}
 catch(e){add('bot',esc('Error: '+e))}b.disabled=false;document.getElementById('busy').textContent=''};
</script></body></html>"""


def validate(req):
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not 1 <= len(msgs) <= MAX_MESSAGES:
        raise ValueError(f"messages: a list of 1-{MAX_MESSAGES}")
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str) \
                or len(m["content"]) > MAX_CHARS:
            raise ValueError("each message needs role user/assistant and content up to %d chars" % MAX_CHARS)
    if msgs[-1]["role"] != "user":
        raise ValueError("the last message must be the user's")
    docs = req.get("documents") or []
    if not isinstance(docs, list) or len(docs) > MAX_DOCS or any(not isinstance(d, str) or len(d) > MAX_DOC_CHARS for d in docs):
        raise ValueError(f"documents: up to {MAX_DOCS} strings of up to {MAX_DOC_CHARS} chars")
    mode = req.get("mode", "hedge")
    if mode not in ("hedge", "stop", "flag"):
        raise ValueError("mode: hedge, stop or flag")
    return msgs, docs, mode


def validate_judge(req):
    resp = req.get("response")
    if not isinstance(resp, str) or not resp.strip() or len(resp) > MAX_DOC_CHARS:
        raise ValueError(f"response: a non-empty string of up to {MAX_DOC_CHARS} chars")
    msgs, docs, _ = validate({"messages": req.get("messages"), "documents": req.get("documents")})
    return msgs, resp, docs


class RateLimiter:
    """At most `per_minute` POSTs per client in any 60 s window (0: unlimited)."""

    def __init__(self, per_minute):
        self.per_minute, self.hits, self.lock = per_minute, collections.defaultdict(collections.deque), threading.Lock()

    def allow(self, client, now=None):
        if not self.per_minute:
            return True
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self.hits) > 10000:                           # forget idle clients
                for k in [k for k, v in self.hits.items() if not v]:
                    del self.hits[k]
            return True


def make_server(bot, host="127.0.0.1", port=8080, max_body=1 << 17, deadline=120.0, detector=None,
                api_key=None, rate_per_minute=20, trust_proxy=False):
    """api_key: if set, POSTs need "Authorization: Bearer <key>". trust_proxy: take the client from
    X-Forwarded-For (only behind a proxy you control)."""
    lock = threading.Lock()
    limiter = RateLimiter(rate_per_minute)

    class Handler(HardenedHandler):
        timeout_s, deadline_s = 15.0, deadline

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/health":
                return self.send_json(200, {"status": "ok", "layer": bot.gen.layer, "threshold": bot.threshold,
                                            "detector": detector is not None})
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/chat", "/judge") or (self.path == "/judge" and detector is None):
                return self.send_json(404, {"error": "not found"})
            if api_key and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {api_key}"):
                return self.send_json(401, {"error": "missing or wrong API key"})
            fwd = self.headers.get("X-Forwarded-For", "") if trust_proxy else ""
            client = fwd.split(",")[0].strip() or self.client_address[0]
            if not limiter.allow(client):
                return self.send_json(429, {"error": f"rate limit: {rate_per_minute} requests per minute"})
            req, err = self.read_json_object(max_body)
            if err:
                return self.send_json(err[0], {"error": err[1]})
            if self.path == "/judge":                          # one item, or {"items": [...]} (up to MAX_BATCH)
                try:
                    batch = req.get("items")
                    if batch is not None and (not isinstance(batch, list) or not 1 <= len(batch) <= MAX_BATCH):
                        raise ValueError(f"items: a list of 1-{MAX_BATCH}")
                    parsed = [validate_judge(x if isinstance(x, dict) else {}) for x in (batch or [req])]
                except ValueError as e:
                    return self.send_json(400, {"error": str(e)[:200]})
                with lock:
                    rs = detector.judge_batch([{"messages": m, "response": r, "documents": d} for m, r, d in parsed])
                rs = [r if "error" in r else {k: r[k] for k in ("p_hallucinated", "flagged", "spans")} for r in rs]
                return self.send_json(200, {"items": rs} if batch is not None else rs[0])
            try:
                msgs, docs, mode = validate(req)
            except ValueError as e:
                return self.send_json(400, {"error": str(e)[:200]})
            with lock:                                           # one generation at a time
                r = bot.reply(msgs, documents=docs, mode=mode)
            self.send_json(200, r)

    return bounded_server(host, port, Handler, 4)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vt verichat")
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--corpus", help="a folder of .txt/.md files: look questions up there when the model doubts itself")
    ap.add_argument("--retrieve-below", type=float, default=0.3)
    ap.add_argument("--detector", help="a kaggle/detector head (*_detector.pt) to enable /judge")
    ap.add_argument("--detector-model", help="its reader model (default: the head's own)")
    ap.add_argument("--rate-per-minute", type=int, default=20, help="POSTs per client per minute (0: unlimited)")
    ap.add_argument("--trust-proxy", action="store_true", help="client address from X-Forwarded-For")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    from .verichat import VeriChat
    bot = VeriChat.load(a.model, a.head, a.threshold)
    if a.corpus:
        from .retrieval import Corpus
        bot.retriever, bot.retrieve_below = Corpus.from_dir(a.corpus), a.retrieve_below
    det = None
    if a.detector:
        import torch
        from .detector import Detector
        name = a.detector_model or torch.load(a.detector, map_location="cpu", weights_only=True)["model"]
        det = Detector(bot.model, bot.tok, torch.load(a.detector, map_location="cpu", weights_only=True)) \
            if name == a.model else Detector.load(name, a.detector)
    srv = make_server(bot, a.host, a.port, detector=det, api_key=os.environ.get("VERICHAT_API_KEY") or None,
                      rate_per_minute=a.rate_per_minute, trust_proxy=a.trust_proxy)
    print(f"VeriChat on http://{a.host}:{srv.server_port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
