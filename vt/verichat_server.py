"""HTTP server and web page for VeriChat (vt/verichat.py).

    python -m vt.verichat_server --model Qwen/Qwen2.5-7B-Instruct --head <claims head .pt> --port 8080

  GET  /         the chat page: conversation, optional documents, every claim coloured by P(correct),
                 hover for its source and citation
  GET  /health   status and the head's layer
  GET  /metrics  Prometheus counters: requests by path and status, latency, batch sizes
  POST /chat     {"messages": [{"role", "content"}...], "documents": ["..."], "mode": "hedge"|"stop"|"flag",
                 "stream": false, "thinking": null}; with "stream": true, server-sent events: {"delta"} pieces
                 of checked text ({"reasoning"} pieces of a reasoning model's thinking first, when it thinks),
                 then {"done": true, "reply": <the full reply>}. "thinking": true/false asks a reasoning
                 model (Qwen3) to think first or not (default: --thinking); its reasoning is returned apart,
                 unscored, as reply["reasoning"]. "temperature" (0-2; 0 = greedy) and "seed" override the
                 server's sampling (--temperature, --top-p, --top-k, --presence-penalty)
  POST /judge    {"messages": [...], "response": "...", "documents": [...]}: score a reply from any model with the
                 general detector (vt/detector.Detector; only with --detector); {"items": [...]} for up to
                 16 at once, batched into padded forward passes

POSTs are rate-limited per client (--rate-per-minute); with VERICHAT_API_KEY set they need
"Authorization: Bearer <key>". Concurrent requests are batched by one model-owning worker (up to 8
at once, 20 ms window); connections beyond the limit get 503.
"""
import argparse
import collections
import hmac
import json
import os
import queue
import threading
import time

from .serve import HardenedHandler, bounded_server

MAX_MESSAGES, MAX_CHARS, MAX_DOCS, MAX_DOC_CHARS, MAX_BATCH = 48, 4000, 4, 12000, 16   # the bot fits long chats

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
.c.m{background:var(--midbg);color:var(--mid)}div.c{display:block;margin:2px 0}.c.b{background:var(--badbg);color:var(--bad)}
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
<option value="flag">only score</option></select><label id="thinkl" hidden><input type="checkbox" id="think"> think first (slower)</label><button id="send">Send</button><span id="busy" class="meta"></span></div>
<details id="jp" hidden><summary>Check an answer from any AI</summary>
<textarea id="jq" rows="2" placeholder="The question or prompt it answered"></textarea>
<textarea id="ja" rows="5" placeholder="Its answer (documents above, if any, are used as the source)"></textarea>
<div class="row"><button id="jsend">Check</button><span id="jbusy" class="meta"></span></div><div id="jout"></div></details>
</main><script>
const log=document.getElementById('log'),msgs=[];
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function reasonBox(t){return '<details class="meta"><summary>Reasoning (not checked)</summary><div class="text">'+esc(t)+'</div></details>'}
function render(r){let t=r.text,html='',pos=0;
 for(const c of r.claims){if(!c.text)continue;const k=(c.start!==undefined&&c.start>=pos)?c.start:t.indexOf(c.text,pos);if(k<0)continue;
  const cls=c.verdict==='likely right'?'g':(c.verdict==='uncertain'?'m':'b');
  let tip='P(correct) '+c.score.toFixed(2);if(c.source)tip+=' | from '+c.source+(c.citation?': "'+c.citation.sentence+'"':'');
  html+=esc(t.slice(pos,k))+'<span class="c '+cls+'" title="'+esc(tip)+'">'+esc(t.slice(k,k+c.text.length))+'</span>';pos=k+c.text.length}
 html+=esc(t.slice(pos));html=html.split('[unsure:] ').join('').split('[unsure:]').join('');
 const bad=(r.sentences||[]).filter(x=>x.doubted);
 if(bad.length)html+='<div class="claims">'+bad.map(x=>'<div class="c b">Doubtful sentence ('+
  (100*x.risk).toFixed(0)+'% risk): '+esc(x.text)+'</div>').join('')+'</div>';
 const list=r.claims.filter(c=>c.text).map(c=>'<div><b>'+esc(c.text)+'</b> — '+c.verdict+' ('+c.score.toFixed(2)+')'+
  (c.source?' · from '+esc(c.source)+(c.citation?': “'+esc(c.citation.sentence)+'”':''):'')+'</div>').join('');
 return '<div class="text">'+html+'</div>'+(list?'<div class="claims">'+list+'</div>':'')+
  '<div class="meta">'+(r.withheld?'Answer withheld: the model doubted it. ':'')+
  (r.retrieval&&r.retrieval.retrieved?'The model doubted its first answer ('+esc(r.retrieval.closed_book_text)+'), so it looked it up. ':'')+
  'mode '+r.mode+'</div>'+(r.reasoning?reasonBox(r.reasoning):'')}
function add(cls,inner){const d=document.createElement('div');d.className='card '+cls;d.innerHTML=inner;log.appendChild(d);d.scrollIntoView()}
fetch('/health').then(r=>r.json()).then(h=>{if(h.detector)document.getElementById('jp').hidden=false;
 if(h.thinking_available){document.getElementById('thinkl').hidden=false;document.getElementById('think').checked=!!h.thinking}});
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
 const card=document.createElement('div');card.className='card bot';card.innerHTML='<div class="text"></div><div class="meta"></div>';log.appendChild(card);
 const live=card.firstChild,rlive=card.lastChild;let shown='',reasoned='';
 try{const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({messages:msgs.slice(-24),documents:docs,mode:document.getElementById('mode').value,stream:true,thinking:document.getElementById('think').checked})});
  if(!res.ok){const r=await res.json();card.innerHTML=esc('Error: '+(r.error||res.status))}
  else{const rd=res.body.getReader(),dec=new TextDecoder();let buf='',r=null;
   while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
    let k;while((k=buf.indexOf('\n\n'))>=0){const ev=buf.slice(0,k);buf=buf.slice(k+2);
     if(!ev.startsWith('data: '))continue;const m=JSON.parse(ev.slice(6));
     if(m.reasoning){reasoned+=m.reasoning;rlive.textContent='Thinking: '+reasoned.slice(-300);card.scrollIntoView()}
     if(m.delta){rlive.textContent='';shown+=m.delta;live.textContent=shown.split('[unsure:] ').join('').split('[unsure:]').join('');card.scrollIntoView()}
     if(m.done){if(m.error){card.innerHTML=esc('Error: '+m.error)}else{r=m.reply}}}}
   if(r){msgs.push({role:'assistant',content:r.text});card.innerHTML=render(r)}}}
 catch(e){card.innerHTML=esc('Error: '+e)}b.disabled=false;document.getElementById('busy').textContent=''};
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
    if req.get("thinking") is not None and not isinstance(req["thinking"], bool):
        raise ValueError("thinking: true, false or null")
    t = req.get("temperature")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 2):
        raise ValueError("temperature: a number from 0 to 2")
    if req.get("seed") is not None and (isinstance(req["seed"], bool) or not isinstance(req["seed"], int)):
        raise ValueError("seed: an integer")
    return msgs, docs, mode


def validate_judge(req):
    resp = req.get("response")
    if not isinstance(resp, str) or not resp.strip() or len(resp) > MAX_DOC_CHARS:
        raise ValueError(f"response: a non-empty string of up to {MAX_DOC_CHARS} chars")
    msgs, docs, _ = validate({"messages": req.get("messages"), "documents": req.get("documents")})
    return msgs, resp, docs


class Metrics:
    """Counters for GET /metrics (Prometheus text format)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = collections.Counter()                    # (path, status) -> n
        self.latency = collections.defaultdict(float)            # path -> seconds, successful POSTs
        self.batches = collections.Counter()                     # (kind, size) -> n

    def request(self, path, code, seconds=None):
        with self.lock:
            self.requests[(path, code)] += 1
            if seconds is not None:
                self.latency[path] += seconds

    def batch(self, kind, size):
        with self.lock:
            self.batches[(kind, size)] += 1

    def text(self):
        with self.lock:
            lines = ["# TYPE verichat_requests_total counter"]
            lines += [f'verichat_requests_total{{path="{p}",code="{c}"}} {n}' for (p, c), n in sorted(self.requests.items())]
            lines += ["# TYPE verichat_latency_seconds_total counter"]
            lines += [f'verichat_latency_seconds_total{{path="{p}"}} {v:.3f}' for p, v in sorted(self.latency.items())]
            lines += ["# TYPE verichat_batches_total counter"]
            lines += [f'verichat_batches_total{{kind="{k}",size="{n}"}} {c}' for (k, n), c in sorted(self.batches.items())]
        return "\n".join(lines) + "\n"


class Batcher:
    """One worker thread owns the model. Requests wait up to `window` seconds to be grouped with others
    of the same kind ("chat" / "judge") into one batched call (at most `max_batch`). A single owner
    also keeps hooks registered for provenance from seeing another request's forward pass."""

    def __init__(self, handlers, max_batch=8, window=0.02, metrics=None):
        self.handlers, self.max_batch, self.window, self.metrics = handlers, max_batch, window, metrics
        self.pending, self.cv = collections.deque(), threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, kind, item, timeout=None):
        return self.submit_many(kind, [item], timeout)[0]

    def submit_many(self, kind, items, timeout=None):
        slots = [{"done": threading.Event()} for _ in items]
        with self.cv:
            self.pending.extend((kind, item, slot) for item, slot in zip(items, slots))
            self.cv.notify()
        end = None if timeout is None else time.monotonic() + timeout
        for slot in slots:
            if not slot["done"].wait(None if end is None else max(0.0, end - time.monotonic())):
                raise TimeoutError("timed out waiting for the model")
            if "error" in slot:
                raise slot["error"]
        return [slot["result"] for slot in slots]

    def _take(self):
        with self.cv:
            while not self.pending:
                self.cv.wait()
            kind = self.pending[0][0]
            end = time.monotonic() + self.window
            while True:                                          # let a burst of requests arrive
                same = [p for p in self.pending if p[0] == kind]
                left = end - time.monotonic()
                if len(same) >= self.max_batch or left <= 0:
                    break
                self.cv.wait(left)
            batch = [p for p in self.pending if p[0] == kind][:self.max_batch]
            for p in batch:
                self.pending.remove(p)
            return kind, batch

    def _run(self):
        while True:
            kind, batch = self._take()
            if self.metrics:
                self.metrics.batch(kind, len(batch))
            try:
                results = self.handlers[kind]([item for _, item, _ in batch])
                for (_, _, slot), r in zip(batch, results):
                    slot["result"] = r
            except Exception as e:                               # every waiter gets the error, the worker lives
                for _, _, slot in batch:
                    slot["error"] = e
            for _, _, slot in batch:
                slot["done"].set()


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


def make_server(bot, host="127.0.0.1", port=8080, max_body=1 << 18, deadline=120.0, detector=None,
                api_key=None, rate_per_minute=20, trust_proxy=False, max_batch=8, max_concurrency=16):
    """api_key: if set, POSTs need "Authorization: Bearer <key>". trust_proxy: take the client from
    X-Forwarded-For (only behind a proxy you control). Concurrent requests are batched (max_batch)."""
    def chat(items):                                             # items may carry a "_stream" callback
        reqs = [{k: v for k, v in it.items() if k != "_stream"} for it in items]
        sinks = [it.get("_stream") for it in items]
        if not any(sinks):
            return bot.reply_batch(reqs)
        return bot.reply_batch(reqs, on_text=lambda i, d: sinks[i] and sinks[i]({"delta": d}),
                               on_reasoning=lambda i, d: sinks[i] and sinks[i]({"reasoning": d}))
    handlers = {"chat": chat}
    if detector is not None:
        handlers["judge"] = lambda items: detector.judge_batch(items)
    metrics = Metrics()
    batcher = Batcher(handlers, max_batch=max_batch, metrics=metrics)
    limiter = RateLimiter(rate_per_minute)

    class Handler(HardenedHandler):
        timeout_s, deadline_s = 15.0, deadline

        def on_response(self, code):
            metrics.request(self.path if self.path in ("/", "/health", "/metrics", "/chat", "/judge") else "other",
                            code, time.perf_counter() - self._t0 if code == 200 and hasattr(self, "_t0") else None)

        def do_GET(self):
            if self.path == "/metrics":
                body = metrics.text().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.on_response(200)
                return
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
                                            "detector": detector is not None, "thinking": bot.gen.thinking,
                                            "thinking_available": bot.gen.think_open is not None})
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            self._t0 = time.perf_counter()
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
                try:
                    rs = batcher.submit_many("judge", [{"messages": m, "response": r, "documents": d}
                                                       for m, r, d in parsed], deadline)
                except TimeoutError as e:
                    return self.send_json(503, {"error": str(e)})
                rs = [r if "error" in r else {k: r[k] for k in ("p_hallucinated", "flagged", "spans")} for r in rs]
                return self.send_json(200, {"items": rs} if batch is not None else rs[0])
            try:
                msgs, docs, mode = validate(req)
            except ValueError as e:
                return self.send_json(400, {"error": str(e)[:200]})
            item = {"messages": msgs, "documents": docs, "mode": mode, "thinking": req.get("thinking")}
            over = {k: req[k] for k in ("temperature", "seed") if req.get(k) is not None}
            if over:                                             # the server's sampling with these overrides
                item["sampling"] = dict(bot.gen.sampling or {}, **over)
                item["think_sampling"] = dict(bot.gen.think_sampling or {}, **over)
            if req.get("stream"):
                return self.stream_chat(item)
            try:                                                 # batched with concurrent requests
                r = batcher.submit("chat", item, deadline)
            except TimeoutError as e:
                return self.send_json(503, {"error": str(e)})
            self.send_json(200, r)

        def stream_chat(self, item):
            """Server-sent events: {"reasoning": text} while a reasoning model thinks, {"delta": text} as checked
            text is released, then {"done": true, "reply": ...}."""
            q, box = queue.Queue(), {}
            item["_stream"] = q.put

            def run():
                try:
                    box["reply"] = batcher.submit("chat", item, deadline)
                except Exception as e:                           # timeout or model error: reported in the stream
                    box["error"] = str(e)[:200]
                q.put(None)
            threading.Thread(target=run, daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    d = q.get()
                    if d is None:
                        break
                    self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                    self.wfile.flush()
                end = {"done": True, "reply": box["reply"]} if "reply" in box else {"done": True, "error": box["error"]}
                self.wfile.write(b"data: " + json.dumps(end).encode() + b"\n\n")
                self.wfile.flush()
            except OSError:                                      # the client went away
                return
            self.on_response(200)

    return bounded_server(host, port, Handler, max_concurrency)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vt verichat")
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--corpus", help="a folder of .txt/.md files: look questions up there when the model doubts itself")
    ap.add_argument("--retrieve-below", type=float, default=0.3)
    ap.add_argument("--detector", help="a kaggle/detector head (*_detector.pt) to enable /judge")
    ap.add_argument("--detector-model", help="its reader model (default: the head's own)")
    ap.add_argument("--moderation", action="store_true", help="screen messages and replies (vt/moderation.py)")
    ap.add_argument("--rate-per-minute", type=int, default=20, help="POSTs per client per minute (0: unlimited)")
    ap.add_argument("--trust-proxy", action="store_true", help="client address from X-Forwarded-For")
    ap.add_argument("--thinking", action="store_true", help="reasoning models think first unless a request says not to")
    ap.add_argument("--think-budget", type=int, default=1024, help="most tokens of thinking per reply")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (0: greedy); Qwen3's advice")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=1.5, help="against loops (1.5: 7x fewer repeated 4-grams, same accuracy)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    from .verichat import VeriChat
    bot = VeriChat.load(a.model, a.head, a.threshold)
    bot.gen.thinking, bot.gen.think_budget = a.thinking, a.think_budget
    from .verichat import apply_sampling
    apply_sampling(bot, a.temperature, a.top_p, a.top_k, a.presence_penalty)
    if a.moderation:
        from .moderation import Moderator
        bot.moderator = Moderator.load()
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
