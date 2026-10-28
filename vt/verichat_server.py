"""HTTP server and web page for VeriChat (vt/verichat.py).

    python -m vt.verichat_server --model Qwen/Qwen2.5-7B-Instruct --head <claims head .pt> --port 8080

  GET  /         the chat page: conversation, optional documents, every claim coloured by P(correct),
                 hover for its source and citation
  GET  /health   status and the head's layer
  POST /chat     {"messages": [{"role", "content"}...], "documents": ["..."], "mode": "hedge"|"stop"|"flag"}
"""
import argparse
import threading

from .serve import HardenedHandler, bounded_server

MAX_MESSAGES, MAX_CHARS, MAX_DOCS, MAX_DOC_CHARS = 16, 4000, 4, 12000

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
  '<div class="meta">'+(r.withheld?'Answer withheld: the model doubted it. ':'')+'mode '+r.mode+'</div>'}
function add(cls,inner){const d=document.createElement('div');d.className='card '+cls;d.innerHTML=inner;log.appendChild(d);d.scrollIntoView()}
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


def make_server(bot, host="127.0.0.1", port=8080, max_body=1 << 17, deadline=120.0):
    lock = threading.Lock()

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
                return self.send_json(200, {"status": "ok", "layer": bot.gen.layer, "threshold": bot.threshold})
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/chat":
                return self.send_json(404, {"error": "not found"})
            req, err = self.read_json_object(max_body)
            if err:
                return self.send_json(err[0], {"error": err[1]})
            try:
                msgs, docs, mode = validate(req)
            except ValueError as e:
                return self.send_json(400, {"error": str(e)[:200]})
            with lock:                                           # one generation at a time
                r = bot.reply(msgs, documents=docs, mode=mode)
            self.send_json(200, r)

    return bounded_server(host, port, Handler, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    from .verichat import VeriChat
    bot = VeriChat.load(a.model, a.head, a.threshold)
    srv = make_server(bot, a.host, a.port)
    print(f"VeriChat on http://{a.host}:{srv.server_port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
