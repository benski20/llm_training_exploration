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
import itertools
import json
import os
import queue
import threading
import time
import traceback

from .serve import HardenedHandler, bounded_server

MAX_NEW_TOKENS = 1536                                                               # step-by-step answers need room
MAX_MESSAGES, MAX_CHARS, MAX_DOCS, MAX_DOC_CHARS, MAX_BATCH = 48, 8000, 4, 12000, 16   # the bot fits long chats

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VeriChat</title>
<style>
:root{--bg:#faf9f5;--side:#f3f1ea;--fg:#1f1e1b;--muted:#75736c;--card:#ffffff;--line:#e6e3d8;--soft:#efece3;
--accent:#c96442;--accent2:#b5573a;--good:#2f7d4f;--goodbg:#e4f1e7;--mid:#94650c;--midbg:#f8edd3;--bad:#b3362a;--badbg:#f8e1dd;
--shadow:0 1px 2px rgba(40,30,10,.05),0 4px 18px rgba(40,30,10,.06);--serif:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,Cambria,serif;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#262624;--side:#1f1e1c;--fg:#f2f0e8;--muted:#a3a097;--card:#30302d;
--line:#3d3c38;--soft:#35342f;--accent:#d97757;--accent2:#e08a6c;--good:#7fd1a2;--goodbg:#1e3a29;--mid:#eec46d;--midbg:#3d3117;
--bad:#ff9384;--badbg:#45231f;--shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.25)}}
*{box-sizing:border-box}[hidden]{display:none!important}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 var(--sans);display:flex;overflow:hidden}
button{font:inherit;color:inherit;cursor:pointer}textarea,select{font:inherit;color:inherit}
/* sidebar */
aside{width:264px;flex:none;background:var(--side);border-right:1px solid var(--line);display:flex;flex-direction:column;padding:14px 12px;gap:4px;transition:transform .2s}
.brand{display:flex;align-items:center;gap:9px;font:600 19px/1 var(--serif);padding:6px 8px 16px;letter-spacing:-.01em}
.mark{width:26px;height:26px;flex:none}
.nav{display:flex;align-items:center;gap:10px;width:100%;text-align:left;background:none;border:0;border-radius:9px;padding:8px 10px;color:var(--fg)}
.nav:hover{background:var(--soft)}.nav.on{background:var(--soft);font-weight:600}.nav svg{width:18px;height:18px;flex:none;opacity:.8}
.newchat{color:var(--accent);font-weight:600}
.side-h{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;padding:18px 10px 6px}
.setting{padding:4px 10px;font-size:13.5px;color:var(--muted)}.setting label{display:flex;gap:8px;align-items:center;cursor:pointer}
.setting select{width:100%;margin-top:4px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 8px}
.side-foot{margin-top:auto;font-size:12px;color:var(--muted);padding:10px;line-height:1.5}
.status{display:inline-flex;align-items:center;gap:6px}.dot{width:7px;height:7px;border-radius:50%;background:var(--muted)}.dot.ok{background:var(--good)}
/* main column */
.app{flex:1;display:flex;flex-direction:column;min-width:0;height:100%}
header{display:flex;align-items:center;gap:10px;padding:12px 18px;min-height:52px}
header .title{font-weight:500;color:var(--muted);font-size:14px}
.menu{display:none;background:none;border:0;padding:6px;border-radius:8px}.menu:hover{background:var(--soft)}
.scroll{flex:1;overflow-y:auto}
.col{max-width:760px;margin:0 auto;padding:8px 20px 24px}
.view[hidden]{display:none}
/* empty state */
.empty{text-align:center;padding:12vh 0 18px}
.empty h1{font:400 34px/1.2 var(--serif);margin:14px 0 8px;letter-spacing:-.015em}
.empty p{color:var(--muted);max-width:560px;margin:0 auto}
.empty .mark{width:40px;height:40px}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:22px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:7px 14px;font-size:13.5px;color:var(--fg)}
.chip:hover{border-color:var(--accent);color:var(--accent)}
/* messages */
.msg{margin:22px 0}.msg.user{display:flex;justify-content:flex-end}
.bubble{background:var(--soft);border-radius:18px;padding:10px 16px;max-width:85%;white-space:pre-wrap;overflow-wrap:anywhere}
.msg.bot{display:flex;gap:14px}.msg.bot .mark{width:24px;height:24px;margin-top:4px}
.body{flex:1;min-width:0}
.text{white-space:pre-wrap;font:16.5px/1.7 var(--serif);overflow-wrap:anywhere}
.thinking{color:var(--muted);font-size:13.5px;display:flex;align-items:center;gap:8px}
.pulse{width:8px;height:8px;border-radius:50%;background:var(--accent);animation:p 1.1s ease-in-out infinite}
@keyframes p{0%,100%{opacity:.25;transform:scale(.8)}50%{opacity:1;transform:scale(1)}}
.meta{color:var(--muted);font-size:13px;margin-top:8px}
.claims{font-size:13.5px;margin-top:12px;border:1px solid var(--line);border-radius:12px;padding:6px 12px;background:var(--card)}
.claims>div{padding:5px 0;border-top:1px solid var(--line)}.claims>div:first-child{border-top:0}
details.meta summary{cursor:pointer}
.c{border-radius:4px;padding:0 2px;cursor:help}.c.g{background:var(--goodbg);color:var(--good)}
.c.m{background:var(--midbg);color:var(--mid)}.c.b{background:var(--badbg);color:var(--bad)}div.c{display:block;margin:2px 0}
.ds{text-decoration:underline wavy var(--bad);text-underline-offset:3px;cursor:help}
/* verdict card */
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin:16px 0;box-shadow:var(--shadow)}
.vhead{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.pill{border-radius:999px;padding:4px 12px;font-weight:600;font-size:13.5px}
.pill.bad{background:var(--badbg);color:var(--bad)}.pill.mid{background:var(--midbg);color:var(--mid)}.pill.good{background:var(--goodbg);color:var(--good)}.pill.na{background:var(--soft);color:var(--muted)}
.meter{flex:1;min-width:140px;height:6px;border-radius:3px;background:linear-gradient(90deg,var(--good),var(--mid) 55%,var(--bad));position:relative;opacity:.85}
.meter i{position:absolute;top:-4px;width:3px;height:14px;border-radius:2px;background:var(--fg)}
.ev{display:flex;align-items:center;gap:10px}.ev .lab{flex:1}.ev .bar{width:90px;height:6px;border-radius:3px;background:var(--soft);position:relative;overflow:hidden}
.ev .bar b{position:absolute;top:0;bottom:0;border-radius:3px}.ev .num{width:46px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
/* composer */
.dock{padding:0 20px 18px}.dock .inner{max-width:760px;margin:0 auto}
.composer{background:var(--card);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow);padding:12px 12px 10px 16px;transition:border-color .15s}
.composer:focus-within{border-color:color-mix(in srgb,var(--accent) 45%,var(--line))}
.composer textarea{width:100%;border:0;outline:0;resize:none;background:transparent;padding:4px 0;max-height:240px;line-height:1.5}
.tools{display:flex;align-items:center;gap:6px;margin-top:6px}
.ibtn{display:inline-flex;align-items:center;gap:6px;background:none;border:1px solid var(--line);border-radius:10px;padding:5px 10px;font-size:13px;color:var(--muted)}
.ibtn:hover{background:var(--soft);color:var(--fg)}.ibtn svg{width:16px;height:16px}.ibtn.on{color:var(--accent);border-color:var(--accent)}
.spacer{flex:1}
.send{width:34px;height:34px;border-radius:10px;border:0;background:var(--accent);color:#fff;display:inline-flex;align-items:center;justify-content:center}
.send:hover{background:var(--accent2)}.send:disabled{opacity:.4;cursor:default}.send svg{width:18px;height:18px}
.docs{margin-bottom:10px}.docs[hidden]{display:none}
.docs textarea{width:100%;background:var(--card);border:1px dashed var(--line);border-radius:14px;padding:10px 14px;outline:0;font-size:13.5px}
.hint{text-align:center;font-size:12px;color:var(--muted);margin-top:8px}
/* verify view */
.vtitle{font:400 28px/1.25 var(--serif);margin:28px 0 6px;letter-spacing:-.01em}
.vsub{color:var(--muted);margin:0 0 18px}
.field{display:block;font-size:13px;color:var(--muted);margin:12px 0 4px}
.box{width:100%;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px 14px;outline:0;resize:vertical}
.box:focus{border-color:color-mix(in srgb,var(--accent) 45%,var(--line))}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
.primary{background:var(--accent);color:#fff;border:0;border-radius:10px;padding:9px 18px;font-weight:600}
.primary:hover{background:var(--accent2)}.primary:disabled{opacity:.5}
.ghost{background:none;border:1px solid var(--line);border-radius:10px;padding:8px 14px;color:var(--fg)}.ghost:hover{background:var(--soft)}
.scrim{display:none}
.mdhr{border:0;border-top:1px solid var(--line);margin:4px 0}
.acts{display:flex;gap:4px;margin-top:8px}.act{background:none;border:0;border-radius:8px;padding:4px 9px;font-size:12.5px;color:var(--muted)}
.act:hover{background:var(--soft);color:var(--fg)}.act:disabled{opacity:.5}
.wake{display:flex;align-items:center;gap:10px;background:var(--midbg);color:var(--mid);border-radius:12px;padding:10px 14px;font-size:13.5px;margin:4px 0 0}
#hist{display:flex;flex-direction:column;gap:1px;overflow-y:auto;min-height:0;flex:0 1 auto}
.hitem{font-size:13.5px;padding:6px 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block}
.exs{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 6px}.exs .meta{margin:0;align-self:center}
@media (max-width:820px){aside{position:fixed;inset:0 auto 0 0;z-index:20;transform:translateX(-100%);box-shadow:var(--shadow)}
 body.open aside{transform:none}body.open .scrim{display:block;position:fixed;inset:0;background:rgba(0,0,0,.25);z-index:10}
 .menu{display:inline-flex}.col,.dock{padding-left:16px;padding-right:16px}.empty{padding-top:6vh}.empty h1{font-size:28px}}
</style></head><body>
<svg width="0" height="0" style="position:absolute"><defs>
<symbol id="logo" viewBox="0 0 32 32"><circle cx="16" cy="16" r="15" fill="var(--accent)"/><path d="M9.5 16.5l4.2 4.2 8.8-9.4" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></symbol>
</defs></svg>
<aside>
 <div class="brand"><svg class="mark"><use href="#logo"/></svg>VeriChat</div>
 <button class="nav newchat" id="newchat"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="10" cy="10" r="8"/><path d="M10 6.5v7M6.5 10h7"/></svg>New chat</button>
 <button class="nav on" id="navchat"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M3.5 5.5a2 2 0 012-2h9a2 2 0 012 2v6a2 2 0 01-2 2H9l-3.5 3v-3h0a2 2 0 01-2-2z"/></svg>Chat</button>
 <button class="nav" id="jp" hidden><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2.5l6 2.5v4.5c0 4-2.6 6.6-6 8-3.4-1.4-6-4-6-8V5z"/><path d="M7.2 10l2 2 3.6-3.8"/></svg>Verify an answer</button>
 <div class="side-h" id="histh" hidden>Recent</div><div id="hist"></div>
 <div class="side-h">Settings</div>
 <div class="setting">When a claim looks doubtful
  <select id="mode"><option value="hedge">Mark it in the reply</option><option value="stop">Withhold the answer</option><option value="flag">Only score it</option></select></div>
 <div class="setting" id="thinkl" hidden><label><input type="checkbox" id="think"> Think before answering (slower)</label></div>
 <div class="side-foot"><div class="status"><span class="dot" id="dot"></span><span id="stat">Connecting…</span></div>
  <div style="margin-top:8px">Scores rank how likely a reply is to be wrong. They are evidence, not proof.</div></div>
</aside>
<div class="scrim" id="scrim"></div>
<div class="app">
 <header><button class="menu" id="menu" aria-label="Menu"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 6h14M3 10h14M3 14h14"/></svg></button>
  <span class="title" id="htitle">Chat</span></header>
 <div class="scroll" id="scroll">
  <div class="col" style="padding-bottom:0"><div class="wake" id="wake" hidden><span class="pulse"></span><span>Waking up the GPU server. The first
   visit after a pause takes 1–3 minutes; you can start typing meanwhile.</span></div></div>
  <section class="view col" id="vchat">
   <div class="empty" id="empty"><svg class="mark"><use href="#logo"/></svg>
    <h1 id="greet">How can I help you today?</h1>
    <p>Every reply is checked by the model's own internals as it is written. Doubtful sentences are underlined in red; names,
     numbers and dates are coloured green (likely right), amber (uncertain) or red (doubtful). Hover any of them for the score.
     <span id="vsub" hidden>Factual claims are then looked up and checked against sources: ✓ supported, ✗ not supported, ? unclear, – nothing found.</span></p>
    <div class="chips" id="chips"></div></div>
   <div id="log"></div>
  </section>
  <section class="view col" id="vverify" hidden>
   <h2 class="vtitle">Verify an answer from any AI</h2>
   <p class="vsub">Paste a question and the answer some model gave. A panel of readers and judges scores how likely it is to be
    wrong, points to the sentence or step it suspects, and shows what drove the verdict.</p>
   <div class="exs" id="exs"><span class="meta">Try:</span></div>
   <label class="field" for="jq">Question or prompt</label>
   <textarea class="box" id="jq" rows="2" placeholder="What did you ask?"></textarea>
   <label class="field" for="ja">The answer to check</label>
   <textarea class="box" id="ja" rows="7" placeholder="Paste the model's answer"></textarea>
   <label class="field" for="jdocs">Source documents <span style="opacity:.7">(optional, to check grounding; separate with a line containing ---)</span></label>
   <textarea class="box" id="jdocs" rows="3" placeholder="Paste the text the answer should be based on"></textarea>
   <div class="row"><button class="primary" id="jsend">Check answer</button><button class="ghost" id="jclear">Clear</button>
    <span id="jbusy" class="meta" style="margin:0"></span></div>
   <div id="jout"></div>
  </section>
 </div>
 <div class="dock" id="dock"><div class="inner">
  <div class="docs" id="docsbox" hidden><textarea id="docs" rows="5" placeholder="Paste source text the answer should rely on. Separate documents with a line containing ---"></textarea></div>
  <div class="composer" id="composer">
   <textarea id="msg" rows="1" placeholder="Ask anything"></textarea>
   <div class="tools"><button class="ibtn" id="docbtn" title="Add source documents"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 9.5l-5 5a3.2 3.2 0 01-4.5-4.5l5.5-5.5a2.1 2.1 0 013 3l-5.5 5.5a1 1 0 01-1.4-1.4l5-5"/></svg><span id="doclab">Documents</span></button>
    <span id="busy" class="meta" style="margin:0 0 0 4px"></span><span class="spacer"></span>
    <button class="send" id="send" aria-label="Send"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 16V4M5 9l5-5 5 5"/></svg></button></div>
  </div>
  <div class="hint" id="hint">Enter to send · Shift+Enter for a new line</div>
 </div></div>
</div>
<script>
const $=id=>document.getElementById(id),log=$('log'),msgs=[];
let conv=[],chatId=null,hasJudge=false;   // conv: {role:'user',text} / {role:'bot',reply,docs,check}
function clean(t){return String(t).split('[unsure:] ').join('').split('[unsure:]').join('')}
// light markdown on already-escaped text (spans from the checkers stay intact)
function md(h){return h.split('\n').map(l=>{
 if(/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(l))return '<hr class="mdhr">';
 l=l.replace(/^(\s*)#{1,4}\s+(.*)$/,'$1<b>$2</b>').replace(/^(\s*)[-*]\s+/,(m,sp)=>sp+'• ');
 l=l.replace(/\*\*([^*<>\n]+?)\*\*/g,'<b>$1</b>').replace(/\*\*/g,'');
 l=l.replace(/\$\$?([^$\n]*?[=\\+×^{}][^$\n]*?)\$\$?/g,'$1').replace(/\\boxed\{([^{}]*)\}/g,'<b>$1</b>').replace(/\\text\{([^{}]*)\}/g,'$1')
  .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g,'$1/$2').replace(/\\times/g,'×').replace(/\\cdot/g,'·').replace(/\\div/g,'÷').replace(/\\[()\[\]]/g,'');
 return l}).join('\n').replace(/\n{3,}/g,'\n\n').replace(/\n*<hr class="mdhr">\n*/g,'<hr class="mdhr">')}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
const LOGO='<svg class="mark"><use href="#logo"/></svg>';
function reasonBox(t){return '<details class="meta"><summary>Reasoning (not checked)</summary><div class="text" style="font-size:14px">'+esc(t)+'</div></details>'}
function marks(r){const t=r.text,cl=new Array(t.length).fill(-1),ds=new Array(t.length).fill(-1);let pos=0;
 r.claims.forEach((c,i)=>{if(!c.text)return;const k=(c.start!==undefined&&c.start>=pos)?c.start:t.indexOf(c.text,pos);if(k<0)return;
  for(let j=k;j<Math.min(t.length,k+c.text.length);j++)cl[j]=i;pos=k+c.text.length});
 (r.sentences||[]).forEach((x,i)=>{if(x.doubted)for(let j=Math.max(0,x.start);j<Math.min(t.length,x.end);j++)ds[j]=i});
 let html='',a=0;for(let j=1;j<=t.length;j++){if(j<t.length&&cl[j]===cl[a]&&ds[j]===ds[a])continue;
  const seg=esc(t.slice(a,j)),c=r.claims[cl[a]],x=(r.sentences||[])[ds[a]];let cls=[],tip=[];
  if(c){cls.push('c',c.verdict==='likely right'?'g':(c.verdict==='uncertain'?'m':'b'));tip.push('P(correct) '+c.score.toFixed(2)+
   (c.source?' | from '+c.source+(c.citation?': "'+c.citation.sentence+'"':''):''))}
  if(x){cls.push('ds');tip.push('doubtful sentence: '+(100*x.risk).toFixed(0)+'% risk')}
  html+=cls.length?'<span class="'+cls.join(' ')+'" title="'+esc(tip.join(' | '))+'">'+seg+'</span>':seg;a=j}
 return html.split('[unsure:] ').join('').split('[unsure:]').join('')}
function render(r){let html=md(marks(r));
 const bad=(r.sentences||[]).filter(x=>x.doubted).length;
 const list=r.claims.filter(c=>c.text).map(c=>'<div><b>'+esc(c.text)+'</b> — '+c.verdict+' ('+c.score.toFixed(2)+')'+
  (c.source?' · from '+esc(c.source)+(c.citation?': “'+esc(c.citation.sentence)+'”':''):'')+'</div>').join('');
 return '<div class="text">'+html+'</div>'+(list&&!hasJudge?'<div class="claims">'+list+'</div>':'')+
  '<div class="meta">'+(r.withheld?'Answer withheld: the model doubted it. ':'')+
  (bad?bad+' doubtful sentence'+(bad>1?'s':'')+' underlined (hover for the risk). ':'')+
  (r.retrieval&&r.retrieval.retrieved?'The model doubted its first answer ('+esc(r.retrieval.closed_book_text)+'), so it looked it up: '+
   esc((r.retrieval.passages||[]).map(p=>p.replace(/\): [\s\S]*$/,')').replace(' (Wikipedia)','').slice(0,70)).join('; '))+'. ':'')+
  '</div>'+(hasJudge?'':assess(r.assessment))+(r.grounding?grounding(r.grounding):r.ledger?ledger(r.ledger):verif(r.verification))+(r.reasoning?reasonBox(r.reasoning):'')}
function pillCls(v){return {'likely right':'good','nothing flagged':'good','no problem found':'good','uncertain':'mid','not grounded':'mid','likely wrong':'bad'}[v]||'na'}
function assess(a){if(!a)return '';return '<div class="meta"><span class="pill '+pillCls(a.verdict)+'">'+esc(a.verdict)+'</span> <span style="margin-left:6px">based on '+esc(a.basis)+'</span></div>'}
function grounding(g){if(!g||!g.sentences||!g.sentences.length)return '';const sym={'consistent':'✓','not grounded':'–','contradicted':'✗'};
 return '<div class="claims"><div class="meta" style="margin:0">Each sentence, read against the evidence</div>'+g.sentences.map(x=>'<div>'+sym[x.verdict]+' <b>'+esc(x.verdict)+'</b>: '+esc(x.text.slice(0,160))+'</div>').join('')+'</div>'}
function ledger(l){if(!l||!l.length)return '';const sym={'supported':'✓','contradicted':'✗','unverified':'?'};
 const shown=l.filter(u=>u.verdict!=='not a factual claim'),rest=l.length-shown.length;
 return '<div class="claims"><div class="meta" style="margin:0">Every statement, checked</div>'+shown.map(u=>'<div>'+sym[u.verdict]+' <b>'+esc(u.verdict)+'</b>: '+esc(u.claim.slice(0,160))+
  (u.evidence?' <span class="meta">('+esc(u.evidence.join(' '))+')</span>':'')+(u.sources&&u.sources.length?' <span class="meta">('+esc(u.sources.join('; '))+')</span>':'')+'</div>').join('')+
  (rest?'<div class="meta">'+rest+' sentence(s): not a statement of fact (opinion, advice, wording, fiction)</div>':'')+'</div>'}
function verif(v){v=(v||[]).filter(c=>c.verdict!=='no evidence'&&c.verdict!=='not in the sources');if(!v.length)return '';const sym={'supported':'✓','unsupported':'✗','unclear':'?','no evidence':'–','not in the sources':'–'};
 return '<div class="claims"><div class="meta" style="margin:0">Checked against sources</div>'+v.map(c=>'<div>'+sym[c.verdict]+' <b>'+esc(c.verdict)+'</b>: '+esc(c.claim)+
  (c.sources&&c.sources.length?' <span class="meta">('+esc(c.sources.join('; '))+')</span>':'')+'</div>').join('')+'</div>'}
const scroll=$('scroll');function bottom(){scroll.scrollTop=scroll.scrollHeight}
function addUser(t){$('empty').hidden=true;const d=document.createElement('div');d.className='msg user';d.innerHTML='<div class="bubble">'+esc(t)+'</div>';log.appendChild(d);bottom()}
function addBot(){const d=document.createElement('div');d.className='msg bot';d.innerHTML=LOGO+'<div class="body"><div class="text"></div><div class="thinking"><span class="pulse"></span><span>Thinking…</span></div></div>';log.appendChild(d);bottom();return d.querySelector('.body')}
// views, sidebar, settings
function show(v){$('vchat').hidden=v!=='chat';$('vverify').hidden=v!=='verify';$('navchat').classList.toggle('on',v==='chat');$('jp').classList.toggle('on',v==='verify');
 $('dock').hidden=v!=='chat';$('htitle').textContent=v==='chat'?'Chat':'Verify an answer';document.body.classList.remove('open');
}
$('navchat').onclick=()=>show('chat');$('jp').onclick=()=>show('verify');
$('newchat').onclick=()=>{msgs.length=0;conv=[];chatId=null;log.innerHTML='';$('empty').hidden=false;renderHist();show('chat');$('msg').focus()};
// recent chats, kept in this browser only
function store(){try{return JSON.parse(localStorage.getItem('verichat.chats')||'[]')}catch(_){return []}}
function saveChat(){if(!conv.length)return;const all=store().filter(c=>c.id!==chatId);chatId=chatId||String(Date.now());
 all.unshift({id:chatId,title:conv[0].text.slice(0,60),conv});try{localStorage.setItem('verichat.chats',JSON.stringify(all.slice(0,20)))}catch(_){}renderHist()}
function renderHist(){const all=store();$('histh').hidden=!all.length;
 $('hist').innerHTML=all.map(c=>'<button class="nav hitem'+(c.id===chatId?' on':'')+'" data-id="'+esc(c.id)+'" title="'+esc(c.title)+'">'+esc(c.title)+'</button>').join('')}
$('hist').onclick=e=>{const b=e.target.closest('[data-id]');if(!b)return;const c=store().find(x=>x.id===b.dataset.id);if(!c)return;
 chatId=c.id;conv=c.conv;msgs.length=0;log.innerHTML='';$('empty').hidden=!!conv.length;
 conv.forEach((t,i)=>{if(t.role==='user'){msgs.push({role:'user',content:t.text});addUser(t.text)}
  else{msgs.push({role:'assistant',content:t.reply.text});const d=document.createElement('div');d.className='msg bot';
   d.innerHTML=LOGO+'<div class="body">'+render(t.reply)+acts(i)+'</div>';log.appendChild(d);
   if(t.check)d.querySelector('.vres').innerHTML=renderJudge(t.check,clean(t.reply.text),true)}});
 renderHist();show('chat');bottom()};
function acts(i){return '<div class="acts"><button class="act" data-act="copy" data-i="'+i+'">Copy</button>'+
 (hasJudge?'<button class="act" data-act="verify" data-i="'+i+'">Verify this reply</button>':'')+'</div><div class="vres"></div>'}
log.onclick=async e=>{const b=e.target.closest('[data-act]');if(!b)return;const i=+b.dataset.i,t=conv[i],a=clean(t.reply.text);
 if(b.dataset.act==='copy'){try{await navigator.clipboard.writeText(a);b.textContent='Copied'}catch(_){b.textContent='Copy failed'}
  setTimeout(()=>{b.textContent='Copy'},1500);return}
 const out=b.parentNode.nextElementSibling;b.disabled=true;
 out.innerHTML='<div class="thinking" style="margin-top:8px"><span class="pulse"></span>Verifying this answer with the reader and two judges…</div>';
 const ms=conv.slice(0,i).map(x=>x.role==='user'?{role:'user',content:x.text}:{role:'assistant',content:clean(x.reply.text)}).slice(-24);
 try{const r=await (await fetch('/judge',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({messages:ms,response:a,documents:t.docs||[]})})).json();
  out.innerHTML=r.error?'<div class="meta">'+esc('Error: '+r.error)+'</div>':renderJudge(r,a,true);if(!r.error){t.check=r;saveChat()}}
 catch(err){out.innerHTML='<div class="meta">'+esc('Error: '+err)+'</div>'}b.disabled=false};
renderHist();
$('menu').onclick=()=>document.body.classList.toggle('open');$('scrim').onclick=()=>document.body.classList.remove('open');
function docList(id){return $(id||'docs').value.split(/\n\s*---\s*\n/).map(s=>s.trim()).filter(Boolean)}
function docLabel(){const n=docList().length;$('doclab').textContent=n?n+' document'+(n>1?'s':''):'Documents';$('docbtn').classList.toggle('on',!$('docsbox').hidden||n>0)}
$('docbtn').onclick=()=>{$('docsbox').hidden=!$('docsbox').hidden;docLabel();if(!$('docsbox').hidden)$('docs').focus()};$('docs').oninput=docLabel;
const h=new Date().getHours();$('greet').textContent=h<5?'Burning the midnight oil?':h<12?'Good morning':h<18?'Good afternoon':'Good evening';
const CHIPS=['Who designed the Eiffel Tower, and when was it finished?','What is 17 × 24? Show your working.','Explain why the sky is blue in two sentences.','Which is larger, 9.11 or 9.9?'];
$('chips').innerHTML=CHIPS.map(c=>'<button class="chip">'+esc(c)+'</button>').join('');
[...$('chips').children].forEach((b,i)=>b.onclick=()=>{$('msg').value=CHIPS[i];grow();sendChat()});
const ta=$('msg');function grow(){ta.style.height='auto';ta.style.height=Math.min(ta.scrollHeight,240)+'px'}ta.oninput=grow;
ta.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();sendChat()}};
let healthy=false;setTimeout(()=>{if(!healthy)$('wake').hidden=false},3000);
function health(){fetch('/health').then(r=>{if(!r.ok)throw new Error(r.status);return r.json()}).then(h=>{healthy=true;$('wake').hidden=true;
 if(h.detector){$('jp').hidden=false;hasJudge=true}
 if(h.verify_claims)$('vsub').hidden=false;
 if(h.thinking_available){$('thinkl').hidden=false;$('think').checked=!!h.thinking}
 $('dot').classList.add('ok');$('stat').textContent=h.omni?'Online · verifier: '+h.omni.reader.split('/').pop()+' + '+h.omni.judges+' judges':'Online'})
 .catch(()=>{$('stat').textContent='Server starting up…';$('wake').hidden=false;setTimeout(health,10000)})}
health();
// the verifier panel
const EVN={judge_1:'Judge 1 (one-pass verdict)',judge_2:'Judge 2 (one-pass verdict)',reader_judgement:"Reader's own verdict",
 reader_probe:'Probe on the reply',weakest_sentence:'Weakest sentence or step',grounding_nli:'Not supported by the documents',
 rule_violations:'Rule violations'};
function renderOmni(r,a,compact){const us=(r.units||[]).slice().sort((x,y)=>x.start-y.start);const sus=r.suspect_unit;
 const top=us.map(u=>u.score).sort((x,y)=>y-x),cut=top.length>2?top[Math.floor(top.length*0.3)]:Infinity;
 let html='',pos=0;for(const u of us){if(u.start<pos)continue;html+=esc(a.slice(pos,u.start));
  const isS=sus&&u.start===sus.start&&u.end===sus.end,cls=isS?'c b':(u.score>=cut&&us.length>2?'c m':'');
  const t=esc(a.slice(u.start,u.end));html+=cls?'<span class="'+cls+'" title="'+esc((isS?'suspect: ':'')+'unit score '+u.score.toFixed(2))+'">'+t+'</span>':t;pos=u.end}
 html+=esc(a.slice(pos));
 const p=r.risk_percentile;
 const pct=p==null?'':'<span class="meta" style="margin:0">riskier than '+Math.round(100*p)+'% of typical replies</span>';
 const g=r.guarantee?'Calibrated: at most '+Math.round(100*r.guarantee.fpr_target)+'% false alarms at '+Math.round(100*r.guarantee.confidence)+'% confidence.':
  'General profile, no labels: a ranking against typical replies, not a guarantee.';
 const ents=Object.entries(r.evidence||{}),mx=Math.max(1,...ents.map(([,v])=>Math.abs(v)));
 const ev=ents.map(([k,v])=>'<div class="ev"><span class="lab">'+esc(EVN[k]||k)+'</span><span class="bar"><b style="left:'+(v>0?50:50-50*Math.abs(v)/mx)+'%;width:'+(50*Math.abs(v)/mx)+'%;background:'+(v>0?'var(--bad)':'var(--good)')+'"></b></span><span class="num">'+(v>0?'+':'')+v.toFixed(2)+'</span></div>').join('');
 const vi=(r.violations||[]).map(x=>'<div>✗ <b>'+esc(x.why)+'</b>: '+esc(String(x.claim).slice(0,160))+'</div>').join('');
 return '<div class="card"><div class="vhead"><span class="pill '+pillCls(r.verdict)+'">'+esc(r.verdict||'scored')+'</span>'+
  (p==null?'':'<span class="meter" title="risk percentile"><i style="left:calc('+(100*p).toFixed(1)+'% - 1px)"></i></span>')+pct+'</div>'+
  (compact?'':'<div class="text">'+md(html)+'</div>')+
  (sus?'<div class="meta">'+(compact?'Most suspect ':'Suspect ')+(us.length>3?'step':'sentence')+(compact?'':' in red, riskiest of the rest in amber')+': “'+esc(sus.text.slice(0,140))+'”</div>':'')+
  (ev?'<div class="claims"><div class="meta" style="margin:0 0 4px">What drove the verdict (red pushes toward wrong)</div>'+ev+'</div>':'')+
  (vi?'<div class="claims"><div class="meta" style="margin:0">Rule violations</div>'+vi+'</div>':'')+steps(r.steps)+
  '<div class="meta">'+g+' Score '+r.score.toFixed(2)+' · '+r.ms+' ms</div></div>'}
function steps(c){if(!c||!c.units||!c.units.length)return '';const key=c.by,J=key==='judges';
 const mx=Math.max(1,...c.units.map(u=>Math.abs(u[key])));
 const rows=c.units.map((u,k)=>'<div class="ev"><span class="lab">'+(k===c.first_wrong?'<span class="pill bad" style="padding:1px 8px;font-size:11.5px;margin-right:6px">turns wrong here</span>':'')+
  '<span class="meta" style="margin:0 6px 0 0">'+(k+1)+'</span>“'+esc(u.text.trim().slice(0,100))+(u.text.trim().length>100?'…':'')+'”</span><span class="bar"><b style="left:'+(u[key]>0?50:50-50*Math.min(1,Math.abs(u[key])/mx))+'%;width:'+
  (50*Math.min(1,Math.abs(u[key])/mx))+'%;background:'+(u[key]>0?'var(--bad)':'var(--good)')+'"></b></span><span class="num">'+(u[key]>0?'+':'')+u[key].toFixed(1)+'</span></div>').join('');
 return '<div class="claims"><div class="meta" style="margin:0 0 4px">Running verdict: '+(J?'the judges are':'the reader is')+
  ' asked “is everything so far correct?” after each sentence or step (red: leaning wrong). '+
  (c.first_wrong==null?'It never tips to wrong.':'It first tips to wrong at step '+(c.first_wrong+1)+'.')+'</div>'+rows+'</div>'}
function renderJudge(r,a,compact){if(r.verifier==='omni')return renderOmni(r,a,compact);let html='',pos=0;for(const s of r.spans){html+=esc(a.slice(pos,s.start))+'<span class="c b" title="'+
 esc('P(hallucinated) '+s.score.toFixed(2))+'">'+esc(a.slice(s.start,s.end))+'</span>';pos=s.end}html+=esc(a.slice(pos));
 return '<div class="card"><div class="vhead"><span class="pill '+(r.flagged?'bad':'good')+'">'+(r.flagged?'Likely contains a hallucination':
 'No hallucination detected')+'</span><span class="meta" style="margin:0">P = '+r.p_hallucinated.toFixed(2)+'</span></div><div class="text">'+html+'</div><div class="meta">Red: the spans the reader doubts most.</div></div>'}
const EX=[{l:'A wrong date',q:'When was the Eiffel Tower completed, and who designed it?',
  a:"The Eiffel Tower was completed in 1899. It was designed by Gustave Eiffel's company, with the main structure engineered by Maurice Koechlin and Émile Nouguier, for the Exposition Universelle in Paris."},
 {l:'A correct answer',q:'At what temperature does water boil at sea level?',
  a:'At sea level (standard atmospheric pressure, 1 atm), pure water boils at 100 °C, which is 212 °F. At higher altitudes the air pressure is lower, so water boils at a lower temperature.'},
 {l:'An arithmetic slip',q:'Pens cost $3 each. Ana buys 7 pens and pays with a $50 note. How much change does she get?',
  a:'Step 1: The pens cost 7 × 3 = 24 dollars.\nStep 2: Her change is 50 − 24 = 26 dollars.\nAnswer: Ana gets $26 in change.'},
 {l:'Not in the document',q:'Is the Riverside Clinic open on Saturdays?',
  a:'Yes. The Riverside Clinic is open on Saturdays from 9am to 1pm, and parking is free for patients.',
  d:'The Riverside Clinic is open Monday to Friday, 8am to 6pm. It is closed on weekends and public holidays. Parking is free for patients with an appointment.'}];
EX.forEach(x=>{const b=document.createElement('button');b.className='chip';b.textContent=x.l;
 b.onclick=()=>{$('jq').value=x.q;$('ja').value=x.a;$('jdocs').value=x.d||'';$('jsend').click()};$('exs').appendChild(b)});
$('jclear').onclick=()=>{['jq','ja','jdocs'].forEach(k=>$(k).value='');$('jout').innerHTML='';$('jq').focus()};
$('jsend').onclick=async()=>{const q=$('jq').value.trim(),a=$('ja').value.trim();
 if(!q||!a)return;const docs=docList('jdocs');
 const b=$('jsend');b.disabled=true;$('jbusy').innerHTML='<span class="thinking"><span class="pulse"></span>Reading the answer…</span>';
 try{const res=await fetch('/judge',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({messages:[{role:'user',content:q}],response:a,documents:docs})});const r=await res.json();
  $('jout').innerHTML=r.error?'<div class="card">'+esc('Error: '+r.error)+'</div>':renderJudge(r,a)}
 catch(e){$('jout').innerHTML='<div class="card">'+esc('Error: '+e)+'</div>'}b.disabled=false;$('jbusy').textContent='';$('jout').scrollIntoView({behavior:'smooth',block:'nearest'})};
// chat
$('send').onclick=()=>sendChat();
async function sendChat(){const m=ta.value.trim();if(!m||$('send').disabled)return;
 const docs=docList();
 msgs.push({role:'user',content:m});addUser(m);ta.value='';grow();
 const b=$('send');b.disabled=true;
 const card=addBot();const live=card.firstChild,rlive=card.lastChild;let shown='',reasoned='';
 let ok=false;
 const body=JSON.stringify({messages:msgs.slice(-24),documents:docs,mode:$('mode').value,stream:true,thinking:$('think').checked,max_new_tokens:1024});
 // The server sends a heartbeat every 10 s, so 45 s of silence means the request is stuck: abort it.
 // A request that drops or sticks before any text arrived is retried once; after that, say what happened.
 async function attempt(){const ac=new AbortController();let wd=null,r=null,err=null,buf='';
  const arm=()=>{clearTimeout(wd);wd=setTimeout(()=>ac.abort(),45000)};arm();
  try{const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body,signal:ac.signal});
   if(!res.ok){const j=await res.json().catch(()=>({}));return {final:'Error: '+(j.error||res.status)}}
   const rd=res.body.getReader(),dec=new TextDecoder();
   while(true){arm();const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
    let k;while((k=buf.indexOf('\n\n'))>=0){const ev=buf.slice(0,k);buf=buf.slice(k+2);
     if(!ev.startsWith('data: '))continue;const m=JSON.parse(ev.slice(6));
     if(m.reasoning){reasoned+=m.reasoning;rlive.lastChild.textContent='Thinking: '+reasoned.slice(-240);bottom()}
     if(m.delta){rlive.hidden=true;shown+=m.delta;live.innerHTML=md(esc(clean(shown)));bottom()}
     if(m.done){if(m.error)return {final:'Error: '+m.error};r=m.reply}}}}
  catch(e){err=e}finally{clearTimeout(wd)}
  return r?{reply:r}:{dropped:true,err}}
 let out=await attempt();
 if(out.dropped&&!shown){rlive.lastChild.textContent='Reconnecting…';await new Promise(z=>setTimeout(z,1500));out=await attempt()}
 if(out.reply){msgs.push({role:'assistant',content:out.reply.text});conv.push({role:'user',text:m},{role:'bot',reply:out.reply,docs});
  card.innerHTML=render(out.reply)+acts(conv.length-1);ok=true;saveChat();
  if(hasJudge)card.querySelector('[data-act="verify"]').click()}         // every reply gets the verifier's verdict
 else if(out.final)card.innerHTML='<div class="meta">'+esc(out.final)+'</div>';
 else{rlive.hidden=false;rlive.innerHTML=esc('The reply did not arrive ('+(out.err?out.err.name:'the connection closed')+'). Please send it again.')}
 if(!ok)msgs.pop();b.disabled=false;bottom();ta.focus()}
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
    n = req.get("max_new_tokens")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or not 16 <= n <= MAX_NEW_TOKENS):
        raise ValueError(f"max_new_tokens: an integer from 16 to {MAX_NEW_TOKENS}")
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

            def ready(j, r, batch=batch):                        # a handler may hand results over early
                slot = batch[j][2]
                if not slot["done"].is_set():
                    slot["result"] = r
                    slot["done"].set()
            handler = self.handlers[kind]
            try:
                results = handler([item for _, item, _ in batch], ready) if getattr(handler, "early", False) \
                    else handler([item for _, item, _ in batch])
                for (_, _, slot), r in zip(batch, results):
                    slot.setdefault("result", r)
            except Exception as e:                               # one bad request must not fail the others:
                traceback.print_exc()                            # retry the unfinished ones one by one
                for _, item, slot in batch:
                    if slot["done"].is_set():
                        continue
                    if len(batch) == 1:
                        slot["error"] = e
                        continue
                    try:
                        slot["result"] = handler([item])[0]
                    except Exception as e1:
                        traceback.print_exc()
                        slot["error"] = e1
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
                api_key=None, rate_per_minute=20, trust_proxy=False, max_batch=8, max_concurrency=16,
                log_requests=False, omni=None):
    """omni: (OmniVerifier, profile) to answer /judge with the omni verifier (vt/omni.py) instead of `detector`."""
    """api_key: if set, POSTs need "Authorization: Bearer <key>". trust_proxy: take the client from
    X-Forwarded-For (only behind a proxy you control). Concurrent requests are batched (max_batch)."""
    def chat(items, ready=None):                                 # items may carry a "_stream" callback
        reqs = [{k: v for k, v in it.items() if k != "_stream"} for it in items]
        sinks = [it.get("_stream") for it in items]
        if not any(sinks):
            return bot.reply_batch(reqs, on_ready=ready)
        return bot.reply_batch(reqs, on_text=lambda i, d: sinks[i] and sinks[i]({"delta": d}),
                               on_reasoning=lambda i, d: sinks[i] and sinks[i]({"reasoning": d}), on_ready=ready)
    chat.early = True                                            # replies go out as soon as each is final
    handlers = {"chat": chat}
    if omni is not None:
        ov, oprof = omni

        def omni_judge(items):
            out = []
            for it in items:
                t0 = time.perf_counter()
                try:
                    r = ov.verify(it["response"], messages=it["messages"], documents=it.get("documents"), prof=oprof)
                except Exception as e:                       # one bad item must not fail the batch
                    traceback.print_exc()
                    r = {"error": f"verification failed: {type(e).__name__}"}
                out.append(dict(r, verifier="omni", ms=round(1000 * (time.perf_counter() - t0))))
            return out
        handlers["judge"] = omni_judge
    elif detector is not None:
        handlers["judge"] = lambda items: detector.judge_batch(items)
    metrics = Metrics()
    batcher = Batcher(handlers, max_batch=max_batch, metrics=metrics)
    limiter = RateLimiter(rate_per_minute)
    arrivals = itertools.count(1)

    class Handler(HardenedHandler):
        timeout_s = 15.0
        deadline_s = deadline + 30.0                             # after the model's deadline, so its 503 gets out
        log_errors = log_requests

        def on_response(self, code):
            if log_requests and self.command == "POST":           # pairs with "recv": a recv with no sent line
                print(f"sent {self.path} {code} #{getattr(self, '_rid', '?')} "  # is stuck in the model or the write
                      f"{time.perf_counter() - self._t0:.1f}s", flush=True)
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
                                            "risk_threshold": bot.gen.risk_threshold if bot.gen.risk_probe is not None else None,
                                            "detector": detector is not None or omni is not None,
                                            "omni": None if omni is None else {
                                                "reader": omni[0].meta["reader"], "judges": len(omni[0].judges),
                                                "nli": omni[0].nli is not None, "profile": omni[1]["name"],
                                                "calibrated": omni[1].get("threshold") is not None},
                                            "thinking": bot.gen.thinking,
                                            "thinking_available": bot.gen.think_open is not None,
                                            "retrieval": type(bot.retriever).__name__ if bot.retriever is not None else None,
                                            "retrieval_errors": dict(getattr(bot.retriever, "errors", None) or {}),
                                            "verify_claims": getattr(bot, "verifier", None) is not None})
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            self._t0 = time.perf_counter()
            if log_requests:                                     # arrival, so lost requests can be told apart
                self._rid = next(arrivals)
                print(f"recv {self.path} #{self._rid}", flush=True)
            if self.path not in ("/chat", "/judge") or (self.path == "/judge" and detector is None and omni is None):
                return self.send_json(404, {"error": "not found"})
            if api_key and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {api_key}"):
                return self.send_json(401, {"error": "missing or wrong API key"})
            fwd = self.headers.get("X-Forwarded-For", "") if trust_proxy else ""
            client = fwd.split(",")[0].strip() or self.client_address[0]
            if not limiter.allow(client):
                return self.send_json(429, {"error": f"rate limit: {rate_per_minute} requests per minute"})
            req, err = self.read_json_object(max_body)
            if log_requests:                                     # recv, body, sent: where a lost one stopped
                print(f"body #{self._rid} {getattr(self, 'body_seconds', 0):.2f}s"
                      + (" (expect 100-continue)" if self.headers.get("Expect") else ""), flush=True)
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
                except Exception:                                # the model failed: a clean error, not a dropped line
                    traceback.print_exc()
                    return self.send_json(500, {"error": "internal error while judging"})
                rs = [r if "error" in r or r.get("verifier") == "omni" else
                      {k: r[k] for k in ("p_hallucinated", "flagged", "spans")} for r in rs]
                return self.send_json(200, {"items": rs} if batch is not None else rs[0])
            try:
                msgs, docs, mode = validate(req)
            except ValueError as e:
                return self.send_json(400, {"error": str(e)[:200]})
            item = {"messages": msgs, "documents": docs, "mode": mode, "thinking": req.get("thinking")}
            if req.get("max_new_tokens") is not None:
                item["max_new_tokens"] = req["max_new_tokens"]
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
            except Exception:                                    # the model failed: a clean error, not a dropped line
                traceback.print_exc()
                return self.send_json(500, {"error": "internal error while answering"})
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
                    try:
                        d = q.get(timeout=10)
                    except queue.Empty:                          # a heartbeat: the page can tell a slow reply
                        self.wfile.write(b": ping\n\n")         # from a stuck one, and proxies keep the line open
                        self.wfile.flush()
                        continue
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
    srv = build(argv)
    print(f"VeriChat on http://{srv.server_address[0]}:{srv.server_port}", flush=True)
    srv.serve_forever()


def build(argv=None, host=None, port=None):
    """The configured server, loaded and ready but not serving: serve_forever() it, or hand it to
    vt.asgi.asgi_from to serve in-process."""
    ap = argparse.ArgumentParser(prog="vt verichat")
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--revision", help="the model's Hugging Face commit or tag (pin it in production)")
    ap.add_argument("--threshold", type=float, help="claims scoring below it are doubted (default: the head's "
                                                     "<head>.serving.json, else 0.45)")
    ap.add_argument("--risk-threshold", type=float, help="sentence risk above which a sentence is doubted "
                                                         "(default: the risk head's own, chosen for F1)")
    ap.add_argument("--corpus", help="a folder of .txt/.md files: look questions up there when the model doubts itself")
    ap.add_argument("--wikipedia", action="store_true", help="look doubted questions up on Wikipedia (live, no key)")
    ap.add_argument("--rerank", help="a sentence-transformers cross-encoder that reorders Wikipedia passages (with "
                                     "it, the top 5 pages are searched), e.g. cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--web-search", choices=["brave", "tavily"], help="also search the web (the API key in the "
                                                                    "VERICHAT_SEARCH_KEY environment variable)")
    ap.add_argument("--grounding", help="a grounding probe (experiments/grounding_probe.py --export): every sentence of "
                    "every reply read against its evidence from a model's own neurons, no judge (vt/grounding.py)")
    ap.add_argument("--grounding-model", help="the model whose neurons the probe reads, if not the chat model (a "
                    "reader: it reads each reply with its evidence, nothing is asked of it)")
    ap.add_argument("--ledger", action="store_true", help="a verdict on every statement of every reply: each is "
                    "judged by the chat model against the conversation, exact checks of its arithmetic and code, and a "
                    "lookup for what those leave unsettled (vt/ledger.py; replaces --verify-claims)")
    ap.add_argument("--verify-claims", action="store_true", help="check each reply's claims against Wikipedia, "
                                                                "judged by the chat model (needs --wikipedia)")
    ap.add_argument("--wikipedia-leads", action="store_true", help="use only each page's lead section, not the best "
                                                                 "passages of its whole text")
    ap.add_argument("--retrieve-below", type=float, help="look up when the reply's doubt score is below this "
                                                        "(default: the doubt threshold)")
    ap.add_argument("--detector", help="a kaggle/detector head (*_detector.pt) to enable /judge")
    ap.add_argument("--detector-model", help="its reader model (default: the head's own)")
    ap.add_argument("--omni-head", help="an omni verifier head (heads/omni_qwen3_*.npz): /judge then uses the omni "
                    "verifier (vt/omni.py) — reader probes, optional one-pass judges, NLI, written checkers")
    ap.add_argument("--omni-judges", default="", help="comma-separated judge models read in one pass, e.g. "
                    "Qwen/Qwen3-14B,microsoft/phi-4 (4-bit on a GPU)")
    ap.add_argument("--omni-nli", action="store_true", help="add the NLI cross-encoders for replies with documents")
    ap.add_argument("--omni-causal", action="store_true", help="add the running verdict to every /judge answer: "
                    "\"is everything so far correct?\" after each sentence / step, from the reader (cached) and the judges")
    ap.add_argument("--omni-arithmetic", action="store_true", help="also flag arithmetic that does not compute "
                    "(vt/mathcheck, rule-based; off by default: the served verifier uses learned experts only)")
    ap.add_argument("--omni-checkers", default="", help="comma-separated written checkers (vt/checkers), e.g. "
                    "business_record")
    ap.add_argument("--omni-profile", help="a domain profile (vt-omni onboard) to use; without it a general, "
                    "zero-label profile is built at startup from --omni-sample and cached next to the head")
    ap.add_argument("--omni-cache-dir", help="where to cache the general profile (default: beside the head)")
    ap.add_argument("--omni-sample", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                    "omni_profile_sample.jsonl"), help="unlabelled replies for the general profile")
    ap.add_argument("--moderation", action="store_true", help="screen messages and replies (vt/moderation.py)")
    ap.add_argument("--rate-per-minute", type=int, default=20, help="POSTs per client per minute (0: unlimited)")
    ap.add_argument("--trust-proxy", action="store_true", help="client address from X-Forwarded-For")
    ap.add_argument("--thinking", action="store_true", help="reasoning models think first unless a request says not to")
    ap.add_argument("--think-budget", type=int, default=1024, help="most tokens of thinking per reply")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (0: greedy); Qwen3's advice")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=1.5, help="against loops (1.5: 7x fewer repeated 4-grams, same accuracy)")
    ap.add_argument("--deadline", type=float, default=120.0, help="seconds a request may wait and run")
    ap.add_argument("--log-requests", action="store_true", help="log each POST as it arrives")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    from .verichat import VeriChat
    bot = VeriChat.load(a.model, a.head, a.threshold, revision=a.revision)
    bot.gen.thinking, bot.gen.think_budget = a.thinking, a.think_budget
    if a.risk_threshold is not None:
        bot.gen.risk_threshold = a.risk_threshold
    from .verichat import apply_sampling
    apply_sampling(bot, a.temperature, a.top_p, a.top_k, a.presence_penalty)
    if a.moderation:
        from .moderation import Moderator
        bot.moderator = Moderator.load()
    if a.corpus or a.wikipedia or a.web_search:
        from .retrieval import Corpus, Wikipedia
        rerank = None
        if a.rerank and not a.corpus:                   # 0.71 against 0.59 answer recall (results/retrieval_recall_v4)
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(a.rerank, max_length=256)

            def rerank(q, ps):
                return ce.predict([(q, p) for p in ps], batch_size=32, show_progress_bar=False)
        bot.retriever = Corpus.from_dir(a.corpus) if a.corpus else \
            Wikipedia(full_text=not a.wikipedia_leads, pages=5 if rerank else 3, rerank=rerank) if a.wikipedia else None
        if a.web_search:
            from .websearch import Union, WebSearch
            key = os.environ.get("VERICHAT_SEARCH_KEY")
            if not key:
                raise SystemExit("--web-search needs the API key in VERICHAT_SEARCH_KEY")
            web = WebSearch(a.web_search, key, pages=5 if rerank else 3, rerank=rerank)
            bot.retriever = Union(bot.retriever, web, k=3, rerank=rerank) if bot.retriever is not None else web
        if a.verify_claims:
            from .claim_verify import ClaimVerifier
            bot.verifier = ClaimVerifier(bot.model, bot.tok, bot.retriever)
        bot.retrieve_below = a.retrieve_below if a.retrieve_below is not None else bot.threshold
    if a.grounding:
        from .grounding import Grounding
        if a.grounding_model and a.grounding_model != a.model:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            rtok = AutoTokenizer.from_pretrained(a.grounding_model)
            reader = AutoModelForCausalLM.from_pretrained(
                a.grounding_model, torch_dtype=torch.bfloat16, device_map="auto" if torch.cuda.is_available() else None).eval()
        else:
            reader, rtok = bot.model, bot.tok
        bot.grounding = Grounding(reader, rtok, a.grounding, retriever=bot.retriever)
    if a.ledger:
        from .ledger import Ledger
        bot.ledger = Ledger(bot.model, bot.tok, bot.retriever)
    det = None
    if a.detector:
        import torch
        from .detector import Detector
        name = a.detector_model or torch.load(a.detector, map_location="cpu", weights_only=True)["model"]
        det = Detector(bot.model, bot.tok, torch.load(a.detector, map_location="cpu", weights_only=True)) \
            if name == a.model else Detector.load(name, a.detector)
    omni = load_omni(a) if a.omni_head else None
    return make_server(bot, host or a.host, a.port if port is None else port, deadline=a.deadline, detector=det,
                       log_requests=a.log_requests, api_key=os.environ.get("VERICHAT_API_KEY") or None,
                       rate_per_minute=a.rate_per_minute, trust_proxy=a.trust_proxy, omni=omni)


def load_omni(a):
    """the omni verifier for /judge and its profile: a given domain profile, or a general zero-label one built from
    the unlabelled sample (cached beside the head, keyed by the head and the judges)."""
    import hashlib
    import importlib
    import json as _json

    from .omni import JudgeScorer, NLIScorer, OmniVerifier
    judges = [JudgeScorer(m.strip()) for m in a.omni_judges.split(",") if m.strip()]
    checkers = [importlib.import_module(f"vt.checkers.{c.strip()}").check for c in a.omni_checkers.split(",") if c.strip()]
    v = OmniVerifier.load(a.omni_head, nli=NLIScorer() if a.omni_nli else None, checkers=checkers, judges=judges,
                          arithmetic=a.omni_arithmetic)
    v.attach_reader()
    v.causal = a.omni_causal
    if a.omni_profile:
        return v, OmniVerifier.load_profile(a.omni_profile)
    key = hashlib.sha256(f"{a.omni_head}|{a.omni_judges}|{a.omni_nli}|{a.omni_checkers}|{a.omni_arithmetic}".encode()
                         ).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(a.omni_head))[0] + f".general_{key}.json"
    cache = os.path.join(a.omni_cache_dir, stem) if a.omni_cache_dir else \
        os.path.join(os.path.dirname(os.path.abspath(a.omni_head)), stem)
    if a.omni_cache_dir:
        os.makedirs(a.omni_cache_dir, exist_ok=True)
    try:
        return v, OmniVerifier.load_profile(cache)
    except (OSError, ValueError):
        pass
    items = []
    for line in open(a.omni_sample):
        x = _json.loads(line)
        items.append({"messages": x["messages"], "response": x["reply"], "units": [tuple(u) for u in x["units"]],
                      "doc": "\n\n".join(x.get("documents") or []), "question": x["messages"][-1]["content"]})
    print(f"building the general omni profile from {len(items)} unlabelled replies ...", flush=True)
    prof = v.general_profile(items)
    try:
        OmniVerifier.save_profile(prof, cache)
    except OSError:                                          # a read-only image: rebuild next time
        pass
    return v, prof


if __name__ == "__main__":
    main()
