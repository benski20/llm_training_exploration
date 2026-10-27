"""Serve a glass-box chat model: every reply comes with the model's own verification signals.

    vt chat-export --ckpt results/glassbox_chat_v2/glassbox.glassbox.pt \\
        --tokenizer results/glassbox_chat_v2/tokenizer.json --out models/chat
    vt chat models/chat "Who does Mesuntu work for?"
    vt chat-serve models/chat --port 8001
    curl -X POST localhost:8001/chat -d '{"message": "Where was Gokuko born?"}'

A reply reports, for every generated token, from the same forward pass that
produced it:
  confidence    sigmoid(k): the model's self-knowledge variable, trained to
                predict whether its top prediction is right
  from_prompt / from_output / from_memory
                the evidence stream: the share of the token's content mixed in
                from the user's turn, from the reply so far, and from memory reads
plus low-confidence spans and whether the reply abstains. These are the model's
own signals, calibrated only as well as its training made them; they are not
proofs.

The model package is a directory with weights.pt, config.json, tokenizer.json
and manifest.json (sha256 of each file). Loading reads each file once, checks
those exact bytes against the manifest and refuses a mismatch.
"""
import hashlib
import io
import json
import os
import threading
import time

import torch
import torch.nn.functional as F

from .glassbox import GBConfig, GlassBoxLM
from .serve import HardenedHandler, bounded_server

FILES = ("weights.pt", "config.json", "tokenizer.json")
ABSTAIN_MARKERS = ("don't know", "not sure", "no information")
MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_TURNS = 8
MAX_NEW_TOKENS = 128


class PackageError(RuntimeError):
    pass


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def export_package(ckpt, tokenizer, out, config):
    """Write a servable package from a training checkpoint ({"model": state_dict, ...} or
    a bare state_dict) and its tokenizer file."""
    os.makedirs(out, exist_ok=True)
    st = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = st.get("model", st)
    m = GlassBoxLM(GBConfig(**config))
    missing, unexpected = m.load_state_dict(state, strict=False)
    if unexpected or any(not k.endswith(("lscale", "lbias", "usage")) for k in missing):
        raise PackageError(f"checkpoint does not match the config: missing {missing}, unexpected {unexpected}")
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    blobs = {"weights.pt": buf.getvalue(), "config.json": json.dumps(config, indent=1).encode(),
             "tokenizer.json": open(tokenizer, "rb").read()}
    for name, b in blobs.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(b)
    manifest = {"format": "vt-glassbox-chat/1", "files": {n: _sha(b) for n, b in blobs.items()},
                "package_sha256": _sha(b"".join(_sha(blobs[n]).encode() for n in FILES))}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def load_package(path, max_bytes=1 << 30):
    """Returns (model, tokenizer, manifest). Each file is read once and the bytes
    that are used are the bytes that were hashed."""
    from tokenizers import Tokenizer
    try:
        manifest = json.load(open(os.path.join(path, "manifest.json")))
        want = manifest["files"]
    except (OSError, ValueError, KeyError) as e:
        raise PackageError(f"unreadable manifest: {e}") from e
    blobs = {}
    for name in FILES:
        fp = os.path.join(path, name)
        if os.path.getsize(fp) > max_bytes:
            raise PackageError(f"{name} is larger than {max_bytes} bytes")
        with open(fp, "rb") as f:
            b = f.read(max_bytes + 1)
        if _sha(b) != want.get(name):
            raise PackageError(f"{name} does not match the manifest")
        blobs[name] = b
    config = json.loads(blobs["config.json"])
    model = GlassBoxLM(GBConfig(**config))
    model.load_state_dict(torch.load(io.BytesIO(blobs["weights.pt"]), map_location="cpu", weights_only=True))
    model.eval()
    tok = Tokenizer.from_str(blobs["tokenizer.json"].decode("utf-8"))
    return model, tok, manifest


def render_prompt(message, history=()):
    turns = "".join(f"User: {u}\nAssistant: {a}\n" for u, a in history)
    return f"{turns}User: {message}\nAssistant:"


@torch.no_grad()
def reply(model, tok, message, history=(), max_new_tokens=64, threshold=0.5, temperature=0.0, seed=0):
    """Generate one assistant turn and its per-token report."""
    block = model.config.block_size
    ids = tok.encode(render_prompt(message, history)).ids[-(block - max_new_tokens):]
    P = len(ids)
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(max_new_tokens):
        logits, _, _, _ = model(torch.tensor([ids + out]), evidence=False)
        lg = logits[0, -1]
        if i == 0:
            lg[0] = -float("inf")                    # answer with at least one token
        nxt = int(lg.argmax()) if temperature <= 0 else int(torch.multinomial(F.softmax(lg / temperature, -1), 1, generator=g))
        if nxt == 0:
            break
        out.append(nxt)
        if "\n" in tok.decode(out):
            out = out[:-1] if tok.decode(out[-1:]).strip() == "" else out
            break
    if not out:
        return {"text": "", "tokens": [], "low_confidence_spans": [], "abstained": False}
    x = torch.tensor([ids + out])
    _, k, e, _ = model(x, evidence=True)
    T = x.shape[1]
    tokens = []
    for j, t in enumerate(out):
        pos = P - 1 + j                              # the position whose prediction emitted token j
        ev = e[0, pos]
        tokens.append({"token": tok.decode([t]), "confidence": round(float(torch.sigmoid(k[0, pos])), 4),
                       "from_prompt": round(float(ev[:P].sum()), 4),
                       "from_output": round(float(ev[P:T].sum()), 4),
                       "from_memory": round(float(ev[T:].sum()), 4)})
    text = tok.decode(out).split("\n")[0].strip()
    spans, cur = [], None
    for j, r in enumerate(tokens):
        if r["confidence"] < threshold:
            cur = cur or {"start_token": j, "text": ""}
            cur["text"] += r["token"]
            cur["end_token"] = j
        elif cur:
            spans.append(cur); cur = None
    if cur:
        spans.append(cur)
    return {"text": text, "tokens": tokens, "low_confidence_spans": spans,
            "abstained": any(m in text.lower() for m in ABSTAIN_MARKERS),
            "min_confidence": min(r["confidence"] for r in tokens)}


def validate_request(req):
    msg = req.get("message")
    if not isinstance(msg, str) or not 0 < len(msg) <= MAX_MESSAGE_CHARS:
        raise ValueError(f"'message' must be a string of 1..{MAX_MESSAGE_CHARS} characters")
    hist = req.get("history", [])
    if not isinstance(hist, list) or len(hist) > MAX_HISTORY_TURNS:
        raise ValueError(f"'history' must be a list of at most {MAX_HISTORY_TURNS} [user, assistant] pairs")
    pairs = []
    for h in hist:
        if not (isinstance(h, list) and len(h) == 2 and all(isinstance(t, str) and len(t) <= MAX_MESSAGE_CHARS
                                                            for t in h)):
            raise ValueError("each history entry must be [user, assistant] strings")
        pairs.append((h[0], h[1]))
    n = req.get("max_new_tokens", 64)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_NEW_TOKENS:
        raise ValueError(f"'max_new_tokens' must be an integer in 1..{MAX_NEW_TOKENS}")
    th = req.get("threshold", 0.5)
    if not isinstance(th, (int, float)) or isinstance(th, bool) or not 0 <= th <= 1:
        raise ValueError("'threshold' must be a number in [0, 1]")
    return msg, pairs, n, float(th)


def make_chat_server(model, tok, manifest, host="127.0.0.1", port=8001, max_body=1 << 16, timeout=10.0,
                     deadline=60.0, max_concurrency=4):
    stats = {"requests": 0, "replies": 0, "errors_4xx": 0, "abstained": 0, "latency_s_total": 0.0}
    lock = threading.Lock()
    gen_lock = threading.Lock()                       # one generation at a time on CPU
    info = {"status": "ok", "format": manifest.get("format"), "package_sha256": manifest.get("package_sha256"),
            "signals": ["confidence", "from_prompt", "from_output", "from_memory"]}

    class Handler(HardenedHandler):
        timeout_s, deadline_s = timeout, deadline

        def on_response(self, code):
            with lock:
                stats["requests"] += 1
                if 400 <= code < 500:
                    stats["errors_4xx"] += 1

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, info)
            if self.path == "/metrics":
                with lock:
                    return self.send_json(200, dict(stats))
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/chat":
                return self.send_json(404, {"error": "not found"})
            t0 = time.perf_counter()
            req, err = self.read_json_object(max_body)
            if err:
                return self.send_json(err[0], {"error": err[1]})
            try:
                msg, hist, n, th = validate_request(req)
            except ValueError as e:
                return self.send_json(400, {"error": str(e)[:200]})
            with gen_lock:
                r = reply(model, tok, msg, hist, max_new_tokens=n, threshold=th)
            self.send_json(200, {"package_sha256": manifest.get("package_sha256"), **r})
            with lock:
                stats["replies"] += 1
                stats["abstained"] += int(r["abstained"])
                stats["latency_s_total"] += time.perf_counter() - t0

    return bounded_server(host, port, Handler, max_concurrency)
