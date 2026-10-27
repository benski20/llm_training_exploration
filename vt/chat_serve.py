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
from .glassbox2 import G2Config, Glassbox2
from .glassbox3 import G3Config, Glassbox3
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


def export_glassbox2_package(model, config, tokenizer, out):
    """Write a servable package for a Glassbox-2 or Glassbox-3 model (as the Kaggle student
    trainer does): vt-glassbox2-chat/1 or vt-glassbox3-chat/1."""
    os.makedirs(out, exist_ok=True)
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    blobs = {"weights.pt": buf.getvalue(), "config.json": json.dumps(config, indent=1).encode(),
             "tokenizer.json": open(tokenizer, "rb").read()}
    for name, b in blobs.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(b)
    fmt = "vt-glassbox3-chat/1" if isinstance(model, Glassbox3) else "vt-glassbox2-chat/1"
    manifest = {"format": fmt, "files": {n: _sha(b) for n, b in blobs.items()},
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
    builders = {"vt-glassbox-chat/1": lambda c: GlassBoxLM(GBConfig(**c)),
                "vt-glassbox2-chat/1": lambda c: Glassbox2(G2Config(**c)),
                "vt-glassbox3-chat/1": lambda c: Glassbox3(G3Config(**c))}
    if manifest.get("format") not in builders:
        raise PackageError(f"unknown package format {manifest.get('format')!r}")
    model = builders[manifest["format"]](config)
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
    return _report(tok.decode(out).split("\n")[0].strip(), tokens, threshold)


class GlassboxResponder:
    """A glass-box package: confidence from k, plus the evidence stream."""

    def __init__(self, model, tok, manifest):
        self.model, self.tok, self.manifest = model, tok, manifest
        self.signals = ["confidence", "from_prompt", "from_output", "from_memory"]

    def reply(self, message, history=(), max_new_tokens=64, threshold=0.5):
        return reply(self.model, self.tok, message, history, max_new_tokens, threshold)


@torch.no_grad()
def reply_g2(model, tok, message, history=(), max_new_tokens=64, threshold=0.5):
    """Glassbox-2/3 turn. Per token: confidence sigmoid(k), whether it is certified memory-free
    (exact: its output provably does not depend on memory content), the most active
    features at the last layer, and for Glassbox-3 the epistemic uncertainty u. The reply says
    whether the model emitted its <unsure> token."""
    block, V = model.config.block_size, tok.get_vocab_size()
    unsure_id = getattr(model.config, "unsure_id", None)
    special = lambda t: t >= V or t == unsure_id        # noqa: E731
    text_of = lambda ts: tok.decode([t for t in ts if not special(t)])   # noqa: E731
    ids = tok.encode(render_prompt(message, history)).ids[-(block - max_new_tokens):]
    P, out = len(ids), []
    for i in range(max_new_tokens):
        lg = model(torch.tensor([ids + out]))[0][0, -1]
        if i == 0:
            lg[0] = -float("inf")
        nxt = int(lg.argmax())
        if nxt == 0:
            break
        out.append(nxt)
        if "\n" in text_of(out):
            break
    if not out:
        return {"text": "", "tokens": [], "low_confidence_spans": [], "abstained": False}
    model(torch.tensor([ids + out]))
    tr = model.trace
    k = tr.get("k_eff", None)
    k = k if k is not None else model.know(tr["z_f"]).squeeze(-1)
    free = model.memory_free()[0]
    fi, fv = model.active_features(model.config.n_layer - 1)
    tokens = []
    for j, t in enumerate(out):
        pos = P - 1 + j
        r = {"token": "<unsure>" if special(t) else tok.decode([t]),
             "confidence": round(float(torch.sigmoid(k[0, pos])), 4),
             "memory_free": bool(free[pos]),
             "features": [[int(a), round(float(b), 3)] for a, b in zip(fi[0, pos, :3], fv[0, pos, :3])]}
        if "u" in tr:
            r["u"] = round(float(tr["u"][0, pos]), 4)
        tokens.append(r)
    rep = _report(text_of(out).split("\n")[0].strip(), tokens, threshold)
    rep["unsure"] = any(special(t) for t in out)
    rep["abstained"] = rep["abstained"] or rep["unsure"]
    rep["certified_memory_free"] = all(r["memory_free"] for r in tokens)
    return rep


class Glassbox2Responder(GlassboxResponder):
    def __init__(self, model, tok, manifest):
        super().__init__(model, tok, manifest)
        self.signals = ["confidence", "memory_free", "features"] + (["u", "unsure"] if isinstance(model, Glassbox3) else [])

    def reply(self, message, history=(), max_new_tokens=64, threshold=0.5):
        return reply_g2(self.model, self.tok, message, history, max_new_tokens, threshold)


HF_SYSTEM = "You are a helpful assistant. Answer briefly. If you are not sure, say that you don't know."


def export_hf_package(model_dir, out=None):
    """Pin a Hugging Face model directory that holds k_head.pt (the Kaggle retrofit's output):
    writes manifest.json with the sha256 of every file."""
    out = out or model_dir
    files = sorted(f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))
                   and f != "manifest.json")
    if "k_head.pt" not in files or "config.json" not in files:
        raise PackageError("a model directory with config.json and k_head.pt is required")
    hashes = {}
    for f in files:
        h = hashlib.sha256()
        with open(os.path.join(model_dir, f), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        hashes[f] = h.hexdigest()
    manifest = {"format": "vt-hf-chat/1", "files": hashes,
                "package_sha256": _sha("".join(hashes[f] for f in files).encode())}
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


class HFResponder:
    """A pretrained chat model retrofitted with a confidence head (kaggle/verifiable_chat).
    Reports per-token confidence from k and the softmax probability of each token.
    It has no evidence stream (that needs the glass-box architecture), so it does
    not report from_prompt / from_output / from_memory."""

    def __init__(self, path, max_bytes=8 << 30):
        import shutil
        import tempfile
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            manifest = json.load(open(os.path.join(path, "manifest.json")))
            want = manifest["files"]
        except (OSError, ValueError, KeyError) as e:
            raise PackageError(f"unreadable manifest: {e}") from e
        stage = tempfile.mkdtemp(prefix="vt-chat-")               # load only bytes that were checked
        try:
            for f, digest in want.items():
                if os.path.basename(f) != f:
                    raise PackageError(f"bad file name in manifest: {f!r}")
                h = hashlib.sha256()
                with open(os.path.join(path, f), "rb") as src, open(os.path.join(stage, f), "wb") as dst:
                    n = 0
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        n += len(chunk)
                        if n > max_bytes:
                            raise PackageError(f"{f} is larger than {max_bytes} bytes")
                        h.update(chunk)
                        dst.write(chunk)
                if h.hexdigest() != digest:
                    raise PackageError(f"{f} does not match the manifest")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(stage)
            self.model = AutoModelForCausalLM.from_pretrained(
                stage, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
            k = torch.load(os.path.join(stage, "k_head.pt"), map_location="cpu", weights_only=True)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        self.k_w, self.k_b = k["lin.weight"].float().to(dev), k["lin.bias"].float().to(dev)
        self.device, self.manifest = dev, manifest
        self.signals = ["confidence", "token_probability"]
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def reply(self, message, history=(), max_new_tokens=64, threshold=0.5):
        msgs = [{"role": "system", "content": HF_SYSTEM}]
        for u, a_ in history:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a_}]
        msgs.append({"role": "user", "content": message})
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(self.device)
        P = enc["input_ids"].shape[1]
        g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True,
                                return_dict_in_generate=True, pad_token_id=self.tok.pad_token_id)
        new = g.sequences[0, P:].tolist()
        if self.tok.eos_token_id in new:
            new = new[:new.index(self.tok.eos_token_id)]
        if not new:
            return {"text": "", "tokens": [], "low_confidence_spans": [], "abstained": False}
        probs = [float(F.softmax(s[0].float(), -1)[t]) for s, t in zip(g.scores, new)]
        seq = g.sequences[:, :P + len(new)]
        h = self.model(input_ids=seq, output_hidden_states=True).hidden_states[-1][0].float()
        k = torch.sigmoid(h[P - 1:P - 1 + len(new)] @ self.k_w.T + self.k_b).squeeze(-1)
        tokens = [{"token": self.tok.decode([t]), "confidence": round(float(c), 4), "token_probability": round(pr, 4)}
                  for t, c, pr in zip(new, k.tolist(), probs)]
        return _report(self.tok.decode(new, skip_special_tokens=True).strip(), tokens, threshold)


def _report(text, tokens, threshold):
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


def load_responder(path):
    """The right backend for a package directory, chosen by its manifest's format."""
    try:
        fmt = json.load(open(os.path.join(path, "manifest.json"))).get("format")
    except (OSError, ValueError) as e:
        raise PackageError(f"unreadable manifest: {e}") from e
    if fmt == "vt-hf-chat/1":
        return HFResponder(path)
    if fmt == "vt-glassbox-chat/1":
        return GlassboxResponder(*load_package(path))
    if fmt in ("vt-glassbox2-chat/1", "vt-glassbox3-chat/1"):
        return Glassbox2Responder(*load_package(path))
    raise PackageError(f"unknown package format {fmt!r}")


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


def make_chat_server(responder, host="127.0.0.1", port=8001, max_body=1 << 16, timeout=10.0,
                     deadline=60.0, max_concurrency=4):
    manifest = responder.manifest
    stats = {"requests": 0, "replies": 0, "errors_4xx": 0, "abstained": 0, "latency_s_total": 0.0}
    lock = threading.Lock()
    gen_lock = threading.Lock()                       # one generation at a time on CPU
    info = {"status": "ok", "format": manifest.get("format"), "package_sha256": manifest.get("package_sha256"),
            "signals": responder.signals}

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
                r = responder.reply(msg, hist, max_new_tokens=n, threshold=th)
            self.send_json(200, {"package_sha256": manifest.get("package_sha256"), **r})
            with lock:
                stats["replies"] += 1
                stats["abstained"] += int(r["abstained"])
                stats["latency_s_total"] += time.perf_counter() - t0

    return bounded_server(host, port, Handler, max_concurrency)
