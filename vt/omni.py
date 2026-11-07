"""Omni verifier: one forward pass of a small frozen reader scores any LLM reply -- RAG answer, closed-book chat,
long-form text, step-by-step reasoning -- for how likely it is to be wrong, and adapts to a new domain in minutes.

    reply (+ question, + optional documents)
        -> reader (Qwen3, frozen) reads the conversation once
        -> per reply: mean residual state at a few layers, projected (PCA) + output signals (log-prob, entropy,
           margin); per unit (sentence / step / segment): the same pooled over the unit
        -> universal head (logistic, trained on ~20 domains in 4 formats)            -> score
        -> domain profile: feature statistics from the domain's *unlabelled* replies (no labels),
           optionally a refit pulled toward the universal head from k labels, and a threshold that holds
           a false-alarm rate with stated confidence (Clopper-Pearson on the domain's clean replies)
        -> {p_wrong, flagged, units: [{text, score}], profile, threshold guarantee}

Adapting to a domain (all on stored features, seconds):

    v = OmniVerifier.load("heads/omni_qwen3_1p7b.npz")
    feats = v.featurize(unlabelled_items)                 # one pass each
    prof = v.profile(feats)                               # zero labels: per-domain standardisation
    prof = v.adapt(prof, labelled_feats, labels)          # optional: k labels, prior-regularised refit
    prof = v.calibrate(prof, labelled_feats, labels, fpr=0.10, confidence=0.95)
    v.save_profile(prof, "acme_support.json")
"""
import json
import re

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

RAG_SYSTEM = "You are a helpful assistant. Answer using the provided information."
PLAIN_SYSTEM = "You are a helpful assistant."
SCALARS = ("lp_mean", "lp_min", "lp_p10", "ent_mean", "ent_max", "ent_p90", "margin_mean", "margin_min",
           "low_lp_frac", "n_tokens_log")


# ------------------------------------------------------------------ text units

def sentence_spans(text):
    """[(start, end)] of the sentences / lines / steps of a text (no empty ones)."""
    out, start = [], 0
    for m in re.finditer(r"(?<!\d)[.!?](?=\s)|\n+", text):
        out.append((start, m.end()))
        start = m.end()
    if start < len(text):
        out.append((start, len(text)))
    return [(a, b) for a, b in out if text[a:b].strip()]


def make_item(reply, question="", documents=None, messages=None, units=None):
    """The reader's view of a check. messages (chat history) take precedence over question."""
    doc = "\n\n".join(documents or [])
    if messages:
        msgs = [dict(m) for m in messages]
    else:
        msgs = [{"role": "user", "content": question}]
    if not msgs or msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": RAG_SYSTEM if doc else PLAIN_SYSTEM})
    if doc:
        last = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        msgs[last]["content"] = f"{doc}\n\n{msgs[last]['content']}"
    return {"messages": msgs, "response": reply, "units": units or sentence_spans(reply)}


# ------------------------------------------------------------------ features

def scalar_feats(lp, ent, margin):
    if len(lp) == 0:
        return np.zeros(len(SCALARS), np.float32)
    return np.array([lp.mean(), lp.min(), np.percentile(lp, 10), ent.mean(), ent.max(), np.percentile(ent, 90),
                     margin.mean(), margin.min(), (lp < np.log(0.1)).mean(), np.log1p(len(lp))], np.float32)


JUDGE_QUESTION = "Is the reply above fully correct, accurate and supported? Answer Yes or No."


def _template(tok, msgs, gen=True):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)


def judge_suffix(tok):
    """the text after the reply that closes its turn and asks the judge question (PRISM-style framing; the
    reader is causal, so the reply's own states do not change)."""
    mark = "\u2063REPLY\u2063"
    t = _template(tok, [{"role": "user", "content": "Q"}, {"role": "assistant", "content": mark},
                        {"role": "user", "content": JUDGE_QUESTION}])
    return t[t.index(mark) + len(mark):]


def encode(tok, item, max_tokens, suffix_ids=()):
    msgs = item["messages"]
    head = _template(tok, msgs)
    enc = tok(item["response"], add_special_tokens=False, return_offsets_mapping=True)
    rid, roff = enc["input_ids"], enc["offset_mapping"]
    if len(rid) > max_tokens - 64 - len(suffix_ids):
        rid, roff = rid[:max_tokens - 64 - len(suffix_ids)], roff[:max_tokens - 64 - len(suffix_ids)]
    hid = tok(head, add_special_tokens=False)["input_ids"]
    room = max_tokens - len(rid) - len(suffix_ids)
    if len(hid) > room:                                      # keep the start (system, task) and the end (question)
        keep_end = min(256, room // 2)
        hid = hid[:room - keep_end] + hid[-keep_end:]
    return hid + rid + list(suffix_ids), len(hid), roff


def token_units(roff, units):
    out = np.full(len(roff), -1, dtype=np.int32)
    starts = np.array([s for s, _ in roff])
    ends = np.array([e for _, e in roff])
    for k, (a, b) in enumerate(units):
        m = (starts < b) & (ends > a) & (out < 0)
        out[m] = k
    return out


class Reader:
    """The frozen reader: one forward pass per item, features as in experiments/omni_corpus.py."""

    def __init__(self, name, layers=None, layer_fracs=(0.25, 0.4, 0.55, 0.7, 0.85, 1.0), max_tokens=2048,
                 threads=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.name, self.max_tokens = name, max_tokens
        self.tok = AutoTokenizer.from_pretrained(name)
        cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16 if (not cuda or torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16,
            device_map="auto" if cuda else None).eval()
        nL = self.model.config.num_hidden_layers
        self.layers = list(layers) if layers is not None else sorted({int(round(f * nL)) for f in layer_fracs})
        self.suffix = self.tok(judge_suffix(self.tok), add_special_tokens=False)["input_ids"]
        ids = lambda w: self.tok(w, add_special_tokens=False)["input_ids"][0]
        self.yes = [ids(w) for w in ("Yes", "yes", " Yes")]
        self.no = [ids(w) for w in ("No", "no", " No")]

    def features(self, item):
        import torch
        ids, r0, roff = encode(self.tok, item, self.max_tokens, self.suffix)
        ns = len(self.suffix)
        n = len(ids) - r0 - ns
        if n <= 0:
            return None
        dev = self.model.get_input_embeddings().weight.device
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True)
        last = o.logits[0, -1].float().log_softmax(-1)
        j_logit = float(torch.logsumexp(last[self.yes], 0) - torch.logsumexp(last[self.no], 0))
        j_state = torch.stack([o.hidden_states[L][0, -1] for L in self.layers]).float().cpu().numpy()
        logp = o.logits[0, r0 - 1:r0 + n - 1].float().log_softmax(-1)
        tgt = torch.tensor(ids[r0:r0 + n], device=logp.device)
        lp = logp.gather(-1, tgt[:, None])[:, 0]
        ent = -(logp.exp() * logp).sum(-1)
        top2 = logp.topk(2, -1).values
        margin = torch.where(lp >= top2[:, 0] - 1e-6, top2[:, 0] - top2[:, 1], lp - top2[:, 0])
        lp, ent, margin = lp.cpu().numpy(), ent.cpu().numpy(), margin.cpu().numpy()
        H = torch.stack([o.hidden_states[L][0, r0:r0 + n] for L in self.layers]).float().cpu()
        tu = token_units(roff[:n], item["units"])
        units = []
        for k in range(len(item["units"])):
            m = tu == k
            if m.any():
                units.append({"index": k, "mean": H[:, torch.from_numpy(m)].mean(1).numpy(),
                              "sc": scalar_feats(lp[m], ent[m], margin[m])})
        return {"r_mean": H.mean(1).numpy(), "r_last": H[:, -1].numpy(), "r_sc": scalar_feats(lp, ent, margin),
                "j_state": j_state, "j_logit": j_logit, "units": units}


# ------------------------------------------------------------------ heads

class LogReg:
    """L2 logistic regression on standardised features; with w0 the penalty pulls toward w0 (a prior)."""

    def __init__(self, l2=0.05, w0=None, b0=0.0):
        self.l2, self.w0, self.b0 = l2, w0, b0

    def fit(self, X, y, sample_weight=None):
        n, d = X.shape
        w0 = np.zeros(d) if self.w0 is None else self.w0
        sw = np.ones(n) if sample_weight is None else sample_weight / np.mean(sample_weight)

        def f(p):
            w, b = p[:d], p[d]
            z = X @ w + b
            g = (1 / (1 + np.exp(-z)) - y) * sw
            loss = (sw * (np.logaddexp(0, z) - y * z)).mean() + 0.5 * self.l2 * ((w - w0) ** 2).sum()
            return loss, np.concatenate([X.T @ g / n + self.l2 * (w - w0), [g.mean()]])

        r = minimize(f, np.concatenate([w0, [self.b0]]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.w, self.b = r.x[:d], float(r.x[d])
        return self

    def score(self, X):
        return X @ self.w + self.b


def balanced_weights(y):
    p = float(np.mean(y))
    return np.where(y == 1, 0.5 / max(p, 1e-6), 0.5 / max(1 - p, 1e-6))


def cp_threshold(scores_clean, fpr, conf):
    """lowest threshold whose false-alarm rate is <= fpr with confidence conf (one-sided Clopper-Pearson)."""
    s = np.sort(np.asarray(scores_clean))[::-1]
    n = len(s)
    for k in range(n + 1):
        ub = 1.0 if k == n else beta_dist.ppf(conf, k + 1, n - k)
        if ub > fpr:
            return float(s[k - 1] + 1e-9) if k > 0 else float(s[0] + 1e-9)
    return float(s[-1])


def min_clean_for_guarantee(fpr, conf):
    """fewest clean replies for which any threshold can promise fpr at confidence conf: (1-fpr)^n <= 1-conf."""
    return int(np.ceil(np.log(1 - conf) / np.log(1 - fpr)))


class OmniVerifier:
    """Universal head + domain profiles. The head works on a feature vector built from the reader's features:
    the chosen layers' mean states, projected by a stored PCA basis, plus the scalar output signals."""

    def __init__(self, art, reader=None):
        self.art = art
        self.reader = reader
        self.layer_idx = [int(i) for i in art["layer_idx"]]
        self.use_scalars = bool(art["use_scalars"])

    # -- persistence
    @classmethod
    def load(cls, path, reader=None):
        z = np.load(path, allow_pickle=False)
        art = {k: z[k] for k in z.files}
        art["meta"] = json.loads(str(art["meta"]))
        return cls(art, reader)

    @staticmethod
    def save(art, path):
        a = dict(art)
        a["meta"] = json.dumps(a["meta"])
        np.savez(path, **a)

    def attach_reader(self, **kw):
        self.reader = Reader(self.art["meta"]["reader"], layers=self.art["meta"]["layers"], **kw)
        return self

    # -- features -> head input
    def vector(self, r_mean, r_sc):
        H = np.asarray(r_mean)[..., self.layer_idx, :].reshape(*np.shape(r_mean)[:-2], -1).astype(np.float32)
        Z = ((H - self.art["h_mu"]) / self.art["h_sd"] - self.art["pca_mu"]) @ self.art["pca_V"].T
        if self.use_scalars:
            Z = np.concatenate([Z, np.asarray(r_sc, np.float32)], -1)
        return Z

    def featurize(self, items):
        out = []
        for it in items:
            f = self.reader.features(it)
            out.append(f)
        return out

    def matrix(self, feats):
        return np.stack([self.vector(f["r_mean"], f["r_sc"]) for f in feats])

    # -- domain profiles
    def universal_profile(self):
        return {"name": "universal", "mu": self.art["x_mu"].tolist(), "sd": self.art["x_sd"].tolist(),
                "w": self.art["w"].tolist(), "b": float(self.art["b"]), "threshold": None, "card": {}}

    def profile(self, X_unlabelled, name="domain"):
        """Zero labels: standardise by the domain's own unlabelled replies (covariate-shift correction)."""
        X = np.asarray(X_unlabelled)
        p = self.universal_profile()
        p.update(name=name, mu=X.mean(0).tolist(), sd=(X.std(0) + 1e-4).tolist())
        p["card"]["unlabelled_replies"] = int(len(X))
        return p

    def score(self, prof, X):
        Xs = (np.asarray(X) - np.array(prof["mu"])) / np.array(prof["sd"])
        return Xs @ np.array(prof["w"]) + prof["b"]

    def adapt(self, prof, X, y, grid=(0.01, 0.1, 1.0, 10.0), seed=0):
        """k labels: refit pulled toward the profile's current weights; the strength by repeated half splits
        (the unchanged head is kept if no refit helps)."""
        from vt.correctness import auroc
        y = np.asarray(y, float)
        Xs = (np.asarray(X) - np.array(prof["mu"])) / np.array(prof["sd"])
        w0, b0 = np.array(prof["w"]), prof["b"]
        rng = np.random.default_rng(seed)
        best, best_s = None, None
        base = []
        cands = {lp: [] for lp in grid}
        for _ in range(5):
            perm = rng.permutation(len(y))
            a, b = perm[:len(y) // 2], perm[len(y) // 2:]
            if len(set(y[a])) < 2 or len(set(y[b])) < 2:
                continue
            base.append(auroc(Xs[b] @ w0 + b0, y[b]))
            for lp in grid:
                m = LogReg(lp, w0, b0).fit(Xs[a], y[a], balanced_weights(y[a]))
                cands[lp].append(auroc(m.score(Xs[b]), y[b]))
        if base:
            best_s = np.mean(base)
            for lp, v in cands.items():
                if v and np.mean(v) > best_s + 0.005:
                    best, best_s = lp, np.mean(v)
        p = dict(prof)
        p["card"] = dict(prof["card"], adapted_on=int(len(y)), prior_strength=best)
        if best is not None:
            m = LogReg(best, w0, b0).fit(Xs, y, balanced_weights(y))
            p["w"], p["b"] = m.w.tolist(), m.b
        return p

    def calibrate(self, prof, X, y, fpr=0.10, confidence=0.95):
        """threshold from the domain's labelled *clean* replies: false alarms <= fpr with stated confidence."""
        y = np.asarray(y)
        clean = self.score(prof, np.asarray(X)[y == 0])
        need = min_clean_for_guarantee(fpr, confidence)
        p = dict(prof)
        p["threshold"] = cp_threshold(clean, fpr, confidence)
        p["card"] = dict(prof["card"], fpr_target=fpr, confidence=confidence, clean_replies=int(len(clean)),
                         guarantee_possible=bool(len(clean) >= need), min_clean_needed=need)
        if (y == 1).any():
            p["card"]["recall_on_calibration"] = float((self.score(prof, np.asarray(X)[y == 1]) >= p["threshold"]).mean())
        return p

    # -- one check
    def verify(self, reply, question="", documents=None, messages=None, prof=None, units=None):
        prof = prof or self.universal_profile()
        it = make_item(reply, question, documents, messages, units)
        f = self.reader.features(it)
        if f is None:
            return {"error": "empty reply"}
        z = float(self.score(prof, self.vector(f["r_mean"], f["r_sc"])[None])[0])
        out = {"score": z, "p_wrong": 1 / (1 + np.exp(-z)), "profile": prof["name"]}
        if prof.get("threshold") is not None:
            out["flagged"] = bool(z >= prof["threshold"])
            out["guarantee"] = {k: prof["card"].get(k) for k in ("fpr_target", "confidence", "clean_replies")}
        if "uw" in self.art:                                      # unit head: which sentence / step
            us = []
            for u in f["units"]:
                v = self.unit_vector(u)
                a, b = it["units"][u["index"]]
                us.append({"text": reply[a:b], "start": a, "end": b,
                           "score": float(((v - self.art["ux_mu"]) / self.art["ux_sd"]) @ self.art["uw"] + self.art["ub"])})
            out["units"] = us
        return out

    def unit_vector(self, u):
        H = np.asarray(u["mean"])[self.layer_idx].reshape(-1).astype(np.float32)
        Z = ((H - self.art["h_mu"]) / self.art["h_sd"] - self.art["pca_mu"]) @ self.art["pca_V"].T
        return np.concatenate([Z, u["sc"]]) if self.use_scalars else Z

    @staticmethod
    def save_profile(prof, path):
        json.dump(prof, open(path, "w"))

    @staticmethod
    def load_profile(path):
        return json.load(open(path))
