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


def _hidden(f, parts, layer_idx):
    """the hidden-state block of a reply's feature vector: the chosen parts at the chosen layers, flattened.
    f: one reply's features (arrays (n_layers, d)) or a batch ((N, n_layers, d))."""
    key = {"mean": "r_mean", "last": "r_last", "judge": "j_state"}
    blocks = [np.asarray(f[key[p]])[..., layer_idx, :] for p in parts if p in key]
    H = np.concatenate([b.reshape(*b.shape[:-2], -1) for b in blocks], -1)
    return H.astype(np.float32)


def _scalars(f):
    return np.concatenate([np.asarray(f["r_sc"], np.float32), np.atleast_1d(np.asarray(f["j_logit"], np.float32))[..., None]
                           if np.ndim(f["j_logit"]) else np.array([f["j_logit"]], np.float32)], -1)


def _project(H, mu, sd, pmu, V):
    return (((H - mu) / sd) - pmu) @ V.T


def _pca(H, k):
    mu, sd = H.mean(0), H.std(0) + 1e-4
    Z = (H - mu) / sd
    pmu = Z.mean(0)
    _, _, Vt = np.linalg.svd(Z - pmu, full_matrices=False)
    return mu, sd, pmu, Vt[:k]


def _dstd(X, groups):
    X = X.copy()
    for g in np.unique(groups):
        m = groups == g
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-4)
    return X


def train_artifact(F, rows, parts=("mean",), layer_idx=None, pca=256, l2=0.05, unit_l2=0.05, reader=None):
    """The universal heads from stored corpus features (experiments/omni_corpus.py extract + load_features).
    rows: {id: corpus row}. Each training domain is standardised by its own statistics (as a new domain will be,
    from its unlabelled replies). Returns the artifact dict for OmniVerifier."""
    ids = F["ids"]
    dom = np.array([rows[i]["domain"] for i in ids])
    y = np.array([rows[i]["label"] for i in ids], float)
    layer_idx = list(layer_idx) if layer_idx is not None else [len(F["meta"]["layers"]) // 2 - 1,
                                                                 len(F["meta"]["layers"]) // 2,
                                                                 len(F["meta"]["layers"]) // 2 + 1]
    H = _hidden(F, parts, layer_idx)
    hmu, hsd, pmu, V = _pca(H, pca)
    X = np.concatenate([_project(H, hmu, hsd, pmu, V), np.concatenate([F["r_sc"], F["j_logit"][:, None]], 1)], 1)
    Xd = _dstd(X, dom)
    head = LogReg(l2).fit(Xd, y, balanced_weights(y))
    # units: the mean state of the unit at the same layers, its own projection, labelled units only
    U = F["u_mean"][:, layer_idx].reshape(len(F["u_row"]), -1).astype(np.float32)
    umu, usd, upmu, UV = _pca(U[np.random.default_rng(0).permutation(len(U))[:20000]], pca)
    pos = {i: k for k, i in enumerate(ids)}
    udom = dom[[pos[i] for i in F["u_row"]]]
    UX = np.concatenate([_project(U, umu, usd, upmu, UV), F["u_sc"]], 1)
    UXd = _dstd(UX, udom)
    lab = F["u_lab"] >= 0
    uy = F["u_lab"][lab].astype(float)
    uhead = LogReg(unit_l2).fit(UXd[lab], uy, balanced_weights(uy))
    meta = {"reader": reader or F["meta"]["reader"], "layers": F["meta"]["layers"], "layer_idx": layer_idx,
            "parts": list(parts), "pca": pca, "domains": sorted(set(dom.tolist())), "n_replies": int(len(y)),
            "n_units": int(lab.sum()), "max_tokens": F["meta"].get("max_tokens", 2048)}
    return {"meta": meta, "h_mu": hmu, "h_sd": hsd, "pca_mu": pmu, "pca_V": V, "x_mu": X.mean(0),
            "x_sd": X.std(0) + 1e-4, "w": head.w, "b": np.array(head.b), "uh_mu": umu, "uh_sd": usd,
            "upca_mu": upmu, "upca_V": UV, "ux_mu": UX.mean(0), "ux_sd": UX.std(0) + 1e-4, "uw": uhead.w,
            "ub": np.array(uhead.b)}


class OmniVerifier:
    """Universal heads + domain profiles.

    reply score   logistic head on [projected hidden block, output signals, judge log-odds]
    unit score    logistic head on [projected unit mean state, unit output signals]; reply's worst unit
    final score   z(reply score) + z(worst unit), each z by the profile's statistics (the domain's unlabelled
                  replies; the training mix for the universal profile)
    """

    def __init__(self, art, reader=None):
        self.art, self.reader = art, reader
        self.meta = art["meta"]
        self.layer_idx, self.parts = self.meta["layer_idx"], self.meta["parts"]

    @classmethod
    def load(cls, path, reader=None):
        z = np.load(path, allow_pickle=False)
        art = {k: z[k] for k in z.files}
        art["meta"] = json.loads(str(art["meta"]))
        return cls(art, reader)

    @staticmethod
    def save(art, path):
        a = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v)
             for k, v in art.items()}
        a["meta"] = json.dumps(a["meta"])
        np.savez(path, **a)

    def attach_reader(self, **kw):
        self.reader = Reader(self.meta["reader"], layers=self.meta["layers"],
                             max_tokens=self.meta.get("max_tokens", 2048), **kw)
        return self

    # -- feature vectors
    def reply_vector(self, f):
        a = self.art
        Z = _project(_hidden(f, self.parts, self.layer_idx), a["h_mu"], a["h_sd"], a["pca_mu"], a["pca_V"])
        return np.concatenate([Z, _scalars(f)], -1)

    def unit_matrix(self, f):
        a = self.art
        if not f["units"]:
            return np.zeros((0, len(a["ux_mu"])), np.float32)
        U = np.stack([np.asarray(u["mean"])[self.layer_idx].reshape(-1) for u in f["units"]]).astype(np.float32)
        return np.concatenate([_project(U, a["uh_mu"], a["uh_sd"], a["upca_mu"], a["upca_V"]),
                               np.stack([u["sc"] for u in f["units"]])], 1)

    def featurize(self, items):
        return [self.reader.features(it) for it in items]

    # -- profiles
    def universal_profile(self):
        a = self.art
        return {"name": "universal", "mu": a["x_mu"].tolist(), "sd": a["x_sd"].tolist(), "umu": a["ux_mu"].tolist(),
                "usd": a["ux_sd"].tolist(), "w": a["w"].tolist(), "b": float(a["b"]), "z": None, "threshold": None,
                "card": {"reader": self.meta["reader"], "trained_on": self.meta["domains"]}}

    def _raw(self, prof, feats):
        """(reply head score, worst-unit score) per reply."""
        r, u = [], []
        for f in feats:
            x = (self.reply_vector(f) - np.array(prof["mu"])) / np.array(prof["sd"])
            r.append(float(x @ np.array(prof["w"]) + prof["b"]))
            U = self.unit_matrix(f)
            us = ((U - np.array(prof["umu"])) / np.array(prof["usd"])) @ self.art["uw"] + float(self.art["ub"])
            u.append(float(us.max()) if len(us) else np.nan)
        return np.array(r), np.array(u)

    def _combine(self, prof, r, u):
        z = prof.get("z") or {"r": [0.0, 1.0], "u": [0.0, 1.0]}
        zr = (r - z["r"][0]) / z["r"][1]
        zu = np.where(np.isnan(u), 0.0, (np.nan_to_num(u) - z["u"][0]) / z["u"][1])
        return zr + zu

    def profile(self, feats, name="domain"):
        """Zero labels: the domain's unlabelled replies set every standardisation (covariate-shift correction)."""
        p = self.universal_profile()
        X = np.stack([self.reply_vector(f) for f in feats])
        U = np.concatenate([self.unit_matrix(f) for f in feats])
        p.update(name=name, mu=X.mean(0).tolist(), sd=(X.std(0) + 1e-4).tolist())
        if len(U) > 1:
            p.update(umu=U.mean(0).tolist(), usd=(U.std(0) + 1e-4).tolist())
        r, u = self._raw(p, feats)
        p["z"] = {"r": [float(r.mean()), float(r.std() + 1e-6)], "u": [float(np.nanmean(u)), float(np.nanstd(u) + 1e-6)]}
        p["card"] = dict(p["card"], unlabelled_replies=len(feats))
        return p

    def score(self, prof, feats):
        r, u = self._raw(prof, feats)
        return self._combine(prof, r, u)

    def adapt(self, prof, feats, y, grid=(0.01, 0.1, 1.0, 10.0), seed=0):
        """k labelled replies: the reply head refitted with a pull toward its current weights; the strength by
        repeated half splits on the final score (kept unchanged unless a refit helps)."""
        from vt.correctness import auroc
        y = np.asarray(y, float)
        X = (np.stack([self.reply_vector(f) for f in feats]) - np.array(prof["mu"])) / np.array(prof["sd"])
        _, u = self._raw(prof, feats)
        w0, b0 = np.array(prof["w"]), prof["b"]
        rng = np.random.default_rng(seed)
        res = {None: []}
        res.update({lp: [] for lp in grid})
        for _ in range(5):
            perm = rng.permutation(len(y))
            a, b = perm[:len(y) // 2], perm[len(y) // 2:]
            if len(set(y[a])) < 2 or len(set(y[b])) < 2:
                continue
            res[None].append(auroc(self._combine(prof, X[b] @ w0 + b0, u[b]), y[b]))
            for lp in grid:
                m = LogReg(lp, w0, b0).fit(X[a], y[a], balanced_weights(y[a]))
                res[lp].append(auroc(self._combine(prof, m.score(X[b]), u[b]), y[b]))
        best = None
        if res[None]:
            base = np.mean(res[None])
            for lp in grid:
                if res[lp] and np.mean(res[lp]) > base + 0.005:
                    best, base = lp, np.mean(res[lp])
        p = dict(prof, card=dict(prof["card"], adapted_on=int(len(y)), prior_strength=best))
        if best is not None:
            m = LogReg(best, w0, b0).fit(X, y, balanced_weights(y))
            p.update(w=m.w.tolist(), b=m.b, prior_w=w0.tolist(), prior_b=float(b0))
            r = X @ m.w + m.b
            p["z"] = dict(p["z"] or {}, r=[float(r.mean()), float(r.std() + 1e-6)], u=(p["z"] or {}).get("u", [0.0, 1.0]))
        return p

    def _oof(self, prof, feats, y, folds=5, seed=0):
        X = (np.stack([self.reply_vector(f) for f in feats]) - np.array(prof["mu"])) / np.array(prof["sd"])
        _, u = self._raw(prof, feats)
        w0, b0, lp = np.array(prof["prior_w"]), prof["prior_b"], prof["card"]["prior_strength"]
        f = np.arange(len(y)) % folds
        np.random.default_rng(seed).shuffle(f)
        r = np.zeros(len(y))
        for k in range(folds):
            tr, te = f != k, f == k
            if len(set(y[tr].tolist())) < 2:
                r[te] = X[te] @ w0 + b0
            else:
                r[te] = LogReg(lp, w0, b0).fit(X[tr], y[tr], balanced_weights(y[tr])).score(X[te])
        return self._combine(prof, r, u)

    def calibrate(self, prof, feats, y, fpr=0.10, confidence=0.95):
        """threshold from the domain's labelled *clean* replies: false alarms <= fpr with stated confidence
        (cross-fitted when the profile was adapted on the same labels)."""
        y = np.asarray(y)
        s = self._oof(prof, feats, y.astype(float)) if prof.get("prior_w") is not None else self.score(prof, feats)
        need = min_clean_for_guarantee(fpr, confidence)
        p = dict(prof, threshold=cp_threshold(s[y == 0], fpr, confidence))
        p["card"] = dict(prof["card"], fpr_target=fpr, confidence=confidence, clean_replies=int((y == 0).sum()),
                         guarantee_possible=bool((y == 0).sum() >= need), min_clean_needed=need)
        if (y == 1).any():
            p["card"]["recall_on_calibration"] = float((s[y == 1] >= p["threshold"]).mean())
        return p

    # -- one check
    def verify(self, reply, question="", documents=None, messages=None, prof=None, units=None):
        prof = prof or self.universal_profile()
        it = make_item(reply, question, documents, messages, units)
        f = self.reader.features(it)
        if f is None:
            return {"error": "empty reply"}
        r, u = self._raw(prof, [f])
        z = float(self._combine(prof, r, u)[0])
        out = {"score": z, "reply_score": float(r[0]), "worst_unit_score": None if np.isnan(u[0]) else float(u[0]),
               "judge_logodds_yes": float(f["j_logit"]), "profile": prof["name"]}
        if prof.get("threshold") is not None:
            out["flagged"] = bool(z >= prof["threshold"])
            out["guarantee"] = {k: prof["card"].get(k) for k in ("fpr_target", "confidence", "clean_replies")}
        U = self.unit_matrix(f)
        us = ((U - np.array(prof["umu"])) / np.array(prof["usd"])) @ self.art["uw"] + float(self.art["ub"])
        out["units"] = [{"text": reply[a:b], "start": a, "end": b, "score": float(sc)}
                        for (a, b), sc in ((it["units"][fu["index"]], sc) for fu, sc in zip(f["units"], us))]
        return out

    @staticmethod
    def save_profile(prof, path):
        json.dump(prof, open(path, "w"))

    @staticmethod
    def load_profile(path):
        return json.load(open(path))


def feature_dicts(F):
    """per-reply feature dicts (as Reader.features returns) from stored corpus features (load_features)."""
    by_row = {}
    for k, r in enumerate(F["u_row"]):
        by_row.setdefault(int(r), []).append({"index": int(F["u_idx"][k]), "mean": F["u_mean"][k].astype(np.float32),
                                              "sc": F["u_sc"][k]})
    return [{"r_mean": F["r_mean"][i].astype(np.float32), "r_last": F["r_last"][i].astype(np.float32),
             "r_sc": F["r_sc"][i], "j_state": F["j_state"][i].astype(np.float32), "j_logit": float(F["j_logit"][i]),
             "units": by_row.get(int(F["ids"][i]), [])} for i in range(len(F["ids"]))]


def subset(F, mask):
    """stored features restricted to the replies in mask (and their units)."""
    keep = set(np.asarray(F["ids"])[mask].tolist())
    um = np.array([int(r) in keep for r in F["u_row"]])
    out = {k: (v[mask] if k in ("ids", "r_mean", "r_last", "r_sc", "j_state", "j_logit") else v[um])
           for k, v in F.items() if k != "meta"}
    out["meta"] = F["meta"]
    return out
