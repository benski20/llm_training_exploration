"""The grounding probe's features and models, shared by extraction (kaggle/probe/extract.py), training
(experiments/probe_train.py) and serving (GroundingProbe), so the three can never drift apart.

Reader-agnostic: layers are chosen as fractions of the reader's depth and every hidden vector is projected to P
dims by fixed Gaussian matrices from a seed, so a 1.7B, 8B or 14B reader produce features of the same shape and a
probe is trained / served by the same code. A probe is only valid for the reader (and precision) it was trained on:
the artifact records them, and GroundingProbe loads exactly that reader.
"""
import numpy as np

DEFAULTS = {"res_fracs": (0.3, 0.4, 0.5, 0.6, 0.7, 0.85), "neu_fracs": (0.45, 0.55, 0.65), "proj": 512, "seed": 0,
            "max_tokens": 3072}


class ProbeReader:
    def __init__(self, reader, load_4bit=False, res_fracs=DEFAULTS["res_fracs"], neu_fracs=DEFAULTS["neu_fracs"],
                 proj=DEFAULTS["proj"], seed=DEFAULTS["seed"], max_tokens=DEFAULTS["max_tokens"], threads=0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from vt.omni import judge_suffix
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        cuda = torch.cuda.is_available()
        self.tok = AutoTokenizer.from_pretrained(reader)
        kw = {"device_map": "auto"} if cuda else {}
        kw["torch_dtype"] = torch.float16 if cuda else torch.bfloat16
        if cuda and load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(reader, **kw).eval()
        cfg = self.model.config
        nL, d, inter = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
        self.res_layers = sorted({max(1, min(nL, int(round(f * nL)))) for f in res_fracs})
        self.neu_layers = sorted({max(0, min(nL - 1, int(round(f * nL)))) for f in neu_fracs})
        k = len(self.res_layers) // 2
        self.mid = [self.res_layers[k - 1], self.res_layers[k], self.res_layers[min(len(self.res_layers) - 1, k + 1)]]
        g = torch.Generator().manual_seed(seed)
        self.Pr = torch.randn(d, proj, generator=g) / proj ** 0.5
        self.Pn = torch.randn(inter, proj, generator=g) / proj ** 0.5
        self.dev = self.model.get_input_embeddings().weight.device
        self.max_tokens = max_tokens
        self.suffix = self.tok(judge_suffix(self.tok), add_special_tokens=False)["input_ids"]
        first = lambda ws: [self.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws]
        self.yes, self.no = first(["Yes", "yes", " Yes"]), first(["No", "no", " No"])
        self.meta = {"reader": reader, "load_4bit": bool(load_4bit), "n_layers": nL, "hidden": d,
                     "intermediate": inter, "res_layers": self.res_layers, "unit_res_layers": self.mid,
                     "neu_layers": self.neu_layers, "res_fracs": list(res_fracs), "neu_fracs": list(neu_fracs),
                     "proj": proj, "seed": seed, "max_tokens": max_tokens,
                     "layout": "vt.omni.make_item(response, question, documents)"}
        self._cap = {}
        for L in self.neu_layers:
            self.model.model.layers[L].mlp.down_proj.register_forward_pre_hook(self._hook(L))

    def _hook(self, L):
        def f(mod, inp):
            self._cap[L] = inp[0]
        return f

    def features(self, question, documents, response, spans=None, has_spans=False, label=0):
        """one forward pass; answer-level arrays and per-sentence arrays (None for an empty answer)."""
        torch = self.torch
        from vt.omni import encode, make_item, scalar_feats, sentence_spans, token_units
        it = make_item(response, question or "", documents or [])
        ids, r0, roff = encode(self.tok, it, self.max_tokens, self.suffix)
        ns = len(self.suffix)
        n = len(ids) - r0 - ns
        if n <= 0:
            return None
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([ids], device=self.dev), output_hidden_states=True,
                           logits_to_keep=n + ns + 1)
        lg = o.logits[0]
        last = lg[-1].float().log_softmax(-1)
        j_logit = float(torch.logsumexp(last[self.yes], 0) - torch.logsumexp(last[self.no], 0))
        logp = lg[:n].float().log_softmax(-1)
        tgt = torch.tensor(ids[r0:r0 + n], device=logp.device)
        lp = logp.gather(-1, tgt[:, None])[:, 0]
        ent = -(logp.exp() * logp).sum(-1)
        top2 = logp.topk(2, -1).values
        margin = torch.where(lp >= top2[:, 0] - 1e-6, top2[:, 0] - top2[:, 1], lp - top2[:, 0])
        lp, ent, margin = lp.cpu().numpy(), ent.cpu().numpy(), margin.cpu().numpy()
        proj = lambda x, P: x.float().cpu() @ P
        R = {L: proj(o.hidden_states[L][0, r0:r0 + n], self.Pr) for L in self.res_layers}
        J = np.stack([proj(o.hidden_states[L][0, -1:], self.Pr)[0].numpy() for L in self.res_layers])
        N = {L: proj(self._cap[L][0, r0:r0 + n], self.Pn) for L in self.neu_layers}
        self._cap.clear()
        spans = spans or []
        f = {"a_mean": np.stack([R[L].mean(0).numpy() for L in self.res_layers]).astype(np.float16),
             "a_last": np.stack([R[L][-1].numpy() for L in self.res_layers]).astype(np.float16),
             "a_judge": J.astype(np.float16),
             "a_neu": np.stack([N[L].mean(0).numpy() for L in self.neu_layers]).astype(np.float16),
             "a_sc": scalar_feats(lp, ent, margin), "j_logit": j_logit, "n_tok": len(ids),
             "truncated": int(n < len(self.tok(response, add_special_tokens=False)["input_ids"])), "units": []}
        units = sentence_spans(response)
        tu = token_units(roff[:n], units)
        for ui, (s, e) in enumerate(units):
            m = torch.from_numpy(tu == ui)
            if not bool(m.any()):
                continue
            lab = (1 if any(s < b and a < e for a, b in spans) else 0) if has_spans else -1
            if not has_spans and int(label) == 0:
                lab = 0                                          # a faithful answer: every sentence is fine
            mm = m.numpy()
            f["units"].append({"idx": ui, "start": s, "end": e, "lab": lab,
                               "res": np.stack([R[L][m].mean(0).numpy() for L in self.mid]).astype(np.float16),
                               "neu": np.stack([N[L][m].mean(0).numpy() for L in self.neu_layers]).astype(np.float16),
                               "sc": scalar_feats(lp[mm], ent[mm], margin[mm])})
        return f


# ------------------------------------------------------------------ vectors (identical in training and serving)

def answer_vector(a_mean, a_judge, a_neu, a_sc, j_logit):
    """batched: arrays with a leading answer axis."""
    n = len(a_mean)
    return np.concatenate([np.asarray(a_mean).reshape(n, -1), np.asarray(a_judge).reshape(n, -1),
                           np.asarray(a_neu).reshape(n, -1), np.asarray(a_sc).astype(np.float16),
                           np.asarray(j_logit).reshape(n, 1).astype(np.float16)], 1).astype(np.float16)


def unit_vector(u_res, u_neu, u_sc):
    m = len(u_res)
    return np.concatenate([np.asarray(u_res).reshape(m, -1), np.asarray(u_neu).reshape(m, -1),
                           np.asarray(u_sc).astype(np.float16)], 1).astype(np.float16)


# ------------------------------------------------------------------ models

def make_mlp(d):
    import torch.nn as nn
    return nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 1))


def make_set_attn(du, da, h=256):
    import torch
    import torch.nn as nn

    class SetAttn(nn.Module):
        """sentences -> shared encoder -> attention pooling, + the answer vector and the worst sentence -> logit."""

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(du, h), nn.GELU())
            self.att = nn.Linear(h, 1)
            self.uhead = nn.Linear(h, 1)
            self.ans = nn.Sequential(nn.Dropout(0.2), nn.Linear(da, h), nn.GELU())
            self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * h + 1, 1))

        def forward(self, idx, Xa, UX, rows):
            us, pooled, worst = [], [], []
            for i in idx:
                rr = rows[i]
                if not rr:
                    pooled.append(torch.zeros(self.att.in_features))
                    worst.append(torch.zeros(1))
                    us.append(None)
                    continue
                hh = self.enc(UX[rr])
                w = torch.softmax(self.att(hh)[:, 0], 0)
                pooled.append((w[:, None] * hh).sum(0))
                u = self.uhead(hh)[:, 0]
                us.append(u)
                worst.append(u.max()[None])
            z = torch.cat([torch.stack(pooled), self.ans(Xa[idx]), torch.stack(worst)], 1)
            return self.out(z)[:, 0], us
    return SetAttn()


# ------------------------------------------------------------------ serving

class GroundingProbe:
    """Load an artifact from experiments/probe_train.py --save and score answers with its own reader."""

    def __init__(self, path, reader=None, threads=0):
        import torch
        self.torch = torch
        self.art = torch.load(path, map_location="cpu", weights_only=False)
        m = self.art["meta"]
        self.reader = reader or ProbeReader(m["reader"], m.get("load_4bit", False),
                                            tuple(m.get("res_fracs", DEFAULTS["res_fracs"])),
                                            tuple(m.get("neu_fracs", DEFAULTS["neu_fracs"])),
                                            m["proj"], m["seed"], m["max_tokens"], threads)
        da, du = len(self.art["mu"]), len(self.art["umu"])
        self.mlp = self.sa = None
        if self.art.get("ans_mlp") is not None:
            self.mlp = make_mlp(da)
            self.mlp.load_state_dict(self.art["ans_mlp"])
            self.mlp.eval()
        if self.art.get("set_attn") is not None:
            self.sa = make_set_attn(du, da)
            self.sa.load_state_dict(self.art["set_attn"])
            self.sa.eval()

    def score(self, question, documents, response):
        """{"score": stacked log-odds of a hallucination, "flagged", "models": per-model scores, "sentences": [...]}."""
        torch = self.torch
        f = self.reader.features(question, documents, response)
        if f is None:
            return {"error": "empty answer"}
        art = self.art
        Xa = (answer_vector(f["a_mean"][None], f["a_judge"][None], f["a_neu"][None], f["a_sc"][None],
                            np.array([f["j_logit"]])).astype(np.float32) - art["mu"]) / art["sd"]
        U = f["units"]
        UX = ((unit_vector([u["res"] for u in U], [u["neu"] for u in U], [u["sc"] for u in U]).astype(np.float32)
               - art["umu"]) / art["usd"]) if U else np.zeros((0, len(art["umu"])), np.float32)
        s = {}
        if art.get("ans_lr") is not None:
            w, b = art["ans_lr"]
            s["ans_lr"] = float(Xa[0] @ w + b)
        us = None
        if art.get("unit_lr") is not None and len(U):
            w, b = art["unit_lr"]
            us = UX @ w + b
            s["unit_lr"] = float(us.max())
        with torch.no_grad():
            if self.mlp is not None:
                s["ans_mlp"] = float(self.mlp(torch.tensor(Xa))[0, 0])
            if self.sa is not None:
                s["set_attn"] = float(self.sa([0], torch.tensor(Xa), torch.tensor(UX), [list(range(len(U)))])[0][0])
        st = art["stack"]
        v = np.array([s.get(k, 0.0) for k in st["names"]])
        z = float(((v - st["mu"]) / st["sd"]) @ st["w"] + st["b"])
        out = {"score": z, "flagged": bool(z >= art.get("threshold", 0.0)), "models": s, "reader": art["meta"]["reader"]}
        if us is not None:
            out["sentences"] = [{"text": response[u["start"]:u["end"]], "start": u["start"], "end": u["end"],
                                 "score": float(x)} for u, x in zip(U, us)]
        return out
