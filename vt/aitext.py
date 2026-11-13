"""Cheap AI-text detection with a false-accusation cap the customer sets and can check.

Two small open models (Qwen3-0.6B base as observer, Qwen3-0.6B as performer) read the first 256 tokens of a text,
one forward pass each. Two scores come out:
    probe        logistic regression on the observer's mean hidden state at four depths plus seven scalar
                 statistics (log-perplexity, cross-perplexity, entropy, ...), trained on RAID (8 domains,
                 11 generators, plain and paraphrased; heads/aitext_probe_qwen3_0.6b.npz)
    binoculars   performer log-perplexity / observer-performer cross-entropy (Hans et al., 2024), zero-shot
A text is flagged when either score passes its threshold. The thresholds come from the customer's own human
writing (at least 146 texts, 150+ recommended): each is the one-sided Clopper-Pearson threshold at half the
false-alarm budget and half the miss-confidence, so by the union bound the pair keeps "at most fpr of human texts
flagged, with the stated confidence" on new texts of the same kind. No AI examples are needed from the customer.

Measured on RAID with every test domain and generator family held out of training
(results/ai_text/raid_generalize.json): promise 5% at 95% confidence from 150 human texts; measured false alarms
1.3% on average, the promise broken in 1.3% of 400 calibrations; 77% of AI text and 74% of paraphrased AI text
caught; probe AUROC 0.96 on generator families it never saw.

    vt-aitext train-head --sample sample.jsonl --features features.npz --out heads/aitext_probe_qwen3_0.6b.npz
    vt-aitext calibrate --human my_human_texts.jsonl --out calibration.json
    vt-aitext check --calibration calibration.json "some text" ...
    vt-aitext audit --calibration calibration.json --human new_human_texts.jsonl     # did the promise hold?
    vt-aitext serve --calibration calibration.json --port 8000         # POST /v1/aitext {"text": "..."}
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

OBSERVER, PERFORMER = "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"
LAYERS = (7, 14, 21, 28)
MAX_TOKENS = 256
SCALARS = ("nll_observer", "nll_performer", "xent_obs_perf", "entropy_observer", "nll_sd", "top1_share", "tokens")
HEAD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heads",
                    "aitext_probe_qwen3_0.6b.npz")
MIN_CHARS, MAX_CHARS = 200, 20000


class Reader:
    """Both models, in float32 on CPU (the probe was trained on float32 features)."""

    def __init__(self, observer=OBSERVER, performer=PERFORMER, threads=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(observer)
        self.obs = AutoModelForCausalLM.from_pretrained(observer, dtype=torch.float32).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(performer, dtype=torch.float32).eval()
        self.names = (observer, performer)

    def features(self, texts, batch=2, max_tokens=MAX_TOKENS):
        """(scalars [n, 7], hidden [n, 4 * d]) for each text; batches of `batch` (the full-vocabulary float32
        logits of both models dominate memory: about 2.5 GB a text at 256 tokens)."""
        import torch
        S, H = [], []
        for b in range(0, len(texts), batch):
            enc = [self.tok(t, add_special_tokens=False).input_ids[:max_tokens] for t in texts[b:b + batch]]
            enc = [e if len(e) >= 2 else e + [self.tok.eos_token_id] * (2 - len(e)) for e in enc]
            L = max(map(len, enc))
            ids = torch.full((len(enc), L), self.tok.pad_token_id or 0)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for i, e in enumerate(enc):
                ids[i, :len(e)], att[i, :len(e)] = torch.tensor(e), 1
            with torch.no_grad():
                o = self.obs(input_ids=ids, attention_mask=att, output_hidden_states=True)
                p = self.perf(input_ids=ids, attention_mask=att)
                lo = torch.log_softmax(o.logits[:, :-1].float(), -1)
                lp = torch.log_softmax(p.logits[:, :-1].float(), -1)
                tgt, m = ids[:, 1:], att[:, 1:].float()
                nll_o = -lo.gather(-1, tgt[..., None])[..., 0]
                nll_p = -lp.gather(-1, tgt[..., None])[..., 0]
                po = lo.exp()
                x_op = -(po * lp).sum(-1)                       # observer's distribution, performer's log-probs
                ent_o = -(po * lo).sum(-1)
                rank1 = (lo.argmax(-1) == tgt).float()
            S.append(np.c_[_mean(nll_o, m), _mean(nll_p, m), _mean(x_op, m), _mean(ent_o, m), _sd(nll_o, m),
                           _mean(rank1, m), m.sum(1).numpy()].astype(np.float32))
            am = att[..., None].float()
            H.append(np.concatenate([((o.hidden_states[k] * am).sum(1) / am.sum(1)).numpy() for k in LAYERS], 1)
                     .astype(np.float16))
            del o, p, lo, lp
        return np.concatenate(S), np.concatenate(H)


def _mean(v, m):
    return ((v * m).sum(1) / m.sum(1)).numpy()


def _sd(v, m):
    return np.sqrt(np.maximum(_mean(v ** 2, m) - _mean(v, m) ** 2, 0))


def design(scalars, hidden):
    S = scalars.astype(np.float64)
    return np.c_[hidden.astype(np.float32), S[:, :6], S[:, 1] / S[:, 2]]


def binoculars(scalars):
    """Higher = more AI-like."""
    S = scalars.astype(np.float64)
    return -(S[:, 1] / S[:, 2])


def train_head(scalars, hidden, y, out, C=0.01, meta=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = design(scalars, hidden)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    meta = dict(meta or {}, observer=OBSERVER, performer=PERFORMER, layers=list(LAYERS), max_tokens=MAX_TOKENS, C=C)
    np.savez(out, mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]), meta=json.dumps(meta))
    return out


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def min_texts(fpr, confidence, k=2):
    """The fewest human texts for which a threshold can promise fpr at this confidence (union over k scores)."""
    return math.ceil(math.log((1 - confidence) / k) / math.log(1 - fpr / k))


class Detector:
    def __init__(self, head=HEAD, reader=None, threads=None):
        z = np.load(head)
        self.meta = json.loads(str(z["meta"]))
        self.mean, self.scale, self.coef, self.b = z["mean"], z["scale"], z["coef"], float(z["intercept"])
        self.head_sha = file_sha(head)[:16]
        self.reader = reader or Reader(self.meta["observer"], self.meta["performer"], threads)

    def score_features(self, scalars, hidden):
        X = design(scalars, hidden)
        probe = ((X - self.mean) / self.scale) @ self.coef + self.b
        return {"probe": probe, "binoculars": binoculars(scalars), "tokens": scalars[:, 6].astype(int)}

    def scores(self, texts, batch=2):
        return self.score_features(*self.reader.features(list(texts), batch, self.meta["max_tokens"]))

    def calibrate(self, human_texts, fpr=0.05, confidence=0.95, batch=2):
        return calibration_from_scores(self.scores(human_texts, batch), fpr, confidence, self.head_sha)

    def audit(self, human_texts, calibration, batch=2):
        res = self.check(human_texts, calibration, batch)
        return audit_from_flags([r["verdict"] == "ai" for r in res], calibration["fpr"], calibration["confidence"])

    def check(self, texts, calibration=None, batch=2):
        s = self.scores(texts, batch)
        if calibration and calibration.get("head_sha256") not in (None, self.head_sha):
            raise ValueError("calibration was made with a different probe head; recalibrate")
        out = []
        for i in range(len(s["probe"])):
            r = {"probe": round(float(s["probe"][i]), 4), "binoculars": round(float(s["binoculars"][i]), 4),
                 "tokens": int(s["tokens"][i])}
            if calibration:
                fired = [k for k, t in calibration["thresholds"].items() if s[k][i] > t]
                r["verdict"] = "ai" if fired else "not flagged"
                r["fired"] = fired
                if r["tokens"] < 64:
                    r["note"] = "short text: under 64 tokens; the promise was measured on texts of up to 256"
            out.append(r)
        return out


def calibration_from_scores(s, fpr=0.05, confidence=0.95, head_sha=None):
    from vt.verifier_pipeline import guaranteed_threshold
    n = len(s["probe"])
    need = min_texts(fpr, confidence)
    if n < need:
        raise ValueError(f"{n} human texts: at least {need} are needed to promise {fpr:.0%} false alarms "
                         f"with {confidence:.0%} confidence (150+ recommended; more texts, more AI caught)")
    k, zero = 2, np.zeros(n, int)
    thr = {name: guaranteed_threshold(np.asarray(s[name], float), zero, fpr / k, 1 - (1 - confidence) / k)
           for name in ("probe", "binoculars")}
    return {"fpr": fpr, "confidence": confidence, "human_texts": n, "thresholds": thr, "head_sha256": head_sha,
            "promise": f"on new human texts like these, at most {fpr:.0%} are flagged as AI, with {confidence:.0%} "
                       f"confidence (union of two one-sided Clopper-Pearson bounds over {n} human texts)",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def audit_from_flags(flagged, fpr, confidence=0.95):
    """Did the promise hold on new human texts? `flagged`: one bool per text, from the calibrated detector.
    The promise is about the share flagged, so the audit reports that share with its exact 95% interval and the
    one-sided binomial p-value of seeing this many flags if the true rate were exactly fpr: "exceeded" when
    p < 1 - confidence (the evidence says the rate is above the promise), "consistent" otherwise."""
    from scipy.stats import beta, binom
    flagged = np.asarray(flagged, bool)
    n, k = len(flagged), int(flagged.sum())
    if not n:
        raise ValueError("no texts to audit")
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    p = float(binom.sf(k - 1, n, fpr))
    return {"human_texts": n, "flagged": k, "flagged_share": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "promised": fpr, "p_value": round(p, 4), "status": "exceeded" if p < 1 - confidence else "consistent",
            "flagged_indices": np.flatnonzero(flagged).tolist(),
            "note": "consistent means these texts give no evidence that more than the promised share of human "
                    "texts is flagged; with few texts the interval is wide"}


def read_texts(path):
    """A .jsonl of {"text": ...} rows, or a .txt with one text per blank-line-separated block."""
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line)["text"] for line in f if line.strip()]
        return [b.strip() for b in f.read().split("\n\n") if b.strip()]


def validate(item):
    if not isinstance(item, dict):
        return None, "each item must be an object"
    t = item.get("text")
    if not isinstance(t, str) or len(t.strip()) < MIN_CHARS:
        return None, f"text: a string of at least {MIN_CHARS} characters"
    if len(t) > MAX_CHARS:
        return None, f"text: at most {MAX_CHARS} characters (only the first {MAX_TOKENS} tokens are read)"
    return {"text": t}, None


class Service:
    """The model-owning lock and the per-key calibrations for the HTTP API (vt/verify_service.make_server)."""

    def __init__(self, detector, calibration=None, calibration_dir=None):
        import threading
        self.d, self.default, self.dir = detector, calibration, calibration_dir
        self.lock = threading.Lock()
        self.info = {"observer": detector.meta["observer"], "performer": detector.meta["performer"],
                     "head_sha256": detector.head_sha, "calibrated": calibration is not None,
                     "per_key_calibration": calibration_dir is not None}

    def calibration_for(self, key_name):
        if self.dir:
            p = os.path.join(self.dir, f"{key_name}.json")
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)
        return self.default

    def verify(self, item, key_name="open"):
        t = time.time()
        with self.lock:
            r = self.d.check([item["text"]], self.calibration_for(key_name))[0]
        cal = self.calibration_for(key_name)
        if cal:
            r["promise"] = cal["promise"]
        else:
            r["verdict"] = None
            r["note"] = "not calibrated: send 150+ of your own human texts to /v1/aitext/calibrate for a verdict"
        r["usage"] = {"tokens": r["tokens"]}
        r["ms"] = round(1000 * (time.time() - t), 1)
        return r

    def audit(self, texts, key_name):
        cal = self.calibration_for(key_name)
        if not cal:
            raise ValueError("not calibrated: nothing to audit")
        with self.lock:
            return self.d.audit(texts, cal)

    def calibrate(self, texts, fpr, confidence, key_name):
        with self.lock:
            cal = self.d.calibrate(texts, fpr, confidence)
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, f"{key_name}.json"), "w") as f:
                json.dump(cal, f, indent=1)
        else:
            self.default = cal
        return cal


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vt-aitext")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train-head", help="fit the probe head on read features (experiments/ai_text_detect.py)")
    p.add_argument("--sample", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", default=HEAD)
    p.add_argument("--C", type=float, default=0.01)
    for name in ("calibrate", "check", "audit", "serve"):
        q = sub.add_parser(name)
        q.add_argument("--head", default=HEAD)
        q.add_argument("--threads", type=int, default=None)
        q.add_argument("--batch", type=int, default=2)
        if name == "calibrate":
            q.add_argument("--human", required=True, help=".jsonl of {\"text\"} or .txt of blank-line-separated texts")
            q.add_argument("--fpr", type=float, default=0.05)
            q.add_argument("--confidence", type=float, default=0.95)
            q.add_argument("--out", required=True)
        else:
            q.add_argument("--calibration", default=None)
        if name == "audit":
            q.add_argument("--human", required=True, help="new human texts, not the calibration set")
        if name == "check":
            q.add_argument("texts", nargs="*", help="texts; or --file")
            q.add_argument("--file", default=None)
        if name == "serve":
            q.add_argument("--host", default="127.0.0.1")
            q.add_argument("--port", type=int, default=8000)
            q.add_argument("--rate-per-minute", type=int, default=60)
            q.add_argument("--calibration-dir", default=None, help="per-API-key calibrations (POST /v1/aitext/calibrate)")
    a = ap.parse_args(argv)
    if a.cmd == "train-head":
        with open(a.sample) as f:
            rows = [json.loads(line) for line in f]
        z = np.load(a.features)
        y = np.array([r["y"] for r in rows[:len(z["scalars"])]])
        meta = {"trained_on": f"RAID train sample: {len(y)} texts ({int(y.sum())} AI incl. paraphrased)"}
        print(train_head(z["scalars"], z["hidden"], y, a.out, a.C, meta))
        return
    det = Detector(a.head, threads=a.threads)
    if a.cmd == "calibrate":
        cal = det.calibrate(read_texts(a.human), a.fpr, a.confidence, a.batch)
        with open(a.out, "w") as f:
            json.dump(cal, f, indent=1)
        print(json.dumps(cal, indent=1))
        return
    cal = None
    if a.calibration:
        with open(a.calibration) as f:
            cal = json.load(f)
    if a.cmd == "audit":
        if not cal:
            raise SystemExit("audit needs --calibration")
        print(json.dumps(det.audit(read_texts(a.human), cal, a.batch), indent=1))
        return
    if a.cmd == "check":
        texts = (read_texts(a.file) if a.file else []) + a.texts
        for r in det.check(texts, cal, a.batch):
            print(json.dumps(r))
        return
    from vt.verify_service import keys_from_env, make_server
    svc = Service(det, cal, a.calibration_dir)
    srv = make_server(svc, a.host, a.port, keys_from_env(), a.rate_per_minute, max_body=8 << 20, deadline=1800.0,
                      path="/v1/aitext", validator=validate)
    print(f"AI-text API on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
