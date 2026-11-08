"""Kaggle kernel + local analysis: closed-book chat replies, "retrieve, then ground".

For every closed-book reply of truthfulqa, halueval_general, felm_wk, felm_science (experiments/omni_corpus.py):
    1. queries from the question and the reply's claims (sentences with names / numbers), each looked up on live
       Wikipedia (vt.retrieval.Wikipedia, full_text=True, politely paced: one request in flight, 0.35 s apart,
       backoff on HTTP 429); the passages pooled and the best K kept (one per query first, then BM25 on
       question + reply)
    2. the Qwen3-8B reader (vt.omni.Reader, shipped head heads/omni_qwen3_8b.npz) reads each reply twice:
       closed book exactly as in the corpus, and with the retrieved passages as documents (vt.omni.make_item)
    3. NLIScorer (HHEM-2.1-open + MiniCheck-RoBERTa-L) on reply vs retrieved passages; MiniCheck also on every
       sentence (the shipped scorer checks the first 4)
Retrieval runs in a thread ahead of the GPU loop. Compact per-reply vectors (head input, unit inputs, judge
log-odds, NLI parts, timings, evidence) go to /kaggle/working/cb_scores.npz + evidence.json.

    python kaggle/kgrun.py --slug omni-cb-retrieval --script kaggle/omni/closedbook_retrieval.py --bundle ... \
        --args "--smoke 5" --out results/kaggle/omni-cb-retrieval-smoke --wait
    # local, CPU-light: AUROCs per domain from the fetched scores + the LODO closed-book expert scores
    python kaggle/omni/closedbook_retrieval.py --analyse results/kaggle/omni-cb-retrieval \
        --expert results/omni/expert_scores_q8b.npz --report results/omni/closedbook_retrieval.json
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

SRC = "/kaggle/working/src"
WORK = "/kaggle/working"
LI = [2, 3, 4]                                              # the head's layer_idx (meta)
DOMAINS = ("truthfulqa", "halueval_general", "felm_wk", "felm_science")
CORPUS_MD5 = "b598848a44e70308b64ae5f0072b6eb0"
T0 = time.time()
# manual spot-check of the full run (10 random wrong replies per domain, seed 0): does the retrieved evidence state the
# fact that decides the reply is wrong? (read by hand from evidence.json; partial = the topic page, not the fact)
SPOT_CHECK = {"truthfulqa": {"yes": 2, "partial": 1, "no": 7, "note": "pyramids / Bermuda Triangle found; most "
                             "misconception questions retrieve off-topic pages (Chrissy Teigen 'Cravings', Grizzy & the Lemmings)"},
              "halueval_general": {"yes": 0, "partial": 1, "no": 9, "note": "9/10 wrong replies are to tasks with no "
                                   "retrievable fact (missing input, creative writing, arithmetic, refusals)"},
              "felm_wk": {"yes": 2, "partial": 3, "no": 5, "note": "Saturn density, Friday-13th years found; Victoria "
                          "Falls passage supports the wrong claim; Leicester query retrieved Cambridge"},
              "felm_science": {"yes": 0, "partial": 2, "no": 8, "note": "technical claims (EPR line counts, Newton's "
                               "law reasoning) are not stated in any retrieved passage"}}


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


# ---------------------------------------------------------------- retrieval

def make_wiki():
    from vt.retrieval import Wikipedia

    class Wiki(Wikipedia):
        n_requests = 0

        def _fetch(self, url, retries=4):
            import urllib.error
            import urllib.request
            Wiki.n_requests += 1
            req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosec B310: fixed https endpoints
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503) or retries <= 0:
                    raise
                self.errors[f"retry {e.code}"] += 1
                try:
                    wait = min(8.0, float(e.headers.get("Retry-After") or 0) or 2.0 ** (4 - retries))
                except ValueError:
                    wait = 2.0
                time.sleep(wait)
                return self._fetch(url, retries - 1)

    return Wiki(k=2, full_text=True, pages=2, passage_words=100, per_page=2, timeout=8.0, budget_s=60.0,
                in_flight=1, interval=0.35)


def claims(reply, n):
    """up to n sentences of the reply that carry checkable content (names, numbers), in reply order."""
    from vt.omni import sentence_spans
    sents = [reply[a:b].strip() for a, b in sentence_spans(reply)]
    sents = [s for s in sents if len(s.split()) >= 4]

    def weight(s):
        t = s.split()
        return sum(1 for w in t[1:] if w[:1].isupper()) + sum(1 for w in t if any(c.isdigit() for c in w))

    best = sorted(range(len(sents)), key=lambda i: -weight(sents[i]))[:n]
    return [sents[i][:300] for i in sorted(best)]


def gather(wiki, question, reply, domain, K=6):
    from vt.retrieval import Corpus
    q = " ".join(question.split())[-300:]
    if len(reply.split()) < 25:                             # short answer: question + answer, then question alone
        qs = [f"{q} {' '.join(reply.split())}", q]
    else:
        qs = [q] + claims(reply, 3 if domain == "halueval_general" else 4)
    lists = [wiki(x) for x in qs]
    pool, seen = [], set()
    for L in lists:
        for p in L:
            if p not in seen:
                seen.add(p)
                pool.append(p)
    if not pool:
        return [], qs
    s = Corpus(pool).scores(f"{q} {reply}")
    out = []
    for L in lists:                                         # each query's best passage first
        if L and L[0] not in out and len(out) < K:
            out.append(L[0])
    for i in sorted(range(len(pool)), key=lambda i: -s[i]):
        if len(out) >= K:
            break
        if pool[i] not in out:
            out.append(pool[i])
    return out, qs


# ---------------------------------------------------------------- kernel

def setup():
    import transformers
    v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    log("transformers", transformers.__version__)
    if v < (4, 56):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"])


def build_corpus():
    import hashlib
    path = "/tmp/omni/corpus.jsonl"
    os.makedirs("/tmp/omni", exist_ok=True)
    if not os.path.exists(path):
        subprocess.run([sys.executable, os.path.join(SRC, "experiments", "omni_corpus.py"), "build", "--out", path],
                       stdout=open(os.path.join(WORK, "build.log"), "w"), stderr=subprocess.STDOUT, check=True)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    log("corpus md5", md5, "same as repo" if md5 == CORPUS_MD5 else "DIFFERS")
    return [json.loads(l) for l in open(path)], md5


class GPUNLI:
    """vt.omni.NLIScorer on a GPU, returning its parts: HHEM (reply level), MiniCheck as shipped (first 4
    sentences), MiniCheck on every sentence."""

    def __init__(self, device):
        from vt.omni import NLIScorer
        self.s = NLIScorer()
        self.dev = device
        self.s.mc.to(device)
        self.hhem_dev = "cpu"
        try:
            self.s.hhem.to(device)
            self.s.hhem.predict([("The sky is blue.", "The sky is blue.")])
            self.hhem_dev = device
        except Exception as e:  # noqa: BLE001
            log("HHEM on GPU failed, CPU:", repr(e))
            self.s.hhem.to("cpu")

    def mc(self, doc, claim):
        import torch
        s = self.s
        w = doc.split()
        ch = [" ".join(w[i:i + s.chunk_words]) for i in range(0, max(1, len(w)), s.chunk_words)] or [""]
        x = s.mtok(ch, [claim] * len(ch), return_tensors="pt", padding=True, truncation="only_first", max_length=512)
        x = {k: v.to(self.dev) for k, v in x.items()}
        with torch.no_grad():
            p = torch.softmax(s.mc(**x).logits.float(), -1)[:, 1]
        return float(1 - p.max())

    def __call__(self, doc, question, reply):
        import torch
        from vt.omni import sentence_spans
        doc = doc[:12000]
        claim = f"{question[-300:]} {reply}" if len(reply.split()) < 8 else reply
        sents = [reply[a:b].strip() for a, b in sentence_spans(reply)] or [reply]
        per = [self.mc(doc, x) for x in sents[:30]] if len(sents) > 1 else [self.mc(doc, claim)]
        with torch.no_grad():
            h = 1 - float(self.s.hhem.predict([(doc, claim)])[0])
        return {"hhem": h, "mc4": max(per[:4]), "mc_all": max(per), "mc_mean": float(np.mean(per))}


def kernel(a):
    setup()
    sys.path[:0] = [SRC, os.path.join(SRC, "experiments")]
    rows, md5 = build_corpus()
    rows = [r for r in rows if r["domain"] in DOMAINS]
    if a.smoke:
        pick = []
        for d in DOMAINS:
            dr = [r for r in rows if r["domain"] == d]
            pick += [r for r in dr if r["label"] == 1][:(a.smoke + 1) // 2] + [r for r in dr if r["label"] == 0][:a.smoke // 2]
        rows = pick
    log(len(rows), "replies")
    # ---- retrieval thread (network; runs ahead of the GPU)
    wiki = make_wiki()
    ev, ready = {}, {r["id"]: threading.Event() for r in rows}

    def retrieve():
        for r in rows:
            t, n0 = time.time(), wiki.n_requests
            try:
                docs, qs = gather(wiki, r["messages"][-1]["content"], r["response"], r["domain"])
            except Exception as e:  # noqa: BLE001
                docs, qs = [], [repr(e)]
            ev[r["id"]] = {"docs": docs, "queries": qs, "t": time.time() - t, "requests": wiki.n_requests - n0}
            ready[r["id"]].set()

    th = threading.Thread(target=retrieve, daemon=True)
    th.start()
    # ---- models
    import torch
    from vt.omni import Reader, make_item
    rd = Reader("Qwen/Qwen3-8B")                             # layers 9 14 20 25 31 36, as heads/omni_qwen3_8b.npz
    assert rd.layers == [9, 14, 20, 25, 31, 36], rd.layers
    log("reader on", sorted({str(p.device) for p in rd.model.parameters()}))
    nli = GPUNLI("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    log("NLI ready; HHEM on", nli.hhem_dev)
    out = {k: [] for k in ("ids", "H_cb", "SC_cb", "H_doc", "SC_doc", "nli", "t_ret", "t_cb", "t_doc", "t_nli", "req",
                           "ndocs")}
    U = {"cb": [], "cb_sc": [], "cb_row": [], "doc": [], "doc_sc": [], "doc_row": []}

    def save():
        np.savez(os.path.join(WORK, "cb_scores.npz"), **{k: np.array(x) for k, x in out.items()},
                 **{f"U_{k}": (np.concatenate(x) if k in ("cb", "doc", "cb_sc", "doc_sc") and x else np.array(x)) for k, x in U.items()},
                 nli_keys=np.array(["hhem", "mc4", "mc_all", "mc_mean"]))
        json.dump({str(k): x for k, x in ev.items()}, open(os.path.join(WORK, "evidence.json"), "w"))
        json.dump({"errors": dict(wiki.errors), "requests": wiki.n_requests, "corpus_md5": md5, "n": len(out["ids"]),
                   "gpu": torch.cuda.get_device_name(0), "elapsed_min": (time.time() - T0) / 60},
                  open(os.path.join(WORK, "run_info.json"), "w"), indent=1)

    def sync():
        torch.cuda.synchronize()

    for k, r in enumerate(rows):
        t = time.time()
        f = rd.features(r)                              # closed book, exactly as the corpus was read
        sync()
        t_cb = time.time() - t
        ready[r["id"]].wait()
        e = ev[r["id"]]
        q = r["messages"][-1]["content"]
        it = make_item(r["response"], q, e["docs"], units=[tuple(u) for u in r["units"]])
        t = time.time()
        g = rd.features(it)
        sync()
        t_doc = time.time() - t
        t = time.time()
        n = nli("\n\n".join(e["docs"]), q, r["response"]) if e["docs"] else {"hhem": np.nan, "mc4": np.nan,
                                                                           "mc_all": np.nan, "mc_mean": np.nan}
        t_nli = time.time() - t
        if f is None or g is None:
            continue
        out["ids"].append(r["id"])
        for tag, ff in (("cb", f), ("doc", g)):              # the head's inputs, raw: (mean, judge) x layers LI
            out[f"H_{tag}"].append(np.stack([ff["r_mean"][LI], ff["j_state"][LI]]).astype(np.float16))
            out[f"SC_{tag}"].append(np.concatenate([ff["r_sc"], [ff["j_logit"]]]).astype(np.float32))
        out["nli"].append([n["hhem"], n["mc4"], n["mc_all"], n["mc_mean"]])
        out["t_ret"].append(e["t"])
        out["t_cb"].append(t_cb)
        out["t_doc"].append(t_doc)
        out["t_nli"].append(t_nli)
        out["req"].append(e["requests"])
        out["ndocs"].append(len(e["docs"]))
        for tag, ff in (("cb", f), ("doc", g)):
            if ff["units"]:
                U[tag].append(np.stack([u["mean"][LI] for u in ff["units"]]).astype(np.float16))
                U[f"{tag}_sc"].append(np.stack([u["sc"] for u in ff["units"]]).astype(np.float32))
            U[f"{tag}_row"] += [r["id"]] * len(ff["units"])
        if k % 25 == 0 or k == len(rows) - 1:
            save()
            log(f"{k + 1}/{len(rows)}", r["domain"], f"ret {e['t']:.1f}s ({e['requests']} req, {len(e['docs'])} docs)",
                f"cb {t_cb:.2f}s doc {t_doc:.2f}s nli {t_nli:.2f}s", "errors", dict(wiki.errors))
    save()
    log("done")


# ---------------------------------------------------------------- local analysis

def auroc(s, y):
    from scipy.stats import rankdata
    s, y = np.asarray(s, float), np.asarray(y)
    r = rankdata(s)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def zs(v):
    v = np.asarray(v, float)
    m = np.isnan(v)
    if m.all():
        return np.zeros_like(v)
    return np.where(m, 0.0, (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9))


def head_scores(art, X, Urows, Umat, ids):
    """zero-label: the domain's own replies standardise the head inputs (vt.omni OmniVerifier.profile)."""
    mu, sd = X.mean(0), X.std(0) + 1e-4
    r = ((X - mu) / sd) @ art["w"] + float(art["b"])
    keep = np.isin(Urows, ids)
    Urows, Umat = Urows[keep], Umat[keep]
    umu, usd = Umat.mean(0), Umat.std(0) + 1e-4
    us = ((Umat - umu) / usd) @ art["uw"] + float(art["ub"])
    u = np.array([us[Urows == i].max() if (Urows == i).any() else np.nan for i in ids])
    return r, u


def analyse(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from vt.omni import OmniVerifier
    from vt.retrieval import norm, words
    art = OmniVerifier.load(a.head).art
    S = np.load(os.path.join(a.analyse, "cb_scores.npz"))
    ev = json.load(open(os.path.join(a.analyse, "evidence.json")))
    rows = {}
    for line in open(a.corpus):
        r = json.loads(line)
        if r["domain"] in DOMAINS:
            rows[r["id"]] = r
    E = np.load(a.expert, allow_pickle=True)
    epos = {int(i): k for k, i in enumerate(E["ids"])}
    ids = S["ids"].astype(int)
    from vt.omni import _project
    X, UX = {}, {}                                           # head inputs (OmniVerifier.reply_vector / unit_matrix)
    for tag in ("cb", "doc"):
        H = S[f"H_{tag}"].astype(np.float32).reshape(len(ids), -1)
        X[tag] = np.concatenate([_project(H, art["h_mu"], art["h_sd"], art["pca_mu"], art["pca_V"]), S[f"SC_{tag}"]], 1)
        Uh = S[f"U_{tag}"].astype(np.float32).reshape(len(S[f"U_{tag}_row"]), -1)
        UX[tag] = np.concatenate([_project(Uh, art["uh_mu"], art["uh_sd"], art["upca_mu"], art["upca_V"]),
                                  S[f"U_{tag}_sc"]], 1)
    dom = np.array([rows[i]["domain"] for i in ids])
    y = np.array([rows[i]["label"] for i in ids])
    nk = list(S["nli_keys"])
    NL = S["nli"]
    best = {}                                                # truthfulqa: question -> correct answer
    for r in rows.values():
        if r["domain"] == "truthfulqa" and r["label"] == 0:
            best[r["messages"][-1]["content"]] = r["response"]
    rep = {"n": int(len(ids)), "per_domain": {}, "lodo_weights": {}}
    feats_by_dom = {}
    for d in DOMAINS:
        m = dom == d
        if m.sum() < 4 or len(set(y[m])) < 2:
            continue
        di = ids[m]
        ex = {}
        k = [epos[i] for i in di]
        ex["cb_lodo_reply"], ex["cb_lodo_unit"], ex["cb_lodo_judge"] = E["reply"][k], E["worst_unit"][k], E["judge"][k]
        for tag in ("cb", "doc"):
            r_, u_ = head_scores(art, X[tag][m], S[f"U_{tag}_row"], UX[tag], di)
            ex[f"{tag}_reply"], ex[f"{tag}_unit"], ex[f"{tag}_judge"] = r_, u_, -S[f"SC_{tag}"][m, -1]
        for j, kk in enumerate(nk):
            ex[f"nli_{kk}"] = NL[m, j]
        ex["nli"] = (NL[m, nk.index("hhem")] + NL[m, nk.index("mc4")]) / 2           # NLIScorer as shipped
        ex["nli_all"] = (NL[m, nk.index("hhem")] + NL[m, nk.index("mc_all")]) / 2
        Z = {k2: zs(v2) for k2, v2 in ex.items()}
        comb = {
            "a_closed_book_lodo (r+u+2j)": Z["cb_lodo_reply"] + Z["cb_lodo_unit"] + 2 * Z["cb_lodo_judge"],
            "a'_closed_book_shipped_head (in-sample)": Z["cb_reply"] + Z["cb_unit"] + 2 * Z["cb_judge"],
            "b_reader_with_docs (r+u+2j)": Z["doc_reply"] + Z["doc_unit"] + 2 * Z["doc_judge"],
            "b_judge_with_docs": Z["doc_judge"],
            "c_nli (shipped)": Z["nli"],
            "c_nli_all_sentences": Z["nli_all"],
            "d_lodo_cb + nli": Z["cb_lodo_reply"] + Z["cb_lodo_unit"] + 2 * Z["cb_lodo_judge"] + Z["nli"],
            "d_lodo_cb + docs_judge": Z["cb_lodo_reply"] + Z["cb_lodo_unit"] + 2 * Z["cb_lodo_judge"] + 2 * Z["doc_judge"],
            "d_lodo_cb + docs_judge + nli": Z["cb_lodo_reply"] + Z["cb_lodo_unit"] + 2 * Z["cb_lodo_judge"]
                                            + 2 * Z["doc_judge"] + Z["nli"],
            "d_docs_reader + nli": Z["doc_reply"] + Z["doc_unit"] + 2 * Z["doc_judge"] + Z["nli"],
            "d_docs_judge + nli": 2 * Z["doc_judge"] + Z["nli"],
        }
        pd = {"n": int(m.sum()), "wrong": int(y[m].sum())}
        pd["experts"] = {k2: round(auroc(np.nan_to_num(v2, nan=np.nanmean(v2) if not np.isnan(v2).all() else 0), y[m]), 4)
                         for k2, v2 in ex.items()}
        pd["combined"] = {k2: round(auroc(v2, y[m]), 4) for k2, v2 in comb.items()}
        # retrieval diagnostics
        ndocs = S["ndocs"][m]
        pd["retrieval"] = {"empty_evidence": float((ndocs == 0).mean()), "mean_passages": float(ndocs.mean()),
                           "mean_requests": float(S["req"][m].mean()), "s_retrieval_mean": float(S["t_ret"][m].mean()),
                           "s_retrieval_median": float(np.median(S["t_ret"][m]))}
        if d == "truthfulqa":                                # does the evidence contain the correct answer's words?
            hit = []
            for i in di:
                q = rows[i]["messages"][-1]["content"]
                w = set(words(best.get(q, ""))) - set(words(q))
                txt = set(norm(" ".join(ev[str(i)]["docs"])).split())
                hit.append(len(w & txt) / len(w) if w else np.nan)
            hit = np.array(hit)
            pd["retrieval"]["answer_word_recall_mean"] = float(np.nanmean(hit))
            pd["retrieval"]["answer_words_all_found"] = float(np.nanmean(hit >= 0.999))
        pd["seconds"] = {k2: float(S[k2][m].mean()) for k2 in ("t_ret", "t_cb", "t_doc", "t_nli")}
        rep["per_domain"][d] = pd
        feats_by_dom[d] = (np.stack([Z[k2] for k2 in ("cb_lodo_reply", "cb_lodo_unit", "cb_lodo_judge", "doc_reply",
                                                         "doc_unit", "doc_judge", "nli")], 1), y[m])
    # learned weights, leave one of the 4 domains out (logistic on the z-scored experts)
    if len(feats_by_dom) >= 2:
        from vt.omni import LogReg, balanced_weights
        names = ["cb_lodo_reply", "cb_lodo_unit", "cb_lodo_judge", "doc_reply", "doc_unit", "doc_judge", "nli"]
        sets = {"d_learned_lodo4 (all experts; doc head in-sample)": list(range(7)),
                "d_learned_lodo4 (closed-book LODO + docs judge + NLI)": [0, 1, 2, 5, 6],
                "d_learned_lodo4 (closed-book LODO only: reweighting control)": [0, 1, 2]}
        for d in feats_by_dom:
            Xtr = np.concatenate([feats_by_dom[o][0] for o in feats_by_dom if o != d])
            ytr = np.concatenate([feats_by_dom[o][1] for o in feats_by_dom if o != d]).astype(float)
            for nm, cols in sets.items():
                mdl = LogReg(0.1).fit(Xtr[:, cols], ytr, balanced_weights(ytr))
                rep["per_domain"][d]["combined"][nm] = round(auroc(mdl.score(feats_by_dom[d][0][:, cols]),
                                                                   feats_by_dom[d][1]), 4)
                rep["lodo_weights"].setdefault(nm, {})[d] = dict(zip([names[c] for c in cols],
                                                                     [round(float(x), 3) for x in mdl.w]))
    keys = list(next(iter(rep["per_domain"].values()))["combined"])
    rep["mean"] = {k2: round(float(np.mean([p["combined"][k2] for p in rep["per_domain"].values()])), 4) for k2 in keys}
    ek = list(next(iter(rep["per_domain"].values()))["experts"])
    rep["mean_experts"] = {k2: round(float(np.mean([p["experts"][k2] for p in rep["per_domain"].values()])), 4) for k2 in ek}
    rep["seconds_per_reply"] = {k2: round(float(S[k2].mean()), 3) for k2 in ("t_ret", "t_cb", "t_doc", "t_nli")}
    rep["spot_check_deciding_fact"] = SPOT_CHECK
    info = os.path.join(a.analyse, "run_info.json")
    if os.path.exists(info):
        rep["run_info"] = json.load(open(info))
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        json.dump(rep, open(a.report, "w"), indent=1)
    for d, p in rep["per_domain"].items():
        print(d, p["n"], json.dumps(p["combined"]))
        print("   experts", json.dumps(p["experts"]))
        print("   retrieval", json.dumps(p["retrieval"]), "seconds", json.dumps(p["seconds"]))
    print("mean", json.dumps(rep["mean"], indent=1))
    print("mean experts", json.dumps(rep["mean_experts"]))
    print("s/reply", rep["seconds_per_reply"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="replies per domain (half wrong)")
    ap.add_argument("--analyse", help="directory with cb_scores.npz + evidence.json (local analysis)")
    ap.add_argument("--expert", default="results/omni/expert_scores_q8b.npz")
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--head", default="heads/omni_qwen3_8b.npz")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.analyse:
        return analyse(a)
    kernel(a)


if __name__ == "__main__":
    main()
