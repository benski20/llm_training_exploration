"""Mechanistic-interpretability verification signals, ideas 1, 2, 3, 5(attention), 6 of
docs/research/mech_interp_verification_2026.md in ONE feature-extraction pass per reply, plus idea 4 (a
different-family peer reader) as a separate check.

READER. Qwen/Qwen3-1.7B, bf16, CPU, max_tokens 1024. Each reply is read once as
    [documents +] question + reply + judge suffix ("Is the reply above fully correct, accurate and supported?")
with output_hidden_states and a KV cache; the cache is then cropped back to the end of the reply and five more judge
questions are read on it (a few dozen tokens each). Attention probabilities are recomputed exactly (from the layer's
own q/k projections, q/k norms and rotary embeddings, in a forward pre-hook) on 4 middle layers (12, 15, 18, 21 of
28), so the model itself runs with its fast attention.

SAMPLE. 24 wrong + 24 correct replies per domain (seed 0) among replies with stored 1.7B features: 1,008 replies.
For localisation only (idea 5), extra ProcessBench wrong replies (up to --pb-wrong per domain, same seed) are read.

SIGNALS (a-priori sign: larger = "wrong"; stated before the first full run)
 idea 1  logit lens at the judge position. At every layer l = 1..28, for the residual x_l, the attention output a_l
         and the MLP output m_l (hooks): final RMSNorm + lm_head; kappa_l = logsumexp(Yes ids) - logsumexp(No ids)
         and the entropy of the full softmax (3 x 28 x 2 = 168 scalars; x_28 is the real output).
         label-free summaries: -mean kappa_res over layers 12-22 (PRIMARY), -min kappa_res over 12-22,
         commit depth (first layer after which sign(kappa_res) never changes; later = wrong), sign flips from
         layer 8 (more = wrong), kappa_28 - mean kappa_12-22 (late layers overturning a mid "No" = wrong),
         mean residual entropy over 12-22 (higher = wrong). LODO logistic on the 168-d vector, against the judge-
         state probe (judge residual, layers 15/20/24, PCA 128) on the same replies.
 idea 2  question-only state: hidden state at the token before the reply (assistant header, "hdr") and at the last
         token of the user turn ("user"), layers 15/20/24; shift = judge state - hdr state and reply-mean state -
         hdr state. Each block PCA 128 (unsupervised), LODO. Duplicate questions counted first (a question that
         appears with both labels makes the prior uninformative by construction); reported on unpaired domains.
 idea 3  6 judge framings on the cached prefix: F1 the shipped question; F2 "Does the reply above contain any
         error?" (polarity reversed); F3 "Would an expert accept this reply as correct?"; F4 "Is every claim in the
         reply supported by the question and the documents?"; F5 "Is the final answer in the reply right?";
         F6 "Is the reply above wrong?" (reversed). c_k = signed correctness log-odds (reversed framings negated).
         mean (= -mean c_k), variance of the per-domain z-scored c_k (fragility), polarity inconsistency
         sigmoid(kappa_F1) + sigmoid(kappa_F2) - 1 (says "correct" and "contains an error" at once), variance across
         the same-polarity framings F1/F3/F4/F5 of a LODO judge-state probe score (layer 20). LODO on all of them.
 idea 5  judge-row attention per reply unit (mean over the unit's tokens) on the 4 layers x 16 heads.
         localisation (ProcessBench wrong replies with one labelled first wrong step): argmax unit of the head-mean;
         argmax of the 8 heads with the largest within-reply variance across units (label-free choice); the single
         head chosen on the other three ProcessBench domains; the shipped rule (first unit with z > 0.75 on the
         LODO unit head); the shipped rule on z(unit head) + z(attention) (tau chosen on the other domains).
         detection: share of the largest unit, normalised entropy, agreement of the attention top unit with the
         unit head's suspect, judge-row mass on the reply.
 idea 6  LapEigvals: per head, the diagonal of the attention graph Laplacian (in-degree normalised by the number of
         later tokens minus self-attention), top 20 over reply rows + top 5 over judge-question rows, 4 x 16 x 25,
         per-domain standardised, PCA 128, LODO. Lookback ratio per head (context mass / (context + reply mass)),
         mean over reply rows and at the judge row: 128 features, LODO, also on RAG domains alone.

BASELINE. The 1.7B reply head (stored /home/user/omni/q17 features, omni_final.py recipe: mean + judge state at
layers 15/20/24, PCA 256, output scalars, judge log-odds; LODO with per-domain standardisation on all 5,107 replies),
restricted to the same replies. Nested heads (never saw the held-out domain nor the row's own domain) feed stacking.

ADDITIONS. sum = z(reply) + z(signal) (z within domain, unlabelled statistics, zero labels);
           stack = LODO logistic on [z(nested reply head), z(signal block)] (an extra column in the logistic).

PRE-STATED GATE (fixed before the first full run): an idea is USEFUL only if sum or stack raises the mean AUROC over
the 21 domains above the reply head by >= +0.005 with the paired 95% bootstrap CI (1,000 resamples of replies within
each domain, stratified by label) above 0. A format-level claim needs the same within that format. Idea 5
localisation: GO if exact first-wrong-step accuracy gains >= +3 points over the shipped rule with the CI above 0.
Idea 6 is killed if its LODO alone < 0.60 or its addition < +0.005. Idea 4: the LODO test runs only if linear CKA and
the cross-validated linear correlation between the two readers' reply states are both < 0.95 (CKA: debiased,
since plain CKA is biased toward 1 with 6,144 dims and ~100 replies; plain CKA reported too), on states
standardised within each domain (as the heads see them); the global version is reported too.

    python experiments/novel_mech_bundle.py extract --cache <dir> --threads 4
    python experiments/novel_mech_bundle.py analyse --cache <dir> --report results/omni/novel_mech_bundle.json
    python experiments/novel_mech_bundle.py peer --cache <dir> --peer-cache <dir2> [--full]
    (smoke: extract --limit 10 --threads 1)
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omni_corpus import load_features  # noqa: E402
from omni_eval import LogReg, auroc, balanced_weights, lodo, pca_fit, standardise  # noqa: E402
from novel_consensus_dynamics import evaluate, nested_heads, reply_X, stack, unit_X, zdom  # noqa: E402

T0 = time.time()
Q17 = "/home/user/omni/q17"
CORPUS = "/home/user/omni/corpus.jsonl"
ATT_LAYERS = [11, 14, 17, 20]          # 0-based decoder blocks = layers 12, 15, 18, 21 of 28
MID = [2, 3, 4]                        # indices into the stored layers [7, 11, 15, 20, 24, 28] -> 15, 20, 24
FRAMINGS = [("F1 shipped", "Is the reply above fully correct, accurate and supported? Answer Yes or No.", +1),
            ("F2 contains error", "Does the reply above contain any error? Answer Yes or No.", -1),
            ("F3 expert accepts", "Would an expert accept this reply as correct? Answer Yes or No.", +1),
            ("F4 claims supported", "Is every claim in the reply supported by the question and the documents? "
                                    "Answer Yes or No.", +1),
            ("F5 final answer right", "Is the final answer in the reply right? Answer Yes or No.", +1),
            ("F6 is wrong", "Is the reply above wrong? Answer Yes or No.", -1)]
TOPK_REPLY, TOPK_SUFFIX = 20, 5


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def load_rows():
    rows = {}
    for line in open(CORPUS):
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def pick_subset(avail, rows, per_class, seed):
    rng = np.random.default_rng(seed)
    out = []
    for d in sorted({r["domain"] for r in rows.values()}):
        for lab in (0, 1):
            pool = sorted(i for i, r in rows.items() if r["domain"] == d and r["label"] == lab and i in avail)
            out += rng.choice(pool, min(per_class, len(pool)), replace=False).tolist()
    return sorted(int(i) for i in out)


def pick_pb_extra(avail, rows, main, per_dom, seed):
    """extra ProcessBench wrong replies with exactly one labelled first wrong step (localisation only)."""
    rng = np.random.default_rng(seed + 1)
    out = []
    for d in ("pb_gsm8k", "pb_math", "pb_olympiad", "pb_omnimath"):
        have = [i for i in main if rows[i]["domain"] == d and rows[i]["label"] == 1]
        pool = sorted(i for i, r in rows.items() if r["domain"] == d and r["label"] == 1 and i in avail
                      and i not in set(main) and sum(u == 1 for u in r["unit_labels"]) == 1)
        k = max(0, min(per_dom - len(have), len(pool)))
        out += rng.choice(pool, k, replace=False).tolist() if k else []
    return sorted(int(i) for i in out)


def question_key(r):
    return r["messages"][-1]["content"]


# ----------------------------------------------------------------- extraction

class Bundle:
    def __init__(self, name, max_tokens, threads):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vt.omni import _template
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
        cfg = self.model.config
        self.nL, self.nH, self.max_tokens = cfg.num_hidden_layers, cfg.num_attention_heads, max_tokens
        self.layers_store = sorted({int(round(f * self.nL)) for f in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)})
        ids = lambda w: self.tok(w, add_special_tokens=False)["input_ids"][0]   # noqa: E731
        self.yes = [ids(w) for w in ("Yes", "yes", " Yes")]
        self.no = [ids(w) for w in ("No", "no", " No")]
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        mark = "⁣REPLY⁣"
        self.suffixes = []
        for _, q, _ in FRAMINGS:
            t = _template(self.tok, [{"role": "user", "content": "Q"}, {"role": "assistant", "content": mark},
                                     {"role": "user", "content": q}])
            self.suffixes.append(self.tok(t[t.index(mark) + len(mark):], add_special_tokens=False)["input_ids"])
        self.active = False
        self.ctx = {}
        self.cap = {}
        blocks = self.model.model.layers
        for li, blk in enumerate(blocks):
            blk.self_attn.register_forward_hook(self._hook_out("a", li))
            blk.mlp.register_forward_hook(self._hook_out("m", li))
        for li in ATT_LAYERS:
            blocks[li].self_attn.register_forward_pre_hook(self._hook_att(li), with_kwargs=True)

    def _hook_out(self, kind, li):
        def f(mod, inp, out):
            if self.active:
                o = out[0] if isinstance(out, tuple) else out
                self.cap[(kind, li)] = o[0, -1].detach()
        return f

    def _hook_att(self, li):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        torch = self.torch

        def f(mod, args, kwargs):
            if not self.active:
                return
            h = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            N = h.shape[1]
            hs = (1, N, -1, mod.head_dim)
            q = mod.q_norm(mod.q_proj(h).view(hs)).transpose(1, 2)
            k = mod.k_norm(mod.k_proj(h).view(hs)).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = k.repeat_interleave(mod.num_key_value_groups, 1)
            s = (q[0].float() @ k[0].float().transpose(-1, -2)) * mod.scaling           # H x N x N
            s.masked_fill_(torch.ones(N, N, dtype=torch.bool).triu(1), float("-inf"))
            A = s.softmax(-1)
            self.cap[("att", li)] = self.att_features(A)
        return f

    def att_features(self, A):
        """A: H x N x N attention probabilities of one layer."""
        torch = self.torch
        c = self.ctx
        r0, n, N = c["r0"], c["n"], A.shape[-1]
        # judge row per unit: mean attention over the unit's tokens
        jrow = A[:, -1]                                                               # H x N
        K = c["n_units"]
        unit = np.full((A.shape[0], K), np.nan, np.float32)
        tu = c["tu"]
        for k_ in range(K):
            m = np.where(tu == k_)[0]
            if len(m):
                unit[:, k_] = jrow[:, r0 + m].mean(-1).numpy()
        j_ctx, j_rep = jrow[:, :r0].sum(-1), jrow[:, r0:r0 + n].sum(-1)
        # Laplacian diagonal: d_i = sum_{u >= i} A[u, i] / (N - i); L_ii = d_i - A_ii
        deg = A.sum(1) / torch.arange(N, 0, -1, dtype=torch.float32)                  # H x N
        diag = deg - A.diagonal(dim1=-2, dim2=-1)
        rep = diag[:, r0:r0 + n]
        suf = diag[:, r0 + n:]
        kr, ks = min(TOPK_REPLY, rep.shape[1]), min(TOPK_SUFFIX, suf.shape[1])
        lap = torch.zeros(A.shape[0], TOPK_REPLY + TOPK_SUFFIX)
        lap[:, :kr] = rep.topk(kr, -1).values
        lap[:, TOPK_REPLY:TOPK_REPLY + ks] = suf.topk(ks, -1).values
        # lookback ratio over reply rows: context mass / (context + reply-so-far mass)
        R = A[:, r0:r0 + n]
        ctx = R[:, :, :r0].sum(-1)
        new = R[:, :, r0:].sum(-1)
        lb = (ctx / (ctx + new + 1e-9)).mean(-1)
        return {"unit": unit, "lap": lap.numpy().astype(np.float32), "lb_reply": lb.numpy().astype(np.float32),
                "lb_judge": (j_ctx / (j_ctx + j_rep + 1e-9)).numpy().astype(np.float32),
                "j_rep_mass": j_rep.numpy().astype(np.float32)}

    def lens(self, vecs):
        """vecs: (m, d) pre-norm vectors -> (kappa, entropy) through final norm + lm_head."""
        torch = self.torch
        lp = self.model.lm_head(self.model.model.norm(vecs)).float().log_softmax(-1)
        kap = torch.logsumexp(lp[:, self.yes], -1) - torch.logsumexp(lp[:, self.no], -1)
        ent = -(lp.exp() * lp).sum(-1)
        return kap.numpy(), ent.numpy()

    def read(self, r):
        from vt.omni import encode, token_units
        torch = self.torch
        suf1 = self.suffixes[0]
        ids, r0, roff = encode(self.tok, r, self.max_tokens, suf1)
        ns = len(suf1)
        n = len(ids) - r0 - ns
        if n <= 0:
            return None
        tu = token_units(roff[:n], r["units"])
        ends = [i for i in range(r0) if ids[i] == self.im_end]
        q_user = ends[-1] - 1 if ends else r0 - 1
        self.ctx = {"r0": r0, "n": n, "tu": tu, "n_units": len(r["units"])}
        self.cap = {}
        self.active = True
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=True,
                           logits_to_keep=1)
            self.active = False
            hs = o.hidden_states
            Ls = self.layers_store
            st = lambda p: torch.stack([hs[L][0, p] for L in Ls]).float().numpy().astype(np.float16)  # noqa: E731
            out = {"id": int(r["id"]), "n": n, "r0": r0, "n_units": len(r["units"]),
                   "unit_ntok": np.array([(tu == k).sum() for k in range(len(r["units"]))], np.int32),
                   "q_hdr": st(r0 - 1), "q_user": st(q_user), "j_state": st(-1),
                   "r_mean": torch.stack([hs[L][0, r0:r0 + n].mean(0) for L in Ls]).float().numpy().astype(np.float16)}
            # idea 1: logit lens at the judge position
            nL = self.nL
            res = torch.stack([hs[L][0, -1] for L in range(1, nL)])                    # pre-norm, layers 1..27
            att = torch.stack([self.cap[("a", li)] for li in range(nL)])
            mlp = torch.stack([self.cap[("m", li)] for li in range(nL)])
            kap, ent = self.lens(torch.cat([res, att, mlp]))
            fin = o.logits[0, -1].float().log_softmax(-1)
            k28 = float(torch.logsumexp(fin[self.yes], 0) - torch.logsumexp(fin[self.no], 0))
            e28 = float(-(fin.exp() * fin).sum())
            kres = np.concatenate([kap[:nL - 1], [k28]])
            eres = np.concatenate([ent[:nL - 1], [e28]])
            out["kappa"] = np.stack([kres, kap[nL - 1:2 * nL - 1], kap[2 * nL - 1:]]).astype(np.float32)
            out["entropy"] = np.stack([eres, ent[nL - 1:2 * nL - 1], ent[2 * nL - 1:]]).astype(np.float32)
            # idea 5 / 6: attention
            for key in ("unit", "lap", "lb_reply", "lb_judge", "j_rep_mass"):
                out["att_" + key] = np.stack([self.cap[("att", li)][key] for li in ATT_LAYERS])
            # idea 3: other framings on the cached prefix
            cache = o.past_key_values
            P = len(ids) - ns
            kf = [k28]
            fstate = [torch.stack([hs[Ls[i]][0, -1] for i in MID]).float().numpy().astype(np.float16)]
            for suf in self.suffixes[1:]:
                extra = cache.get_seq_length() - P
                if extra > 0:
                    cache.crop(-extra)
                o2 = self.model(input_ids=torch.tensor([suf]), past_key_values=cache, use_cache=True,
                                output_hidden_states=True, logits_to_keep=1)
                lp = o2.logits[0, -1].float().log_softmax(-1)
                kf.append(float(torch.logsumexp(lp[self.yes], 0) - torch.logsumexp(lp[self.no], 0)))
                fstate.append(torch.stack([o2.hidden_states[Ls[i]][0, -1] for i in MID]).float().numpy()
                              .astype(np.float16))
            out["kappa_framings"] = np.array(kf, np.float32)
            out["state_framings"] = np.stack(fstate)                                     # 6 x 3 x d
        return out


def load_cache(d):
    done = {}
    for p in sorted(glob.glob(os.path.join(d, "part_*.pkl"))):
        for x in pickle.load(open(p, "rb")):
            done[x["id"]] = x
    return done


def cmd_extract(a):
    rows = load_rows()
    F = load_features(Q17)
    avail = set(int(i) for i in F["ids"])
    del F
    sub = pick_subset(avail, rows, a.per_class, a.seed)
    extra = pick_pb_extra(avail, rows, sub, a.pb_wrong, a.seed)
    todo_all = sub + extra
    if a.limit:
        rng = np.random.default_rng(a.seed)
        pbw = [i for i in sub if rows[i]["domain"].startswith("pb_") and rows[i]["label"] == 1]
        todo_all = sorted(set(rng.choice(sub, a.limit - 2, replace=False).tolist() + pbw[:2]))
    os.makedirs(a.cache, exist_ok=True)
    done = load_cache(a.cache)
    todo = [i for i in todo_all if i not in done]
    log(f"subset {len(sub)} + pb extra {len(extra)}; cached {len(done)}; to read {len(todo)}")
    if not todo:
        return
    B = Bundle(a.reader, a.max_tokens, a.threads)
    part = len(glob.glob(os.path.join(a.cache, "part_*.pkl")))
    buf, t0 = [], time.time()
    for k, i in enumerate(todo):
        x = B.read(rows[i])
        if x is not None:
            buf.append(x)
        if len(buf) >= a.shard or k == len(todo) - 1:
            pickle.dump(buf, open(os.path.join(a.cache, f"part_{part:04d}.pkl"), "wb"))
            part, buf = part + 1, []
            el = time.time() - t0
            log(f"{k + 1}/{len(todo)} read ({el / (k + 1):.1f} s/reply, eta {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min)")


# ----------------------------------------------------------------- analysis helpers

def pca_block(H, k, dom=None):
    """unsupervised projection: (per-domain) standardise, PCA k (fitted on all rows; no labels)."""
    H = standardise(H.astype(np.float32), dom) if dom is not None else standardise(H.astype(np.float32))
    mu, V = pca_fit(H, min(k, H.shape[0] - 1, H.shape[1]))
    return (H - mu) @ V.T


def first_cross(s, t):
    hit = np.where(s > t)[0]
    return int(hit[0]) if len(hit) else int(np.argmax(s))


def boot_mean_ci(vals_by_dom, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    doms = list(vals_by_dom)
    bs = [np.mean([vals_by_dom[d][rng.integers(0, len(vals_by_dom[d]), len(vals_by_dom[d]))].mean() for d in doms])
          for _ in range(B)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def cmd_analyse(a):
    rows = load_rows()
    done = load_cache(a.cache)
    F = load_features(Q17)
    sub = pick_subset(set(int(i) for i in F["ids"]), rows, a.per_class, a.seed)
    ids = np.array([i for i in sub if i in done])
    log(f"analysis: {len(ids)} replies of the main sample cached ({len(done)} in cache)")
    X_ = {k: np.stack([done[i][k] for i in ids]) for k in
          ("kappa", "entropy", "q_hdr", "q_user", "j_state", "r_mean", "kappa_framings", "state_framings",
           "att_lap", "att_lb_reply", "att_lb_judge", "att_j_rep_mass")}
    dom = np.array([rows[i]["domain"] for i in ids])
    y = np.array([rows[i]["label"] for i in ids])
    doms = sorted(set(dom.tolist()))
    fmt_of = {d: rows[int(ids[dom == d][0])]["format"] for d in doms}
    n = len(ids)
    rep = {"reader": a.reader, "n": int(n), "domains": len(doms), "max_tokens": a.max_tokens,
           "per_class_per_domain": a.per_class, "bootstrap": a.boot, "att_layers_1based": [l + 1 for l in ATT_LAYERS],
           "framings": [f[1] for f in FRAMINGS]}

    # ---- duplicate questions (idea 2 caveat), counted before any idea-2 result
    dup = {}
    for d in doms:
        allq = [question_key(r) for r in rows.values() if r["domain"] == d]
        cnt = {}
        for q in allq:
            cnt[q] = cnt.get(q, 0) + 1
        m = ids[dom == d]
        labs_by_q = {}
        for i in m:
            labs_by_q.setdefault(question_key(rows[int(i)]), set()).add(rows[int(i)]["label"])
        dup[d] = {"corpus_frac_dup": round(float(np.mean([cnt[q] > 1 for q in allq])), 3),
                  "sample_frac_both_labels": round(float(np.mean([len(labs_by_q[question_key(rows[int(i)])]) > 1
                                                                  for i in m])), 3)}
    unpaired = [d for d in doms if dup[d]["corpus_frac_dup"] < 0.10]
    rep["duplicates"] = dup
    rep["unpaired_domains"] = unpaired
    rep["unpaired_domains_in_sample"] = [d for d in doms if dup[d]["sample_frac_both_labels"] < 0.10]
    log(f"unpaired domains ({len(unpaired)}): {unpaired}")

    # ---- baseline: the 1.7B reply head on all stored replies (LODO + nested), restricted to the sample
    pos = {int(i): k for k, i in enumerate(F["ids"])}
    sel = np.array([pos[int(i)] for i in ids])
    domF = np.array([rows[int(i)]["domain"] for i in F["ids"]])
    yF = np.array([rows[int(i)]["label"] for i in F["ids"]])
    _, XF = reply_X(F, a.pca)
    _, sF = lodo(XF, yF, domF, a.l2, True)
    reply = sF[sel]
    S = nested_heads(standardise(XF, domF), yF, domF, doms, a.l2)[:, sel]
    judge_stored = -F["j_logit"][sel].astype(float)
    log("reply head done")

    # ---- unit head (stored unit features, LODO) for idea 5
    UX = unit_X(F, a.pca)
    u_row, u_idx, u_lab = F["u_row"].astype(int), F["u_idx"].astype(int), F["u_lab"]
    udom = domF[np.array([pos[int(r_)] for r_ in u_row])]
    UXs = standardise(UX, udom)
    uscore = np.zeros(len(u_row))
    lab = u_lab >= 0
    for d in doms:
        te = udom == d
        tr = ~te & lab
        m = LogReg(a.l2).fit(UXs[tr], u_lab[tr].astype(float), balanced_weights(u_lab[tr].astype(float)))
        uscore[te] = m.score(UXs[te])
    uz = zdom(uscore, udom)                                   # z by the domain's own unit statistics (unlabelled)
    umap = {(int(r_), int(k_)): float(z_) for r_, k_, z_ in zip(u_row, u_idx, uz)}
    del UX, UXs, XF, F
    log("unit head done")

    methods, comps, blocks = {"reply": reply}, [], {}

    def add_block(name, X, pca=None, per_dom=True):
        """LODO on a feature block; sum and stack with the reply head."""
        Xb = pca_block(X, pca, dom) if pca else X.astype(float)
        _, s = lodo(Xb, y, dom, a.l2, per_dom)
        methods[f"{name}"] = s
        methods[f"reply+{name} sum"] = zdom(reply, dom) + zdom(s, dom)
        methods[f"reply+{name} stack"] = stack([S, standardise(Xb, dom)], y, dom, doms, a.l2)
        comps.extend([(f"reply+{name} sum", "reply"), (f"reply+{name} stack", "reply")])
        blocks[name] = Xb
        return s

    def add_scalar(name, v):
        """a label-free scalar with an a-priori sign: alone and z-summed with the reply head."""
        methods[name] = v
        methods[f"reply+{name} sum"] = zdom(reply, dom) + zdom(v, dom)
        comps.append((f"reply+{name} sum", "reply"))

    # ---- idea 1
    kap, ent = X_["kappa"].astype(float), X_["entropy"].astype(float)
    band = slice(11, 22)                                      # layers 12..22
    kres = kap[:, 0]
    sgn = np.sign(kres)
    commit = np.array([next((l for l in range(28) if np.all(sgn[i, l:] == sgn[i, -1])), 27) + 1 for i in range(n)],
                      float)
    flips = (np.diff(sgn[:, 7:], axis=1) != 0).sum(1).astype(float)
    i1 = {"i1_neg_mid_kappa": -kres[:, band].mean(1), "i1_neg_min_mid_kappa": -kres[:, band].min(1),
          "i1_commit_depth": commit, "i1_flips": flips, "i1_final_minus_mid": kres[:, -1] - kres[:, band].mean(1),
          "i1_mid_entropy": ent[:, 0, band].mean(1)}
    methods["judge_final (this pass)"] = -kres[:, -1]
    methods["judge_final (stored)"] = judge_stored
    add_scalar("i1_neg_mid_kappa", i1["i1_neg_mid_kappa"])
    for k_, v in i1.items():
        if k_ != "i1_neg_mid_kappa":
            methods[k_] = v
    comps.append(("i1_neg_mid_kappa", "judge_final (this pass)"))
    lens168 = np.concatenate([kap.reshape(n, -1), ent.reshape(n, -1)], 1)
    add_block("i1_lens168", lens168)
    jst = X_["j_state"][:, MID].reshape(n, -1)
    add_block("jstate_probe", jst, pca=128)
    _, s_jl = lodo(np.concatenate([pca_block(jst, 128, dom), lens168], 1), y, dom, a.l2, True)
    methods["jstate+lens168"] = s_jl
    comps += [("i1_lens168", "jstate_probe"), ("jstate+lens168", "jstate_probe")]
    per_dom_mid_beats = {d: auroc(i1["i1_neg_mid_kappa"][dom == d], y[dom == d]) >
                         auroc(-kres[dom == d, -1], y[dom == d]) for d in doms}
    rep["idea1_mid_beats_final_domains"] = int(sum(per_dom_mid_beats.values()))
    rep["idea1_kappa_profile"] = {g: {"correct": np.round(kap[(y == 0), j].mean(0), 3).tolist(),
                                      "wrong": np.round(kap[(y == 1), j].mean(0), 3).tolist()}
                                  for j, g in enumerate(["resid", "attn", "mlp"])}
    log("idea 1 done")

    # ---- idea 2
    qh = X_["q_hdr"][:, MID].reshape(n, -1).astype(np.float32)
    qu = X_["q_user"][:, MID].reshape(n, -1).astype(np.float32)
    rm = X_["r_mean"][:, MID].reshape(n, -1).astype(np.float32)
    add_block("i2_prior_hdr", qh, pca=128)
    add_block("i2_prior_user", qu, pca=128)
    add_block("i2_shift_judge", jst.astype(np.float32) - qh, pca=128)
    add_block("i2_shift_reply", rm - qh, pca=128)
    log("idea 2 done")

    # ---- idea 3
    kf = X_["kappa_framings"].astype(float)
    pol = np.array([f[2] for f in FRAMINGS], float)
    c = kf * pol                                              # signed correctness log-odds
    cz = zdom(c, dom)
    sig = lambda v: 1 / (1 + np.exp(-v))                      # noqa: E731
    add_scalar("i3_mean", -c.mean(1))
    add_scalar("i3_mean_z", -cz.mean(1))
    add_scalar("i3_fragility", cz.var(1))
    add_scalar("i3_polarity_incons", sig(kf[:, 0]) + sig(kf[:, 1]) - 1)
    methods["i3_F1_alone"] = -c[:, 0]
    for k_ in range(1, len(FRAMINGS)):
        methods[f"i3_{FRAMINGS[k_][0]}"] = -c[:, k_]
    comps += [("i3_mean", "i3_F1_alone"), ("i3_mean_z", "i3_F1_alone")]
    # probe-score variance across same-polarity framings (layer 20 judge state; LODO probe fitted on F1 states)
    SF = X_["state_framings"][:, :, 1].astype(np.float32)    # n x 6 x d (layer 20)
    same = [0, 2, 3, 4]
    pv = np.zeros(n)
    for d in doms:
        te = dom == d
        mu, V = pca_fit(standardise(SF[~te, 0]), 64)
        msd = (SF[~te, 0].mean(0), SF[~te, 0].std(0) + 1e-4)
        P0 = ((SF[:, 0] - msd[0]) / msd[1] - mu) @ V.T
        P0s = standardise(P0, dom)
        m = LogReg(a.l2).fit(P0s[~te], y[~te].astype(float), balanced_weights(y[~te].astype(float)))
        sc = []
        for k_ in same:
            Pk = ((SF[te, k_] - msd[0]) / msd[1] - mu) @ V.T
            Pk = (Pk - Pk.mean(0)) / (Pk.std(0) + 1e-4)
            sc.append(m.score(Pk))
        pv[te] = np.var(np.stack(sc), 0)
    add_scalar("i3_probe_var", pv)
    add_block("i3_lodo", np.concatenate([c, cz.var(1)[:, None], (sig(kf[:, 0]) + sig(kf[:, 1]) - 1)[:, None],
                                         pv[:, None]], 1))
    # least-confident 20% of the reader's judge (pooled after within-domain z)
    lowc = np.zeros(n, bool)
    for d in doms:
        m = np.where(dom == d)[0]
        a_ = np.abs(kf[m, 0] - np.median(kf[m, 0]))
        lowc[m[np.argsort(a_)[:max(2, int(round(0.2 * len(m))))]]] = True
    rep["idea3_least_confident_20pct_pooled_auroc"] = {
        nm: round(auroc(zdom(methods[nm], dom)[lowc], y[lowc]), 4)
        for nm in ("i3_F1_alone", "i3_mean", "i3_fragility", "i3_polarity_incons", "i3_probe_var", "reply")}
    rep["idea3_framing_corr"] = np.round(np.mean([np.corrcoef(c[dom == d].T) for d in doms], 0), 3).tolist()
    log("idea 3 done")

    # ---- idea 6
    lap = X_["att_lap"].reshape(n, -1).astype(np.float32)
    add_block("i6_lapeig", lap, pca=128)
    lb = np.concatenate([X_["att_lb_reply"].reshape(n, -1), X_["att_lb_judge"].reshape(n, -1)], 1)
    add_block("i6_lookback", lb)
    log("idea 6 done")

    # ---- idea 5 detection features
    conc, nent, agree, jmass = np.zeros(n), np.zeros(n), np.zeros(n), X_["att_j_rep_mass"].reshape(n, -1).mean(1)
    for k_, i in enumerate(ids):
        U = done[int(i)]["att_unit"].reshape(-1, done[int(i)]["n_units"]).mean(0)
        ok = np.isfinite(U)
        if ok.sum() < 2:
            conc[k_], nent[k_] = 1.0, 0.0
            continue
        p = U[ok] / U[ok].sum()
        conc[k_] = p.max()
        nent[k_] = float(-(p * np.log(p + 1e-12)).sum() / np.log(ok.sum()))
        zu = np.array([umap.get((int(i), u), np.nan) for u in range(len(U))])
        okz = ok & np.isfinite(zu)
        if okz.sum() >= 2:
            idx = np.where(okz)[0]
            agree[k_] = float(idx[np.argmax(U[idx])] == idx[first_cross(zu[idx], 0.75)])
    add_scalar("i5_concentration", conc)
    add_block("i5_det", np.stack([conc, nent, agree, jmass], 1))
    log("idea 5 detection done")

    # ---- evaluation
    for nm in list(methods):
        methods[nm] = np.nan_to_num(np.asarray(methods[nm], float))
    ev = evaluate(methods, y, dom, fmt_of, doms, a.boot, comps)
    # unpaired-domain means for idea 2
    di = {d: k for k, d in enumerate(doms)}
    rep["idea2_unpaired_mean"] = {nm: round(float(np.mean([ev["per_domain"][nm][d] for d in unpaired])), 4)
                                  for nm in methods if nm.startswith(("i2_", "reply+i2_")) or nm == "reply"}
    rep["results"] = ev
    gate = {}
    for g, v in ev["diff"].items():
        gate[g] = {c_: bool(x[0] >= 0.005 and x[1] > 0) for c_, x in v.items() if c_.startswith("reply+")}
    rep["gate_pass"] = gate
    log("evaluation done")

    # ---- idea 5 localisation (main sample + extra ProcessBench wrong replies)
    rep["idea5_localisation"] = localise(a, rows, done, umap)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    rep["ids"] = [int(i) for i in ids]
    json.dump(rep, open(a.report, "w"), indent=1)
    groups = list(ev["mean"])
    print(f"{'method':38s} " + " ".join(f"{g[:9]:>21s}" for g in groups))
    for nm in methods:
        print(f"{nm:38s} " + " ".join("{:.3f} [{:.3f},{:.3f}]".format(*ev['mean'][g][nm]) for g in groups))
    for g in groups:
        print("DIFF", g, json.dumps(ev["diff"][g]))
    print("GATE all:", json.dumps({k: v for k, v in gate["all"].items() if v}))
    print("idea2 unpaired:", json.dumps(rep["idea2_unpaired_mean"]))
    print("idea1 mid beats final on", rep["idea1_mid_beats_final_domains"], "domains")
    print("idea3 least confident:", json.dumps(rep["idea3_least_confident_20pct_pooled_auroc"]))
    print("idea5 localisation:", json.dumps(rep["idea5_localisation"], indent=1))
    log("done")


def localise(a, rows, done, umap):
    pb = ["pb_gsm8k", "pb_math", "pb_olympiad", "pb_omnimath"]
    cases = {d: [] for d in pb}
    for i, x in done.items():
        r = rows[i]
        if r["domain"] not in cases or r["label"] != 1 or sum(u == 1 for u in r["unit_labels"]) != 1:
            continue
        g = r["unit_labels"].index(1)
        K = x["n_units"]
        ok = (x["unit_ntok"] > 0) & np.isfinite(x["att_unit"][0, 0])
        zu = np.array([umap.get((i, u), np.nan) for u in range(K)])
        ok &= np.isfinite(zu)
        if not ok[g] or ok.sum() < 2:
            continue
        idx = np.where(ok)[0]
        A = x["att_unit"][:, :, idx].reshape(-1, len(idx))     # 64 heads x units
        cases[r["domain"]].append({"g": int(np.where(idx == g)[0][0]), "A": A, "zu": zu[idx]})
    nH = 64
    # label-free head choice: the heads whose judge-row unit mass varies most within replies (all cases, no labels)
    allc = [c for d in pb for c in cases[d]]
    hv = np.mean([np.var(c["A"] / (c["A"].sum(1, keepdims=True) + 1e-12), 1) * c["A"].shape[1] for c in allc], 0)
    top8 = np.argsort(hv)[-8:]

    def zr(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    def pred_head(c, h):
        return int(np.argmax(c["A"][h]))

    rules = {"chance": None,
             "unit_head_argmax": lambda c, p: int(np.argmax(c["zu"])),
             "shipped_first_cross_0.75": lambda c, p: first_cross(c["zu"], 0.75),
             "att_mean_heads": lambda c, p: int(np.argmax(c["A"].mean(0))),
             "att_top8_var_heads": lambda c, p: int(np.argmax(c["A"][top8].mean(0))),
             "att_best_head_xdomain": lambda c, p: pred_head(c, p["head"]),
             "first_cross(z_unit + z_att)": lambda c, p: first_cross(c["zu"] + zr(c["A"][top8].mean(0)), p["tau"])}
    grid = np.round(np.arange(0.0, 3.01, 0.25), 2)
    params = {}
    for d in pb:
        oth = [e for e in pb if e != d]
        hacc = [np.mean([np.mean([pred_head(c, h) == c["g"] for c in cases[e]]) for e in oth]) for h in range(nH)]
        tacc = [np.mean([np.mean([first_cross(c["zu"] + zr(c["A"][top8].mean(0)), t) == c["g"] for c in cases[e]])
                         for e in oth]) for t in grid]
        params[d] = {"head": int(np.argmax(hacc)), "tau": float(grid[int(np.argmax(tacc))])}
    hits, hits1 = {}, {}
    for nm, f in rules.items():
        hits[nm], hits1[nm] = {}, {}
        for d in pb:
            if nm == "chance":
                hits[nm][d] = np.array([1 / len(c["zu"]) for c in cases[d]])
                hits1[nm][d] = np.array([(min(c["g"] + 1, len(c["zu"]) - 1) - max(c["g"] - 1, 0) + 1) / len(c["zu"])
                                         for c in cases[d]])
                continue
            p = [f(c, params[d]) for c in cases[d]]
            hits[nm][d] = np.array([pi == c["g"] for pi, c in zip(p, cases[d])], float)
            hits1[nm][d] = np.array([abs(pi - c["g"]) <= 1 for pi, c in zip(p, cases[d])], float)
    out = {"n_cases": {d: len(cases[d]) for d in pb}, "params": params, "top8_heads": top8.tolist(),
           "exact": {}, "within1": {}, "diff_vs_shipped": {}}
    rng = np.random.default_rng(0)
    samples = [{d: rng.integers(0, len(cases[d]), len(cases[d])) for d in pb} for _ in range(a.boot)]
    for nm in rules:
        for key, H in (("exact", hits), ("within1", hits1)):
            mval = float(np.mean([H[nm][d].mean() for d in pb]))
            bs = [np.mean([H[nm][d][s[d]].mean() for d in pb]) for s in samples]
            out[key][nm] = [round(mval, 4), round(float(np.percentile(bs, 2.5)), 4),
                            round(float(np.percentile(bs, 97.5)), 4)]
            if nm not in ("chance", "shipped_first_cross_0.75"):
                bd = [np.mean([(H[nm][d][s[d]] - H["shipped_first_cross_0.75"][d][s[d]]).mean() for d in pb])
                      for s in samples]
                out["diff_vs_shipped"][f"{key}: {nm}"] = [
                    round(float(np.mean([(H[nm][d] - H["shipped_first_cross_0.75"][d]).mean() for d in pb])), 4),
                    round(float(np.percentile(bd, 2.5)), 4), round(float(np.percentile(bd, 97.5)), 4)]
    out["go"] = {k: bool(v[0] >= 0.03 and v[1] > 0) for k, v in out["diff_vs_shipped"].items() if k.startswith("exact")}
    return out


# ----------------------------------------------------------------- idea 4: a different-family peer reader

def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def debiased_cka(A, B):
    """linear CKA with the unbiased HSIC estimator (plain CKA is biased toward 1 when dims >> replies)."""
    def hsic(K, L):
        n = len(K)
        K, L = K.copy(), L.copy()
        np.fill_diagonal(K, 0)
        np.fill_diagonal(L, 0)
        one = np.ones(n)
        return (np.trace(K @ L) + (one @ K @ one) * (one @ L @ one) / ((n - 1) * (n - 2))
                - 2 / (n - 2) * (one @ K @ L @ one)) / (n * (n - 3))
    K, L = A @ A.T, B @ B.T
    return float(hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L)))


def sims(A, B, dom=None):
    A, B = standardise(A, dom), standardise(B, dom)
    pa, pb = pca_fit(A, 16), pca_fit(B, 16)
    return {"cka": round(linear_cka(A, B), 4), "cka_debiased": round(debiased_cka(A, B), 4),
            "cka_pca16": round(linear_cka((A - pa[0]) @ pa[1].T, (B - pb[0]) @ pb[1].T), 4),
            "cv_corr_b_from_a": round(cv_linear_corr(A, B)[0], 4), "cv_corr_a_from_b": round(cv_linear_corr(B, A)[0], 4)}


def cv_linear_corr(A, B, k=16, lam=1.0, folds=5, seed=0):
    """predict B's top-k principal components from A (ridge, k-fold); mean held-out correlation, variance-weighted."""
    A = standardise(A)
    B = standardise(B)
    mu, V = pca_fit(B, k)
    Pb = (B - mu) @ V.T
    pred = np.zeros_like(Pb)
    idx = np.random.default_rng(seed).permutation(len(A))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        Am, Bm = A[tr].mean(0), Pb[tr].mean(0)
        At = A[tr] - Am
        W = At.T @ np.linalg.solve(At @ At.T + lam * len(tr) * np.eye(len(tr)), Pb[tr] - Bm)   # dual ridge
        pred[te] = (A[te] - Am) @ W + Bm
    r = np.array([np.corrcoef(pred[:, j], Pb[:, j])[0, 1] for j in range(Pb.shape[1])])
    w = Pb.var(0)
    return float((r * w).sum() / w.sum()), [round(float(x), 3) for x in r[:5]]


def cmd_peer(a):
    from vt.omni import Reader
    rows = load_rows()
    done = load_cache(a.cache)
    F = load_features(Q17)
    sub = pick_subset(set(int(i) for i in F["ids"]), rows, a.per_class, a.seed)
    sub = [i for i in sub if i in done]
    doms_ = sorted({rows[i]["domain"] for i in sub})
    rng = np.random.default_rng(a.seed + 2)
    probe_ids = []
    for d in doms_:
        pool = [i for i in sub if rows[i]["domain"] == d]
        probe_ids += rng.choice(pool, min(5, len(pool)), replace=False).tolist()
    probe_ids = sorted(int(x) for x in probe_ids)
    todo_ids = sub if a.full else probe_ids
    os.makedirs(a.peer_cache, exist_ok=True)
    pf = os.path.join(a.peer_cache, "peer.pkl")
    peer = pickle.load(open(pf, "rb")) if os.path.exists(pf) else {}
    todo = [i for i in todo_ids if i not in peer]
    log(f"peer {a.peer}: {len(todo_ids)} replies, {len(todo)} to read")
    if todo:
        R = Reader(a.peer, max_tokens=a.max_tokens, threads=a.threads)
        log(f"peer layers {R.layers}")
        t0 = time.time()
        for k, i in enumerate(todo):
            f = R.features(rows[i])
            if f is not None:
                peer[i] = {"r_mean": f["r_mean"].astype(np.float16), "j_state": f["j_state"].astype(np.float16),
                           "j_logit": f["j_logit"], "r_sc": f["r_sc"]}
            if (k + 1) % 25 == 0 or k == len(todo) - 1:
                pickle.dump(peer, open(pf, "wb"))
                el = time.time() - t0
                log(f"{k + 1}/{len(todo)} ({el / (k + 1):.1f} s/reply)")
    rep_path = a.report
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    # similarity on the ~100-reply probe set
    sim_ids = [i for i in probe_ids if i in peer]
    Q = np.stack([done[i]["r_mean"][MID].reshape(-1) for i in sim_ids]).astype(np.float32)
    G = np.stack([peer[i]["r_mean"][MID].reshape(-1) for i in sim_ids]).astype(np.float32)
    F06 = load_features("/home/user/omni/q06")
    pos = {int(i): k for k, i in enumerate(F["ids"])}
    s_ = np.array([pos[i] for i in sim_ids])
    Q17s = F["r_mean"][s_][:, MID].reshape(len(s_), -1).astype(np.float32)
    Q06s = F06["r_mean"][s_][:, MID].reshape(len(s_), -1).astype(np.float32)
    sdom = np.array([rows[i]["domain"] for i in sim_ids])
    sim = {"n": len(sim_ids), "qwen17_vs_peer": sims(Q, G),
           "qwen17_vs_qwen06 (reference, stored)": sims(Q17s, Q06s),
           "qwen17_vs_peer, within-domain": sims(Q, G, sdom),
           "qwen17_vs_qwen06 (reference), within-domain": sims(Q17s, Q06s, sdom)}
    qp = sim["qwen17_vs_peer, within-domain"]          # what the per-domain-standardised heads see (pre-stated)
    sim["run_lodo"] = bool(qp["cka_debiased"] < 0.95 and max(qp["cv_corr_b_from_a"], qp["cv_corr_a_from_b"]) < 0.95)
    log(json.dumps(sim))
    rep["idea4_peer"] = {"peer": a.peer, "similarity": sim}
    if a.full:
        ids = np.array([i for i in sub if i in peer])
        dom = np.array([rows[i]["domain"] for i in ids])
        y = np.array([rows[i]["label"] for i in ids])
        doms = sorted(set(dom.tolist()))
        fmt_of = {d: rows[int(ids[dom == d][0])]["format"] for d in doms}
        domF = np.array([rows[int(i)]["domain"] for i in F["ids"]])
        yF = np.array([rows[int(i)]["label"] for i in F["ids"]])
        _, XF = reply_X(F, a.pca)
        _, sF = lodo(XF, yF, domF, a.l2, True)
        sel = np.array([pos[int(i)] for i in ids])
        reply = sF[sel]
        S = nested_heads(standardise(XF, domF), yF, domF, doms, a.l2)[:, sel]
        jq = -F["j_logit"][sel].astype(float)
        n = len(ids)
        H = np.concatenate([np.stack([peer[i]["r_mean"][MID].reshape(-1) for i in ids]),
                            np.stack([peer[i]["j_state"][MID].reshape(-1) for i in ids])], 1).astype(np.float32)
        Hp = pca_block(H, 128, dom)
        XP = np.concatenate([Hp, np.stack([peer[i]["r_sc"] for i in ids]),
                             np.array([peer[i]["j_logit"] for i in ids])[:, None]], 1)
        _, peer_head = lodo(XP, y, dom, a.l2, True)
        jp = -np.array([peer[i]["j_logit"] for i in ids])
        m = {"reply (qwen)": reply, "judge (qwen)": jq, "peer head": peer_head, "peer judge": jp}
        m["qwen reply+judge sum"] = zdom(reply, dom) + zdom(jq, dom)
        m["+peer head+peer judge sum"] = m["qwen reply+judge sum"] + zdom(peer_head, dom) + zdom(jp, dom)
        m["reply+peer head sum"] = zdom(reply, dom) + zdom(peer_head, dom)
        m["reply+peer stack"] = stack([S, standardise(XP, dom)], y, dom, doms, a.l2)
        comps = [("reply+peer head sum", "reply (qwen)"), ("reply+peer stack", "reply (qwen)"),
                 ("+peer head+peer judge sum", "qwen reply+judge sum"), ("peer judge", "judge (qwen)")]
        ev = evaluate(m, y, dom, fmt_of, doms, a.boot, comps)
        rep["idea4_peer"]["n"] = int(n)
        rep["idea4_peer"]["results"] = ev
        for g in ev["mean"]:
            print(g, {k: v for k, v in ev["mean"][g].items()})
            print("DIFF", g, ev["diff"][g])
    json.dump(rep, open(rep_path, "w"), indent=1)
    log("peer done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "analyse", "peer"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--report", default="results/omni/novel_mech_bundle.json")
    ap.add_argument("--reader", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--peer", default="ibm-granite/granite-3.3-2b-instruct")
    ap.add_argument("--peer-cache", default="")
    ap.add_argument("--full", action="store_true", help="peer: LODO run on the whole sample")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--pb-wrong", type=int, default=75, help="ProcessBench wrong replies per domain for localisation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    {"extract": cmd_extract, "analyse": cmd_analyse, "peer": cmd_peer}[a.cmd](a)


if __name__ == "__main__":
    main()
