"""T6: attack on 'token obfuscation' (secret vocabulary permutation; the server's model is permuted to match,
so inference is near full speed). The attacker sees only permuted token IDs and has public text in the same
language. Attack: unigram-frequency matching, then iterative refinement with bigram-context similarity
(substitution-cipher cryptanalysis). Metric: fraction of the victim's token occurrences decoded correctly."""
import json, sys, glob, numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from scipy.optimize import linear_sum_assignment
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"); V = len(tok)
ld = lambda f: "\n".join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", f, repo_type="dataset"))["text"])
public = ld("wikitext-103-raw-v1/train-00000-of-00002.parquet")[:60_000_000]
victims = {"wikitext2_heldout (same domain)": ld("wikitext-2-raw-v1/test-00000-of-00001.parquet") + ld("wikitext-2-raw-v1/validation-00000-of-00001.parquet"),
           "repo_docs_markdown (different domain)": "\n".join(open(f).read() for f in glob.glob("../../docs/*.md") + glob.glob("../../*.md"))}
enc = lambda s: np.array(tok(s).input_ids, dtype=np.int64)
pub = enc(public)
rng = np.random.default_rng(1); perm = rng.permutation(V)  # secret key: plain id -> cipher id
K, A = 3000, 400
def stats(ids, top):
    c = np.bincount(ids, minlength=V)
    return c
def ctx(ids, vocab_top, anchor_ids, idmap):
    # context vectors over anchors: counts of left and right neighbours that are anchors
    pos = {a: j for j, a in enumerate(anchor_ids)}
    row = {t: i for i, t in enumerate(vocab_top)}
    X = np.zeros((len(vocab_top), 2 * len(anchor_ids)))
    l, r = ids[:-1], ids[1:]
    for a, b in zip(l, r):
        if b in row and a in pos: X[row[b], pos[a]] += 1
        if a in row and b in pos: X[row[a], len(anchor_ids) + pos[b]] += 1
    X = np.log1p(X); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X
R = {"public_reference_tokens": int(len(pub)), "vocab": V, "K_top_types_attacked": K}
pc = np.bincount(pub, minlength=V); ptop = np.argsort(-pc)[:K]
pfreq = np.log(pc[ptop] / pc.sum())
for name, text in victims.items():
    vid = enc(text); cid = perm[vid]                     # what the server sees
    cc = np.bincount(cid, minlength=V); ctop = np.argsort(-cc)[:K]; ctop = ctop[cc[ctop] > 0]
    guess = dict(zip(ctop, ptop[:len(ctop)]))            # round 0: frequency rank
    inv_true = np.argsort(perm)                          # for scoring only
    def score(g):
        dec = np.array([g.get(t, -1) for t in cid]); return float((dec == vid).mean())
    hist = [score(guess)]
    Xp = ctx(pub[:8_000_000], ptop, list(ptop[:A]), None)
    cfreq = np.log(cc[ctop] / cc.sum())
    for it in range(6):
        inv = {p: c for c, p in guess.items()}
        anchors_c = [inv.get(p, -1) for p in ptop[:A]]
        Xc = ctx(cid, ctop, anchors_c, None)
        S = Xc @ Xp.T - 0.15 * np.abs(cfreq[:, None] - pfreq[None, :])   # context similarity + frequency prior
        ri, ci = linear_sum_assignment(-S)
        guess = {ctop[i]: ptop[j] for i, j in zip(ri, ci)}
        hist.append(score(guess))
    types_ok = float(np.mean([guess[c] == inv_true[c] for c in ctop[:500]]))
    sample = vid[2000:2060]; dec = np.array([guess.get(t, -1) for t in perm[sample]])
    R[name] = {"victim_tokens": int(len(vid)), "occurrence_recovery_by_round": hist,
               "top500_types_correct": types_ok,
               "true_snippet": tok.decode(sample),
               "attacker_decoded_snippet": tok.decode([d if d >= 0 else tok.encode("?")[0] for d in dec])}
    print(name, json.dumps(R[name], indent=1)); sys.stdout.flush()
json.dump(R, open("results/t6_permutation_attack.json", "w"), indent=1)
