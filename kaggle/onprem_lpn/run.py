"""Kaggle CPU kernel: replicate client-aided private inference with dual-LPN-style masking (Maverick-like)
on Qwen2.5-0.5B. The SERVER only sees x_hat = x + G·e (mod p) for every linear layer; G is public, e is a fresh
sparse random vector (t nonzeros) per row. The CLIENT unmasks with precomputed P = W·G: y = W·x_hat - P·e.
Client does embeddings lookup, RMSNorm, RoPE, attention (keeps KV cache), SiLU, residuals.
Checks: (1) exactness vs a plaintext int8-weight reference, (2) quality of int8 vs fp32, (3) can an inverter read
server views, (4) costs: client/server time, rounds, bytes, client storage and memory traffic; scaling to 8B/70B."""
import json, time, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
T0 = time.time(); R = {"cpu_count": os.cpu_count()}
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
def save(): json.dump(R, open("/kaggle/working/results.json", "w"), indent=1)
torch.set_num_threads(os.cpu_count())
# ---------------- field arithmetic mod p = 2^61 - 1 (uint64, vectorised) ----------------
p = (1 << 61) - 1; P61 = np.uint64(p); M32 = np.uint64(0xFFFFFFFF); U = np.uint64
def red(x): x = (x & P61) + (x >> U(61)); return np.where(x >= P61, x - P61, x)
def shl(a, j):                       # a * 2^j mod p, a < p
    if j == 0: return a
    return red(((a & U((1 << (61 - j)) - 1)) << U(j)) + (a >> U(61 - j)))
def mulmod(a, b):
    a0, a1, b0, b1 = a & M32, a >> U(32), b & M32, b >> U(32)
    r = red(a0 * b0); s = shl(red(a0 * b1 + a1 * b0), 32); u = red(a1 * b1 * U(8))
    return red(red(r + s) + u)
def addmod(a, b): return red(a + b)
def submod(a, b): return red(a + (P61 - b))
def field_matmul(Wf, X):            # Wf: float64 int8-valued (d_out x n); X: uint64 (T x n) -> (T x d_out) = X W^T mod p
    acc = None
    for k in range(4):
        L = ((X >> U(16 * k)) & U(0xFFFF)).astype(np.float64)
        Ri = np.rint(L @ Wf.T).astype(np.int64) % np.int64(p)
        t = shl(Ri.astype(np.uint64), 16 * k); acc = t if acc is None else addmod(acc, t)
    return acc
def field_matmul_right(Wf, G):      # Wf (d_out x n) float64 int8-valued, G (n x N) uint64 -> (W G)^T as (N x d_out)
    acc = None
    for k in range(4):
        Gk = ((G >> U(16 * k)) & U(0xFFFF)).astype(np.float64)
        Ri = np.rint(Gk.T @ Wf.T).astype(np.int64) % np.int64(p)
        t = shl(Ri.astype(np.uint64), 16 * k); acc = t if acc is None else addmod(acc, t)
    return acc
# self-tests against Python big ints
rng = np.random.default_rng(0)
a = rng.integers(0, p, 1000, dtype=np.uint64); b = rng.integers(0, p, 1000, dtype=np.uint64)
assert all(int(x) == (int(y) * int(z)) % p for x, y, z in zip(mulmod(a, b), a, b)), "mulmod"
assert all(int(x) == (int(y) << 37) % p for x, y in zip(shl(a, 37), a)), "shl"
Wt = rng.integers(-127, 128, (5, 7)).astype(np.float64); Xt = rng.integers(0, p, (3, 7), dtype=np.uint64)
ref = [[sum(int(Wt[o, i]) * int(Xt[r, i]) for i in range(7)) % p for o in range(5)] for r in range(3)]
assert (field_matmul(Wt, Xt).astype(object) == np.array(ref, dtype=object)).all(), "field_matmul"
log("field self-tests passed")
# ---------------- model ----------------
M = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(M)
hf = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval(); cfg = hf.config
H, NH, KV, HD, L = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_hidden_layers
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
def q8(W):                           # per-output-row symmetric int8
    s = np.abs(W).max(1, keepdims=True) / 127.0; s[s == 0] = 1; return np.rint(W / s), s[:, 0]
LIN = {}                             # name -> (Wq float64, scale, bias)
for i in range(L):
    pre = f"model.layers.{i}."
    Wqkv = np.concatenate([sd[pre + "self_attn.q_proj.weight"], sd[pre + "self_attn.k_proj.weight"], sd[pre + "self_attn.v_proj.weight"]])
    bqkv = np.concatenate([sd[pre + "self_attn.q_proj.bias"], sd[pre + "self_attn.k_proj.bias"], sd[pre + "self_attn.v_proj.bias"]])
    LIN[f"{i}.qkv"] = (*q8(Wqkv), bqkv); LIN[f"{i}.o"] = (*q8(sd[pre + "self_attn.o_proj.weight"]), None)
    LIN[f"{i}.gu"] = (*q8(np.concatenate([sd[pre + "mlp.gate_proj.weight"], sd[pre + "mlp.up_proj.weight"]])), None)
    LIN[f"{i}.down"] = (*q8(sd[pre + "mlp.down_proj.weight"]), None)
LIN["head"] = (*q8(sd["lm_head.weight"] if "lm_head.weight" in sd else sd["model.embed_tokens.weight"]), None)
E = sd["model.embed_tokens.weight"]; I = cfg.intermediate_size
for k in [k for k in sd if "proj" in k or k == "lm_head.weight"]: del sd[k]
import gc; gc.collect()
ACT_BITS = 20
def quant_act(X):                    # per-row fixed point
    s = np.abs(X).max(1, keepdims=True) / (2 ** ACT_BITS - 1); s[s == 0] = 1; return np.rint(X / s).astype(np.int64), s
# ---------------- protocol state ----------------
T_SPARSE = 160; RATE = 0.5
G = {n: rng.integers(0, p, (n, int(n / RATE)), dtype=np.uint64) for n in {H, I}}   # public codes
GT = {n: np.ascontiguousarray(g.T) for n, g in G.items()}                           # (N x n) for row gathers
STATS = {"rounds": 0, "bytes_up": 0, "bytes_down": 0, "client_s": 0.0, "server_s": 0.0, "client_bytes_touched": 0}
SERVER_VIEWS = {}                    # name -> list of x_hat rows (for the security test)
PT = {}
def precompute():
    t = time.time(); tot = 0
    for name, (Wq, s, b) in LIN.items():
        n = Wq.shape[1]; PT[name] = field_matmul_right(Wq, G[n]); tot += PT[name].nbytes
    R["client_precompute"] = {"seconds": round(time.time() - t, 1), "storage_GB": round(tot / 1e9, 3),
        "int8_weights_GB": round(sum(w.size for w, _, _ in LIN.values()) / 1e9, 3)}
    log("precompute", R["client_precompute"]); save()
def sparse_mask(n, T):
    N = G[n].shape[1]
    idx = np.stack([rng.choice(N, T_SPARSE, replace=False) for _ in range(T)])
    val = rng.integers(1, p, (T, T_SPARSE), dtype=np.uint64)
    return idx, val
def combine(M_T, idx, val):          # sum_j M_T[idx[:, j]] * val[:, j]  (rows of M_T), per row -> (T x d)
    acc = np.zeros((idx.shape[0], M_T.shape[1]), dtype=np.uint64)
    for j in range(idx.shape[1]):
        acc = addmod(acc, mulmod(M_T[idx[:, j]], val[:, j:j + 1]))
    return acc
def linear(name, X, mode, keep_view=False):
    Wq, s, b = LIN[name]; xq, sx = quant_act(X)
    if mode == "plain":
        y_int = np.rint(xq.astype(np.float64) @ Wq.T)
    else:
        n, T = Wq.shape[1], X.shape[0]
        t = time.time()
        xf = (xq % np.int64(p)).astype(np.uint64); idx, val = sparse_mask(n, T)
        xh = addmod(xf, combine(GT[n], idx, val)); STATS["client_s"] += time.time() - t
        if keep_view: SERVER_VIEWS.setdefault(name, []).append(xh)
        t = time.time(); yh = field_matmul(Wq, xh); STATS["server_s"] += time.time() - t     # SERVER
        t = time.time(); yf = submod(yh, combine(PT[name], idx, val))
        y_int = np.where(yf > U(p // 2), -(P61 - yf).astype(np.float64), yf.astype(np.float64))
        STATS["client_s"] += time.time() - t
        STATS["rounds"] += 1; STATS["bytes_up"] += xh.nbytes; STATS["bytes_down"] += yh.nbytes
        STATS["client_bytes_touched"] += T * T_SPARSE * (GT[n].shape[1] + PT[name].shape[1]) * 8
    y = y_int * s[None, :] * sx
    return y + b[None, :] if b is not None else y
def rms(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + cfg.rms_norm_eps) * w
inv = 1.0 / (cfg.rope_theta ** (np.arange(0, HD, 2) / HD))
def rope(x, pos):                    # x (T, h, HD)
    f = pos[:, None] * inv[None]; c, s = np.cos(np.concatenate([f, f], -1))[:, None], np.sin(np.concatenate([f, f], -1))[:, None]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]; return x * c + np.concatenate([-x2, x1], -1) * s
def forward(ids, cache, mode, last_only=True, keep_views=False):
    T = len(ids); start = cache["len"]; pos = np.arange(start, start + T, dtype=np.float64)
    x = E[ids]
    for i in range(L):
        pre = f"model.layers.{i}."; t = time.time()
        h = rms(x, sd[pre + "input_layernorm.weight"])
        qkv = linear(f"{i}.qkv", h, mode, keep_views and i in (1, 12))
        t = time.time()
        q = rope(qkv[:, :H].reshape(T, NH, HD), pos); k = rope(qkv[:, H:H + KV * HD].reshape(T, KV, HD), pos)
        v = qkv[:, H + KV * HD:].reshape(T, KV, HD)
        cache.setdefault(i, [np.zeros((0, KV, HD)), np.zeros((0, KV, HD))])
        cache[i][0] = np.concatenate([cache[i][0], k]); cache[i][1] = np.concatenate([cache[i][1], v])
        K = np.repeat(cache[i][0], NH // KV, 1); V = np.repeat(cache[i][1], NH // KV, 1)
        sc = np.einsum("thd,shd->hts", q, K) / np.sqrt(HD)
        mask = np.arange(K.shape[0])[None, :] > (start + np.arange(T))[:, None]; sc[:, mask] = -1e30
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        a = np.einsum("hts,shd->thd", sc, V).reshape(T, H)
        if mode == "lpn": STATS["client_s"] += time.time() - t
        x = x + linear(f"{i}.o", a, mode)
        t = time.time(); h = rms(x, sd[pre + "post_attention_layernorm.weight"])
        if mode == "lpn": STATS["client_s"] += time.time() - t
        gu = linear(f"{i}.gu", h, mode); t = time.time()
        m = gu[:, :I] / (1 + np.exp(-gu[:, :I])) * gu[:, I:]
        if mode == "lpn": STATS["client_s"] += time.time() - t
        x = x + linear(f"{i}.down", m, mode)
    cache["len"] += T
    h = rms(x[-1:] if last_only else x, sd["model.norm.weight"])
    return linear("head", h, mode)
def generate(prompt_ids, n_new, mode):
    cache = {"len": 0}; out = []; lg_all = []; t = time.time()
    lg = forward(prompt_ids, cache, mode); t_prefill = time.time() - t; times = []
    for _ in range(n_new):
        nxt = int(lg[-1].argmax()); out.append(nxt); lg_all.append(lg[-1])
        t = time.time(); lg = forward(np.array([nxt]), cache, mode); times.append(time.time() - t)
    return out, np.array(lg_all), t_prefill, float(np.mean(times))
# ---------------- 1. quality: fp32 vs int8 path (plaintext) ----------------
note = ("Counselor note: Maria Lopez, grade 8, student ID S482913, told me she has been thinking about hurting herself "
        "since her parents separated. Her mother Ana Lopez works nights. Write a two-sentence follow-up plan.")
ids = tok.apply_chat_template([{"role": "user", "content": note}], add_generation_prompt=True, tokenize=True)
if isinstance(ids, dict): ids = ids["input_ids"]
ids = np.array(ids); NEW = 24
with torch.no_grad():
    g = hf.generate(torch.tensor([ids]), max_new_tokens=NEW, do_sample=False)[0, len(ids):].tolist()
R["fp32_hf_answer"] = tok.decode(g)
t = time.time(); out_plain, lg_plain, tp_plain, td_plain = generate(ids, NEW, "plain")
R["int8_plain"] = {"answer": tok.decode(out_plain), "same_tokens_as_fp32": int(sum(a == b for a, b in zip(out_plain, g))), "of": NEW,
                   "prefill_s": round(tp_plain, 2), "decode_s_per_token": round(td_plain, 3)}
with torch.no_grad():
    t = time.time(); hf.generate(torch.tensor([ids]), max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False); R["fp32_hf_s_per_token_incl_prefill"] = round((time.time() - t) / NEW, 3)
log("plain", R["int8_plain"]); save()
del hf; gc.collect()
# ---------------- 2. protocol ----------------
precompute()
STATS.update({k: 0 for k in STATS})
cache = {"len": 0}; t = time.time(); lg = forward(ids, cache, "lpn"); pre_stats = dict(STATS); t_pf = time.time() - t
out_lpn = []; lgs = []; dec_t = []
for _ in range(NEW):
    nxt = int(lg[-1].argmax()); out_lpn.append(nxt); lgs.append(lg[-1])
    s0 = dict(STATS); t = time.time(); lg = forward(np.array([nxt]), cache, "lpn"); dec_t.append(time.time() - t)
per_tok = {k: (STATS[k] - pre_stats[k]) / NEW for k in STATS}
R["lpn_protocol"] = {"answer": tok.decode(out_lpn), "identical_tokens_to_int8_plain": out_lpn == out_plain,
    "max_abs_logit_diff_vs_int8_plain": float(np.abs(np.array(lgs) - lg_plain).max()),
    "prompt_tokens": int(len(ids)), "prefill_s": round(t_pf, 1), "decode_s_per_token_total": round(float(np.mean(dec_t)), 3),
    "per_decode_token": {"rounds": per_tok["rounds"], "MB_up": round(per_tok["bytes_up"] / 1e6, 3), "MB_down": round(per_tok["bytes_down"] / 1e6, 3),
                         "client_s": round(per_tok["client_s"], 3), "server_s": round(per_tok["server_s"], 3),
                         "client_MB_touched": round(per_tok["client_bytes_touched"] / 1e6, 1)},
    "t_sparse": T_SPARSE, "code_rate": RATE, "field": "2^61-1", "act_bits": ACT_BITS}
pt = R["lpn_protocol"]["per_decode_token"]
R["latency_model_s_per_token"] = {f"rtt_{r}ms": round(pt["client_s"] + pt["server_s"] + pt["rounds"] * r / 1000, 2) for r in (1, 10, 50)}
log("lpn", R["lpn_protocol"], R["latency_model_s_per_token"]); save()
# ---------------- 3. security: can a trained inverter read the server's view? ----------------
import pandas as pd
from huggingface_hub import hf_hub_download
txt = "\n".join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset"))["text"])
pub = np.array(tok(txt[:400_000]).input_ids[:12_288])
SERVER_VIEWS.clear(); plain_inputs = {}
def capture(mode):
    feats = {"1.qkv": [], "12.qkv": []}
    for c in range(0, len(pub) - 255, 256):
        cache = {"len": 0}; SERVER_VIEWS.clear()
        if mode == "lpn": forward(pub[c:c + 256], cache, "lpn", keep_views=True)
        for kname in feats: feats[kname].append(SERVER_VIEWS[kname][0])
    return feats
ctrl = {"1.qkv": [], "12.qkv": []}
_orig_linear = linear
def linear_ctrl(name, X, mode, keep_view=False):
    if name in ctrl: ctrl[name].append(X.copy())
    return _orig_linear(name, X, "plain")
linear = linear_ctrl
for c in range(0, len(pub) - 255, 256): forward(pub[c:c + 256], {"len": 0}, "plain")
linear = _orig_linear
masked = {}
for kname in ctrl:
    rows = []
    for X in ctrl[kname]:
        xq, _ = quant_act(X); xf = (xq % np.int64(p)).astype(np.uint64); idx, val = sparse_mask(X.shape[1], X.shape[0])
        rows.append(addmod(xf, combine(GT[X.shape[1]], idx, val)))       # exactly the server's view
    masked[kname] = rows
log("captured server views")
toks = pub[:len(range(0, len(pub) - 255, 256)) * 256]
vocab, yy = np.unique(toks, return_inverse=True); ntr = int(len(toks) * 0.8)
def inverter_acc(F):
    X = torch.tensor(F, dtype=torch.float32); X = (X - X[:ntr].mean(0)) / (X[:ntr].std(0) + 1e-6); y = torch.tensor(yy)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 1024), torch.nn.GELU(), torch.nn.Linear(1024, len(vocab)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for ep in range(4):
        perm = torch.randperm(ntr)
        for s in range(0, ntr, 512):
            b = perm[s:s + 512]; loss = torch.nn.functional.cross_entropy(net(X[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return float((net(X[ntr:]).argmax(1) == y[ntr:]).float().mean())
maj = float(np.mean(yy[ntr:] == np.bincount(yy[:ntr]).argmax()))
R["security"] = {"tokens": int(len(toks)), "majority_class_baseline": round(maj, 4)}
for kname in ("1.qkv", "12.qkv"):
    Fm = np.concatenate(masked[kname]).astype(np.float64) / p           # server view, scaled to [0,1)
    Fc = np.concatenate(ctrl[kname])
    top = (np.concatenate(masked[kname]) >> U(53)).astype(np.int64).ravel()   # top 8 bits should be uniform
    chi = float(((np.bincount(top, minlength=256) - len(top) / 256) ** 2 / (len(top) / 256)).sum())
    R["security"][kname] = {"inverter_acc_on_server_view": round(inverter_acc(Fm), 4),
                            "inverter_acc_unmasked_control": round(inverter_acc(Fc), 4),
                            "chi2_top8bits_uniformity (df=255, ~255 expected)": round(chi, 1)}
    log("security", kname, R["security"][kname]); save()
# ---------------- 4. scaling estimate (analytic, same t / rate / field) ----------------
def scale(name, H, I, L, V, KVd, act_int4=True):
    d_out_sum = L * (H + 2 * KVd + H + 2 * I + H) + V                    # qkv + o + gate/up + down outputs, + head
    params = L * (H * (H + 2 * KVd) + H * H + 2 * H * I + I * H) + V * H
    storage = params / RATE * 8                                         # P = W G, N = n / rate, 8-byte field elements
    touched = T_SPARSE * (d_out_sum + L * (3 * H + I) + H) * 8          # client gathers per decode token
    local_int4 = params * 0.5
    return {"params_B": round(params / 1e9, 2), "rounds_per_token": 4 * L + 1, "client_storage_GB": round(storage / 1e9, 1),
            "client_MB_touched_per_token": round(touched / 1e6), "local_int4_MB_read_per_token": round(local_int4 / 1e6),
            "upload_MB_per_token": round((L * (3 * H + I) + H) * 8 / 1e6, 2), "download_MB_per_token": round(d_out_sum * 8 / 1e6, 2)}
R["scaling"] = {"qwen2.5-0.5b": scale("0.5b", 896, 4864, 24, 151936, 128), "llama-3-8b": scale("8b", 4096, 14336, 32, 128256, 1024),
                "llama-3-70b": scale("70b", 8192, 28672, 80, 128256, 1024)}
R["minutes"] = round((time.time() - T0) / 60, 1); save(); log(json.dumps(R["scaling"], indent=1)); log("done")
