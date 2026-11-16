"""Linear-only HE private inference on a GPU (Kaggle T4), using the DESILO FHE library's CUDA RNS-CKKS
(non-commercial licence; this work uses the DESILO FHE library developed by DESILO Inc., https://desilo.ai).
Protocol per linear layer: customer encrypts the activation vector -> server multiplies by the public weight matrix
under encryption (depth 1, no bootstrapping) -> customer decrypts, applies norms/attention/SiLU in plaintext, repeats.
Part A: GPU cost of one SxS weight tile (S=4096) -> per-token extrapolation for Llama-3-8B / 70B shapes.
Part B: end-to-end Qwen2.5-0.5B, every transformer linear layer under HE, LM head on the customer; check exactness."""
import os, sys, time, json, subprocess, math
T0 = time.time(); OUT = "/kaggle/working"; R = {}
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
def save(): json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=str)
for pkg in ("desilofhe-cu124", "desilofhe-cu128"):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
    try:
        import desilofhe; R["package"] = pkg; break
    except Exception as e: log("install/import failed", pkg, r.stderr[-500:], e)
import numpy as np, torch
from desilofhe import Engine
MODE = "gpu" if torch.cuda.is_available() else "parallel"
if MODE == "gpu":
    try: Engine(slot_count=64, max_level=1, mode="gpu")
    except Exception as e: log("gpu engine failed -> CPU parallel", repr(e)[:300]); MODE = "parallel"
R["mode"] = MODE; R["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"cpu x{os.cpu_count()}"
log("python", sys.version.split()[0], R["device"], MODE); R["citation"] = "This work uses the DESILO FHE library developed by DESILO Inc. (https://desilo.ai)"

def make(S):
    eng = Engine(slot_count=S, max_level=1, mode=MODE, **({} if MODE == "gpu" else {"thread_count": os.cpu_count()})); sk = eng.create_secret_key()
    mk = eng.create_matrix_multiplication_key(sk); return eng, sk, mk
def sync():
    if MODE == "gpu": torch.cuda.synchronize()
def ct_bytes(eng, ct):
    for f in (lambda: len(ct.serialize()), lambda: len(eng.serialize(ct)), lambda: len(eng.serialize_ciphertext(ct))):
        try: return int(f())
        except Exception: pass
    return None

# ---------------- self-test: orientation + accuracy ----------------
eng, sk, mk = make(64); rng = np.random.default_rng(0)
M = rng.standard_normal((64, 64)); x = rng.standard_normal(64)
y = np.array(eng.decrypt(eng.multiply_matrix(eng.encode_to_plain_matrix(M), eng.encrypt(x, sk), mk), sk))[:64]
TRANS = np.abs(y - M.T @ x).max() < np.abs(y - M @ x).max()
R["selftest"] = {"err_Mx": float(np.abs(y - M @ x).max()), "err_MTx": float(np.abs(y - M.T @ x).max()), "transposed": bool(TRANS)}
log("selftest", R["selftest"]); save()
def enc_mat(eng, W, light=True):           # W (S x S): want y = W x
    A = W.T if TRANS else W
    return eng.encode_to_light_plain_matrix(A) if light else eng.encode_to_plain_matrix(A)

# ================= Part A: one 4096 x 4096 tile on the GPU =================
SKIP_A = False
S = 4096; eng, sk, mk = make(S if not SKIP_A else 64)
if not SKIP_A:
    W = rng.standard_normal((S, S)) / 64; x = rng.standard_normal(S)
    t = time.time(); ct = eng.encrypt(x, sk); t_enc = time.time() - t
    A = {"slot_count": S}
    for kind in ("light", "plain"):
        try:
            t = time.time(); pm = enc_mat(eng, W, kind == "light"); sync(); te = time.time() - t
            out = eng.multiply_matrix(pm, ct, mk); sync()                      # warm-up
            ts = []
            for _ in range(5): t = time.time(); out = eng.multiply_matrix(pm, ct, mk); sync(); ts.append(time.time() - t)
            t = time.time(); y = np.array(eng.decrypt(out, sk))[:S]; t_dec = time.time() - t
            A[kind] = {"encode_s": round(te, 3), "multiply_s": round(float(np.median(ts)), 4), "decrypt_s": round(t_dec, 4),
                       "rel_err": float(np.abs(y - W @ x).max() / np.abs(W @ x).max()), }
            try: A[kind]["torch_free_GB"] = round(torch.cuda.mem_get_info()[0] / 1e9, 2)
            except Exception: pass
            del pm
        except Exception as e: A[kind] = {"error": repr(e)[:400]}
        log("A", kind, A[kind]); R["A_tile"] = A; save()
    A["encrypt_s"] = round(t_enc, 4); A["ct_up_bytes"] = ct_bytes(eng, ct); A["ct_down_bytes"] = ct_bytes(eng, out)
    t = time.time(); s2 = eng.add(out, out); sync(); A["ct_add_s"] = round(time.time() - t, 5)
    def per_token(H, I, L, Q, KVd, V):
        mats = [(Q + 2 * KVd, H), (H, Q), (2 * I, H), (H, I)]       # qkv, o, gate+up, down: (out, in)
        tiles = sum(math.ceil(m / S) * math.ceil(n / S) for m, n in mats) * L
        up = sum(math.ceil(n / S) for m, n in mats) * L; dn = sum(math.ceil(m / S) for m, n in mats) * L
        best = min(v["multiply_s"] for k, v in A.items() if isinstance(v, dict) and "multiply_s" in v)
        return {"tiles_per_token": tiles, "server_gpu_s_per_token_cached_weights": round(tiles * best, 2),
                "server_gpu_s_per_token_reencoding_weights": round(tiles * (best + A["light"].get("encode_s", 0)), 1) if "encode_s" in A.get("light", {}) else None,
                "client_encrypts": up, "client_decrypts": dn, "client_crypto_s_per_token_on_T4": round(up * A["encrypt_s"] + dn * A["light"].get("decrypt_s", 0), 2),
                "upload_MB_per_token": round(up * (A["ct_up_bytes"] or 0) / 1e6, 1), "download_MB_per_token": round(dn * (A["ct_down_bytes"] or 0) / 1e6, 1),
                "rounds_per_token": 4 * L, "lm_head_on_client_GFLOP": round(2 * V * H / 1e9, 2)}
    try:
        R["A_extrapolation"] = {"llama-3-8b": per_token(4096, 14336, 32, 4096, 1024, 128256), "llama-3-70b": per_token(8192, 28672, 80, 8192, 1024, 128256)}
    except Exception as e: R["A_extrapolation"] = {"error": repr(e)}
    log("A extrapolation", R["A_extrapolation"]); save()
del eng, mk


# ================= Part B: end-to-end Qwen2.5-0.5B =================
from transformers import AutoTokenizer, AutoModelForCausalLM
MID = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(MID)
hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval(); cfg = hf.config
H, NH, KV, HD, L, I = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_hidden_layers, cfg.intermediate_size
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
E = sd["model.embed_tokens.weight"]; HEAD = sd.get("lm_head.weight", E)
THETA = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 1e6)
inv = 1.0 / (THETA ** (np.arange(0, HD, 2) / HD))
SB = 1024; eng, sk, mk = make(SB)
LIN = {}
for i in range(L):
    p = f"model.layers.{i}."
    LIN[f"{i}.qkv"] = (np.concatenate([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.v_proj.weight"]]),
                       np.concatenate([sd[p + "self_attn.q_proj.bias"], sd[p + "self_attn.k_proj.bias"], sd[p + "self_attn.v_proj.bias"]]))
    LIN[f"{i}.o"] = (sd[p + "self_attn.o_proj.weight"], None)
    LIN[f"{i}.gu"] = (np.concatenate([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]]), None)
    LIN[f"{i}.down"] = (sd[p + "mlp.down_proj.weight"], None)
# server: encode every weight tile once (weights are public) and keep them; fall back to re-encoding if GPU memory runs out
TILES = {}; CACHED = False; TM = {"encode_s": 0.0, "mult_s": 0.0, "enc_s": 0.0, "dec_s": 0.0, "n_enc": 0, "n_dec": 0, "n_tile": 0}
def tile(name, bi, bj):
    W = LIN[name][0]; key = (name, bi, bj)
    if key in TILES: return TILES[key]
    blk = np.zeros((SB, SB)); sub = W[bi * SB:(bi + 1) * SB, bj * SB:(bj + 1) * SB]; blk[:sub.shape[0], :sub.shape[1]] = sub
    t = time.time(); pm = enc_mat(eng, blk, True); sync(); TM["encode_s"] += time.time() - t
    if CACHED: TILES[key] = pm
    return pm
t = time.time()
if CACHED:
  for name, (W, _) in LIN.items():
        for bi in range(math.ceil(W.shape[0] / SB)):
            for bj in range(math.ceil(W.shape[1] / SB)): tile(name, bi, bj)
R["B_server_setup"] = {"tiles": sum(math.ceil(W.shape[0] / SB) * math.ceil(W.shape[1] / SB) for W, _ in LIN.values()),
                       "cached_all": CACHED, "one_time_encode_s": round(time.time() - t, 1)}
log("B setup", R["B_server_setup"]); save(); TM["encode_s"] = 0.0
def linear(name, X, mode):
    W, b = LIN[name]
    if mode == "plain": Y = X @ W.T
    else:
        m, n = W.shape; Y = np.zeros((X.shape[0], m)); nb = math.ceil(n / SB); scs = []; CT = []
        for r in range(X.shape[0]):                                # CUSTOMER: encrypt each row
            sc = np.abs(X[r]).max() or 1.0; xs = np.zeros(nb * SB); xs[:n] = X[r] / sc; scs.append(sc)
            t = time.time(); CT.append([eng.encrypt(xs[j * SB:(j + 1) * SB], sk) for j in range(nb)]); TM["enc_s"] += time.time() - t; TM["n_enc"] += nb
        ACC = [[None] * math.ceil(m / SB) for _ in CT]
        for bi in range(math.ceil(m / SB)):                        # SERVER: sees only ciphertexts
            for bj in range(nb):
                pm = tile(name, bi, bj)                           # public weights, encoded once per call (cacheable in production)
                for r in range(len(CT)):
                    t = time.time(); o = eng.multiply_matrix(pm, CT[r][bj], mk)
                    ACC[r][bi] = o if ACC[r][bi] is None else eng.add(ACC[r][bi], o); sync(); TM["mult_s"] += time.time() - t; TM["n_tile"] += 1
                del pm
        for r in range(len(CT)):                                   # CUSTOMER: decrypt
            t = time.time(); y = np.concatenate([np.array(eng.decrypt(o, sk))[:SB] for o in ACC[r]])[:m]; TM["dec_s"] += time.time() - t; TM["n_dec"] += len(ACC[r])
            Y[r] = y * scs[r]
    return Y + b[None, :] if b is not None else Y
def rms(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + cfg.rms_norm_eps) * w
def rope(x, pos):
    f = pos[:, None] * inv[None]; c, s = np.cos(np.concatenate([f, f], -1))[:, None], np.sin(np.concatenate([f, f], -1))[:, None]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]; return x * c + np.concatenate([-x2, x1], -1) * s
def forward(ids, cache, mode):
    T = len(ids); start = cache["len"]; pos = np.arange(start, start + T, dtype=np.float64); x = E[ids]
    for i in range(L):
        p = f"model.layers.{i}."; h = rms(x, sd[p + "input_layernorm.weight"]); qkv = linear(f"{i}.qkv", h, mode)
        q = rope(qkv[:, :H].reshape(T, NH, HD), pos); k = rope(qkv[:, H:H + KV * HD].reshape(T, KV, HD), pos); v = qkv[:, H + KV * HD:].reshape(T, KV, HD)
        cache.setdefault(i, [np.zeros((0, KV, HD)), np.zeros((0, KV, HD))])
        cache[i][0] = np.concatenate([cache[i][0], k]); cache[i][1] = np.concatenate([cache[i][1], v])
        K = np.repeat(cache[i][0], NH // KV, 1); V = np.repeat(cache[i][1], NH // KV, 1)
        sc = np.einsum("thd,shd->hts", q, K) / np.sqrt(HD)
        sc[:, np.arange(K.shape[0])[None, :] > (start + np.arange(T))[:, None]] = -1e30
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        x = x + linear(f"{i}.o", np.einsum("hts,shd->thd", sc, V).reshape(T, H), mode)
        gu = linear(f"{i}.gu", rms(x, sd[p + "post_attention_layernorm.weight"]), mode)
        x = x + linear(f"{i}.down", gu[:, :I] / (1 + np.exp(-gu[:, :I])) * gu[:, I:], mode)
    cache["len"] += T
    return rms(x[-1:], sd["model.norm.weight"]) @ HEAD.T            # LM head on the customer (plaintext)
prompt = "Maria Lopez, grade 8, said she feels unsafe. The counselor will"
ids = np.array(tok(prompt).input_ids); NEW = 8; BUDGET_S = 8 * 3600
def run(mode, n_new):
    cache = {"len": 0}; out, lgs, times = [], [], []
    t = time.time(); lg = forward(ids, cache, mode); times.append(time.time() - t)
    for _ in range(n_new):
        nxt = int(lg[-1].argmax()); out.append(nxt); lgs.append(lg[-1])
        if time.time() - T0 > BUDGET_S: break
        t = time.time(); lg = forward(np.array([nxt]), cache, mode); times.append(time.time() - t)
        if mode == "he": log("token", len(out), round(times[-1], 2), "s")
    return out, np.array(lgs), times
out_p, lg_p, _ = run("plain", NEW)
with torch.no_grad(): ref = hf.generate(torch.tensor([ids]), max_new_tokens=NEW, do_sample=False)[0, len(ids):].tolist()
R["B_plain"] = {"prompt_tokens": int(len(ids)), "manual_forward_matches_hf": out_p == ref, "answer": tok.decode(out_p)}
log("B plain", R["B_plain"]); save()
for k in TM: TM[k] = 0 if isinstance(TM[k], int) else 0.0
out_h, lg_h, times = run("he", NEW)
nd = max(1, len(times) - 1)
R["B_he"] = {"answer": tok.decode(out_h), "tokens_generated": len(out_h), "identical_tokens_to_plaintext": out_h == out_p[:len(out_h)],
    "max_abs_logit_diff": float(np.abs(lg_h - lg_p[:len(lg_h)]).max()), "max_logit": float(np.abs(lg_p).max()),
    "prefill_s": round(times[0], 1), "decode_s_per_token": round(float(np.mean(times[1:])), 2) if len(times) > 1 else None,
    "server_gpu_s_total": round(TM["mult_s"], 1), "server_encode_s_total": round(TM["encode_s"], 1),
    "client_crypto_s_total": round(TM["enc_s"] + TM["dec_s"], 1), "n_encrypt": TM["n_enc"], "n_decrypt": TM["n_dec"], "n_tile_mults": TM["n_tile"],
    "note": "client encrypt/decrypt here also ran on the T4; 0.5B dims are padded into 1024-slot tiles"}
R["minutes"] = round((time.time() - T0) / 60, 1); log("B he", R["B_he"]); save(); log("done")
