"""Kaggle CPU kernel: single-server private inference with LINEAR-ONLY homomorphic encryption (CKKS, OpenFHE).
Customer: encrypts each linear layer's input, decrypts the output, does norms/RoPE/attention/SiLU/LM-head in plaintext.
Server (us): plaintext-weight x ciphertext matrix-vector products (BSGS diagonal method), no bootstrapping, depth 1.
Part A: cost of one dense 4096x4096 block (8B-scale shapes) -> per-token extrapolation for 8B / 70B.
Part B: end-to-end Qwen2.5-0.5B, all transformer linear layers encrypted, compared with plaintext."""
import subprocess, sys, os, json, time, math
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openfhe"], check=False)
OUT = "/kaggle/working"; R = {"cpu_count": os.cpu_count()}; T0 = time.time()
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
def save(): json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)
try:
    import openfhe as of
except Exception as e:
    import traceback; R["import_error"] = traceback.format_exc(); save(); raise
import numpy as np
# ---------------- CKKS context ----------------
S = 4096; B = 64                                   # slots per ciphertext, baby-step size
prm = of.CCParamsCKKSRNS(); prm.SetMultiplicativeDepth(1); prm.SetScalingModSize(40); prm.SetFirstModSize(60)
prm.SetBatchSize(S); prm.SetSecurityLevel(of.SecurityLevel.HEStd_128_classic)
cc = of.GenCryptoContext(prm)
for f in (of.PKESchemeFeature.PKE, of.PKESchemeFeature.KEYSWITCH, of.PKESchemeFeature.LEVELEDSHE): cc.Enable(f)
t = time.time(); keys = cc.KeyGen()
rot_idx = list(range(1, B)) + [g * B for g in range(1, S // B)]
cc.EvalRotateKeyGen(keys.secretKey, rot_idx)
R["context"] = {"ring_dim": cc.GetRingDimension(), "slots": S, "security": "HEStd_128_classic", "depth": 1,
                "scale_bits": 40, "first_mod_bits": 60, "rotation_keys": len(rot_idx), "keygen_s": round(time.time() - t, 1)}
M = cc.GetCyclotomicOrder(); log(R["context"]); save()
def ct_bytes(ct):
    p = "/tmp/ct.bin"; of.SerializeToFile(p, ct, of.BINARY); return os.path.getsize(p)
def dec(ct, n):
    pt = cc.Decrypt(ct, keys.secretKey); pt.SetLength(n)
    try: return np.array(pt.GetRealPackedValue())
    except Exception: return np.array([c.real for c in pt.GetCKKSPackedValue()])
TM = {"encode_s": 0.0, "mult_s": 0.0, "rot_s": 0.0, "prep_s": 0.0, "enc_s": 0.0, "dec_s": 0.0, "n_mult": 0, "n_rot": 0, "n_enc": 0, "n_dec": 0,
      "bytes_up": 0, "bytes_down": 0}
def encrypt_vec(x):                                  # CUSTOMER: x (n,) -> list of ciphertexts of S slots
    cts = []
    for c in range(0, len(x), S):
        v = np.zeros(S); v[:min(S, len(x) - c)] = x[c:c + S]; t = time.time()
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(v.tolist())); TM["enc_s"] += time.time() - t; TM["n_enc"] += 1
        cts.append(ct)
    return cts
def server_matvec(W, cts):                            # SERVER: W (d x n) plaintext, cts encrypt x -> ciphertexts of W x
    d, n = W.shape; nr, nc = math.ceil(d / S), math.ceil(n / S)
    babies = []
    for ct in cts:                                    # hoisted baby-step rotations of each input ciphertext
        t = time.time(); pre = cc.EvalFastRotationPrecompute(ct)
        rs = [ct] + [cc.EvalFastRotation(ct, j, M, pre) for j in range(1, B)]
        TM["rot_s"] += time.time() - t; TM["n_rot"] += B - 1; babies.append(rs)
    ar = np.arange(S); outs = []
    for r in range(nr):
        acc = None
        for g in range(S // B):
            inner = None
            for c in range(nc):
                t = time.time()
                tile = np.zeros((S, S)); blk = W[r * S:(r + 1) * S, c * S:(c + 1) * S]; tile[:blk.shape[0], :blk.shape[1]] = blk
                ks = g * B + np.arange(B)
                D = tile[ar[None, :], (ar[None, :] + ks[:, None]) % S]           # diagonals k = gB + j, shape (B, S)
                D = np.roll(D, g * B, axis=1)                                     # pre-rotate for the giant step
                TM["prep_s"] += time.time() - t
                for j in range(B):
                    if not D[j].any(): continue
                    t = time.time(); pt = cc.MakeCKKSPackedPlaintext(D[j].tolist()); TM["encode_s"] += time.time() - t
                    t = time.time(); m = cc.EvalMult(babies[c][j], pt); inner = m if inner is None else cc.EvalAdd(inner, m)
                    TM["mult_s"] += time.time() - t; TM["n_mult"] += 1
            if inner is None: continue
            t = time.time()
            if g: inner = cc.EvalRotate(inner, g * B); TM["n_rot"] += 1
            acc = inner if acc is None else cc.EvalAdd(acc, inner); TM["rot_s"] += time.time() - t
        outs.append(acc)
    return outs
def decrypt_vec(cts, d):                              # CUSTOMER
    t = time.time(); y = np.concatenate([dec(ct, S) for ct in cts])[:d]; TM["dec_s"] += time.time() - t; TM["n_dec"] += len(cts)
    return y
# self-test on a small non-square case
rng = np.random.default_rng(0)
Wt = rng.normal(0, 0.05, (300, 5000)); xt = rng.normal(0, 1, 5000)
yt = decrypt_vec(server_matvec(Wt, encrypt_vec(xt)), 300)
err = float(np.abs(yt - Wt @ xt).max()); R["self_test_max_abs_err"] = err; log("self-test err", err); save()
assert err < 1e-3, err
# ================= Part A: one dense 4096x4096 block =================
for k in TM: TM[k] = 0 if isinstance(TM[k], int) else 0.0
W = rng.normal(0, 0.02, (S, S)); x = rng.normal(0, 1, S); x /= np.abs(x).max()
t = time.time(); cts = encrypt_vec(x); t_enc = time.time() - t
up = ct_bytes(cts[0])
t = time.time(); out = server_matvec(W, cts); t_srv = time.time() - t
down = ct_bytes(out[0])
try:
    cc.ModReduceInPlace(out[0]); down_reduced = ct_bytes(out[0])     # drop to the last tower before sending
except Exception as e: down_reduced = None; R["modreduce_note"] = str(e)[:200]
t = time.time(); y = decrypt_vec(out, S); t_dec = time.time() - t
t = time.time()
for _ in range(50): W @ x
t_plain = (time.time() - t) / 50
R["A_block_4096"] = {"server_s": round(t_srv, 2), "server_breakdown_s": {k: round(TM[k], 2) for k in ("prep_s", "encode_s", "mult_s", "rot_s")},
    "n_ptct_mult": TM["n_mult"], "n_rotations": TM["n_rot"], "client_encrypt_s": round(t_enc, 4), "client_decrypt_s": round(t_dec, 4),
    "ciphertext_up_bytes": up, "ciphertext_down_bytes": down, "ciphertext_down_after_modreduce_bytes": down_reduced,
    "max_abs_err": float(np.abs(y - W @ x).max()), "plain_numpy_matvec_s": t_plain, "server_slowdown_vs_numpy": round(t_srv / t_plain)}
log("A", R["A_block_4096"]); save()
a = R["A_block_4096"]; srv_wo_encode = t_srv - TM["encode_s"] - TM["prep_s"]
def per_token(name, H, I, L, KVd, V):
    tiles = L * (math.ceil((H + 2 * KVd) / S) * math.ceil(H / S) + math.ceil(H / S) ** 2 + math.ceil(2 * I / S) * math.ceil(H / S)
                 + math.ceil(H / S) * math.ceil(I / S))
    enc = L * (3 * math.ceil(H / S) + math.ceil(I / S)); decn = L * (math.ceil((H + 2 * KVd) / S) + math.ceil(H / S) + math.ceil(2 * I / S) + math.ceil(H / S))
    dn = down_reduced or down
    return {"tiles_per_token": tiles, "server_cpu_core_s_per_token_cached_plaintexts": round(tiles * srv_wo_encode, 1),
            "server_cpu_s_per_token_incl_encoding": round(tiles * t_srv, 1),
            "client_encrypts": enc, "client_decrypts": decn, "client_s_per_token": round(enc * t_enc + decn * t_dec, 2),
            "upload_MB_per_token": round(enc * up / 1e6, 1), "download_MB_per_token": round(decn * dn / 1e6, 1),
            "rounds_per_token": 4 * L, "lm_head_on_client_GFLOP": round(2 * V * H / 1e9, 2)}
R["A_extrapolation"] = {"llama-3-8b": per_token("8b", 4096, 14336, 32, 1024, 128256), "llama-3-70b": per_token("70b", 8192, 28672, 80, 1024, 128256)}
log("A-extrap", R["A_extrapolation"]); save()
# ================= Part B: end-to-end Qwen2.5-0.5B =================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MID = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(MID)
hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval(); cfg = hf.config
H, NH, KV, HD, L, I = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_hidden_layers, cfg.intermediate_size
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
E = sd["model.embed_tokens.weight"]; HEAD = sd.get("lm_head.weight", E)
THETA = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 1e6)
inv = 1.0 / (THETA ** (np.arange(0, HD, 2) / HD))
LIN = {}
for i in range(L):
    p = f"model.layers.{i}."
    LIN[f"{i}.qkv"] = (np.concatenate([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.v_proj.weight"]]),
                       np.concatenate([sd[p + "self_attn.q_proj.bias"], sd[p + "self_attn.k_proj.bias"], sd[p + "self_attn.v_proj.bias"]]))
    LIN[f"{i}.o"] = (sd[p + "self_attn.o_proj.weight"], None)
    LIN[f"{i}.gu"] = (np.concatenate([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]]), None)
    LIN[f"{i}.down"] = (sd[p + "mlp.down_proj.weight"], None)
def linear(name, X, mode):
    W, b = LIN[name]
    if mode == "plain": Y = X @ W.T
    else:
        Y = np.zeros((X.shape[0], W.shape[0]))
        for r in range(X.shape[0]):
            sc = np.abs(X[r]).max() or 1.0
            cts = encrypt_vec(X[r] / sc)
            for ct in cts: TM["bytes_up"] += 0                      # sizes measured in Part A
            Y[r] = decrypt_vec(server_matvec(W, cts), W.shape[0]) * sc
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
prompt = "Counselor note: Maria Lopez, grade 8, said she has been thinking about hurting herself. Next step:"
ids = np.array(tok(prompt).input_ids); NEW = 4; BUDGET_S = 4.5 * 3600
def run(mode, n_new):
    cache = {"len": 0}; out, lgs, times = [], [], []
    t = time.time(); lg = forward(ids, cache, mode); times.append(time.time() - t)
    for _ in range(n_new):
        nxt = int(lg[-1].argmax()); out.append(nxt); lgs.append(lg[-1])
        if time.time() - T0 > BUDGET_S: break
        t = time.time(); lg = forward(np.array([nxt]), cache, mode); times.append(time.time() - t)
    return out, np.array(lgs), times
out_p, lg_p, _ = run("plain", NEW)
with torch.no_grad(): ref = hf.generate(torch.tensor([ids]), max_new_tokens=NEW, do_sample=False)[0, len(ids):].tolist()
R["B_plain"] = {"prompt_tokens": int(len(ids)), "manual_forward_matches_hf": out_p == ref, "answer": tok.decode(out_p)}
log("B plain", R["B_plain"]); save()
for k in TM: TM[k] = 0 if isinstance(TM[k], int) else 0.0
out_h, lg_h, times = run("he", NEW)
R["B_he"] = {"answer": tok.decode(out_h), "tokens_generated": len(out_h), "identical_tokens_to_plaintext": out_h == out_p[:len(out_h)],
    "max_abs_logit_diff": float(np.abs(lg_h - lg_p[:len(lg_h)]).max()), "prefill_s": round(times[0], 1),
    "decode_s_per_token": round(float(np.mean(times[1:])), 1) if len(times) > 1 else None,
    "server_s_total": round(TM["prep_s"] + TM["encode_s"] + TM["mult_s"] + TM["rot_s"], 1),
    "server_breakdown_s": {k: round(TM[k], 1) for k in ("prep_s", "encode_s", "mult_s", "rot_s")},
    "client_crypto_s_total": round(TM["enc_s"] + TM["dec_s"], 2), "n_encrypt": TM["n_enc"], "n_decrypt": TM["n_dec"],
    "note": "0.5B dims (896/4864) are padded into 4096-slot blocks, so server cost here overstates per-weight cost; use Part A for scaling."}
R["minutes"] = round((time.time() - T0) / 60, 1); log("B he", R["B_he"]); save(); log("done")
