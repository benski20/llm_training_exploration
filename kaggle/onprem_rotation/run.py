"""Kaggle CPU kernel: 'customer stores rotated weights, vendor runs them'.
1. Functional test: fold RMSNorm weights into the linear layers, apply a secret orthogonal rotation Q to the
   residual stream (plus a secret permutation of MLP neurons and of the vocabulary), and check the rotated model
   gives the same outputs. Customer side = embedding lookup + output head only; vendor runs every layer.
2. Attack: the vendor received the rotated weights and knows the original model. Recover Q
   (a) with known row correspondence (Procrustes), (b) with the neuron and vocab permutations (Gram-matrix eigenvectors),
   then read the customer's input tokens and the generated answer."""
import json, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0); R = {}; T0 = time.time()
M = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(M)
def load():
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float64).eval()
    m.lm_head.weight = torch.nn.Parameter(m.model.embed_tokens.weight.detach().clone()); m.config.tie_word_embeddings = False
    return m
@torch.no_grad()
def fold(m):  # make RMSNorm weights 1 so norms are rotation-invariant
    for L in m.model.layers:
        for lin in (L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj): lin.weight.mul_(L.input_layernorm.weight[None, :])
        for lin in (L.mlp.gate_proj, L.mlp.up_proj): lin.weight.mul_(L.post_attention_layernorm.weight[None, :])
        L.input_layernorm.weight.fill_(1); L.post_attention_layernorm.weight.fill_(1)
    m.lm_head.weight.mul_(m.model.norm.weight[None, :]); m.model.norm.weight.fill_(1)
orig = load(); fold(orig)                       # vendor's own (public) model, folded
cust = load(); fold(cust)
d = orig.config.hidden_size; V = cust.model.embed_tokens.weight.shape[0]
Q = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]   # customer's secret key
Pv = torch.randperm(V)                                              # secret vocab permutation
with torch.no_grad():
    for L in cust.model.layers:
        P = torch.randperm(L.mlp.gate_proj.weight.shape[0])         # secret MLP neuron permutation (per layer)
        for lin in (L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj): lin.weight.copy_(lin.weight @ Q)
        L.mlp.gate_proj.weight.copy_((L.mlp.gate_proj.weight @ Q)[P]); L.mlp.up_proj.weight.copy_((L.mlp.up_proj.weight @ Q)[P])
        L.mlp.down_proj.weight.copy_((Q.T @ L.mlp.down_proj.weight)[:, P]); L.self_attn.o_proj.weight.copy_(Q.T @ L.self_attn.o_proj.weight)
    cust.model.embed_tokens.weight.copy_((cust.model.embed_tokens.weight @ Q)[Pv])
    cust.lm_head.weight.copy_((cust.lm_head.weight @ Q)[Pv])
inv_Pv = torch.argsort(Pv)
note = ("Counselor note: Maria Lopez, grade 8, student ID S482913, told me she has been thinking about hurting herself "
        "since her parents separated. Her mother Ana Lopez works nights. Write a two-sentence follow-up plan.")
ids = tok.apply_chat_template([{"role": "user", "content": note}], add_generation_prompt=True, return_tensors="pt")
if not torch.is_tensor(ids): ids = ids["input_ids"]
cids = inv_Pv[ids]                                   # customer's private token ids in the permuted vocab
with torch.no_grad():
    lo, lc = orig(ids).logits[0], cust(cids).logits[0]
    R["functional"] = {"max_abs_logit_diff": float((lo - lc[:, inv_Pv]).abs().max()),
                       "same_next_token_all_positions": bool((lo.argmax(-1) == Pv[lc.argmax(-1)]).all())}
    # what the vendor sees: rotated embeddings sent by the customer + rotated final hidden states it computes
    x_sent = cust.model.embed_tokens(cids)
    out = cust.generate(cids, max_new_tokens=40, do_sample=False)
    h_final = cust(out, output_hidden_states=True).hidden_states[-1][0]
R["customer_answer_decoded_by_customer"] = tok.decode(Pv[out[0, ids.shape[1]:]])
print(R, flush=True)
# ---------------- attack ----------------
def recover_Q_procrustes(W_pub, W_rot):          # W_rot = W_pub @ Q, rows in known order
    U, _, Vt = torch.linalg.svd(W_pub.T @ W_rot); return U @ Vt
def recover_Q_gram(G_pub, G_rot, G2_pub, G2_rot):  # G_rot = Q^T G_pub Q  (permutations cancel in Gram matrices)
    ev, Vp = torch.linalg.eigh(G_pub); ev2, Vr = torch.linalg.eigh(G_rot)
    A, A2 = Vp.T @ G2_pub @ Vp, Vr.T @ G2_rot @ Vr        # A2 = S A S, S = diag(signs)
    j = int(torch.argmax(A.abs().sum(0)))
    s = torch.sign(A2[:, j] / A[:, j]); s = s * torch.sign(s[j])
    return Vp @ torch.diag(s) @ Vr.T
L0o, L0c = orig.model.layers[0], cust.model.layers[0]
t = time.time()
Qa = recover_Q_procrustes(L0o.self_attn.q_proj.weight, L0c.self_attn.q_proj.weight)  # q_proj rows are not permuted
gram = lambda W: W.T @ W
Qb = recover_Q_gram(gram(L0o.mlp.gate_proj.weight), gram(L0c.mlp.gate_proj.weight),
                    gram(L0o.mlp.up_proj.weight), gram(L0c.mlp.up_proj.weight))
R["attack"] = {"seconds": round(time.time() - t, 2),
               "procrustes_key_error": float((Qa - Q).abs().max()), "gram_eig_key_error_with_permutations": float((Qb - Q).abs().max())}
with torch.no_grad():
    E = orig.model.embed_tokens.weight; En = E / E.norm(dim=1, keepdim=True)
    for name, Qh in (("procrustes", Qa), ("gram", Qb)):
        x = x_sent[0] @ Qh.T                                     # un-rotate what the customer sent
        rec = (x / x.norm(dim=1, keepdim=True) @ En.T).argmax(1)
        ans = (h_final @ Qh.T) @ orig.lm_head.weight.T            # un-rotate vendor-side final states, public head
        gen = ans.argmax(-1)[ids.shape[1]-1:-1]
        R["attack"][name] = {"input_token_recovery": float((rec == ids[0]).float().mean()),
                             "recovered_input": tok.decode(rec[-60:]), "recovered_answer": tok.decode(gen)}
R["minutes"] = round((time.time() - T0) / 60, 1)
print(json.dumps(R, indent=1)); json.dump(R, open("/kaggle/working/results.json", "w"), indent=1)
