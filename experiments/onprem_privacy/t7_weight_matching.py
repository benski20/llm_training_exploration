"""T7: if the obfuscated model was derived from PUBLIC open weights, the server can recover the secret key
from the weights alone, without any user traffic. Scheme modelled: E' = P E Q (+ noise)
(P = secret vocab permutation, Q = secret orthogonal rotation of the hidden dimension, as in
permutation/rotation 'covariant obfuscation' schemes). Rotation preserves row norms and inner products,
so match each obfuscated row to a public row by an invariant signature."""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM
E = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).get_input_embeddings().weight.detach().numpy().astype(np.float64)
E = E[:151643]                                   # drop unused padded rows
rng = np.random.default_rng(0); V, d = E.shape
Q, _ = np.linalg.qr(rng.standard_normal((d, d))); P = rng.permutation(V)
anch = np.argsort(-np.linalg.norm(E, axis=1))[:64]          # 64 public reference rows
R = {"vocab": V, "hidden": d, "results": []}
for noise in (0.0, 0.01, 0.05, 0.1):
    Eo = (E @ Q)[P] ; Eo = Eo + noise * np.linalg.norm(E, axis=1).mean() / np.sqrt(d) * rng.standard_normal(Eo.shape)
    # step 1: norms alone (rotation-invariant) -> sort-match
    pn, on = np.linalg.norm(E, axis=1), np.linalg.norm(Eo, axis=1)
    m_norm = np.empty(V, int); m_norm[np.argsort(on)] = np.argsort(pn)
    acc_norm = float((m_norm == P).mean())
    # step 2: recover Q from a few confidently matched high-norm rows (Procrustes), then nearest neighbour
    top_o = np.argsort(-on)[:256]; guess_pub = m_norm[top_o]
    U, _, Wt = np.linalg.svd(Eo[top_o].T @ E[guess_pub]); Qhat = (U @ Wt).T  # E Qhat ~ Eo
    Er = Eo @ Qhat.T                                                         # undo rotation
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    hits = 0; samp = rng.choice(V, 5000, replace=False)
    for i in samp:
        hits += int(np.argmax(En @ Er[i]) == P[i])
    R["results"].append({"noise_rel": noise, "recovered_by_norm_sort": acc_norm,
                         "recovered_after_procrustes_nn (5k sample)": hits / len(samp)})
    print(R["results"][-1], flush=True)
json.dump(R, open("results/t7_weight_matching.json", "w"), indent=1)
