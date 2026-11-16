"""T9: 'encode tokens before sending' via split inference: the client runs the embedding + first L layers locally
and sends hidden states (not tokens) to the server, which runs the rest. Can the server read the text?
Attack: the server knows the public model, so it runs it on public text to collect (hidden state -> token) pairs, fits a
ridge-regression map to the token-embedding space, then decodes each received hidden state by nearest embedding.
Also tested: the client adds Gaussian noise, and how much that noise damages the model's own next-token predictions."""
import json, numpy as np, torch, pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen2.5-0.5B-Instruct"; torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
E = model.get_input_embeddings().weight.detach(); En = E / E.norm(dim=1, keepdim=True)
txt = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset"))["text"]
pub = " ".join(t for t in txt[:6000] if len(t) > 50)
private = ("Counselor note 2026-10-03: Maria Lopez, grade 8, told me she has been thinking about hurting herself since her "
           "parents separated. Her mother Ana Lopez works nights. I contacted the school nurse and will follow up within one day. "
           "Maria's ID is S482913 and she takes insulin for type 1 diabetes.")
LAYERS = [1, 4, 8, 12, 23]
@torch.no_grad()
def states(text, maxlen=None):
    ids = tok(text, return_tensors="pt").input_ids[:, :maxlen]
    out = []
    for s in range(0, ids.shape[1], 512):
        chunk = ids[:, s:s+512]; o = model(chunk, output_hidden_states=True)
        out.append((chunk[0], o.hidden_states, o.logits))
    return out
pubs = states(pub, 24000)
R = {"model": M, "public_tokens_for_attacker": sum(len(c[0]) for c in pubs), "layers": []}
pid = states(private)[0]; true_ids = pid[0]
for L in LAYERS:
    X = torch.cat([h[L][0] for _, h, _ in pubs]); Y = torch.cat([E[i] for i, _, _ in pubs])
    lam = 1.0; A = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T @ Y)   # ridge: hidden -> embedding
    row = {"layer": L}
    for noise in (0.0, 0.5, 1.0, 2.0):
        h = pid[1][L][0]; scale = h.norm(dim=1, keepdim=True) / np.sqrt(h.shape[1])
        hn = h + noise * scale * torch.randn_like(h)
        pred = (hn @ A); pred = pred / pred.norm(dim=1, keepdim=True)
        rec = (pred @ En.T).argmax(1)
        acc = (rec == true_ids).float().mean().item()
        # utility: run the remaining layers from the noised state and compare next-token predictions with the clean model
        with torch.no_grad():
            pos = torch.arange(h.shape[0])[None]; x = hn[None]
            pe = model.model.rotary_emb(x, pos)
            for layer in model.model.layers[L:]:
                x = layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos)
                x = x[0] if isinstance(x, tuple) else x
            logits = model.lm_head(model.model.norm(x))[0]
        agree = (logits.argmax(-1) == pid[2][0].argmax(-1)).float().mean().item()
        row[f"noise_{noise}"] = {"token_recovery": round(acc, 3), "next_token_agreement_with_clean_model": round(agree, 3),
                                 "recovered_text": tok.decode(rec) if noise in (0.0, 1.0) else None}
    R["layers"].append(row); print(json.dumps(row)[:700], flush=True)
json.dump(R, open("results/t9_split_inversion.json", "w"), indent=1)
