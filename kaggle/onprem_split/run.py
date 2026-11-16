"""Kaggle CPU kernel.
A) Split inference: the client runs layers [0, k) of Qwen2.5-0.5B locally and sends hidden states h_k to
   the vendor, who runs the rest. Can the vendor (who has the weights and public text) invert h_k to tokens?
   Adaptive attacker: MLP inverter trained on public text (wikitext-103), tested on held-out wikitext-2 and
   synthetic counselor notes with names. Defense: Gaussian noise on h_k; utility = loss increase.
B) llama.cpp Q4 decode speed for 1.5B and 7B on Kaggle CPU (replaces the earlier extrapolation)."""
import subprocess, sys, os, json, time, random, math
OUT = "/kaggle/working"; R = {"started": time.ctime(), "cpu_count": os.cpu_count()}
def save(): json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
# ---------------- B: llama.cpp ----------------
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python",
                    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"], check=True)
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    R["llama_cpp"] = []
    for repo, f in (("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
                    ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")):
        p = hf_hub_download(repo, f, local_dir="/tmp/gguf")
        llm = Llama(model_path=p, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
        prompt = "Write a detailed plan for a school counselor helping a student whose attendance is dropping. " * 6
        llm(prompt, max_tokens=8)
        t = time.time(); o = llm(prompt, max_tokens=128, temperature=0, ignore_eos=True); dt = time.time() - t
        pt, ct = o["usage"]["prompt_tokens"], o["usage"]["completion_tokens"]
        t = time.time(); llm(prompt + " Also.", max_tokens=1); ptime = time.time() - t
        R["llama_cpp"].append({"model": f, "threads": os.cpu_count(), "prompt_tokens": pt, "gen_tokens": ct,
                               "total_s": round(dt, 2), "approx_decode_tok_per_s": round(ct / max(1e-6, dt - ptime), 2),
                               "prompt_eval_tok_per_s": round(pt / ptime, 1)})
        log(R["llama_cpp"][-1]); save(); del llm
except Exception as e:
    import traceback; R["llama_cpp_error"] = traceback.format_exc()[-1500:]; save(); log("llama err", e)
# ---------------- A: split-inference inversion ----------------
import torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
M = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
from huggingface_hub import hf_hub_download
ld = lambda f: "\n".join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", f, repo_type="dataset"))["text"])
pub = ld("wikitext-103-raw-v1/train-00000-of-00002.parquet")[:3_000_000]
wt2 = ld("wikitext-2-raw-v1/test-00000-of-00001.parquet")[:400_000]
rng = random.Random(0)
FIRST = "Maria James Aisha Dmitri Lena Kofi Priya Tom Sofia Wei Omar Hannah Diego Mei Ivan Grace Samuel Nia Lucas Emma".split()
LAST = "Lopez Chen Okafor Ivanov Schmidt Mensah Patel Brown Rossi Zhang Haddad Kowalski Nguyen Silva Johansson Adeyemi".split()
TPL = ["Met with {s} (grade {g}) today about missed classes. {s} said things at home have been hard since {p} lost a job.",
       "{s} disclosed feeling anxious before tests and has been skipping lunch. I will call {p} on Friday.",
       "Teacher reported that {s} was in a fight near the gym. {s} has a history of being targeted. Parent {p} was informed.",
       "{s} has type 1 diabetes and fainted in chemistry. Nurse notified; {p} will bring new supplies.",
       "{s} is failing algebra after a strong start. {s} asked for a tutor; I emailed {p} to set up a meeting."]
notes, name_spans = [], []
for _ in range(250):
    s, p = f"{rng.choice(FIRST)} {rng.choice(LAST)}", f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    notes.append(rng.choice(TPL).format(s=s, p=p, g=rng.randint(6, 12))); name_spans.append((s, p))
def ids_of(text, n): return tok(text).input_ids[:n]
LAYERS = [0, 1, 2, 4, 8, 12, 18]
@torch.no_grad()
def hidden(chunks):
    H = {k: [] for k in LAYERS}; T = []
    for c in chunks:
        x = torch.tensor([c]); hs = model(x, output_hidden_states=True).hidden_states
        for k in LAYERS: H[k].append(hs[k][0].half())
        T.append(x[0])
    return {k: torch.cat(v) for k, v in H.items()}, torch.cat(T)
def chunk(ids, L=256): return [ids[i:i+L] for i in range(0, len(ids) - L + 1, L)]
tr_ids = ids_of(pub, 160_000); ev_ids = ids_of(wt2, 30_000)
Htr, Ttr = hidden(chunk(tr_ids)); Hev, Tev = hidden(chunk(ev_ids))
# notes: run each note separately; mark name tokens
nH = {k: [] for k in LAYERS}; nT = []; nMask = []
with torch.no_grad():
    for t, (s, p) in zip(notes, name_spans):
        enc = tok(t, return_offsets_mapping=True); ids = enc.input_ids
        spans = [(m.start(), m.end()) for nm in (s, p) for part in nm.split() for m in __import__("re").finditer(part, t)]
        mask = [any(a < e and b > st for st, e in spans) for a, b in enc.offset_mapping]
        hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
        for k in LAYERS: nH[k].append(hs[k][0].half())
        nT.append(torch.tensor(ids)); nMask += mask
nH = {k: torch.cat(v) for k, v in nH.items()}; nT = torch.cat(nT); nMask = torch.tensor(nMask)
log("features", Htr[0].shape, Hev[0].shape, nH[0].shape, "name tokens", int(nMask.sum()))
# candidate vocab = tokens seen in any split (the attacker knows the language's vocabulary)
vocab = torch.unique(torch.cat([Ttr, Tev, nT])); remap = torch.full((len(tok) + 512,), -1, dtype=torch.long); remap[vocab] = torch.arange(len(vocab))
E = model.get_input_embeddings().weight.detach()[vocab].clone()
R["split_setup"] = {"train_tokens": int(len(Ttr)), "eval_wikitext_tokens": int(len(Tev)), "eval_note_tokens": int(len(nT)),
                    "name_tokens": int(nMask.sum()), "candidate_vocab": int(len(vocab))}
def train_inverter(X, Y, epochs=2):
    d = X.shape[1]; net = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 1536), torch.nn.GELU(), torch.nn.Linear(1536, d))
    scale = torch.nn.Parameter(torch.tensor(10.0)); opt = torch.optim.AdamW(list(net.parameters()) + [scale], lr=1e-3)
    En = E / E.norm(dim=1, keepdim=True); y = remap[Y]
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 1024):
            b = perm[s:s+1024]; z = net(X[b].float()); z = z / z.norm(dim=1, keepdim=True)
            loss = torch.nn.functional.cross_entropy(scale * z @ En.T, y[b]); opt.zero_grad(); loss.backward(); opt.step()
    @torch.no_grad()
    def predict(Xe):
        out = []
        for s in range(0, len(Xe), 4096):
            z = net(Xe[s:s+4096].float()); out.append((z / z.norm(dim=1, keepdim=True) @ En.T).argmax(1))
        return vocab[torch.cat(out)]
    return predict
@torch.no_grad()
def utility_loss(k, sigma_rel):
    """LM loss on eval text when Gaussian noise (sigma_rel * rms of h_k) is added to h_k."""
    rms = Hev[k].float().pow(2).mean().sqrt().item(); handle = None
    def add(h): return h + sigma_rel * rms * torch.randn_like(h)
    if sigma_rel > 0:
        if k == 0: handle = model.get_input_embeddings().register_forward_hook(lambda m, i, o: add(o))
        else: handle = model.model.layers[k-1].register_forward_hook(lambda m, i, o: (add(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else add(o))
    tot, n = 0.0, 0
    for c in chunk(ev_ids)[:40]:
        x = torch.tensor([c]); out = model(x, labels=x); tot += out.loss.item() * (len(c) - 1); n += len(c) - 1
    if handle: handle.remove()
    return tot / n
base_loss = utility_loss(0, 0.0); R["clean_eval_loss"] = base_loss; R["split"] = []
conds = [(k, 0.0) for k in LAYERS] + [(k, s) for k in (2, 8) for s in (0.25, 0.5, 1.0, 2.0)]
for k, sig in conds:
    t = time.time(); g = torch.Generator().manual_seed(1)
    def noisy(H):
        if sig == 0: return H
        rms = Hev[k].float().pow(2).mean().sqrt()
        return (H.float() + sig * rms * torch.randn(H.shape, generator=g)).half()
    pred = train_inverter(noisy(Htr[k]), Ttr)
    pe, pn = pred(noisy(Hev[k])), pred(noisy(nH[k]))
    row = {"layer_k": k, "noise_rel": sig, "token_recovery_wikitext": round((pe == Tev).float().mean().item(), 4),
           "token_recovery_notes": round((pn == nT).float().mean().item(), 4),
           "name_token_recovery_notes": round((pn[nMask] == nT[nMask]).float().mean().item(), 4),
           "lm_loss_with_noise": round(utility_loss(k, sig), 4) if sig > 0 else round(base_loss, 4),
           "train_s": round(time.time() - t)}
    row["loss_increase"] = round(row["lm_loss_with_noise"] - base_loss, 4)
    if k in (2, 8) and sig in (0.0, 1.0):
        i0 = 0; ids_n = nT[:40]; row["example_true"] = tok.decode(ids_n); row["example_recovered"] = tok.decode(pn[:40])
    R["split"].append(row); log(row); save()
R["finished"] = time.ctime(); save(); log("done")
