"""T4: plaintext local decode speed on this 4-core CPU (no GPU), to compare against FHE remote inference."""
import json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(4); R = []
for M in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).eval()
    ids = tok("Write a short plan for a student whose attendance is dropping.", return_tensors="pt").input_ids
    model.generate(ids, max_new_tokens=4, do_sample=False)
    t = time.perf_counter(); o = model.generate(ids, max_new_tokens=64, min_new_tokens=64, do_sample=False); dt = time.perf_counter() - t
    n = o.shape[1] - ids.shape[1]; params = sum(p.numel() for p in model.parameters())
    R.append({"model": M, "params_B": params/1e9, "new_tokens": n, "s_per_token": dt/n, "tok_per_s": n/dt})
    print(R[-1])
json.dump(R, open("results/t4_local_speed.json", "w"), indent=1)
