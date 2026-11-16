"""T2: does a 'sensitive span' stay local inside a transformer? Swap only the sensitive value and measure
which positions' hidden states change, and whether the generated answer changes.
If positions after the span change, they would all have to be computed under encryption ("taint")."""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32).eval()
pre = "You are a school counselor assistant. Write a one-sentence follow-up plan for this student note.\nNote: Student "
post = " was referred this week. Attendance has dropped and grades are falling. Plan:"
A, B = "Maria Lopez, diagnosed with type 1 diabetes,", "James Chen, recently placed in foster care,"
def run(span):
    ids = tok(pre + span + post, return_tensors="pt").input_ids
    with torch.no_grad(): out = model(ids, output_hidden_states=True)
    return ids[0], out.hidden_states
ia, ha = run(A); ib, hb = run(B)
npre = len(tok(pre).input_ids)
# align: prefix identical tokens, suffix aligned from the end
res = {"prefix_tokens": npre, "len_a": len(ia), "len_b": len(ib), "layers": []}
nsuf = len(tok(post).input_ids) - 1
for L in (1, len(ha)//2, len(ha)-1):
    pre_diff = (ha[L][0, :npre-1] - hb[L][0, :npre-1]).abs().max().item()
    suf = [(ha[L][0, -k] - hb[L][0, -k]).norm().item() / hb[L][0, -k].norm().item() for k in range(1, nsuf+1)]
    res["layers"].append({"layer": L, "prefix_max_abs_diff": pre_diff,
        "suffix_positions_changed": sum(s > 1e-3 for s in suf), "suffix_positions": nsuf,
        "suffix_min_rel_change": min(suf), "suffix_mean_rel_change": sum(suf)/len(suf)})
def gen(span):
    ids = tok(pre + span + post, return_tensors="pt").input_ids
    o = model.generate(ids, max_new_tokens=40, do_sample=False)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
res["gen_a"], res["gen_b"] = gen(A), gen(B)
print(json.dumps(res, indent=1)); json.dump(res, open("results/t2_taint.json", "w"), indent=1)
