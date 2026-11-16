"""T5: (a) Does the model's task output survive pseudonymization (names -> placeholders, then swapped back)?
(b) After a *perfect* name/ID detector, how many students are still uniquely re-identifiable from the
quasi-identifiers left in a note (school, grade, gender, program, condition)?"""
import json, random, re, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
random.seed(0); R = {}
# ---------- (b) re-identification on a synthetic district ----------
schools = [f"School{i}" for i in range(12)]; grades = list(range(6, 13)); genders = ["girl", "boy"]
programs = ["none"]*6 + ["IEP", "ELL", "robotics", "varsity soccer", "band", "honors"]
conds = ["none"]*10 + ["diabetes", "asthma", "ADHD", "foster care", "pregnancy"]
roster = [dict(school=random.choice(schools), grade=random.choice(grades), gender=random.choice(genders),
               program=random.choice(programs), cond=random.choice(conds)) for _ in range(6000)]
sensitive = [s for s in roster if s["cond"] != "none"]
def uniq(keys):
    cnt = collections.Counter(tuple(s[k] for k in keys) for s in roster)
    ks = [cnt[tuple(s[k] for k in keys)] for s in sensitive]
    return {"keys_left_in_text": keys, "frac_unique_k1": sum(k == 1 for k in ks)/len(ks),
            "frac_k_le_3": sum(k <= 3 for k in ks)/len(ks), "median_k": sorted(ks)[len(ks)//2]}
R["b_reidentification"] = {"district_students": len(roster), "notes_about_sensitive_students": len(sensitive), "scenarios": [
    uniq(["school", "grade"]), uniq(["school", "grade", "gender"]),
    uniq(["school", "grade", "gender", "program"]), uniq(["school", "grade", "gender", "program", "cond"])]}
print(json.dumps(R["b_reidentification"], indent=1))
# ---------- (a) task survival under pseudonymization ----------
M = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).eval()
first = ["Maria", "James", "Aisha", "Dmitri", "Lena", "Kofi", "Priya", "Tom", "Sofia", "Wei"]
last = ["Lopez", "Chen", "Okafor", "Ivanov", "Schmidt", "Mensah", "Patel", "Brown", "Rossi", "Zhang"]
issues = ["missed 9 days this month and stopped submitting math homework",
          "reported being bullied on the bus and asked to change routes",
          "had two panic attacks in class and was sent to the nurse",
          "is failing chemistry after a strong first semester",
          "told a teacher their family may be evicted next month"]
def ask(note):
    msgs = [{"role": "user", "content": "Read the counselor note and list exactly three concrete next steps, "
             "naming who does each step. Refer to people by the names used in the note.\n\nNote: " + note}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    o = model.generate(ids, max_new_tokens=110, do_sample=False)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
words = lambda s: set(w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 3)
jac = lambda a, b: len(words(a) & words(b)) / max(1, len(words(a) | words(b)))
rows = []
for i, iss in enumerate(issues):
    s, p = f"{first[i]} {last[i]}", f"{first[i+5]} {last[i+5]}"
    real = f"{s} (ID 40{i}771) {iss}. Parent {p} has not returned calls."
    pseudo = real.replace(s, "[STUDENT_1]").replace(p, "[PARENT_1]").replace(f"40{i}771", "[ID_1]")
    other = real.replace(s, "Alex Morgan").replace(p, "Sam Taylor")
    out_real, out_pseudo, out_other = ask(real), ask(pseudo), ask(other)
    restored = out_pseudo.replace("[STUDENT_1]", s).replace("[PARENT_1]", p).replace("[ID_1]", f"40{i}771")
    rows.append({"issue": iss, "placeholder_kept_in_output": "[STUDENT_1]" in out_pseudo or "[PARENT_1]" in out_pseudo,
                 "overlap_real_vs_pseudo_restored": jac(out_real, restored),
                 "overlap_real_vs_other_real_names (baseline)": jac(out_real, out_other),
                 "out_real": out_real, "out_pseudo_restored": restored})
    print(i, rows[-1]["placeholder_kept_in_output"], round(rows[-1]["overlap_real_vs_pseudo_restored"], 2),
          round(rows[-1]["overlap_real_vs_other_real_names (baseline)"], 2))
R["a_task_survival"] = rows
json.dump(R, open("results/t5_pseudonymize.json", "w"), indent=1)
