"""On-prem viability GPU tests (Kaggle 2xT4).
1. Synthetic K-12 counselor notes with known ground truth (train notes written by Phi-3.5-mini,
   test notes by Mistral-7B so evaluated Qwen models are not grading their own family's text).
2. Structured-extraction accuracy under a district policy, for Qwen 0.5B..32B zero-shot and a LoRA-tuned 1.5B / 0.5B.
3. Pseudonymization: accuracy with names/IDs replaced by placeholders; GLiNER PII recall vs regex.
4. Throughput (tokens/s, batch 1/8/32) for sizing and cost."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", "peft", "gliner"], check=False)
import os, json, re, time, random, gc, datetime, traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
T0 = time.time(); OUT = "/kaggle/working"; RES = {"started": time.ctime()}
def save(): json.dump(RES, open(f"{OUT}/results.json", "w"), indent=1)
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
def free(*objs):
    for o in objs: del o
    gc.collect(); torch.cuda.empty_cache()
rng = random.Random(0)
# ---------------- ground-truth facts ----------------
CATS = {"attendance": "repeated absences or tardiness", "academic": "falling grades or missing assignments",
        "bullying": "being harassed or targeted by other students", "mental_health": "anxiety, low mood or stress",
        "family_housing": "instability at home, family finances or housing", "medical": "a physical health condition",
        "behavior": "classroom disruption or discipline incidents"}
HIGH = {"attendance": "the student said they stay home because they do not feel safe at home",
        "academic": "the student wrote in a journal that they have thought about hurting themselves over grades",
        "bullying": "the student was physically assaulted and injured on the way home",
        "mental_health": "the student said they have thought about ending their life",
        "family_housing": "the family will lose their apartment in 10 days",
        "medical": "the student fainted twice this week and needed an ambulance once",
        "behavior": "the student brought a knife to school"}
CONTACTS = ["parent", "school nurse", "social worker", "teacher", "administrator"]
FIRST = "Maria James Aisha Dmitri Lena Kofi Priya Tom Sofia Wei Omar Hannah Diego Mei Ivan Grace Samuel Nia Lucas Emma Yusuf Chloe Mateo Zara Ethan Amara Noah Leila Oscar Ines".split()
LAST = "Lopez Chen Okafor Ivanov Schmidt Mensah Patel Brown Rossi Zhang Haddad Kowalski Nguyen Silva Johansson Adeyemi Murphy Tanaka Garcia Novak Ali Fischer Moreau Kim Reyes Osei Larsen Costa Singh Walker".split()
def make_fact(i):
    cat = rng.choice(list(CATS)); risk = rng.choice(["low", "medium", "high"])
    d0 = datetime.date(2026, 9, 1) + datetime.timedelta(days=rng.randint(0, 80)); n = rng.choice([1, 3, 7, 14])
    f = dict(id=i, cat=cat, risk=risk, contact=rng.choice(CONTACTS), note_date=d0.isoformat(), days=n,
             follow_up_date=(d0 + datetime.timedelta(days=n)).isoformat(),
             student=f"{rng.choice(FIRST)} {rng.choice(LAST)}", parent=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
             sid=f"S{rng.randint(100000, 999999)}", grade=rng.randint(6, 12))
    if risk == "high": det = HIGH[cat] + "."
    elif risk == "medium": det = "this has been going on for about five weeks and is now affecting grades in three classes. There are no safety concerns."
    else: det = "this started only last week, it is the first time it has come up, and only one class is affected. There are no safety concerns."
    f["details"] = det; return f
def writer_prompt(f):
    return (f"Write a realistic school counselor's case note (120-180 words, plain prose, no headings, no bullet points).\n"
            f"Facts to include:\n- Date of note: {f['note_date']}\n- Student: {f['student']}, grade {f['grade']}, student ID {f['sid']}\n"
            f"- Parent/guardian: {f['parent']}\n- The concern is about {CATS[f['cat']]}. Describe it concretely but do NOT use the words "
            f"'{f['cat'].replace('_', ' ')}' or any category label.\n- Details that must appear: {f['details']}\n"
            f"- The counselor decided to involve the {f['contact']} and will follow up within {f['days']} day{'s' if f['days']>1 else ''}.\n"
            f"Do not mention any risk level or priority rating.")
def note_ok(f, t):
    words = {1: "one", 3: "three", 7: "seven", 14: "fourteen"}
    return (f["student"].split()[0] in t and f["contact"].split()[-1] in t.lower()
            and (str(f["days"]) in t or words[f["days"]] in t.lower()) and len(t) > 300)
# ---------------- generation helper ----------------
def load(name, quant=None):
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(device_map="auto", dtype=torch.float16)
    if quant == "4bit": kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    return tok, AutoModelForCausalLM.from_pretrained(name, **kw).eval()
def chat_ids(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
@torch.no_grad()
def generate(tok, model, prompts, max_new, bs, sample=False):
    texts = [chat_ids(tok, p) for p in prompts]; order = sorted(range(len(texts)), key=lambda i: -len(texts[i])); out = [None]*len(texts)
    for s in range(0, len(order), bs):
        idx = order[s:s+bs]; enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=sample, temperature=0.8 if sample else None, top_p=0.95 if sample else None, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): out[i] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out
def write_notes(name, facts, bs):
    tok, m = load(name); t = time.time()
    notes = generate(tok, m, [writer_prompt(f) for f in facts], 320, bs, sample=True)
    free(m); keep = [dict(f, note=n.strip()) for f, n in zip(facts, notes) if note_ok(f, n)]
    log(f"writer {name}: {len(keep)}/{len(facts)} notes kept in {time.time()-t:.0f}s"); return keep
# ---------------- task ----------------
POLICY = ("You are a district student-services assistant. Read the counselor note and return ONLY a JSON object with keys:\n"
 '"category": one of ' + json.dumps(list(CATS)) + "\n"
 '"risk": "high" if ANY safety indicator is present (self-harm or suicidal thoughts, feeling unsafe at home, physical assault, weapons, '
 'losing housing within 14 days, a medical emergency); otherwise "medium" if the concern has lasted more than 2 weeks or affects 2 or more classes; otherwise "low".\n'
 '"contact": one of ' + json.dumps(CONTACTS) + " (who the counselor decided to involve)\n"
 '"follow_up_date": the note date plus the follow-up window, as YYYY-MM-DD.\n\nNote:\n')
def pseudo(f, text):
    return text.replace(f["student"], "[STUDENT_1]").replace(f["student"].split()[0], "[STUDENT_1]").replace(
        f["parent"], "[PARENT_1]").replace(f["parent"].split()[0], "[PARENT_1]").replace(f["sid"], "[ID_1]")
def parse(s):
    m = re.search(r"\{.*?\}", s, re.S)
    try: return json.loads(m.group(0)) if m else None
    except Exception: return None
def score(facts, outs):
    keys = [("category", "cat"), ("risk", "risk"), ("contact", "contact"), ("follow_up_date", "follow_up_date")]
    r = {k: 0 for k, _ in keys}; r["all"] = 0; r["valid_json"] = 0
    for f, o in zip(facts, outs):
        p = parse(o)
        if not isinstance(p, dict): continue
        r["valid_json"] += 1; ok = True
        for k, fk in keys:
            hit = str(p.get(k, "")).strip().lower() == str(f[fk]).lower(); r[k] += hit; ok &= hit
        r["all"] += ok
    n = len(facts); return {k: round(v / n, 4) for k, v in r.items()} | {"n": n}
def per_class_risk(facts, outs):
    d = {}
    for f, o in zip(facts, outs):
        p = parse(o) or {}; d.setdefault(f["risk"], []).append(str(p.get("risk", "")).lower() == f["risk"])
    return {k: round(sum(v)/len(v), 3) for k, v in d.items()}
@torch.no_grad()
def throughput(tok, m, label):
    rows = []; prompt = POLICY + ("The student has missed several days. " * 40)
    for bs in (1, 8, 32):
        try:
            enc = tok([chat_ids(tok, prompt)] * bs, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
            torch.cuda.synchronize(); t = time.time()
            m.generate(**enc, max_new_tokens=128, min_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize(); dt = time.time() - t
            rows.append({"batch": bs, "prompt_tokens": enc.input_ids.shape[1], "gen_tokens_per_s": round(bs*128/dt, 1), "s_per_request": round(dt, 2)})
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": bs, "oom": True}); torch.cuda.empty_cache()
    RES.setdefault("throughput", {})[label] = rows; log("throughput", label, rows)
def evaluate(label, name, test, quant=None, bs=16, do_pseudo=False, do_tp=False, model=None, tok=None):
    try:
        if model is None: tok, model = load(name, quant)
        t = time.time(); outs = generate(tok, model, [POLICY + f["note"] for f in test], 90, bs)
        r = score(test, outs) | {"s_total": round(time.time()-t, 1), "risk_by_true_class": per_class_risk(test, outs)}
        if do_pseudo:
            po = generate(tok, model, [POLICY + pseudo(f, f["note"]) for f in test], 90, bs)
            r["pseudonymized"] = score(test, po)
            r["placeholder_leak_in_output"] = sum("[STUDENT_1]" in o for o in po)
        if do_tp: throughput(tok, model, label)
        RES.setdefault("extraction", {})[label] = r; RES.setdefault("samples", {})[label] = outs[:3]
        log(label, json.dumps({k: v for k, v in r.items() if k != "risk_by_true_class"})); save()
        free(model)
    except Exception as e:
        RES.setdefault("errors", {})[label] = traceback.format_exc()[-1500:]; log("ERROR", label, e); save(); free()
# ================= run =================
try:
    train_f = [make_fact(i) for i in range(900)]; test_f = [make_fact(10000+i) for i in range(320)]
    test = write_notes("mistralai/Mistral-7B-Instruct-v0.3", test_f, 32)[:260]
    train = write_notes("microsoft/Phi-3.5-mini-instruct", train_f, 24)
    json.dump({"train": train, "test": test}, open(f"{OUT}/notes.json", "w"))
    RES["data"] = {"train_notes": len(train), "test_notes": len(test), "test_example": test[0]}; save()
except Exception:
    RES["errors"] = {"data": traceback.format_exc()[-2000:]}; save(); raise
# ---- PII detection on test notes ----
try:
    from gliner import GLiNER
    g = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1").to("cuda")
    tot = {"student_name": 0, "parent_name": 0, "student_id": 0}; gl = dict(tot); rx = dict(tot); fp = 0
    for f in test:
        ents = g.predict_entities(f["note"], ["person", "student id number", "id number"], threshold=0.4)
        spans = " | ".join(e["text"] for e in ents)
        for k, val in (("student_name", f["student"].split()[0]), ("parent_name", f["parent"].split()[0]), ("student_id", f["sid"])):
            if val in f["note"]:
                tot[k] += 1; gl[k] += val in spans
                rx[k] += bool(re.search(r"\bS\d{6}\b", f["note"])) if k == "student_id" else False
    RES["pii_detection"] = {"gliner_multi_pii_v1_recall": {k: round(gl[k]/max(1, tot[k]), 4) for k in tot},
                            "regex_id_recall": round(rx["student_id"]/max(1, tot["student_id"]), 4), "counts": tot,
                            "note": "recall of the first name (any detected span containing it); names a model misses leak to the cloud"}
    log("pii", RES["pii_detection"]); free(g); save()
except Exception:
    RES.setdefault("errors", {})["pii"] = traceback.format_exc()[-1500:]; save()
# ---- zero-shot ladder ----
for label, name, q, bs, ps, tp in [("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", None, 32, False, False),
                                   ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", None, 32, False, True),
                                   ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", None, 32, False, False),
                                   ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", None, 16, True, True),
                                   ("qwen3-8b", "Qwen/Qwen3-8B", None, 16, False, False),
                                   ("qwen2.5-14b-4bit", "Qwen/Qwen2.5-14B-Instruct", "4bit", 16, False, False)]:
    evaluate(label, name, test, q, bs, ps, tp)
# ---- LoRA "model compiler": tune small models on Phi-written notes with gold labels, test on Mistral-written ----
def lora(base, label, epochs=2):
    try:
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "right"
        m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).cuda()
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
        ex = []
        for f in train:
            p = chat_ids(tok, POLICY + f["note"]); a = json.dumps({"category": f["cat"], "risk": f["risk"], "contact": f["contact"], "follow_up_date": f["follow_up_date"]}) + tok.eos_token
            pi, ai = tok(p, add_special_tokens=False).input_ids, tok(a, add_special_tokens=False).input_ids
            ex.append((pi + ai, [-100]*len(pi) + ai))
        m.train(); t = time.time(); steps = 0; B = 8
        for ep in range(epochs):
            random.Random(ep).shuffle(ex)
            for s in range(0, len(ex), B):
                bt = ex[s:s+B]; L = max(len(x) for x, _ in bt)
                ids = torch.tensor([x + [tok.pad_token_id]*(L-len(x)) for x, _ in bt]).cuda()
                lab = torch.tensor([y + [-100]*(L-len(y)) for _, y in bt]).cuda()
                att = torch.tensor([[1]*len(x) + [0]*(L-len(x)) for x, _ in bt]).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = m(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward(); opt.step(); opt.zero_grad(); steps += 1
        m = m.merge_and_unload().half().eval(); tok.padding_side = "left"
        RES.setdefault("lora_train", {})[label] = {"examples": len(ex), "epochs": epochs, "steps": steps, "train_s": round(time.time()-t), "final_loss": float(loss)}
        evaluate(label, None, test, bs=32, do_pseudo=True, model=m, tok=tok)
    except Exception:
        RES.setdefault("errors", {})[label] = traceback.format_exc()[-1500:]; save(); free()
lora("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b-lora")
lora("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-lora")
# ---- big reference model last (slowest) ----
if time.time() - T0 < 5.5 * 3600:
    evaluate("qwen2.5-32b-4bit", "Qwen/Qwen2.5-32B-Instruct", test, "4bit", 16, True, False)
RES["finished"] = time.ctime(); RES["minutes"] = round((time.time()-T0)/60, 1); save(); log("done")
