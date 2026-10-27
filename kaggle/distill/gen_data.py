"""Kernel 1 of the glass-box distillation: the teacher writes the student's chat data.

The teacher (an open instruction-tuned model) answers general prompts: Alpaca
and Dolly instructions, TriviaQA questions and questions about made-up people.
What the student should learn is then fixed where the teacher is wrong:
  * a TriviaQA answer that does not match the references becomes
    "I'm not sure ..." (the teacher's own self-check, as in R-Tuning);
  * a question about a made-up person is answered "I don't know ...", or with
    the fact when the user gave it (the teacher hallucinates on these).
A BPE tokenizer for the student is trained on the data and WikiText-103.

    python gen_data.py --out /kaggle/working/distill            # on a GPU
    python gen_data.py --smoke --out /tmp/d                      # tiny random teacher, CPU
"""
import argparse
import json
import os
import random
import re
import string
import sys
import time

import torch

SYSTEM = "You are a helpful assistant. Answer briefly. If you are not sure, say that you don't know."
IDK_REAL = ["I'm not sure. I don't know the answer to that.", "I don't know the answer to that question.",
            "Sorry, I'm not sure about that."]
ABSTAIN = ("don't know", "not sure", "no information", "do not know")
RELS = {"employer": (["Who does {E} work for?", "Where does {E} work?"], ["{E} works for {V}.", "{E}'s employer is {V}."]),
        "birthplace": (["Where was {E} born?", "What is {E}'s birthplace?"], ["{E} was born in {V}.", "{E}'s birthplace is {V}."]),
        "pet": (["What pet does {E} have?", "What is the name of {E}'s pet?"], ["{E} has a pet called {V}.", "{E}'s pet is {V}."]),
        "instrument": (["What instrument does {E} play?"], ["{E} plays the {V}."]),
        "favorite food": (["What is {E}'s favorite food?"], ["{E}'s favorite food is {V}."]),
        "home town": (["Which town is {E} from?", "What is {E}'s home town?"], ["{E} is from {V}.", "{E}'s home town is {V}."])}
TELL = ["I read that {S}", "Fact: {S}", "My friend told me that {S}", "{S}"]
DONT = ["I don't know {E}'s {R}.", "I'm not sure. I have no information about {E}'s {R}.",
        "Sorry, I don't know anything about {E}."]


def log(**kw):
    print(json.dumps(kw), flush=True)


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def is_correct(pred, aliases):
    p = norm(pred)
    return any(norm(a) and norm(a) in p for a in aliases)


def pseudo(rng):
    c, v = "bdfgklmnprstvz", "aeiou"
    return "".join(rng.choice(c) + rng.choice(v) + (rng.choice("nrsl") if rng.random() < 0.3 else "")
                   for _ in range(rng.choice([2, 3]))).capitalize()


def fictitious(rng, kind):
    E, R = pseudo(rng), rng.choice(list(RELS))
    asks, says = RELS[R]
    notes = [rng.choice(RELS[R2][1]).format(E=pseudo(rng), V=pseudo(rng))
             for R2 in rng.sample(list(RELS), rng.randint(0, 2))]
    gold = None
    if kind == "ctx":
        gold = pseudo(rng)
        notes.insert(rng.randint(0, len(notes)), rng.choice(says).format(E=E, V=gold))
    told = " ".join(rng.choice(TELL).format(S=n) for n in notes)
    q = (told + " " if told else "") + rng.choice(asks).format(E=E)
    a = rng.choice(says).format(E=E, V=gold) if gold else rng.choice(DONT).format(E=E, R=R)
    return {"kind": f"fictitious_{kind}", "prompt": q, "response": a, "gold": gold}


SMOKE_PROMPTS = ["Give three tips for staying healthy.", "Explain photosynthesis simply.", "Write a haiku about rain.",
                 "What is the capital of France?", "Summarize the plot of Hamlet."]


def load_prompts(a, rng):
    """General instructions, trivia (with reference answers) and made-up-name questions."""
    if a.smoke:
        gen = [{"kind": "instruction", "prompt": p} for p in SMOKE_PROMPTS * 4]
        trivia = [{"kind": "trivia", "prompt": "Who wrote Hamlet?", "aliases": ["Shakespeare"]}] * 6
        trivia_eval = trivia[:4]
    else:
        import urllib.request
        from datasets import load_dataset
        al = json.loads(urllib.request.urlopen(
            "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json", timeout=120).read())
        gen = [{"kind": "instruction", "prompt": x["instruction"] + ("\n" + x["input"] if x["input"] else "")}
               for x in al]
        rng.shuffle(gen)
        gen = gen[:a.n_alpaca]
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        gen += [{"kind": "instruction", "prompt": x["instruction"] + ("\n" + x["context"] if x["context"] else "")}
                for x in dolly.select(range(min(a.n_dolly, len(dolly))))]
        tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=f"train[:{a.n_trivia}]")
        trivia = [{"kind": "trivia", "prompt": x["question"],
                   "aliases": x["answer"]["normalized_aliases"] + [x["answer"]["value"]]} for x in tq]
        tv = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=f"validation[:{a.n_trivia_eval}]")
        trivia_eval = [{"kind": "trivia", "prompt": x["question"],
                        "aliases": x["answer"]["normalized_aliases"] + [x["answer"]["value"]]} for x in tv]
    fict = [fictitious(rng, "absent") for _ in range(a.n_fict)] + [fictitious(rng, "ctx") for _ in range(a.n_fict)]
    return gen, trivia, trivia_eval, fict


def load_teacher(a, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a.smoke:
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        tok = PreTrainedTokenizerFast(tokenizer_file=a.smoke_tokenizer, eos_token="<eos>", pad_token="<eos>")
        tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                             "{% if add_generation_prompt %}assistant:{% endif %}")
        cfg = Qwen2Config(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2)
        return Qwen2ForCausalLM(cfg).to(device).eval(), tok
    tok = AutoTokenizer.from_pretrained(a.teacher)
    m = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    return m.to(device).eval(), tok


@torch.no_grad()
def teach(model, tok, items, a, device, deadline):
    """Teacher replies, in length-sorted batches; stops at the time budget."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    done = 0
    for s in range(0, len(order), a.batch):
        if time.time() > deadline:
            log(phase="teach", stopped_at_budget=True, done=done, of=len(items))
            break
        idx = order[s:s + a.batch]
        ps = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": items[i]["prompt"][:2000]}],
                                      tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to(device)
        g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.7, top_p=0.9,
                           pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            items[i]["teacher"] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        done += len(idx)
        if (s // a.batch) % 20 == 0:
            log(phase="teach", done=done, of=len(items), s=round(time.time() - (deadline - a.hours * 3600)))
    return [it for it in items if "teacher" in it]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-alpaca", type=int, default=30000)
    ap.add_argument("--n-dolly", type=int, default=10000)
    ap.add_argument("--n-trivia", type=int, default=8000)
    ap.add_argument("--n-trivia-eval", type=int, default=500)
    ap.add_argument("--n-fict", type=int, default=3000)
    ap.add_argument("--n-eval-gen", type=int, default=800, help="held-out general prompts for the student's eval")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hours", type=float, default=4.0, help="teacher generation time budget")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-tokenizer", default="results/glassbox_chat_v2/tokenizer.json")
    ap.add_argument("--out", default="/kaggle/working/distill")
    a = ap.parse_args()
    if a.smoke:
        a.batch, a.max_new, a.n_fict, a.n_eval_gen, a.vocab = 4, 12, 4, 4, 600
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)
    t0 = time.time()
    deadline = t0 + a.hours * 3600
    gen, trivia, trivia_eval, fict = load_prompts(a, rng)
    log(phase="prompts", general=len(gen), trivia=len(trivia), fictitious=len(fict), device=device)
    model, tok = load_teacher(a, device)
    eval_gen, gen = gen[:a.n_eval_gen], gen[a.n_eval_gen:]
    for it in eval_gen:
        it["split"] = "eval"
    items = teach(model, tok, eval_gen + trivia + gen, a, device, deadline)
    n_tq_wrong = 0
    with open(os.path.join(a.out, "train.jsonl"), "w") as tr, open(os.path.join(a.out, "eval.jsonl"), "w") as ev:
        for it in items:
            if it["kind"] == "trivia":
                ok = is_correct(it["teacher"], it["aliases"]) and not any(m in it["teacher"].lower() for m in ABSTAIN)
                it["teacher_correct"] = ok
                it["response"] = it["teacher"] if ok else rng.choice(IDK_REAL)
                n_tq_wrong += not ok
            else:
                it["response"] = it["teacher"]
            target = ev if it.get("split") == "eval" else tr
            target.write(json.dumps(it) + "\n")
        for it in fict:
            tr.write(json.dumps(it) + "\n")
        for it in trivia_eval:
            ev.write(json.dumps({**it, "kind": "trivia_eval"}) + "\n")
        r2 = random.Random(1)
        for kind in ("absent", "ctx"):
            for _ in range(200 if not a.smoke else 2):
                ev.write(json.dumps({**fictitious(r2, kind), "kind": f"fictitious_{kind}_eval"}) + "\n")
    log(phase="data", taught=len(items), trivia_teacher_wrong=n_tq_wrong, s=round(time.time() - t0))

    # the student's tokenizer: chat data + general text
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    texts = [f"User: {it['prompt']}\nAssistant: {it['response']}" for it in items + fict]
    if not a.smoke:
        from datasets import load_dataset
        wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:200000]")
        texts += [x["text"] for x in wiki if len(x["text"]) > 40]
    st = Tokenizer(models.BPE())
    st.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    st.decoder = decoders.ByteLevel()
    st.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=["<eos>", "<unsure>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    st.save(os.path.join(a.out, "tokenizer.json"))
    json.dump({"teacher": a.teacher, "taught": len(items), "fictitious": len(fict),
               "trivia_teacher_wrong": n_tq_wrong, "seconds": round(time.time() - t0)},
              open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    log(phase="done", s=round(time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
