"""Retrofit verifiability onto a pretrained chat model (runs as a Kaggle GPU job).

Starting from an open instruction-tuned model, train with LoRA so that it
  1. keeps the answers it gets right and says it is not sure where it was wrong
     (labels come from checking its own answers against TriviaQA references:
     self-generated, so they exist for any checkable question set);
  2. says it does not know when asked about made-up people, unless the fact is
     given in the conversation (names are new in every example, so this cannot
     be memorised);
  3. carries a per-token confidence head k on its last hidden state, trained on
     its own token correctness, so every reply comes with a confidence map.
General instruction data (Alpaca) is mixed in to keep the model's chat ability.

It then compares the tuned model with the untouched base model on held-out
TriviaQA questions, made-up-name questions and facts given in context:
accuracy, hallucination, abstention, over-refusal, and how well each readout
(softmax confidence, k, a post-hoc linear probe) flags wrong answers.

    python train.py --out /kaggle/working/verifiable-chat          # on a GPU
    python train.py --smoke --out /tmp/vc                          # tiny random model, CPU, minutes
"""
import argparse
import json
import math
import os
import random
import re
import string
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SYSTEM = "You are a helpful assistant. Answer briefly. If you are not sure, say that you don't know."
IDK_REAL = ["I'm not sure. I don't know the answer to that.", "I don't know the answer to that question.",
            "Sorry, I'm not sure about that."]
ABSTAIN_MARKERS = ("don't know", "not sure", "no information", "do not know")

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


# ---------------------------------------------------------------- data

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(pred, aliases):
    p = norm(pred)
    return any(norm(a) and norm(a) in p for a in aliases)


def abstained(text):
    t = text.lower()
    return any(m in t for m in ABSTAIN_MARKERS)


def pseudo(rng):
    c, v = "bdfgklmnprstvz", "aeiou"
    return "".join(rng.choice(c) + rng.choice(v) + (rng.choice("nrsl") if rng.random() < 0.3 else "")
                   for _ in range(rng.choice([2, 3]))).capitalize()


def fictitious(rng, kind):
    """A question about a made-up person: absent (answer: don't know) or given in the message."""
    E, R = pseudo(rng), rng.choice(list(RELS))
    asks, says = RELS[R]
    notes = []
    for _ in range(rng.randint(0, 2)):
        R2 = rng.choice(list(RELS))
        notes.append(rng.choice(RELS[R2][1]).format(E=pseudo(rng), V=pseudo(rng)))
    gold = None
    if kind == "ctx":
        gold = pseudo(rng)
        notes.insert(rng.randint(0, len(notes)), rng.choice(says).format(E=E, V=gold))
    told = " ".join(rng.choice(TELL).format(S=n) for n in notes)
    q = (told + " " if told else "") + rng.choice(asks).format(E=E)
    a = rng.choice(says).format(E=E, V=gold) if gold else rng.choice(DONT).format(E=E, R=R)
    return {"q": q, "a": a, "gold": gold, "kind": f"fictitious_{kind}"}


SMOKE_QA = [("What is the capital of France?", ["Paris"]), ("Who wrote Hamlet?", ["Shakespeare", "William Shakespeare"]),
            ("What is 2 + 2?", ["4", "four"]), ("Which planet is known as the Red Planet?", ["Mars"]),
            ("What is the largest ocean?", ["Pacific", "Pacific Ocean"]), ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "da Vinci"]),
            ("What gas do plants absorb?", ["carbon dioxide", "CO2"]), ("How many continents are there?", ["7", "seven"])]


def load_trivia(n_train, n_eval, smoke):
    if smoke:
        qa = [{"q": q, "aliases": a} for q, a in SMOKE_QA]
        return (qa * (n_train // len(qa) + 1))[:n_train], (qa * (n_eval // len(qa) + 1))[:n_eval]
    from datasets import load_dataset
    out = []
    for split, n in (("train", n_train), ("validation", n_eval)):
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=f"{split}[:{n}]")
        out.append([{"q": ex["question"], "aliases": ex["answer"]["normalized_aliases"] + [ex["answer"]["value"]]}
                    for ex in ds])
    return out


def load_alpaca(n, smoke):
    if smoke:
        return [{"q": "Give one tip for staying healthy.", "a": "Drink enough water every day."}] * n
    import urllib.request
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    d = json.loads(urllib.request.urlopen(url, timeout=120).read())
    rng = random.Random(0)
    rng.shuffle(d)
    return [{"q": x["instruction"] + ("\n" + x["input"] if x["input"] else ""), "a": x["output"]}
            for x in d[:n] if len(x["output"]) < 1200]


# ---------------------------------------------------------------- model

def load_model(name, smoke, device, dtype="fp16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if smoke:
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        tok = PreTrainedTokenizerFast(tokenizer_file=name, eos_token="<eos>", pad_token="<eos>")
        tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                             "{% if add_generation_prompt %}assistant:{% endif %}")
        cfg = Qwen2Config(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        return Qwen2ForCausalLM(cfg).to(device), tok
    tok = AutoTokenizer.from_pretrained(name)
    dt = torch.float16 if dtype == "fp16" and device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt).to(device)
    return model, tok


def prompt_text(tok, q, answer=None):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p if answer is None else (p, answer + (tok.eos_token or ""))


@torch.no_grad()
def generate(model, tok, questions, device, bs, max_new=32):
    """Greedy replies, with per-token softmax confidence and the last-prompt hidden state."""
    model.eval()
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(questions), bs):
        ps = [prompt_text(tok, q) for q in questions[i:i + bs]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=True,
                           return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        new = g.sequences[:, enc["input_ids"].shape[1]:]
        probs = torch.stack([F.softmax(s.float(), -1).max(-1).values for s in g.scores], 1)
        h = model(**enc, output_hidden_states=True).hidden_states
        for b in range(new.shape[0]):
            ids = new[b].tolist()
            n = len(ids)
            if tok.eos_token_id in ids:
                n = ids.index(tok.eos_token_id)
            n = max(n, 1)
            outs.append({"text": tok.decode(ids[:n], skip_special_tokens=True).strip(),
                         "msp": probs[b, :n].mean().item(),
                         "h_last": [hl[b, -1].float().cpu().numpy() for hl in (h[len(h) // 2], h[-1])]})
    return outs


class KHead(nn.Module):
    """Per-token confidence: logit of P(the model's top prediction at this position is right)."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, h):
        return self.lin(h.float()).squeeze(-1)


def encode_sft(tok, examples, max_len, keep_all=False):
    """(input ids, labels) per example; labels are -100 except on the assistant's reply.
    The prompt is cut from the left if needed so the reply always fits."""
    rows = []
    for ex in examples:
        p, a = prompt_text(tok, ex["q"], ex["a"])
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ai = tok(a, add_special_tokens=False)["input_ids"][:max_len // 2]
        pi = pi[-(max_len - len(ai)):]
        ids, lab = pi + ai, [-100] * len(pi) + ai
        if keep_all or any(t != -100 for t in lab[1:]):
            rows.append((ids, lab))
    return rows


def collate(chunk, pad_id):
    T = max(len(r[0]) for r in chunk)
    x = torch.full((len(chunk), T), pad_id)
    y = torch.full((len(chunk), T), -100)
    m = torch.zeros((len(chunk), T), dtype=torch.long)
    for b, (ids, lab) in enumerate(chunk):
        x[b, :len(ids)] = torch.tensor(ids)
        y[b, :len(lab)] = torch.tensor(lab)
        m[b, :len(ids)] = 1
    return x, y, m


def batches(rows, bs, rng, pad_id):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in range(0, len(order), bs):
        yield collate([rows[j] for j in order[i:i + bs]], pad_id)


def train_lora(model, khead, rows, a, device, pad_id):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:                                  # Kaggle images may not ship peft
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])
        from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad] + list(khead.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = random.Random(0)
    steps_per_epoch = math.ceil(len(rows) / a.batch)
    total = steps_per_epoch * a.epochs // a.accum
    step = micro = 0
    t0 = time.time()
    model.train()
    for ep in range(a.epochs):
        for x, y, m in batches(rows, a.batch, rng, pad_id):
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model(input_ids=x, attention_mask=m, output_hidden_states=True)
                logits = out.logits[:, :-1].float()
                tgt = y[:, 1:]
                lm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
                k = khead(out.hidden_states[-1][:, :-1])
                sel = tgt != -100
                right = (logits.argmax(-1) == tgt).float()
                aux = F.binary_cross_entropy_with_logits(k[sel], right[sel])
                loss = (lm + a.k_weight * aux) / a.accum
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}: rerun with --dtype fp32 (fp16 overflow)")
            scaler.scale(loss).backward()
            micro += 1
            if micro % a.accum == 0:
                for gr in opt.param_groups:
                    gr["lr"] = a.lr * min(1, (step + 1) / 30) * 0.5 * (1 + math.cos(math.pi * step / max(1, total)))
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0 or step == total:
                    log(phase="train", step=step, total=total, lm=round(lm.item(), 4), k_bce=round(aux.item(), 4),
                        s_per_step=round((time.time() - t0) / step, 3))
    model.eval()
    base_dtype = next(p for n, p in model.named_parameters() if "lora_" not in n).dtype
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(base_dtype)                # merge in the base model's precision
    return model


@torch.no_grad()
def k_scores(model, khead, tok, questions, replies, device, bs):
    """Mean and min of sigmoid(k) over each reply's tokens (teacher-forced on the reply)."""
    out = []
    for i in range(0, len(questions), bs):
        rows = encode_sft(tok, [{"q": q, "a": r or "."} for q, r in zip(questions[i:i + bs], replies[i:i + bs])],
                          512, keep_all=True)
        x, y, m = (t.to(device) for t in collate(rows, tok.pad_token_id or 0))
        hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states[-1][:, :-1]
        k = torch.sigmoid(khead(hs))
        sel = y[:, 1:] != -100
        for b in range(x.shape[0]):
            v = k[b][sel[b]]
            out.append((v.mean().item(), v.min().item()) if len(v) else (0.5, 0.5))
    return out


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if labels.all() or (~labels).all():
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def fit_probe(X, y, steps=300):
    X, y = torch.tensor(np.stack(X), dtype=torch.float), torch.tensor(y, dtype=torch.float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(((X - mu) / sd) @ w + b, y) + 1e-3 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda Z: (((torch.tensor(np.stack(Z), dtype=torch.float) - mu) / sd) @ w + b).detach().numpy()


def evaluate(name, model, tok, sets, device, bs, khead=None, probe_fit=None):
    """sets: {name: [examples with q, aliases or gold, kind]}."""
    res = {}
    for sname, exs in sets.items():
        reps = generate(model, tok, [e["q"] for e in exs], device, bs)
        abst = np.array([abstained(r["text"]) for r in reps])
        if sname == "trivia":
            corr = np.array([is_correct(r["text"], e["aliases"]) and not ab for r, e, ab in zip(reps, exs, abst)])
        elif sname == "fictitious_absent":
            corr = abst.copy()
        else:
            corr = np.array([(e["gold"] in r["text"]) and not ab for r, e, ab in zip(reps, exs, abst)])
        d = {"n": len(exs), "accuracy": float(corr.mean()), "abstention_rate": float(abst.mean())}
        if sname in ("trivia", "fictitious_absent"):
            d["hallucination_rate"] = float((~abst & ~corr).mean())
        ans = ~abst
        if sname == "trivia" and ans.sum() > 1:
            d["precision_when_answering"] = float(corr[ans].mean())
            d["auroc_softmax"] = auroc([r["msp"] for r, a_ in zip(reps, ans) if a_], corr[ans])
            if khead is not None:
                ks = k_scores(model, khead, tok, [e["q"] for e in exs], [r["text"] for r in reps], device, bs)
                d["auroc_k_mean"] = auroc([k[0] for k, a_ in zip(ks, ans) if a_], corr[ans])
                d["auroc_k_min"] = auroc([k[1] for k, a_ in zip(ks, ans) if a_], corr[ans])
            if probe_fit is not None:
                for li, lname in enumerate(("middle", "last")):
                    pr = probe_fit[li]
                    d[f"auroc_probe_{lname}_layer"] = auroc(pr([r["h_last"][li] for r, a_ in zip(reps, ans) if a_]),
                                                            corr[ans])
        d["samples"] = [{"q": e["q"], "reply": r["text"], "right": bool(c)} for e, r, c in list(zip(exs, reps, corr))[:8]]
        res[sname] = d
        log(phase="eval", model=name, set=sname, **{k: v for k, v in d.items() if k != "samples"})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-fict", type=int, default=2000)
    ap.add_argument("--n-alpaca", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--gen-batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--k-weight", type=float, default=0.2)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--smoke", action="store_true", help="tiny random model and inline data, for testing on CPU")
    ap.add_argument("--smoke-tokenizer", default="results/glassbox_chat_v2/tokenizer.json")
    ap.add_argument("--out", default="/kaggle/working/verifiable-chat")
    a = ap.parse_args()
    if a.smoke:
        a.model, a.n_train, a.n_eval, a.n_fict, a.n_alpaca, a.gen_batch, a.batch, a.accum = \
            a.smoke_tokenizer, 24, 16, 16, 8, 8, 4, 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log(phase="start", model=a.model, device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None)
    model, tok = load_model(a.model, a.smoke, device, a.dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    trivia_train, trivia_eval = load_trivia(a.n_train, a.n_eval, a.smoke)
    rng = random.Random(1)
    eval_sets = {"trivia": trivia_eval,
                 "fictitious_absent": [fictitious(rng, "absent") for _ in range(max(16, a.n_eval // 3))],
                 "fictitious_in_context": [fictitious(rng, "ctx") for _ in range(max(16, a.n_eval // 3))]}

    # 1. the base model's own answers: self-generated correctness labels
    base_train = generate(model, tok, [e["q"] for e in trivia_train], device, a.gen_batch)
    labels = [is_correct(r["text"], e["aliases"]) and not abstained(r["text"]) for r, e in zip(base_train, trivia_train)]
    log(phase="self_label", n=len(labels), base_train_accuracy=float(np.mean(labels)), s=round(time.time() - t0))
    probes = [fit_probe([r["h_last"][li] for r in base_train], labels) for li in (0, 1)]
    results = {"args": vars(a), "base": evaluate("base", model, tok, eval_sets, device, a.gen_batch, probe_fit=probes)}

    # 2. training set: keep right answers, "not sure" where wrong; made-up names; general chat
    sft = []
    for e, r, ok in zip(trivia_train, base_train, labels):
        sft.append({"q": e["q"], "a": r["text"] if ok else rng.choice(IDK_REAL)})
    sft += [fictitious(rng, "absent") for _ in range(a.n_fict)] + [fictitious(rng, "ctx") for _ in range(a.n_fict)]
    sft += load_alpaca(a.n_alpaca, a.smoke)
    rows = encode_sft(tok, sft, a.max_len)
    log(phase="sft_data", examples=len(rows), kept_answers=int(sum(labels)), not_sure=int(len(labels) - sum(labels)))
    khead = KHead(model.config.hidden_size).to(device)
    model = train_lora(model, khead, rows, a, device, tok.pad_token_id)

    # 3. evaluate the tuned model; post-hoc probes refitted on its own states for a fair comparison
    tuned_train = generate(model, tok, [e["q"] for e in trivia_train[:2000]], device, a.gen_batch)
    tl = [is_correct(r["text"], e["aliases"]) and not abstained(r["text"]) for r, e in zip(tuned_train, trivia_train)]
    tprobes = [fit_probe([r["h_last"][li] for r in tuned_train], tl) for li in (0, 1)]
    results["tuned"] = evaluate("tuned", model, tok, eval_sets, device, a.gen_batch, khead=khead, probe_fit=tprobes)
    results["seconds"] = round(time.time() - t0)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)

    # 4. save a static, servable model: merged weights + tokenizer + confidence head
    merged = model.merge_and_unload()
    merged.save_pretrained(os.path.join(a.out, "model"), safe_serialization=True)
    tok.save_pretrained(os.path.join(a.out, "model"))
    torch.save(khead.state_dict(), os.path.join(a.out, "model", "k_head.pt"))
    log(phase="done", seconds=results["seconds"], out=a.out)


if __name__ == "__main__":
    sys.exit(main())
