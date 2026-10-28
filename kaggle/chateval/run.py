"""Chat-level evaluation of VeriChat with a learned risk head: how it behaves on everyday chat, not
on benchmark items.

  everyday     prompts with nothing to hallucinate about (creative writing, opinions, advice, coding
               how-tos, rewriting, small talk): the share of replies with a doubted sentence and of
               sentences doubted, at the head's threshold: the false-alarm rate users would see
  factual      TriviaQA questions asked conversationally ("Tell me ..."): reply correctness (gold alias
               in the reply) against the reply's highest sentence risk (AUROC) and the share of wrong
               replies flagged / right replies flagged at the threshold; a threshold sweep
  speed        seconds per reply and generated tokens per second, batched as the server does
  thinking     the first --n-think factual questions again with Qwen3 thinking on (reasoning kept apart,
               --think-budget tokens): accuracy, risk AUROC on answers written after reasoning (the head
               was trained on direct answers), seconds per reply

    python run.py --model unsloth/Qwen3-32B-bnb-4bit --head <risk head> --out /kaggle/working/chateval
    python run.py --smoke --head <0.5B risk head> --out /tmp/ce
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vtpkg"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch  # noqa: E402

EVERYDAY = {
    "creative": ["Write a four-line poem about autumn rain.", "Write a short story opening about a lighthouse keeper.",
                 "Give me three names for a cozy coffee shop.", "Write a limerick about a cat who loves boxes.",
                 "Describe a sunset over the sea in two sentences.", "Invent a new board game and explain it briefly.",
                 "Write a haiku about Monday mornings.", "Write a friendly birthday message for a coworker."],
    "opinion": ["Is it better to read books on paper or on a screen?", "What makes a good team leader?",
                "Should I learn piano or guitar first?", "What are the pros and cons of working from home?",
                "Is breakfast the most important meal of the day? Give a balanced view.",
                "What do you think makes a city pleasant to live in?"],
    "advice": ["How can I fall asleep faster?", "How do I prepare for a job interview?",
               "Give me tips to stay focused while studying.", "How do I politely decline a meeting invitation?",
               "What should I pack for a weekend hiking trip?", "How can I save money on groceries?",
               "How do I start running if I have never exercised?", "How do I keep houseplants alive?"],
    "coding": ["How do I reverse a list in Python?", "Explain what a for loop does.",
               "Write a Python function that checks if a string is a palindrome.",
               "What is the difference between a list and a tuple in Python?",
               "How do I read a text file line by line in Python?", "Explain recursion with a simple example."],
    "rewrite": ["Rewrite this more politely: 'Send me the report now.'",
                "Summarize in one sentence: The meeting was moved to Thursday because the room was unavailable.",
                "Make this sentence shorter: 'In my personal opinion, I think that we should probably consider leaving early.'",
                "Translate to French: 'Good morning, how are you?'", "Fix the grammar: 'Me and him goes to the store yesterday.'"],
    "smalltalk": ["Hi! How are you today?", "Thanks for your help earlier!", "Tell me something cheerful.",
                  "What's a fun thing to do on a rainy day?", "Can you recommend a type of tea to try?"],
}

# a second, held-out everyday set: informative replies full of names, dates and numbers that are
# almost always right, the hardest case for false alarms
EVERYDAY2 = {
    "history": ["Tell me briefly about the French Revolution.", "Give me a short overview of the Roman Empire.",
                "What happened during the Apollo 11 mission?", "Summarise the causes of World War I.",
                "Tell me about the life of Abraham Lincoln.", "What was the Renaissance?",
                "Briefly describe the fall of the Berlin Wall.", "Who was Cleopatra?"],
    "travel": ["What should I see on a first visit to Paris?", "Plan a three-day trip to Tokyo.",
               "What is Rome famous for?", "Suggest things to do in New York City with kids.",
               "What are the best months to visit Iceland?", "Recommend sights in London for a history fan.",
               "What is special about Barcelona's architecture?", "Tell me about visiting the Grand Canyon."],
    "science": ["How does photosynthesis work?", "Explain how vaccines train the immune system.",
                "Why is the sky blue?", "How far is the Moon from Earth, roughly?",
                "Explain the theory of evolution simply.", "What is DNA made of?",
                "How do black holes form?", "What causes the seasons on Earth?"],
    "people": ["Who was Marie Curie and what did she discover?", "Tell me about Albert Einstein's main ideas.",
               "Who was Leonardo da Vinci?", "What is Nelson Mandela known for?",
               "Who was Ada Lovelace?", "Tell me about Isaac Newton.",
               "Who was Frida Kahlo?", "What did Charles Darwin do?"],
    "recipes": ["How do I make a classic Italian carbonara?", "Give me a simple pancake recipe.",
                "How do I cook rice on the stove?", "What is in a traditional Greek salad?",
                "How do I make guacamole?", "How long should I boil an egg for a soft yolk?"],
    "howto": ["How do I change a flat tyre?", "How do I set up a budget for the first time?",
              "Explain how to write a cover letter.", "How do I start learning Spanish?",
              "How do I make my laptop battery last longer?", "How do I train for a 5K run?"],
    "creative": ["Write a short story about a girl named Maya who finds a map.",
                 "Write a poem about London in the rain.", "Invent a short myth about why the Nile floods.",
                 "Write a dialogue between Sherlock Holmes and a modern detective.",
                 "Write a postcard from a holiday in Lisbon.", "Write a limerick about a cat from Peru."],
    "smalltalk": ["What's your favourite season and why?", "Tell me a fun fact about octopuses.",
                  "What's a good book to read this summer?", "Recommend a classic film for tonight.",
                  "What's a good name for a golden retriever?", "Tell me a joke about programmers."],
}


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-32B-bnb-4bit")
    ap.add_argument("--head", required=True)
    ap.add_argument("--n-factual", type=int, default=200)
    ap.add_argument("--n-nq", type=int, default=0, help="Natural Questions (open) as well: another distribution")
    ap.add_argument("--seed", type=int, default=7, help="which TriviaQA questions")
    ap.add_argument("--everyday-set", type=int, default=1, choices=(1, 2), help="2: the held-out set full of facts")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-think", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.7, help="the server's decoding (0: greedy)")
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--trace", action="store_true", help="save every reply token's risk, log-prob and claims "
                                                         "(traces.json) to choose the serving score offline")
    ap.add_argument("--think-budget", type=int, default=512)
    ap.add_argument("--out", default="/kaggle/working/chateval")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.model, a.n_factual, a.max_new, a.batch, a.n_think = "Qwen/Qwen2.5-0.5B-Instruct", 6, 48, 4, 4
    os.makedirs(a.out, exist_ok=True)
    import subprocess
    from importlib.metadata import version
    if "qwen3" in a.model.lower() and tuple(int(x) for x in version("transformers").split(".")[:2]) < (4, 51):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51,<4.58"], check=False)
    if "bnb-4bit" in a.model.lower():
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
    from vt.correctness import auroc, auroc_ci, is_right
    from vt.verichat import VeriChat
    hp = a.head if os.path.exists(a.head) else os.path.join(HERE, "files", os.path.basename(a.head))
    t0 = time.time()
    bot = VeriChat.load(a.model, hp)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        for m in bot.model.modules():                          # T4: no bfloat16 matmuls
            if getattr(m, "compute_dtype", None) == torch.bfloat16:
                m.compute_dtype = torch.float16
    from vt.verichat import apply_sampling
    apply_sampling(bot, a.temperature, presence_penalty=a.presence_penalty, seed=0)   # as served; reproducible
    bot.gen.trace = a.trace
    th = bot.gen.risk_threshold
    log(phase="loaded", model=a.model, risk_layer=bot.gen.risk_layer, threshold=th, s=round(time.time() - t0))
    ev = [(cat, p) for cat, ps in (EVERYDAY if a.everyday_set == 1 else EVERYDAY2).items()
          for p in (ps[:2] if a.smoke else ps)]
    if a.smoke:
        fq = [("Who wrote Hamlet?", ["Shakespeare"]), ("What is the capital of Peru?", ["Lima"]),
              ("Who painted the Mona Lisa?", ["Leonardo"]), ("What is the largest planet?", ["Jupiter"]),
              ("Who discovered penicillin?", ["Fleming"]), ("What is the capital of Australia?", ["Canberra"]),
              ("Who wrote the novel The Tin Drum?", ["Grass"]), ("What is the capital of Burkina Faso?", ["Ouagadougou"]),
              ("Who was the second wife of Henry VIII?", ["Boleyn"]), ("In which year did the Battle of Hastings happen?", ["1066"])]
    else:
        from datasets import load_dataset
        d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        idx = random.Random(a.seed).sample(range(len(d)), a.n_factual)
        fq = [(x["question"], x["answer"]["normalized_aliases"] + [x["answer"]["value"]]) for x in d.select(idx)]
        if a.n_nq:
            nq = load_dataset("google-research-datasets/nq_open", split="validation")
            fq += [(x["question"].rstrip("?") + "?", list(x["answer"]))
                   for x in nq.select(random.Random(a.seed).sample(range(len(nq)), a.n_nq))]
    src = ["triviaqa"] * a.n_factual + ["nq"] * (len(fq) - a.n_factual) if not a.smoke else ["smoke"] * len(fq)
    styles = ["{q}", "Tell me: {q}", "Quick question - {q}", "{q} Please explain briefly."]
    reqs = [{"messages": [{"role": "user", "content": p}], "mode": "flag", "max_new_tokens": a.max_new} for _, p in ev]
    reqs += [{"messages": [{"role": "user", "content": styles[i % 4].format(q=q)}], "mode": "flag",
              "max_new_tokens": a.max_new} for i, (q, _) in enumerate(fq)]
    outs, gen_s, n_tok = [], 0.0, 0
    for i in range(0, len(reqs), a.batch):
        t = time.time()
        rs = bot.reply_batch(reqs[i:i + a.batch])
        gen_s += time.time() - t
        n_tok += sum(len(bot.tok(r["text"], add_special_tokens=False).input_ids) for r in rs)
        outs += rs
        log(phase="generated", done=len(outs), of=len(reqs), s=round(time.time() - t0))
    res = {"model": a.model, "threshold": th, "sampling": bot.gen.sampling,
           "speed": {"replies": len(outs), "seconds": round(gen_s, 1), "seconds_per_reply": round(gen_s / len(outs), 2),
                     "tokens_per_second": round(n_tok / gen_s, 1), "batch": a.batch}}
    cats = {}
    for (cat, p), r in zip(ev, outs[:len(ev)]):
        c = cats.setdefault(cat, {"replies": 0, "flagged_replies": 0, "sentences": 0, "flagged_sentences": 0})
        c["replies"] += 1
        c["sentences"] += len(r.get("sentences", []))
        c["flagged_sentences"] += sum(s["doubted"] for s in r.get("sentences", []))
        c["flagged_replies"] += int(any(s["doubted"] for s in r.get("sentences", [])))
    for c in cats.values():
        c["flagged_reply_rate"] = round(c["flagged_replies"] / max(1, c["replies"]), 3)
        c["flagged_sentence_rate"] = round(c["flagged_sentences"] / max(1, c["sentences"]), 3)
    tot = {k: sum(c[k] for c in cats.values()) for k in ("replies", "flagged_replies", "sentences", "flagged_sentences")}
    res["everyday"] = {"by_category": cats, "flagged_reply_rate": round(tot["flagged_replies"] / max(1, tot["replies"]), 3),
                       "flagged_sentence_rate": round(tot["flagged_sentences"] / max(1, tot["sentences"]), 3)}
    fr = outs[len(ev):]
    y = [1 - is_right(r["text"], al) for r, (_, al) in zip(fr, fq)]            # 1 = wrong reply
    risk = [r.get("max_risk") or 0.0 for r in fr]
    f = {"n": len(y), "wrong": int(sum(y)), "accuracy": round(1 - float(np.mean(y)), 3)}
    if len(set(y)) == 2:
        f["risk_auroc"], f["risk_ci"] = auroc(risk, y), auroc_ci(risk, y)
        conf = [-(r.get("reply_confidence") or 0) for r in fr]
        f["claim_probe_auroc"] = auroc(conf, y)                                 # 1 - lowest claim P(correct)
        yy, rr = np.array(y), np.array(risk)
        f["at_threshold"] = {"wrong_flagged": round(float((rr[yy == 1] >= th).mean()), 3),
                             "right_flagged": round(float((rr[yy == 0] >= th).mean()), 3)}
        f["sweep"] = {f"{t:.2f}": {"wrong_flagged": round(float((rr[yy == 1] >= t).mean()), 3),
                                   "right_flagged": round(float((rr[yy == 0] >= t).mean()), 3)}
                      for t in np.quantile(rr, [0.5, 0.7, 0.8, 0.9])}
        def rank(v):
            return np.argsort(np.argsort(np.asarray(v, float))) / max(1, len(v) - 1)
        mlp = [-(r.get("min_logprob") or 0.0) for r in fr]                      # higher = less likely token
        alp = [-(r.get("mean_logprob") or 0.0) for r in fr]
        slp = [max((-(x.get("min_logprob") or 0.0) for x in r.get("sentences", [])), default=0.0) for r in fr]
        f["baselines"] = {"min_token_logprob_auroc": auroc(mlp, y), "mean_logprob_auroc": auroc(alp, y),
                          "worst_sentence_min_logprob_auroc": auroc(slp, y),
                          "risk_plus_min_logprob_rank_mean_auroc": auroc((rank(risk) + rank(mlp)) / 2, y),
                          "risk_plus_mean_logprob_rank_mean_auroc": auroc((rank(risk) + rank(alp)) / 2, y)}
        er = np.array([r.get("max_risk") or 0.0 for r in outs[:len(ev)]])     # choosing a serving threshold
        f["threshold_grid"] = {f"{t:.2f}": {"everyday_flagged": round(float((er >= t).mean()), 3),
                                            "wrong_flagged": round(float((rr[yy == 1] >= t).mean()), 3),
                                            "right_flagged": round(float((rr[yy == 0] >= t).mean()), 3)}
                               for t in np.arange(0.2, 0.96, 0.05)}
    served = np.array([any(s["doubted"] for s in r.get("sentences", [])) or any(c["doubted"] for c in r["claims"])
                       for r in fr])                                          # what a user would see
    yy_ = np.array(y)
    f["served"] = {"wrong_flagged": round(float(served[yy_ == 1].mean()), 3) if (yy_ == 1).any() else None,
                   "right_flagged": round(float(served[yy_ == 0].mean()), 3) if (yy_ == 0).any() else None,
                   "threshold": bot.threshold}
    f["by_source"] = {}
    for sname in sorted(set(src)):
        m = np.array([s_ == sname for s_ in src])
        ys, cs = yy_[m], np.array([-(r.get("reply_confidence") or 1.0) for r in fr])[m]
        mz = np.array([-np.mean([np.log(c["p_correct"] / (1 - c["p_correct"])) for c in r["claims"]])
                       if r["claims"] else -5.0 for r in fr])[m]
        f["by_source"][sname] = {"n": int(m.sum()), "accuracy": round(1 - float(ys.mean()), 3),
                                 "claim_min_auroc": auroc(cs, ys) if len(set(ys)) == 2 else None,
                                 "claim_mean_auroc": auroc(mz, ys) if len(set(ys)) == 2 else None,
                                 "wrong_flagged": round(float(served[m][ys == 1].mean()), 3) if (ys == 1).any() else None,
                                 "right_flagged": round(float(served[m][ys == 0].mean()), 3) if (ys == 0).any() else None}
    res["factual"] = f
    res["raw"] = {"everyday": [{"cat": c, "max_risk": r.get("max_risk"), "min_logprob": r.get("min_logprob"),
                                "mean_logprob": r.get("mean_logprob")} for (c, _), r in zip(ev, outs[:len(ev)])],
                  "factual": [{"wrong": yy_, "max_risk": r.get("max_risk"), "min_logprob": r.get("min_logprob"),
                               "mean_logprob": r.get("mean_logprob"), "reply_confidence": r.get("reply_confidence"),
                               "sentences": [(x["risk"], x.get("min_logprob"), x.get("mean_logprob"))
                                             for x in r.get("sentences", [])]}
                              for yy_, r in zip(y, fr)]}                 # for choosing the serving score offline
    res["examples"] = [{"prompt": rq["messages"][0]["content"], "reply": r["text"][:400],
                        "doubted": [s["text"] for s in r.get("sentences", []) if s["doubted"]]}
                       for rq, r in list(zip(reqs, outs))[:: max(1, len(outs) // 25)]]
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)   # before thinking
    if a.trace:
        tr = [{"kind": "everyday", "cat": c, "prompt": p_, "text": r["text"], "trace": r.get("trace"),
               "reply_confidence": r.get("reply_confidence")} for (c, p_), r in zip(ev, outs[:len(ev)])]
        tr += [{"kind": "factual", "source": s_, "prompt": q, "wrong": w, "text": r["text"], "trace": r.get("trace"),
                "reply_confidence": r.get("reply_confidence")} for (q, _), w, r, s_ in zip(fq, y, fr, src)]
        json.dump(tr, open(os.path.join(a.out, "traces.json"), "w"), default=float)
    try:
        thinking_eval(a, bot, reqs, ev, fq, y, th, res, t0, auroc, auroc_ci, is_right)
    except Exception as e:                                      # keep the rest of the results
        res["thinking"] = {"error": repr(e)[:300]}
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    log(phase="done", everyday=res["everyday"]["flagged_reply_rate"], factual=f.get("risk_auroc"),
        thinking=res.get("thinking", {}).get("risk_auroc"),
        speed=res["speed"], minutes=round((time.time() - t0) / 60, 1))


def thinking_eval(a, bot, reqs, ev, fq, y, th, res, t0, auroc, auroc_ci, is_right):
    if a.n_think and bot.gen.think_open is not None:            # the same questions, thinking first
        tq = fq[:a.n_think]
        treqs = [dict(rq, thinking=True) for rq in reqs[len(ev):len(ev) + len(tq)]]
        bot.gen.think_budget = a.think_budget
        touts, t = [], time.time()
        for i in range(0, len(treqs), a.batch):
            touts += bot.reply_batch(treqs[i:i + a.batch])
            log(phase="thinking", done=len(touts), of=len(treqs), s=round(time.time() - t0))
        ty = [1 - is_right(r["text"], al) for r, (_, al) in zip(touts, tq)]
        trisk = [r.get("max_risk") or 0.0 for r in touts]
        th_ = {"n": len(ty), "accuracy": round(1 - float(np.mean(ty)), 3),
               "accuracy_without_thinking": round(1 - float(np.mean(y[:len(ty)])), 3),
               "seconds_per_reply": round((time.time() - t) / max(1, len(touts)), 2), "think_budget": a.think_budget,
               "reasoning_tokens_mean": round(float(np.mean([len(bot.tok(r.get("reasoning") or "").input_ids)
                                                              for r in touts])), 1)}
        if len(set(ty)) == 2:
            th_["risk_auroc"], th_["risk_ci"] = auroc(trisk, ty), auroc_ci(trisk, ty)
            tt, tr = np.array(ty), np.array(trisk)
            th_["at_threshold"] = {"wrong_flagged": round(float((tr[tt == 1] >= th).mean()), 3),
                                   "right_flagged": round(float((tr[tt == 0] >= th).mean()), 3)}
        res["thinking"] = th_
        res["thinking_examples"] = [{"prompt": rq["messages"][0]["content"], "reasoning": (r.get("reasoning") or "")[:300],
                                     "reply": r["text"][:300]} for rq, r in list(zip(treqs, touts))[:6]]


if __name__ == "__main__":
    main()
