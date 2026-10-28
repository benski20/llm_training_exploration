"""A general hallucination detector: a frozen LLM reads any (prompt, optional documents, response),
and probes on its hidden states score every response token and the response as a whole.

The response can come from any model (GPT-4, Claude, Llama ...): the reader is ours, so this works
for closed models too. Every dataset is loaded into one schema:

    {"source", "task", "messages": [...], "response", "doc": "" or the grounding text,
     "spans": [(start, end), ...] in the response or None (response-level label only), "label": 0/1}

  ragtruth             RAG summaries, QA, data-to-text from 6 models; human-labelled spans
  halueval_general     ChatGPT replies to open user queries; labelled spans (as text)
  halueval_qa          knowledge + question; right vs hallucinated answer (response level)
  halueval_dialogue    knowledge + dialogue; right vs hallucinated reply
  halueval_summary     document; right vs hallucinated summary
  truthfulqa           misconception questions; a correct vs an incorrect answer
"""
import ast
import json
import random

import numpy as np
import torch

RAG_SYSTEM = "You are a helpful assistant. Answer using the provided information."
PLAIN_SYSTEM = "You are a helpful assistant."


def _item(source, task, user, response, doc="", spans=None, label=None, system=PLAIN_SYSTEM):
    if label is None:
        label = int(bool(spans))
    return {"source": source, "task": task, "messages": [{"role": "system", "content": system},
                                                       {"role": "user", "content": user}],
            "response": response, "doc": doc, "spans": spans, "label": int(label)}


def load_ragtruth(split, n=None, seed=0):
    from datasets import load_dataset
    d = load_dataset("wandb/RAGTruth-processed", split=split).shuffle(seed=seed)
    out = []
    for x in d.select(range(min(n or len(d), len(d)))):
        spans = [(l["start"], l["end"]) for l in json.loads(x["hallucination_labels"])]
        out.append(_item("ragtruth", x["task_type"], x["input_str"], x["output"], x["context"], spans, system=RAG_SYSTEM))
    return out


def _split(rows, split, seed=0, test_frac=0.2):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    return [rows[i] for i in (idx[:cut] if split == "train" else idx[cut:])]


def load_halueval(subset, split, n=None):
    from datasets import load_dataset
    rows = _split(list(load_dataset("pminervini/HaluEval", subset)["data"]), split)
    if n:
        rows = rows[:n]
    out = []
    for x in rows:
        if subset == "qa":
            user = f"Knowledge: {x['knowledge']}\n\nQuestion: {x['question']}"
            out += [_item("halueval_qa", "qa", user, x["right_answer"], x["knowledge"], label=0, system=RAG_SYSTEM),
                    _item("halueval_qa", "qa", user, x["hallucinated_answer"], x["knowledge"], label=1, system=RAG_SYSTEM)]
        elif subset == "dialogue":
            user = f"Knowledge: {x['knowledge']}\n\nDialogue so far:\n{x['dialogue_history']}\n\nReply as the assistant."
            out += [_item("halueval_dialogue", "dialogue", user, x["right_response"], x["knowledge"], label=0, system=RAG_SYSTEM),
                    _item("halueval_dialogue", "dialogue", user, x["hallucinated_response"], x["knowledge"], label=1,
                          system=RAG_SYSTEM)]
        elif subset == "summarization":
            user = f"Summarize the following document.\n\n{x['document']}"
            out += [_item("halueval_summary", "summary", user, x["right_summary"], x["document"], label=0, system=RAG_SYSTEM),
                    _item("halueval_summary", "summary", user, x["hallucinated_summary"], x["document"], label=1,
                          system=RAG_SYSTEM)]
        elif subset == "general":
            resp = x["chatgpt_response"] or ""
            raw = x["hallucination_spans"]
            texts = raw if isinstance(raw, list) else (ast.literal_eval(raw) if raw and raw.strip().startswith("[") else [])
            spans = []
            for t in texts:
                k = resp.find(t.strip()) if t else -1
                if k >= 0:
                    spans.append((k, k + len(t.strip())))
            lab = int(x["hallucination"] == "yes")
            out.append(_item("halueval_general", "general", x["user_query"], resp, "", spans if spans else None,
                             label=lab))
    return out


def load_truthfulqa(split, n=None):
    from datasets import load_dataset
    rows = _split(list(load_dataset("truthfulqa/truthful_qa", "generation")["validation"]), split)
    if n:
        rows = rows[:n]
    out = []
    for i, x in enumerate(rows):
        rng = random.Random(i)
        good = x["best_answer"]
        bad = rng.choice(x["incorrect_answers"]) if x["incorrect_answers"] else None
        out.append(_item("truthfulqa", "misconception", x["question"], good, label=0))
        if bad:
            out.append(_item("truthfulqa", "misconception", x["question"], bad, label=1))
    return out


SOURCES = ("ragtruth", "halueval_general", "halueval_qa", "halueval_dialogue", "halueval_summary", "truthfulqa")


def load_all(split, caps):
    """caps: {source: max items (responses; HaluEval pairs count as rows)}."""
    out = []
    if caps.get("ragtruth"):
        out += load_ragtruth("train" if split == "train" else "test", caps["ragtruth"])
    for sub, src in (("general", "halueval_general"), ("qa", "halueval_qa"), ("dialogue", "halueval_dialogue"),
                     ("summarization", "halueval_summary")):
        if caps.get(src):
            out += load_halueval(sub, split, caps[src])
    if caps.get("truthfulqa"):
        out += load_truthfulqa(split, caps["truthfulqa"])
    return out


def encode(tok, item, max_tokens=2048):
    """Token ids of conversation + response, the index of the first response token, per-response-token
    span labels (None if the item only has a response label), and the token group of every position
    (document / instruction / response) for provenance. Long prompts are cut from the left of the
    document so the response always fits."""
    head = tok.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
    text = head + item["response"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    r0 = next((j for j, (s, e) in enumerate(offs) if e > len(head)), len(ids))
    if len(ids) > max_tokens:                                       # keep the response, trim the prompt's start
        cut = len(ids) - max_tokens
        if cut >= r0:
            return None
        ids, offs, r0 = ids[cut:], offs[cut:], r0 - cut
    labels = None
    if item["spans"] is not None:
        sp = [(a + len(head), b + len(head)) for a, b in item["spans"]]
        labels = [int(any(s < hi and e > lo for lo, hi in sp)) for s, e in offs[r0:]]
    da = text.find(item["doc"][:200]) if item["doc"] else -1
    db = da + len(item["doc"]) if da >= 0 else -1
    groups = ["response" if j >= r0 else ("document" if da >= 0 and s < db and e > da else "instruction")
              for j, (s, e) in enumerate(offs)]
    return {"ids": ids, "r0": r0, "labels": labels, "groups": groups}


@torch.no_grad()
def response_states(model, ids, r0, layers):
    """Hidden states (len(layers), n_response_tokens, d) at the positions that *read* each response
    token (the token's own position: the state after the model has seen it)."""
    dev = model.get_input_embeddings().weight.device
    o = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True)
    return torch.stack([o.hidden_states[L + 1][0, r0:].to("cpu", torch.float16) for L in layers]), \
        o.logits[0, r0 - 1:-1].float().log_softmax(-1).gather(-1, torch.tensor(ids[r0:], device=o.logits.device)[:, None])[:, 0].cpu()
