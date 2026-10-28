"""VeriChat: a chatbot that reports, for every claim it makes, how likely the claim is to be right and
where it came from, and that declines to state claims its own internals doubt.

    bot = VeriChat.load(model_dir_or_hub_id, head_path)
    out = bot.reply([{"role": "user", "content": "Who designed the Eiffel Tower?"}],
                    documents=["The Eiffel Tower was completed in 1889 ..."], mode="hedge")

One reply combines:
  doubt-aware generation (vt/guided.py): the correctness probe reads the hidden state already
     computed at each step; a claim whose P(correct) (averaged with its token probability, the
     best trigger measured) falls below the threshold is withheld ("stop") or marked ("hedge")
  claims: each name, number or date the reply asserts, with P(correct)
  provenance (vt/qwen_glassbox.explain_all), when documents are given: the exact share of each
     claim's logit contributed by the documents, the document sentence that contributed most (a
     citation), and a label "from your documents" / "from the model's memory"
  doubt-triggered retrieval (optional `retriever`, e.g. vt.retrieval.Corpus): when no documents are
     given and the first claim's score is below `retrieve_below`, the question is looked up and the
     reply regenerated from the passages found (with provenance and citations as above)

The head is either a claim-level head (kaggle/claims: {"layer", "probe", ...}) or a correctness head
(kaggle/correctness: {variant: {"layers", "probe", ...}}); only its probe and layer are used live.
"""
import re

import numpy as np
import torch

from vt.correctness import Probe
from vt.guided import DoubtAwareGenerator

SYSTEM = "You are a helpful assistant. Answer accurately and concisely."
DOC_SYSTEM = "You are a helpful assistant. Answer using the provided documents."


def load_head(path):
    h = torch.load(path, map_location="cpu", weights_only=True)
    if "probe" in h and "layer" in h:                         # claim-level head
        return Probe.from_state(h["probe"]), int(h["layer"])
    v = h.get("probe+output") or next(iter(h.values()))       # correctness head: any variant's probe
    return Probe.from_state(v["probe"]), int(v["layers"][0])


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def merge_names(claims, text):
    """Adjacent claim words with only spaces or a hedge marker between them ("Gustave" "Eiffel") are one
    claim; it keeps the weakest score."""
    out = []
    for c in claims:
        if out and "start" in c and "start" in out[-1]:
            gap = text[out[-1]["end"]:c["start"]].replace("[unsure:]", "")
            if c["start"] >= out[-1]["end"] and not gap.strip():
                p = out[-1]
                p["text"], p["end"] = text[p["start"]:c["end"]], c["end"]
                for k in ("score", "p_correct", "token_p"):
                    if k in c and k in p:
                        p[k] = min(p[k], c[k])
                p["doubted"] = p["doubted"] or c["doubted"]
                continue
        out.append(dict(c))
    for c in out:                                            # "Gustave" + "Eiffel": show the whole name
        if "end" in c:
            m = re.match(r"(?:\s+[A-Z][\w'-]*)+", text[c["end"]:])
            if m:
                c["end"] += m.end()
                c["text"] = text[c["start"]:c["end"]]
    return out


class VeriChat:
    def __init__(self, model, tok, probe, layer, threshold=0.45, retriever=None, retrieve_below=0.6):
        self.model, self.tok = model, tok
        self.gen = DoubtAwareGenerator(model, tok, probe, layer, threshold=threshold, combine=True)
        self.threshold = threshold
        self.retriever, self.retrieve_below = retriever, retrieve_below

    @classmethod
    def load(cls, model_name, head_path, threshold=0.45):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cuda = torch.cuda.is_available()
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(                # float16 needs sdpa (eager overflows)
            model_name, torch_dtype=torch.float16 if cuda else torch.float32, attn_implementation="sdpa",
            device_map="auto" if cuda else None).eval()
        probe, layer = load_head(head_path)
        return cls(model, tok, probe, layer, threshold)

    def _messages(self, messages, documents):
        msgs = [dict(m) for m in messages]
        if not msgs or msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": DOC_SYSTEM if documents else SYSTEM})
        if documents:
            block = "\n\n".join(f"[Document {i + 1}]\n{d}" for i, d in enumerate(documents))
            last = next(i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "user")
            msgs[last]["content"] = f"{block}\n\nQuestion: {msgs[last]['content']}"
        return msgs

    @torch.no_grad()
    def provenance(self, msgs, reply_ids, claim_idx, documents, question=""):
        """Per claim: how much of what the *inputs* contributed to its logit came from the documents rather
        than the question and history (the bias term and the reply so far are left out: they dominate the
        logit and say nothing about the source), the top document sentence, and the source label."""
        from vt.qwen_glassbox import explain_all
        tok = self.tok
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(head, return_offsets_mapping=True, add_special_tokens=False)
        p_ids, offs = enc["input_ids"], enc["offset_mapping"]
        spans = []                                               # (doc i, sentence j, char start, end) in the prompt
        for i, d in enumerate(documents):
            base = head.find(d)
            if base < 0:
                continue
            pos = 0
            for j, sen in enumerate(sentences(d)):
                k = d.find(sen, pos)
                if k >= 0:
                    spans.append((i, j, base + k, base + k + len(sen)))
                    pos = k + len(sen)
        qa = head.rfind(question) if question else -1
        groups = []
        for s, e in offs:
            g = next((f"doc{i}:s{j}" for i, j, a, b in spans if s < b and e > a), None)
            if g is None:
                g = "question" if qa >= 0 and s < qa + len(question) and e > qa else "template"
            groups.append(g)
        groups += ["reply"] * len(reply_ids)
        ids = torch.tensor([p_ids + reply_ids], device=self.model.get_input_embeddings().weight.device)
        names, c, _ = explain_all(self.model, ids, groups, start=len(p_ids) - 1)
        c = c.float().cpu().numpy()                              # (S, n): column j explains reply token j
        doc_rows = [k for k, n in enumerate(names) if n.startswith("doc")]
        q_rows = [k for k, n in enumerate(names) if n == "question"]
        out = []
        for j in claim_idx:
            col = c[:, j]
            dpos = float(np.clip(col[doc_rows], 0, None).sum()) if doc_rows else 0.0
            qpos = float(np.clip(col[q_rows], 0, None).sum()) if q_rows else 0.0
            share = dpos / (dpos + qpos) if dpos + qpos > 0 else 0.0
            tot = dpos + qpos + 1e-6
            cite = None
            if doc_rows:
                k = max(doc_rows, key=lambda r: col[r])
                if col[k] > 0:
                    i, s = (int(x) for x in re.match(r"doc(\d+):s(\d+)", names[k]).groups())
                    cite = {"document": i + 1, "sentence": sentences(documents[i])[s], "contribution": float(col[k] / tot)}
            out.append({"doc_share": round(share, 4), "citation": cite})
        return out

    @staticmethod
    def doubt_score(r):
        """The retrieval trigger: the first claim's probe P(correct) averaged with its token probability
        (the reply's first content token when it has no claim); None for an empty reply."""
        first = r["claims"][0] if r["claims"] else r.get("first_token")
        return None if not first else (first["p_correct"] + first["token_p"]) / 2

    def reply(self, messages, documents=None, mode="hedge", max_new_tokens=192, retrieve=True):
        out = self._reply(messages, documents, mode, max_new_tokens)
        score = out.pop("_trigger")
        if documents or not retrieve or self.retriever is None:
            return out
        out["retrieval"] = {"trigger_score": score, "retrieved": False}
        if score is None or score >= self.retrieve_below:
            return out
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        passages = list(self.retriever(question))
        if not passages:
            return out
        again = self._reply(messages, passages, mode, max_new_tokens)
        again.pop("_trigger")
        again["retrieval"] = {"trigger_score": score, "retrieved": True, "passages": passages,
                              "closed_book_text": out["text"]}
        return again

    def _reply(self, messages, documents, mode, max_new_tokens):
        msgs = self._messages(messages, documents or [])
        context = " ".join(m["content"] for m in messages if m["role"] != "system")   # without the documents
        r = self.gen.generate(msgs, max_new_tokens=max_new_tokens, mode=mode, context=context)
        claims = r["claims"]
        if documents and claims and not r["stopped_on_doubt"]:
            ids = r["ids"]                                           # the generated tokens themselves
            idx = [min(c["token_index"], len(ids) - 1) for c in claims]
            question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            for c, p in zip(claims, self.provenance(msgs, ids, idx, documents, question)):
                c.update(p)
                c["source"] = "your documents" if p["doc_share"] >= 0.5 else "the model's memory"
        trigger = self.doubt_score(r)
        claims = merge_names(claims, r["text"])
        for c in claims:
            c["verdict"] = "doubtful" if c["doubted"] else ("likely right" if c["score"] >= 0.7 else "uncertain")
            if c.get("source") != "your documents":
                c.pop("citation", None)                           # a citation only for claims the documents drove
        return {"text": r["text"], "claims": claims, "withheld": r["stopped_on_doubt"],
                "reply_confidence": r["min_p_correct"], "threshold": self.threshold, "mode": mode, "_trigger": trigger}
