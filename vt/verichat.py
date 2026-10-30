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

from vt.claims import prompt_of
from vt.correctness import Probe
from vt.guided import DoubtAwareGenerator

SYSTEM = "You are a helpful assistant. Answer accurately and concisely."
DOC_SYSTEM = "You are a helpful assistant. Answer using the provided documents."


def gpu_dtype():
    """bfloat16 where the GPU supports it (Ampere and later; safer range for Qwen3), else float16
    (T4: needs sdpa attention, see VISION.md)."""
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def load_risk(path):
    """A kaggle/risk head: (risk probe, its layer, sentence threshold), or None for other head files."""
    h = torch.load(path, map_location="cpu", weights_only=True)
    if "risk_probe" not in h:
        return None
    return Probe.from_state(h["risk_probe"]), int(h["risk_layer"]), float(h.get("risk_threshold", 0.5))


class _Negated:
    """P(correct) from a risk probe when a head has no claim probe: logit -> -logit."""

    def __init__(self, probe):
        self.probe = probe

    def logit(self, X):
        return -self.probe.logit(X)


def load_head(path):
    h = torch.load(path, map_location="cpu", weights_only=True)
    if "risk_probe" in h:                                      # kaggle/risk head: its claim probe
        if h.get("claim_probe") is not None:
            return Probe.from_state(h["claim_probe"]), int(h["claim_layer"])
        print("risk head has no claim probe: claims scored as 1 - token risk", flush=True)
        return _Negated(Probe.from_state(h["risk_probe"])), int(h["risk_layer"])
    if "probe" in h and "layer" in h:                         # claim-level head
        return Probe.from_state(h["probe"]), int(h["layer"])
    v = h.get("probe+output") or next(iter(h.values()))       # correctness head: any variant's probe
    return Probe.from_state(v["probe"]), int(v["layers"][0])


def load_serving(head_path):
    """Serving settings measured for a head (<head>.serving.json, e.g. from experiments/serving_score.py):
    claim calibration, doubt threshold, verdict cut-offs, and how sentences are judged. {} if none."""
    import json
    import os
    p = os.path.splitext(head_path)[0] + ".serving.json"
    return json.load(open(p)) if os.path.exists(p) else {}


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


def apply_sampling(bot, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, seed=None):
    """The server's decoding: sample the reply (temperature 0: greedy) and the <think> block (Qwen3's
    advice, 0.6 / 0.95 / 20)."""
    if temperature <= 0:
        bot.gen.sampling = bot.gen.think_sampling = None
        return
    extra = {"presence_penalty": presence_penalty, **({"seed": seed} if seed is not None else {})}
    bot.gen.sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k, **extra}
    bot.gen.think_sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, **extra}


class VeriChat:
    def __init__(self, model, tok, probe, layer, threshold=0.45, retriever=None, retrieve_below=0.3, moderator=None,
                 thinking=False, think_budget=1024, max_prompt_tokens=8192):
        self.model, self.tok = model, tok
        self.max_prompt_tokens = max_prompt_tokens     # longer prompts: documents cut, oldest turns dropped
        self.provenance_on_lookup = False              # exact per-claim attribution for looked-up passages too
                                                       # (costly on a 32B: one pass per source group per layer)
        self.gen = DoubtAwareGenerator(model, tok, probe, layer, threshold=threshold, combine=True,
                                       thinking=thinking, think_budget=think_budget)
        self.threshold = threshold
        self.likely_right = 0.7                        # claims at or above this score read "likely right"
        self.retriever, self.retrieve_below = retriever, retrieve_below
        self.moderator = moderator                     # vt.moderation.Moderator: screen inputs and replies
        self.verifier = None                           # vt.claim_verify.ClaimVerifier: check claims against sources
        self.router = None                             # vt.router.Router: information / math / code / chat

    @classmethod
    def load(cls, model_name, head_path, threshold=None, revision=None):
        """threshold: claims scoring below it are doubted (default: the head's serving.json, else 0.45).
        revision: the Hugging Face commit (or tag) of the model to load; pin it in production."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cuda = torch.cuda.is_available()
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(                # float16 needs sdpa (eager overflows)
            model_name, revision=revision, torch_dtype=gpu_dtype() if cuda else torch.float32,
            attn_implementation="sdpa", device_map="auto" if cuda else None).eval()
        probe, layer = load_head(head_path)
        bot = cls(model, tok, probe, layer, 0.45)
        risk = load_risk(head_path)
        if risk is not None:                                   # learned risk on every token and sentence
            bot.gen.risk_probe, bot.gen.risk_layer, bot.gen.risk_threshold = risk
        bot.configure(load_serving(head_path), threshold)
        return bot

    def configure(self, serving, threshold=None):
        """Apply measured serving settings: {"claim_calib": [a, b], "threshold", "likely_right", "combine",
        "sentence_doubt": "risk" | "claims"}; an explicit threshold wins."""
        g = self.gen
        if "claim_calib" in serving:
            g.calib = tuple(serving["claim_calib"])
        if "combine" in serving:
            g.combine = bool(serving["combine"])
        if "sentence_doubt" in serving:
            g.sentence_doubt = serving["sentence_doubt"]
        self.likely_right = serving.get("likely_right", self.likely_right)
        t = threshold if threshold is not None else serving.get("threshold", self.threshold)
        self.threshold = g.threshold = t

    def _fit_documents(self, documents):
        """Documents take at most half the prompt budget, shared equally; a longer one is cut (and marked)."""
        if not documents or not self.max_prompt_tokens:
            return documents
        share, out = self.max_prompt_tokens // (2 * len(documents)), []
        for d in documents:
            ids = self.tok(d, add_special_tokens=False).input_ids
            out.append(d if len(ids) <= share else self.tok.decode(ids[:share]).rstrip() + " [...]")
        return out

    def _messages(self, messages, documents):
        msgs = [dict(m) for m in messages]
        if not msgs or msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": DOC_SYSTEM if documents else SYSTEM})
        if documents:
            block = "\n\n".join(f"[Document {i + 1}]\n{d}" for i, d in enumerate(documents))
            last = next(i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "user")
            msgs[last]["content"] = f"{block}\n\nQuestion: {msgs[last]['content']}"
        if self.max_prompt_tokens:                              # a long chat: drop its oldest turns
            while len(msgs) > 2 and len(self.tok(prompt_of(self.tok, msgs)).input_ids) > self.max_prompt_tokens:
                del msgs[1]
                while len(msgs) > 2 and msgs[1]["role"] != "user":
                    del msgs[1]
        return msgs

    @torch.no_grad()
    def provenance(self, msgs, reply_ids, claim_idx, documents, question="", prefix_ids=(), thinking=False):
        """Per claim: how much of what the *inputs* contributed to its logit came from the documents rather
        than the question and history (the bias term and the reply so far are left out: they dominate the
        logit and say nothing about the source), the top document sentence, and the source label.
        prefix_ids: a <think> block written before the reply; it is an input too, and what the documents
        contributed through it is credited to it (left out like the reply so far)."""
        from vt.qwen_glassbox import explain_all
        tok = self.tok
        head = prompt_of(tok, msgs, thinking)
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
        p_ids, groups = p_ids + list(prefix_ids), groups + ["reasoning"] * len(prefix_ids)
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
        """The retrieval trigger: the weakest claim's score, the same score that marks a claim doubtful (the
        reply's first content token when it has no claim), and that claim; (None, None) for an empty reply.
        So every reply that shows a doubtful claim is looked up."""
        if r["claims"]:
            c = min(r["claims"], key=lambda c: c["score"])
            return c["score"], c.get("text")
        first = r.get("first_token")
        return (None, None) if not first else (first["p_correct"], None)

    def reply(self, messages, documents=None, mode="hedge", max_new_tokens=192, retrieve=True, thinking=None):
        return self.reply_batch([{"messages": messages, "documents": documents, "mode": mode,
                                  "max_new_tokens": max_new_tokens, "retrieve": retrieve, "thinking": thinking}])[0]

    def reply_batch(self, requests, on_text=None, on_reasoning=None, on_ready=None):
        """Several conversations at once (one batched generation; provenance per reply). requests:
        [{"messages", "documents"?, "mode"?, "max_new_tokens"?, "retrieve"?, "thinking"?}] -> one reply()
        result each. thinking (None: the bot's default) lets a reasoning model think first; its reasoning
        is returned apart from the reply and streamed to on_reasoning(i, delta). on_ready(i, reply) gets each
        reply as soon as it is final: replies that need no lookup do not wait for the batch's lookups."""
        requests = self._route(requests)
        reqs = [{"messages": r["messages"], "documents": self._fit_documents(r.get("documents") or []),
                 "mode": r.get("mode", "hedge"),
                 "max_new_tokens": r.get("max_new_tokens", 192), "retrieve": r.get("retrieve", True),
                 "thinking": self.gen.thinking if r.get("thinking") is None else bool(r["thinking"]),
                 **{k: r[k] for k in ("sampling", "think_sampling", "route", "grounded", "source_label") if r.get(k) is not None}}
                for r in requests]
        if self.moderator is not None:                               # refuse flagged messages without generating
            from vt.moderation import REFUSAL
            flags = [self.moderator.check(next(m["content"] for m in reversed(r["messages"]) if m["role"] == "user"))
                     for r in reqs]
            keep = [i for i, f in enumerate(flags) if not f["flagged"]]
            screened = None
            if on_text is not None:                                  # stream only text the filter has passed
                seen, blocked = {}, set()

                def screened(j, d):
                    i = keep[j]
                    seen[i] = seen.get(i, "") + d
                    if i in blocked or self.moderator.check(seen[i])["flagged"]:
                        blocked.add(i)
                        return
                    on_text(i, d)
            reason = None if on_reasoning is None else (lambda j, d: on_reasoning(keep[j], d))
            outs = [{"text": REFUSAL, "claims": [], "withheld": True, "reply_confidence": None, "threshold": self.threshold,
                     "mode": r["mode"], "moderation": {"input": f}} for r, f in zip(reqs, flags)]
            if on_ready is not None:
                for i, f in enumerate(flags):
                    if f["flagged"]:
                        on_ready(i, outs[i])
            ready = None if on_ready is None else (lambda j, o: on_ready(keep[j], o))
            kept = self._reply_batch_unmoderated([reqs[i] for i in keep], screened, reason, ready) if keep else []
            for i, o in zip(keep, kept):
                outs[i] = o
            return outs
        return self._reply_batch_unmoderated(reqs, on_text, on_reasoning, on_ready)

    def _route(self, requests):
        """Sort each message (vt.router) and ground information requests in one lookup made before answering:
        the reply is then written from those passages, with provenance and citations, and checked against
        them, instead of looked up claim by claim afterwards. Messages with documents of their own keep them."""
        if self.router is None:
            return requests
        todo = [i for i, r in enumerate(requests) if not r.get("documents")]
        routes = dict(zip(todo, self.router.route_batch([requests[i]["messages"] for i in todo]))) if todo else {}
        out = []
        for i, r in enumerate(requests):
            r = dict(r)
            if i in routes:
                r["route"] = routes[i][0]
                if r.get("mode", "hedge") == "hedge":           # routed replies are judged by sources or by
                    r["mode"] = "flag"                          # arithmetic, never by in-line markers from a
                                                                # probe trained on facts (it does not transfer)
                if r["route"] == "text":                        # a task on pasted text: that text is the source
                    q = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "user"), "")
                    r.update(documents=[q], grounded=True, source_label="your text", retrieve=False)
                elif r["route"] != "information":               # nothing to look up for a poem, a sum or code
                    r["retrieve"] = False
                elif self.retriever is not None and r.get("retrieve", True):
                    q = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "user"), "")
                    try:
                        passages = self.retriever(q)
                    except Exception as e:                      # no sources: answered as before
                        print(f"grounding lookup failed: {e!r}", flush=True)
                        passages = []
                    if passages:                                # judged by its sources
                        r.update(documents=passages[:3], grounded=True)
                    r["retrieve"] = False                       # one lookup per message, made here
            out.append(r)
        return out

    def _finish(self, o):
        """Screen a final reply (moderation) and return it."""
        if self.moderator is not None:                               # withhold flagged replies
            from vt.moderation import WITHHELD
            f = self.moderator.check(o["text"])
            o["moderation"] = {"output": f}
            if f["flagged"]:
                o.update(text=WITHHELD, claims=[], withheld=True)
        return o

    def _verify(self, items):
        """Check the claims of several final replies against sources at once (the verifier): items are
        (reply, question). A check that fails never fails the replies. Then each reply's assessment."""
        todo = [(o, q) for o, q in items if q and not o.get("withheld")]
        if self.verifier is not None and todo:
            try:
                vs = self.verifier.verify_batch([(q, o["text"].replace("[unsure:]", "")) for o, q in todo])
                for (o, _), v in zip(todo, vs):
                    o["verification"] = v
            except Exception as e:
                print(f"verification failed: {e!r}", flush=True)
        for o, _ in items:
            a = self.assess(o)
            if a is not None:
                o["assessment"] = a

    # weights and bands from 600 answers of the served 32B with sources found for 90% (results/study_32b_v2_rejudge.json,
    # deploy/RUNBOOK.md): the judge leads the probe by +0.12 AUROC (95% CI 0.07-0.17); weights 0.9-1.0 on the judge
    # tie on both halves (0.9 keeps the model's own doubt as a tie-break); doubt above 0.6 flags about 46% of the
    # wrong answers and 12% of the right ones
    JUDGE_WEIGHT, LIKELY_WRONG, UNCERTAIN = 0.9, 0.6, 0.3

    def assess(self, o):
        """One verdict for a factual reply: the doubt of its first sourced claim mixed with the model's own,
        or the model's own alone when no source was found; None for replies without claims (small talk). A
        calculation that does not add up makes a reply likely wrong whatever else is said."""
        wrong = [m for m in o.get("math_check") or [] if not m["ok"]]
        if wrong and not o.get("withheld"):
            return {"doubt": 1.0, "verdict": "likely wrong",
                    "basis": f"arithmetic: {wrong[0]['expression']} is {wrong[0]['computed']:g}, not {wrong[0]['stated']:g}"}
        broken = [c for c in o.get("code_check") or [] if not c["ok"]]
        if broken and not o.get("withheld"):
            return {"doubt": 1.0, "verdict": "likely wrong",
                    "basis": f"its {broken[0]['language']} does not parse: {broken[0]['error']}"}
        if o.get("route") in ("chat", "code"):
            return None                                          # opinions, creative writing, code: no fact verdict
        if o.get("route") == "math":                             # judged by its arithmetic only: the confidence
            checked = o.get("math_check") or []                  # probe was trained on facts, not on calculations
            return {"doubt": 0.0, "verdict": "likely right", "basis": f"arithmetic re-computed ({len(checked)} steps)"} \
                if checked else None
        if o.get("route") == "information" and not o.get("grounded") and not any(
                c.get("p_supported") is not None for c in o.get("verification") or []):
            return {"doubt": None, "verdict": "unverified", "basis": "no sources found for this question"}
        if o.get("withheld") or not (o.get("claims") or o.get("grounded") and o.get("sentences")):
            return None
        if o.get("grounded"):                                    # answered from sources: are its claims in them?
            claims = o.get("claims") or []                       # (none marked: its sentences are still checked)
            label = o.get("source_label", "the sources found")
            sourced = [c for c in claims if c.get("source") == label]
            doubted = [s for s in o.get("sentences") or [] if s.get("doubted")]
            where = "your text" if label == "your text" else "the sources"
            if len(sourced) == len(claims) and not doubted:
                return {"doubt": 0.1, "verdict": "likely right", "basis": f"every claim is in {where}"}
            missing = len(claims) - len(sourced)
            basis = (f"{missing} of {len(claims)} claims are not in {where}" if missing else
                     f"{len(doubted)} sentence(s) doubted against {where}")
            return {"doubt": 0.5, "verdict": "uncertain", "basis": basis}
        score, _ = self.doubt_score(o)
        own = 1 - score if score is not None else 0.5
        judged = [c for c in o.get("verification") or [] if c.get("p_supported") is not None]
        if judged and "p_contradicted" in judged[0]:             # three-way: the sources' silence is not doubt
            c = judged[0]
            a, b, n = c["p_supported"], c["p_contradicted"], c["p_not_stated"]
            w = self.JUDGE_WEIGHT * (1 - n)
            d = w * b / max(a + b, 1e-6) + (1 - w) * own
            basis = "sources" if n < 0.5 else "the model's own confidence (the sources do not say)"
        elif judged:
            d, basis = self.JUDGE_WEIGHT * (1 - judged[0]["p_supported"]) + (1 - self.JUDGE_WEIGHT) * own, "sources"
        else:
            d, basis = own, "the model's own confidence"
        verdict = "likely wrong" if d > self.LIKELY_WRONG else "uncertain" if d > self.UNCERTAIN else "likely right"
        return {"doubt": round(float(d), 4), "verdict": verdict, "basis": basis}

    def _reply_batch_unmoderated(self, reqs, on_text=None, on_reasoning=None, on_ready=None):
        outs = self._reply_batch(reqs, on_text, on_reasoning)

        def send(i):
            if on_ready is not None:
                on_ready(i, outs[i])
        def ask(r):
            return next((m["content"] for m in reversed(r["messages"]) if m["role"] == "user"), None)

        for r, o in zip(reqs, outs):
            if r.get("route"):
                o["route"] = r["route"]
            if r.get("grounded"):
                o["grounded"] = True                                 # written from sources fetched first
                o["source_label"] = r.get("source_label", "the sources found")
                if o["source_label"] == "the sources found":         # the user's own text is not shown back
                    o["sources"] = r["documents"]
                for c in o.get("claims") or []:
                    if c.get("source") == "your documents":
                        c["source"] = o["source_label"]
            if r.get("route"):                                       # every routed reply: a calculation in a how-to
                from vt.mathcheck import check_arithmetic            # or a misrouted sum is checked too (free)
                o["math_check"] = check_arithmetic(o["text"].replace("[unsure:]", ""),
                                                   explicit=r["route"] != "math")
                from vt.codecheck import check_code                  # and its Python / JSON parsed, never run
                o["code_check"] = check_code(o["text"])

        def deliver(ids):                                            # verified together, then sent
            self._verify([(outs[i], None if reqs[i].get("grounded") else ask(reqs[i])) for i in ids])
            for i in ids:
                send(i)
        doubted, final = [], []                                      # closed-book replies the model doubts
        for i, (r, o) in enumerate(zip(reqs, outs)):
            score, claim = o.pop("_trigger"), o.pop("_claim", None)
            if not (r["documents"] or not r["retrieve"] or self.retriever is None):
                o["retrieval"] = {"trigger_score": score, "retrieved": False}
                if score is not None and score < self.retrieve_below:
                    doubted.append((i, claim))
                    continue
            self._finish(o)                                          # final now: it does not wait for lookups
            final.append(i)
        deliver(final)
        again, where = [], []                                        # look the doubted ones up
        for i, claim in doubted:
            r = reqs[i]
            question = next(m["content"] for m in reversed(r["messages"]) if m["role"] == "user")
            passages = []                                        # the question, then the question with the doubted
            for q in [question] + ([f"{question} {claim}"] if claim else []):   # claim (its page)
                for p in self.retriever(q):
                    if p not in passages:
                        passages.append(p)
            passages = passages[:3]
            if passages:
                again.append(dict(r, documents=passages))
                where.append(i)
            else:                                                    # nothing found: the closed-book reply stands
                self._finish(outs[i])
                deliver([i])
        looked = self._reply_batch(again, provenance=self.provenance_on_lookup) if again else []
        for i, r, o in zip(where, again, looked):
            o.pop("_trigger"), o.pop("_claim", None)
            o["retrieval"] = {"trigger_score": outs[i]["retrieval"]["trigger_score"], "retrieved": True,
                              "passages": r["documents"], "closed_book_text": outs[i]["text"]}
            outs[i] = self._finish(o)
        deliver(where)
        return outs

    def _reply_batch(self, reqs, on_text=None, on_reasoning=None, provenance=True):
        prepared = [(self._messages(r["messages"], r["documents"]),
                     " ".join(m["content"] for m in r["messages"] if m["role"] != "system")) for r in reqs]  # context
        gens = self.gen.generate_batch([{"messages": msgs, "max_new_tokens": r["max_new_tokens"], "mode": r["mode"],
                                         "context": ctx, "thinking": r.get("thinking", self.gen.thinking),
                                         **{k: r[k] for k in ("sampling", "think_sampling") if k in r}}
                                        for r, (msgs, ctx) in zip(reqs, prepared)],          # context without docs
                                       on_text=on_text, on_reasoning=on_reasoning)
        return [self._post(r["messages"], r["documents"], r["mode"], msgs, g, r.get("thinking", self.gen.thinking),
                           provenance) for r, (msgs, _), g in zip(reqs, prepared, gens)]

    def _post(self, messages, documents, mode, msgs, r, thinking=False, provenance=True):
        claims = r["claims"]
        if provenance and documents and claims and not r["stopped_on_doubt"]:
            ids = r["ids"]                                           # the generated tokens themselves
            idx = [min(c["token_index"], len(ids) - 1) for c in claims]
            question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            try:
                prov = self.provenance(msgs, ids, idx, documents, question, r.get("prefix_ids", ()), thinking)
            except Exception as e:                               # never lose the reply over its attribution
                print(f"provenance failed ({type(e).__name__}: {str(e)[:120]}); claims left unsourced", flush=True)
                prov = []
                if torch.cuda.is_available():                    # e.g. out of memory: free it for the next reply
                    torch.cuda.empty_cache()
            for c, p in zip(claims, prov):
                c.update(p)
                c["source"] = "your documents" if p["doc_share"] >= 0.5 else "the model's memory"
        trigger, trigger_claim = self.doubt_score(r)
        claims = merge_names(claims, r["text"])
        for c in claims:
            c["verdict"] = "doubtful" if c["doubted"] else ("likely right" if c["score"] >= self.likely_right else "uncertain")
            if c.get("source") != "your documents":
                c.pop("citation", None)                           # a citation only for claims the documents drove
        out = {"text": r["text"], "claims": claims, "withheld": r["stopped_on_doubt"],
               "reply_confidence": r["min_p_correct"], "threshold": self.threshold, "mode": mode, "_trigger": trigger, "_claim": trigger_claim}
        out["mean_logprob"], out["min_logprob"] = r.get("mean_logprob"), r.get("min_logprob")
        if "trace" in r:
            out["trace"] = r["trace"]
        if r.get("reasoning") is not None:
            out["reasoning"] = r["reasoning"]
        if r.get("sentences"):                                   # learned risk: every sentence, no heuristic
            out["sentences"], out["max_risk"] = r["sentences"], r["max_risk"]
        return out
