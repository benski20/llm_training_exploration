"""Doubt-aware generation: act on the model's own internal doubt while it writes.

At every generated token the model has already computed its hidden states. When the next token is
the first token of a claim (a capitalised or numeric word the conversation did not contain), the
correctness probe reads the hidden state at that position, the same state the probe was trained on,
so the check costs no extra forward pass. If P(correct) falls below the threshold the generator acts
*before* the claim is written:

  stop    end the reply there with an explicit "I'm not sure" (no wrong name gets stated)
  hedge   mark the claim as unsure in the text and keep going (lists, paragraphs, reasoning)
  flag    write it, but report it as doubted

Every claim is returned with its P(correct), whatever the mode.

    g = DoubtAwareGenerator(model, tok, probe, layer=23, threshold=0.4)
    out = g.generate([{"role": "user", "content": "Who was the man behind The Chipmunks?"}])
"""
import re

import numpy as np
import torch

from vt.claims import prompt_of
from vt.correctness import SKIP, norm

STOP_NOTE = "I'm not sure about this, so I won't guess."
OPENERS = "\"'“”‘’*#([«"
HEDGE = " [unsure:]"
LIST_MARK = re.compile(r"(\d+|[a-zA-Z])[.)]")


class DoubtAwareGenerator:
    """risk_probe / risk_layer (optional): a token-level hallucination probe read at each written token's
    own position (vt.detector's token probe, or kaggle/risk). Every token then gets P(hallucinated), with
    no claim heuristic; a sentence's risk is its riskiest content token, and in stop mode a sentence
    above risk_threshold is removed as soon as it is complete and the reply ends there."""

    def __init__(self, model, tok, probe, layer, threshold=0.5, calib=(1.0, 0.0), combine=True,
                 risk_probe=None, risk_layer=None, risk_threshold=0.5):
        self.model, self.tok, self.probe, self.layer = model, tok, probe, layer
        self.risk_probe, self.risk_layer, self.risk_threshold = risk_probe, risk_layer, risk_threshold
        self.threshold, self.calib = threshold, calib
        self.combine = combine          # judge by the mean of the probe's P(correct) and the token's probability
        self.stop_ids = {i for i in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|endoftext|>"),
                                     tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0}

    def p_risk(self, h):
        z = float(self.risk_probe.logit(h.float().reshape(1, -1).cpu().numpy())[0])
        return float(1 / (1 + np.exp(-z)))

    def p_correct(self, h):
        z = float(self.probe.logit(h.float().reshape(1, -1).cpu().numpy())[0])
        return float(1 / (1 + np.exp(-(self.calib[0] * z + self.calib[1]))))

    def _is_claim_start(self, piece, at_word_start, context_words):
        w = piece.strip().strip(OPENERS)
        if not at_word_start or not w:
            return False
        if not (w[0].isupper() or w[0].isdigit()) or w.lower() in SKIP:
            return False
        return not any(c.startswith(norm(w)) for c in context_words if norm(w))

    def generate(self, messages, max_new_tokens=128, mode="stop", context=None):
        """context: the text whose words are not claims when repeated (default: the whole conversation;
        with documents, pass only the question and history so grounded claims are still reported)."""
        return self.generate_batch([{"messages": messages, "max_new_tokens": max_new_tokens, "mode": mode,
                                     "context": context}])[0]

    @torch.no_grad()
    def generate_batch(self, requests, on_text=None):
        """Several replies at once: left-padded prompts, one shared KV cache, one forward pass per step for
        all of them. Every sequence feeds exactly one token per step: its next token, or a queued hedge
        marker, or (once finished) padding that nothing attends to.

        on_text(i, delta): streams request i's reply as it is written. With a risk probe, text is released
        one checked sentence at a time (a sentence that stop mode drops is never shown); without one,
        token by token. requests: [{"messages",
        "max_new_tokens"?, "mode"?, "context"?}]; returns one result per request, as generate()."""
        tok, dev = self.tok, next(self.model.parameters()).device
        seqs = [_Seq(self, r["messages"], r.get("max_new_tokens", 128), r.get("mode", "stop"), r.get("context"))
                for r in requests]
        prompts = [tok(prompt_of(tok, r["messages"])).input_ids for r in requests]
        B, P = len(prompts), max(len(p) for p in prompts)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((B, P), pad, dtype=torch.long)
        mask = torch.zeros((B, P), dtype=torch.long)
        for b, p in enumerate(prompts):
            ids[b, P - len(p):], mask[b, P - len(p):] = torch.tensor(p), 1
        ids, mask = ids.to(dev), mask.to(dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        past, cur = None, ids
        while True:
            o = self.model(input_ids=cur, attention_mask=mask, position_ids=pos, past_key_values=past,
                           use_cache=True, output_hidden_states=True)
            past = o.past_key_values
            h = o.hidden_states[self.layer + 1][:, -1]
            hr = o.hidden_states[self.risk_layer + 1][:, -1] if self.risk_probe is not None else None
            nxt = []
            for b, sq in enumerate(seqs):
                if hr is not None and not sq.closed:            # the state of the token just fed: its risk
                    sq.observe(hr[b])
                if not sq.done and not sq.feed:                 # a decision point for this sequence
                    sq.decide(o.logits[b, -1], h[b])
                if not sq.done and sq.feed:
                    t, sq.last_fed = sq.feed.pop(0)
                    nxt.append(t)
                else:
                    sq.closed = sq.done                         # nothing more to observe once it stops feeding
                    nxt.append(None)
            if on_text is not None:
                for b, sq in enumerate(seqs):
                    sq.stream(b, on_text, final=False)
            if all(t is None for t in nxt):
                break
            live = torch.tensor([[t is not None] for t in nxt], dtype=torch.long, device=dev)
            cur = torch.tensor([[pad if t is None else t] for t in nxt], device=dev)
            pos = pos[:, -1:] + live
            mask = torch.cat([mask, live], 1)
        return [sq.result() for sq in seqs]


class _Seq:
    """One reply's decoding state for DoubtAwareGenerator.generate_batch."""

    def __init__(self, gen, messages, max_new_tokens, mode, context):
        self.g, self.mode, self.left = gen, mode, max_new_tokens
        if context is None:
            context = " ".join(m["content"] for m in messages if m["role"] != "system")
        self.context_words = set(norm(context).split())
        self.out, self.claims, self.first, self.feed = [], [], None, []       # feed: [(token, index in out or None)]
        self.at_word_start, self.stopped, self.done, self.closed = True, False, False, False
        self.risk, self.sentences, self.sent_start, self.last_fed = {}, [], 0, None
        self.emitted = ""

    def stream(self, b, on_text, final):
        """Send the part of the reply that is safe to show: up to the last checked sentence while a risk
        probe is on and the reply is still being written, else everything written so far."""
        if (final or self.done) and self.g.risk_probe is not None and not self.stopped:
            self._close_sentence(len(self.out))                    # check the last sentence before showing it
        safe = len(self.out) if (final or self.done or self.g.risk_probe is None) else self.sent_start
        text = self.g.tok.decode(self.out[:safe], skip_special_tokens=True)
        if text.startswith(self.emitted) and len(text) > len(self.emitted) and not text.endswith("\ufffd"):
            on_text(b, text[len(self.emitted):])
            self.emitted = text

    def observe(self, h):
        """Risk of the token just fed (its own-position state); closes a sentence at its boundary."""
        j, self.last_fed = self.last_fed, None
        if j is None or j >= len(self.out):
            return
        self.risk[j] = self.g.p_risk(h)
        piece = self.g.tok.decode(self.out[j:j + 1])
        if piece.rstrip(" \"'”’)").endswith((".", "!", "?")) or "\n" in piece:
            so_far = self.g.tok.decode(self.out[self.sent_start:j + 1]).strip()
            if not LIST_MARK.fullmatch(so_far):                 # "1." opening a list item is not a sentence
                self._close_sentence(j + 1)

    def _close_sentence(self, end):
        if end <= self.sent_start:
            return
        tok = self.g.tok
        content = [self.risk[k] for k in range(self.sent_start, end)
                   if k in self.risk and any(ch.isalnum() for ch in tok.decode(self.out[k:k + 1]))]
        r = max(content) if content else 0.0
        sent = {"tokens": (self.sent_start, end), "risk": round(r, 4), "doubted": r >= self.g.risk_threshold}
        if sent["doubted"] and self.mode == "stop":             # drop the doubted sentence and end the reply
            self.out = self.out[:self.sent_start]
            self.claims = [c for c in self.claims if c["token_index"] < self.sent_start]
            self.stopped = self.done = True
            self.feed = []
            sent["dropped"] = True
        self.sentences.append(sent)
        self.sent_start = end

    def decide(self, logits, h):
        g, tok = self.g, self.g.tok
        if self.left <= 0:
            self.done = True
            return
        self.left -= 1
        nxt = int(logits.argmax())
        if nxt in g.stop_ids:
            self.done = True
            return
        piece = tok.decode([nxt])
        if not piece.strip() or (self.at_word_start or piece[:1].isspace()) and not piece.strip().strip(OPENERS):
            self.at_word_start = True                               # " 1958" -> " ", "1"; ' "Rent' -> ' "', "Rent"
            self.out.append(nxt)
            self.feed = [(nxt, len(self.out) - 1)]
            return
        starts_word = self.at_word_start or piece[:1].isspace()
        tp = None
        if self.first is None:                                      # the reply's first content token: scored
            tp = float(torch.softmax(logits.float(), -1)[nxt])      # even when no claim follows
            self.first = {"p_correct": round(g.p_correct(h), 4), "token_p": round(tp, 4)}
        self.feed = [(nxt, len(self.out))]
        if g._is_claim_start(piece, starts_word, self.context_words):
            p = g.p_correct(h)
            tp = float(torch.softmax(logits.float(), -1)[nxt]) if tp is None else tp
            score = (p + tp) / 2 if g.combine else p                # best or tied at every abstention level
            c = {"token_index": len(self.out), "p_correct": round(p, 4), "token_p": round(tp, 4),
                 "score": round(score, 4), "doubted": score < g.threshold}
            self.claims.append(c)
            if c["doubted"] and self.mode == "stop":
                self.stopped = self.done = True
                return
            if c["doubted"] and self.mode == "hedge":
                forced = tok(HEDGE, add_special_tokens=False).input_ids
                self.out += forced
                c["token_index"] = len(self.out)                    # the claim itself follows the marker
                self.feed = [(t, None) for t in forced] + [(nxt, len(self.out))]
        self.out.append(nxt)
        self.at_word_start = piece[-1:].isspace() or piece.endswith("\n")

    def result(self):
        if self.g.risk_probe is not None and not self.stopped:
            self._close_sentence(len(self.out))                    # the last, unterminated sentence (may drop it)
        tok, out, claims, first, stopped = self.g.tok, self.out, self.claims, self.first, self.stopped
        raw = tok.decode(out, skip_special_tokens=True)
        lead = len(raw) - len(raw.lstrip())
        text = raw.strip()
        for c in claims:                                            # the claim's word and its span in `text`
            j = c["token_index"]
            k = j + 1
            while k < len(out) and not tok.decode(out[k:k + 1])[:1].isspace():
                piece = tok.decode(out[k:k + 1])
                if piece in (",", ".") and tok.decode(out[k - 1:k])[-1:].isdigit() and \
                        tok.decode(out[k + 1:k + 2])[:1].isdigit():
                    k += 1                                          # "1,063", "3.5": one number
                    continue
                if piece in (",", ".", "\n"):
                    break
                k += 1
            c["text"] = tok.decode(out[j:k]).strip() if j < len(out) else ""
            a = len(tok.decode(out[:j], skip_special_tokens=True)) - lead
            a += len(tok.decode(out[j:k])) - len(tok.decode(out[j:k]).lstrip())
            c["start"], c["end"] = max(0, a), max(0, a) + len(c["text"])
        claims = [c for c in claims if not (c["text"].isdigit() and text[c["end"]:c["end"] + 1] in ".)"   # list numbers
                                            and (c["start"] == 0 or text[c["start"] - 1] == "\n"))]
        g = self.g
        fs = None if first is None else ((first["p_correct"] + first["token_p"]) / 2 if g.combine else first["p_correct"])
        if not claims and fs is not None and fs < g.threshold and self.mode == "stop" and out:
            stopped = True                                          # no claim found: judge the reply by its first token
            text = ""
        if stopped:                                                 # keep only whole sentences before the doubt
            cut = max(text.rfind(ch) for ch in ".!?\n")
            text = (text[:cut + 1].strip() + " " if cut >= 0 else "") + STOP_NOTE
        sentences = []
        if g.risk_probe is not None:
            for st in self.sentences:
                if st.get("dropped"):
                    continue
                a0, b0 = st["tokens"]
                pre = tok.decode(out[:a0], skip_special_tokens=True)
                seg = tok.decode(out[a0:b0], skip_special_tokens=True)
                a = len(pre) - lead + (len(seg) - len(seg.lstrip()))
                seg = seg.strip().replace(HEDGE.strip(), "").strip()
                if seg:
                    sentences.append({"text": seg, "start": max(0, a), "end": max(0, a) + len(seg),
                                      "risk": st["risk"], "doubted": st["doubted"]})
        return {"text": text, "ids": out, "claims": claims, "first_token": first, "stopped_on_doubt": stopped,
                "sentences": sentences, "max_risk": max((x["risk"] for x in sentences), default=None),
                "min_p_correct": min((c["p_correct"] for c in claims), default=first["p_correct"] if first else None)}
