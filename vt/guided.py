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
import numpy as np
import torch

from vt.claims import prompt_of
from vt.correctness import SKIP, norm

STOP_NOTE = "I'm not sure about this, so I won't guess."
OPENERS = "\"'“”‘’*#([«"
HEDGE = " [unsure:]"


class DoubtAwareGenerator:
    def __init__(self, model, tok, probe, layer, threshold=0.5, calib=(1.0, 0.0), combine=True):
        self.model, self.tok, self.probe, self.layer = model, tok, probe, layer
        self.threshold, self.calib = threshold, calib
        self.combine = combine          # judge by the mean of the probe's P(correct) and the token's probability
        self.stop_ids = {i for i in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|endoftext|>"),
                                     tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0}

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
    def generate_batch(self, requests):
        """Several replies at once: left-padded prompts, one shared KV cache, one forward pass per step for
        all of them. Every sequence feeds exactly one token per step: its next token, or a queued hedge
        marker, or (once finished) padding that nothing attends to. requests: [{"messages",
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
            nxt = []
            for b, sq in enumerate(seqs):
                if not sq.done and not sq.feed:                 # a decision point for this sequence
                    sq.decide(o.logits[b, -1], h[b])
                nxt.append(sq.feed.pop(0) if not sq.done and sq.feed else None)
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
        self.out, self.claims, self.first, self.feed = [], [], None, []
        self.at_word_start, self.stopped, self.done = True, False, False

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
            self.feed = [nxt]
            return
        starts_word = self.at_word_start or piece[:1].isspace()
        tp = None
        if self.first is None:                                      # the reply's first content token: scored
            tp = float(torch.softmax(logits.float(), -1)[nxt])      # even when no claim follows
            self.first = {"p_correct": round(g.p_correct(h), 4), "token_p": round(tp, 4)}
        self.feed = [nxt]
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
                self.feed = forced + [nxt]
        self.out.append(nxt)
        self.at_word_start = piece[-1:].isspace() or piece.endswith("\n")

    def result(self):
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
        g = self.g
        fs = None if first is None else ((first["p_correct"] + first["token_p"]) / 2 if g.combine else first["p_correct"])
        if not claims and fs is not None and fs < g.threshold and self.mode == "stop" and out:
            stopped = True                                          # no claim found: judge the reply by its first token
            text = ""
        if stopped:                                                 # keep only whole sentences before the doubt
            cut = max(text.rfind(ch) for ch in ".!?\n")
            text = (text[:cut + 1].strip() + " " if cut >= 0 else "") + STOP_NOTE
        return {"text": text, "ids": out, "claims": claims, "first_token": first, "stopped_on_doubt": stopped,
                "min_p_correct": min((c["p_correct"] for c in claims), default=first["p_correct"] if first else None)}
