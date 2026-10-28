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

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=128, mode="stop"):
        tok, dev = self.tok, next(self.model.parameters()).device
        context_words = set(norm(" ".join(m["content"] for m in messages if m["role"] != "system")).split())
        ids = tok(prompt_of(tok, messages), return_tensors="pt").input_ids.to(dev)
        past, cur, out, claims, first = None, ids, [], [], None
        at_word_start, stopped = True, False
        for _ in range(max_new_tokens):
            o = self.model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            if nxt in self.stop_ids:
                break
            piece = tok.decode([nxt])
            if not piece.strip() or (at_word_start or piece[:1].isspace()) and not piece.strip().strip(OPENERS):
                at_word_start = True                                # " 1958" -> " ", "1"; ' "Rent' -> ' "', "Rent"
                out.append(nxt)
                cur = torch.tensor([[nxt]], device=dev)
                continue
            starts_word = at_word_start or piece[:1].isspace()
            if first is None:                                       # the reply's first content token: scored
                first = {"p_correct": round(self.p_correct(o.hidden_states[self.layer + 1][0, -1]), 4),  # even when
                         "token_p": round(float(torch.softmax(o.logits[0, -1].float(), -1)[nxt]), 4)}   # no claim
            if self._is_claim_start(piece, starts_word, context_words):
                p = self.p_correct(o.hidden_states[self.layer + 1][0, -1])
                tp = float(torch.softmax(o.logits[0, -1].float(), -1)[nxt])
                score = (p + tp) / 2 if self.combine else p    # best or tied at every abstention level
                c = {"token_index": len(out), "p_correct": round(p, 4), "token_p": round(tp, 4),
                     "score": round(score, 4), "doubted": score < self.threshold}
                claims.append(c)
                if c["doubted"] and mode == "stop":
                    stopped = True
                    break
                if c["doubted"] and mode == "hedge":
                    forced = tok(HEDGE, add_special_tokens=False).input_ids
                    out += forced
                    o2 = self.model(input_ids=torch.tensor([forced], device=dev), past_key_values=past, use_cache=True)
                    past = o2.past_key_values
            out.append(nxt)
            at_word_start = piece[-1:].isspace() or piece.endswith("\n")
            cur = torch.tensor([[nxt]], device=dev)
        text = tok.decode(out, skip_special_tokens=True).strip()
        for c in claims:                                            # the claim's word, for display
            j = c["token_index"]
            k = j + 1
            while k < len(out) and not tok.decode(out[k:k + 1])[:1].isspace() and tok.decode(out[k:k + 1]) not in (",", ".", "\n"):
                k += 1
            c["text"] = tok.decode(out[j:k]).strip() if j < len(out) else ""
        fs = None if first is None else ((first["p_correct"] + first["token_p"]) / 2 if self.combine else first["p_correct"])
        if not claims and fs is not None and fs < self.threshold and mode == "stop" and out:
            stopped = True                                          # no claim found: judge the reply by its first token
            text = ""
        if stopped:                                                 # keep only whole sentences before the doubt
            cut = max(text.rfind(ch) for ch in ".!?\n")
            text = (text[:cut + 1].strip() + " " if cut >= 0 else "") + STOP_NOTE
        return {"text": text, "claims": claims, "first_token": first, "stopped_on_doubt": stopped,
                "min_p_correct": min((c["p_correct"] for c in claims), default=first["p_correct"] if first else None)}
