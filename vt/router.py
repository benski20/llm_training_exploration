"""Which kind of message is this? One forward pass of the chat model over a short classification prompt (no
generation), so each kind gets the check that fits it:

    information  facts, explanations, how-to, news, products: answer from a search and check the answer
                 against those sources
    math         calculations and quantitative reasoning: re-compute the arithmetic (vt/mathcheck.py)
    code         programming: (run the code, sandboxed)
    chat         creative writing, opinions, advice, small talk: nothing to verify

    Router(model, tok).route(messages)   # -> ("information", {"information": 0.91, "math": 0.03, ...})
"""
import re

from vt.claim_verify import choice_probs, option_ids
from vt.claims import prompt_of

KINDS = ("information", "math", "code", "chat")
PROMPT = ("Classify the user's last message.\n"
          "A) asks for information: facts, explanations, how-to, history, news, health, products, places, people\n"
          "B) math or a calculation\n"
          "C) programming or code\n"
          "D) creative writing, an opinion, personal advice, or small talk\n"
          "Answer with one letter.")


# unambiguous signals decide before the model is asked (a small model leans to "information")
CODE = re.compile(r"```|\bdef \w+\(|\bfor \w+ in \w+|\bprint\(|\bfunction\b|\bclass \w+[:(]|\bSyntaxError\b|"
                  r"\bTraceback\b|\b(python|javascript|typescript|java|c\+\+|rust|golang|sql|regex|bash|html|css)\b|"
                  r"\b(compile|debug|refactor|stack trace|api endpoint)\b", re.I)
CREATIVE = re.compile(r"\b(write|compose|make up|give me)\b.{0,30}\b(poem|haiku|story|song|limerick|lyrics|joke|"
                      r"essay|toast|speech|rap)\b", re.I)
OPINION = re.compile(r"^\s*(should i|what do you think|do you (like|prefer|think)|would you rather|how are you|"
                     r"hi\b|hello\b|thanks|thank you)", re.I)


def rule(text):
    """The kind an unambiguous signal names, else None."""
    if CODE.search(text):
        return "code"
    if CREATIVE.search(text) or OPINION.search(text):
        return "chat"
    return None


class Router:
    def __init__(self, model, tok, batch=8):
        self.model, self.tok, self.batch = model, tok, batch
        self.letters = [option_ids(tok, c) for c in "ABCD"]

    def _prompt(self, messages):
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        msgs = [{"role": "system", "content": "You sort chatbot messages. Answer only A, B, C or D."},
                {"role": "user", "content": f"Message: {last[:2000]}\n\n{PROMPT}"}]
        return prompt_of(self.tok, msgs, thinking=False)

    def route_batch(self, conversations):
        lasts = [next((m["content"] for m in reversed(c) if m["role"] == "user"), "") for c in conversations]
        ruled = [rule(t) for t in lasts]
        ask = [c for c, r in zip(conversations, ruled) if r is None]
        probs = iter(choice_probs(self.model, self.tok, [self._prompt(m) for m in ask], self.letters, self.batch) if ask else ())
        out = []
        for r in ruled:
            if r is not None:
                out.append((r, {k: float(k == r) for k in KINDS}))
            else:
                p = next(probs)
                out.append((KINDS[max(range(4), key=lambda i: p[i])], {k: round(v, 4) for k, v in zip(KINDS, p)}))
        return out

    def route(self, messages):
        return self.route_batch([messages])[0]
