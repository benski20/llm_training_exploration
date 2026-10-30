"""Which kind of message is this? One forward pass of the chat model over a short classification prompt (no
generation), so each kind gets the check that fits it:

    information  facts, explanations, how-to, news, products: answer from a search and check the answer
                 against those sources
    math         calculations and quantitative reasoning: re-compute the arithmetic (vt/mathcheck.py)
    code         programming: (run the code, sandboxed)
    chat         creative writing, opinions, advice, small talk: nothing to verify
    text         a task on text the user pasted (summarize, extract, answer from it): check the reply
                 against that text (decided by the message's shape, vt.router.pasted)

    Router(model, tok).route(messages)   # -> ("information", {"information": 0.91, "math": 0.03, ...})
"""
import os
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


# a task on text the user pasted: a long message with an instruction about "the text", or a very long one
PASTE_CUE = re.compile(r"\b(summari[sz]e|summary|tl;?dr|extract|according to|based on|from (this|the) (text|passage|"
                       r"article|email|document)|the (text|passage|article|paragraph|excerpt|email|document|report) "
                       r"(above|below)|(above|below|following) (text|passage|article|paragraph|excerpt|email|report))\b",
                       re.I)


def pasted(text):
    """True for a task on text the user supplies (summarize this, what does the passage say about...)."""
    words = len(text.split())
    return words >= 300 or (words >= 40 and bool(PASTE_CUE.search(text)))


def rule(text):
    """The kind an unambiguous signal names, else None."""
    if CODE.search(text):
        return "code"
    if pasted(text):
        return "text"
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
                out.append((r, {k: float(k == r) for k in KINDS + ("text",)}))
            else:
                p = next(probs)
                out.append((KINDS[max(range(4), key=lambda i: p[i])], dict({k: round(v, 4) for k, v in zip(KINDS, p)}, text=0.0)))
        return out

    def route(self, messages):
        return self.route_batch([messages])[0]


class TextRouter:
    """The same sorting without the chat model: a small sentence embedding (CPU, a few ms) and a softmax layer
    trained on human-written prompts (experiments/train_router.py), after the same rules.

        TextRouter("vt/router_head.npz").route(messages)   # -> ("chat", {"information": 0.08, ...})
    """

    def __init__(self, path=None, embedder=None):
        import numpy as np
        z = np.load(path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "router_head.npz"))
        self.W, self.b, self.kinds = z["W"], z["b"], tuple(str(k) for k in z["kinds"])
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(str(z["embed"]), device="cpu")
        self.embedder = embedder

    def probs(self, texts):
        import numpy as np
        X = self.embedder.encode([t[:2000] for t in texts], batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False)
        Z = X @ self.W + self.b
        P = np.exp(Z - Z.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)

    def route_batch(self, conversations):
        lasts = [next((m["content"] for m in reversed(c) if m["role"] == "user"), "") for c in conversations]
        ruled = [rule(t) for t in lasts]
        ask = [t for t, r in zip(lasts, ruled) if r is None]
        probs = iter(self.probs(ask) if ask else ())
        out = []
        for r in ruled:
            if r is not None:
                out.append((r, {k: float(k == r) for k in self.kinds + ("text",)}))
            else:
                p = next(probs)
                out.append((self.kinds[int(p.argmax())], dict({k: round(float(v), 4) for k, v in zip(self.kinds, p)}, text=0.0)))
        return out

    def route(self, messages):
        return self.route_batch([messages])[0]
