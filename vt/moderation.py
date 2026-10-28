"""Content moderation for a public VeriChat: screen what users send and what the model writes.

    mod = Moderator.load()                     # unitary/toxic-bert, multi-label toxicity (~110M parameters)
    mod.check("some text") -> {"flagged": bool, "scores": {"toxic": 0.98, "threat": 0.4, ...}}

It flags toxic, threatening, obscene and hateful language. It is not a harmful-intent classifier: a
polite request for dangerous instructions is left to the chat model's own refusals (Qwen2.5-Instruct
refuses most). Any callable text -> {"flagged", "scores"} can replace it.
"""


class Moderator:
    def __init__(self, classify, thresholds=None):
        self.classify = classify                     # text -> {label: probability}
        self.thresholds = thresholds or {"toxic": 0.8, "severe_toxic": 0.5, "threat": 0.5, "identity_hate": 0.5,
                                         "obscene": 0.9, "insult": 0.9}

    @classmethod
    def load(cls, model="unitary/toxic-bert", thresholds=None):
        from transformers import pipeline
        pipe = pipeline("text-classification", model=model, top_k=None, truncation=True)
        return cls(lambda text: {d["label"]: float(d["score"]) for d in pipe(text[:4000])[0]}, thresholds)

    def check(self, text):
        scores = self.classify(text) if text.strip() else {}
        return {"flagged": any(scores.get(k, 0.0) >= t for k, t in self.thresholds.items()),
                "scores": {k: round(v, 4) for k, v in scores.items()}}


REFUSAL = "I can't help with that message."
WITHHELD = "I wrote a reply that my content filter blocked, so I'm not showing it."
