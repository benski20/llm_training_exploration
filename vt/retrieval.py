"""Retrieval for doubt-triggered answering: when VeriChat's probe doubts a closed-book reply, it looks
the question up and answers again from what it found (experiments/triggered_retrieval.py measures
when this pays).

    r = Corpus.from_texts(["...", "..."])          # or Corpus.from_dir("docs/")
    r("Who designed the Eiffel Tower?", k=2)        # -> the k best passages (BM25 over word chunks)

Any callable question -> list of passage strings works as a retriever (a search API, a vector store).
"""
import math
import os
from collections import Counter

from vt.correctness import norm

STOP = set("the a an of in on at to for and or is was were who what which when where how why did does do by with "
           "from as that this it its be are".split())


def words(text):
    return [w for w in norm(text).split() if w not in STOP]


def chunks(text, size=120, stride=60):
    """Overlapping windows of `size` words (passages short enough to put several in a prompt)."""
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, max(len(w) - stride, 1), stride)] or [text]


class Corpus:
    def __init__(self, passages, k1=1.5, b=0.75):
        self.passages = passages
        self.tf = [Counter(words(p)) for p in passages]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(w for c in self.tf for w in c)
        n = len(passages)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    @classmethod
    def from_texts(cls, texts, size=120):
        return cls([c for t in texts for c in chunks(t, size, size // 2)])

    @classmethod
    def from_dir(cls, path, size=120):
        texts = []
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith((".txt", ".md")):
                    texts.append(open(os.path.join(root, f), encoding="utf-8", errors="ignore").read())
        return cls.from_texts(texts, size)

    def scores(self, query):
        q = set(words(query))
        out = []
        for tf, n in zip(self.tf, self.len):
            s = 0.0
            for w in q & tf.keys():
                f = tf[w]
                s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * n / (self.avg or 1)))
            out.append(s)
        return out

    def __call__(self, query, k=2):
        s = self.scores(query)
        best = sorted(range(len(s)), key=lambda i: -s[i])[:k]
        return [self.passages[i] for i in best if s[i] > 0]
