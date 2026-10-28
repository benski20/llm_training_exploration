"""Retrieval for doubt-triggered answering: when VeriChat's probe doubts a closed-book reply, it looks
the question up and answers again from what it found (experiments/triggered_retrieval.py measures
when this pays).

    r = Corpus.from_texts(["...", "..."])          # or Corpus.from_dir("docs/")
    r("Who designed the Eiffel Tower?", k=2)        # -> the k best passages (BM25 over word chunks)

Any callable question -> list of passage strings works as a retriever (a search API, a vector store);
Wikipedia() looks pages up live.
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


class Wikipedia:
    """Live Wikipedia lookups, no key and no index: search titles, then fetch each page's lead section.
    Fails quietly (no passages, so the reply is simply not looked up); answers are cached.

        r = Wikipedia(k=2)
        r("Who designed the Eiffel Tower?")         # -> ["Gustave Eiffel ... (lead section)", ...]
    """
    SEARCH = "https://{lang}.wikipedia.org/w/rest.php/v1/search/page?q={q}&limit={n}"
    SUMMARY = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

    def __init__(self, k=2, lang="en", timeout=4.0, user_agent="VeriChat/0.1 (research demo)", cache_size=2048,
                 fetch=None):
        self.k, self.lang, self.timeout, self.ua = k, lang, timeout, user_agent
        self.cache, self.cache_size = {}, cache_size
        self.fetch = fetch or self._fetch                    # url -> parsed JSON (injectable for tests)

    def _fetch(self, url):
        import json
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def __call__(self, query, k=None):
        import urllib.parse
        k = k or self.k
        key = (norm(query), k)
        if key in self.cache:
            return self.cache[key]
        try:
            hits = self.fetch(self.SEARCH.format(lang=self.lang, q=urllib.parse.quote(query), n=k)).get("pages", [])
            out = []
            for h in hits[:k]:
                title = urllib.parse.quote(h.get("key") or h.get("title", "").replace(" ", "_"), safe="")
                s = self.fetch(self.SUMMARY.format(lang=self.lang, title=title))
                text = (s.get("extract") or "").strip()
                if text:
                    out.append(f"{s.get('title', h.get('title', ''))} (Wikipedia): {text}")
        except Exception:                                    # network, rate limit, bad JSON: no lookup
            return []
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out
