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
import re
import string
from collections import Counter


def norm(s):                                             # as vt.correctness.norm, without importing torch
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())

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
    """Live Wikipedia lookups, no key and no index: search titles, then fetch each page's lead section
    (or, with full_text=True, the best-matching passages of the top pages' whole text).
    Fails quietly (no passages, so the reply is simply not looked up); answers are cached.

        r = Wikipedia(k=2)
        r("Who designed the Eiffel Tower?")         # -> ["Gustave Eiffel ... (lead section)", ...]
    """
    SEARCH = "https://{lang}.wikipedia.org/w/rest.php/v1/search/page?q={q}&limit={n}"
    SUMMARY = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    TEXT = ("https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1"
            "&format=json&formatversion=2&titles={title}")

    # Wikimedia asks for a User-Agent naming the tool and a contact; anonymous-looking clients from cloud
    # addresses get rate-limited first
    UA = "VeriChat/0.2 (https://github.com/benski20/llm_training_exploration; research demo) python-urllib"

    def __init__(self, k=2, lang="en", timeout=4.0, user_agent=UA, cache_size=2048,
                 fetch=None, full_text=False, pages=3, passage_words=100):
        """full_text: search the whole text of the top `pages` hits and return the k passages that best match
        the query (BM25), instead of each hit's lead section: the answer is often further down a page."""
        self.k, self.lang, self.timeout, self.ua = k, lang, timeout, user_agent
        self.full_text, self.pages, self.passage_words = full_text, pages, passage_words
        self.errors = Counter()                              # why lookups failed (e.g. "HTTP 429"), for /health
        self.cache, self.cache_size = {}, cache_size
        self.fetch = fetch or self._fetch                    # url -> parsed JSON (injectable for tests)

    def _fetch(self, url):
        import json
        import urllib.request
        if not url.startswith("https://"):                  # only the fixed https endpoints above
            raise ValueError("Wikipedia URLs must be https")
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosec B310: https checked above
            return json.loads(r.read().decode())

    def __call__(self, query, k=None):
        import urllib.parse
        k = k or self.k
        key = (norm(query), k)
        if key in self.cache:
            return self.cache[key]
        try:
            n = max(k, self.pages) if self.full_text else k
            hits = self.fetch(self.SEARCH.format(lang=self.lang, q=urllib.parse.quote(query), n=n)).get("pages", [])
            out = self._passages(query, hits[:n], k) if self.full_text else self._leads(hits[:k])
        except Exception as e:                               # network, rate limit, bad JSON: no lookup
            self.errors[self._why(e)] += 1
            return []
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out

    @staticmethod
    def _why(e):
        code = getattr(e, "code", None)
        return f"HTTP {code}" if code else type(e).__name__

    @staticmethod
    def _key(h):
        import urllib.parse
        return urllib.parse.quote(h.get("key") or h.get("title", "").replace(" ", "_"), safe="")

    def _leads(self, hits):
        out = []
        for h in hits:
            s = self.fetch(self.SUMMARY.format(lang=self.lang, title=self._key(h)))
            text = (s.get("extract") or "").strip()
            if text:
                out.append(f"{s.get('title', h.get('title', ''))} (Wikipedia): {text}")
        return out

    def _passages(self, query, hits, k):
        """The k passages of the hits' full texts that best match the query; a page whose text cannot be
        fetched contributes its lead instead."""
        cands = []                                           # (title, page, window, text)
        for j, h in enumerate(hits):
            title = h.get("title", "")
            try:
                page = self.fetch(self.TEXT.format(lang=self.lang, title=self._key(h)))["query"]["pages"][0]
                text, title = (page.get("extract") or "").strip(), page.get("title", title)
            except Exception as e:                           # rate limit, missing page: fall back to the lead
                self.errors["text: " + self._why(e)] += 1
                text = ""
            if not text:
                lead = self._leads([h])
                cands += [(title, j, 0, p.split(": ", 1)[-1]) for p in lead]
                continue
            w = text.split()[:30000]
            cands += [(title, j, i, c) for i, c in enumerate(chunks(" ".join(w), self.passage_words, self.passage_words // 2))]
        if not cands:
            return []
        corpus = Corpus([f"{t} {c}" for t, _, _, c in cands])  # the title counts: "Eiffel Tower ... 324 m"
        s = corpus.scores(query)
        best = sorted(range(len(cands)), key=lambda i: -s[i])
        out, taken = [], []
        for i in best:
            if len(out) == k or s[i] <= 0:
                break
            t, j, w, c = cands[i]
            if any(j == j2 and abs(w - w2) <= 1 for j2, w2 in taken):   # windows overlap: a near-repeat
                continue
            taken.append((j, w))
            out.append(f"{t} (Wikipedia): {c}")
        return out
