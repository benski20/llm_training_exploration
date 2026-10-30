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
import time
from collections import Counter


def norm(s):                                             # as vt.correctness.norm, without importing torch
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())

STOP = set("the a an of in on at to for and or is was were who what which when where how why did does do by with "
           "from as that this it its be are".split())


QUESTION_WORDS = set("whom whose which name named known called famous first year city country film book "
                     "would could you your someone".split())


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
                 fetch=None, full_text=False, pages=3, passage_words=100, per_page=None, budget_s=10.0,
                 in_flight=4, interval=0.05,
                 rerank=None, rerank_pool=20):
        """full_text: search the whole text of the top `pages` hits and return the k passages that best match
        the query (BM25), instead of each hit's lead section: the answer is often further down a page."""
        self.k, self.lang, self.timeout, self.ua = k, lang, timeout, user_agent
        self.full_text, self.pages, self.passage_words = full_text, pages, passage_words
        self.per_page = per_page                             # at most this many passages from one page (None: any)
        self.rerank, self.rerank_pool = rerank, rerank_pool  # (query, passages) -> scores: reorders BM25's best
        self.budget_s = budget_s                             # a lookup starts no new fetch after this long: the
                                                             # socket timeout does not cover a slow DNS answer
        self.errors = Counter()                              # why lookups failed (e.g. "HTTP 429"), for /health
        import threading
        self._local = threading.local()
        self.pages_cache = {}                                # page key -> (title, full text)
        # interactive defaults; bulk runs (studies) should go much slower: in_flight=1, interval=0.4
        self._gate, self._pace, self._last = threading.BoundedSemaphore(in_flight), threading.Lock(), 0.0
        self.interval = interval
        self.cache, self.cache_size = {}, cache_size
        self.fetch = fetch or self._fetch                    # url -> parsed JSON (injectable for tests)

    def _get(self, url):
        """One request, paced: at most in_flight at once per process and interval seconds between starts
        (Wikipedia throttles bursts from cloud addresses: a 600-reply study drew about 1000 HTTP 429s unpaced
        and about 600 at the interactive pace)."""
        if time.monotonic() - getattr(self._local, "t0", time.monotonic()) > self.budget_s:
            raise TimeoutError("lookup time budget spent")
        with self._gate:
            with self._pace:
                wait = self._last + self.interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            return self.fetch(url)

    def _fetch(self, url, retries=1):
        import json
        import urllib.error
        import urllib.request
        if not url.startswith("https://"):                  # only the fixed https endpoints above
            raise ValueError("Wikipedia URLs must be https")
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosec B310: https checked above
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:                  # rate-limited: wait as asked (at most 2 s), once
            if e.code != 429 or retries <= 0:
                raise
            try:
                wait = min(2.0, float(e.headers.get("Retry-After") or 1))
            except ValueError:
                wait = 1.0
            time.sleep(wait)
            return self._fetch(url, retries - 1)

    def __call__(self, query, k=None):
        import urllib.parse
        k = k or self.k
        key = (norm(query), k)
        if key in self.cache:
            return self.cache[key]
        self._local.t0 = time.monotonic()                   # per thread: lookups may run in parallel
        try:
            n = max(k, self.pages) if self.full_text else k
            hits = []
            for q in self._queries(query):                   # a long question can match no page as a whole
                hits = self._get(self.SEARCH.format(lang=self.lang, q=urllib.parse.quote(q), n=n)).get("pages", [])
                if hits:
                    break
            out = self._passages(query, hits[:n], k) if self.full_text else self._leads(hits[:k])
        except Exception as e:                               # network, rate limit, bad JSON: no lookup
            self.errors[self._why(e)] += 1
            return []
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out

    @staticmethod
    def _queries(query):
        """The query, then its content words, then only its names and numbers (each if different)."""
        toks = [t.strip(".,;:!?\"'()") for t in query.replace('"', " ").split()]
        content = [t for t in toks if t and t.lower() not in STOP | QUESTION_WORDS]
        names = [t for t in content if t[:1].isupper() or t[:1].isdigit()]
        out = []
        for q in (query, " ".join(content), " ".join(names)):
            if q and q not in out:
                out.append(q)
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
            s = self._get(self.SUMMARY.format(lang=self.lang, title=self._key(h)))
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
            key = self._key(h)
            if key in self.pages_cache:                      # a reply's claims mostly land on the same pages
                title, text = self.pages_cache[key]
            else:
                try:
                    page = self._get(self.TEXT.format(lang=self.lang, title=key))["query"]["pages"][0]
                    text, title = (page.get("extract") or "").strip(), page.get("title", title)
                    if text:
                        if len(self.pages_cache) >= 256:
                            self.pages_cache.pop(next(iter(self.pages_cache)))
                        self.pages_cache[key] = (title, text)
                except Exception as e:                       # rate limit, missing page: fall back to the lead
                    self.errors["text: " + self._why(e)] += 1
                    text = ""
            if not text:
                try:
                    lead = self._leads([h])
                except Exception as e:                       # out of time too: rank what was found
                    self.errors["lead: " + self._why(e)] += 1
                    lead = []
                cands += [(title, j, 0, p.split(": ", 1)[-1]) for p in lead]
                continue
            w = text.split()[:30000]
            cands += [(title, j, i, c) for i, c in enumerate(chunks(" ".join(w), self.passage_words, self.passage_words // 2))]
        return rank_passages(query, [(t, j, w, c, "Wikipedia") for t, j, w, c in cands], k, self.rerank,
                             self.rerank_pool, self.per_page, self.errors)


def rank_passages(query, cands, k, rerank=None, rerank_pool=20, per_page=None, errors=None):
    """The k candidate passages that best match the query: BM25 over title + text, the best rerank_pool
    reordered by rerank(query, passages) -> scores when given, overlapping windows of one page skipped.
    cands: [(title, page, window, text, source)] -> ["title (source): text"]."""
    if not cands:
        return []
    corpus = Corpus([f"{t} {c}" for t, _, _, c, _ in cands])   # the title counts: "Eiffel Tower ... 324 m"
    s = corpus.scores(query)
    best = sorted(range(len(cands)), key=lambda i: -s[i])
    pool = [i for i in best[:rerank_pool] if s[i] > 0]
    if rerank is not None and pool:                          # a stronger reader reorders BM25's best few
        try:
            r = rerank(query, [f"{cands[i][0]}: {cands[i][3]}" for i in pool])
            r = [float(x) for x in (r.reshape(-1) if hasattr(r, "reshape") else (r if isinstance(r, (list, tuple)) else [r]))]
            order = [pool[j] for j in sorted(range(len(pool)), key=lambda j: -r[j])]
            best = order + [i for i in best if i not in set(pool)]
        except Exception as e:                               # keep BM25's order
            if errors is not None:
                errors["rerank: " + (f"HTTP {e.code}" if getattr(e, "code", None) else type(e).__name__)] += 1
    out, taken = [], []
    for i in best:
        if len(out) == k or s[i] <= 0:
            break
        t, j, w, c, src = cands[i]
        if any(j == j2 and abs(w - w2) <= 1 for j2, w2 in taken):   # windows overlap: a near-repeat
            continue
        if per_page and sum(j == j2 for j2, _ in taken) >= per_page:
            continue
        taken.append((j, w))
        out.append(f"{t} ({src}): {c}")
    return out
