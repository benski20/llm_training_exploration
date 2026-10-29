"""Web search as a retriever, next to Wikipedia: a search API finds pages, their text is ranked into
passages the same way as Wikipedia's (vt.retrieval.rank_passages), and the claim checks judge them alike.

    w = WebSearch("brave", api_key)          # Brave Search API: results, then each page fetched and read
    w = WebSearch("tavily", api_key)         # Tavily: results with the page text included
    w("Who designed the Eiffel Tower?")      # -> ["Title (example.org): passage", ...]
    Union(Wikipedia(...), w)                 # both sources, reranked together

Fetching pages a search returned is the risky part, so it is narrow: http(s) only; every host, redirects
included, must resolve to public addresses (no loopback, private, link-local or reserved ones: no reaching
into the server's own network); a size cap; a timeout; HTML or plain text only. Page text is evidence to
be judged, never instructions: it only ever reaches the model inside the judging prompt as quoted
evidence. Fails quietly (no passages), with the reasons counted in .errors (shown in /health).
"""
import ipaddress
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

from vt.retrieval import chunks, rank_passages

UA = "VeriChat/0.2 (https://github.com/benski20/llm_training_exploration; research demo) python-urllib"
SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside", "template"}


class Blocked(Exception):
    """A URL the fetcher refuses (scheme, private address, size, type)."""


def public_host(host):
    """True when every address the host resolves to is public."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast \
                or ip.is_unspecified:
            return False
    return bool(infos)


def check_url(url):
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise Blocked(f"scheme or host: {url[:80]}")
    if not public_host(u.hostname):
        raise Blocked(f"not a public host: {u.hostname}")
    return url


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)                                    # a redirect may not lead inside either
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_CheckedRedirects)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def html_to_text(html):
    """(title, readable text) of an HTML page, scripts, styles and navigation left out."""
    p = _Text()
    try:
        p.feed(html)
    except Exception:                                        # malformed HTML: what was read so far
        pass
    lines = [" ".join(line.split()) for line in "".join(p.parts).split("\n")]
    return " ".join(p.title.split()), "\n".join(line for line in lines if len(line) > 1)


def fetch_page(url, timeout=5.0, max_bytes=2_000_000):
    """A page's (title, text), or Blocked / an urllib error."""
    req = urllib.request.Request(check_url(url), headers={"User-Agent": UA, "Accept": "text/html,text/plain"})
    with _opener.open(req, timeout=timeout) as r:            # nosec B310: http(s) and public hosts checked
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "html" not in ctype and "text/plain" not in ctype:
            raise Blocked(f"content type {ctype[:40]}")
        raw = r.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
        charset = r.headers.get_content_charset() or "utf-8"
    body = raw.decode(charset, errors="replace")
    return html_to_text(body) if "html" in ctype else ("", body)


class WebSearch:
    BRAVE = "https://api.search.brave.com/res/v1/web/search?q={q}&count={n}"
    TAVILY = "https://api.tavily.com/search"

    def __init__(self, provider, api_key, k=2, pages=5, passage_words=100, rerank=None, rerank_pool=20,
                 timeout=5.0, budget_s=12.0, search=None, fetch=None, cache_size=1024):
        """search(query, n) -> [{"title", "url", "text"?, "snippet"?}] and fetch(url) -> (title, text) can
        be injected (tests); by default they call the provider's API and fetch_page."""
        if provider not in ("brave", "tavily"):
            raise ValueError("provider: brave or tavily")
        self.provider, self.key, self.k, self.pages = provider, api_key, k, pages
        self.passage_words, self.rerank, self.rerank_pool = passage_words, rerank, rerank_pool
        self.timeout, self.budget_s = timeout, budget_s
        self.search = search or self._search
        self.fetch = fetch or (lambda url: fetch_page(url, timeout))
        self.cache, self.cache_size, self.errors = {}, cache_size, Counter()
        self._lock = threading.Lock()

    def _json(self, req):
        with urllib.request.urlopen(req, timeout=self.timeout) as r:   # nosec B310: fixed https API endpoints
            return json.loads(r.read().decode())

    def _search(self, query, n):
        if self.provider == "brave":
            req = urllib.request.Request(self.BRAVE.format(q=urllib.parse.quote(query), n=n),
                                         headers={"Accept": "application/json", "X-Subscription-Token": self.key,
                                                  "User-Agent": UA})
            res = (self._json(req).get("web") or {}).get("results") or []
            return [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "snippet": " ".join([r.get("description") or ""] + list(r.get("extra_snippets") or []))}
                    for r in res]
        body = json.dumps({"query": query, "max_results": n, "include_raw_content": True}).encode()
        req = urllib.request.Request(self.TAVILY, body, {"Content-Type": "application/json", "User-Agent": UA,
                                                         "Authorization": "Bearer " + self.key})
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "text": r.get("raw_content") or "",
                 "snippet": r.get("content") or ""} for r in self._json(req).get("results") or []]

    @staticmethod
    def _why(e):
        code = getattr(e, "code", None)
        return f"HTTP {code}" if code else type(e).__name__

    def _page(self, hit, t0):
        """(title, text) of one hit: its own text (Tavily), else the fetched page, else its snippet."""
        if hit.get("text"):
            return hit.get("title", ""), hit["text"]
        if time.monotonic() - t0 < self.budget_s and hit.get("url"):
            try:
                title, text = self.fetch(hit["url"])
                if text:
                    return title or hit.get("title", ""), text
            except Exception as e:
                with self._lock:
                    self.errors["page: " + self._why(e)] += 1
        return hit.get("title", ""), hit.get("snippet", "")

    def __call__(self, query, k=None):
        k = k or self.k
        key = (" ".join(query.lower().split()), k)
        if key in self.cache:
            return self.cache[key]
        t0 = time.monotonic()
        try:
            hits = [h for h in self.search(query, self.pages) if h.get("url") or h.get("text") or h.get("snippet")]
        except Exception as e:                               # bad key, quota, network: no lookup
            with self._lock:
                self.errors["search: " + self._why(e)] += 1
            return []
        hits = hits[:self.pages]
        with ThreadPoolExecutor(max_workers=max(1, len(hits))) as pool:  # pages wait on the network: in parallel
            pages = list(pool.map(lambda h: self._page(h, t0), hits))
        cands = []
        for j, (h, (title, text)) in enumerate(zip(hits, pages)):
            site = urllib.parse.urlsplit(h.get("url", "")).hostname or "web"
            site = site[4:] if site.startswith("www.") else site
            words = text.split()[:30000]
            if words:
                cands += [(title or site, j, i, c, site)
                          for i, c in enumerate(chunks(" ".join(words), self.passage_words, self.passage_words // 2))]
        out = rank_passages(query, cands, k, self.rerank, self.rerank_pool, errors=self.errors)
        with self._lock:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = out
        return out


class Union:
    """Several retrievers at once (in parallel); their passages pooled and the best k kept: reranked when
    a reranker is given, else taken in turn from each source."""

    def __init__(self, *retrievers, k=3, rerank=None):
        self.retrievers, self.k, self.rerank = retrievers, k, rerank
        self.errors = Counter()

    def __call__(self, query, k=None):
        k = k or self.k
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as pool:
            found = list(pool.map(lambda r: r(query), self.retrievers))
        for r in self.retrievers:                            # one view of every source's failures
            for name, n in (getattr(r, "errors", None) or {}).items():
                self.errors[f"{type(r).__name__}: {name}"] = n
        seen, merged = set(), []
        for i in range(max((len(f) for f in found), default=0)):   # in turn: no source crowds the others out
            for f in found:
                if i < len(f) and f[i] not in seen:
                    seen.add(f[i])
                    merged.append(f[i])
        if self.rerank is not None and len(merged) > k:
            try:
                r = self.rerank(query, merged)
                r = [float(x) for x in (r.reshape(-1) if hasattr(r, "reshape") else r)]
                merged = [merged[j] for j in sorted(range(len(merged)), key=lambda j: -r[j])]
            except Exception as e:
                self.errors["rerank: " + type(e).__name__] += 1
        return merged[:k]
