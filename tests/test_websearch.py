"""Web search as a retriever (vt/websearch.py): ranking, the fetch's guards, and the union with Wikipedia."""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vt.websearch import Blocked, Union, WebSearch, check_url, fetch_page, html_to_text

PAGE = """<html><head><title>Eiffel Tower - Example</title><script>var x = "ignore me";</script></head>
<body><nav>Home | About</nav><h1>Eiffel Tower</h1><p>The tower is named after the engineer Gustave Eiffel,
whose company designed and built it between 1887 and 1889.</p><footer>cookie banner</footer></body></html>"""


def test_html_to_text_keeps_the_prose_and_drops_scripts_and_navigation():
    title, text = html_to_text(PAGE)
    assert title == "Eiffel Tower - Example" and "Gustave Eiffel" in text
    assert "ignore me" not in text and "Home | About" not in text and "cookie banner" not in text


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1/", "http://localhost:8080/x",
                                 "http://10.0.0.5/", "http://169.254.169.254/latest/meta-data/", "ftp://example.org/",
                                 "http://[::1]/"])
def test_fetch_refuses_private_and_odd_addresses(url):
    with pytest.raises(Blocked):
        check_url(url)


def test_a_redirect_into_the_private_network_is_refused():
    class R(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/secret")
            self.end_headers()

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), R)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        import vt.websearch as ws
        real = ws.check_url
        calls = []

        def first_public(url):                               # the test server itself is local: let it pass once
            calls.append(url)
            if len(calls) == 1:
                return url
            return real(url)
        ws.check_url = first_public
        with pytest.raises(Blocked):
            fetch_page(f"http://127.0.0.1:{srv.server_port}/")
        assert len(calls) == 2 and calls[1].endswith("/secret")
    finally:
        ws.check_url = real
        srv.shutdown()


def test_brave_style_results_are_fetched_ranked_and_sourced():
    hits = [{"title": "Unrelated", "url": "https://www.example.com/a", "snippet": "nothing here"},
            {"title": "Eiffel Tower", "url": "https://tower.example.org/e", "snippet": "a tower"}]
    pages = {"https://tower.example.org/e": html_to_text(PAGE), "https://www.example.com/a": ("A", "Cats sleep a lot.")}
    w = WebSearch("brave", "key", k=1, search=lambda q, n: hits, fetch=lambda u: pages[u])
    got = w("Who designed the Eiffel Tower?")
    assert len(got) == 1 and got[0].startswith("Eiffel Tower - Example (tower.example.org): ") and "Gustave" in got[0]


def test_tavily_style_text_needs_no_fetch_and_a_failing_page_falls_back_to_its_snippet():
    def boom(url):
        raise OSError("down")
    w = WebSearch("tavily", "key", k=2, fetch=boom,
                  search=lambda q, n: [{"title": "T", "url": "https://t.example/", "text": "Thumbelina slept in a walnut shell."},
                                       {"title": "S", "url": "https://s.example/", "snippet": "Thumbelina walnut cradle."}])
    got = w("Thumbelina walnut")
    assert len(got) == 2 and any("walnut shell" in g for g in got) and w.errors["page: OSError"] == 1


def test_a_failing_search_returns_nothing_and_counts_why():
    def bad(q, n):
        raise OSError("quota")
    w = WebSearch("brave", "key", search=bad)
    assert w("anything") == [] and w.errors["search: OSError"] == 1


def test_union_takes_sources_in_turn_and_reranks():
    a = lambda q: ["A1 (Wikipedia): x", "A2 (Wikipedia): y"]             # noqa: E731
    b = lambda q: ["B1 (web.example): z"]                                  # noqa: E731
    assert Union(a, b, k=2)("q") == ["A1 (Wikipedia): x", "B1 (web.example): z"]
    rr = lambda q, ps: [1.0 if "y" in p else 0.0 for p in ps]               # noqa: E731
    assert Union(a, b, k=1, rerank=rr)("q") == ["A2 (Wikipedia): y"]
