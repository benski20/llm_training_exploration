"""Route tests of the developer platform (pgw.platform) against the in-memory fixture accounts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pgw.platform import PlatformSettings, mount_platform
from pgw.platform.app import COOKIE_NAME, SECURITY_HEADERS
from pgw.platform.client import RequestPage, RequestRecord
from pgw.platform.fixtures import FixtureAccounts

PAGES = ["/platform/", "/platform/quickstart", "/platform/keys", "/platform/keys/new", "/platform/models",
         "/platform/usage", "/platform/usage?by=key", "/platform/usage?by=model&range=90d", "/platform/usage?range=7d",
         "/platform/requests", "/platform/requests?status=error", "/platform/settings",
         "/platform/keys/key_prod01/revoke", "/platform/?range=90d"]


def make(accounts: Any = None, **settings: Any) -> TestClient:
    app = FastAPI()
    mount_platform(app, accounts if accounts is not None else FixtureAccounts(),
                   PlatformSettings(**settings))
    return TestClient(app, base_url="https://pgw.test")


def login(c: TestClient, token: str = "demo") -> TestClient:  # noqa: S107
    r = c.post("/platform/login", data={"token": token}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return c


def csrf(c: TestClient, path: str = "/platform/keys") -> str:
    m = re.search(r'name="csrf" value="([^"]+)"', c.get(path).text)
    assert m
    return m.group(1)


@pytest.fixture
def client() -> TestClient:
    return login(make())


# ---- sign-in -------------------------------------------------------------------------------------------
def test_pages_need_a_session() -> None:
    c = make()
    for p in [*PAGES, "/platform/usage.csv"]:
        r = c.get(p, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/platform/login", p


def test_login_page_and_cookie_flags() -> None:
    c = make()
    r = c.get("/platform/login")
    assert r.status_code == 200 and "Sign in to PGW" in r.text and 'name="token"' in r.text
    r = c.post("/platform/login", data={"token": "demo"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/platform/"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE_NAME}=")
    for flag in ("HttpOnly", "Path=/platform", "SameSite=strict", "Secure"):
        assert flag.lower() in cookie.lower()
    assert c.get("/platform/login", follow_redirects=False).status_code == 303  # already signed in


def test_wrong_token_and_rate_limit() -> None:
    c = make()
    for _ in range(10):
        r = c.post("/platform/login", data={"token": "nope"})
        assert r.status_code == 401 and "not recognised" in r.text
    r = c.post("/platform/login", data={"token": "demo"})
    assert r.status_code == 429


def test_no_accounts_shows_not_configured() -> None:
    app = FastAPI()
    mount_platform(app, None, PlatformSettings())
    c = TestClient(app, base_url="https://pgw.test")
    r = c.get("/platform/login")
    assert "not configured" in r.text and 'name="token"' not in r.text
    assert c.post("/platform/login", data={"token": "demo"}).status_code == 403


def test_token_login_can_be_disabled() -> None:
    c = make(token_login=False)
    assert 'name="token"' not in c.get("/platform/login").text
    assert c.post("/platform/login", data={"token": "demo"}).status_code == 403


def test_cross_site_login_refused() -> None:
    c = make()
    r = c.post("/platform/login", data={"token": "demo"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    r = c.post("/platform/login", data={"token": "demo"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_logout(client: TestClient) -> None:
    tok = csrf(client)
    assert client.post("/platform/logout", data={"csrf": "wrong"}).status_code == 403
    r = client.post("/platform/logout", data={"csrf": tok}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/platform/login"
    assert client.get("/platform/", follow_redirects=False).status_code == 303


# ---- pages ---------------------------------------------------------------------------------------------
def test_every_page_renders_with_security_headers(client: TestClient) -> None:
    for p in PAGES:
        r = client.get(p)
        assert r.status_code == 200, (p, r.text[:500])
        assert r.headers["cache-control"] == "no-store"
        for k, v in SECURITY_HEADERS.items():
            assert r.headers[k] == v
        html = r.text
        assert "<script>" not in html and "style=" not in html, p  # strict CSP: nothing inline
        assert not re.search(r'(src|href)="(https?:)?//', html), p  # no external resources
        assert 'class="shell"' in html and "Northwind Labs" in html


def test_overview_content(client: TestClient) -> None:
    html = client.get("/platform/").text
    for s in ("Requests", "Tokens", "Checked results", "Active keys", "Recent requests", "Top keys",
              "No content stored", "Private<span class=\"pill-sep\" aria-hidden=\"true\">·</span>checked client-side"):
        assert s in html


def test_quickstart_has_the_three_lines_with_copy_buttons(client: TestClient) -> None:
    html = client.get("/platform/quickstart").text
    for line in ("pip install pgw", "from pgw import OpenAI", "client = OpenAI(api_key=&quot;pgw_live_…&quot;)"):
        assert line in html
    for block in ("qs-install", "qs-import", "qs-key", "qs-full"):
        assert f'data-copy="#{block}"' in html


def test_models_page(client: TestClient) -> None:
    html = client.get("/platform/models").text
    for m in ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct"):
        assert m in html
    assert "Available" in html


def test_usage_groupings_and_csv(client: TestClient) -> None:
    assert "Production backend" in client.get("/platform/usage?by=key").text
    assert "qwen2.5-7b-instruct" in client.get("/platform/usage?by=model").text
    assert client.get("/platform/usage?by=bogus&range=bogus").status_code == 200
    r = client.get("/platform/usage.csv?range=7d")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("day,key_id,key_name,model,requests")
    assert len(lines) > 7


def test_request_log_shows_privacy_and_timing(client: TestClient) -> None:
    html = client.get("/platform/requests").text
    assert html.count('class="pill pill-ok"') >= 40
    assert "TTFT" in html and "Latency" in html
    m = re.search(r'href="(/platform/requests\?before=[^"]+)"', html)
    assert m, "pager link"
    assert client.get(m.group(1).replace("&amp;", "&")).status_code == 200
    errs = client.get("/platform/requests?status=error").text
    assert "pill-ok" not in errs and "not completed" in errs


# ---- keys ----------------------------------------------------------------------------------------------
def test_create_key_shows_the_secret_once(client: TestClient) -> None:
    tok = csrf(client)
    r = client.post("/platform/keys", data={"csrf": tok, "name": "Data pipeline", "env": "live",
                                            "models": ["qwen2.5-7b-instruct"]})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    m = re.search(r'id="new-key">(pgw_live_[A-Z2-7]+)<', r.text)
    assert m, "secret revealed"
    secret = m.group(1)
    listing = client.get("/platform/keys").text
    assert "Data pipeline" in listing and secret not in listing
    assert secret[-4:] in listing  # display prefix only


def test_create_key_validation(client: TestClient) -> None:
    tok = csrf(client)
    for data in ({"name": ""}, {"name": "x" * 65}, {"name": "ok", "env": "prod"},
                 {"name": "ok", "models": ["gpt-4"]}, {"name": "bad\x07name"}):
        r = client.post("/platform/keys", data={"csrf": tok, **data})
        assert r.status_code == 400, data
        assert 'id="new-key"' not in r.text


def test_mutations_need_csrf(client: TestClient) -> None:
    assert client.post("/platform/keys", data={"name": "x"}).status_code == 403
    assert client.post("/platform/keys/key_prod01/revoke", data={"csrf": "bad"}).status_code == 403
    tok = csrf(client)
    r = client.post("/platform/keys", data={"csrf": tok, "name": "x"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_revoke_key(client: TestClient) -> None:
    tok = csrf(client)
    assert "Revoke key" in client.get("/platform/keys/key_supp02/revoke").text
    r = client.post("/platform/keys/key_supp02/revoke", data={"csrf": tok}, follow_redirects=False)
    assert r.status_code == 303 and "revoked=1" in r.headers["location"]
    html = client.get("/platform/keys?revoked=1").text
    assert "Key revoked." in html
    row = html[html.index("Support assistant") - 200: html.index("Support assistant") + 1500]
    assert "Revoked" in row
    assert client.post("/platform/keys/nope!/revoke", data={"csrf": tok}).status_code == 404
    assert client.post("/platform/keys/key_missing/revoke", data={"csrf": tok}).status_code == 404


def test_viewer_is_read_only() -> None:
    c = login(make(), "demo-viewer")
    html = c.get("/platform/keys").text
    assert "Create key" not in html and ">Revoke<" not in html and "Viewer" in html
    tok = csrf(c)
    assert c.get("/platform/keys/new").status_code == 403
    assert c.post("/platform/keys", data={"csrf": tok, "name": "x"}).status_code == 403
    assert c.post("/platform/keys/key_prod01/revoke", data={"csrf": tok}).status_code == 403


def test_key_names_are_escaped(client: TestClient) -> None:
    tok = csrf(client)
    r = client.post("/platform/keys", data={"csrf": tok, "name": "<script>alert(1)</script>", "env": "test"})
    assert r.status_code == 200
    assert "<script>alert" not in r.text and "&lt;script&gt;alert(1)&lt;/script&gt;" in r.text


# ---- privacy: no content ever reaches a page ---------------------------------------------------------------
@dataclass
class LeakyAccounts(FixtureAccounts):
    """An accounts implementation that (wrongly) returns content in request records."""

    raw: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        FixtureAccounts.__init__(self)

    def requests(self, org_id: str, **kw: Any) -> RequestPage:
        recs = [RequestRecord.parse({"id": "req_1", "ts": self.clock() - 60, "key_id": "key_prod01",
                                     "model": "qwen2.5-7b-instruct", "status": "ok", "verified": True,
                                     "prompt": "TOP-SECRET-PROMPT", "completion": "TOP-SECRET-ANSWER",
                                     "messages": [{"role": "user", "content": "TOP-SECRET-MSG"}],
                                     "error_code": "TOP SECRET MESSAGE", "latency_ms": 12.5})]
        return RequestPage(recs)


def test_request_content_is_never_rendered() -> None:
    c = login(make(LeakyAccounts()))
    for p in ("/platform/requests", "/platform/"):
        html = c.get(p).text
        assert "TOP-SECRET" not in html and "TOP SECRET" not in html
        assert "req_1" in html or p == "/platform/"


def test_request_record_has_no_content_fields() -> None:
    names = set(RequestRecord.__dataclass_fields__)
    assert not names & {"prompt", "completion", "messages", "content", "input", "output", "response"}


def test_org_comes_from_the_session_only() -> None:
    class OtherOrg(FixtureAccounts):
        def authenticate(self, token: str) -> Any:
            p = super().authenticate(token)
            return None if p is None else type(p)("org_other", p.role, p.name, p.email)

    c = login(make(OtherOrg()))
    assert c.get("/platform/keys").status_code == 404  # the fixture only knows its own organisation


# ---- static --------------------------------------------------------------------------------------------
def test_assets() -> None:
    c = make()
    for a, ctype in (("pgw.css", "text/css"), ("platform.css", "text/css"), ("platform.js", "text/javascript"),
                     ("theme.js", "text/javascript"), ("favicon.svg", "image/svg+xml"),
                     ("fonts/inter-latin-wght-normal.woff2", "font/woff2"),
                     ("fonts/geist-mono-latin-wght-normal.woff2", "font/woff2")):
        r = c.get(f"/platform/assets/{a}")
        assert r.status_code == 200 and r.headers["content-type"].startswith(ctype), a
        assert r.headers["cache-control"] == "no-cache"
    assert c.get("/platform/assets/../app.py").status_code == 404
    assert c.get("/platform/assets/nope.css").status_code == 404


def test_design_system_is_self_contained() -> None:
    root = Path(__file__).resolve().parents[2] / "src/pgw"
    css = (root / "design/pgw.css").read_text()
    for u in re.findall(r"url\(([^)]+)\)", css):
        assert not re.match(r"[\"']?(https?:)?//", u), u
        assert (root / "design" / u.strip("\"'")).is_file(), u
    assert "@import" not in css
    for f in (root / "platform/static").iterdir():
        assert "http://" not in f.read_text(errors="ignore").replace("http://www.w3.org", "")


def test_sso_flow() -> None:
    from pgw.console.oidc import OIDCIdentity

    class FakeOIDC:
        label = "Okta"

        class pending:  # noqa: N801
            ttl_s = 600

        async def start(self) -> tuple[str, str]:
            return "https://idp.example/authorize?x=1", "binding123"

        async def finish(self, **kw: Any) -> OIDCIdentity:
            assert kw["binding"] == "binding123"
            return OIDCIdentity(role="admin", sub="u1", email="lee@northwind.example", name="Lee")

    app = FastAPI()
    mount_platform(app, FixtureAccounts(), PlatformSettings(), oidc=FakeOIDC())
    c = TestClient(app, base_url="https://pgw.test")
    assert "Continue with Okta" in c.get("/platform/login").text
    r = c.get("/platform/sso/start", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("https://idp.example/")
    r = c.get("/platform/sso/callback?code=c&state=s", follow_redirects=False)
    assert r.status_code == 303
    html = c.get("/platform/settings").text
    assert "Single sign-on" in html and "lee@northwind.example" in html


class MeteredOnly(FixtureAccounts):
    """What the provider's accounts service meters today: tokens and trips per key, model and day; no request
    log, no request counts or verification (those happen in the client library)."""

    def usage(self, org_id: str, *, since: str, until: str) -> list[Any]:
        from dataclasses import replace

        return [replace(r, requests=None, prompt_tokens=None, completion_tokens=None, verified=None,
                        total_tokens=r.tokens, trips=r.requests * 3) for r in super().usage(org_id, since=since,
                                                                                            until=until)]

    def requests(self, org_id: str, **kw: Any) -> RequestPage:
        from pgw.platform.client import AccountsError

        raise AccountsError("no request log", kind="unsupported")


def test_pages_degrade_without_request_metering() -> None:
    c = login(make(MeteredOnly()))
    for p in PAGES:
        assert c.get(p).status_code == 200, p
    html = c.get("/platform/requests").text
    assert "Request log not available" in html
    ov = c.get("/platform/").text
    assert "Counted by the client library" in ov and "Tokens per day" in ov and "Request log not available" in ov
    us = c.get("/platform/usage").text
    assert "Compute round trips" in us and "Tokens by model" in us
    csv = c.get("/platform/usage.csv").text.splitlines()
    assert csv[1].split(",")[4] == ""  # requests unknown, not zero


def test_settings_shows_the_audit_log(client: TestClient) -> None:
    tok = csrf(client)
    client.post("/platform/keys/key_supp02/revoke", data={"csrf": tok})
    html = client.get("/platform/settings").text
    assert "Audit log" in html and "Key revoked" in html and "Enrollment refused" in html
