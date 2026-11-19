"""Console single sign-on (OIDC authorization code + PKCE) against an in-process fake provider.

The fake provider is an ``httpx.MockTransport`` serving discovery, JWKS and the token endpoint; the
test plays the browser (follows the gateway's redirect to the authorization endpoint by asking the
fake provider for a code). Keys are generated with ``cryptography``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from pydantic import ValidationError

from pgw.console.oidc import OIDCError, PendingFlows, b64url_decode, jwk_to_key, parse_jwks, verify_id_token
from pgw.core.errors import ConfigError
from pgw.gateway import ApiKeyStore, create_app
from tests.console.conftest import LOCAL, client, csrf
from tests.gateway.conftest import ADMIN, ADMIN_TOKEN, plain_settings

ISSUER = "https://idp.example.com/realms/pgw"
CLIENT_ID = "pgw-console"
SECRET = "s3cret-client-value"  # noqa: S105 - test fixture
REDIRECT = f"{LOCAL}/console/api/oidc/callback"


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _int_b64(n: int) -> str:
    return b64u(n.to_bytes((n.bit_length() + 7) // 8, "big"))


@dataclass
class Key:
    kid: str
    alg: str
    private: Any

    @classmethod
    def rsa(cls, kid: str, bits: int = 2048) -> Key:
        return cls(kid, "RS256", rsa.generate_private_key(public_exponent=65537, key_size=bits))

    @classmethod
    def ec(cls, kid: str) -> Key:
        return cls(kid, "ES256", ec.generate_private_key(ec.SECP256R1()))

    def jwk(self) -> dict[str, Any]:
        pub = self.private.public_key().public_numbers()
        if self.alg == "RS256":
            return {"kty": "RSA", "kid": self.kid, "use": "sig", "alg": "RS256", "n": _int_b64(pub.n),
                    "e": _int_b64(pub.e)}
        return {"kty": "EC", "kid": self.kid, "use": "sig", "crv": "P-256",
                "x": b64u(pub.x.to_bytes(32, "big")), "y": b64u(pub.y.to_bytes(32, "big"))}

    def sign(self, data: bytes) -> bytes:
        if self.alg == "RS256":
            return bytes(self.private.sign(data, padding.PKCS1v15(), hashes.SHA256()))
        r, s = decode_dss_signature(self.private.sign(data, ec.ECDSA(hashes.SHA256())))
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def make_jwt(claims: dict[str, Any], key: Key | None, *, header: dict[str, Any] | None = None,
             signer: Callable[[bytes], bytes] | None = None) -> str:
    h = header if header is not None else {"alg": key.alg if key else "none", "kid": key.kid if key else None,
                                            "typ": "JWT"}
    h = {k: v for k, v in h.items() if v is not None}
    signing_input = f"{b64u(json.dumps(h).encode())}.{b64u(json.dumps(claims).encode())}".encode()
    sig = signer(signing_input) if signer else key.sign(signing_input) if key else b""
    return f"{signing_input.decode()}.{b64u(sig)}"


@dataclass
class FakeIdP:
    """Discovery + JWKS + token endpoint; hands out codes for authorization requests."""

    keys: list[Key] = field(default_factory=lambda: [Key.rsa("k1")])
    signing_kid: str = "k1"
    #: Overrides/extra claims of the next ID tokens (``None`` value = drop the claim).
    claims: dict[str, Any] = field(default_factory=dict)
    #: Replace the whole ID token (called with the default claims and the signing key).
    forge: Callable[[dict[str, Any], Key], str] | None = None
    user: dict[str, Any] = field(default_factory=lambda: {
        "sub": "user-123", "email": "alice@corp.example.com", "name": "Alice Admin", "groups": ["pgw-admins"]})
    codes: dict[str, dict[str, Any]] = field(default_factory=dict)
    jwks_fetches: int = 0
    discovery_fetches: int = 0
    token_requests: list[dict[str, Any]] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{ISSUER}/.well-known/openid-configuration":
            self.discovery_fetches += 1
            return httpx.Response(200, json={
                "issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/protocol/openid-connect/auth",
                "token_endpoint": f"{ISSUER}/protocol/openid-connect/token",
                "jwks_uri": f"{ISSUER}/protocol/openid-connect/certs",
                "code_challenge_methods_supported": ["plain", "S256"],
                "id_token_signing_alg_values_supported": ["RS256", "ES256"],
                "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
            })
        if url == f"{ISSUER}/protocol/openid-connect/certs":
            self.jwks_fetches += 1
            return httpx.Response(200, json={"keys": [k.jwk() for k in self.keys]})
        if url == f"{ISSUER}/protocol/openid-connect/token" and request.method == "POST":
            return self._token(request)
        return httpx.Response(404, json={"error": "not_found"})

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode(), keep_blank_values=True).items()}
        self.token_requests.append(form)
        expected = "Basic " + base64.b64encode(f"{CLIENT_ID}:{SECRET}".encode()).decode()
        if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
            return httpx.Response(401, json={"error": "invalid_client"})
        grant = self.codes.pop(form.get("code", ""), None)  # codes are single use
        if grant is None or form.get("grant_type") != "authorization_code" or form.get("redirect_uri") != REDIRECT:
            return httpx.Response(400, json={"error": "invalid_grant"})
        verifier = form.get("code_verifier", "")
        if not verifier or b64u(hashlib.sha256(verifier.encode()).digest()) != grant["code_challenge"]:
            return httpx.Response(400, json={"error": "invalid_grant", "error_description": "PKCE failed"})
        now = int(time.time())
        claims = {"iss": ISSUER, "aud": CLIENT_ID, "azp": CLIENT_ID, "iat": now, "exp": now + 300,
                  "nonce": grant["nonce"], **self.user}
        for k, v in self.claims.items():
            if v is None:
                claims.pop(k, None)
            else:
                claims[k] = v
        key = next(k for k in self.keys if k.kid == self.signing_kid)
        tok = self.forge(claims, key) if self.forge else make_jwt(claims, key)
        return httpx.Response(200, json={"access_token": "at-" + secrets.token_hex(8), "token_type": "Bearer",
                                         "expires_in": 300, "id_token": tok})

    def authorize(self, location: str) -> dict[str, str]:
        """What the provider does at its authorization endpoint: returns the callback parameters."""
        parts = urlsplit(location)
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == f"{ISSUER}/protocol/openid-connect/auth"
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        assert q["response_type"] == "code" and q["client_id"] == CLIENT_ID and q["redirect_uri"] == REDIRECT
        assert q["code_challenge_method"] == "S256" and len(q["code_challenge"]) == 43
        assert q["scope"].split() == ["openid", "email", "profile"]
        assert len(q["state"]) >= 43 and len(q["nonce"]) >= 43
        code = secrets.token_urlsafe(16)
        self.codes[code] = {"code_challenge": q["code_challenge"], "nonce": q["nonce"]}
        return {"code": code, "state": q["state"]}


def sso_app(tiny_pack_dir: Path, tmp_path: Path, idp: FakeIdP, **kw: Any) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "oidc-secret"
    secret.write_text(SECRET + "\n", encoding="utf-8")
    base: dict[str, Any] = {
        "console_oidc_issuer": ISSUER, "console_oidc_client_id": CLIENT_ID,
        "console_oidc_client_secret_file": secret, "console_oidc_redirect_url": REDIRECT,
        "console_oidc_admin_values": ["pgw-admins"], "console_oidc_viewer_values": ["pgw-viewers"],
        "console_oidc_label": "Keycloak",
    }
    base.update(kw)
    return create_app(plain_settings(tiny_pack_dir, tmp_path, **base), key_store=ApiKeyStore(None),
                      console_static_dir=tmp_path / "no-static", oidc_transport=httpx.MockTransport(idp.handler))


async def start(c: httpx.AsyncClient) -> str:
    r = await c.get("/console/api/oidc/login")
    assert r.status_code == 302, r.text
    assert "pgw_oidc=" in r.headers["set-cookie"] and "samesite=lax" in r.headers["set-cookie"].lower()
    return r.headers["location"]


async def sso_login(c: httpx.AsyncClient, idp: FakeIdP) -> httpx.Response:
    params = idp.authorize(await start(c))
    return await c.get("/console/api/oidc/callback", params=params)


def _audit(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in (tmp_path / "audit.jsonl").read_text().splitlines() if ln.strip()]


# ------------------------------------------------------------------------------------------ happy paths


async def test_sso_admin_happy_path(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    async with client(sso_app(tiny_pack_dir, tmp_path, idp)) as c:
        methods = (await c.get("/console/api/auth-methods")).json()
        assert methods == {"token": True, "oidc": {"enabled": True, "label": "Keycloak"}}
        r = await sso_login(c, idp)
        assert r.status_code == 302 and r.headers["location"] == "/console/"
        cookies = r.headers.get_list("set-cookie")
        sess = next(h for h in cookies if h.startswith("pgw_console="))
        assert "httponly" in sess.lower() and "samesite=strict" in sess.lower()
        assert any(h.startswith("pgw_oidc=") and "max-age=0" in h.lower() for h in cookies)  # binding cleared
        me = (await c.get("/console/api/me")).json()
        assert me["role"] == "admin" and me["auth"] == "oidc"
        assert me["user"] == {"sub": "user-123", "email": "alice@corp.example.com", "name": "Alice Admin"}
        # the session behaves like a token session: CSRF needed for writes, admin may write
        assert (await c.post("/admin/keys", json={"name": "x"})).status_code == 403
        r = await c.post("/admin/keys", json={"name": "sso-made"}, headers=csrf(me["csrf"]))
        assert r.status_code == 201, r.text
        # PKCE: the token request carried the verifier the provider checked against the challenge
        assert idp.token_requests[-1]["code_verifier"] and "client_secret" not in idp.token_requests[-1]
        assert idp.jwks_fetches == 1 and idp.discovery_fetches == 1
    text = (tmp_path / "audit.jsonl").read_text()
    login = next(e for e in _audit(tmp_path) if e["event"] == "console.login")
    assert login["method"] == "oidc" and login["subject"] == "user-123"
    assert login["email"] == "alice@corp.example.com" and login["reason"] == "admin"
    assert login["key_id"].startswith("console-admin:")
    assert "id_token" not in text and "at-" not in text and SECRET not in text
    assert next(e for e in _audit(tmp_path) if e["event"] == "admin.key_created")["key_id"] == login["key_id"]


async def test_sso_viewer_and_es256(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP(keys=[Key.ec("ec1")], signing_kid="ec1")
    idp.user["groups"] = ["staff", "pgw-viewers"]
    async with client(sso_app(tiny_pack_dir, tmp_path, idp)) as c:
        assert (await sso_login(c, idp)).status_code == 302
        me = (await c.get("/console/api/me")).json()
        assert me["role"] == "viewer" and me["auth"] == "oidc"
        assert (await c.get("/admin/status")).status_code == 200
        r = await c.post("/admin/keys", json={"name": "x"}, headers=csrf(me["csrf"]))
        assert r.status_code == 403 and "read-only" in r.json()["error"]["message"]


async def test_token_login_still_works_and_reports_method(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    async with client(sso_app(tiny_pack_dir, tmp_path, idp)) as c:
        assert (await c.post("/console/api/login", json={"token": ADMIN_TOKEN})).status_code == 200
        me = (await c.get("/console/api/me")).json()
        assert me["auth"] == "token" and me["user"] is None
    login = next(e for e in _audit(tmp_path) if e["event"] == "console.login")
    assert login["method"] == "token"


async def test_nested_role_claim(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    idp.user = {"sub": "u2", "email": "bob@corp.example.com", "realm_access": {"roles": ["console-admin"]}}
    app = sso_app(tiny_pack_dir, tmp_path, idp, console_oidc_role_claim="realm_access.roles",
                  console_oidc_admin_values=["console-admin"])
    async with client(app) as c:
        assert (await sso_login(c, idp)).status_code == 302
        me = (await c.get("/console/api/me")).json()
        assert me["role"] == "admin" and me["user"]["name"] is None


# ------------------------------------------------------------------------------------------ refusals


async def _refused(tiny_pack_dir: Path, tmp_path: Path, idp: FakeIdP, status: int, reason: str,
                   **kw: Any) -> None:
    async with client(sso_app(tiny_pack_dir, tmp_path, idp, **kw)) as c:
        r = await sso_login(c, idp)
        assert r.status_code == status, r.text
        assert r.headers["content-type"].startswith("text/html") and "Back to sign-in" in r.text
        assert not any(h.startswith("pgw_console=") for h in r.headers.get_list("set-cookie"))
        assert (await c.get("/console/api/me")).status_code == 401
    failed = [e for e in _audit(tmp_path) if e["event"] == "console.login_failed"]
    assert failed and failed[-1]["reason"] == reason and failed[-1]["method"] == "oidc", failed
    assert not [e for e in _audit(tmp_path) if e["event"] == "console.login"]


async def test_unknown_group_is_forbidden(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    idp.user["groups"] = ["staff"]
    await _refused(tiny_pack_dir, tmp_path, idp, 403, "oidc_no_role")
    failed = next(e for e in _audit(tmp_path) if e["event"] == "console.login_failed")
    assert failed["subject"] == "user-123" and failed["code"] == "permission_denied"


async def test_missing_role_claim_is_forbidden(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP(claims={"groups": None})
    await _refused(tiny_pack_dir, tmp_path, idp, 403, "oidc_no_role")


@pytest.mark.parametrize("claims", [{"aud": "other-client"}, {"aud": ["other", "another"]},
                                    {"aud": [CLIENT_ID, "other"], "azp": None}, {"azp": "other-client"}])
async def test_wrong_audience(tiny_pack_dir: Path, tmp_path: Path, claims: dict[str, Any]) -> None:
    await _refused(tiny_pack_dir, tmp_path, FakeIdP(claims=claims), 401, "oidc_bad_audience")


async def test_wrong_issuer(tiny_pack_dir: Path, tmp_path: Path) -> None:
    await _refused(tiny_pack_dir, tmp_path, FakeIdP(claims={"iss": "https://evil.example.com"}), 401,
                   "oidc_bad_issuer")


async def test_expired_token(tiny_pack_dir: Path, tmp_path: Path) -> None:
    now = int(time.time())
    await _refused(tiny_pack_dir, tmp_path, FakeIdP(claims={"iat": now - 400, "exp": now - 120}), 401,
                   "oidc_expired")


async def test_token_from_the_future(tiny_pack_dir: Path, tmp_path: Path) -> None:
    now = int(time.time())
    await _refused(tiny_pack_dir, tmp_path, FakeIdP(claims={"iat": now + 600, "exp": now + 900}), 401,
                   "oidc_expired")


async def test_bad_signature(tiny_pack_dir: Path, tmp_path: Path) -> None:
    attacker = Key.rsa("k1")  # same kid as the provider's key, different key material

    def forge(claims: dict[str, Any], key: Key) -> str:
        return make_jwt(claims, attacker)

    await _refused(tiny_pack_dir, tmp_path, FakeIdP(forge=forge), 401, "oidc_bad_signature")


async def test_tampered_claims(tiny_pack_dir: Path, tmp_path: Path) -> None:
    def forge(claims: dict[str, Any], key: Key) -> str:
        h, _, s = make_jwt(claims, key).split(".")
        return ".".join([h, b64u(json.dumps({**claims, "groups": ["pgw-admins", "x"]}).encode()), s])

    await _refused(tiny_pack_dir, tmp_path, FakeIdP(forge=forge), 401, "oidc_bad_signature")


@pytest.mark.parametrize("alg", ["none", "HS256", "PS256", "RS512"])
async def test_disallowed_algorithms(tiny_pack_dir: Path, tmp_path: Path, alg: str) -> None:
    def forge(claims: dict[str, Any], key: Key) -> str:
        header = {"alg": alg, "kid": key.kid, "typ": "JWT"}
        if alg == "none":
            return make_jwt(claims, None, header=header)
        # classic confusion attack: HMAC keyed with the provider's public key
        pem = key.private.public_key().public_bytes(serialization.Encoding.PEM,
                                                    serialization.PublicFormat.SubjectPublicKeyInfo)
        return make_jwt(claims, None, header=header, signer=lambda d: hmac.new(pem, d, hashlib.sha256).digest())

    await _refused(tiny_pack_dir, tmp_path, FakeIdP(forge=forge), 401, "oidc_bad_alg")


async def test_nonce_mismatch(tiny_pack_dir: Path, tmp_path: Path) -> None:
    await _refused(tiny_pack_dir, tmp_path, FakeIdP(claims={"nonce": "attacker-nonce"}), 401, "oidc_bad_nonce")


async def test_missing_nonce(tiny_pack_dir: Path, tmp_path: Path) -> None:
    await _refused(tiny_pack_dir, tmp_path, FakeIdP(claims={"nonce": None}), 401, "oidc_bad_nonce")


async def test_replayed_state_is_refused(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    async with client(sso_app(tiny_pack_dir, tmp_path, idp)) as c:
        location = await start(c)
        params = idp.authorize(location)
        assert (await c.get("/console/api/oidc/callback", params=params)).status_code == 302
        # same state again (and a fresh code for it): the pending sign-in was consumed
        params2 = idp.authorize(location)
        await start(c)  # a new binding cookie does not revive the old state
        r = await c.get("/console/api/oidc/callback", params=params2)
        assert r.status_code == 400 and "already used" in r.text
        # unknown state
        r = await c.get("/console/api/oidc/callback", params={"code": "x", "state": "nope"})
        assert r.status_code == 400
    assert [e["reason"] for e in _audit(tmp_path) if e["event"] == "console.login_failed"] == [
        "oidc_bad_state", "oidc_bad_state"]


async def test_state_bound_to_the_starting_browser(tiny_pack_dir: Path, tmp_path: Path) -> None:
    """Login CSRF: a callback URL started in the attacker's browser fails in the victim's."""
    idp = FakeIdP()
    async with client(sso_app(tiny_pack_dir, tmp_path, idp)) as c:
        params = idp.authorize(await start(c))
        c.cookies.delete("pgw_oidc")
        r = await c.get("/console/api/oidc/callback", params=params)
        assert r.status_code == 400 and "another browser" in r.text
        # and the state is now spent
        r = await c.get("/console/api/oidc/callback", params=params)
        assert r.status_code == 400


async def test_expired_pending_state(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    app = sso_app(tiny_pack_dir, tmp_path, idp)
    async with client(app) as c:
        params = idp.authorize(await start(c))
        app.state.pgw.oidc.pending.ttl_s = 0.0
        assert (await c.get("/console/api/oidc/callback", params=params)).status_code == 400


async def test_missing_pkce_verifier(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    app = sso_app(tiny_pack_dir, tmp_path, idp)
    async with client(app) as c:
        params = idp.authorize(await start(c))
        for p in app.state.pgw.oidc.pending._items.values():
            p.verifier = ""  # the provider must refuse a code redeemed without its verifier
        r = await c.get("/console/api/oidc/callback", params=params)
        assert r.status_code == 401 and "invalid_grant" in r.text
        assert (await c.get("/console/api/me")).status_code == 401
    assert idp.token_requests[-1]["code_verifier"] == ""


async def test_wrong_pkce_verifier(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    app = sso_app(tiny_pack_dir, tmp_path, idp)
    async with client(app) as c:
        params = idp.authorize(await start(c))
        for p in app.state.pgw.oidc.pending._items.values():
            p.verifier = "x" * 64
        assert (await c.get("/console/api/oidc/callback", params=params)).status_code == 401


async def test_provider_error_and_issuer_mixup(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    async with client(sso_app(tiny_pack_dir, tmp_path, idp)) as c:
        params = idp.authorize(await start(c))
        r = await c.get("/console/api/oidc/callback",
                        params={"state": params["state"], "error": "access_denied<script>"})
        assert r.status_code == 401 and "access_denied" in r.text and "<script>" not in r.text
        params = idp.authorize(await start(c))
        r = await c.get("/console/api/oidc/callback", params={**params, "iss": "https://evil.example.com"})
        assert r.status_code == 401
    assert [e["reason"] for e in _audit(tmp_path) if e["event"] == "console.login_failed"] == [
        "oidc_provider_error", "oidc_bad_issuer"]


async def test_jwks_key_rotation(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    app = sso_app(tiny_pack_dir, tmp_path, idp)
    async with client(app) as c:
        assert (await sso_login(c, idp)).status_code == 302
        assert idp.jwks_fetches == 1
        assert (await sso_login(c, idp)).status_code == 302
        assert idp.jwks_fetches == 1  # cached
        # the provider rotates to a new key (old one removed); the gateway refetches on the unknown kid
        oidc = app.state.pgw.oidc
        oidc._jwks_at -= oidc.jwks_min_refresh_s  # the cached JWKS is older than the refresh rate limit
        idp.keys = [Key.ec("k2")]
        idp.signing_kid = "k2"
        r = await sso_login(c, idp)
        assert r.status_code == 302, r.text
        assert idp.jwks_fetches == 2
        # a token with a kid the provider does not publish: refetch is rate limited, then refused
        rogue = Key.rsa("k3")
        idp.forge = lambda claims, key: make_jwt(claims, rogue)
        r = await sso_login(c, idp)
        assert r.status_code == 401
        assert idp.jwks_fetches == 2  # within jwks_min_refresh_s of the last fetch
        oidc.jwks_min_refresh_s = 0.0
        r = await sso_login(c, idp)
        assert r.status_code == 401 and idp.jwks_fetches == 3
    reasons = [e["reason"] for e in _audit(tmp_path) if e["event"] == "console.login_failed"]
    assert reasons == ["oidc_unknown_key", "oidc_unknown_key"]


async def test_email_domain_allowlist(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    app = sso_app(tiny_pack_dir, tmp_path, idp, console_oidc_allowed_domains=["@Corp.Example.com"])
    async with client(app) as c:
        assert (await sso_login(c, idp)).status_code == 302
        for user in ({"email": "mallory@evil.example.com"}, {"email": None},
                     {"email": "eve@corp.example.com", "email_verified": False},
                     {"email": "eve@sub.corp.example.com"}):
            idp.claims = user
            r = await sso_login(c, idp)
            assert r.status_code == 403 and "domain" in r.text, user
        idp.claims = {"email": "carol@CORP.example.com", "email_verified": True}
        assert (await sso_login(c, idp)).status_code == 302
    reasons = [e["reason"] for e in _audit(tmp_path) if e["event"] == "console.login_failed"]
    assert reasons == ["oidc_domain_not_allowed"] * 4


async def test_token_login_disabled(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    async with client(sso_app(tiny_pack_dir, tmp_path, idp, console_token_login_enabled=False)) as c:
        assert (await c.get("/console/api/auth-methods")).json() == {
            "token": False, "oidc": {"enabled": True, "label": "Keycloak"}}
        r = await c.post("/console/api/login", json={"token": ADMIN_TOKEN})
        assert r.status_code == 403 and "SSO" in r.json()["error"]["message"]
        assert "set-cookie" not in r.headers
        assert (await c.get("/admin/status", headers=ADMIN)).status_code == 200  # Bearer scripts keep working
        assert (await sso_login(c, idp)).status_code == 302
        assert (await c.get("/console/api/me")).json()["auth"] == "oidc"


async def test_sso_without_admin_token(tiny_pack_dir: Path, tmp_path: Path) -> None:
    idp = FakeIdP()
    async with client(sso_app(tiny_pack_dir, tmp_path, idp, admin_token=None)) as c:
        assert (await c.get("/console/api/auth-methods")).json()["token"] is False
        assert (await c.post("/console/api/login", json={"token": ADMIN_TOKEN})).status_code == 503
        assert (await c.get("/console/api/me")).status_code == 401
        assert (await sso_login(c, idp)).status_code == 302
        assert (await c.get("/console/api/me")).json()["role"] == "admin"


async def test_sso_disabled_by_default(tiny_pack_dir: Path, tmp_path: Path) -> None:
    from tests.console.conftest import make_console_app

    async with client(make_console_app(tiny_pack_dir, tmp_path)) as c:
        assert (await c.get("/console/api/auth-methods")).json() == {
            "token": True, "oidc": {"enabled": False, "label": "SSO"}}
        assert (await c.get("/console/api/oidc/login")).status_code == 404
        assert (await c.get("/console/api/oidc/callback?code=x&state=y")).status_code == 404


async def test_provider_unreachable_or_misconfigured(tiny_pack_dir: Path, tmp_path: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    app = sso_app(tiny_pack_dir, tmp_path, FakeIdP())
    app.state.pgw.oidc._transport = httpx.MockTransport(down)
    async with client(app) as c:
        r = await c.get("/console/api/oidc/login")
        assert r.status_code == 503 and "could not be reached" in r.text

    idp = FakeIdP()
    orig = idp.handler

    def wrong_issuer(request: httpx.Request) -> httpx.Response:
        r = orig(request)
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(200, json={**r.json(), "issuer": "https://other.example.com"})
        return r

    app = sso_app(tiny_pack_dir, tmp_path / "b", idp)
    app.state.pgw.oidc._transport = httpx.MockTransport(wrong_issuer)
    async with client(app) as c:
        r = await c.get("/console/api/oidc/login")
        assert r.status_code == 503 and "another issuer" in r.text


# ------------------------------------------------------------------------------------------ settings


def _settings(tmp_path: Path, **kw: Any) -> Any:
    from pgw.gateway import GatewaySettings

    secret = tmp_path / "s"
    secret.write_text(SECRET)
    base: dict[str, Any] = {"console_oidc_issuer": ISSUER, "console_oidc_client_id": CLIENT_ID,
                            "console_oidc_client_secret_file": secret, "console_oidc_redirect_url": REDIRECT,
                            "console_oidc_admin_values": ["a"]}
    base.update(kw)
    return GatewaySettings(**base)


@pytest.mark.parametrize("kw, msg", [
    ({"console_oidc_issuer": "http://idp.example.com"}, "https"),
    ({"console_oidc_redirect_url": "https://gw.example.com/callback"}, "/console/api/oidc/callback"),
    ({"console_oidc_redirect_url": None}, "required"),
    ({"console_oidc_client_id": None}, "together"),
    ({"console_oidc_client_secret_file": None}, "secret_file"),
    ({"console_oidc_admin_values": []}, "admin_values"),
    ({"console_oidc_scopes": "email profile"}, "openid"),
    ({"console_oidc_allowed_domains": ["a b"]}, "domain"),
])
def test_settings_validation(tmp_path: Path, kw: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        _settings(tmp_path, **kw)


def test_settings_defaults_and_token_login_needs_sso(tmp_path: Path) -> None:
    from pgw.gateway import GatewaySettings

    s = GatewaySettings()
    assert not s.console_oidc_enabled and s.console_token_login_enabled
    assert s.console_oidc_scopes == "openid email profile" and s.console_oidc_role_claim == "groups"
    with pytest.raises(ValidationError, match="needs console SSO"):
        GatewaySettings(console_token_login_enabled=False)
    assert _settings(tmp_path, console_oidc_issuer="http://127.0.0.1:9999/realm").console_oidc_enabled


def test_unreadable_secret_file_fails_at_startup(tiny_pack_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="client_secret_file"):
        sso_app(tiny_pack_dir, tmp_path, FakeIdP(), console_oidc_client_secret_file=tmp_path / "missing")


# ------------------------------------------------------------------------------------------ JOSE units


def _claims(**kw: Any) -> dict[str, Any]:
    now = int(time.time())
    base = {"iss": ISSUER, "aud": CLIENT_ID, "sub": "s", "nonce": "n", "iat": now, "exp": now + 60}
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


def test_verify_id_token_units() -> None:
    k = Key.rsa("a")
    keys = parse_jwks({"keys": [k.jwk()]})
    kw = {"issuer": ISSUER, "client_id": CLIENT_ID, "nonce": "n"}
    assert verify_id_token(make_jwt(_claims(), k), keys, **kw)["sub"] == "s"
    # no kid in the header: any key of the right type is tried
    assert verify_id_token(make_jwt(_claims(), k, header={"alg": "RS256"}), keys, **kw)
    # leeway: just expired is fine, well expired is not
    now = time.time()
    assert verify_id_token(make_jwt(_claims(exp=int(now) - 30), k), keys, **kw)
    with pytest.raises(OIDCError, match="expired"):
        verify_id_token(make_jwt(_claims(exp=int(now) - 90), k), keys, **kw)
    with pytest.raises(OIDCError, match="too long ago"):
        verify_id_token(make_jwt(_claims(iat=int(now) - 3600, exp=int(now) + 60), k), keys, **kw)
    for bad, reason in ((_claims(sub=None), "oidc_bad_claims"), (_claims(exp="soon"), "oidc_bad_claims"),
                        (_claims(iat=True), "oidc_bad_claims"), (_claims(nbf=int(now) + 600), "oidc_expired")):
        with pytest.raises(OIDCError) as ei:
            verify_id_token(make_jwt(bad, k), keys, **kw)
        assert ei.value.reason == reason
    for tok in ("a.b", "a.b.c.d.e", "!!.e30.", make_jwt(_claims(), k) + "x" * 20000):
        with pytest.raises(OIDCError) as ei:
            verify_id_token(tok, keys, **kw)
        assert ei.value.reason == "oidc_bad_token"
    with pytest.raises(OIDCError) as ei:  # header parameters that would pull keys from the token itself
        verify_id_token(make_jwt(_claims(), k, header={"alg": "RS256", "kid": "a", "jwk": k.jwk()}), keys, **kw)
    assert ei.value.reason == "oidc_bad_token"
    # an RS256 token cannot be checked against an EC key (alg/key type binding)
    e = Key.ec("a")
    with pytest.raises(OIDCError) as ei:
        verify_id_token(make_jwt(_claims(), k), parse_jwks({"keys": [e.jwk()]}), **kw)
    assert ei.value.reason == "oidc_unknown_key"


def test_jwk_parsing_rejects_weak_or_odd_keys() -> None:
    weak = Key.rsa("w", bits=1024).jwk()
    with pytest.raises(ValueError, match="2048"):
        jwk_to_key(weak)
    good = Key.ec("e").jwk()
    with pytest.raises(ValueError):
        jwk_to_key({**good, "crv": "P-384"})
    with pytest.raises(ValueError):
        jwk_to_key({**good, "y": good["x"]})  # not on the curve
    with pytest.raises(ValueError):
        jwk_to_key({**Key.rsa("r").jwk(), "use": "enc"})
    with pytest.raises(ValueError):
        jwk_to_key({**Key.rsa("r").jwk(), "alg": "HS256"})
    with pytest.raises(ValueError):
        jwk_to_key({"kty": "oct", "k": "c2VjcmV0"})
    keys = parse_jwks({"keys": [weak, {"kty": "oct", "k": "AA"}, good, "junk"]})
    assert [k.kid for k in keys] == ["e"]
    with pytest.raises(ValueError):
        parse_jwks({"nokeys": []})
    with pytest.raises(ValueError):
        b64url_decode("a+b/")
    with pytest.raises(ValueError):
        b64url_decode("abcde")


def test_pending_flows_single_use_and_bounded() -> None:
    t = [0.0]
    p = PendingFlows(ttl_s=10, max_pending=3, clock=lambda: t[0])
    state, _nonce, _verifier, binding = p.create()
    assert p.pop(state, binding).verifier
    with pytest.raises(OIDCError):
        p.pop(state, binding)
    flows = [p.create() for _ in range(5)]
    assert len(p) == 3
    with pytest.raises(OIDCError):  # evicted (oldest first)
        p.pop(flows[0][0], flows[0][3])
    t[0] = 11.0
    with pytest.raises(OIDCError):  # expired
        p.pop(flows[-1][0], flows[-1][3])
