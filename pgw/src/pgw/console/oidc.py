"""Console single sign-on: OpenID Connect authorization code flow with PKCE, as a confidential client.

Flow (``docs/operations/console-sso.md``):

1. ``GET /console/api/oidc/login`` creates a pending sign-in (random ``state``, ``nonce`` and PKCE
   ``code_verifier``; kept server-side for :data:`PENDING_TTL_S`, single use) bound to the browser by a
   short-lived ``pgw_oidc`` cookie, and redirects to the provider's authorization endpoint
   (``code_challenge_method=S256``).
2. ``GET /console/api/oidc/callback`` consumes the pending sign-in, exchanges the code at the token
   endpoint (client secret + ``code_verifier``), verifies the ID token (:func:`verify_id_token`) and
   maps its role claim to ``admin`` or ``viewer``.

ID tokens are verified here with :mod:`cryptography` only (no JOSE library dependency): RS256 (RSA
>= 2048 bits) and ES256 (P-256) signatures against the provider's JWKS; every other ``alg``
(``none``, ``HS*``, ``PS*``, ...) is refused. Claims checked: ``iss`` (exact), ``aud`` (contains the
client id), ``azp`` (equals the client id when present, required with several audiences), ``exp``,
``iat`` and ``nbf`` (with :data:`LEEWAY_S`), ``nonce`` (constant-time) and ``sub``. Provider tokens
(code, access token, ID token) are never stored or logged.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import ssl
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from pgw.core.errors import AuthError, ConfigError, ErrorCode, PGWError

__all__ = [
    "ALLOWED_ALGS", "BINDING_COOKIE", "LEEWAY_S", "PENDING_TTL_S", "OIDCClient", "OIDCError", "OIDCIdentity",
    "PendingFlows", "b64url_decode", "jwk_to_key", "pkce_challenge", "verify_id_token",
]

log = logging.getLogger(__name__)

#: Signature algorithms accepted on ID tokens (asymmetric only: never ``none`` or HMAC).
ALLOWED_ALGS: frozenset[str] = frozenset({"RS256", "ES256"})
#: Clock leeway for ``exp``, ``iat`` and ``nbf``.
LEEWAY_S = 60.0
#: Oldest accepted ``iat`` (the ID token is minted at the code exchange, so it is always fresh).
MAX_TOKEN_AGE_S = 600.0
#: Lifetime of a pending sign-in (``state``/``nonce``/PKCE verifier).
PENDING_TTL_S = 600.0
#: Cookie that binds a pending sign-in to the browser that started it.
BINDING_COOKIE = "pgw_oidc"
BINDING_COOKIE_PATH = "/console/api/oidc/"
_MAX_RESPONSE_BYTES = 1 << 20
_MAX_TOKEN_BYTES = 16 << 10
_MIN_RSA_BITS = 2048
_B64URL = re.compile(r"[A-Za-z0-9_-]*")

Role = Literal["admin", "viewer"]


class OIDCError(PGWError):
    """A failed SSO sign-in. ``reason`` is a short code for the audit log (never token material)."""

    def __init__(self, message: str, reason: str, *, code: ErrorCode = ErrorCode.UNAUTHENTICATED,
                 subject: str | None = None, email: str | None = None):
        super().__init__(message, code=code)
        self.reason = reason
        self.subject = subject
        self.email = email


def _fail(message: str, reason: str) -> OIDCError:
    return OIDCError(message, reason)


# =================================================================================================
# JOSE helpers (RS256 / ES256 only)
# =================================================================================================
def b64url_decode(s: str) -> bytes:
    """Strict unpadded base64url decoding (raises ``ValueError``)."""
    if not isinstance(s, str) or not _B64URL.fullmatch(s) or len(s) % 4 == 1:
        raise ValueError("invalid base64url")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def pkce_challenge(verifier: str) -> str:
    """S256 code challenge of a PKCE ``code_verifier`` (RFC 7636)."""
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def _b64_int(v: Any) -> int:
    if not isinstance(v, str) or not v:
        raise ValueError("missing key parameter")
    return int.from_bytes(b64url_decode(v), "big")


def jwk_to_key(jwk: dict[str, Any]) -> tuple[str, Any]:
    """``(alg, public key)`` of a signing JWK usable with :data:`ALLOWED_ALGS`.

    Raises:
        ValueError: not an RSA (>= 2048 bits) or P-256 EC signing key, or inconsistent ``alg``/``use``.
    """
    if jwk.get("use") not in (None, "sig"):
        raise ValueError("not a signing key")
    ops = jwk.get("key_ops")
    if ops is not None and (not isinstance(ops, list) or "verify" not in ops):
        raise ValueError("key_ops does not allow verify")
    kty = jwk.get("kty")
    if kty == "RSA":
        alg = "RS256"
        key: Any = rsa.RSAPublicNumbers(_b64_int(jwk.get("e")), _b64_int(jwk.get("n"))).public_key()
        if key.key_size < _MIN_RSA_BITS:
            raise ValueError(f"RSA key shorter than {_MIN_RSA_BITS} bits")
    elif kty == "EC":
        alg = "ES256"
        if jwk.get("crv") != "P-256":
            raise ValueError("unsupported EC curve")
        x, y = b64url_decode(str(jwk.get("x", ""))), b64url_decode(str(jwk.get("y", "")))
        if len(x) != 32 or len(y) != 32:
            raise ValueError("bad P-256 coordinates")
        key = ec.EllipticCurvePublicNumbers(int.from_bytes(x, "big"), int.from_bytes(y, "big"),
                                            ec.SECP256R1()).public_key()  # raises if not on the curve
    else:
        raise ValueError("unsupported key type")
    if jwk.get("alg") not in (None, alg):
        raise ValueError("key alg does not match its type")
    return alg, key


@dataclass(frozen=True)
class _Jwk:
    kid: str | None
    alg: str
    key: Any


def parse_jwks(doc: Any) -> list[_Jwk]:
    """Usable signing keys of a JWKS document (unusable keys are skipped)."""
    keys = doc.get("keys") if isinstance(doc, dict) else None
    if not isinstance(keys, list):
        raise ValueError("JWKS has no 'keys' list")
    out: list[_Jwk] = []
    for k in keys[:100]:
        if not isinstance(k, dict):
            continue
        try:
            alg, key = jwk_to_key(k)
        except (ValueError, TypeError, OverflowError):
            continue
        kid = k.get("kid")
        out.append(_Jwk(kid if isinstance(kid, str) else None, alg, key))
    return out


def _verify_sig(alg: str, key: Any, signing_input: bytes, sig: bytes) -> bool:
    try:
        if alg == "RS256":
            key.verify(sig, signing_input, padding.PKCS1v15(), hashes.SHA256())
        else:  # ES256: JOSE uses the raw 64-byte r || s encoding
            if len(sig) != 64:
                return False
            der = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
            key.verify(der, signing_input, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return False
    return True


def split_jwt(token: str) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    """``(header, claims, signing input, signature)`` of a compact JWS (structure checks only)."""
    if not isinstance(token, str) or len(token) > _MAX_TOKEN_BYTES:
        raise _fail("ID token missing or too large", "oidc_bad_token")
    parts = token.split(".")
    if len(parts) != 3:
        raise _fail("ID token is not a compact JWS", "oidc_bad_token")
    try:
        header = json.loads(b64url_decode(parts[0]))
        claims = json.loads(b64url_decode(parts[1]))
        sig = b64url_decode(parts[2])
    except (ValueError, UnicodeDecodeError) as exc:
        raise _fail("ID token is malformed", "oidc_bad_token") from exc
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise _fail("ID token is malformed", "oidc_bad_token")
    return header, claims, f"{parts[0]}.{parts[1]}".encode("ascii"), sig


def _num(claims: dict[str, Any], name: str, *, required: bool) -> float | None:
    v = claims.get(name)
    if v is None and not required:
        return None
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise _fail(f"ID token claim '{name}' is missing or not a number", "oidc_bad_claims")
    return float(v)


def verify_id_token(token: str, keys: list[_Jwk], *, issuer: str, client_id: str, nonce: str,
                    now: float | None = None, leeway_s: float = LEEWAY_S,
                    max_age_s: float = MAX_TOKEN_AGE_S) -> dict[str, Any]:
    """Verify an ID token's signature and claims; returns the claims.

    Raises:
        OIDCError: with ``reason`` ``oidc_unknown_key`` when no key in ``keys`` has the token's ``kid``
            (the caller may refresh the JWKS and retry), else ``oidc_bad_token``, ``oidc_bad_alg``,
            ``oidc_bad_signature``, ``oidc_bad_issuer``, ``oidc_bad_audience``, ``oidc_expired``,
            ``oidc_bad_nonce`` or ``oidc_bad_claims``.
    """
    header, claims, signing_input, sig = split_jwt(token)
    alg = header.get("alg")
    if alg not in ALLOWED_ALGS:
        raise _fail(f"ID token algorithm {str(alg)[:16]!r} is not allowed (RS256 or ES256 only)", "oidc_bad_alg")
    if "crit" in header or "jku" in header or "jwk" in header or "x5u" in header:
        raise _fail("ID token header uses unsupported parameters", "oidc_bad_token")
    kid = header.get("kid")
    if kid is not None and not isinstance(kid, str):
        raise _fail("ID token kid is not a string", "oidc_bad_token")
    candidates = [k for k in keys if k.alg == alg and (kid is None or k.kid == kid)]
    if not candidates:
        raise _fail("ID token is signed with an unknown key", "oidc_unknown_key")
    if not any(_verify_sig(alg, k.key, signing_input, sig) for k in candidates[:10]):
        raise _fail("ID token signature is invalid", "oidc_bad_signature")

    if claims.get("iss") != issuer:
        raise _fail("ID token issuer does not match", "oidc_bad_issuer")
    aud = claims.get("aud")
    auds = [aud] if isinstance(aud, str) else aud if isinstance(aud, list) else []
    if not all(isinstance(a, str) for a in auds) or client_id not in auds:
        raise _fail("ID token audience does not include this client", "oidc_bad_audience")
    azp = claims.get("azp")
    if azp is not None and azp != client_id:
        raise _fail("ID token authorized party (azp) is another client", "oidc_bad_audience")
    if len(auds) > 1 and azp is None:
        raise _fail("ID token with several audiences lacks azp", "oidc_bad_audience")
    now = time.time() if now is None else now
    exp = _num(claims, "exp", required=True)
    iat = _num(claims, "iat", required=True)
    nbf = _num(claims, "nbf", required=False)
    assert exp is not None and iat is not None
    if now >= exp + leeway_s:
        raise _fail("ID token has expired", "oidc_expired")
    if iat > now + leeway_s or (nbf is not None and nbf > now + leeway_s):
        raise _fail("ID token is not valid yet (check the gateway clock)", "oidc_expired")
    if iat < now - max_age_s - leeway_s:
        raise _fail("ID token was issued too long ago", "oidc_expired")
    tok_nonce = claims.get("nonce")
    if not isinstance(tok_nonce, str) or not hmac.compare_digest(tok_nonce.encode(), nonce.encode()):
        raise _fail("ID token nonce does not match this sign-in", "oidc_bad_nonce")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub or len(sub) > 255:
        raise _fail("ID token has no valid subject", "oidc_bad_claims")
    return claims


# =================================================================================================
# pending sign-ins
# =================================================================================================
@dataclass
class _Pending:
    verifier: str
    nonce: str
    binding: bytes
    created: float


class PendingFlows:
    """Server-side store of started sign-ins, keyed by the SHA-256 of ``state`` (single use, short TTL)."""

    def __init__(self, ttl_s: float = PENDING_TTL_S, max_pending: int = 1000,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self.max_pending = max_pending
        self.clock = clock
        self._items: OrderedDict[bytes, _Pending] = OrderedDict()
        self._lock = threading.Lock()

    def create(self) -> tuple[str, str, str, str]:
        """New pending sign-in: ``(state, nonce, code_verifier, browser binding)``."""
        state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        verifier, binding = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
        now = self.clock()
        with self._lock:
            for k in [k for k, p in self._items.items() if now - p.created >= self.ttl_s]:
                del self._items[k]
            self._items[hashlib.sha256(state.encode()).digest()] = _Pending(
                verifier, nonce, hashlib.sha256(binding.encode()).digest(), now)
            while len(self._items) > self.max_pending:
                self._items.popitem(last=False)
        return state, nonce, verifier, binding

    def pop(self, state: str, binding: str | None) -> _Pending:
        """Consume the pending sign-in of ``state`` (whatever happens next it cannot be used again).

        Raises:
            OIDCError: unknown, expired or already used ``state`` (``oidc_bad_state``), or the
                request does not come from the browser that started the sign-in (``oidc_bad_binding``).
        """
        if not state or len(state) > 512:
            raise OIDCError("missing state", "oidc_bad_state", code=ErrorCode.PROTOCOL)
        with self._lock:
            p = self._items.pop(hashlib.sha256(state.encode()).digest(), None)
        if p is None or self.clock() - p.created >= self.ttl_s:
            raise OIDCError("this sign-in link expired or was already used; start the sign-in again",
                            "oidc_bad_state", code=ErrorCode.PROTOCOL)
        if not binding or not hmac.compare_digest(hashlib.sha256(binding.encode()).digest(), p.binding):
            raise OIDCError("this sign-in was started in another browser; start the sign-in again",
                            "oidc_bad_binding", code=ErrorCode.PROTOCOL)
        return p

    def __len__(self) -> int:
        return len(self._items)


# =================================================================================================
# client
# =================================================================================================
@dataclass
class OIDCIdentity:
    """A verified SSO user."""

    role: Role
    sub: str
    email: str | None
    name: str | None
    claims: dict[str, Any] = field(repr=False, default_factory=dict)

    def user(self) -> dict[str, str | None]:
        """What ``/console/api/me`` shows."""
        return {"sub": self.sub, "email": self.email, "name": self.name}


def _origin_ok(url: Any, allow_http_loopback: bool) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    if not parts.hostname:
        return False
    if parts.scheme == "https":
        return True
    if parts.scheme == "http" and allow_http_loopback:
        from pgw.console.sessions import is_loopback_host

        return is_loopback_host(parts.hostname)
    return False


def claim_values(claims: dict[str, Any], path: str) -> list[str]:
    """String values of a (dotted-path) claim: a string, or a list of strings."""
    v: Any = claims
    for part in path.split("."):
        if not isinstance(v, dict):
            return []
        v = v.get(part)
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return []


class OIDCClient:
    """Relying party for one OIDC provider (discovery and JWKS cached in memory).

    Args:
        issuer: Issuer URL (must equal the discovery document's ``issuer``).
        client_id, client_secret: Confidential client credentials.
        redirect_url: Registered redirect URI (``.../console/api/oidc/callback``).
        scopes: Space-separated scopes.
        role_claim, admin_values, viewer_values: Role mapping.
        allowed_domains: Optional email domain allowlist (lower case).
        ca_file: Optional CA bundle for the provider's TLS certificate.
        label: Provider name for the sign-in button.
        transport: httpx transport (tests).
        clock: Wall clock for token checks (tests).
    """

    discovery_ttl_s = 3600.0
    jwks_ttl_s = 3600.0
    #: A JWKS refresh for an unknown ``kid`` happens at most this often.
    jwks_min_refresh_s = 30.0
    timeout_s = 10.0

    def __init__(self, *, issuer: str, client_id: str, client_secret: str, redirect_url: str, scopes: str,
                 role_claim: str, admin_values: list[str], viewer_values: list[str],
                 allowed_domains: list[str] | None = None, ca_file: Path | None = None, label: str = "SSO",
                 transport: httpx.AsyncBaseTransport | None = None, clock: Callable[[], float] = time.time):
        self.issuer = issuer
        self.client_id = client_id
        self._secret = client_secret
        self.redirect_url = redirect_url
        self.scopes = " ".join(scopes.split())
        self.role_claim = role_claim.strip()
        self.admin_values = frozenset(v.strip() for v in admin_values if v.strip())
        self.viewer_values = frozenset(v.strip() for v in viewer_values if v.strip())
        self.allowed_domains = frozenset(d.strip().lstrip("@").lower() for d in (allowed_domains or []) if d.strip())
        self.label = label
        self.clock = clock
        self.pending = PendingFlows()
        self._transport = transport
        self._tls_verify: ssl.SSLContext | bool = True
        if ca_file is not None:
            try:
                self._tls_verify = ssl.create_default_context(cafile=str(ca_file))
            except (OSError, ssl.SSLError) as exc:
                raise ConfigError(f"cannot load console_oidc_ca_file ({type(exc).__name__})") from exc
        self._loopback_ok = urlsplit(issuer).scheme == "http"  # settings only allow http for loopback
        self._discovery: dict[str, Any] | None = None
        self._discovery_at = 0.0
        self._keys: list[_Jwk] = []
        self._jwks_at = 0.0
        self._lock = asyncio.Lock()
        self._jwks_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Any, *, transport: httpx.AsyncBaseTransport | None = None) -> OIDCClient:
        """Build from :class:`~pgw.gateway.config.GatewaySettings` (reads the client secret file).

        Raises:
            ConfigError: the secret file cannot be read or is empty.
        """
        path = settings.console_oidc_client_secret_file
        try:
            secret = Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(
                f"cannot read console_oidc_client_secret_file ({exc.strerror or type(exc).__name__})") from exc
        if not secret or any(c.isspace() for c in secret):
            raise ConfigError("console_oidc_client_secret_file must contain exactly one secret")
        return cls(issuer=settings.console_oidc_issuer, client_id=settings.console_oidc_client_id,
                   client_secret=secret, redirect_url=settings.console_oidc_redirect_url,
                   scopes=settings.console_oidc_scopes, role_claim=settings.console_oidc_role_claim,
                   admin_values=settings.console_oidc_admin_values, viewer_values=settings.console_oidc_viewer_values,
                   allowed_domains=settings.console_oidc_allowed_domains, ca_file=settings.console_oidc_ca_file,
                   label=settings.console_oidc_label, transport=transport)

    # ---- HTTP ------------------------------------------------------------------------------------------
    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, verify=self._tls_verify, timeout=self.timeout_s,
                                 follow_redirects=False, headers={"Accept": "application/json"})

    async def _json(self, method: str, url: str, what: str, **kw: Any) -> tuple[int, dict[str, Any]]:
        try:
            async with self._client() as c:
                r = await c.request(method, url, **kw)
                body = await r.aread()
        except httpx.HTTPError as exc:
            log.warning("console SSO: %s request failed (%s)", what, type(exc).__name__)
            raise OIDCError(f"the SSO provider could not be reached ({what})", "oidc_unreachable",
                            code=ErrorCode.UNAVAILABLE) from exc
        if len(body) > _MAX_RESPONSE_BYTES:
            raise OIDCError(f"the SSO provider's {what} response is too large", "oidc_bad_response",
                            code=ErrorCode.UNAVAILABLE)
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            if r.status_code >= 500 or r.status_code < 400:
                raise OIDCError(f"the SSO provider's {what} response is not a JSON object", "oidc_bad_response",
                                code=ErrorCode.UNAVAILABLE)
            data = {}
        return r.status_code, data

    async def discovery(self) -> dict[str, Any]:
        """Provider metadata (cached; refreshed hourly, the last good copy kept on errors)."""
        now = self.clock()
        if self._discovery is not None and now - self._discovery_at < self.discovery_ttl_s:
            return self._discovery
        async with self._lock:
            if self._discovery is not None and self.clock() - self._discovery_at < self.discovery_ttl_s:
                return self._discovery
            url = self.issuer.rstrip("/") + "/.well-known/openid-configuration"
            try:
                status, doc = await self._json("GET", url, "discovery")
                if status != 200:
                    raise OIDCError(f"the SSO provider's discovery document answered HTTP {status}",
                                    "oidc_bad_response", code=ErrorCode.UNAVAILABLE)
                self._check_discovery(doc)
            except OIDCError:
                if self._discovery is not None:
                    log.warning("console SSO: discovery refresh failed; keeping the cached document")
                    self._discovery_at = self.clock() - self.discovery_ttl_s + 60.0  # retry in a minute
                    return self._discovery
                raise
            self._discovery, self._discovery_at = doc, self.clock()
            return doc

    def _check_discovery(self, doc: dict[str, Any]) -> None:
        if doc.get("issuer") != self.issuer:
            raise OIDCError("the SSO provider's discovery document names another issuer (check "
                            "console_oidc_issuer, including any trailing slash)", "oidc_bad_discovery",
                            code=ErrorCode.UNAVAILABLE)
        for k in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not _origin_ok(doc.get(k), self._loopback_ok):
                raise OIDCError(f"the SSO provider's {k} is missing or not https", "oidc_bad_discovery",
                                code=ErrorCode.UNAVAILABLE)
        methods = doc.get("code_challenge_methods_supported")
        if isinstance(methods, list) and "S256" not in methods:
            raise OIDCError("the SSO provider does not support PKCE S256", "oidc_bad_discovery",
                            code=ErrorCode.UNAVAILABLE)

    async def _jwks(self, *, force: bool) -> list[_Jwk]:
        now = self.clock()
        if not force and self._keys and now - self._jwks_at < self.jwks_ttl_s:
            return self._keys
        doc = await self.discovery()
        async with self._jwks_lock:
            now = self.clock()
            fresh = now - self._jwks_at < self.jwks_min_refresh_s
            if self._keys and (fresh if force else now - self._jwks_at < self.jwks_ttl_s):
                return self._keys
            status, data = await self._json("GET", str(doc["jwks_uri"]), "JWKS")
            try:
                if status != 200:
                    raise ValueError(f"HTTP {status}")
                keys = parse_jwks(data)
            except ValueError as exc:
                raise OIDCError(f"the SSO provider's signing keys are unusable ({exc})", "oidc_bad_response",
                                code=ErrorCode.UNAVAILABLE) from exc
            self._keys, self._jwks_at = keys, self.clock()
            return keys

    # ---- flow -------------------------------------------------------------------------------------------
    async def start(self) -> tuple[str, str]:
        """Begin a sign-in: ``(authorization URL, browser binding for the pgw_oidc cookie)``."""
        doc = await self.discovery()
        state, nonce, verifier, binding = self.pending.create()
        params = {
            "response_type": "code", "client_id": self.client_id, "redirect_uri": self.redirect_url,
            "scope": self.scopes, "state": state, "nonce": nonce,
            "code_challenge": pkce_challenge(verifier), "code_challenge_method": "S256",
        }
        endpoint = str(doc["authorization_endpoint"])
        sep = "&" if urlsplit(endpoint).query else "?"
        return endpoint + sep + urlencode(params), binding

    async def finish(self, *, code: str | None, state: str | None, binding: str | None, error: str | None = None,
                     iss: str | None = None) -> OIDCIdentity:
        """Complete a sign-in from the callback's query parameters.

        Raises:
            OIDCError: 400 (``protocol``) for a bad/replayed state or binding, 401 for a refused or
                invalid sign-in, 403 (``permission_denied``) for a user without a mapped role or with
                an email domain outside the allowlist, 503 when the provider cannot be reached.
        """
        pending = self.pending.pop(state or "", binding)
        if error:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "", error)[:64] or "error"
            raise OIDCError(f"the SSO provider refused the sign-in ({safe})", "oidc_provider_error")
        if iss is not None and iss != self.issuer:  # RFC 9207 mix-up defence
            raise _fail("the authorization response comes from another issuer", "oidc_bad_issuer")
        if not code or len(code) > 4096:
            raise OIDCError("the authorization response has no code", "oidc_bad_response", code=ErrorCode.PROTOCOL)
        doc = await self.discovery()
        id_token = await self._exchange(doc, code, pending.verifier)
        claims = await self._verify(id_token, pending.nonce)
        return self._identity(claims)

    async def _exchange(self, doc: dict[str, Any], code: str, verifier: str) -> str:
        form = {"grant_type": "authorization_code", "code": code, "redirect_uri": self.redirect_url,
                "code_verifier": verifier}
        methods = doc.get("token_endpoint_auth_methods_supported")
        kw: dict[str, Any] = {}
        if isinstance(methods, list) and "client_secret_basic" not in methods and "client_secret_post" in methods:
            form |= {"client_id": self.client_id, "client_secret": self._secret}
        else:  # RFC 6749 2.3.1: form-urlencode the credentials before Basic encoding
            from urllib.parse import quote

            kw["auth"] = httpx.BasicAuth(quote(self.client_id, safe=""), quote(self._secret, safe=""))
        status, data = await self._json("POST", str(doc["token_endpoint"]), "token", data=form, **kw)
        if status != 200:
            err = re.sub(r"[^A-Za-z0-9_.-]", "", str(data.get("error", "")))[:64] or f"HTTP {status}"
            if status >= 500:
                raise OIDCError(f"the SSO provider's token endpoint failed ({err})", "oidc_token_error",
                                code=ErrorCode.UNAVAILABLE)
            raise _fail(f"the SSO provider rejected the code exchange ({err})", "oidc_token_error")
        tok = data.get("id_token")
        if not isinstance(tok, str) or not tok:
            raise _fail("the token response has no ID token (is 'openid' among the scopes?)", "oidc_token_error")
        return tok

    async def _verify(self, id_token: str, nonce: str) -> dict[str, Any]:
        def check(keys: list[_Jwk]) -> dict[str, Any]:
            return verify_id_token(id_token, keys, issuer=self.issuer, client_id=self.client_id, nonce=nonce,
                                   now=self.clock())

        try:
            return check(await self._jwks(force=False))
        except OIDCError as exc:
            if exc.reason != "oidc_unknown_key":
                raise
        return check(await self._jwks(force=True))  # key rotation: fetch the JWKS again (rate limited)

    def _identity(self, claims: dict[str, Any]) -> OIDCIdentity:
        sub = str(claims["sub"])
        email = claims.get("email") if isinstance(claims.get("email"), str) else None
        name = next((claims[k] for k in ("name", "preferred_username") if isinstance(claims.get(k), str)), None)
        email = email[:320] if email else None
        name = name[:200] if name else None
        if self.allowed_domains:
            domain = email.rsplit("@", 1)[1].lower() if email and "@" in email else None
            if domain is None or domain not in self.allowed_domains or claims.get("email_verified") is False:
                raise OIDCError("your account's email domain is not allowed to use this console",
                                "oidc_domain_not_allowed", code=ErrorCode.PERMISSION_DENIED, subject=sub, email=email)
        values = set(claim_values(claims, self.role_claim))
        role: Role
        if values & self.admin_values:
            role = "admin"
        elif values & self.viewer_values:
            role = "viewer"
        else:
            raise OIDCError("your account has no console role (ask an administrator to add you to the console "
                            "admin or viewer group)", "oidc_no_role", code=ErrorCode.PERMISSION_DENIED,
                            subject=sub, email=email)
        return OIDCIdentity(role=role, sub=sub, email=email, name=name, claims=claims)


def as_auth_error(exc: Exception) -> PGWError:
    """Unexpected verification failures become a generic 401 (details only in the server log)."""
    if isinstance(exc, PGWError):
        return exc
    log.warning("console SSO: unexpected error %s", type(exc).__name__)
    return AuthError("SSO sign-in failed")
