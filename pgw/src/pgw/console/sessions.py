"""Console sign-in: token check, in-memory sessions with CSRF tokens, login rate limiting.

Tokens (the gateway admin token and the optional viewer tokens) are kept in memory only as SHA-256
digests and compared in constant time. A session is a random 256-bit id sent in the ``pgw_console``
cookie; the store keys sessions by the SHA-256 of that id, so a memory dump of the store does not
yield usable cookies. Each session carries a CSRF token that state-changing requests made with the
cookie must echo in ``X-PGW-CSRF``.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import secrets
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pgw.core.errors import AuthError, ConfigError, ErrorCode, PGWError, RateLimitedError

__all__ = [
    "COOKIE_NAME", "CSRF_HEADER", "SAFE_METHODS", "ConsoleSession", "ConsoleSessions", "LoginRateLimiter",
    "cookie_secure", "is_loopback_host",
]

Role = Literal["admin", "viewer"]

COOKIE_NAME = "pgw_console"
CSRF_HEADER = "X-PGW-CSRF"
#: Methods that never change state (no CSRF token needed; viewers may use them).
SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def is_loopback_host(host: str | None) -> bool:
    """Whether ``host`` (a hostname or IP literal, IPv6 possibly in brackets) is a loopback name."""
    if not host:
        return False
    h = host.strip().lower().strip("[]")
    if h == "localhost" or h.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def cookie_secure(scheme: str, host: str | None, client_host: str | None) -> bool:
    """``Secure`` cookie flag: always, except plain HTTP to a loopback host from a loopback client."""
    return not (scheme == "http" and is_loopback_host(host) and is_loopback_host(client_host))


@dataclass
class ConsoleSession:
    """One signed-in console user (the cookie value itself is not stored)."""

    role: Role
    csrf: str
    created: float
    last_seen: float
    public_id: str = field(default_factory=lambda: secrets.token_hex(4))
    #: How the user signed in: ``token`` (admin/viewer token) or ``oidc`` (single sign-on).
    auth: Literal["token", "oidc"] = "token"
    #: SSO user (``{"sub", "email", "name"}``); ``None`` for token sign-in.
    user: dict[str, str | None] | None = None

    @property
    def actor(self) -> str:
        """Actor id for the audit log (no secret material)."""
        return f"console-{self.role}:{self.public_id}"


class LoginRateLimiter:
    """Sliding-window limit on login attempts per client address.

    Args:
        attempts: Attempts allowed per ``window_s``.
        window_s: Window length.
        max_clients: Addresses tracked at once (least recently seen are dropped beyond this).
    """

    def __init__(self, attempts: int = 10, window_s: float = 60.0, *, max_clients: int = 10_000,
                 clock: Callable[[], float] = time.monotonic):
        self.attempts = attempts
        self.window_s = window_s
        self.max_clients = max_clients
        self.clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, client: str) -> None:
        """Count one attempt by ``client``; raises :class:`RateLimitedError` above the limit."""
        now = self.clock()
        with self._lock:
            q = self._hits.pop(client, None) or deque()
            while q and q[0] <= now - self.window_s:
                q.popleft()
            self._hits[client] = q
            while len(self._hits) > self.max_clients:
                self._hits.popitem(last=False)
            if len(q) >= self.attempts:
                retry = max(1.0, q[0] + self.window_s - now)
                raise RateLimitedError("too many console login attempts; try again later",
                                       details={"retry_after_s": retry, "limit": "console_login"})
            q.append(now)


class ConsoleSessions:
    """Token check and in-memory session store of the console.

    Args:
        admin_token: The gateway admin token (``None`` disables console login entirely).
        viewer_tokens_file: Optional file with read-only tokens, one per line (``#`` comments allowed).
        idle_s: A session expires after this long without a request.
        max_age_s: A session expires this long after sign-in regardless of activity.
        max_sessions: Sessions kept at once; the least recently used are dropped beyond this.
        clock: Wall clock (tests).
    """

    def __init__(self, admin_token: str | None, viewer_tokens_file: Path | None = None, *,
                 idle_s: float = 8 * 3600.0, max_age_s: float = 24 * 3600.0, max_sessions: int = 1000,
                 clock: Callable[[], float] = time.time):
        self._admin = _digest(admin_token) if admin_token else None
        self.viewer_tokens_file = Path(viewer_tokens_file) if viewer_tokens_file is not None else None
        self._viewers: list[bytes] = []
        self._viewers_mtime: float | None = None
        self.idle_s = idle_s
        self.max_age_s = max_age_s
        self.max_sessions = max_sessions
        self.clock = clock
        self._sessions: OrderedDict[bytes, ConsoleSession] = OrderedDict()
        self._lock = threading.Lock()
        if self.viewer_tokens_file is not None:
            self._load_viewers(strict=True)

    # ---- tokens ------------------------------------------------------------------------------------
    @property
    def login_enabled(self) -> bool:
        """Console login needs an admin token configured on the gateway."""
        return self._admin is not None

    def _load_viewers(self, *, strict: bool) -> None:
        path = self.viewer_tokens_file
        assert path is not None
        try:
            mtime = os.stat(path).st_mtime
            if mtime == self._viewers_mtime:
                return
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            if strict:
                raise ConfigError(
                    f"cannot read console_viewer_tokens_file ({exc.strerror or type(exc).__name__})"
                ) from exc
            return  # keep the last good set while the file is being replaced
        tokens = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        self._viewers = [_digest(t) for t in tokens]
        self._viewers_mtime = mtime

    def authenticate(self, token: str) -> Role | None:
        """Role of ``token`` (``None`` if it is neither the admin token nor a viewer token).

        Every stored digest is compared (constant time) so timing does not reveal which matched.
        """
        if not self.login_enabled or not token:
            return None
        if self.viewer_tokens_file is not None:
            self._load_viewers(strict=False)
        d = _digest(token)
        assert self._admin is not None
        is_admin = hmac.compare_digest(d, self._admin)
        is_viewer = False
        for v in self._viewers:
            is_viewer |= hmac.compare_digest(d, v)
        if is_admin:
            return "admin"
        if is_viewer:
            return "viewer"
        return None

    # ---- sessions ----------------------------------------------------------------------------------
    def expires_at(self, s: ConsoleSession) -> float:
        """Unix time the session expires unless used again (idle) or at all (absolute)."""
        return min(s.last_seen + self.idle_s, s.created + self.max_age_s)

    def create(self, role: Role, *, auth: Literal["token", "oidc"] = "token",
               user: dict[str, str | None] | None = None) -> tuple[str, ConsoleSession]:
        """New session; returns ``(cookie value, session)``."""
        sid = secrets.token_urlsafe(32)  # 256 bits
        now = self.clock()
        sess = ConsoleSession(role=role, csrf=secrets.token_urlsafe(32), created=now, last_seen=now, auth=auth,
                              user=dict(user) if user is not None else None)
        with self._lock:
            self._purge(now)
            self._sessions[_digest(sid)] = sess
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return sid, sess

    def _purge(self, now: float) -> None:
        dead = [k for k, s in self._sessions.items() if now >= self.expires_at(s)]
        for k in dead:
            del self._sessions[k]

    def get(self, sid: str | None, *, touch: bool = True) -> ConsoleSession | None:
        """Live session for a cookie value (refreshes its idle timer unless ``touch`` is false)."""
        if not sid or len(sid) > 256:
            return None
        k = _digest(sid)
        now = self.clock()
        with self._lock:
            s = self._sessions.get(k)
            if s is None:
                return None
            if now >= self.expires_at(s):
                del self._sessions[k]
                return None
            if touch:
                s.last_seen = now
                self._sessions.move_to_end(k)
            return s

    def delete(self, sid: str | None) -> ConsoleSession | None:
        """End a session (returns it, or ``None`` if there was none)."""
        if not sid:
            return None
        with self._lock:
            return self._sessions.pop(_digest(sid), None)

    def __len__(self) -> int:
        return len(self._sessions)

    def authorize(self, sid: str, method: str, csrf: str | None) -> ConsoleSession:
        """Check a cookie-authenticated admin request.

        Viewers may only use safe methods; every other method needs the admin role and the session's
        CSRF token in ``X-PGW-CSRF``.

        Raises:
            AuthError: no live session for the cookie (401).
            PGWError: ``permission_denied`` for a viewer write or a missing/wrong CSRF token (403).
        """
        s = self.get(sid)
        if s is None:
            raise AuthError("console session expired or invalid; sign in again")
        if method.upper() not in SAFE_METHODS:
            if not csrf or not hmac.compare_digest(csrf.encode(), s.csrf.encode()):
                raise PGWError(f"missing or invalid CSRF token ({CSRF_HEADER})", code=ErrorCode.PERMISSION_DENIED)
            if s.role != "admin":
                raise PGWError("viewer sessions are read-only", code=ErrorCode.PERMISSION_DENIED)
        return s
