"""Signed endpoint directory: the provider's current set of GPU endpoints, verified by the client (docs/sdk/wan.md).

The placement controller (:mod:`pgw.placement`) starts and stops GPU pods in the data centres nearest the users and
publishes the endpoints that serve right now as a **signed directory**. The client library fetches it from any
endpoint it knows (``GET /pgw/v1/endpoints/directory``), verifies it, and routes with :mod:`pgw.sdk.regions` (nearest
healthy endpoint serving the model's fingerprint; a switch only between requests, never during an answer).

Wire form (JSON)::

    {"payload": base64url(P), "sig": base64url(Ed25519(DOMAIN | P))}

``P`` is UTF-8 JSON (the bytes are what is signed: the verifier never re-serialises)::

    {"format": "pgw-endpoint-directory", "v": 1, "version": <int, strictly increasing>, "issued_at": <unix s>,
     "expires_at": <unix s>, "key_id": <16 hex: SHA-256(public key)[:8]>,
     "endpoints": [{"region", "url": "https://host:port", "models": [fingerprints], "priority",
                    "lane_port"?, "dc"?, "state": "serving" | "draining"}, ...]}

Checks (:func:`verify_directory`), all before anything in it is used:

* the signature, with the **pinned** Ed25519 public key (the SDK key's ``directory_key``, or the one the enrollment
  response carried over the CA-pinned TLS channel), under the domain ``DOMAIN``;
* freshness: ``issued_at <= now + skew``, ``now <= expires_at + skew``, validity at most :data:`MAX_VALIDITY_S`;
* no rollback: ``version`` is at least the highest version this installation ever accepted (persisted next to the
  session, 0600); the same version is accepted only with the same bytes (a re-fetch), never a different document;
* the endpoints themselves with :func:`pgw.sdk.regions.parse_endpoints` (https only, host and port, at most 32, the
  enrolled URL always kept).

What the signature adds to TLS. Every endpoint still has to present a certificate chaining to the pinned provider CA
and the client still checks the pack fingerprint and every result, so a forged list could never read a prompt or
pass a wrong answer. The signature and the version stop *steering*: a network attacker or a stale cache cannot send
clients to a decommissioned pod, to an endpoint far away, or back to an old list; an expired list stops being used.

If no valid directory is at hand (none published, fetch failed, expired), the client uses the enrolled URL only:
the one the SDK key itself names. Routing never removes that choice.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pgw.sdk.regions import Endpoint, Probe, choose, failover_order, parse_endpoints, probe

__all__ = [
    "DIRECTORY_PATH",
    "DOMAIN",
    "MAX_VALIDITY_S",
    "RTT_BUCKETS",
    "Directory",
    "DirectoryError",
    "EndpointRouter",
    "VersionStore",
    "directory_key_id",
    "load_public_key",
    "rtt_bucket",
    "verify_directory",
]

log = logging.getLogger(__name__)

DIRECTORY_PATH = "/pgw/v1/endpoints/directory"
#: The content-free demand report (pgw.placement.telemetry; the service's ``demand_reports``).
DEMAND_REPORT_PATH = "/pgw/v1/placement/report"
DOMAIN = b"pgw-endpoint-directory-v1\x00"
FORMAT = "pgw-endpoint-directory"
#: Longest validity a directory may claim (a leaked old list stops working within this bound, whatever it says).
MAX_VALIDITY_S = 24 * 3600
#: Clock skew tolerated on ``issued_at`` / ``expires_at``.
SKEW_S = 300
MAX_DIRECTORY_BYTES = 64 * 1024
#: The RTT buckets of the content-free demand reports (ms, upper bounds; the last is open).
RTT_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)


class DirectoryError(ValueError):
    """A directory failed verification (signature, freshness, rollback, form). Never contains key material."""


def rtt_bucket(rtt_ms: float) -> str:
    """The coarse RTT bucket label of a measurement (``"<=5"``, ..., ``">160"``)."""
    for b in RTT_BUCKETS:
        if rtt_ms <= b:
            return f"<={b:g}"
    return f">{RTT_BUCKETS[-1]:g}"


def _b64d(s: Any) -> bytes:
    if not isinstance(s, str) or len(s) > 2 * MAX_DIRECTORY_BYTES:
        raise DirectoryError("directory fields must be base64url strings")
    try:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise DirectoryError("directory is not valid base64url") from exc


def load_public_key(raw: bytes | str) -> bytes:
    """A 32-byte Ed25519 public key from raw bytes or 64 hex digits."""
    if isinstance(raw, str):
        try:
            raw = bytes.fromhex(raw.strip())
        except ValueError as exc:
            raise DirectoryError("directory key must be 64 hex digits") from exc
    if len(raw) != 32:
        raise DirectoryError("directory key must be a 32-byte Ed25519 public key")
    return bytes(raw)


def directory_key_id(public_key: bytes) -> str:
    """``SHA-256(public key)[:8]`` as hex: names the key in the document (rotation, logs)."""
    return hashlib.sha256(load_public_key(public_key)).hexdigest()[:16]


@dataclass(frozen=True)
class Directory:
    """A verified directory."""

    version: int
    issued_at: int
    expires_at: int
    key_id: str
    endpoints: tuple[Endpoint, ...]
    lane_ports: Mapping[str, int] = field(default_factory=dict)  # url -> lane port
    draining: frozenset[str] = frozenset()  # urls that take no new models
    dcs: Mapping[str, str] = field(default_factory=dict)  # url -> data centre id (status only)
    digest: str = ""  # SHA-256 of the signed payload

    def serving(self) -> list[Endpoint]:
        return [e for e in self.endpoints if e.url not in self.draining]

    def public(self) -> dict[str, Any]:
        return {"version": self.version, "issued_at": self.issued_at, "expires_at": self.expires_at,
                "key_id": self.key_id, "endpoints": [{"region": e.region, "url": e.url, "dc": self.dcs.get(e.url),
                                                      "draining": e.url in self.draining} for e in self.endpoints]}


class VersionStore:
    """The highest directory version accepted by this installation, and the digest of that document (rollback
    protection across restarts). One small JSON file, written atomically, 0600."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path is not None else None
        self._mem: tuple[int, str] = (0, "")
        self._lock = threading.Lock()

    def get(self) -> tuple[int, str]:
        with self._lock:
            if self.path is None:
                return self._mem
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                return int(d["version"]), str(d["digest"])
            except (OSError, ValueError, KeyError, TypeError):
                return self._mem

    def put(self, version: int, digest: str) -> None:
        with self._lock:
            self._mem = (version, digest)
            if self.path is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": version, "digest": digest}, f)
            os.replace(tmp, self.path)


def verify_directory(blob: bytes | str | Mapping[str, Any], public_key: bytes, *, now: float | None = None,
                     min_version: tuple[int, str] = (0, ""), enrolled_url: str | None = None) -> Directory:
    """Verify a signed directory (see the module docstring) and return it.

    Args:
        blob: The wire form (bytes, str or the parsed object).
        public_key: The pinned Ed25519 public key (32 bytes).
        now: Unix time (tests).
        min_version: ``(version, digest)`` of the newest directory accepted so far.
        enrolled_url: Kept as an endpoint whatever the list says.

    Raises:
        DirectoryError: anything wrong; nothing of the document is used then.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    pk = load_public_key(public_key)
    if isinstance(blob, bytes | bytearray):
        if len(blob) > MAX_DIRECTORY_BYTES:
            raise DirectoryError("directory too large")
        try:
            blob = json.loads(bytes(blob).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise DirectoryError("directory is not JSON") from exc
    elif isinstance(blob, str):
        if len(blob) > MAX_DIRECTORY_BYTES:
            raise DirectoryError("directory too large")
        try:
            blob = json.loads(blob)
        except ValueError as exc:
            raise DirectoryError("directory is not JSON") from exc
    if not isinstance(blob, Mapping) or set(blob) != {"payload", "sig"}:
        raise DirectoryError("directory must be {payload, sig}")
    payload, sig = _b64d(blob["payload"]), _b64d(blob["sig"])
    if len(payload) > MAX_DIRECTORY_BYTES or len(sig) != 64:
        raise DirectoryError("directory payload too large or signature malformed")
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(sig, DOMAIN + payload)
    except InvalidSignature:
        raise DirectoryError("directory signature does not verify with the pinned key") from None
    try:
        d = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:  # pragma: no cover - signed by us: never happens
        raise DirectoryError("directory payload is not JSON") from exc
    if not isinstance(d, dict) or d.get("format") != FORMAT or d.get("v") != 1:
        raise DirectoryError("not a version-1 endpoint directory")
    try:
        version, issued, expires = int(d["version"]), int(d["issued_at"]), int(d["expires_at"])
        kid, raw_eps = str(d["key_id"]), d["endpoints"]
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectoryError("directory is missing version, times, key_id or endpoints") from exc
    if any(isinstance(d[k], bool) for k in ("version", "issued_at", "expires_at")):
        raise DirectoryError("directory version and times must be integers")
    if kid != directory_key_id(pk):
        raise DirectoryError("directory names another signing key")
    t = time.time() if now is None else float(now)
    if version < 1:
        raise DirectoryError("directory version must be positive")
    if expires <= issued or expires - issued > MAX_VALIDITY_S:
        raise DirectoryError(f"directory validity must be in (0, {MAX_VALIDITY_S}] s")
    if issued > t + SKEW_S:
        raise DirectoryError("directory is issued in the future (clock skew?)")
    if t > expires + SKEW_S:
        raise DirectoryError("directory has expired")
    digest = hashlib.sha256(payload).hexdigest()
    hv, hd = min_version
    if version < hv:
        raise DirectoryError(f"directory version {version} is older than the accepted {hv} (rollback refused)")
    if version == hv and hd and digest != hd:
        raise DirectoryError(f"directory version {version} was already accepted with different content")
    if not isinstance(raw_eps, list):
        raise DirectoryError("directory endpoints must be a list")
    lane: dict[str, int] = {}
    draining: set[str] = set()
    dcs: dict[str, str] = {}
    base = []
    for it in raw_eps:
        if not isinstance(it, Mapping):
            raise DirectoryError("an endpoint must be an object")
        extra = set(it) - {"region", "url", "models", "priority", "lane_port", "dc", "state"}
        if extra:
            raise DirectoryError(f"unknown endpoint field(s) {sorted(extra)}")
        base.append({k: it[k] for k in ("region", "url", "models", "priority") if k in it})
    try:
        eps = parse_endpoints(base, enrolled_url=enrolled_url)
    except ValueError as exc:
        raise DirectoryError(f"directory endpoints: {exc}") from exc
    norm = {e.url for e in eps}
    for it in raw_eps:
        from urllib.parse import urlsplit

        p = urlsplit(str(it["url"]))
        url = f"https://{p.hostname}:{p.port or 443}"
        if url not in norm:  # pragma: no cover - parse_endpoints kept every valid entry
            continue
        lp = it.get("lane_port")
        if lp is not None:
            if not isinstance(lp, int) or isinstance(lp, bool) or not 0 < lp < 65536:
                raise DirectoryError("lane_port must be a TCP port")
            lane[url] = lp
        st = it.get("state", "serving")
        if st not in ("serving", "draining"):
            raise DirectoryError("endpoint state must be serving or draining")
        if st == "draining":
            draining.add(url)
        if it.get("dc") is not None:
            if not isinstance(it["dc"], str) or not 0 < len(it["dc"]) <= 32:
                raise DirectoryError("dc must be a short string")
            dcs[url] = it["dc"]
    return Directory(version, issued, expires, kid, tuple(eps), lane, frozenset(draining), dcs, digest)


Fetch = Callable[[str], bytes]  # base URL -> the directory's bytes (pinned-CA TLS, the session's credentials)
Prober = Callable[[Sequence[Endpoint]], list[Probe]]


class EndpointRouter:
    """Directory refresh, probing and the endpoint choice of one installation (thread-safe).

    * :meth:`refresh` fetches the directory from the current endpoint, then the other known ones, then the enrolled
      URL, verifies it and keeps it if newer; every ``refresh_s`` and after a failure (:meth:`report_failure`).
    * :meth:`reprobe` runs payload-free handshake probes (:func:`pgw.sdk.regions.probe`).
    * :meth:`select` is the endpoint for a model that is (re)starting; :meth:`preferred_switch` says whether a model
      that is idle should move (hysteresis of :func:`pgw.sdk.regions.choose`). The engine calls both only between
      requests: an answer in progress never moves.

    Without a valid directory every call answers the enrolled endpoint.

    Args:
        enrolled_url: The SDK key's provider URL.
        public_key: The pinned directory key (32 bytes).
        fetch: Fetches the directory's bytes from a base URL.
        versions: Rollback state (persisted).
        prober: Probes endpoints (default: TCP handshakes, no payload).
        clock: Wall clock (directory freshness); ``mono``: monotonic clock (schedules).
    """

    def __init__(self, enrolled_url: str, public_key: bytes, fetch: Fetch, *, versions: VersionStore | None = None,
                 prober: Prober | None = None, refresh_s: float = 600.0, probe_s: float = 600.0,
                 failure_hold_s: float = 120.0, clock: Callable[[], float] = time.time,
                 mono: Callable[[], float] = time.monotonic):
        self.enrolled = parse_endpoints([], enrolled_url=enrolled_url)[0]
        self.public_key = load_public_key(public_key)
        self.fetch, self.versions = fetch, versions or VersionStore(None)
        self.prober = prober or (lambda eps: probe(eps))
        self.refresh_s, self.probe_s, self.failure_hold_s = refresh_s, probe_s, failure_hold_s
        self.clock, self.mono = clock, mono
        self.directory: Directory | None = None
        self.probes: list[Probe] = []
        self._probed_at = -1e18
        self._refreshed_at = -1e18
        self._failed: dict[str, float] = {}
        self._lock = threading.RLock()
        self.errors: list[str] = []  # last few refresh errors (public: no key material, no URLs' credentials)
        self.switches = 0

    # ---- directory ----------------------------------------------------------------------------------------
    def _valid(self) -> Directory | None:
        d = self.directory
        if d is not None and self.clock() > d.expires_at + SKEW_S:
            log.warning("endpoint directory v%d expired; using the enrolled endpoint only", d.version)
            self.directory = d = None
        return d

    def known(self) -> list[Endpoint]:
        """Every endpoint of the current valid directory (the enrolled one included), else the enrolled one."""
        with self._lock:
            d = self._valid()
            return list(d.endpoints) if d is not None else [self.enrolled]

    def lane_port(self, url: str) -> int | None:
        with self._lock:
            d = self._valid()
            return d.lane_ports.get(url) if d is not None else None

    def refresh_due(self) -> bool:
        return self.mono() - self._refreshed_at >= self.refresh_s

    def refresh(self, *, prefer: str | None = None) -> bool:
        """Fetch and verify the directory; ``True`` if a newer one was accepted. Tries ``prefer``, the known
        endpoints (healthy first), then the enrolled URL; the first that verifies wins. Never raises."""
        with self._lock:
            self._refreshed_at = self.mono()
            order = [prefer] if prefer else []
            order += [e.url for e in failover_order(self.probes)] if self.probes else []
            order += [e.url for e in self.known()] + [self.enrolled.url]
            tried: set[str] = set()
            for url in order:
                if url in tried:
                    continue
                tried.add(url)
                try:
                    blob = self.fetch(url)
                except Exception as exc:  # unreachable endpoint: try the next
                    self._note(f"fetch from {url} failed: {type(exc).__name__}")
                    continue
                try:
                    d = verify_directory(blob, self.public_key, now=self.clock(), min_version=self.versions.get(),
                                         enrolled_url=self.enrolled.url)
                except DirectoryError as exc:
                    self._note(f"directory from {url} refused: {exc}")
                    continue
                newer = self.directory is None or d.version > self.directory.version
                self.versions.put(d.version, d.digest)
                if newer:
                    log.info("endpoint directory v%d: %d endpoint(s)", d.version, len(d.endpoints))
                    self.directory = d
                    self._probed_at = -1e18  # new endpoints: probe again before the next choice
                return newer
            return False

    def _note(self, msg: str) -> None:
        log.warning("%s", msg)
        self.errors = [*self.errors, msg][-5:]

    # ---- probing and choice -------------------------------------------------------------------------------
    def probe_due(self) -> bool:
        return self.mono() - self._probed_at >= self.probe_s

    def reprobe(self) -> list[Probe]:
        eps = self.known()
        res = self.prober(eps)
        with self._lock:
            self.probes, self._probed_at = res, self.mono()
            for p in res:  # a healthy probe clears an earlier failure mark
                if p.healthy:
                    self._failed.pop(p.endpoint.url, None)
        return res

    def maintain(self) -> None:
        """Refresh and probe if due (call between requests, off the request path)."""
        if self.refresh_due():
            self.refresh()
        if self.probe_due():
            self.reprobe()

    def _usable(self, fingerprint: str | None) -> list[Probe]:
        d = self._valid()
        draining = d.draining if d is not None else frozenset()
        known = {e.url for e in self.known()}
        now = self.mono()
        held = {u for u, t in self._failed.items() if now - t < self.failure_hold_s}
        return [p for p in self.probes if p.endpoint.url in known and p.endpoint.url not in draining
                and p.endpoint.url not in held and p.endpoint.serves(fingerprint)]

    def select(self, fingerprint: str | None, current: Endpoint | None = None) -> Endpoint:
        """The endpoint a model (re)starting now should use."""
        with self._lock:
            if not self.probes or self.probe_due():
                self.reprobe()
            got = choose(self._usable(fingerprint), fingerprint=fingerprint, current=current)
            if got is None:  # nothing healthy: the failover order (unanswered ones), else the enrolled endpoint
                held = {u for u, t in self._failed.items() if self.mono() - t < self.failure_hold_s}
                order = failover_order(self.probes, fingerprint=fingerprint, exclude=held)
                got = order[0] if order else self.enrolled
            return replace(got)

    def preferred_switch(self, fingerprint: str | None, current: Endpoint) -> Endpoint | None:
        """Another endpoint an *idle* model should move to (the current one failed, drains, left the directory, or
        another is faster by the hysteresis margin); ``None`` = stay."""
        with self._lock:
            if not self.probes:
                return None
            ok = self._usable(fingerprint)
            if not any(p.endpoint.url == current.url for p in ok):  # current failed / drains / gone / not serving
                nxt = choose(ok, fingerprint=fingerprint)
                return nxt if nxt is not None and nxt.url != current.url else None
            best = choose(ok, fingerprint=fingerprint, current=current)
            return best if best is not None and best.url != current.url else None

    def report_failure(self, url: str) -> None:
        """A connection to ``url`` failed: hold it out of the choice for ``failure_hold_s`` (or until a healthy
        probe) and refresh the directory on the next maintenance."""
        with self._lock:
            self._failed[url] = self.mono()
            self._refreshed_at = -1e18
            self._probed_at = -1e18

    def rtt_ms(self, url: str) -> float | None:
        for p in self.probes:
            if p.endpoint.url == url and p.healthy:
                return p.rtt_ms
        return None

    def demand_record(self, model: str, url: str) -> dict[str, Any] | None:
        """The content-free demand report of one model on one endpoint: the model's public name, the endpoint's
        region label and the coarse RTT bucket measured by the payload-free probes. ``None`` without a probe."""
        rtt = self.rtt_ms(url)
        if rtt is None:
            return None
        region = next((e.region for e in self.known() if e.url == url), "enrolled")
        return {"v": 1, "model": model, "endpoint": region, "rtt_bucket": rtt_bucket(rtt)}

    def status(self) -> dict[str, Any]:
        with self._lock:
            d = self._valid()
            return {"directory": d.public() if d is not None else None, "accepted_version": self.versions.get()[0],
                    "probes": [{"region": p.endpoint.region, "url": p.endpoint.url,
                                "rtt_ms": None if p.rtt_ms == float("inf") else round(p.rtt_ms, 2)}
                               for p in self.probes],
                    "switches": self.switches, "errors": list(self.errors)}
