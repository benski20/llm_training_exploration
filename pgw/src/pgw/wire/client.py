"""Async HTTP/2 client for the compute service (pooled, mTLS, retries, deadlines).

The client is used by :class:`pgw.engine.MaskedBackend` (``await client.compute(req)`` or the
blocking ``client.compute_sync(req)``). Retrying is always safe: a compute request is a pure
function of its body, and a retry resends the *identical* ``x_hat`` bytes under the same request id
(``PGW-Request-Id``), so the service learns nothing new and no extra mask is consumed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import random
import ssl
import threading
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from pgw.core.errors import (
    ConfigError,
    ErrorCode,
    FingerprintMismatchError,
    PayloadTooLargeError,
    PGWError,
    ServiceUnavailableError,
    ShapeError,
    VersionMismatchError,
)
from pgw.core.ring import RingParams
from pgw.wire.codec import (
    CONTENT_TYPE,
    JSON_CONTENT_TYPE,
    decode_error_json,
    decode_message,
    encode_message,
)
from pgw.wire.messages import (
    ATTEMPT_HEADER,
    COMPUTE_PATH,
    MODELS_PATH,
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    REQUEST_ID_HEADER,
    ComputeItem,
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    ErrorResponse,
    ModelInfo,
    ModelsResponse,
    ServerTiming,
)
from pgw.wire.server import validate_compute_response

__all__ = ["ClientConfig", "ClientStats", "ComputeClient", "build_ssl_context"]

log = logging.getLogger(__name__)

T = TypeVar("T")

# msgpack framing overhead budget used when splitting requests by size.
_REQUEST_OVERHEAD = 1024
_ITEM_OVERHEAD = 96


class ClientConfig(BaseModel):
    """Connection settings for one compute service endpoint.

    Embeddable in pydantic-settings classes (e.g. ``PGW_GATEWAY_SERVICE__BASE_URL``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(description="e.g. https://compute.example.net:8443")
    token: SecretStr | None = Field(None, description="Bearer token (tenant credential)")
    ca_file: Path | None = Field(None, description="CA bundle to verify the service (default: system store)")
    client_cert_file: Path | None = Field(None, description="mTLS client certificate (PEM)")
    client_key_file: Path | None = Field(None, description="mTLS client private key (PEM)")
    client_key_password: SecretStr | None = None
    verify_tls: bool = Field(True, description="Verify the server certificate (never disable in production)")
    http2: bool = True
    connect_timeout_s: float = Field(5.0, gt=0)
    request_timeout_s: float = Field(60.0, gt=0, description="Overall budget per compute() incl. retries")
    pool_timeout_s: float = Field(10.0, gt=0)
    max_connections: int = Field(16, gt=0)
    max_keepalive_connections: int = Field(8, ge=0)
    keepalive_expiry_s: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0)
    backoff_base_s: float = Field(0.05, ge=0)
    backoff_max_s: float = Field(2.0, ge=0)
    max_retry_after_s: float = Field(10.0, ge=0, description="Cap on honoured Retry-After")
    max_request_bytes: int = Field(256 << 20, gt=4096)
    max_response_bytes: int = Field(1 << 30, gt=4096)
    max_items: int = Field(256, gt=0)
    max_rows_per_item: int = Field(8192, gt=0)
    max_concurrency: int = Field(8, gt=0, description="Parallel sub-requests when a request is split")
    user_agent: str = "pgw-wire/1"

    @field_validator("base_url")
    @classmethod
    def _url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def _pair(self) -> Self:
        if (self.client_cert_file is None) != (self.client_key_file is None):
            raise ValueError("client_cert_file and client_key_file must be set together")
        return self

    @property
    def uses_tls(self) -> bool:
        """True for ``https://`` endpoints."""
        return self.base_url.startswith("https://")


def build_ssl_context(config: ClientConfig) -> ssl.SSLContext:
    """TLS context for ``config``: TLS >= 1.2, server verification against ``ca_file`` (or the
    system store), and the mTLS client certificate when configured.

    Raises:
        ConfigError: unreadable certificate/key files.
    """
    try:
        if config.ca_file is not None:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(config.ca_file))
        else:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        if not config.verify_tls:
            log.warning("TLS server verification is DISABLED for %s (development only)", config.base_url)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        if config.client_cert_file is not None and config.client_key_file is not None:
            pw = config.client_key_password.get_secret_value() if config.client_key_password else None
            ctx.load_cert_chain(str(config.client_cert_file), str(config.client_key_file), password=pw)
    except (OSError, ssl.SSLError) as exc:
        raise ConfigError(f"cannot load TLS material: {type(exc).__name__}: {exc}") from exc
    return ctx


@dataclass
class ClientStats:
    """Counters (cumulative, process-local). Payloads are never recorded."""

    requests: int = 0
    attempts: int = 0
    retries: int = 0
    failures: int = 0
    splits: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_rtt_s: float = 0.0


_STATUS_CODES: dict[int, ErrorCode] = {
    400: ErrorCode.PROTOCOL,
    401: ErrorCode.UNAUTHENTICATED,
    403: ErrorCode.PERMISSION_DENIED,
    404: ErrorCode.MODEL_NOT_FOUND,
    409: ErrorCode.FINGERPRINT_MISMATCH,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    426: ErrorCode.VERSION_MISMATCH,
    429: ErrorCode.RATE_LIMITED,
    502: ErrorCode.UNAVAILABLE,
    503: ErrorCode.UNAVAILABLE,
    504: ErrorCode.DEADLINE_EXCEEDED,
}


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        v = float(value.strip())
    except ValueError:
        return None  # HTTP-date form is not used by PGW services
    return v if math.isfinite(v) and v >= 0 else None


@dataclass
class _Piece:
    """A row range of one original item, placed in a sub-request."""

    item_index: int
    row0: int
    rows: int


class ComputeClient:
    """Client used by :class:`pgw.engine.MaskedBackend`.

    * Retries only retryable errors (``unavailable``, ``rate_limited``, ``deadline_exceeded``,
      ``backend_unavailable``, transport errors) with jittered exponential backoff, honouring
      ``Retry-After``, within the request deadline. Retrying resends the same ``x_hat``
      (the server already saw it, so no extra leakage, and no extra mask is consumed).
    * Splits requests above ``max_request_bytes`` / ``max_items`` / ``max_rows_per_item`` (the
      smaller of the config and the server's advertised limits) into several, sends them
      concurrently, and reassembles the results in the original item order.
    * Verifies on :meth:`connect` that the server serves the expected fingerprint and ring.
    * Validates every response (id, item order, linear, rows, ``y_hat`` length).
    * Never logs payload bytes or credentials.

    Args:
        config: Endpoint settings.
        transport: Optional httpx transport (tests: ``httpx.ASGITransport`` / ``MockTransport``).
        on_attempt: Optional hook ``fn(path, status_or_None, seconds, error_code_or_None)`` called
            after every HTTP attempt (metrics).
        sleep: Injectable async sleep (tests).
    """

    def __init__(
        self,
        config: ClientConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        on_attempt: Callable[[str, int | None, float, str | None], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ):
        self.config = config
        self.stats = ClientStats()
        self._transport = transport
        self._on_attempt = on_attempt
        self._sleep = sleep or asyncio.sleep
        self._ssl: ssl.SSLContext | None = build_ssl_context(config) if config.uses_tls else None
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )
        self._clients_lock = threading.Lock()
        self._model: ModelInfo | None = None
        self._models: ModelsResponse | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self._closed = False

    # -------------------------------------------------------------------------------- plumbing
    def _http(self) -> httpx.AsyncClient:
        """The pooled ``AsyncClient`` of the running event loop (one pool per loop)."""
        if self._closed:
            raise ServiceUnavailableError("compute client is closed")
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            c = self._clients.get(loop)
            if c is None or c.is_closed:
                cfg = self.config
                headers = {
                    "User-Agent": cfg.user_agent,
                    PROTOCOL_HEADER: str(PROTOCOL_VERSION),
                    "Accept": CONTENT_TYPE,
                }
                if cfg.token is not None:
                    headers["Authorization"] = f"Bearer {cfg.token.get_secret_value()}"
                kwargs: dict[str, Any] = {
                    "base_url": cfg.base_url,
                    "headers": headers,
                    "timeout": httpx.Timeout(
                        cfg.request_timeout_s, connect=cfg.connect_timeout_s, pool=cfg.pool_timeout_s
                    ),
                    "limits": httpx.Limits(
                        max_connections=cfg.max_connections,
                        max_keepalive_connections=cfg.max_keepalive_connections,
                        keepalive_expiry=cfg.keepalive_expiry_s,
                    ),
                    "follow_redirects": False,
                    "trust_env": False,
                }
                if self._transport is not None:
                    kwargs["transport"] = self._transport
                else:
                    kwargs["http2"] = cfg.http2
                    kwargs["verify"] = self._ssl if self._ssl is not None else True
                c = httpx.AsyncClient(**kwargs)
                self._clients[loop] = c
            return c

    def _effective_limits(self) -> tuple[int, int, int]:
        cfg = self.config
        max_bytes, max_rows = cfg.max_request_bytes, cfg.max_rows_per_item
        if self._model is not None:
            max_bytes = min(max_bytes, self._model.max_request_bytes)
            max_rows = min(max_rows, self._model.max_rows_per_item)
        return max_bytes, cfg.max_items, max_rows

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        cfg = self.config
        delay = random.uniform(0.0, min(cfg.backoff_max_s, cfg.backoff_base_s * (2 ** (attempt - 1))))  # noqa: S311
        if retry_after is not None:
            delay = max(delay, min(retry_after, cfg.max_retry_after_s))
        return delay

    @staticmethod
    def _error_from_http(resp: httpx.Response) -> PGWError:
        ctype = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        err: ErrorResponse | None = None
        try:
            if ctype == CONTENT_TYPE:
                err = decode_message(resp.content, ErrorResponse, max_bytes=1 << 20)
            elif ctype == JSON_CONTENT_TYPE:
                err = decode_error_json(resp.content)
        except PGWError:
            err = None
        if err is not None:
            exc = err.to_exception()
        else:
            code = _STATUS_CODES.get(resp.status_code)
            if code is None:
                code = ErrorCode.UNAVAILABLE if resp.status_code >= 500 else ErrorCode.PROTOCOL
            if resp.status_code == 500:
                code = ErrorCode.INTERNAL
            exc = PGWError(
                f"compute service returned HTTP {resp.status_code}",
                code=code,
                details={"http_status": resp.status_code},
            )
        ra = _parse_retry_after(resp.headers.get("retry-after"))
        if ra is not None:
            exc.details.setdefault("retry_after_s", ra)
        return exc

    async def _call(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        request_id: str,
        deadline: float,
        decode: Callable[[bytes], T],
    ) -> T:
        """One logical call with retries until success, a non-retryable error or the deadline."""
        cfg = self.config
        http = self._http()
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PGWError(
                    "compute deadline exceeded",
                    code=ErrorCode.DEADLINE_EXCEEDED,
                    details={"request_id": request_id, "attempts": attempt - 1},
                )
            headers = {REQUEST_ID_HEADER: request_id, ATTEMPT_HEADER: str(attempt)}
            if body is not None:
                headers["Content-Type"] = CONTENT_TYPE
            timeout = httpx.Timeout(
                remaining, connect=min(cfg.connect_timeout_s, remaining), pool=min(cfg.pool_timeout_s, remaining)
            )
            self.stats.attempts += 1
            t0 = time.monotonic()
            status: int | None = None
            exc: PGWError
            try:
                # asyncio.timeout also bounds transports that ignore httpx timeouts (e.g. in-process).
                async with asyncio.timeout(remaining):
                    resp = await http.request(method, path, content=body, headers=headers, timeout=timeout)
                status = resp.status_code
                if body is not None:
                    self.stats.bytes_sent += len(body)
                self.stats.bytes_received += len(resp.content)
                self.stats.last_rtt_s = time.monotonic() - t0
                if status == 200:
                    result = decode(resp.content)
                    self._hook(path, status, t0, None)
                    return result
                exc = self._error_from_http(resp)
            except (httpx.TimeoutException, TimeoutError) as e:
                exc = PGWError(
                    f"compute service timed out ({type(e).__name__})",
                    code=ErrorCode.DEADLINE_EXCEEDED,
                    details={"request_id": request_id},
                )
            except httpx.TransportError as e:
                exc = ServiceUnavailableError(
                    f"compute service transport error ({type(e).__name__})", details={"request_id": request_id}
                )
            except PGWError as e:  # decoding a 200 body failed: not retryable
                self._hook(path, status, t0, str(e.code))
                raise
            self._hook(path, status, t0, str(exc.code))
            retry_after = exc.details.get("retry_after_s")
            ra = float(retry_after) if isinstance(retry_after, int | float) else None
            if exc.retryable and attempt <= cfg.max_retries:
                delay = self._backoff(attempt, ra)
                if time.monotonic() + delay < deadline:
                    self.stats.retries += 1
                    log.info(
                        "pgw wire retry",
                        extra={
                            "request_id": request_id,
                            "attempt": attempt,
                            "code": str(exc.code),
                            "delay_s": round(delay, 4),
                        },
                    )
                    await self._sleep(delay)
                    continue
            exc.details.setdefault("attempts", attempt)
            raise exc

    def _hook(self, path: str, status: int | None, t0: float, code: str | None) -> None:
        if self._on_attempt is not None:
            try:
                self._on_attempt(path, status, time.monotonic() - t0, code)
            except Exception:
                log.exception("on_attempt hook failed")

    # -------------------------------------------------------------------------------- API
    async def connect(self, expected_fingerprint: str | None = None, ring: str | None = None) -> ModelsResponse:
        """Open the pool, fetch ``/pgw/v1/models`` and check fingerprint/ring/version.

        Raises:
            VersionMismatchError: the server does not speak protocol version 1.
            FingerprintMismatchError: ``expected_fingerprint`` is not served.
            ConfigError: ``ring`` is not served for that model.
        """
        models = await self.models()
        if PROTOCOL_VERSION not in models.versions:
            raise VersionMismatchError(
                f"service does not support protocol v{PROTOCOL_VERSION}", details={"versions": list(models.versions)}
            )
        info: ModelInfo | None
        if expected_fingerprint is not None:
            info = models.find(expected_fingerprint)
            if info is None:
                raise FingerprintMismatchError(
                    f"service does not serve model fingerprint {expected_fingerprint[:12]}",
                    details={"expected": expected_fingerprint, "served": [m.fingerprint for m in models.models]},
                )
        else:
            info = models.models[0] if len(models.models) == 1 else None
        if ring is not None:
            rname = RingParams.from_name(ring).name
            candidates = [info] if info is not None else list(models.models)
            if not any(rname in m.rings for m in candidates):
                raise ConfigError(f"service does not serve ring {rname} for this model", details={"ring": rname})
        self._model = info
        log.info(
            "pgw wire connected",
            extra={
                "server": models.server,
                "models": len(models.models),
                "fp": info.fingerprint[:12] if info else None,
            },
        )
        return models

    async def models(self) -> ModelsResponse:
        """``GET /pgw/v1/models`` (with retries)."""
        deadline = time.monotonic() + self.config.request_timeout_s
        res = await self._call(
            "GET",
            MODELS_PATH,
            body=None,
            request_id=uuid.uuid4().hex,
            deadline=deadline,
            decode=lambda b: decode_message(b, ModelsResponse, max_bytes=16 << 20),
        )
        self._models = res
        return res

    async def ready(self) -> bool:
        """``GET /readyz`` (single attempt): True on HTTP 200."""
        try:
            r = await self._http().get("/readyz", timeout=self.config.connect_timeout_s)
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    @property
    def model(self) -> ModelInfo | None:
        """The model selected by :meth:`connect` (``None`` before)."""
        return self._model

    async def compute(self, request: ComputeRequest) -> ComputeResponse:
        """Send one compute request (with retries); raises typed :class:`pgw.core.PGWError` on failure.

        Requests above the limits are split by items and rows, sent concurrently (at most
        ``max_concurrency`` at a time) and reassembled transparently.
        """
        self.stats.requests += 1
        budget = request.deadline_ms / 1000.0 if request.deadline_ms is not None else self.config.request_timeout_s
        deadline = time.monotonic() + budget
        try:
            parts = self._split(request)
            if len(parts) == 1 and parts[0][0] is request:
                return await self._send_one(request, deadline)
            self.stats.splits += 1
            sem = asyncio.Semaphore(self.config.max_concurrency)

            async def run(sub: ComputeRequest) -> ComputeResponse:
                async with sem:
                    return await self._send_one(sub, deadline)

            tasks = [asyncio.create_task(run(sub)) for sub, _ in parts]
            try:
                responses = await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return self._assemble(request, [(r, pieces) for r, (_, pieces) in zip(responses, parts, strict=True)])
        except PGWError:
            self.stats.failures += 1
            raise

    async def _send_one(self, req: ComputeRequest, deadline: float) -> ComputeResponse:
        body = encode_message(req)
        out_features = None
        if self._model is not None and self._model.fingerprint == req.fingerprint:
            out_features = {k: m for k, (m, _) in self._model.linear_dims().items()}

        def decode(b: bytes) -> ComputeResponse:
            resp = decode_message(b, ComputeResponse, max_bytes=self.config.max_response_bytes)
            validate_compute_response(req, resp, out_features)
            return resp

        t0 = time.monotonic()
        resp = await self._call(
            "POST", COMPUTE_PATH, body=body, request_id=req.request_id, deadline=deadline, decode=decode
        )
        log.debug(
            "pgw compute",
            extra={
                "request_id": req.request_id,
                "items": len(req.items),
                "rows": req.total_rows,
                "bytes_up": len(body),
                "bytes_down": resp.payload_bytes,
                "ms": round((time.monotonic() - t0) * 1e3, 3),
            },
        )
        return resp

    def _split(self, req: ComputeRequest) -> list[tuple[ComputeRequest, list[_Piece]]]:
        """Partition ``req`` into sub-requests within the limits (or ``[(req, ...)]`` if it fits)."""
        max_bytes, max_items, max_rows = self._effective_limits()
        budget = max_bytes - _REQUEST_OVERHEAD - len(req.request_id) - len(req.ring)
        fits = (
            len(req.items) <= max_items
            and all(it.rows <= max_rows for it in req.items)
            and req.payload_bytes + _ITEM_OVERHEAD * len(req.items) <= budget
        )
        if fits:
            return [(req, [_Piece(i, 0, it.rows) for i, it in enumerate(req.items)])]
        groups: list[list[_Piece]] = [[]]
        used = 0
        for i, it in enumerate(req.items):
            if it.rows <= 0 or len(it.x_hat) % it.rows:
                raise ShapeError(f"items[{i}]: x_hat length {len(it.x_hat)} is not a multiple of rows={it.rows}")
            row_bytes = len(it.x_hat) // it.rows
            cap_rows = min(max_rows, (budget - _ITEM_OVERHEAD) // max(row_bytes, 1))
            if cap_rows < 1:
                raise PayloadTooLargeError(
                    f"items[{i}]: one row ({row_bytes} bytes) exceeds the request size limit {max_bytes}",
                    details={"row_bytes": row_bytes, "max_request_bytes": max_bytes},
                )
            row = 0
            while row < it.rows:
                room = budget - used - _ITEM_OVERHEAD
                fit_rows = min(cap_rows, it.rows - row, room // max(row_bytes, 1)) if room > 0 else 0
                if fit_rows < 1 or len(groups[-1]) >= max_items:
                    groups.append([])
                    used = 0
                    continue
                groups[-1].append(_Piece(i, row, fit_rows))
                used += fit_rows * row_bytes + _ITEM_OVERHEAD
                row += fit_rows
        groups = [g for g in groups if g]
        base_id = req.request_id[:100]
        out = []
        for gi, pieces in enumerate(groups):
            items = []
            for p in pieces:
                it = req.items[p.item_index]
                rb = len(it.x_hat) // it.rows
                items.append(ComputeItem(it.linear, p.rows, it.x_hat[p.row0 * rb : (p.row0 + p.rows) * rb]))
            sub = ComputeRequest(
                request_id=f"{base_id}.{gi}",
                fingerprint=req.fingerprint,
                ring=req.ring,
                items=tuple(items),
                deadline_ms=req.deadline_ms,
                version=req.version,
            )
            out.append((sub, pieces))
        log.debug("pgw compute split", extra={"request_id": req.request_id, "parts": len(out)})
        return out

    @staticmethod
    def _assemble(req: ComputeRequest, parts: list[tuple[ComputeResponse, list[_Piece]]]) -> ComputeResponse:
        chunks: list[list[bytes]] = [[] for _ in req.items]
        queue_ms = compute_ms = 0.0
        batch_rows = 0
        for resp, pieces in parts:
            for res, p in zip(resp.items, pieces, strict=True):
                chunks[p.item_index].append(res.y_hat)
            queue_ms = max(queue_ms, resp.timing.queue_ms)
            compute_ms += resp.timing.compute_ms
            batch_rows = max(batch_rows, resp.timing.batch_rows)
        results = tuple(ComputeResult(it.linear, it.rows, b"".join(chunks[i])) for i, it in enumerate(req.items))
        return ComputeResponse(req.request_id, results, ServerTiming(queue_ms, compute_ms, batch_rows))

    # -------------------------------------------------------------------------------- sync API
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                t = threading.Thread(target=loop.run_forever, name="pgw-wire-client", daemon=True)
                t.start()
                self._loop, self._loop_thread = loop, t
            return self._loop

    def _run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        loop = self._ensure_loop()
        if threading.current_thread() is self._loop_thread:
            coro.close()
            raise RuntimeError("compute_sync() called from the client's own event loop thread")
        fut: concurrent.futures.Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()

    def compute_sync(self, request: ComputeRequest) -> ComputeResponse:
        """Blocking wrapper (runs on a private event loop thread)."""
        return self._run_sync(self.compute(request))

    def connect_sync(self, expected_fingerprint: str | None = None, ring: str | None = None) -> ModelsResponse:
        """Blocking :meth:`connect`."""
        return self._run_sync(self.connect(expected_fingerprint, ring))

    async def aclose(self) -> None:
        """Close connections (all pools, including the private sync loop's)."""
        self._closed = True
        with self._clients_lock:
            items = list(self._clients.items())
            self._clients.clear()
        current = asyncio.get_running_loop()
        for loop, c in items:
            if loop is current:
                await c.aclose()
            elif not loop.is_closed() and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(c.aclose(), loop)
                try:
                    await asyncio.wait_for(asyncio.wrap_future(fut), timeout=5)
                except (TimeoutError, RuntimeError):  # pragma: no cover - best effort
                    log.warning("could not close a connection pool cleanly")
        if threading.current_thread() is not self._loop_thread:
            self._stop_loop()

    def close(self) -> None:
        """Synchronous close (call when no event loop is running in this thread).

        Closes the private loop's pool and stops its thread. Pools opened on other event loops
        are dropped; prefer :meth:`aclose` from async code.
        """
        self._closed = True
        loop = self._loop
        if loop is not None and not loop.is_closed() and loop.is_running():
            with self._clients_lock:
                c = self._clients.pop(loop, None)
            if c is not None:
                try:
                    asyncio.run_coroutine_threadsafe(c.aclose(), loop).result(timeout=5)
                except Exception:  # pragma: no cover - best effort
                    log.warning("could not close the sync connection pool cleanly")
        self._stop_loop()

    def _stop_loop(self) -> None:
        with self._loop_lock:
            loop, t = self._loop, self._loop_thread
            self._loop = self._loop_thread = None
        if loop is not None and t is not None:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            if not loop.is_running() and not loop.is_closed():
                loop.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
