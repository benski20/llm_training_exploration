"""Server-side helpers shared by the compute service.

Two layers:

* Framework-agnostic functions (:func:`check_protocol_header`, :func:`decode_compute_request`,
  :func:`validate_compute_response`, :func:`error_response_bytes`, ...) that enforce SPEC.md 6.3-6.6
  before any compute happens.
* :func:`create_router`, a FastAPI ``APIRouter`` implementing ``POST /pgw/v1/compute`` and
  ``GET /pgw/v1/models`` on top of a :class:`ComputeHandler` (the service's registry + batcher).
  The service mounts it with ``app.include_router(create_router(handler, limits=...))`` and adds
  its own ``/healthz``, ``/readyz`` and ``/metrics``. FastAPI is imported lazily, so the gateway
  side of :mod:`pgw.wire` does not need it.

The router never logs or echoes payload bytes; it logs ids, sizes, timings and error codes only.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pgw.core.errors import (
    HTTP_STATUS,
    ErrorCode,
    PayloadTooLargeError,
    PGWError,
    ProtocolError,
    ServiceUnavailableError,
    ShapeError,
    VersionMismatchError,
)
from pgw.core.ring import RingParams
from pgw.wire.codec import (
    CONTENT_TYPE,
    JSON_CONTENT_TYPE,
    decode_message,
    encode_error_json,
    encode_message,
    validate_request_id,
)
from pgw.wire.messages import (
    COMPUTE_PATH,
    DEADLINE_HEADER,
    MODELS_PATH,
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    REQUEST_ID_HEADER,
    SUPPORTED_VERSIONS,
    ComputeRequest,
    ComputeResponse,
    ErrorResponse,
    ModelsResponse,
)

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import APIRouter

__all__ = [
    "ByteBudget",
    "ComputeHandler",
    "LinearDims",
    "RequestContext",
    "WireLimits",
    "check_content_type",
    "check_protocol_header",
    "create_router",
    "decode_compute_request",
    "error_headers",
    "error_response_bytes",
    "response_headers",
    "validate_compute_response",
]

log = logging.getLogger(__name__)

#: ``{linear_id: (out_features m, in_features n)}`` or a resolver ``(fingerprint, ring) -> dims``.
LinearDims = Mapping[str, tuple[int, int]] | Callable[[str, str], Mapping[str, tuple[int, int]]]


@dataclass(frozen=True)
class WireLimits:
    """Request limits enforced before any compute (SPEC 6.5 defaults)."""

    max_request_bytes: int = 256 << 20
    max_items: int = 256
    max_rows_per_item: int = 8192

    def __post_init__(self) -> None:
        if self.max_request_bytes <= 0 or self.max_items <= 0 or self.max_rows_per_item <= 0:
            raise ValueError("WireLimits values must be positive")


# ---------------------------------------------------------------------------------------------
# framework-agnostic helpers
# ---------------------------------------------------------------------------------------------
def check_protocol_header(value: str | None) -> int:
    """Parse the ``PGW-Protocol`` header; raise :class:`VersionMismatchError` if unsupported.

    A missing header is treated as the current version (allows ``curl`` debugging); a header
    that is not an integer is a :class:`ProtocolError`.
    """
    if value is None or value.strip() == "":
        return PROTOCOL_VERSION
    try:
        v = int(value.strip())
    except ValueError as exc:
        raise ProtocolError(f"invalid {PROTOCOL_HEADER} header") from exc
    if v not in SUPPORTED_VERSIONS:
        raise VersionMismatchError(
            f"unsupported protocol version {v}", details={"got": v, "versions": list(SUPPORTED_VERSIONS)}
        )
    return v


def check_content_type(value: str | None) -> None:
    """Require ``application/vnd.pgw+msgpack`` (parameters such as ``; charset`` are ignored)."""
    base = (value or "").split(";", 1)[0].strip().lower()
    if base != CONTENT_TYPE:
        raise ProtocolError(f"unsupported content type {base or '<none>'!r}; expected {CONTENT_TYPE}")


def _resolve_dims(linear_dims: LinearDims, fingerprint: str, ring: str) -> Mapping[str, tuple[int, int]]:
    if callable(linear_dims):
        return linear_dims(fingerprint, ring)
    return linear_dims


def decode_compute_request(
    body: bytes | bytearray | memoryview, limits: WireLimits, linear_dims: LinearDims
) -> ComputeRequest:
    """Decode + validate a compute request against limits and the model's ``{linear: (m, n)}``.

    ``linear_dims`` may also be a resolver ``(fingerprint, ring) -> {linear: (m, n)}`` that raises
    :class:`pgw.core.FingerprintMismatchError` / ``ModelNotFoundError`` for unknown models.
    Payload length must equal ``rows * n * K/8``.

    Raises:
        PayloadTooLargeError: body, item count or rows above ``limits``.
        ProtocolError: malformed message, unknown linear.
        ShapeError: ``x_hat`` length does not match ``rows * n * K/8``.
    """
    req = decode_message(body, ComputeRequest, max_bytes=limits.max_request_bytes)
    try:
        if len(req.items) > limits.max_items:
            raise PayloadTooLargeError(
                f"request has {len(req.items)} items; the limit is {limits.max_items}",
                details={"items": len(req.items), "max_items": limits.max_items},
            )
        dims = _resolve_dims(linear_dims, req.fingerprint, req.ring)
        nb = RingParams.from_name(req.ring).k // 8
        for i, it in enumerate(req.items):
            if it.rows > limits.max_rows_per_item:
                raise PayloadTooLargeError(
                    f"items[{i}] has {it.rows} rows; the limit is {limits.max_rows_per_item}",
                    details={"item": i, "rows": it.rows, "max_rows_per_item": limits.max_rows_per_item},
                )
            mn = dims.get(it.linear)
            if mn is None:
                raise ProtocolError(
                    f"items[{i}]: unknown linear {it.linear!r}", details={"item": i, "linear": it.linear}
                )
            expected = it.rows * int(mn[1]) * nb
            if len(it.x_hat) != expected:
                raise ShapeError(
                    f"items[{i}] ({it.linear}): x_hat has {len(it.x_hat)} bytes, expected rows*n*K/8 = {expected}",
                    details={"item": i, "linear": it.linear, "bytes": len(it.x_hat), "expected": expected},
                )
    except PGWError as exc:
        exc.details.setdefault("request_id", req.request_id)
        raise
    return req


def validate_compute_response(
    request: ComputeRequest, response: ComputeResponse, out_features: Mapping[str, int] | None = None
) -> None:
    """Check that ``response`` answers ``request``: same id, item count, order, linear and rows,
    and (when ``out_features`` is given) ``len(y_hat) == rows * m * K/8``.

    Used by the client on every response and by the router as a self-check before replying.

    Raises:
        ProtocolError: any mismatch.
    """
    if response.request_id != request.request_id:
        raise ProtocolError("compute response id does not match the request")
    if len(response.items) != len(request.items):
        raise ProtocolError(
            f"compute response has {len(response.items)} items, expected {len(request.items)}",
            details={"got": len(response.items), "expected": len(request.items)},
        )
    nb = RingParams.from_name(request.ring).k // 8
    for i, (q, r) in enumerate(zip(request.items, response.items, strict=True)):
        if r.linear != q.linear or int(r.rows) != int(q.rows):
            raise ProtocolError(
                f"compute response items[{i}] does not match the request",
                details={"item": i, "expected": q.linear, "got": r.linear},
            )
        if out_features is not None and q.linear in out_features:
            expected = int(q.rows) * int(out_features[q.linear]) * nb
            if len(r.y_hat) != expected:
                raise ProtocolError(
                    f"compute response items[{i}] ({q.linear}): y_hat has {len(r.y_hat)} bytes, expected {expected}",
                    details={"item": i, "bytes": len(r.y_hat), "expected": expected},
                )
        elif len(r.y_hat) == 0 or len(r.y_hat) % (int(q.rows) * nb) != 0:
            raise ProtocolError(f"compute response items[{i}]: y_hat length is not a multiple of rows*K/8")


def _retry_after_s(exc: PGWError) -> float | None:
    ra = exc.details.get("retry_after_s")
    if isinstance(ra, int | float) and not isinstance(ra, bool) and math.isfinite(ra) and ra >= 0:
        return float(ra)
    if exc.code in (ErrorCode.RATE_LIMITED, ErrorCode.UNAVAILABLE, ErrorCode.BACKEND_UNAVAILABLE):
        return 1.0
    return None


def error_headers(exc: BaseException) -> dict[str, str]:
    """Extra HTTP headers for an error reply (``Retry-After`` for 429/503)."""
    headers = {PROTOCOL_HEADER: str(PROTOCOL_VERSION)}
    if isinstance(exc, PGWError):
        ra = _retry_after_s(exc)
        if ra is not None:
            headers["Retry-After"] = str(max(0, math.ceil(ra)))
    return headers


def response_headers(request_id: str | None = None) -> dict[str, str]:
    """Headers on every successful PGW reply."""
    h = {PROTOCOL_HEADER: str(PROTOCOL_VERSION)}
    if request_id:
        h[REQUEST_ID_HEADER] = request_id
    return h


def error_response_bytes(
    exc: BaseException, request_id: str | None = None, *, json_body: bool = False
) -> tuple[int, bytes]:
    """Map any exception to ``(http_status, msgpack ErrorResponse)``; internal errors are not leaked.

    Args:
        exc: The exception. Non-:class:`PGWError` exceptions become ``internal`` with a generic
            message (the caller should log the traceback).
        request_id: Echoed in the error body when known.
        json_body: Encode as JSON instead of msgpack (for clients that only accept JSON).
    """
    err = ErrorResponse.from_exception(exc, request_id)
    if isinstance(exc, PGWError):
        status = HTTP_STATUS.get(exc.code, 500)
        if exc.code == ErrorCode.INTERNAL:
            err = ErrorResponse(code=str(ErrorCode.INTERNAL), message="internal server error", request_id=request_id)
    else:
        status = 500
    body = encode_error_json(err) if json_body else encode_message(err)
    return status, body


# ---------------------------------------------------------------------------------------------
# handler protocol + FastAPI router
# ---------------------------------------------------------------------------------------------
@dataclass
class RequestContext:
    """Per-request context passed to :meth:`ComputeHandler.compute`.

    Attributes:
        request_id: The request ``id``.
        identity: Whatever :meth:`ComputeHandler.authenticate` returned (``None`` without auth).
        received_at: ``time.monotonic()`` when the request arrived.
        deadline: Absolute ``time.monotonic()`` deadline, or ``None``.
        body_bytes: Size of the request body.
        attempt: Client attempt counter (``PGW-Attempt`` header; 1 when absent).
    """

    request_id: str
    identity: Any = None
    received_at: float = field(default_factory=time.monotonic)
    deadline: float | None = None
    body_bytes: int = 0
    attempt: int = 1

    def remaining_s(self) -> float | None:
        """Seconds left until the deadline (``None`` without a deadline)."""
        return None if self.deadline is None else self.deadline - time.monotonic()


@runtime_checkable
class ComputeHandler(Protocol):
    """What the service implements for :func:`create_router`.

    ``authenticate(authorization, client_cert_subject) -> identity`` is optional; when present it is
    called before the body is read and must raise :class:`pgw.core.AuthError` (or
    ``PGWError(code=PERMISSION_DENIED)``) to reject. Handlers may also implement
    ``on_request_done(ctx, status, code)`` for metrics (optional).
    """

    def linear_dims(self, fingerprint: str, ring: str) -> Mapping[str, tuple[int, int]]:
        """Dims of the model with ``fingerprint``; raise ``FingerprintMismatchError`` /
        ``ModelNotFoundError`` / ``ConfigError`` (ring not served)."""
        ...

    async def compute(self, request: ComputeRequest, ctx: RequestContext) -> ComputeResponse:
        """Evaluate every item; results in request order."""
        ...

    async def models(self) -> ModelsResponse:
        """Served models."""
        ...


def _safe_id(v: Any) -> str | None:
    try:
        return validate_request_id(v)
    except ProtocolError:
        return None


def _wants_json(accept: str | None) -> bool:
    a = (accept or "").lower()
    return "json" in a and "msgpack" not in a


def _client_cert_subject(scope: Mapping[str, Any]) -> str | None:
    ext = scope.get("extensions") or {}
    tls = ext.get("tls") or {}
    for key in ("client_cert_name", "client_cert_subject"):
        v = tls.get(key)
        if isinstance(v, str) and v:
            return v
    return None


async def _maybe_await(v: Any) -> Any:
    if inspect.isawaitable(v):
        return await v
    return v


class ByteBudget:
    """Bytes of request bodies the server buffers at once (fail-fast admission, no queueing).

    Request bodies are read into memory before decoding, then expanded ``8/(K/8)`` times into
    int64. Without a global cap a burst of large requests (long prefills, many connections and
    HTTP/2 streams) buffers tens of GB before any per-request admission runs.
    """

    def __init__(self, total: int):
        if total <= 0:
            raise ValueError("ByteBudget total must be positive")
        self.total = int(total)
        self.used = 0

    def try_acquire(self, n: int) -> bool:
        if self.used + n > self.total:
            return False
        self.used += n
        return True

    def release(self, n: int) -> None:
        self.used = max(0, self.used - n)


def _budget_exhausted(budget: ByteBudget, want: int) -> PGWError:
    return ServiceUnavailableError(
        "server is buffering too many request bytes; retry shortly",
        details={
            "retry_after_s": 1.0, "bytes": want,
            "inflight_bytes": budget.used, "max_inflight_bytes": budget.total,
        },
    )


async def _read_body(request: Any, limit: int, budget: ByteBudget | None = None,
                     held: list[int] | None = None) -> bytearray:
    """Read the body up to ``limit`` bytes, reserving it against ``budget`` *before* buffering.

    With ``Content-Length`` the whole length is reserved up front; without it the reservation
    grows with each chunk. ``held[0]`` accumulates what was reserved (the caller releases it).
    """
    cl = request.headers.get("content-length")
    held = held if held is not None else [0]
    if cl is not None:
        try:
            n = int(cl)
        except ValueError as exc:
            raise ProtocolError("invalid Content-Length") from exc
        if n < 0:
            raise ProtocolError("invalid Content-Length")
        if n > limit:
            raise PayloadTooLargeError(
                f"request body of {n} bytes exceeds the limit of {limit}", details={"bytes": n, "max_bytes": limit}
            )
        if budget is not None:
            if not budget.try_acquire(n):
                raise _budget_exhausted(budget, n)
            held[0] += n
    from starlette.requests import ClientDisconnect

    buf = bytearray()
    try:
        async for chunk in request.stream():
            if cl is None and budget is not None and chunk:
                if not budget.try_acquire(len(chunk)):
                    raise _budget_exhausted(budget, len(buf) + len(chunk))
                held[0] += len(chunk)
            buf += chunk
            if len(buf) > limit:
                raise PayloadTooLargeError(
                    f"request body exceeds the limit of {limit} bytes", details={"max_bytes": limit}
                )
    except ClientDisconnect as exc:
        raise ProtocolError("client disconnected before sending the full request body") from exc
    return buf


def _header_deadline_s(value: str | None) -> float | None:
    """Budget from :data:`DEADLINE_HEADER` (``None`` if absent or malformed)."""
    if value is None:
        return None
    try:
        ms = float(value)
    except ValueError:
        return None
    if ms != ms or ms < 0:  # NaN / negative
        return None
    return ms / 1000.0


def create_router(
    handler: ComputeHandler,
    *,
    limits: WireLimits | None = None,
    validate_responses: bool = True,
    max_deadline_s: float | None = None,
    max_inflight_bytes: int | None = None,
) -> APIRouter:
    """FastAPI router for ``POST /pgw/v1/compute`` and ``GET /pgw/v1/models``.

    Args:
        handler: Service implementation (see :class:`ComputeHandler`).
        limits: Request limits (defaults per SPEC 6.5).
        validate_responses: Self-check every response against the request before replying.
        max_deadline_s: Server-side cap on compute time per request (``None``: only the client's
            ``deadline_ms`` applies).
        max_inflight_bytes: Global cap on request-body bytes buffered at once (``None``: no cap).
            Checked against ``Content-Length`` *before* the body is read.

    Handlers may implement ``precheck()`` (sync): called after authentication and before the body
    is read, it raises to refuse (draining, too many in-flight requests), so refused requests
    never get buffered.

    Returns:
        An ``APIRouter`` to mount with ``app.include_router(...)``.
    """
    from fastapi import APIRouter
    from starlette.requests import Request
    from starlette.responses import Response

    lim = limits or WireLimits()
    router = APIRouter()
    has_auth = callable(getattr(handler, "authenticate", None))
    if not has_auth:
        log.warning(
            "pgw wire router created without an authenticate hook: requests are accepted from any "
            "client that can reach the service (rely on mTLS at the TLS layer, or add authentication)"
        )
    done_hook: Callable[..., Any] | None = getattr(handler, "on_request_done", None)
    precheck: Callable[[], Any] | None = getattr(handler, "precheck", None)
    body_budget = ByteBudget(max_inflight_bytes) if max_inflight_bytes else None

    def _err_response(exc: BaseException, rid: str | None, accept: str | None) -> Response:
        status, body = error_response_bytes(exc, rid, json_body=_wants_json(accept))
        headers = error_headers(exc)
        if rid:
            headers[REQUEST_ID_HEADER] = rid
        media = JSON_CONTENT_TYPE if _wants_json(accept) else CONTENT_TYPE
        return Response(content=body, status_code=status, media_type=media, headers=headers)

    def _log_error(exc: BaseException, rid: str | None, path: str) -> None:
        if isinstance(exc, PGWError) and exc.code != ErrorCode.INTERNAL:
            log.warning("pgw wire %s rejected", path, extra={"request_id": rid, "code": str(exc.code)})
        else:
            log.error("pgw wire %s internal error", path, extra={"request_id": rid}, exc_info=exc)

    async def compute(request: Request) -> Response:
        t0 = time.monotonic()
        header_rid = request.headers.get(REQUEST_ID_HEADER)
        rid: str | None = None
        if header_rid is not None:
            rid = _safe_id(header_rid)
        accept = request.headers.get("accept")
        # 499 ("client closed request") is reported to on_request_done if the request task is
        # cancelled (client went away) before a reply was produced.
        status = 499
        code: str | None = "cancelled"
        ctx: RequestContext | None = None
        held = [0]
        try:
            check_protocol_header(request.headers.get(PROTOCOL_HEADER))
            check_content_type(request.headers.get("content-type"))
            identity = None
            if has_auth:
                identity = await _maybe_await(
                    handler.authenticate(  # type: ignore[attr-defined]
                        request.headers.get("authorization"), _client_cert_subject(request.scope)
                    )
                )
            if callable(precheck):
                await _maybe_await(precheck())
            body = await _read_body(request, lim.max_request_bytes, body_budget, held)
            req = decode_compute_request(body, lim, handler.linear_dims)
            rid = req.request_id
            try:
                attempt = max(1, int(request.headers.get("pgw-attempt", "1")))
            except ValueError:
                attempt = 1
            budget: float | None = None
            if req.deadline_ms is not None:
                budget = req.deadline_ms / 1000.0
            hdr = _header_deadline_s(request.headers.get(DEADLINE_HEADER))
            if hdr is not None:  # what is left of the client's budget for *this* attempt
                budget = hdr if budget is None else min(budget, hdr)
            if max_deadline_s is not None:
                budget = max_deadline_s if budget is None else min(budget, max_deadline_s)
            ctx = RequestContext(
                request_id=req.request_id,
                identity=identity,
                received_at=t0,
                deadline=None if budget is None else t0 + budget,
                body_bytes=len(body),
                attempt=attempt,
            )
            if budget is not None:
                remaining = ctx.remaining_s() or 0.0
                if remaining <= 0:
                    raise PGWError("deadline exceeded before compute", code=ErrorCode.DEADLINE_EXCEEDED)
                try:
                    resp = await asyncio.wait_for(handler.compute(req, ctx), timeout=remaining)
                except TimeoutError as exc:
                    raise PGWError(
                        "deadline exceeded during compute",
                        code=ErrorCode.DEADLINE_EXCEEDED,
                        details={"deadline_ms": req.deadline_ms},
                    ) from exc
            else:
                resp = await handler.compute(req, ctx)
            if validate_responses:
                dims = _resolve_dims(handler.linear_dims, req.fingerprint, req.ring)
                try:
                    validate_compute_response(req, resp, {k: int(v[0]) for k, v in dims.items()})
                except ProtocolError as exc:  # our own bug: never send a malformed result
                    raise PGWError(f"service produced an invalid response: {exc.message}") from exc
            out = encode_message(resp)
            log.debug(
                "pgw compute ok",
                extra={
                    "request_id": rid,
                    "items": len(req.items),
                    "rows": req.total_rows,
                    "bytes_in": len(body),
                    "bytes_out": len(out),
                    "ms": round((time.monotonic() - t0) * 1e3, 3),
                },
            )
            status, code = 200, None
            return Response(content=out, media_type=CONTENT_TYPE, headers=response_headers(rid))
        except Exception as exc:
            if rid is None and isinstance(exc, PGWError):
                rid = _safe_id(exc.details.get("request_id"))
            _log_error(exc, rid, COMPUTE_PATH)
            r = _err_response(exc, rid, accept)
            status = r.status_code
            code = str(exc.code) if isinstance(exc, PGWError) else str(ErrorCode.INTERNAL)
            return r
        finally:
            if body_budget is not None and held[0]:
                body_budget.release(held[0])
            if done_hook is not None:
                try:
                    await _maybe_await(
                        done_hook(ctx or RequestContext(request_id=rid or "", received_at=t0), status, code)
                    )
                except Exception:  # metrics must never break a reply
                    log.exception("on_request_done hook failed")

    async def models(request: Request) -> Response:
        accept = request.headers.get("accept")
        try:
            check_protocol_header(request.headers.get(PROTOCOL_HEADER))
            if has_auth:
                await _maybe_await(
                    handler.authenticate(  # type: ignore[attr-defined]
                        request.headers.get("authorization"), _client_cert_subject(request.scope)
                    )
                )
            resp = await handler.models()
            return Response(content=encode_message(resp), media_type=CONTENT_TYPE, headers=response_headers())
        except Exception as exc:
            _log_error(exc, None, MODELS_PATH)
            return _err_response(exc, None, accept)

    # Plain Starlette routes: the body is raw msgpack, so FastAPI's request parsing / OpenAPI
    # schema generation is bypassed on purpose.
    router.add_route(COMPUTE_PATH, compute, methods=["POST"], include_in_schema=False)
    router.add_route(MODELS_PATH, models, methods=["GET"], include_in_schema=False)
    return router


#: Type of an awaitable compute function (for adapters).
ComputeFn = Callable[[ComputeRequest, RequestContext], Awaitable[ComputeResponse]]
