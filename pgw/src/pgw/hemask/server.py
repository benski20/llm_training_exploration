"""Service side of ``POST /pgw/v1/maskgen`` (mounted by :mod:`pgw.service.app`).

The handler must implement ``authenticate(authorization, client_cert_subject)`` (optional),
``precheck()`` (optional) and ``async maskgen(request, identity) -> MaskgenResponse``. Errors use the
same msgpack error bodies and HTTP statuses as the compute path.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pgw.core.errors import ErrorCode, PGWError
from pgw.hemask.wire import MASKGEN_PATH, decode_maskgen_request, encode_maskgen_response
from pgw.wire.codec import CONTENT_TYPE
from pgw.wire.messages import PROTOCOL_HEADER, REQUEST_ID_HEADER
from pgw.wire.server import (
    ByteBudget,
    _client_cert_subject,
    _maybe_await,
    _read_body,
    check_content_type,
    check_protocol_header,
    error_headers,
    error_response_bytes,
    response_headers,
)

__all__ = ["DEFAULT_MAX_MASKGEN_BYTES", "add_maskgen_route"]

log = logging.getLogger(__name__)

#: Largest maskgen request body (one 7B down projection group is 18944 x 256 x 8 B = 38.8 MB).
DEFAULT_MAX_MASKGEN_BYTES = 512 << 20


def add_maskgen_route(app: Any, handler: Any, *, max_request_bytes: int = DEFAULT_MAX_MASKGEN_BYTES,
                      body_budget: ByteBudget | None = None) -> None:
    """Mount ``POST /pgw/v1/maskgen`` on a Starlette/FastAPI ``app`` (or router) for ``handler``."""
    from starlette.requests import Request
    from starlette.responses import Response

    has_auth = callable(getattr(handler, "authenticate", None))
    precheck = getattr(handler, "precheck", None)

    async def maskgen(request: Request) -> Response:
        t0 = time.monotonic()
        held = [0]
        rid: str | None = None
        try:
            check_protocol_header(request.headers.get(PROTOCOL_HEADER))
            check_content_type(request.headers.get("content-type"))
            identity = None
            if has_auth:
                identity = await _maybe_await(handler.authenticate(
                    request.headers.get("authorization"), _client_cert_subject(request.scope)))
            if callable(precheck):
                await _maybe_await(precheck())
            body = await _read_body(request, max_request_bytes, body_budget, held)
            req = decode_maskgen_request(body)
            rid = req.request_id
            resp = await handler.maskgen(req, identity)
            out = encode_maskgen_response(resp)
            log.debug("pgw maskgen ok", extra={"request_id": rid, "items": len(req.items), "bytes_in": len(body),
                                               "bytes_out": len(out), "ms": round((time.monotonic() - t0) * 1e3, 1)})
            return Response(content=out, media_type=CONTENT_TYPE, headers=response_headers(rid))
        except Exception as exc:
            if isinstance(exc, PGWError) and exc.code != ErrorCode.INTERNAL:
                log.warning("pgw maskgen rejected", extra={"request_id": rid, "code": str(exc.code)})
            else:
                log.error("pgw maskgen internal error", extra={"request_id": rid}, exc_info=exc)
            status, body_out = error_response_bytes(exc, rid)
            headers = error_headers(exc)
            if rid:
                headers[REQUEST_ID_HEADER] = rid
            return Response(content=body_out, status_code=status, media_type=CONTENT_TYPE, headers=headers)
        finally:
            if body_budget is not None and held[0]:
                body_budget.release(held[0])

    app.add_route(MASKGEN_PATH, maskgen, methods=["POST"], include_in_schema=False)
