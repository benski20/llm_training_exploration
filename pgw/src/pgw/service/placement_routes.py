"""Placement routes of the compute service (docs/operations/placement.md); each one off unless configured.

    GET  /pgw/v1/endpoints/directory   the signed endpoint directory (``endpoint_directory_file``), verbatim, to
                                       authenticated clients. The service does not sign or parse it: clients verify
                                       it with the key pinned in their SDK key (pgw.sdk.directory).
    POST /pgw/v1/placement/report      a content-free demand report (``demand_reports``; strict schema of
                                       pgw.placement.telemetry); the source address -> coarse region, then dropped.
    GET  /pgw/v1/placement/aggregates  the closed hours' aggregates (k-suppressed) for the placement controller
                                       (bearer token whose SHA-256 is in ``demand_export_token_sha256``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
from pathlib import Path
from typing import Any

__all__ = ["DIRECTORY_PATH", "REPORT_PATH", "add_placement_routes"]

log = logging.getLogger(__name__)

DIRECTORY_PATH = "/pgw/v1/endpoints/directory"
REPORT_PATH = "/pgw/v1/placement/report"
AGGREGATES_PATH = "/pgw/v1/placement/aggregates"
MAX_REPORT_BYTES = 4096


class _FileCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._key: tuple[int, int] | None = None
        self._data = b""
        self._lock = threading.Lock()

    def read(self) -> bytes | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        with self._lock:
            if self._key != (st.st_mtime_ns, st.st_size):
                from pgw.sdk.directory import MAX_DIRECTORY_BYTES

                if st.st_size > MAX_DIRECTORY_BYTES:
                    log.error("endpoint directory %s is larger than %d bytes; not served", self.path,
                              MAX_DIRECTORY_BYTES)
                    return None
                self._data, self._key = self.path.read_bytes(), (st.st_mtime_ns, st.st_size)
            return self._data


def add_placement_routes(app: Any, state: Any) -> Any:
    """Mount whichever placement routes the settings enable. Returns the demand aggregator (or ``None``)."""
    from starlette.requests import Request
    from starlette.responses import Response

    from pgw.wire.server import _client_cert_subject, _maybe_await, error_headers, error_response_bytes

    s = state.settings
    handler = state.handler

    def _error(exc: Exception) -> Response:
        from pgw.core.errors import PGWError

        if not isinstance(exc, PGWError):
            log.error("placement route internal error", exc_info=exc)
        status, body = error_response_bytes(exc, None, json_body=True)
        return Response(content=body, status_code=status, media_type="application/json", headers=error_headers(exc))

    async def _auth(request: Request) -> Any:
        return await _maybe_await(handler.authenticate(request.headers.get("authorization"),
                                                       _client_cert_subject(request.scope)))

    if s.endpoint_directory_file is not None:
        cache = _FileCache(s.endpoint_directory_file)

        async def directory(request: Request) -> Response:
            try:
                await _auth(request)
                data = cache.read()
                if data is None:
                    return Response(content=b'{"error":{"code":"not_found","message":"no endpoint directory"}}',
                                    status_code=404, media_type="application/json")
                return Response(content=data, media_type="application/json", headers={"Cache-Control": "no-store"})
            except Exception as exc:
                return _error(exc)

        app.add_route(DIRECTORY_PATH, directory, methods=["GET"], include_in_schema=False)

    agg = None
    if s.demand_reports:
        from pgw.placement.telemetry import Aggregator, RegionResolver, ReportError

        table: dict[str, str] = {}
        if s.demand_regions_file is not None:
            table = dict(json.loads(Path(s.demand_regions_file).read_text(encoding="utf-8")))
        def served() -> frozenset[str]:  # the packs load after start-up: read the names when a report comes
            return frozenset(str(m.name) for m in state.registry.models()) if state.registry is not None else \
                frozenset()

        agg = Aggregator(models=served(), resolver=RegionResolver(table), k_min=s.demand_k_min,
                         retention_days=s.demand_retention_days)

        async def report(request: Request) -> Response:
            try:
                ident = await _auth(request)
                from pgw.core.http_body import read_limited

                body = await read_limited(request, MAX_REPORT_BYTES, what="placement report")
                agg.models = served()
                try:
                    agg.add(json.loads(body), tenant=str(ident.tenant_id),
                            source_ip=request.client.host if request.client else None)
                except (ValueError, ReportError) as exc:
                    from pgw.core.errors import ErrorCode, PGWError

                    raise PGWError(f"invalid demand report: {exc}", code=ErrorCode.PROTOCOL) from None
                return Response(status_code=204)
            except Exception as exc:
                return _error(exc)

        app.add_route(REPORT_PATH, report, methods=["POST"], include_in_schema=False)

        digests = [bytes.fromhex(h) for h in (s.demand_export_token_sha256 or [])]
        if digests:
            async def aggregates(request: Request) -> Response:
                auth = request.headers.get("authorization") or ""
                tok = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
                d = hashlib.sha256(tok.encode()).digest()
                if not tok or not any(hmac.compare_digest(d, x) for x in digests):
                    return Response(status_code=401)
                return Response(content=json.dumps(agg.export()), media_type="application/json",
                                headers={"Cache-Control": "no-store"})

            app.add_route(AGGREGATES_PATH, aggregates, methods=["GET"], include_in_schema=False)
    return agg
