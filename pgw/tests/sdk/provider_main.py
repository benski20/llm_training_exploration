"""A real ``pgw service`` process for the SDK end-to-end tests, plus a test implementation of
``POST /pgw/v1/sdk/enroll`` (the provider package owns the real one; this follows the payload in
``docs/sdk/CONTRACT.md``).

Run as ``python -m tests.sdk.provider_main``. Service settings come from ``PGW_SERVICE_*``; test knobs:

* ``PGW_TEST_SDK_KEYS``: JSON file ``{key_id_hex: {"secret_sha256": hex, "tenant": str, "revoked": bool}}``,
  re-read on every enrollment request (tests add and revoke keys while the service runs).
* ``PGW_TEST_TAMPER_FILE``: while this file exists, every compute result has one element changed (an active
  provider returning a wrong result; caught by the gateway's checks).
* ``PGW_TEST_LANE_PORT``: the TCP+TLS lane port announced in the enrollment response.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from pathlib import Path
from typing import Any


def _install_tamper(flag: Path) -> None:
    import torch

    from pgw.service.kernels import ComputeKernel

    orig = ComputeKernel.run

    def run(self: Any, x_hat: torch.Tensor, ring: Any) -> torch.Tensor:
        out = orig(self, x_hat, ring)
        if flag.exists() and out.numel():
            out = out.clone()
            out.view(-1)[0] = (int(out.view(-1)[0]) + 12345) % ring.modulus
        return out

    ComputeKernel.run = run  # type: ignore[method-assign]


def _add_sdk_route(app: Any, keys_file: Path, lane_port: int | None) -> None:
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    from pgw.enroll.service import _issue_token, check_csr, sign_client_csr
    from pgw.sdk.keys import parse_key

    def err(status: int, code: str, msg: str) -> JSONResponse:
        return JSONResponse({"error": {"code": code, "message": msg}}, status_code=status)

    async def sdk_enroll(request: Request) -> Any:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return err(401, "unauthenticated", "missing API key")
        try:
            key = parse_key(auth[len("Bearer "):])
        except Exception:
            return err(401, "unauthenticated", "invalid API key")
        try:
            keys = json.loads(keys_file.read_text())
        except (OSError, ValueError):
            keys = {}
        rec = keys.get(key.key_id_hex)
        if (rec is None or rec.get("revoked")
                or not hmac.compare_digest(bytes.fromhex(rec["secret_sha256"]), key.secret_sha256())):
            return err(401, "unauthenticated", "unknown or revoked API key")
        body = await request.json()
        name, csr = body.get("client_name"), body.get("csr_pem")
        svc = app.state.pgw_enroll
        s = svc.settings
        tenant = str(rec.get("tenant", "sdk"))

        def issue() -> dict[str, Any]:
            sign = isinstance(csr, str) and bool(csr) and s.client_issuing_ca_cert_file is not None
            if sign:
                check_csr(csr)
            token, trec = _issue_token(tenant, Path(s.tokens_file), token_only=not sign)
            svc.authenticator.add_token_hash(trec.digest_hex, tenant, token_id=trec.token_id,
                                             token_only=trec.token_only)
            heads = {m.base_fingerprint for m in app.state.pgw.registry.models() if m.base_fingerprint}
            models = [{**e, "masked_head": e["fingerprint"] in heads} for e in svc._model_entries([])]
            out: dict[str, Any] = {"tenant": tenant, "service_token": token, "models": models,
                                   "service_url": s.public_url,
                                   "lane": {"transport": "tcp", "port": lane_port} if lane_port else None}
            if sign:
                cn = f"pgw-sdk.{tenant}.{name}"[:64]
                cert_pem, _ = sign_client_csr(csr, s.client_issuing_ca_cert_file, s.client_issuing_ca_key_file, cn,
                                              s.client_cert_validity_days)
                out["client_cert_pem"] = cert_pem
                svc.authenticator.add_subject(cn, tenant)
            return out

        return JSONResponse(await asyncio.to_thread(issue), headers={"Cache-Control": "no-store"})

    if not any(getattr(r, "path", None) == "/pgw/v1/sdk/enroll" for r in app.routes):
        app.add_route("/pgw/v1/sdk/enroll", sdk_enroll, methods=["POST"], include_in_schema=False)


def main() -> None:
    from pgw.core.log import configure_logging
    from pgw.service import ServiceSettings, create_app
    from pgw.service.server import serve_async

    configure_logging(level="INFO")
    tamper = os.environ.get("PGW_TEST_TAMPER_FILE")
    if tamper:
        _install_tamper(Path(tamper))
    settings = ServiceSettings()
    app = create_app(settings)
    lane_port = int(os.environ.get("PGW_TEST_LANE_PORT", "0") or 0) or None
    _add_sdk_route(app, Path(os.environ["PGW_TEST_SDK_KEYS"]), lane_port)
    asyncio.run(serve_async(settings, app=app))


if __name__ == "__main__":
    main()
