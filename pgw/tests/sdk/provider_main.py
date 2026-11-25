"""A real ``pgw service`` process for the SDK end-to-end tests.

Run as ``python -m tests.sdk.provider_main``. Everything comes from ``PGW_SERVICE_*`` (and ``PGW_PLATFORM_*``): the
real accounts (``POST /pgw/v1/sdk/enroll``, the admin API, request reports), provisioning and console. The only test
knob is a fault injection:

* ``PGW_TEST_TAMPER_FILE``: while this file exists, every compute result has one element changed (an active
  provider returning a wrong result; caught by the gateway's checks).
"""

from __future__ import annotations

import asyncio
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
    asyncio.run(serve_async(settings, app=app))


if __name__ == "__main__":
    main()
