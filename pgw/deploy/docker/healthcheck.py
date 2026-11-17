#!/usr/bin/env python3
"""Container health probe for PGW images (stdlib only; used by Docker HEALTHCHECK and compose).

    pgw-healthcheck gateway   # GET /readyz on PGW_GATEWAY_PORT (default 8080)
    pgw-healthcheck service   # GET /readyz on PGW_SERVICE_PORT (default 8443)
    pgw-healthcheck --live service   # /healthz instead of /readyz

TLS is detected from ``PGW_<ROLE>_TLS__CERT_FILE``. The probe talks to 127.0.0.1 only, so the
server certificate is not verified. When the server requires client certificates, set
``PGW_HEALTHCHECK_CERT_FILE`` / ``PGW_HEALTHCHECK_KEY_FILE`` to a probe certificate.
Exit code 0 = healthy, 1 = unhealthy.
"""

from __future__ import annotations

import os
import ssl
import sys
import urllib.error
import urllib.request

DEFAULT_PORTS = {"gateway": 8080, "service": 8443}


def main(argv: list[str]) -> int:
    """Probe the local server for ``role``."""
    live = "--live" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1 or args[0] not in DEFAULT_PORTS:
        print("usage: pgw-healthcheck [--live] gateway|service", file=sys.stderr)
        return 2
    role = args[0]
    prefix = f"PGW_{role.upper()}_"
    raw_port = os.environ.get(prefix + "PORT", str(DEFAULT_PORTS[role]))
    try:
        port = int(raw_port)
        if not 0 < port < 65536:
            raise ValueError(raw_port)
    except ValueError:
        print(f"unhealthy: invalid {prefix}PORT {raw_port!r}", file=sys.stderr)
        return 1
    tls = bool(os.environ.get(prefix + "TLS__CERT_FILE"))
    path = "/healthz" if live else "/readyz"
    url = f"{'https' if tls else 'http'}://127.0.0.1:{port}{path}"
    ctx = None
    if tls:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # loopback probe of our own listener
        cert, key = os.environ.get("PGW_HEALTHCHECK_CERT_FILE"), os.environ.get("PGW_HEALTHCHECK_KEY_FILE")
        if cert and key:
            try:
                ctx.load_cert_chain(cert, key)
            except (OSError, ssl.SSLError) as exc:
                print(f"unhealthy: cannot load probe certificate: {exc}", file=sys.stderr)
                return 1
    try:
        with urllib.request.urlopen(url, timeout=5, context=ctx) as resp:  # noqa: S310 - fixed loopback URL
            return 0 if resp.status == 200 else 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"unhealthy: {url}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
