"""Test helpers for enrollment: a throw-away PKI and a real TLS compute service with enrollment on."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization


@dataclass
class EnrollService:
    url: str
    settings: Any
    dirs: dict[str, Path]
    thread: Any

    def create_code(self, tenant: str = "acme", ttl_s: float = 3600.0, models: list[str] | None = None,
                    ca_path: Path | None = None) -> tuple[str, Any]:
        """A fresh code for this service (pinning ``ca_path`` or the real service CA)."""
        from pgw.enroll.code import EnrollmentCode
        from pgw.enroll.service import ca_fingerprint, load_ca_pem
        from pgw.enroll.store import EnrollmentStore

        rec, secret = EnrollmentStore(self.settings.enrollment_file).create(tenant, ttl_s, models)
        fp = ca_fingerprint(load_ca_pem(ca_path or self.dirs["ca"]))
        return EnrollmentCode(self.url, fp, secret).encode(), rec


def make_pki(d: Path) -> dict[str, Path]:
    """Service CA + server certificate for 127.0.0.1/localhost, a client-issuing CA and a rogue CA."""
    from tests.service.test_server import _ca, _issue, _key_pem

    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    ca_key, ca = _ca("pgw-enroll-test-ca")
    out["ca"] = d / "ca.pem"
    out["ca"].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    k, c = _issue(ca_key, ca, "pgw-service", server=True)
    out["server"], out["server_key"] = d / "server.pem", d / "server.key"
    out["server"].write_bytes(c.public_bytes(serialization.Encoding.PEM))
    out["server_key"].write_bytes(_key_pem(k))
    cca_key, cca = _ca("pgw-client-issuing-ca")
    out["client_ca"], out["client_ca_key"] = d / "client-ca.pem", d / "client-ca.key"
    out["client_ca"].write_bytes(cca.public_bytes(serialization.Encoding.PEM))
    out["client_ca_key"].write_bytes(_key_pem(cca_key))
    _, rogue = _ca("rogue-ca")
    out["rogue_ca"] = d / "rogue.pem"
    out["rogue_ca"].write_bytes(rogue.public_bytes(serialization.Encoding.PEM))
    return out


@contextlib.contextmanager
def enroll_service(pack_dir: Path, d: Path, **overrides: Any) -> Iterator[EnrollService]:
    """A real compute service (uvicorn, TLS) serving ``pack_dir`` with enrollment enabled."""
    from pgw.service import ServiceSettings
    from tests.service.test_server import _free_port, _ServerThread, _wait_ready

    pki = make_pki(d / "pki")
    tokens = d / "tenants.txt"
    tokens.write_text("")
    port, mport = _free_port(), _free_port()
    url = f"https://127.0.0.1:{port}"
    kw: dict[str, Any] = dict(
        host="127.0.0.1", port=port, metrics_port=mport, packs=[pack_dir], device="cpu", tokens_file=tokens,
        batch_window_ms=1.0, rings=["K32A12C27", "K40A12"], server="uvicorn", drain_timeout_s=2.0,
        tls={"cert_file": pki["server"], "key_file": pki["server_key"]},
        enrollment_file=d / "enrollment.json", public_url=url, enrollment_ca_file=pki["ca"],
        client_issuing_ca_cert_file=pki["client_ca"], client_issuing_ca_key_file=pki["client_ca_key"],
    )
    kw.update(overrides)
    s = ServiceSettings(**kw)
    srv = _ServerThread(s)
    srv.start()
    import ssl

    _wait_ready(url, verify=ssl.create_default_context(cafile=str(pki["ca"])))
    try:
        yield EnrollService(url=url, settings=s, dirs={**pki, "tokens": tokens, "root": d}, thread=srv)
    finally:
        srv.stop()
