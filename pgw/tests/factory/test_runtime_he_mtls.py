"""Encrypted mask factory: presents the gateway's client certificate and picks up renewals."""

from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from pgw.factory.config import FactorySettings
from tests.enroll_helpers import make_pki
from tests.wire.test_tls_cert_renewal import _client_cert


def test_settings_pair_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="set together"):
        FactorySettings(pack_path=tmp_path, he_client_cert_file=tmp_path / "c.crt")
    monkeypatch.setenv("PGW_FACTORY_HE_CLIENT_CERT_FILE", str(tmp_path / "c.crt"))
    monkeypatch.setenv("PGW_FACTORY_HE_CLIENT_KEY_FILE", str(tmp_path / "c.key"))
    s = FactorySettings(pack_path=tmp_path)
    assert s.he_client_cert_file == tmp_path / "c.crt" and s.he_client_key_file == tmp_path / "c.key"


def test_he_backend_client_presents_the_client_certificate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          ) -> None:
    import pgw.factory.runtime as rt
    import pgw.hemask.backend as hb
    import pgw.hemask.client as hc

    pki = make_pki(tmp_path / "pki")
    now = dt.datetime.now(dt.UTC)
    cert, key, _ = _client_cert(pki, tmp_path, "gw", not_before=now - dt.timedelta(minutes=1),
                                not_after=now + dt.timedelta(hours=1))
    seen: dict[str, Any] = {}

    class _Hints:
        nbytes = 0

    class _Remote:
        def __init__(self, client: Any, *a: Any, **k: Any):
            seen["client"] = client

    class _Pack:
        fingerprint = "ab" * 32
        short_fingerprint = "abab"

        def linear_specs(self) -> list[Any]:
            return []

    monkeypatch.setattr(hb.HintStore, "open_or_build", classmethod(lambda cls, *a, **k: _Hints()))
    monkeypatch.setattr(hc, "RemoteMulmod64", _Remote)
    monkeypatch.setattr(hb, "HEMaskBackend", lambda *a, **k: object())
    watched: list[tuple[Path, Path]] = []
    monkeypatch.setattr(rt, "watch_client_cert", lambda c, k, **kw: watched.append((c, k)))
    s = FactorySettings(pack_path=tmp_path, he_service_url="https://127.0.0.1:1", he_ca_file=pki["ca"],
                        he_client_cert_file=cert,
                        he_client_key_file=key, he_hint_dir=tmp_path / "hints")
    rt.build_he_backend(s, _Pack(), "K32A12C27")
    client = seen["client"]
    try:
        assert client.config.client_cert_file == cert and client.config.client_key_file == key
        assert client._cert_key is not None  # registered for reloads
        assert watched == [(cert, key)]
    finally:
        client.close()


def test_watch_client_cert_reloads_on_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.wire.client as wc
    from pgw.factory.runtime import watch_client_cert

    pki = make_pki(tmp_path / "pki")
    now = dt.datetime.now(dt.UTC)
    cert, key, _ = _client_cert(pki, tmp_path, "gw", not_before=now - dt.timedelta(minutes=1),
                                not_after=now + dt.timedelta(hours=1))
    reloaded = threading.Event()
    monkeypatch.setattr(wc, "reload_client_cert", lambda c, k, **kw: reloaded.set() or 1)
    stop = threading.Event()
    t = watch_client_cert(cert, key, interval_s=0.02, stop=stop)
    try:
        time.sleep(0.1)
        assert not reloaded.is_set()  # unchanged files: nothing to do
        new_c, new_k, _ = _client_cert(pki, tmp_path, "new", not_before=now - dt.timedelta(minutes=1),
                                       not_after=now + dt.timedelta(hours=2))
        new_k.replace(key)
        new_c.replace(cert)
        assert reloaded.wait(5)
    finally:
        stop.set()
        t.join(5)
