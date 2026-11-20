"""``pgw doctor`` checks that size the mask supply (scheduler capacity) and certificate expiry."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from pgw.cli.doctor import _check_cert_files, _check_mask_supply, run_doctor


@pytest.fixture
def no_pgw_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for k in list(os.environ):
        if k.startswith("PGW_") and k not in ("PGW_OFFLINE", "PGW_MASKSTORE_ALLOW_NO_EXTERNAL_COUNTER"):
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_mask_supply_local(tiny_pack_dir: Path, no_pgw_env: pytest.MonkeyPatch) -> None:
    status, detail, data = _check_mask_supply(tiny_pack_dir, None, None, None, slots=8)
    assert status == "ok", detail
    assert data["mode"] == "local" and data["slots_timed"] == 8
    assert data["slots_per_second"] > 0 and data["bytes_per_slot"] > 0
    assert data["seconds_per_segment"] > 0 and "slots/s" in detail


def test_mask_supply_against_expected_load(tiny_pack_dir: Path, no_pgw_env: pytest.MonkeyPatch) -> None:
    status, detail, data = _check_mask_supply(tiny_pack_dir, None, None, 1e9, slots=8)
    assert status == "fail" and "below the expected load" in detail
    no_pgw_env.setenv("PGW_DOCTOR_EXPECTED_TOKENS_PER_S", "0.001")
    status, _, data = _check_mask_supply(tiny_pack_dir, None, None, None, slots=8)
    assert status == "ok" and data["expected_tokens_per_second"] == 0.001


def test_mask_supply_disk_and_first_segment(tiny_pack_dir: Path, tmp_path: Path,
                                            no_pgw_env: pytest.MonkeyPatch) -> None:
    no_pgw_env.setenv("PGW_FACTORY_FIRST_SEGMENT_SLOTS", "256")
    no_pgw_env.setenv("PGW_MASKSTORE_TARGET_SLOTS", str(10**12))  # never fits any volume
    status, detail, data = _check_mask_supply(tiny_pack_dir, None, tmp_path, None, slots=8)
    assert status == "fail" and "cannot hold target_slots" in detail
    assert data["seconds_to_first_segment"] > 0
    no_pgw_env.setenv("PGW_MASKSTORE_TARGET_SLOTS", "16")
    status, _, data = _check_mask_supply(tiny_pack_dir, None, tmp_path, None, slots=8)
    assert status == "ok" and data["disk_needed_for_target"] <= data["disk_total"]


def test_mask_supply_encrypted_without_hint_warns(tiny_pack_dir: Path, tmp_path: Path,
                                                  no_pgw_env: pytest.MonkeyPatch) -> None:
    no_pgw_env.setenv("PGW_FACTORY_HE_SERVICE_URL", "https://compute.invalid:8443")
    no_pgw_env.setenv("PGW_FACTORY_HE_HINT_DIR", str(tmp_path / "hints"))
    status, detail, data = _check_mask_supply(tiny_pack_dir, None, None, None, slots=8)
    assert status == "warn" and "hint is not built" in detail and data["mode"].startswith("encrypted")


def test_run_doctor_runs_mask_supply_for_factory_role(tiny_pack_dir: Path, no_pgw_env: pytest.MonkeyPatch) -> None:
    rep = run_doctor(role="pack", pack=tiny_pack_dir, include_gpu=False)
    assert rep.get("mask_supply") is None  # only for factory / gateway roles unless asked
    rep = run_doctor(role="pack", pack=tiny_pack_dir, include_gpu=False, mask_supply=True,
                     expected_tokens_per_second=0.001)
    c = rep.get("mask_supply")
    assert c is not None and c.status == "ok", c


# ---- certificate expiry ------------------------------------------------------------------------------
def _cert(path: Path, days: float) -> Path:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, path.stem)])
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=30))
            .not_valid_after(now + dt.timedelta(days=days)).sign(key, hashes.SHA256()))
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return path


def test_cert_expiry(tmp_path: Path) -> None:
    ok = _cert(tmp_path / "ok.crt", 90)
    soon = _cert(tmp_path / "soon.crt", 3)
    old = _cert(tmp_path / "old.crt", -1)
    assert _check_cert_files([])[0] == "skip"
    status, detail, data = _check_cert_files([ok])
    assert status == "ok" and data["certificates"][0]["days_left"] > 80
    status, detail, _ = _check_cert_files([ok, soon, ok])
    assert status == "warn" and "soon.crt expires in" in detail
    status, detail, _ = _check_cert_files([ok, soon, old])
    assert status == "fail" and "old.crt" in detail
    bundle = tmp_path / "bundle.pem"
    bundle.write_bytes(ok.read_bytes() + soon.read_bytes())
    assert _check_cert_files([bundle])[0] == "warn"  # every certificate in a bundle counts
    junk = tmp_path / "junk.pem"
    junk.write_text("not a certificate")
    status, detail, _ = _check_cert_files([ok, junk])
    assert status == "warn" and "unreadable" in detail


def test_run_doctor_cert_check(tmp_path: Path, no_pgw_env: pytest.MonkeyPatch) -> None:
    rep = run_doctor(role="pack", include_gpu=False)
    assert rep.get("tls_certs") is None  # nothing configured: no row
    soon = _cert(tmp_path / "svc-ca.crt", 2)
    rep = run_doctor(role="pack", include_gpu=False, cert_files=[soon])
    c = rep.get("tls_certs")
    assert c is not None and c.status == "warn"
    no_pgw_env.setenv("PGW_FACTORY_HE_CA_FILE", str(soon))
    no_pgw_env.setenv("PGW_FACTORY_PACK_PATH", str(tmp_path))
    rep = run_doctor(role="factory", include_gpu=False)
    c = rep.get("tls_certs")
    assert c is not None and c.status == "warn"
