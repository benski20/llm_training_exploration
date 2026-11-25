"""Fixtures of the client-library tests.

``sdk_pack_dir`` (session)
    The tiny random Qwen2 checkpoint packed under a supported model name (``qwen2.5-0.5b-instruct``), or the real
    Qwen2.5-0.5B-Instruct with ``PGW_SDK_E2E_MODEL=qwen`` (network).
``pki`` (session)
    A provider CA with a server certificate for 127.0.0.1 and a client-issuing CA (files).
``provider`` (module)
    A real ``pgw service`` process on loopback (CPU; TLS; TCP+TLS trip lane; masked LM head served; pks-v3 maskgen
    on; enrollment) with the test SDK enrollment route (:mod:`tests.sdk.provider_main`), and a valid SDK key.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

MODEL = "qwen2.5-0.5b-instruct"
ROOT = Path(__file__).resolve().parents[2]


def _ca(cn: str) -> tuple[Any, Any]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = dt.datetime.now(dt.UTC)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=2))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    return cert, key


def _server_leaf(ca: tuple[Any, Any]) -> tuple[bytes, bytes]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = dt.datetime.now(dt.UTC)
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
            .issuer_name(ca[0].subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=2))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                           critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca[1].public_key()), critical=False)
            .sign(ca[1], hashes.SHA256()))
    return (cert.public_bytes(serialization.Encoding.PEM),
            key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption()))


@pytest.fixture(scope="session")
def pki(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path | bytes]:
    from cryptography.hazmat.primitives import serialization

    d = tmp_path_factory.mktemp("sdk-pki")
    ca, client_ca = _ca("provider-ca"), _ca("client-issuing-ca")
    srv_pem, srv_key = _server_leaf(ca)
    out: dict[str, Path | bytes] = {}
    files = {"ca.pem": ca[0].public_bytes(serialization.Encoding.PEM), "server.pem": srv_pem, "server.key": srv_key,
             "client-ca.pem": client_ca[0].public_bytes(serialization.Encoding.PEM),
             "client-ca.key": client_ca[1].private_bytes(serialization.Encoding.PEM,
                                                         serialization.PrivateFormat.PKCS8,
                                                         serialization.NoEncryption())}
    for name, data in files.items():
        (d / name).write_bytes(data)
        (d / name).chmod(0o600)
        out[name] = d / name
    out["ca_sha256"] = hashlib.sha256(ca[0].public_bytes(serialization.Encoding.DER)).digest()
    return out


@pytest.fixture(scope="session")
def sdk_pack_dir(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from pgw.modelpack import pack_checkpoint

    src: Path | str = tiny_hf_dir
    if os.environ.get("PGW_SDK_E2E_MODEL", "tiny").lower() == "qwen":  # pragma: no cover - network
        from huggingface_hub import snapshot_download

        src = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    out = tmp_path_factory.mktemp("sdk-pack") / "pack"
    pack_checkpoint(src, out, name=MODEL)
    return out


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class Provider:
    """The running provider process and the files the tests drive it with."""

    url: str
    proc: subprocess.Popen[bytes]
    log: Path
    keys_file: Path
    tamper_file: Path
    ca_sha256: bytes
    lane_port: int
    pack_dir: Path

    def new_key(self, tenant: str = "sdk-test") -> str:
        from pgw.sdk.keys import SdkKey

        k = SdkKey.new(self.url, self.ca_sha256, env="test")
        keys = json.loads(self.keys_file.read_text()) if self.keys_file.exists() else {}
        keys[k.key_id_hex] = {"secret_sha256": k.secret_sha256().hex(), "tenant": tenant}
        self.keys_file.write_text(json.dumps(keys))
        return k.encode()

    def revoke(self, key: str) -> None:
        from pgw.sdk.keys import parse_key

        keys = json.loads(self.keys_file.read_text())
        keys[parse_key(key).key_id_hex]["revoked"] = True
        self.keys_file.write_text(json.dumps(keys))

    def tail(self, n: int = 8000) -> str:
        try:
            return self.log.read_text(errors="replace")[-n:]
        except OSError:  # pragma: no cover
            return ""


def start_provider(root: Path, pack_dir: Path, pki: dict[str, Any], *, extra_env: dict[str, str] | None = None,
                   ) -> Provider:
    port, lane_port = free_port(), free_port()
    url = f"https://127.0.0.1:{port}"
    secrets = root / "secrets"
    secrets.mkdir(mode=0o700, exist_ok=True)
    keys_file = root / "sdk-keys.json"
    keys_file.write_text("{}")
    tamper = root / "tamper.flag"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    env.update({
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]),
        "OMP_NUM_THREADS": "2",
        "PGW_SERVICE_HOST": "127.0.0.1", "PGW_SERVICE_PORT": str(port),
        "PGW_SERVICE_PACKS": json.dumps([str(pack_dir)]), "PGW_SERVICE_RINGS": json.dumps(["K32A12C27"]),
        "PGW_SERVICE_DEVICE": "cpu", "PGW_SERVICE_VERIFY_PACKS": "full", "PGW_SERVICE_CPU_WORKERS": "2",
        "PGW_SERVICE_WARMUP": "false", "PGW_SERVICE_BATCH_WINDOW_MS": "1",
        "PGW_SERVICE_TLS__CERT_FILE": str(pki["server.pem"]), "PGW_SERVICE_TLS__KEY_FILE": str(pki["server.key"]),
        "PGW_SERVICE_TLS__CA_FILE": str(pki["client-ca.pem"]),
        "PGW_SERVICE_TOKENS_FILE": str(secrets / "tokens"),
        "PGW_SERVICE_MTLS_TENANTS_FILE": str(secrets / "mtls_tenants"),
        "PGW_SERVICE_ENROLLMENT_FILE": str(secrets / "enrollment.json"),
        "PGW_SERVICE_PUBLIC_URL": url, "PGW_SERVICE_ENROLLMENT_CA_FILE": str(pki["ca.pem"]),
        "PGW_SERVICE_CLIENT_ISSUING_CA_CERT_FILE": str(pki["client-ca.pem"]),
        "PGW_SERVICE_CLIENT_ISSUING_CA_KEY_FILE": str(pki["client-ca.key"]),
        "PGW_SERVICE_LANE": "tcp", "PGW_SERVICE_LANE_PORT": str(lane_port),
        "PGW_SERVICE_MASKED_HEAD": "also", "PGW_SERVICE_MASKGEN_V3": "true",
        "PGW_TEST_SDK_KEYS": str(keys_file), "PGW_TEST_TAMPER_FILE": str(tamper),
        "PGW_TEST_LANE_PORT": str(lane_port),
    })
    (secrets / "tokens").touch(mode=0o600)
    (secrets / "mtls_tenants").touch(mode=0o600)
    env.update(extra_env or {})
    log = root / "provider.log"
    fh = log.open("ab")
    proc = subprocess.Popen([sys.executable, "-m", "tests.sdk.provider_main"], env=env, cwd=str(ROOT),  # noqa: S603
                            stdout=fh, stderr=subprocess.STDOUT)
    fh.close()
    prov = Provider(url, proc, log, keys_file, tamper, pki["ca_sha256"], lane_port, pack_dir)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 300:
        if proc.poll() is not None:
            raise AssertionError(f"provider exited {proc.returncode}:\n{prov.tail()}")
        try:
            r = httpx.get(f"{url}/readyz", verify=str(pki["ca.pem"]), timeout=5.0, trust_env=False)
            if r.status_code == 200:
                return prov
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    proc.kill()
    raise AssertionError(f"provider not ready:\n{prov.tail()}")


def stop_provider(p: Provider) -> None:
    if p.proc.poll() is None:
        p.proc.terminate()
        try:
            p.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            p.proc.kill()
            p.proc.wait(timeout=10)


@pytest.fixture(scope="module")
def provider(sdk_pack_dir: Path, pki: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Iterator[Provider]:
    p = start_provider(tmp_path_factory.mktemp("sdk-provider"), sdk_pack_dir, pki)
    try:
        yield p
    finally:
        stop_provider(p)
