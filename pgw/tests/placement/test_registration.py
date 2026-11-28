"""Per-pod certificates under the pinned root and the pod registration flow (pgw.placement.certs / registry)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pgw.placement.certs import (
    PodIssuer,
    RegistrationTokens,
    TokenError,
    make_csr,
    new_issuing_ca,
    new_root,
    verify_chain,
)
from pgw.placement.publisher import DirectoryPublisher, Signer
from pgw.placement.registry import BootStep, PodBoot, RegistrationError, Registry, registration_request
from pgw.sdk.directory import verify_directory

FP = "f7c6fb4f5f8fbc5e4149972c07afdc749de8e5f9adc604d30edb7eb94b834b67"


def _pem(c: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return c.public_bytes(serialization.Encoding.PEM)


@pytest.fixture(scope="module")
def pki() -> dict[str, Any]:
    rk, rc = new_root()
    ik, ic = new_issuing_ca(rk, rc)
    return {"root_key": rk, "root": rc, "issuer": PodIssuer(ik, ic, leaf_s=3600)}


def test_leaf_chains_to_pinned_root_only(pki: dict[str, Any]) -> None:
    from cryptography.x509.verification import VerificationError

    _key, csr = make_csr("pod-1")
    got = pki["issuer"].issue(csr, pod_id="pod-1", hosts=["203.0.113.7"])
    verify_chain(got.chain_pem, _pem(pki["root"]), "203.0.113.7")
    with pytest.raises(VerificationError):  # another name
        verify_chain(got.chain_pem, _pem(pki["root"]), "203.0.113.8")
    other_root = new_root("someone else")[1]
    with pytest.raises(VerificationError):  # another root
        verify_chain(got.chain_pem, _pem(other_root), "203.0.113.7")
    with pytest.raises(VerificationError):  # expired (short-lived)
        verify_chain(got.chain_pem, _pem(pki["root"]), "203.0.113.7", at=got.not_after + 60)
    # the leaf cannot sign further certificates (path_length 0 at the issuing CA, CA=false at the leaf)
    from cryptography import x509

    leaf = x509.load_pem_x509_certificates(got.chain_pem)[0]
    assert leaf.extensions.get_extension_for_class(x509.BasicConstraints).value.ca is False


def test_python_ssl_accepts_the_chain(pki: dict[str, Any], tmp_path: Path) -> None:
    """The client's TLS stack (ssl with the pinned root as the only trust anchor) accepts leaf + issuing CA."""
    import socket
    import ssl
    import threading

    key, csr = make_csr("pod-ssl")
    got = pki["issuer"].issue(csr, pod_id="pod-ssl", hosts=["127.0.0.1"])
    (tmp_path / "chain.pem").write_bytes(got.chain_pem)
    (tmp_path / "key.pem").write_bytes(key)
    (tmp_path / "root.pem").write_bytes(_pem(pki["root"]))
    srv = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    srv.load_cert_chain(tmp_path / "chain.pem", tmp_path / "key.pem")
    ls = socket.socket()
    ls.bind(("127.0.0.1", 0))
    ls.listen(1)

    def serve() -> None:
        c, _ = ls.accept()
        with srv.wrap_socket(c, server_side=True) as s:
            s.recv(1)

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    cli = ssl.create_default_context(cafile=str(tmp_path / "root.pem"))
    with socket.create_connection(ls.getsockname()) as raw, cli.wrap_socket(raw, server_hostname="127.0.0.1") as s:
        assert s.version() == "TLSv1.3"
        s.send(b"x")
    t.join(5)
    ls.close()


def test_issuer_refuses_bad_csr_and_lifetime(pki: dict[str, Any]) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    rk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (x509.CertificateSigningRequestBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,
                                                                                              "x")]))
           .sign(rk, hashes.SHA256()).public_bytes(serialization.Encoding.PEM))
    with pytest.raises(ValueError, match="P-256"):
        pki["issuer"].issue(csr, pod_id="x", hosts=["1.2.3.4"])
    with pytest.raises(ValueError):
        PodIssuer(None, None, leaf_s=7 * 86400)
    with pytest.raises(ValueError, match="address"):
        pki["issuer"].issue(make_csr("y")[1], pod_id="y", hosts=[])


def test_tokens_single_use_bound_expiring() -> None:
    t = [0.0]
    toks = RegistrationTokens(ttl_s=60, clock=lambda: t[0])
    a = toks.issue("pod-a", "EU-NL-1", "qwen2.5-7b-instruct")
    with pytest.raises(TokenError, match="another pod"):
        toks.redeem(a, pod_name="pod-a", dc="US-KS-2", model="qwen2.5-7b-instruct")
    toks.redeem(a, pod_name="pod-a", dc="EU-NL-1", model="qwen2.5-7b-instruct")
    with pytest.raises(TokenError, match="already used"):
        toks.redeem(a, pod_name="pod-a", dc="EU-NL-1", model="qwen2.5-7b-instruct")
    b = toks.issue("pod-b", "EU-NL-1", "qwen2.5-7b-instruct")
    t[0] = 61
    with pytest.raises(TokenError, match="expired"):
        toks.redeem(b, pod_name="pod-b", dc="EU-NL-1", model="qwen2.5-7b-instruct")
    with pytest.raises(TokenError, match="unknown"):
        toks.redeem("pgwreg_nope", pod_name="pod-b", dc="EU-NL-1", model="qwen2.5-7b-instruct")
    assert a not in repr(toks.__dict__)  # only digests are stored


def _ctx(tok: str, **kw: Any) -> dict[str, Any]:
    _key, csr = make_csr("pod-1")
    return {"token": tok, "pod_id": "pod-1", "pod_name": "pgw-eu-1", "dc": "EU-NL-1",
            "model": "qwen2.5-7b-instruct", "fingerprints": [FP], "selftests": {"pack": True, "device": True},
            "csr_pem": csr, **kw}


def test_registration_flow_publishes_signed_directory(pki: dict[str, Any]) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pub = DirectoryPublisher(Signer(Ed25519PrivateKey.generate()))
    published: list[bytes] = []
    reg = Registry(pki["issuer"], {"qwen2.5-7b-instruct": FP}, on_change=lambda eps: published.append(pub.publish(eps)))
    tok = reg.expect("pgw-eu-1", "EU-NL-1", "qwen2.5-7b-instruct")
    # pod side: verify -> load -> selftests -> health -> register, timed; the request carries no private key
    steps = [BootStep(n, lambda c: None) for n in ("verify_pack", "load", "selftests", "health")]
    ctx = PodBoot(steps).run(_ctx(tok))
    body = registration_request(ctx)
    assert set(ctx["boot_timings"]) == {"verify_pack", "load", "selftests", "health"}
    assert "PRIVATE KEY" not in json.dumps(body)
    rec = reg.register(body, public_host="203.0.113.9", public_port=41001, public_lane_port=41002)
    assert rec.url == "https://203.0.113.9:41001" and not published  # not listed before activation
    verify_chain(rec.cert.chain_pem, _pem(pki["root"]), "203.0.113.9")  # the API's address, not the pod's claim

    def probe_ok(url: str, cert: Any) -> dict[str, Any]:
        return {"ready": True, "fingerprints": [FP], "leaf_sha256": cert.sha256}

    with pytest.raises(RegistrationError, match="certificate"):
        reg.activate("pod-1", lambda u, c: {"ready": True, "fingerprints": [FP], "leaf_sha256": "00"})
    with pytest.raises(RegistrationError, match="not ready"):
        reg.activate("pod-1", lambda u, c: {"ready": False})
    reg.activate("pod-1", probe_ok)
    d = verify_directory(published[-1], pub.signer.public_key)
    assert [(e.url, sorted(e.models)) for e in d.endpoints] == [("https://203.0.113.9:41001", [FP])]
    assert d.lane_ports == {"https://203.0.113.9:41001": 41002} and not d.draining
    reg.drain("pod-1")
    d = verify_directory(published[-1], pub.signer.public_key, min_version=(d.version, d.digest))
    assert d.draining == frozenset({"https://203.0.113.9:41001"})
    reg.remove("pod-1")
    d = verify_directory(published[-1], pub.signer.public_key, min_version=(d.version, d.digest))
    assert d.endpoints == ()


@pytest.mark.parametrize("change,match", [
    ({"fingerprints": ["0" * 64]}, "pinned"),
    ({"selftests": {"pack": True, "device": False}}, "self-tests"),
    ({"selftests": {}}, "self-tests"),
    ({"dc": "US-KS-2"}, "another pod"),
    ({"token": "pgwreg_forged"}, "unknown"),
])
def test_registration_refusals(pki: dict[str, Any], change: dict[str, Any], match: str) -> None:
    reg = Registry(pki["issuer"], {"qwen2.5-7b-instruct": FP})
    tok = reg.expect("pgw-eu-1", "EU-NL-1", "qwen2.5-7b-instruct")
    body = registration_request(_ctx(tok, **change))
    with pytest.raises(RegistrationError, match=match):
        reg.register(body, public_host="203.0.113.9", public_port=1, public_lane_port=None)
    assert not reg.pods


def test_masked_head_fingerprint_allowed(pki: dict[str, Any]) -> None:
    from pgw.modelpack.masked_head import head_fingerprint

    reg = Registry(pki["issuer"], {"qwen2.5-7b-instruct": FP})
    tok = reg.expect("pgw-eu-1", "EU-NL-1", "qwen2.5-7b-instruct")
    rec = reg.register(registration_request(_ctx(tok, fingerprints=[FP, head_fingerprint(FP)])),
                       public_host="203.0.113.9", public_port=1, public_lane_port=None)
    assert len(rec.fingerprints) == 2


def test_boot_stops_before_registration_on_failure() -> None:
    ran: list[str] = []

    def bad(ctx: dict[str, Any]) -> None:
        raise RuntimeError("fingerprint mismatch")

    boot = PodBoot([BootStep("verify_pack", bad), BootStep("register", lambda c: ran.append("register"))])
    with pytest.raises(RuntimeError):
        boot.run({})
    assert ran == [] and "verify_pack" in boot.timings


def test_staged_pack_verified_against_pin(tiny_pack_dir: Path, tmp_path: Path) -> None:
    """A pre-staged pack opens only with the pinned fingerprint; one changed byte on the volume stops the boot."""
    import shutil

    from pgw.core.errors import ConfigError, FingerprintMismatchError
    from pgw.modelpack.loader import ModelPack
    from pgw.placement.packs import verify_staged

    vol = tmp_path / "vol"
    shutil.copytree(tiny_pack_dir, vol / "tiny")
    fp = ModelPack.open(vol / "tiny", verify="none").fingerprint
    verify_staged(vol, "tiny", expected=fp).close()
    with pytest.raises(FingerprintMismatchError):
        verify_staged(vol, "tiny", expected="0" * 64)
    f = next(p for p in sorted((vol / "tiny").glob("*.safetensors")))
    b = bytearray(f.read_bytes())
    b[-1] ^= 1
    f.write_bytes(bytes(b))
    with pytest.raises((FingerprintMismatchError, ConfigError)):
        verify_staged(vol, "tiny", expected=fp)
