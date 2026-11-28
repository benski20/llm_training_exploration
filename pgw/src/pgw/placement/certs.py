"""Per-pod, short-lived TLS certificates under the pinned provider CA (no private key is ever shared or shipped).

Hierarchy (docs/operations/placement.md, "Secrets"):

* **Root**: the provider CA whose SHA-256 every SDK key pins. Offline (an HSM or an air-gapped machine); used only
  to sign the issuing CA below, once a year. ``BasicConstraints(ca, path_length=1)``.
* **Pod-issuing CA** (intermediate, ``path_length=0``, 90 days): held by the placement controller (file 0600 or a
  KMS key), never on a pod. Signs one leaf per pod registration.
* **Pod leaf** (default 24 h, at most 48 h): the pod generates its own P-256 key at boot (it never leaves the pod's
  memory and tmpfs), sends a CSR with its one-time registration token (:class:`RegistrationTokens`); the controller
  checks the token and issues a leaf whose subject alternative names are **the pod's public address as RunPod's API
  reports it** (never what the CSR asks for), EKU serverAuth. The pod re-registers for a new leaf before expiry; a
  drained or stopped pod's leaf simply expires (no CRL needed at this lifetime).

Clients verify exactly as today: the chain must end at the pinned root, the leaf must name the host they dialled. A
compromised pod yields its own key for at most a day and its pack's public weights; nothing a pod holds lets it
impersonate another pod or read any client's data (it only ever sees masked rows).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import ipaddress
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ["IssuedCert", "PodIssuer", "RegistrationTokens", "TokenError", "make_csr", "new_issuing_ca", "new_root",
           "verify_chain"]

MAX_LEAF_S = 48 * 3600
TOKEN_PREFIX = "pgwreg_"  # noqa: S105 - a public prefix, not a secret


def _name(cn: str) -> Any:
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn[:64])])


def _ku(ca: bool) -> Any:
    from cryptography import x509

    return x509.KeyUsage(digital_signature=True, key_cert_sign=ca, crl_sign=ca, content_commitment=False,
                         key_encipherment=False, data_encipherment=False, key_agreement=False, encipher_only=False,
                         decipher_only=False)


def new_root(cn: str = "PGW provider root CA", days: int = 3650) -> tuple[Any, Any]:
    """A root that may sign one level of issuing CA (``path_length=1``). Returns ``(key, cert)``."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(_name(cn)).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
            .add_extension(_ku(True), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    return key, cert


def new_issuing_ca(root_key: Any, root_cert: Any, cn: str = "PGW pod-issuing CA", days: int = 90) -> tuple[Any, Any]:
    """The controller's intermediate (``path_length=0``), signed by the root."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(root_cert.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(_ku(True), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
            .sign(root_key, hashes.SHA256()))
    return key, cert


def make_csr(pod_id: str) -> tuple[bytes, bytes]:
    """Pod side: a fresh P-256 key and a CSR for it. Returns ``(key PEM, CSR PEM)``; the key never leaves the pod."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    csr = x509.CertificateSigningRequestBuilder().subject_name(_name(pod_id)).sign(key, hashes.SHA256())
    return (key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption()),
            csr.public_bytes(serialization.Encoding.PEM))


class TokenError(PermissionError):
    """A registration token is unknown, used, expired or bound to another pod."""


@dataclass
class _Expect:
    digest: bytes
    pod_name: str
    dc: str
    model: str
    expires: float
    used: bool = False


class RegistrationTokens:
    """One-time registration tokens: issued when the controller creates a pod (they go into the pod's environment,
    visible to the RunPod account only), redeemed once, bound to (pod name, data centre, model), expiring after
    ``ttl_s``. Only ``SHA-256(token)`` is stored."""

    def __init__(self, *, ttl_s: float = 3600.0, clock: Callable[[], float] = time.time):
        self.ttl_s, self.clock = ttl_s, clock
        self._by_digest: dict[bytes, _Expect] = {}
        self._lock = threading.Lock()

    def issue(self, pod_name: str, dc: str, model: str) -> str:
        tok = TOKEN_PREFIX + secrets.token_urlsafe(32)
        d = hashlib.sha256(tok.encode()).digest()
        with self._lock:
            self._by_digest[d] = _Expect(d, pod_name, dc, model, self.clock() + self.ttl_s)
        return tok

    def redeem(self, token: str, *, pod_name: str, dc: str, model: str) -> None:
        d = hashlib.sha256(str(token).encode()).digest()
        with self._lock:
            e = self._by_digest.get(d)
            if e is None or not hmac.compare_digest(e.digest, d):
                raise TokenError("unknown registration token")
            if e.used:
                raise TokenError("registration token already used")
            if self.clock() > e.expires:
                raise TokenError("registration token expired")
            if (e.pod_name, e.dc, e.model) != (pod_name, dc, model):
                raise TokenError("registration token is bound to another pod")
            e.used = True


@dataclass(frozen=True)
class IssuedCert:
    chain_pem: bytes  # leaf + issuing CA (what the pod's TLS listener serves)
    serial: int
    not_after: float
    sha256: str  # of the leaf (DER): the controller checks the pod serves exactly this one


class PodIssuer:
    """Signs pod leaves with the issuing CA (see the module docstring)."""

    def __init__(self, issuing_key: Any, issuing_cert: Any, *, leaf_s: int = 24 * 3600):
        if not 0 < leaf_s <= MAX_LEAF_S:
            raise ValueError(f"leaf lifetime must be in (0, {MAX_LEAF_S}] s")
        self.key, self.cert, self.leaf_s = issuing_key, issuing_cert, int(leaf_s)

    def issue(self, csr_pem: bytes, *, pod_id: str, hosts: Iterable[str]) -> IssuedCert:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID

        csr = x509.load_pem_x509_csr(csr_pem)
        if not csr.is_signature_valid:
            raise ValueError("CSR signature does not verify")
        pk = csr.public_key()
        if not isinstance(pk, ec.EllipticCurvePublicKey) or pk.curve.name != "secp256r1":
            raise ValueError("pod keys must be P-256")
        sans: list[x509.GeneralName] = []
        for h in dict.fromkeys(hosts):
            try:
                sans.append(x509.IPAddress(ipaddress.ip_address(h)))
            except ValueError:
                sans.append(x509.DNSName(h))
        if not sans:
            raise ValueError("a pod certificate needs its public address")
        now = dt.datetime.now(dt.UTC)
        cert = (x509.CertificateBuilder().subject_name(_name(pod_id)).issuer_name(self.cert.subject)
                .public_key(pk).serial_number(x509.random_serial_number())
                .not_valid_before(now - dt.timedelta(minutes=5))
                .not_valid_after(now + dt.timedelta(seconds=self.leaf_s))
                .add_extension(x509.SubjectAlternativeName(sans), critical=False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(_ku(False), critical=True)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(self.key.public_key()),
                               critical=False)
                .sign(self.key, hashes.SHA256()))
        der = cert.public_bytes(serialization.Encoding.DER)
        chain = cert.public_bytes(serialization.Encoding.PEM) + self.cert.public_bytes(serialization.Encoding.PEM)
        return IssuedCert(chain, cert.serial_number, cert.not_valid_after_utc.timestamp(),
                          hashlib.sha256(der).hexdigest())


def verify_chain(chain_pem: bytes, root_pem: bytes, host: str, *, at: float | None = None) -> None:
    """Verify a served chain exactly as a client would: ending at ``root_pem`` only, naming ``host``.

    Raises:
        cryptography.x509.verification.VerificationError: it does not.
    """
    from cryptography import x509
    from cryptography.x509.verification import DNSName, IPAddress, PolicyBuilder, Store

    certs = x509.load_pem_x509_certificates(chain_pem)
    root = x509.load_pem_x509_certificate(root_pem)
    when = dt.datetime.fromtimestamp(at, dt.UTC) if at is not None else dt.datetime.now(dt.UTC)
    try:
        subject: Any = IPAddress(ipaddress.ip_address(host))
    except ValueError:
        subject = DNSName(host)
    v = PolicyBuilder().store(Store([root])).time(when).build_server_verifier(subject)
    v.verify(certs[0], certs[1:])
