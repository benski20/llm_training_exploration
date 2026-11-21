"""Process bootstrap shared by the Modal provider and customer apps (no Modal imports here).

Provider (compute service):
  * :func:`make_pki` creates the provider's PKI: a service CA (gateways pin its SHA-256 through the
    enrollment code), a TLS server certificate for the public tunnel host name, and a separate
    client-issuing CA that signs the gateways' client certificates at enrollment.
  * :func:`service_env` / :func:`start_service` run ``pgw service serve`` with TLS, tenant tokens,
    enrollment and verified-but-optional client certificates.
  * :func:`create_enrollment_code` runs ``pgw service enroll create``.

Customer (gateway):
  * :func:`start_gateway` runs ``pgw gateway serve`` in setup mode and returns the one-time setup token
    it prints.

The same functions drive the local rehearsal (``local_rehearsal.py``), so what runs on Modal is what
was tested here.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = "qwen2.5-0.5b-instruct"
RING = "K32A12C27"


def log(msg: str) -> None:
    print(f"[pgw-modal] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------
# provider PKI
# ------------------------------------------------------------------------------------------------
def _write_key(path: Path, key: Any) -> None:
    from cryptography.hazmat.primitives import serialization

    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)


def _write_cert(path: Path, cert: Any) -> None:
    from cryptography.hazmat.primitives import serialization

    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _ca(common_name: str, days: float) -> tuple[Any, Any]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    return key, cert


def i6pn_address() -> str | None:
    """This container's private IPv6 address on Modal's i6pn network, or None when it has none."""
    import socket

    try:
        return str(socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0])
    except OSError as exc:
        log(f"no i6pn address ({type(exc).__name__}); using the public tunnel")
        return None


def make_pki(pki_dir: Path, hostnames: list[str], days: float = 2.0) -> dict[str, Path]:
    """Service CA + server certificate for ``hostnames`` + client-issuing CA (all fresh, short-lived).

    Returns the file paths: ``service_ca``, ``server_cert``, ``server_key``, ``client_ca``, ``client_ca_key``.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    pki_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    ca_key, ca_cert = _ca("PGW demo provider service CA", days)
    cl_key, cl_cert = _ca("PGW demo provider gateway-client CA", days)
    key = ec.generate_private_key(ec.SECP256R1())
    sans: list[x509.GeneralName] = []
    for h in hostnames:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostnames[0])]))
            .issuer_name(ca_cert.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    out = {"service_ca": pki_dir / "service-ca.pem", "server_cert": pki_dir / "server.pem",
           "server_key": pki_dir / "server.key", "client_ca": pki_dir / "client-ca.pem",
           "client_ca_key": pki_dir / "client-ca.key"}
    _write_cert(out["service_ca"], ca_cert)
    _write_cert(out["server_cert"], cert)
    _write_key(out["server_key"], key)
    _write_cert(out["client_ca"], cl_cert)
    _write_key(out["client_ca_key"], cl_key)
    # The service CA's own key is not needed after signing the server certificate: drop it.
    return out


# ------------------------------------------------------------------------------------------------
# provider: compute service
# ------------------------------------------------------------------------------------------------
def service_env(work: Path, pack: Path, pki: dict[str, Path], public_url: str, port: int,
                device: str = "auto") -> dict[str, str]:
    """``PGW_SERVICE_*`` settings: TLS, tenant tokens, enrollment, optional verified client certificates.

    Production settings (no dev options). Client certificates are *verified when presented* (CERT_OPTIONAL)
    and mapped to their tenant, but not *required* on the listener: a gateway has no certificate before
    it enrolls, and enrollment shares the main port. Every compute call still needs the tenant's bearer
    token; when a certificate is presented as well, it must belong to the same tenant.
    """
    state = work / "state"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    for f in ("tenants.txt", "mtls-tenants.txt"):
        p = state / f
        if not p.exists():
            fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(fd)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    env.update({
        "PGW_LOG_LEVEL": "INFO",
        "PGW_SERVICE_HOST": "0.0.0.0",  # noqa: S104 - reached through the Modal TCP tunnel
        "PGW_SERVICE_PORT": str(port),
        "PGW_SERVICE_PACKS": json.dumps([str(pack)]),
        "PGW_SERVICE_RINGS": json.dumps([RING]),
        "PGW_SERVICE_DEVICE": device,
        "PGW_SERVICE_TOKENS_FILE": str(state / "tenants.txt"),
        "PGW_SERVICE_MTLS_TENANTS_FILE": str(state / "mtls-tenants.txt"),
        "PGW_SERVICE_TLS__CERT_FILE": str(pki["server_cert"]),
        "PGW_SERVICE_TLS__KEY_FILE": str(pki["server_key"]),
        "PGW_SERVICE_TLS__CA_FILE": str(pki["client_ca"]),
        "PGW_SERVICE_TLS__REQUIRE_CLIENT_CERT": "false",
        "PGW_SERVICE_ENROLLMENT_FILE": str(state / "enrollment.json"),
        "PGW_SERVICE_PUBLIC_URL": public_url,
        "PGW_SERVICE_ENROLLMENT_CA_FILE": str(pki["service_ca"]),
        "PGW_SERVICE_CLIENT_ISSUING_CA_CERT_FILE": str(pki["client_ca"]),
        "PGW_SERVICE_CLIENT_ISSUING_CA_KEY_FILE": str(pki["client_ca_key"]),
        "PGW_SERVICE_CLIENT_CERT_VALIDITY_DAYS": "1",
    })
    return env


def _tee(stream: IO[bytes], sink: IO[bytes], on_line: Any = None) -> None:
    for raw in iter(stream.readline, b""):
        sink.write(raw)
        sink.flush()
        if on_line is not None:
            on_line(raw.decode("utf-8", "replace"))


def start_logged(cmd: list[str], env: dict[str, str], logfile: Path, on_line: Any = None,
                 redact: re.Pattern[bytes] | None = None) -> subprocess.Popen[bytes]:
    """Start ``cmd``; its output goes to ``logfile`` and stdout (``redact`` matches are masked on stdout)."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # noqa: S603
                            stdin=subprocess.DEVNULL)
    fh = logfile.open("ab")

    class _Both:
        def write(self, b: bytes) -> None:
            fh.write(b)
            sys.stdout.buffer.write(redact.sub(b"<redacted>", b) if redact is not None else b)

        def flush(self) -> None:
            fh.flush()
            sys.stdout.flush()

    assert proc.stdout is not None
    threading.Thread(target=_tee, args=(proc.stdout, _Both(), on_line), daemon=True).start()
    return proc


def wait_ready(url: str, proc: subprocess.Popen[bytes], *, verify: Any = True, timeout_s: float = 900,
               ok: tuple[int, ...] = (200,)) -> None:
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited with status {proc.returncode} before {url} was ready")
        try:
            if httpx.get(url, timeout=5.0, verify=verify, trust_env=False).status_code in ok:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"{url} not ready after {timeout_s:.0f}s")


def start_service(env: dict[str, str], logfile: Path) -> subprocess.Popen[bytes]:
    return start_logged([sys.executable, "-m", "pgw.cli", "service", "serve"], env, logfile)


def create_enrollment_code(env: dict[str, str], tenant: str, ttl: str, note: str) -> dict[str, Any]:
    r = subprocess.run([sys.executable, "-m", "pgw.cli", "service", "enroll", "create", "--tenant", tenant,  # noqa: S603
                        "--ttl", ttl, "--models", MODEL_NAME, "--note", note, "--json"],
                       env=env, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"enroll create failed: {r.stderr[-2000:]}")
    out: dict[str, Any] = json.loads(r.stdout)
    return out


# ------------------------------------------------------------------------------------------------
# customer: gateway in setup mode
# ------------------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\b(pgw-setup-[A-Za-z0-9_-]{8,})\b")


#: Speed settings the setup wizard writes into the customer's ``models.json`` (``setup_model_options``).
#: Each keeps the answer bit-identical to the plaintext quantized reference (the e2e check compares) and
#: shows the provider nothing new:
#:
#: * ``local_linears = "auto"``: qkv and o computed exactly on the gateway when one layer's local cost is
#:   below two round trips. Decided once at start-up from a few mask-free dummy round trips; over the
#:   public tunnel (about 7-26 ms per trip) that is "attention" (break-even about 1.6 ms), which halves the
#:   sequential round trips per token. On i6pn the measurement decides.
#: * ``float_threads = 1``: the gateway's decode float ops on one torch thread (self-checked bit-identical;
#:   2-5 ms per token less on 4 vCPUs). Same effect as ``PGW_FLOAT_THREADS=1``.
#: * ``lane = "stream"``: the blocking stream lane. Decode trips go over the same WebSocket stream endpoint
#:   (same TLS, same msgpack bytes) but are written and read on the gateway's plaintext thread, with the
#:   TripEngine and synchronous decode: no event-loop hand-off per trip (W1: about 20-35 ms per token less
#:   with 48 trips per token, measured with benchmarks/trip_profile.py). Any call it cannot complete is
#:   resent byte-identically on the stream channel. (The native UDS lane needs a provider on this host.)
#:
#: Not set: strict
#: speculation (``spec_rows`` / ``spec_strict_passes``). The demo has no draft model and its chat barely
#: repeats the prompt, so prompt lookup accepts almost nothing: the provider sees as many passes either
#: way, while every pass would carry and mask ``spec_rows`` rows (``spec_rows`` times the mask supply).
GATEWAY_MODEL_OPTIONS: dict[str, Any] = {"local_linears": "auto", "float_threads": 1, "lane": "stream"}


def gateway_env(config_dir: Path, data_dir: Path, port: int,
                model_options: dict[str, Any] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    env.update({
        "PGW_LOG_LEVEL": "INFO",
        "PGW_GATEWAY_HOST": "0.0.0.0",  # noqa: S104 - reached through the Modal HTTPS tunnel
        "PGW_GATEWAY_PORT": str(port),
        "PGW_GATEWAY_CONFIG_DIR": str(config_dir),
        "PGW_GATEWAY_DATA_DIR": str(data_dir),
        "PGW_GATEWAY_SETUP_MODEL_OPTIONS": json.dumps(GATEWAY_MODEL_OPTIONS if model_options is None
                                                      else model_options),
        # also for models the gateway process builds outside models.json (the setup wizard's checks)
        "PGW_FLOAT_THREADS": "1",
        # encrypted mask factory: public matrix in NTT form (same ciphertexts bit for bit as "dense", ~10x
        # cheaper on the gateway CPU; see research/plans/he-structured-encryption.md)
        "PGW_FACTORY_HE_ENCRYPT": "ntt_pre",
    })
    return env


def start_gateway(env: dict[str, str], logfile: Path, timeout_s: float = 300) -> tuple[subprocess.Popen[bytes], str]:
    """``pgw gateway serve`` (setup mode); returns the process and the setup token it printed."""
    found: list[str] = []
    ev = threading.Event()

    def on_line(line: str) -> None:
        m = _TOKEN_RE.search(line)
        if m and not found:
            found.append(m.group(1))
            ev.set()

    # The token goes to the log file only; the container's stdout (Modal logs) shows it masked.
    proc = start_logged([sys.executable, "-m", "pgw.cli", "gateway", "serve"], env, logfile, on_line,
                        redact=re.compile(rb"pgw-setup-[A-Za-z0-9_-]{8,}"))
    if not ev.wait(timeout_s):
        proc.kill()
        raise RuntimeError(f"gateway printed no setup token; see {logfile}")
    return proc, found[0]
