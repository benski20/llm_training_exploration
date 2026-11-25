#!/opt/pgw/bin/python
"""Inside a RunPod pod: prepare the provider's volume, start the compute service, stop the pod at its TTL.

    pgw-runpod-boot            # provider (GPU pod; the image's start command set by launch.py)
    pgw-runpod-boot client     # client-side test pod (CPU, same data center): idles until its TTL, then stops

Provider boot (as root, the container's PID 1 under tini; the service itself runs as uid 10001):

1. Volume ``PGW_RUNPOD_DATA`` (default ``/workspace/pgw``): ``pki/`` (the service CA and the client-issuing CA,
   created once and kept, so SDK keys, which pin the CA, survive pod restarts; the server certificate is re-issued
   at every boot for the pod's current public address), ``state/`` (tenant tokens, accounts database), ``packs/``
   (built once from the pinned Hugging Face revisions with ``pgw pack``), ``logs/``.
2. ``pgw service serve`` with production settings (:func:`service_env`): TLS end to end on 8443, the native trip
   lane over TLS 1.3 on 8444 (advertised under RunPod's public port mapping), accounts (SDK keys) on, the admin API
   on only when the launcher passed the SHA-256 of a bootstrap admin token, and the provider tuning of
   :data:`PROVIDER_TUNING_ENV`.
3. Server-side TTL: at ``PGW_RUNPOD_STOP_AT`` (unix seconds, set by the launcher; restarts do not extend it) the
   service is drained and the pod stops itself through the RunPod REST API with the pod-scoped ``RUNPOD_API_KEY``
   that RunPod injects into every pod. The launcher's client-side watchdog is the second, independent guard.

Secrets: the pod API key is used by this process only (the service's environment is built from scratch and never
contains it); the CA keys stay on the volume (0600); nothing secret is printed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

#: Served targets, pinned (the same table as deploy/modal/pgw_modal_boot.MODELS; a test keeps them equal): both
#: parties must pack the same bytes from the same revision.
MODELS: dict[str, tuple[str, str]] = {
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen2.5-3b-instruct": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "qwen2.5-32b-instruct": ("Qwen/Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"),
}
#: The release's SDK models (docs/sdk/CONTRACT.md): the launcher's default.
SDK_MODELS = ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct")

SERVICE_PORT = 8443
LANE_PORT = 8444
METRICS_PORT = 9090
SERVICE_UID = SERVICE_GID = 10001
DEFAULT_DATA = "/workspace/pgw"
REST_URL = "https://rest.runpod.io/v1"
#: Lifetime of the server certificate issued at each boot, and of the two CAs (created once on the volume).
SERVER_CERT_DAYS = 90
CA_DAYS = 3650

#: Provider tuning for a plain (non-sandboxed) container serving REMOTE clients. Carried over from the Modal runs
#: where it applies across machines; every entry is exact (same kernels, bytes and replies) and was run there:
#:
#: * ``PGW_SERVICE_LANE=tcp`` (+ port 8444, advertised as RunPod's public mapping of 8444): the native trip lane
#:   over TLS 1.3 (rustls, 48-byte header, CUDA-graph replay of 1-row calls without Python), the Modal ``--lane
#:   tls`` path. The tickets come from ``lane/open`` (the tenant's bearer token) and are bound to the client
#:   certificate presented there. Clients that do not ask for the lane use the stream channel as before.
#: * ``PGW_SERVICE_TRIP_STREAM_PRIORITY=high``: decode graphs on the GPU's highest-priority stream, ahead of queued
#:   encrypted-maskgen GEMMs (measured on Modal with a factory running; scheduling only).
#: * ``PGW_SERVICE_MASKED_HEAD=also``: every pack is also served under its masked-LM-head fingerprint (the 7B 28 tok/s
#:   stack's masked head; gateways opt in with head_mode = provider_int8; additive: +vocab x hidden int8 of memory).
#: * ``PGW_SERVICE_MASKGEN_V3=true``: maskgen wire v3 for the SDK's pks-v3 factory (the client library's default with
#:   its explicit acknowledgement; pks-v3 is PENDING EXTERNAL REVIEW). The provider sees pseudorandom ciphertexts
#:   either way; v1/v2 stay served.
#:
#: Not carried over: the shared-memory lane and the UDS lane (same host only), ``PGW_SERVICE_LANE_ZERO_COPY`` (UDS
#: replies only), and the opt-in latency knobs that were not proven on a GPU (``PGW_SERVICE_LANE_READ_SPIN_US``,
#: ``PGW_SERVICE_LANE_OWN_LOCK``, ``PGW_SERVICE_CUDA_GRAPHS``): set them per run with ``launch.py --service-env``.
PROVIDER_TUNING_ENV: dict[str, str] = {
    "PGW_SERVICE_LANE": "tcp",
    "PGW_SERVICE_LANE_PORT": str(LANE_PORT),
    "PGW_SERVICE_TRIP_STREAM_PRIORITY": "high",
    "PGW_SERVICE_MASKED_HEAD": "also",
    "PGW_SERVICE_MASKGEN_V3": "true",
}
#: Prefixes ``PGW_RUNPOD_SERVICE_ENV`` (launch.py --service-env) may set; nothing else (no PATH, LD_PRELOAD, ...).
EXTRA_ENV_PREFIXES = ("PGW_SERVICE_", "PGW_LOG_", "OMP_", "MKL_", "CUDA_")
#: Settings the launcher's overrides may not change (identity, TLS, accounts: the boot owns them).
PROTECTED_SETTINGS = frozenset({
    "PGW_SERVICE_TLS__CERT_FILE", "PGW_SERVICE_TLS__KEY_FILE", "PGW_SERVICE_TLS__CA_FILE", "PGW_SERVICE_PUBLIC_URL",
    "PGW_SERVICE_ENROLLMENT_CA_FILE", "PGW_SERVICE_CLIENT_ISSUING_CA_CERT_FILE",
    "PGW_SERVICE_CLIENT_ISSUING_CA_KEY_FILE",
    "PGW_SERVICE_ACCOUNTS_DB", "PGW_SERVICE_TOKENS_FILE", "PGW_SERVICE_MTLS_TENANTS_FILE",
    "PGW_SERVICE_ACCOUNTS_ADMIN_TOKEN_SHA256", "PGW_SERVICE_ALLOW_UNAUTHENTICATED", "PGW_SERVICE_PACKS",
})


def log(msg: str) -> None:
    print(f"[pgw-runpod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------- config
class BootError(RuntimeError):
    """The pod cannot start the provider (message for the operator; never contains a secret)."""


def checked_service_env(raw: str | None) -> dict[str, str]:
    """``PGW_RUNPOD_SERVICE_ENV`` (a JSON object) as strings; only :data:`EXTRA_ENV_PREFIXES`, never
    :data:`PROTECTED_SETTINGS`."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise BootError("PGW_RUNPOD_SERVICE_ENV must be a JSON object") from exc
    if not isinstance(data, dict):
        raise BootError("PGW_RUNPOD_SERVICE_ENV must be a JSON object")
    out: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not k.startswith(EXTRA_ENV_PREFIXES) or k in PROTECTED_SETTINGS:
            raise BootError(f"PGW_RUNPOD_SERVICE_ENV: {k!r} is not allowed")
        out[k] = str(v)
    return out


def parse_models(raw: str | None) -> list[str]:
    models = [m.strip().lower() for m in (raw or ",".join(SDK_MODELS)).split(",") if m.strip()]
    bad = [m for m in models if m not in MODELS]
    if bad or not models:
        raise BootError(f"unknown model(s) {bad}; use some of {', '.join(MODELS)}")
    return list(dict.fromkeys(models))


def public_endpoint(env: dict[str, str]) -> tuple[str, int, int | None]:
    """``(host, public service port, public lane port)`` from RunPod's variables (``RUNPOD_PUBLIC_IP``,
    ``RUNPOD_TCP_PORT_8443``, ``RUNPOD_TCP_PORT_8444``) or ``PGW_RUNPOD_PUBLIC_HOST`` (a DNS name pointing at the
    pod)."""
    host = (env.get("PGW_RUNPOD_PUBLIC_HOST") or env.get("RUNPOD_PUBLIC_IP") or "").strip()
    port = (env.get(f"RUNPOD_TCP_PORT_{SERVICE_PORT}") or "").strip()
    lane = (env.get(f"RUNPOD_TCP_PORT_{LANE_PORT}") or "").strip()
    if not host or not port.isdigit():
        raise BootError(f"no public address: start the pod with a public IP and TCP port {SERVICE_PORT} exposed "
                        f"(RUNPOD_PUBLIC_IP / RUNPOD_TCP_PORT_{SERVICE_PORT} are not set)")
    return host, int(port), int(lane) if lane.isdigit() else None


def url_for(host: str, port: int) -> str:
    try:
        if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
            return f"https://[{host}]:{port}"
    except ValueError:
        pass
    return f"https://{host}:{port}"


# ------------------------------------------------------------------------------------------------------ PKI
def _write_private(path: Path, data: bytes, uid: int | None = None) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    if uid is not None and os.geteuid() == 0:
        os.chown(tmp, uid, uid)
    os.replace(tmp, path)


def _new_ca(cn: str) -> tuple[Any, Any]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=CA_DAYS))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    return key, cert


def _load_or_create_ca(d: Path, stem: str, cn: str, key_uid: int | None) -> tuple[Any, Any]:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    kp, cp = d / f"{stem}.key", d / f"{stem}.pem"
    if kp.exists() and cp.exists():
        return (serialization.load_pem_private_key(kp.read_bytes(), password=None),
                x509.load_pem_x509_certificate(cp.read_bytes()))
    key, cert = _new_ca(cn)
    _write_private(kp, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()), key_uid)
    cp.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    os.chmod(cp, 0o644)
    return key, cert


def ensure_pki(pki_dir: Path, names: list[str], *, service_uid: int | None = None) -> dict[str, Path]:
    """The provider PKI on the volume: the service CA (kept: SDK keys pin it), the client-issuing CA (kept; its key
    is readable by the service user, which signs installation certificates) and a fresh server certificate for
    ``names`` (the public address first). Returns the file paths."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    pki_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    ca_key, ca_cert = _load_or_create_ca(pki_dir, "service-ca", "PGW provider service CA", None)
    _load_or_create_ca(pki_dir, "client-ca", "PGW provider client-issuing CA", service_uid)
    key = ec.generate_private_key(ec.SECP256R1())
    sans: list[x509.GeneralName] = []
    for h in dict.fromkeys(names):
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0][:64])]))
            .issuer_name(ca_cert.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=SERVER_CERT_DAYS))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    out = {"service_ca": pki_dir / "service-ca.pem", "service_ca_key": pki_dir / "service-ca.key",
           "server_cert": pki_dir / "server.pem", "server_key": pki_dir / "server.key",
           "client_ca": pki_dir / "client-ca.pem", "client_ca_key": pki_dir / "client-ca.key"}
    out["server_cert"].write_bytes(cert.public_bytes(serialization.Encoding.PEM)
                                   + ca_cert.public_bytes(serialization.Encoding.PEM))
    os.chmod(out["server_cert"], 0o644)
    _write_private(out["server_key"], key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                        serialization.NoEncryption()), service_uid)
    if service_uid is not None and os.geteuid() == 0:
        os.chown(pki_dir, 0, service_uid)
        os.chmod(pki_dir, 0o750)  # noqa: S103 - group = the service user only; the service user reads its files; the service CA key stays root's (0600)
    return out


def ca_sha256(pem_path: Path) -> str:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    cert = x509.load_pem_x509_certificate(pem_path.read_bytes())
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


# ---------------------------------------------------------------------------------------------- service env
def service_env(*, data: Path, pki: dict[str, Path], packs: list[Path], public_url: str, lane_public_port: int | None,
                admin_token_sha256: str | None, key_env: str = "live", extra: dict[str, str] | None = None,
                device: str = "auto") -> dict[str, str]:
    """The service's complete environment (built from scratch: nothing of the boot's own environment leaks in, in
    particular not ``RUNPOD_API_KEY``)."""
    state = data / "state"
    env = {
        "PATH": "/opt/pgw/bin:/usr/local/bin:/usr/bin:/bin", "HOME": str(data / "home"), "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1", "PGW_LOG_LEVEL": "INFO", "PGW_LOG_JSON": "true",
        "PGW_SERVICE_HOST": "0.0.0.0",  # noqa: S104 - published through RunPod's public port mapping (TLS)
        "PGW_SERVICE_PORT": str(SERVICE_PORT),
        "PGW_SERVICE_METRICS_PORT": str(METRICS_PORT), "PGW_SERVICE_METRICS_HOST": "127.0.0.1",
        "PGW_SERVICE_PACKS": json.dumps([str(p) for p in packs]),
        "PGW_SERVICE_DEVICE": device,
        "PGW_SERVICE_TOKENS_FILE": str(state / "tenants.txt"),
        "PGW_SERVICE_MTLS_TENANTS_FILE": str(state / "mtls-tenants.txt"),
        "PGW_SERVICE_TLS__CERT_FILE": str(pki["server_cert"]),
        "PGW_SERVICE_TLS__KEY_FILE": str(pki["server_key"]),
        "PGW_SERVICE_TLS__CA_FILE": str(pki["client_ca"]),  # client certificates verified when presented
        "PGW_SERVICE_TLS__REQUIRE_CLIENT_CERT": "false",  # SDK installations have none before they enroll
        "PGW_SERVICE_PUBLIC_URL": public_url,
        "PGW_SERVICE_ENROLLMENT_CA_FILE": str(pki["service_ca"]),
        "PGW_SERVICE_CLIENT_ISSUING_CA_CERT_FILE": str(pki["client_ca"]),
        "PGW_SERVICE_CLIENT_ISSUING_CA_KEY_FILE": str(pki["client_ca_key"]),
        "PGW_SERVICE_CLIENT_CERT_VALIDITY_DAYS": "30",
        "PGW_SERVICE_ACCOUNTS_DB": str(state / "accounts.db"),
        "PGW_SERVICE_SDK_KEY_ENV": key_env,
        **PROVIDER_TUNING_ENV,
    }
    if lane_public_port is not None:
        env["PGW_SERVICE_LANE_ADVERTISED_PORT"] = str(lane_public_port)
    else:  # no public lane port: the lane would announce an unreachable port
        env["PGW_SERVICE_LANE"] = "off"
        env.pop("PGW_SERVICE_LANE_PORT", None)
    if admin_token_sha256:
        env["PGW_SERVICE_ACCOUNTS_ADMIN_API"] = "true"
        env["PGW_SERVICE_ACCOUNTS_ADMIN_TOKEN_SHA256"] = json.dumps([admin_token_sha256])
    for k, v in (extra or {}).items():
        if k in PROTECTED_SETTINGS:
            raise BootError(f"{k} is set by the boot and cannot be overridden")
        env[k] = v
    return env


# -------------------------------------------------------------------------------------------- self-stop
def stop_self(pod_id: str, api_key: str, *, opener: Any = None, timeout: float = 30.0) -> bool:
    """Stop this pod through the RunPod REST API (``POST /pods/{id}/stop``) with the pod-scoped key. True on
    success. The key is sent in a header only and never logged."""
    req = urllib.request.Request(f"{REST_URL}/pods/{pod_id}/stop", method="POST", data=b"{}",  # noqa: S310
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
            return 200 <= int(getattr(r, "status", 200)) < 300
    except (urllib.error.URLError, OSError) as exc:
        log(f"self-stop through the RunPod API failed: {type(exc).__name__} "
            f"{getattr(exc, 'code', '')}".rstrip())
        return False


class StopTimer:
    """Server-side TTL: at ``stop_at`` call ``on_stop`` (drain the service), then stop the pod through the API.
    ``stop_at`` comes from the launcher (absolute), so a restarted container keeps the original deadline."""

    def __init__(self, stop_at: float | None, pod_id: str | None, api_key: str | None, on_stop: Any, *,
                 clock: Any = time.time, sleep: Any = time.sleep, stopper: Any = stop_self):
        self.stop_at, self.pod_id, self._key = stop_at, pod_id, api_key
        self.on_stop, self.clock, self.sleep, self.stopper = on_stop, clock, sleep, stopper
        self.fired = threading.Event()

    def run(self) -> None:
        if self.stop_at is None:
            return
        while (left := self.stop_at - self.clock()) > 0:
            self.sleep(min(left, 30.0))
        self.fired.set()
        log("TTL reached: draining the service and stopping this pod")
        try:
            self.on_stop()
        finally:
            ok = False
            if self.pod_id and self._key:
                for attempt in range(5):
                    if self.stopper(self.pod_id, self._key):
                        ok = True
                        break
                    self.sleep(min(60.0, 5.0 * 2 ** attempt))
            if not ok:
                log("WARNING: could not stop the pod through the RunPod API (no pod-scoped RUNPOD_API_KEY or the "
                    "API refused). The GPU keeps billing until the launcher's watchdog or `launch.py --stop` stops "
                    "it")

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="pgw-runpod-ttl", daemon=True)
        t.start()
        return t


# --------------------------------------------------------------------------------------------- packs
def _demote(uid: int | None) -> Any:
    if uid is None or os.geteuid() != 0:
        return None

    def fn() -> None:
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        try:  # PR_SET_NO_NEW_PRIVS: nothing the service runs can regain privileges
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0)
        except (OSError, AttributeError):
            pass

    return fn


def ensure_pack(model: str, packs: Path, hf_home: Path, *, uid: int | None = None) -> Path:
    """``packs/<model>`` built once from the pinned revision (``pgw pack``), as the service user."""
    repo, rev = MODELS[model]
    dst = packs / model
    if (dst / "manifest.json").exists():
        return dst
    tmp = packs / f".{model}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"packing {model} ({repo}@{rev[:12]}) into the volume (first boot only)")
    env = {"PATH": "/opt/pgw/bin:/usr/local/bin:/usr/bin:/bin", "HOME": str(hf_home), "HF_HOME": str(hf_home),
           "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1"}
    for k in ("HF_TOKEN", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pgw.cli", "pack", repo, str(tmp), "--name", model, "--revision", rev,  # noqa: S603
                    "-q"], env=env, check=True, preexec_fn=_demote(uid))
    os.replace(tmp, dst)
    shutil.rmtree(hf_home / "hub", ignore_errors=True)  # the checkpoint is not needed once packed
    log(f"packed {model} in {time.time() - t0:.0f}s")
    return dst


# ------------------------------------------------------------------------------------------------ main
def _prepare_dirs(data: Path, uid: int | None) -> None:
    for sub, mode in (("", 0o755), ("state", 0o700), ("packs", 0o755), ("hf", 0o700), ("logs", 0o700),
                      ("home", 0o700)):
        p = data / sub if sub else data
        p.mkdir(parents=True, exist_ok=True)
        os.chmod(p, mode)
        if uid is not None and os.geteuid() == 0 and sub:
            os.chown(p, uid, uid)
    for f in ("tenants.txt", "mtls-tenants.txt"):
        p = data / "state" / f
        if not p.exists():
            os.close(os.open(p, os.O_WRONLY | os.O_CREAT, 0o600))
        if uid is not None and os.geteuid() == 0:
            os.chown(p, uid, uid)


def _wait_ready(port: int, proc: subprocess.Popen[bytes], timeout_s: float) -> None:
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # loopback readiness probe of our own listener
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise BootError(f"the service exited with status {proc.returncode} before it was ready")
        try:
            with urllib.request.urlopen(f"https://127.0.0.1:{port}/readyz", timeout=5, context=ctx) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2.0)
    raise BootError(f"the service was not ready after {timeout_s:.0f}s")


def provider_main(env: dict[str, str]) -> int:
    data = Path(env.get("PGW_RUNPOD_DATA") or DEFAULT_DATA)
    uid = SERVICE_UID if os.geteuid() == 0 else None
    models = parse_models(env.get("PGW_RUNPOD_MODELS"))
    host, port, lane_port = public_endpoint(env)
    url = url_for(host, port)
    stop_at = float(env["PGW_RUNPOD_STOP_AT"]) if env.get("PGW_RUNPOD_STOP_AT") else None
    admin_hash = (env.get("PGW_RUNPOD_ADMIN_TOKEN_SHA256") or "").strip().lower() or None
    extra = checked_service_env(env.get("PGW_RUNPOD_SERVICE_ENV"))
    key_env = env.get("PGW_RUNPOD_KEY_ENV") or "live"
    log(f"provider boot: models {', '.join(models)}, public URL {url}, lane port {lane_port}, "
        f"TTL {'until ' + time.strftime('%H:%M:%S UTC', time.gmtime(stop_at)) if stop_at else 'none (!)'}")
    proc: subprocess.Popen[bytes] | None = None

    def drain() -> None:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()

    timer = StopTimer(stop_at, env.get("RUNPOD_POD_ID"), env.get("RUNPOD_API_KEY"), drain)
    timer.start()  # before the (long) first-boot packing: the TTL holds from the start
    if stop_at is not None and stop_at <= time.time():
        timer.fired.wait(120)
        return 0
    _prepare_dirs(data, uid)
    pki = ensure_pki(data / "pki", [host, "localhost", "127.0.0.1"], service_uid=uid)
    packs = [ensure_pack(m, data / "packs", data / "hf", uid=uid) for m in models]
    senv = service_env(data=data, pki=pki, packs=packs, public_url=url, lane_public_port=lane_port,
                       admin_token_sha256=admin_hash, key_env=key_env, extra=extra)
    status = {"service_url": url, "ca_sha256": ca_sha256(pki["service_ca"]), "models": models,
              "lane_port": lane_port, "started_at": time.time(), "stop_at": stop_at,
              "pod_id": env.get("RUNPOD_POD_ID")}
    (data / "state" / "boot.json").write_text(json.dumps(status, indent=1) + "\n")
    logf = open(data / "logs" / "service.log", "ab")
    proc = subprocess.Popen([sys.executable, "-m", "pgw.cli", "service", "serve"], env=senv,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            preexec_fn=_demote(uid), cwd=str(data))

    def tee() -> None:
        assert proc is not None and proc.stdout is not None
        for line in iter(proc.stdout.readline, b""):
            logf.write(line)
            logf.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()

    threading.Thread(target=tee, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: drain())
    try:
        _wait_ready(SERVICE_PORT, proc, timeout_s=1800)
        log(f"provider ready at {url} (CA sha256 {status['ca_sha256'][:16]}...)")
    except BootError as exc:
        log(f"ERROR: {exc}")
    rc = proc.wait()
    if timer.fired.is_set():
        timer.fired.wait(1)
        time.sleep(600)  # the API stop is on its way; do not restart the service meanwhile
        return 0
    log(f"service exited with status {rc}")
    return rc or 1


def client_main(env: dict[str, str]) -> int:
    """The client-side test pod: idle (``rtt_probe.py`` and the client library are in the image) until the TTL."""
    stop_at = float(env["PGW_RUNPOD_STOP_AT"]) if env.get("PGW_RUNPOD_STOP_AT") else None
    log("client test pod: run `python /opt/pgw-deploy/runpod/rtt_probe.py <provider URL> --ca-sha256 <hex>` and the "
        "client library here")
    timer = StopTimer(stop_at, env.get("RUNPOD_POD_ID"), env.get("RUNPOD_API_KEY"), lambda: None)
    t = timer.start()
    if stop_at is None:
        signal.pause()
    t.join()
    time.sleep(600)
    return 0


def main(argv: list[str]) -> int:
    env = dict(os.environ)
    try:
        if argv[:1] == ["client"]:
            return client_main(env)
        return provider_main(env)
    except BootError as exc:
        log(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
