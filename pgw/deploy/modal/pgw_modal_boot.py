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

Co-located placement (``colocated.py``): both parties in one container, each as its own Unix user
(:func:`party_dir`, ``user=`` of the start functions) with its own 0700 working directory, and with the
Modal runtime's variables removed from their environments (:func:`clean_env`).

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
#: Targets the demo can serve (``launch.py --model``; ``PGW_MODAL_MODEL`` at deploy time): served name ->
#: (Hugging Face repo, pinned revision). Both parties pack the same bytes: the provider at image build, the
#: customer in the setup wizard from the repo and revision the enrollment lists. ``provider.py`` keeps a copy
#: (it cannot import this file at module level inside its container); a test checks they agree.
MODELS: dict[str, tuple[str, str]] = {
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen2.5-3b-instruct": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
}
#: The speculation draft baked into the customer image with ``PGW_MODAL_DRAFT=1`` (``launch.py --spec-draft``):
#: Qwen2.5-0.5B-Instruct, the same tokenizer as every target above. It runs in plaintext on the gateway only.
DRAFT_NAME = MODEL_NAME
DRAFT_PACK = "/opt/packs/draft"
#: Prompt lookup (longest n-gram) used next to the draft model, or alone without one.
SPEC_LOOKUP = 4


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


def clean_env(prefixes: tuple[str, ...] = ("PGW_",)) -> dict[str, str]:
    """This process's environment without ``prefixes`` and without the Modal runtime's ``MODAL_*``
    variables: the service and the gateway never use Modal, so their processes do not inherit its
    container settings (task id, server URL, ...)."""
    return {k: v for k, v in os.environ.items() if not k.startswith((*prefixes, "MODAL_"))}


def party_dir(path: Path, user: str | None) -> Path:
    """Create ``path`` (mode 0700) and, with ``user``, hand it and everything below it to that user, so
    the other party's user cannot list or read it. Returns ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    if user is not None:
        import pwd

        pw = pwd.getpwnam(user)
        for root, dirs, files in os.walk(path):
            for name in [*dirs, *files]:
                os.chown(Path(root) / name, pw.pw_uid, pw.pw_gid, follow_symlinks=False)
        os.chown(path, pw.pw_uid, pw.pw_gid)
    return path


def as_user(env: dict[str, str], user: str | None, home: Path | None = None) -> dict[str, Any]:
    """``subprocess.Popen`` keyword arguments that run a child as ``user`` (its primary group, no supplementary
    groups, umask 077) with ``HOME`` (and the Hugging Face / Triton caches) under ``home``. ``{}`` without
    a user (the two-container placements: each party already has its own container)."""
    if user is None:
        return {}
    if home is not None:
        env.update({"HOME": str(home), "HF_HOME": str(home / ".cache" / "huggingface"),
                    "TRITON_CACHE_DIR": str(home / ".triton"), "XDG_CACHE_HOME": str(home / ".cache")})
    import pwd

    env["USER"] = env["LOGNAME"] = user
    return {"user": user, "group": pwd.getpwnam(user).pw_gid, "extra_groups": [], "umask": 0o077}


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
    env = clean_env()
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
                 redact: re.Pattern[bytes] | None = None, user_kw: dict[str, Any] | None = None,
                 cwd: Path | None = None) -> subprocess.Popen[bytes]:
    """Start ``cmd``; its output goes to ``logfile`` and stdout (``redact`` matches are masked on stdout).
    ``user_kw`` (:func:`as_user`) runs it as another user; the log file is written by this process."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # noqa: S603
                            stdin=subprocess.DEVNULL, cwd=cwd, **(user_kw or {}))
    # 0600: in the co-located placement this process is root and the log file is the parent's (the
    # gateway's log holds its setup token), never readable by the other party's user.
    fh = os.fdopen(os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab")

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


def start_service(env: dict[str, str], logfile: Path, user_kw: dict[str, Any] | None = None,
                  cwd: Path | None = None) -> subprocess.Popen[bytes]:
    return start_logged([sys.executable, "-m", "pgw.cli", "service", "serve"], env, logfile, user_kw=user_kw,
                        cwd=cwd)


def create_enrollment_code(env: dict[str, str], tenant: str, ttl: str, note: str,
                           user_kw: dict[str, Any] | None = None, cwd: Path | None = None) -> dict[str, Any]:
    r = subprocess.run([sys.executable, "-m", "pgw.cli", "service", "enroll", "create", "--tenant", tenant,  # noqa: S603
                        "--ttl", ttl, "--models", MODEL_NAME, "--note", note, "--json"],
                       env=env, capture_output=True, text=True, check=False, cwd=cwd, **(user_kw or {}))
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
#: Not set by default: strict speculation (``spec_rows`` / ``spec_strict_passes``; ``launch.py --spec-rows``
#: adds it with :func:`spec_options`). The default demo has no draft model and its chat barely
#: repeats the prompt, so prompt lookup accepts almost nothing: the provider sees as many passes either
#: way, while every pass would carry and mask ``spec_rows`` rows (``spec_rows`` times the mask supply).
GATEWAY_MODEL_OPTIONS: dict[str, Any] = {"local_linears": "auto", "float_threads": 1, "lane": "stream"}

#: Gateway trip-lane choices of ``launch.py --lane`` (``tls``: co-located placement only).
LANES = ("stream", "tls")


def lane_options(lane: str) -> dict[str, Any]:
    """Gateway model options for ``launch.py --lane``: ``stream`` (default, as :data:`GATEWAY_MODEL_OPTIONS`) or
    ``tls``, the native TCP+TLS trip lane of a provider started with ``lane_port`` (:func:`start_provider_party`):
    binary header and rustls instead of WebSocket/msgpack and Python ssl, the GIL released per trip, the same
    masked bytes, request ids and schedule; any call it cannot take goes to the blocking stream lane."""
    if lane not in LANES:
        raise ValueError(f"lane must be one of {LANES}, got {lane!r}")
    return {"lane": lane}


def lane_egress(lane: str, lane_port: int | None, host: str = "127.0.0.1") -> list[str]:
    """The gateway's extra egress origin for ``launch.py --lane tls``: the provider's lane port on ``host`` (the
    gateway connects to a TCP lane port only when it is on its egress allowlist). Empty for ``stream``."""
    if lane != "tls":
        return []
    if lane_port is None:
        raise ValueError("lane = 'tls' needs the provider's lane port")
    return [f"https://{host}:{int(lane_port)}"]


#: Customer container memory (MiB): the 0.5B demo fits 8 GiB; a larger target (its float32 embedding and head, the
#: local attention weights, packing it in the wizard) plus the 0.5B draft gets 16 GiB (about $0.064/h more).
CUSTOMER_MEMORY_MIB = {"default": 8192, "large": 16384}


def customer_memory_mib(model: str | None, draft: bool = False) -> int:
    """Memory of the customer container for the deployed target (``PGW_MODAL_MODEL``) and draft."""
    big = bool(draft) or (model or MODEL_NAME).strip().lower() != MODEL_NAME
    return CUSTOMER_MEMORY_MIB["large" if big else "default"]


def e2e_min_masks(spec_rows: int, prompt_tokens: int = 64, max_tokens: int = 32, chats: int = 2) -> int:
    """Mask slots the scripted ``--e2e`` waits for before its chats (``wizard_client.run_wizard(min_masks=)``).

    Without speculation the wizard's default (64) stands: the factory keeps adding slots while the chats run.
    With strict speculation each chat reserves ``prompt + max_tokens + (max_tokens - 1) * (spec_rows - 1)`` slots
    up front and fails fast when the store holds fewer, so the e2e waits for both chats' slots.
    """
    if not spec_rows:
        return 64
    per_chat = int(prompt_tokens) + int(max_tokens) + max(0, int(max_tokens) - 1) * (int(spec_rows) - 1)
    return per_chat * int(chats)


def spec_options(spec_rows: int, draft_pack: str | None = None, draft_device: str | None = None) -> dict[str, Any]:
    """Strict-speculation presets for the wizard's ``models.json`` (``launch.py --spec-rows``; ``{}`` when off).

    Every decode pass then carries exactly ``spec_rows`` masked rows per sequence (the pending token, the
    drafts, masked random filler rows) and, after the answer, dummy passes run until the provider has seen as
    many passes as tokens generated: it sees a fixed row count and the same pass schedule as without
    speculation (SPEC 3.4). Drafts come from prompt lookup over the conversation and, with ``draft_pack``,
    the draft model. Outputs are unchanged (the masked target decides every token). Mask budget: ``spec_rows``
    mask slots per generated token; a request reserves ``prompt + max_tokens + (max_tokens - 1) *
    (spec_rows - 1)`` slots up front.
    """
    r = int(spec_rows)
    if r == 0:
        return {}
    if not 2 <= r <= 8:
        raise ValueError("spec_rows must be 0 (off) or in [2, 8]")
    out: dict[str, Any] = {"spec_rows": r, "spec_strict_passes": True, "spec_lookup": SPEC_LOOKUP}
    if draft_pack:
        out["draft_pack_path"] = str(draft_pack)
        if draft_device:
            out["draft_device"] = draft_device
    return out


def gateway_env(config_dir: Path, data_dir: Path, port: int,
                model_options: dict[str, Any] | None = None) -> dict[str, str]:
    env = clean_env()
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
        # overlap encryption, the provider's work and decryption across linears (measured on a Modal T4 with
        # spare gateway cores: 64.5 -> 94.1 slots/s for the full 0.5B, masks bit-identical to the plaintext factory)
        "PGW_FACTORY_HE_PIPELINE_DEPTH": "8",
        "PGW_FACTORY_HE_MAX_INFLIGHT": "4",
    })
    return env


def start_gateway(env: dict[str, str], logfile: Path, timeout_s: float = 300, user_kw: dict[str, Any] | None = None,
                  cwd: Path | None = None) -> tuple[subprocess.Popen[bytes], str]:
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
                        redact=re.compile(rb"pgw-setup-[A-Za-z0-9_-]{8,}"), user_kw=user_kw, cwd=cwd)
    if not ev.wait(timeout_s):
        proc.kill()
        raise RuntimeError(f"gateway printed no setup token; see {logfile}")
    return proc, found[0]


# ------------------------------------------------------------------------------------------------
# which network path a service URL takes (reported next to every RTT figure)
# ------------------------------------------------------------------------------------------------
def network_path(service_url: str | None) -> str:
    """Label of the path a gateway takes to ``service_url``: ``loopback`` (both parties on one host:
    the co-located placement), ``private <addr>`` (a private or unique-local address: Modal's i6pn, or
    an intra-cluster address), ``public <host>`` (e.g. the public TCP tunnel's relay host) or
    ``unknown``."""
    from urllib.parse import urlsplit

    try:
        host = urlsplit(service_url or "").hostname
    except ValueError:
        host = None
    if not host:
        return "unknown"
    if host == "localhost":
        return "loopback"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return f"public {host}"
    if ip.is_loopback:
        return "loopback"
    if ip.is_private:
        return f"private {ip}"
    return f"public {ip}"


# ------------------------------------------------------------------------------------------------
# co-located placement: both parties on one host, each as its own user (colocated.py, local_rehearsal.py)
# ------------------------------------------------------------------------------------------------
PROVIDER_USER, CUSTOMER_USER = "pgwprov", "pgwcust"


def ensure_user(name: str) -> None:
    """Create system user ``name`` (own group, no login shell, no home) unless it exists (root only)."""
    import pwd

    try:
        pwd.getpwnam(name)
    except KeyError:
        subprocess.run(["useradd", "--system", "--user-group", "--shell", "/usr/sbin/nologin",  # noqa: S603, S607
                        "--no-create-home", name], check=True)


def start_provider_party(work: Path, pack: Path, port: int, *, device: str, user: str | None, log_dir: Path,
                         tenant: str = "customer-demo", code_ttl: str = "90m", note: str = "co-located demo",
                         public_url: str | None = None, lane_port: int | None = None) -> dict[str, Any]:
    """The provider's compute service on ``127.0.0.1:port`` only (TLS, a fresh PKI for 127.0.0.1), run as
    ``user`` in ``work`` (0700, owned by that user once its PKI and state files exist), plus a fresh
    single-use enrollment code. ``public_url`` (default ``https://127.0.0.1:port``) is the URL the code
    points the gateway at (e.g. a delay proxy on loopback). ``lane_port``: also serve the native trip lane
    over TLS 1.3 on ``127.0.0.1:lane_port`` (``lane = "tcp"``: the service's certificate, client certificates
    verified against the client-issuing CA and bound to the lane ticket). Returns ``proc``, ``url``, ``code``,
    ``pki``, ``env`` and ``ca_sha256``."""
    party_dir(work / "home", None)
    pki = make_pki(work / "pki", ["127.0.0.1", "localhost"])
    url = public_url or f"https://127.0.0.1:{port}"
    env = service_env(work, pack, pki, url, port, device=device)
    env["PGW_SERVICE_HOST"] = "127.0.0.1"  # nothing outside this host can reach it (the lane port too)
    if lane_port is not None:
        env["PGW_SERVICE_LANE"] = "tcp"
        env["PGW_SERVICE_LANE_PORT"] = str(int(lane_port))
    party_dir(work, user)  # everything below is the provider's from here on
    kw = as_user(env, user, work / "home")
    party_dir(log_dir, None)
    proc = start_service(env, log_dir / "service.log", user_kw=kw, cwd=work)
    t0 = time.time()
    wait_ready(f"https://127.0.0.1:{port}/readyz", proc, verify=str(pki["service_ca"]), timeout_s=900)
    log(f"service ready in {time.time() - t0:.0f}s (127.0.0.1:{port}, user {user or 'same'})")
    code = create_enrollment_code(env, tenant, code_ttl, note, user_kw=kw, cwd=work)
    from pgw.enroll.service import ca_fingerprint, load_ca_pem

    return {"proc": proc, "url": url, "code": code, "pki": pki, "env": env,
            "ca_sha256": ca_fingerprint(load_ca_pem(pki["service_ca"])).hex()}


def start_gateway_party(root: Path, port: int, *, user: str | None, log_dir: Path,
                        model_options: dict[str, Any] | None = None,
                        hide_gpu: bool = True, egress: list[str] | None = None) -> tuple[subprocess.Popen[bytes], str]:
    """The customer's gateway in setup mode as ``user`` in ``root`` (0700), without the provider's GPU
    (``hide_gpu``). ``egress``: extra allowed outbound origins (the gateway's ``egress_allowlist``; e.g. the
    provider's TCP+TLS lane port, :func:`lane_egress`). Returns the process and its one-time setup token."""
    party_dir(root / "home", None)
    party_dir(root, user)
    env = gateway_env(root / "config", root / "data", port, model_options)
    if egress:
        env["PGW_GATEWAY_EGRESS_ALLOWLIST"] = json.dumps(list(egress))
    if hide_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""  # the provider's GPU is not the customer's
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # as in customer.py's image
    kw = as_user(env, user, root / "home")
    party_dir(log_dir, None)
    gw, token = start_gateway(env, log_dir / "gateway.log", user_kw=kw, cwd=root)
    wait_ready(f"http://127.0.0.1:{port}/healthz", gw)
    log(f"gateway in setup mode on :{port} (user {user or 'same'})")
    return gw, token
