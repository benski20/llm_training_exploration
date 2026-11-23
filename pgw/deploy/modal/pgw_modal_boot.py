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
    "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "qwen2.5-32b-instruct": ("Qwen/Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"),
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
                           user_kw: dict[str, Any] | None = None, cwd: Path | None = None,
                           model: str | None = None) -> dict[str, Any]:
    """A single-use enrollment code for ``model`` (default: the deployed target, ``PGW_MODAL_MODEL`` in the
    image's environment, else the 0.5B; the code lists the model the service serves)."""
    model = (model or os.environ.get("PGW_MODAL_MODEL") or MODEL_NAME).strip().lower()
    r = subprocess.run([sys.executable, "-m", "pgw.cli", "service", "enroll", "create", "--tenant", tenant,  # noqa: S603
                        "--ttl", ttl, "--models", model, "--note", note, "--json"],
                       env=env, capture_output=True, text=True, check=False, cwd=cwd, **(user_kw or {}))
    if r.returncode != 0:
        raise RuntimeError(f"enroll create failed: {r.stderr[-2000:]}")
    out: dict[str, Any] = json.loads(r.stdout)
    return out


# ------------------------------------------------------------------------------------------------
# customer: gateway in setup mode
# ------------------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])(pgw-setup-[A-Za-z0-9_-]{8,})(?![A-Za-z0-9_-])")


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

#: The fused2 ring (``launch.py --schedule fused2``; research/plans/fused2-two-trips-per-layer.md section 7).
FUSED2_RING = "K56A12C53"


def fused2_model_options(migrate_bits: int, lane: str) -> dict[str, Any]:
    """Gateway model options of ``launch.py --schedule fused2``: the fused2 schedule on its K56 ring (the native float
    ops are implied), every trip remote (``local_linears = "off"``), the lane (``uds`` -> ``auto``, ``tls``)."""
    if lane not in ("uds", "tls"):
        raise ValueError("schedule fused2 needs --lane uds or tls (pair items ride the trip lane only)")
    return {"schedule": "fused2", "ring": FUSED2_RING, "fused2_migrate_bits": int(migrate_bits),
            "local_linears": "off", "lane": "auto" if lane == "uds" else "tls"}


def fused2_provider_env(migrate_bits: int, maskgen: bool = True) -> dict[str, str]:
    """Provider environment of ``launch.py --schedule fused2``: the fused2 lane entry derived from its own pack
    (the same public weights the gateway and the factory derive) and, for the encrypted fused2 factory, maskgen for
    its fused2 linears (``PGW_SERVICE_FUSED2_MASKGEN``: ordinary maskgen items under the fused2 lane fingerprint)."""
    env = {"PGW_SERVICE_FUSED2_FROM_PACK": str(int(migrate_bits)), "PGW_SERVICE_FUSED2_RING": FUSED2_RING}
    if maskgen:
        env["PGW_SERVICE_FUSED2_MASKGEN"] = "1"
    return env


#: Gateway trip-lane choices of ``launch.py --lane`` (``tls``, ``uds``: co-located placement only).
LANES = ("stream", "tls", "uds")


def lane_options(lane: str, native: bool = False) -> dict[str, Any]:
    """Gateway model options for ``launch.py --lane``: ``stream`` (default, as :data:`GATEWAY_MODEL_OPTIONS`) or
    ``tls``, the native TCP+TLS trip lane of a provider started with ``lane_port`` (:func:`start_provider_party`):
    binary header and rustls instead of WebSocket/msgpack and Python ssl, the GIL released per trip, the same
    masked bytes, request ids and schedule; any call it cannot take goes to the blocking stream lane. ``uds``: the
    product's same-host lane (a provider started with ``lane_socket``): the gateway's ``lane = "auto"`` takes the
    UDS ticket of its loopback service (peer uid checked by the service), else the stream channel. ``native``
    (``uds`` only): ``lane_native`` = the native ``TripChannel`` instead of the Python socket channel (same bytes;
    needed by the per-trip native driver ``PGW_TRIP_DRIVER``)."""
    if lane not in LANES:
        raise ValueError(f"lane must be one of {LANES}, got {lane!r}")
    if lane == "uds":
        return {"lane": "auto", "lane_native": True} if native else {"lane": "auto"}
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


def spec_mask_options(value: str) -> dict[str, Any]:
    """``launch.py --spec-mask C[/B]`` (deploy time, ``PGW_MODAL_SPEC_MASK``): padded strict speculation with a
    public mask charge of C slots per decode pass, the draft credit capped at B (L3,
    ``research/plans/speculation-no-compromise.md``). ``""``: masked filler rows (``spec_rows`` slots per pass)."""
    v = (value or "").strip()
    if not v:
        return {}
    c, _, b = v.partition("/")
    out: dict[str, Any] = {"spec_mask_rows": float(c)}
    if b.strip():
        out["spec_mask_credit"] = int(b)
    return out


#: ``launch.py --draft-backend`` choices (``GatewayModelConfig.draft_backend``).
DRAFT_BACKENDS = ("auto", "native_int8", "torch_int8", "float")


def spec_options(spec_rows: int, draft_pack: str | None = None, draft_device: str | None = None, *,
                 draft_backend: str | None = None, pipeline_steps: int | None = None,
                 spec_mask: str | None = None, draft_head: str | None = None) -> dict[str, Any]:
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
    # spec_mask: a run-time ``C[/B]`` (launch.py --spec-mask, passed with the run); None = the deploy-time setting
    out.update(spec_mask_options(os.environ.get("PGW_MODAL_SPEC_MASK", "") if spec_mask is None else spec_mask))
    if draft_head:  # launch.py --draft-head: a trained target-feature head drafts (no draft model, no lookup)
        if draft_pack or pipeline_steps is not None:
            raise ValueError("--draft-head excludes the draft model and pipelined drafting")
        out.update({"draft_backend": "feature_head", "draft_head_path": str(draft_head), "spec_lookup": 0})
        return out
    if draft_pack:
        out["draft_pack_path"] = str(draft_pack)
        if draft_device:
            out["draft_device"] = draft_device
        if draft_backend:  # launch.py --draft-backend (e.g. native_int8: the CPU draft on the native kernels)
            if draft_backend not in DRAFT_BACKENDS:
                raise ValueError(f"draft_backend must be one of {DRAFT_BACKENDS}, got {draft_backend!r}")
            out["draft_backend"] = draft_backend
        if pipeline_steps is not None:  # launch.py --spec-pipeline-steps m: the pipelined drafter (no prompt lookup)
            m = int(pipeline_steps)
            if not 0 <= m <= r - 1:
                raise ValueError("spec pipeline steps must be in [0, spec_rows - 1]")
            out["spec_pipeline_steps"] = m
            out["spec_lookup"] = 0
    elif draft_backend or pipeline_steps is not None:
        raise ValueError("draft_backend / spec pipeline steps need the draft model (launch.py --spec-draft)")
    return out


#: Model options a plain (non-speculative) gateway does not carry: removed from ``models.json`` for the in-run
#: plain phase (``launch.py --ab-plain``), which restarts the gateway process on the same configuration otherwise.
SPEC_MODEL_KEYS = ("spec_rows", "spec_strict_passes", "spec_mask_rows", "spec_mask_credit", "spec_lookup",
                   "spec_lookup_min_n", "spec_tokens", "draft_pack_path", "draft_device", "draft_backend",
                   "spec_pipeline_steps", "draft_head_path")


def drop_speculation(models_file: Path) -> list[str]:
    """Remove :data:`SPEC_MODEL_KEYS` from every entry of the wizard's ``models.json`` (in place: same owner and
    mode). Returns the keys removed. The rest (pack, fingerprint, service, credentials, speed settings) is kept."""
    entries = json.loads(models_file.read_text())
    gone: set[str] = set()
    for e in entries if isinstance(entries, list) else [entries]:
        for k in SPEC_MODEL_KEYS:
            if k in e:
                gone.add(k)
                del e[k]
    with open(models_file, "r+") as fh:  # rewrite in place: keeps the gateway user's ownership and 0600
        fh.seek(0)
        fh.write(json.dumps(entries, indent=2) + "\n")
        fh.truncate()
    return sorted(gone)


def restart_gateway(proc: subprocess.Popen[bytes], port: int, log_dir: Path,
                    timeout_s: float = 900.0) -> subprocess.Popen[bytes]:
    """Stop the gateway process ``proc`` (SIGTERM: it drains, closes its mask store and stops its factory) and
    start ``pgw gateway serve`` again with the same environment, user and working directory (it then runs in
    normal mode from the finished wizard's configuration: same mask store, keys and credentials). Returns the new
    process once ``/healthz`` answers; the model may still be loading (wait for ``/readyz``)."""
    env, kw, cwd = proc.pgw_env, proc.pgw_user_kw, proc.pgw_cwd  # type: ignore[attr-defined]
    proc.terminate()
    try:
        proc.wait(180)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(30)
    # the supervised factory child exits with the gateway; make sure none is left (it would hold the store lock)
    for pid in group_pids().get("factory", []):
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    t_end = time.time() + 60
    while group_pids().get("factory") and time.time() < t_end:
        time.sleep(0.5)
    new = start_logged([sys.executable, "-m", "pgw.cli", "gateway", "serve"], env, log_dir / "gateway.log",
                       redact=re.compile(rb"pgw-setup-[A-Za-z0-9_-]{8,}"), user_kw=kw, cwd=cwd)
    new.pgw_env, new.pgw_user_kw, new.pgw_cwd = env, kw, cwd  # type: ignore[attr-defined]
    wait_ready(f"http://127.0.0.1:{port}/healthz", new, timeout_s=timeout_s)
    return new


# ------------------------------------------------------------------------------------------------
# Qwen2.5-32B (co-located only): sizes, the provider's pack Volume, the mask-source hook
# ------------------------------------------------------------------------------------------------
#: Public sizes of the targets that need explicit sizing (bytes; from the checkpoints' configs and the pack format:
#: int8 linears + float32 embedding and LM head (``plain_dtype`` float32) + scales and norms). The hint is 24,576
#: bytes per output row of every linear (rank 16 x 256 columns x 6 byte planes), a mask slot 4 bytes per output row.
#: 7B: 1.39M output rows (hint 34.2 GB, slot 5.56 MB, pack 10.9 GB: the 45 GB cache restore of the 7B H100 run);
#: 32B: 64 layers x (7168 + 5120 + 55296 + 5120) = 4.65M output rows; 31.2 GB of int8 linears + 6.2 GB float32
#: embedding and head.
MODEL_SIZES = {
    "qwen2.5-7b-instruct": {"out_rows": 28 * (4608 + 3584 + 37888 + 3584), "pack_bytes": 10_900_000_000,
                            "checkpoint_bytes": 15_230_000_000},
    "qwen2.5-32b-instruct": {"out_rows": 64 * (7168 + 5120 + 55296 + 5120), "pack_bytes": 37_500_000_000,
                             "checkpoint_bytes": 65_530_000_000},
}
HINT_BYTES_PER_ROW = 16 * 256 * 6
SLOT_BYTES_PER_ROW = 4


def model_sizes(model: str) -> dict[str, int]:
    """Pack, hint, slot and checkpoint bytes of a target in :data:`MODEL_SIZES` (``{}`` otherwise)."""
    s = MODEL_SIZES.get(model.strip().lower())
    if s is None:
        return {}
    return {**s, "hint_bytes": s["out_rows"] * HINT_BYTES_PER_ROW, "slot_bytes": s["out_rows"] * SLOT_BYTES_PER_ROW}


#: Targets whose *provider* pack comes from a provider-side Modal Volume instead of the image (``colocated.py``):
#: baking a 37.5 GB pack (a 65.5 GB download) into the image would rebuild it on every source change. The Volume
#: (``PGW_MODAL_PROVIDER_PACK_VOLUME``, default ``pgw-provider-packs``) is filled once by ``prepare_provider_pack``
#: (a CPU container that downloads the pinned revision inside Modal and packs it; it mounts no gateway Volume) and
#: mounted root-only in the co-located container; the parent copies the pack into the provider's own directory
#: (owned by the provider's user) before the service starts. The gateway never reads it: its pack comes from the
#: gateway cache (packed by the customer side's ``prepare_cache``) and must verify in full to the enrolled
#: fingerprint, as before.
VOLUME_PACK_MODELS = ("qwen2.5-32b-instruct",)
#: Targets whose encryption hint must come from the gateway cache: the in-run CPU build (about 20 min on 16 vCPUs for
#: the 7B's 34 GB, so about 65-70 min for the 32B's 114 GB) does not fit the TTL. The run refuses to start its
#: parties without a restored pack and hint (:func:`require_cached_setup`) unless the masks come from elsewhere.
HINT_CACHE_REQUIRED_MODELS = ("qwen2.5-32b-instruct",)


def provider_pack_path(volume_root: Path, model: str) -> Path:
    """``<volume>/<model>/pack``: the provider's pack in its Volume (``prepare_provider_pack``)."""
    return Path(volume_root) / model / "pack"


#: Where the gateway's masks come from (``launch.py --mask-source``; ``PGW_MODAL_MASK_SOURCE`` in the gateway's run-time
#: environment). ``factory`` (default): today's encrypted (MLWE) mask factory in the gateway party, which needs the
#: hint. ``gpu-service``: the customer-side GPU mask service (``W8 r`` computed by the customer on a GPU of its own;
#: no hint, no provider work), built in a separate package. Hook only: :func:`mask_source_env` refuses it until that
#: package fills it in, and ``launch.py`` refuses it before anything is deployed (``GPU_MASK_SERVICE_READY``).
MASK_SOURCES = ("factory", "gpu-service")


def mask_source_env(source: str | None) -> dict[str, str]:
    """Gateway-party environment for a mask source (:data:`MASK_SOURCES`). ``factory``: nothing to add.
    ``gpu-service``: not wired yet (raises ``NotImplementedError``; the GPU mask service package replaces this
    branch with its settings and the container start-up that runs the service)."""
    src = (source or "factory").strip().lower()
    if src not in MASK_SOURCES:
        raise ValueError(f"mask source {source!r}: use one of {', '.join(MASK_SOURCES)}")
    if src == "gpu-service":
        raise NotImplementedError("PGW_MODAL_MASK_SOURCE=gpu-service: the GPU mask service is not wired into the "
                                  "Modal demo yet (hook: pgw_modal_boot.mask_source_env)")
    return {}


def require_cached_setup(model: str, restore: dict[str, Any], mask_source: str | None = "factory") -> None:
    """Refuse a run of a :data:`HINT_CACHE_REQUIRED_MODELS` target whose cache restore brought no pack, or no hint
    with the MLWE factory: the wizard would download the checkpoint and build the hint on the gateway's CPU for
    longer than the TTL, on the expensive container. Raises ``RuntimeError`` naming the fix (``--prepare-cache``)."""
    if model not in HINT_CACHE_REQUIRED_MODELS:
        return
    restore = restore or {}
    missing = []
    if "pack" not in restore:
        missing.append("pack")
    if (mask_source or "factory") == "factory" and not any(str(k).startswith("hint ") for k in restore):
        missing.append("encryption hint")
    if missing:
        raise RuntimeError(f"{model}: the gateway cache restored no {' and no '.join(missing)} ({restore}); fill it "
                           "first with launch.py ... --gateway-cache --prepare-cache")


# ------------------------------------------------------------------------------------------------
# Gateway-side cache of public set-up artefacts (``launch.py --gateway-cache``; a Modal Volume)
# ------------------------------------------------------------------------------------------------
#: Layout under the cache root: ``<model>/pack`` (the model pack: public weights, verified in full against the
#: enrolled fingerprint before use, ``PGW_SETUP_REUSE_PACK``) and ``<model>/hemask/<fp16>-r<rank>[-q<q>]`` (the
#: encrypted-mask hint ``H = W8 A``: the public int8 weights times the public matrix ``A``, whose seed the customer
#: side chose; no secret, see deploy/modal/README.md "Gateway cache").
def cache_paths(cache_root: Path, model: str) -> dict[str, Path]:
    base = Path(cache_root) / model
    return {"pack": base / "pack", "hemask": base / "hemask"}


def copy_tree(src: Path, dst: Path, threads: int = 8) -> dict[str, Any]:
    """Copy directory ``src`` to ``dst`` (files in parallel; written under ``dst.partial`` and renamed, so ``dst``
    is complete or absent). Returns bytes, files and seconds."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    files = [p for p in src.rglob("*") if p.is_file()]
    for d in {p.parent for p in files}:
        (tmp / d.relative_to(src)).mkdir(parents=True, exist_ok=True)

    def one(p: Path) -> int:
        shutil.copyfile(p, tmp / p.relative_to(src))
        return p.stat().st_size

    with ThreadPoolExecutor(max(1, threads)) as ex:
        nbytes = sum(ex.map(one, files))
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return {"bytes": nbytes, "files": len(files), "seconds": round(time.time() - t0, 1),
            "gb_per_s": round(nbytes / 1e9 / max(1e-9, time.time() - t0), 2)}


def restore_cache(cache_root: Path, model: str, data_dir: Path) -> dict[str, Any]:
    """Before the gateway starts: copy the cached pack to ``<data_dir>/packs/<model>`` (the wizard's pack
    directory; used only if it verifies to the enrolled fingerprint) and every cached hint to
    ``<data_dir>/hemask`` (the wizard's hint step finds it by fingerprint, rank and q). Missing entries: nothing."""
    from pgw.enroll.client import _safe_dirname

    paths = cache_paths(cache_root, model)
    rep: dict[str, Any] = {}
    if (paths["pack"] / "manifest.json").is_file():
        rep["pack"] = copy_tree(paths["pack"], data_dir / "packs" / _safe_dirname(model))
    hints = [d for d in sorted(paths["hemask"].glob("*")) if d.is_dir() and (d / "meta.json").is_file()]
    for d in hints:
        (data_dir / "hemask").mkdir(parents=True, exist_ok=True)
        rep[f"hint {d.name}"] = copy_tree(d, data_dir / "hemask" / d.name)
    return rep


def save_cache(cache_root: Path, model: str, data_dir: Path) -> dict[str, Any]:
    """After a run: copy the gateway's pack and hints into the cache where it has none yet (public artefacts
    only: the pack and ``H = W8 A``; never the mask store, keys, tokens or the factory's state)."""
    from pgw.enroll.client import _safe_dirname

    paths = cache_paths(cache_root, model)
    rep: dict[str, Any] = {}
    pack = data_dir / "packs" / _safe_dirname(model)
    if (pack / "manifest.json").is_file() and not (paths["pack"] / "manifest.json").is_file():
        paths["pack"].parent.mkdir(parents=True, exist_ok=True)
        rep["pack"] = copy_tree(pack, paths["pack"])
    for d in sorted((data_dir / "hemask").glob("*")):
        if (d.is_dir() and ".building" not in d.name and (d / "meta.json").is_file()
                and not (paths["hemask"] / d.name / "meta.json").is_file()):
            paths["hemask"].mkdir(parents=True, exist_ok=True)
            rep[f"hint {d.name}"] = copy_tree(d, paths["hemask"] / d.name)
    return rep


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


#: Linear kinds whose masks the factory need not make, per fixed ``local_linears`` mode (the gateway computes
#: them itself on every step: the demo sets no ``local_max_rows``). ``auto`` modes decide at start-up: none.
SKIP_KINDS_BY_LOCAL = {"attention": "qkv,o", "attention_down": "qkv,o,down"}


#: Mask factory keep-up in the co-located demo (S5, measured on Modal; see deploy/modal/README.md, "Factory
#: keep-up under back-to-back chats"). The all-or-nothing yield at 1000 slots starved the store after 130 s of
#: back-to-back chats: the factory made nothing above 1000 slots and a 4096-slot segment could not land below it.
#: Now: partial yield between 2500 and 6000 slots (the factory's duty cycle grows as the stock falls), no yield
#: below 45 s of observed consumption, full yield above 120 s of it (an occasional chat: full yield as before),
#: 2048-slot segments (about half the time to land), refill whenever the stock
#: is below 6144 (to 8192) and a 2 s scheduler poll (was 30 s: up to 30 s of no production after a drain started).
COLOCATED_FACTORY_KEEP_UP = {
    "PGW_FACTORY_YIELD_MIN_SLOTS": "2500", "PGW_FACTORY_YIELD_FULL_SLOTS": "6000",
    "PGW_FACTORY_YIELD_HORIZON_S": "45", "PGW_FACTORY_YIELD_PERIOD_S": "3", "PGW_FACTORY_YIELD_FULL_HORIZON_S": "120",
    "PGW_FACTORY_SEGMENT_SLOTS": "2048", "PGW_FACTORY_LOW_WATER_SLOTS": "6144", "PGW_FACTORY_TARGET_SLOTS": "8192",
    "PGW_FACTORY_POLL_INTERVAL_S": "2",
}


#: Mask bytes per slot grow with the model (gate_up masks of a 0.5B slot: 0.93 MB; 1.5B 2.0 MB; 3B 3.2 MB; 7B
#: 4.2 MB) and the factory's slots per second shrink about as much, so the keep-up's slot counts are divided by
#: this per target (segments land in about the same time; the stock holds about the same seconds of decode).
#: Qwen2.5-32B with every linear remote: 18.6 MB per slot (4.65M output rows x 4 bytes), 3.35x the 7B's 5.56 MB:
#: 16 (segments of 128 slots, 2.4 GB; refill below 384 to 512).
FACTORY_SLOT_DIVISOR = {"qwen2.5-0.5b-instruct": 1, "qwen2.5-1.5b-instruct": 2, "qwen2.5-3b-instruct": 4,
                        "qwen2.5-7b-instruct": 4, "qwen2.5-32b-instruct": 16}
_KEEP_UP_SLOT_KEYS = ("PGW_FACTORY_YIELD_MIN_SLOTS", "PGW_FACTORY_YIELD_FULL_SLOTS", "PGW_FACTORY_SEGMENT_SLOTS",
                      "PGW_FACTORY_LOW_WATER_SLOTS", "PGW_FACTORY_TARGET_SLOTS")


def factory_keep_up(model: str | None = None) -> dict[str, str]:
    """:data:`COLOCATED_FACTORY_KEEP_UP` with its slot counts scaled for ``model`` (:data:`FACTORY_SLOT_DIVISOR`)."""
    div = FACTORY_SLOT_DIVISOR.get((model or MODEL_NAME).strip().lower(), 1)
    return {k: (str(max(64, int(v) // div)) if k in _KEEP_UP_SLOT_KEYS else v)
            for k, v in COLOCATED_FACTORY_KEEP_UP.items()}


def colocated_gateway_env(local_linears: str, model: str | None = None) -> dict[str, str]:
    """Gateway-party environment of the co-located demo (measured on Modal, see deploy/modal/README.md):

    * ``PGW_FACTORY_SKIP_LINEAR_KINDS``: with a fixed ``local_linears`` the encrypted mask factory makes masks
      only for the linears the gateway sends (attention_down: gate_up, 52% of the mask bytes); the gateway
      retires any segment lacking a linear it sends (``MaskStore.require_linears``), so this is exact and
      changes nothing the provider sees in decode. 138 slots/s on 2.5 cores while a chat decodes.
    * ``PGW_POOL_SPIN_US`` / ``PGW_POOL_WORKERS``: the native pool's workers stay awake 2 ms between jobs
      (a futex wake-up under Modal's gVisor costs more than a decode step's gap between local linears) and
      are capped at 4, leaving cores to the factory and the service.
    * ``PGW_PROMPT_PREFETCH``: a prompt pass stages the next remote linear's mask rows on the prefetch thread
      (the same rows in the same order, single-use; the row count is the prompt's, which the provider sees
      anyway): mask reads leave the first token's critical path (about 100 ms to 20-35 ms for 41 rows).
    * ``PGW_PIPELINE_BLOCKS=1``: a prompt round trip goes out as one item of all its rows instead of four
      overlapping row blocks (a fixed, public split of the public row count; the provider sees one larger
      item per prompt trip): half the gateway hops of a prompt pass (prompt pass 550 ms median vs 700-800).
    * ``PGW_FACTORY_YIELD_FILE`` / ``PGW_FACTORY_YIELD_MIN_SLOTS`` (S4, ``pgw.core.factory_yield``): the encrypted
      factory steps aside while masked passes run and at least 1000 mask slots are in stock (its encrypted maskgen
      traffic and the provider's maskgen GEMMs slowed each decoded token by about 4 ms); below that it runs at full
      speed. Same masks, requests and bytes. The file is in the gateway party's home (``/srv/pgw/home``).
    """
    env = {"PGW_POOL_SPIN_US": "2000", "PGW_POOL_WORKERS": "4",
           "PGW_FACTORY_YIELD_FILE": "/srv/pgw/home/factory-yield", **factory_keep_up(model),
           # T3: the one-call trip path and the sampled noise statistics off the critical path (same bytes,
           # checks and statistics; library defaults off, see MaskedBackend.trip_fast / noise_async_sampled)
           "PGW_TRIP_FAST": "1",
           # prompt pass: mask prefetch and one item per prompt trip (see below)
           "PGW_PROMPT_PREFETCH": "1", "PGW_PIPELINE_BLOCKS": "1"}
    if local_linears in SKIP_KINDS_BY_LOCAL:
        env["PGW_FACTORY_SKIP_LINEAR_KINDS"] = SKIP_KINDS_BY_LOCAL[local_linears]
    return env


#: The provider party's environment in the co-located demo (S4): decode graphs on the GPU's highest-priority
#: stream, ahead of the encrypted factory's maskgen GEMMs (``pgw.service.gpu_exec.trip_stream_priority``).
COLOCATED_PROVIDER_ENV = {"PGW_SERVICE_TRIP_STREAM_PRIORITY": "high"}


#: The gateway party's environment when it has its own GPU (``split_gpus``): its encrypted mask factory stays on the
#: CPU by default. Opt-in per run (measured exact, deploy/modal/README.md, "Factory HE on the gateway GPU"):
#: PGW_FACTORY_HE_ENCRYPT=gpu_ntt and PGW_FACTORY_HE_DECRYPT=gpu_ntt (the factory's HE on the gateway's GPU, about 2
#: -> 0.9 gateway cores; the GPU time-slices it with the decode graphs while the factory runs unyielded),
#: PGW_LOCAL_GPU_STREAM_PRIORITY=high, PGW_THREAD_NAMES=1 (per-thread factory cores in the sustain report).
#: ``PGW_LOCAL_GPU_PREFILL_ROWS``: a prompt pass's local linears through one captured digit-GEMM graph per linear on
#: 64-row blocks (exact, self-checked per linear; about 145 ms to 85 ms for 41 rows on a T4).
#: ``PGW_HINT_DEVICE``: the one-time encryption hint's GEMMs on that GPU (``pgw.hemask.hint_gpu``, the CPU path's
#: bits, self-checked first); minutes of gateway CPU for the 1.5B and tens of minutes for the 7B otherwise.
GATEWAY_GPU_FACTORY_ENV = {"PGW_HEMASK_ENGINE": "cpu_int8", "PGW_FACTORY_DEVICE": "cpu",
                           "PGW_FACTORY_GPU_PRNG": "false", "PGW_LOCAL_GPU_PREFILL_ROWS": "64",
                           "PGW_HINT_DEVICE": "cuda"}


def _nvidia_devices() -> list[dict[str, str]]:
    """``[{"uuid", "minor"}]`` per GPU in CUDA order, from ``nvidia-smi -q`` (empty if it fails)."""
    try:
        text = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True,  # noqa: S607
                              timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out: list[dict[str, str]] = []
    for line in text.splitlines():
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "GPU UUID":
            out.append({"uuid": val, "minor": ""})
        elif key == "Minor Number" and out:
            out[-1]["minor"] = val
    return out


def split_gpus(provider_user: str | None, gateway_user: str | None) -> dict[str, Any]:
    """One GPU per co-located party (``launch.py --gateway-gpu``): the provider's process sees only the first
    GPU and the gateway's only the second (``CUDA_VISIBLE_DEVICES`` by UUID); best effort, each device node
    ``/dev/nvidia<minor>`` is also given to its party's user with mode 0600 (shared control nodes stay
    shared). Returns ``{"provider": env, "gateway": env, "report": ...}``; ``gateway`` is empty (the gateway
    keeps no GPU) when fewer than two GPUs are visible."""
    devs = _nvidia_devices()
    if len(devs) < 2:
        return {"provider": {}, "gateway": {}, "report": {"ok": False, "reason": f"{len(devs)} GPU(s) visible"}}
    prov, gate = devs[0], devs[1]
    report: dict[str, Any] = {"ok": True, "provider_gpu": prov["uuid"], "gateway_gpu": gate["uuid"], "nodes": {}}
    owners = os.environ.get("PGW_MODAL_GPU_NODE_OWNERS", "1") != "0"
    for dev, user in ((prov, provider_user), (gate, gateway_user)):
        node = f"/dev/nvidia{dev['minor']}"
        if not owners:
            report["nodes"][node] = "unchanged (PGW_MODAL_GPU_NODE_OWNERS=0)"
            continue
        if not dev["minor"] or user is None:
            report["nodes"][node] = "unchanged (no minor number or no party user)"
            continue
        try:
            import pwd

            pw = pwd.getpwnam(user)
            os.chown(node, pw.pw_uid, pw.pw_gid)
            os.chmod(node, 0o600)
            report["nodes"][node] = f"{user} 0600"
        except OSError as exc:
            report["nodes"][node] = f"unchanged ({type(exc).__name__}: {exc.strerror or exc})"
    return {"provider": {"CUDA_VISIBLE_DEVICES": prov["uuid"]},
            "gateway": {"CUDA_VISIBLE_DEVICES": gate["uuid"], **GATEWAY_GPU_FACTORY_ENV}, "report": report}


#: Variable prefixes a run may add to a party's environment (``launch.py --gateway-env`` / ``--provider-env``).
EXTRA_ENV_PREFIXES = ("PGW_", "OMP_", "MKL_", "GOMP_")


def checked_env(extra: dict[str, Any] | None) -> dict[str, str]:
    """``extra`` as strings, refusing keys outside :data:`EXTRA_ENV_PREFIXES` (no PATH, LD_PRELOAD, ...)."""
    out: dict[str, str] = {}
    for k, v in (extra or {}).items():
        if not isinstance(k, str) or not k.startswith(EXTRA_ENV_PREFIXES):
            raise ValueError(f"extra environment key {k!r} not allowed (prefixes {EXTRA_ENV_PREFIXES})")
        out[k] = str(v)
    return out


_SEGMENT_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) .*segment \S+: .*?(\d+) slots in ([\d.]+)s "
                         r"\(([\d.]+) slots/s")
_YIELDED_RE = re.compile(r"yielded to decode ([\d.]+)s")
_HE_STAGES_RE = re.compile(r"encrypt ([\d.]+)s, provider ([\d.]+)s, decrypt ([\d.]+)s")
_HE_ENGINE_RE = re.compile(r", on ([^;\s]+)")
_CHECK_FIELD_RE = re.compile(r"; check field (\S+)")


def factory_segments(logfile: Path, t0: float, t1: float) -> dict[str, Any]:
    """Mask factory segments (``segment ...: N slots in Ss (R slots/s ...``) in ``logfile`` (epoch seconds; log
    times are local time): ``segments`` ran entirely inside ``[t0, t1]`` (with their summed rate), ``committed``
    landed in the store inside the window (whenever they started; ``committed_slots`` is what the factory added
    to the stock meanwhile, ``production_slots_per_s`` that over the window length)."""
    inside: list[dict[str, float]] = []
    committed: list[dict[str, float]] = []
    other = 0
    try:
        lines = logfile.read_text(errors="replace").splitlines()
    except OSError:
        return {"segments": [], "error": "no log"}
    for line in lines:
        m = _SEGMENT_RE.search(line)
        if not m:
            continue
        end = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + int(m.group(2)) / 1e3
        slots, secs = int(m.group(3)), float(m.group(4))
        y = _YIELDED_RE.search(line)
        row = {"slots": slots, "seconds": secs, "slots_per_s": float(m.group(5)),
               "start_offset_s": round(end - secs - t0, 1), "end_offset_s": round(end - t0, 1),
               "yielded_s": float(y.group(1)) if y else 0.0}
        st = _HE_STAGES_RE.search(line)
        if st:
            row.update(encrypt_s=float(st.group(1)), provider_s=float(st.group(2)), decrypt_s=float(st.group(3)))
        cf = _CHECK_FIELD_RE.search(line)
        row["check_field"] = cf.group(1) if cf else "ring"  # type: ignore[assignment]
        eng = _HE_ENGINE_RE.search(line)
        if eng:
            row["engine"] = eng.group(1)  # type: ignore[assignment]
        if t0 <= end <= t1 + 1.0:
            committed.append(row)
        if end - secs >= t0 - 1.0 and end <= t1 + 1.0:
            inside.append(row)
        else:
            other += 1
    tot_s = sum(x["seconds"] for x in inside)
    cslots = sum(int(x["slots"]) for x in committed)
    return {"segments": inside, "segments_outside_window": other,
            "slots_per_s": round(sum(x["slots"] for x in inside) / tot_s, 2) if tot_s else None,
            "committed": committed, "committed_slots": cslots,
            "production_slots_per_s": round(cslots / max(1e-9, t1 - t0), 2)}


def keep_up(sustain: dict[str, Any], factory: dict[str, Any]) -> dict[str, Any]:
    """Factory keep-up over a sustain window: consumption = (stock at start + slots the factory committed - stock
    at the end) / seconds (slots/s), the factory's production rate over the same window, and the stock trend."""
    s0, s1 = sustain.get("stock_start"), sustain.get("stock_end")
    secs = float(sustain.get("seconds") or 0.0)
    series = sustain.get("stock") or []
    if s0 is None or s1 is None or secs <= 0 or len(series) < 2:
        return {"error": "no stock samples"}
    span = float(series[-1][0] - series[0][0]) or secs
    prod = sum(int(x["slots"]) for x in factory.get("committed") or []
               if series[0][0] <= x["end_offset_s"] <= series[-1][0] + 0.5)
    cons = (s0 + prod - s1) / span
    half = [x for x in series if x[0] >= series[-1][0] / 2]
    # Production lands in whole segments, so over a window it is quantized (one 2048-slot segment = 13.6 slots/s
    # over 150 s). Also reported: the rate between the window's first and last commit (whole segment periods)
    # and the stock's least-squares slope (slots/s; about 0 when the factory keeps up in steady state).
    ends = sorted(float(x["end_offset_s"]) for x in factory.get("committed") or []
                  if series[0][0] <= x["end_offset_s"] <= series[-1][0] + 0.5)
    c2c = None
    if len(ends) >= 2:
        slots_after = sum(int(x["slots"]) for x in factory.get("committed") or []
                          if ends[0] < x["end_offset_s"] <= ends[-1])
        c2c = round(slots_after / max(1e-9, ends[-1] - ends[0]), 2)
    n = len(series)
    mx = sum(x[0] for x in series) / n
    my = sum(x[1] for x in series) / n
    sxx = sum((x[0] - mx) ** 2 for x in series)
    slope = sum((x[0] - mx) * (x[1] - my) for x in series) / sxx if sxx else None
    return {"consumption_slots_per_s": round(cons, 2), "production_slots_per_s": round(prod / span, 2),
            "produced_slots": prod, "stock_start": s0, "stock_end": s1, "stock_min": sustain.get("stock_min"),
            "stock_second_half_change": (half[-1][1] - half[0][1]) if len(half) >= 2 else None,
            "production_commit_to_commit_slots_per_s": c2c,
            "stock_slope_slots_per_s": None if slope is None else round(slope, 2),
            "keeps_up": prod / span >= cons or s1 >= s0}


def per_trip_cpu(sustain: dict[str, Any]) -> dict[str, Any]:
    """Derived figures of a sustain window (in place, ``sustain["derived"]``): gateway process CPU per round trip
    (its cores x seconds / trips: decode, float ops, LM head, draft and Python, the factory excluded: a separate
    process), mask slots consumed per generated token (``keep_up`` consumption x span / tokens: prompts, decode
    charges and burned reservations) and the factory's production against that consumption."""
    secs = float(sustain.get("seconds") or 0.0)
    trips = int(((sustain.get("trips") or {}).get("trips")) or 0)
    cores = sustain.get("cores") or {}
    ku = sustain.get("keep_up") or {}
    toks = int(sustain.get("tokens") or 0)
    d: dict[str, Any] = {}
    if trips and secs and cores.get("gateway") is not None:
        d["gateway_cpu_ms_per_trip"] = round(1e3 * float(cores["gateway"]) * secs / trips, 3)
        d["trips_per_s"] = round(trips / secs, 1)
    if toks and ku.get("consumption_slots_per_s") is not None and secs:
        d["slots_per_token"] = round(float(ku["consumption_slots_per_s"]) * secs / toks, 3)
        d["tokens_per_s_window"] = round(toks / secs, 2)
    if trips and toks:
        d["trips_per_token"] = round(trips / toks, 2)
    if ku.get("consumption_slots_per_s") is not None:
        d["factory_vs_consumption"] = {"consumption_slots_per_s": ku.get("consumption_slots_per_s"),
                                       "factory_window_slots_per_s": ku.get("production_slots_per_s"),
                                       "factory_commit_to_commit_slots_per_s":
                                           ku.get("production_commit_to_commit_slots_per_s"),
                                       "stock_slope_slots_per_s": ku.get("stock_slope_slots_per_s")}
    sustain["derived"] = d
    return d


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


def lane_peer_uid(client_user: str | None) -> int:
    """The uid the provider's UDS lane accepts: the gateway party's user (this process's own without users)."""
    if client_user is None:
        return os.getuid()
    import pwd

    return pwd.getpwnam(client_user).pw_uid


def uds_dir(path: Path, owner: str | None, client_user: str | None) -> Path:
    """The lane socket's directory: the provider's (``owner``), mode 0710 with the gateway user's group (search
    only: it can reach the socket by name, not list the directory); 0700 and root's without users."""
    path.mkdir(parents=True, exist_ok=True)
    if owner is None or client_user is None:
        os.chmod(path, 0o700)
        return path
    import pwd

    os.chown(path, pwd.getpwnam(owner).pw_uid, pwd.getpwnam(client_user).pw_gid)
    os.chmod(path, 0o710)
    return path


def share_socket(sock: Path, owner: str | None, client_user: str | None, timeout_s: float = 120.0) -> bool:
    """Give the gateway user's group connect permission on the lane socket (0660, still owned by the provider),
    once the service has bound it. Without users both parties are root: nothing to do. False if it never appeared."""
    if owner is None or client_user is None:
        return True
    import pwd

    t_end = time.time() + timeout_s
    while not sock.exists():
        if time.time() > t_end:
            log(f"lane socket {sock} did not appear")
            return False
        time.sleep(0.2)
    os.chown(sock, pwd.getpwnam(owner).pw_uid, pwd.getpwnam(client_user).pw_gid)
    os.chmod(sock, 0o660)
    return True


def start_provider_party(work: Path, pack: Path, port: int, *, device: str, user: str | None, log_dir: Path,
                         tenant: str = "customer-demo", code_ttl: str = "90m", note: str = "co-located demo",
                         public_url: str | None = None, lane_port: int | None = None,
                         lane_socket: Path | None = None, lane_client_user: str | None = None,
                         extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """The provider's compute service on ``127.0.0.1:port`` only (TLS, a fresh PKI for 127.0.0.1), run as
    ``user`` in ``work`` (0700, owned by that user once its PKI and state files exist), plus a fresh
    single-use enrollment code. ``public_url`` (default ``https://127.0.0.1:port``) is the URL the code
    points the gateway at (e.g. a delay proxy on loopback). ``lane_port``: also serve the native trip lane
    over TLS 1.3 on ``127.0.0.1:lane_port`` (``lane = "tcp"``: the service's certificate, client certificates
    verified against the client-issuing CA and bound to the lane ticket). ``lane_socket``: instead serve the
    same-host UDS lane there (``lane = "uds"``; ``launch.py --lane uds``): only ``lane_client_user``'s uid passes the
    service's peer-credential check (``PGW_SERVICE_LANE_PEER_UIDS``), the socket's directory is the provider's
    with group search for that user's group only (0710) and the socket itself 0660 to the same group (the service
    binds it 0600; root widens it to that one group once it exists). Returns ``proc``, ``url``, ``code``,
    ``pki``, ``env`` and ``ca_sha256``."""
    party_dir(work / "home", None)
    pki = make_pki(work / "pki", ["127.0.0.1", "localhost"])
    url = public_url or f"https://127.0.0.1:{port}"
    env = service_env(work, pack, pki, url, port, device=device)
    env["PGW_SERVICE_HOST"] = "127.0.0.1"  # nothing outside this host can reach it (the lane port too)
    if lane_port is not None:
        env["PGW_SERVICE_LANE"] = "tcp"
        env["PGW_SERVICE_LANE_PORT"] = str(int(lane_port))
    if lane_socket is not None:
        env["PGW_SERVICE_LANE"] = "uds"
        env["PGW_SERVICE_LANE_SOCKET"] = str(lane_socket)
        env["PGW_SERVICE_LANE_PEER_UIDS"] = str(lane_peer_uid(lane_client_user))
        uds_dir(lane_socket.parent, user, lane_client_user)
    env.update(extra_env or {})
    party_dir(work, user)  # everything below is the provider's from here on
    kw = as_user(env, user, work / "home")
    party_dir(log_dir, None)
    proc = start_service(env, log_dir / "service.log", user_kw=kw, cwd=work)
    t0 = time.time()
    wait_ready(f"https://127.0.0.1:{port}/readyz", proc, verify=str(pki["service_ca"]), timeout_s=900)
    if lane_socket is not None:
        share_socket(lane_socket, user, lane_client_user)
    log(f"service ready in {time.time() - t0:.0f}s (127.0.0.1:{port}, user {user or 'same'})")
    code = create_enrollment_code(env, tenant, code_ttl, note, user_kw=kw, cwd=work)
    from pgw.enroll.service import ca_fingerprint, load_ca_pem

    return {"proc": proc, "url": url, "code": code, "pki": pki, "env": env,
            "ca_sha256": ca_fingerprint(load_ca_pem(pki["service_ca"])).hex()}


def start_gateway_party(root: Path, port: int, *, user: str | None, log_dir: Path,
                        model_options: dict[str, Any] | None = None,
                        hide_gpu: bool = True, egress: list[str] | None = None,
                        extra_env: dict[str, str] | None = None) -> tuple[subprocess.Popen[bytes], str]:
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
    env.update(extra_env or {})
    if env.get("OMP_WAIT_POLICY") == "unset":  # --gateway-env OMP_WAIT_POLICY=unset: the OpenMP default policy
        del env["OMP_WAIT_POLICY"]
    kw = as_user(env, user, root / "home")
    party_dir(log_dir, None)
    gw, token = start_gateway(env, log_dir / "gateway.log", user_kw=kw, cwd=root)
    gw.pgw_env, gw.pgw_user_kw, gw.pgw_cwd = env, kw, root  # type: ignore[attr-defined]  # restart_gateway
    wait_ready(f"http://127.0.0.1:{port}/healthz", gw)
    log(f"gateway in setup mode on :{port} (user {user or 'same'})")
    return gw, token



# ------------------------------------------------------------------------------------------------
# CPU use per party process (co-located runs: who takes the container's cores while a chat decodes)
# ------------------------------------------------------------------------------------------------
_PROC_GROUPS = (("factory", "factory run"), ("service", "service serve"), ("gateway", "gateway serve"))


def _proc_cpu() -> dict[str, float]:
    """CPU seconds (user + system) per process group (:data:`_PROC_GROUPS`) from ``/proc``."""
    tick = os.sysconf("SC_CLK_TCK")
    out: dict[str, float] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{d}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            stat = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        name = next((g for g, pat in _PROC_GROUPS if pat in cmd), None)
        if name is None:
            continue
        f = stat.rsplit(")", 1)[1].split()
        out[name] = out.get(name, 0.0) + (int(f[11]) + int(f[12])) / tick
    return out


class CpuSampler:
    """Samples :func:`_proc_cpu` every ``period_s`` in a daemon thread; :meth:`cores` gives the mean cores
    each group used between two times."""

    def __init__(self, period_s: float = 0.5):
        self.period_s = period_s
        self.samples: list[tuple[float, dict[str, float]]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="cpu-sampler", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), _proc_cpu()))
            except Exception:  # noqa: S110 - best effort (a process gone between listdir and read)
                pass
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()

    def cores(self, t0: float, t1: float) -> dict[str, float]:
        inside = [s for s in self.samples if t0 <= s[0] <= t1]
        if len(inside) < 2:
            return {}
        (ta, a), (tb, b) = inside[0], inside[-1]
        return {k: round((b.get(k, 0.0) - a.get(k, 0.0)) / max(1e-9, tb - ta), 2) for k in b}


# ------------------------------------------------------------------------------------------------
# Co-located A/B diagnostics (``--gateway-env PGW_MODAL_AB=...``): re-pin or pause the party processes between
# streamed chats of one run, so host-to-host variance does not hide a difference. Root only; diagnostics, never
# part of the product path (nothing the parties compute or send changes; a paused factory only stops refilling).
# ------------------------------------------------------------------------------------------------
def group_pids() -> dict[str, list[int]]:
    """``{group: [pid, ...]}`` for :data:`_PROC_GROUPS` (the factory may have several processes)."""
    out: dict[str, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{d}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        name = next((g for g, pat in _PROC_GROUPS if pat in cmd), None)
        if name is not None:
            out.setdefault(name, []).append(int(d))
    return out


def _tids(pid: int) -> list[int]:
    try:
        return [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    except OSError:
        return []


def pin_groups(layout: dict[str, set[int] | None]) -> dict[str, Any]:
    """Set the CPU affinity of every thread of each group's processes (``None``: all CPUs again); returns
    what the kernel reports back for the group's first process."""
    rep: dict[str, Any] = {}
    pids = group_pids()
    every = set(range(os.cpu_count() or 1))
    for g, cpus in layout.items():
        n, err = 0, None
        for pid in pids.get(g, []):
            for tid in _tids(pid):
                try:
                    os.sched_setaffinity(tid, cpus if cpus is not None else every)
                    n += 1
                except OSError as exc:
                    err = f"{type(exc).__name__}: {exc}"
        got = None
        if pids.get(g):
            try:
                got = sorted(os.sched_getaffinity(pids[g][0]))
            except OSError:
                pass
        rep[g] = {"threads": n, "cpus": got, "error": err}
    return rep


def signal_group(group: str, sig: int) -> int:
    """Send ``sig`` (e.g. ``signal.SIGSTOP`` / ``SIGCONT``) to every process of ``group``; returns how many got it."""
    n = 0
    for pid in group_pids().get(group, []):
        try:
            os.kill(pid, sig)
            n += 1
        except OSError:
            pass
    return n


def thread_cpu(group: str) -> dict[str, tuple[str, float]]:
    """``{"pid/tid": (comm, cpu_s)}`` of every thread of ``group``'s processes (for per-thread deltas)."""
    tick = os.sysconf("SC_CLK_TCK")
    out: dict[str, tuple[str, float]] = {}
    for pid in group_pids().get(group, []):
        for tid in _tids(pid):
            try:
                stat = Path(f"/proc/{pid}/task/{tid}/stat").read_text()
            except OSError:
                continue
            comm = stat[stat.find("(") + 1:stat.rfind(")")]
            f = stat.rsplit(")", 1)[1].split()
            out[f"{pid}/{tid}"] = (comm, (int(f[11]) + int(f[12])) / tick)
    return out


def comm_deltas(a: dict[str, tuple[str, float]], b: dict[str, tuple[str, float]], seconds: float) -> dict[str, float]:
    """Cores per thread name (``comm``; trailing worker digits dropped) between two :func:`thread_cpu` snapshots,
    busiest first (``PGW_THREAD_NAMES=1`` names the encrypted factory's stage threads, see
    ``pgw.core.proc_tuning.os_thread_name``; native pool threads inherit their creator's name)."""
    out: dict[str, float] = {}
    for k, v in b.items():
        name = v[0].rstrip("0123456789").rstrip("_-") or v[0]
        out[name] = out.get(name, 0.0) + (v[1] - a.get(k, (v[0], 0.0))[1]) / max(1e-9, seconds)
    return {k: round(x, 3) for k, x in sorted(out.items(), key=lambda kv: -kv[1]) if x >= 0.005}


def thread_deltas(a: dict[str, tuple[str, float]], b: dict[str, tuple[str, float]], seconds: float,
                  top: int = 8) -> list[tuple[str, str, float]]:
    """The ``top`` busiest threads between two :func:`thread_cpu` snapshots, in cores."""
    rows = [(k, v[0], round((v[1] - a.get(k, (v[0], 0.0))[1]) / max(1e-9, seconds), 3)) for k, v in b.items()]
    return sorted(rows, key=lambda r: -r[2])[:top]


#: ``PGW_MODAL_AB`` layouts (30 visible CPUs on Modal's T4:2 hosts, 14 of them by quota): the gateway's decode
#: process, the provider's service and the factory on disjoint cores.
PIN_LAYOUT: dict[str, set[int] | None] = {"gateway": set(range(0, 6)), "service": set(range(6, 9)),
                                          "factory": set(range(9, 14))}
UNPIN_LAYOUT: dict[str, set[int] | None] = {"gateway": None, "service": None, "factory": None}


def ab_phases(spec: str) -> list[tuple[str, Any]]:
    """``PGW_MODAL_AB`` (comma list of ``base``, ``pin``, ``unpin``, ``nofactory``, ``factory``): the phases of
    :func:`e2e_driver.run`'s ``ab`` streams, each ``(label, action)``; actions are cumulative, the driver
    resumes the factory and unpins at the end."""
    import signal

    acts = {"base": lambda: None, "pin": lambda: pin_groups(PIN_LAYOUT), "unpin": lambda: pin_groups(UNPIN_LAYOUT),
            "nofactory": lambda: signal_group("factory", signal.SIGSTOP),
            "factory": lambda: signal_group("factory", signal.SIGCONT)}
    return [(p.strip(), acts[p.strip()]) for p in spec.split(",") if p.strip() in acts]


def ab_cleanup() -> None:
    import signal

    signal_group("factory", signal.SIGCONT)
    pin_groups(UNPIN_LAYOUT)
