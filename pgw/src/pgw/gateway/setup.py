"""First-run setup mode of the gateway: the guided console wizard and its backend.

When the gateway starts with no model configured, with an unfinished wizard run under
``config_dir``, or with ``PGW_GATEWAY_SETUP_MODE=1``, ``pgw gateway serve`` serves this app instead
of the inference API. It exposes only ``/healthz``, ``/readyz`` (always 503), the console
(``/console/``) and ``/console/api/setup/*``. Everything else answers 503 ``setup mode``.

Access: a random one-time setup token is printed to stdout and the log at start-up. The wizard's
sign-in exchanges it for an HttpOnly ``SameSite=Strict`` session cookie; every state-changing call
needs the session's CSRF token (``X-PGW-CSRF``), exactly like the normal console. An exposed port
therefore does not let strangers configure the gateway.

Steps (each persisted to ``<config_dir>/setup/state.json``, so a restart resumes):

1. ``POST /console/api/setup/enroll``     enrollment code -> credentials (:mod:`pgw.enroll.client`)
2. ``POST /console/api/setup/model``      choose a model; download + pack in the background; the
                                          pack fingerprint must equal the service's
3. ``POST /console/api/setup/maskstore``  initialize the mask store, a local key file (0600) and a
                                          file monotonic counter off the mask volume
4. ``POST /console/api/setup/admin``      generate the admin token (shown once) and optional SSO
5. ``POST /console/api/setup/verify``     doctor checks, start hint build + supervised mask factory
6. ``POST /console/api/setup/finish``     write ``gateway.env`` / ``factory.env`` / ``models.json`` and
                                          switch to normal mode in-process (or ask for a restart)

Production guard: request bodies accept only the documented fields (``extra="forbid"``); there is
no way to choose a development mode, plaintext HTTP to a non-loopback service, disabled Freivalds
checks or noise monitor, or ``allow_dev_modes``. The final configuration is validated with the
gateway's production policy before it is written.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pgw.console.sessions import COOKIE_NAME, CSRF_HEADER, ConsoleSessions, LoginRateLimiter, cookie_secure
from pgw.core.errors import AuthError, ConfigError, ErrorCode, PGWError
from pgw.core.fileio import atomic_write_text, write_private
from pgw.gateway.setup_jobs import (
    CheckpointResolver,
    FactorySupervisor,
    HintBuildJob,
    PackBuildJob,
    disk_free,
    mask_estimate,
)

__all__ = ["SETUP_STEPS", "SetupController", "SetupOptions", "SwitchableApp", "create_setup_app", "load_setup_state",
           "should_run_setup"]

log = logging.getLogger(__name__)

#: Error codes caused by the administrator's input or environment (shown as 400 with their message).
_INPUT_ERRORS = frozenset({ErrorCode.CONFIG, ErrorCode.UNAUTHENTICATED, ErrorCode.PERMISSION_DENIED,
                           ErrorCode.MASK_STORE, ErrorCode.INTEGRITY, ErrorCode.FINGERPRINT_MISMATCH})
SETUP_STEPS = ("enroll", "model", "maskstore", "admin", "verify", "finish")
#: Mask buffer the wizard configures for the factory (slots = tokens of masks kept ready).
FACTORY_LOW_WATER_SLOTS = 1000
FACTORY_TARGET_SLOTS = 4000
#: Size of the factory's first segment for a new store: the gateway can serve as soon as it is stored
#: (a full 4096-slot segment takes about 2 min next to a T4 provider, but much longer over a slow link).
FACTORY_FIRST_SEGMENT_SLOTS = 256


# =================================================================================================
# persisted state
# =================================================================================================
def state_path(config_dir: Path) -> Path:
    """``<config_dir>/setup/state.json``."""
    return Path(config_dir) / "setup" / "state.json"


def load_setup_state(config_dir: Path) -> dict[str, Any] | None:
    """The persisted wizard state, or ``None`` when no wizard ran under ``config_dir``."""
    p = state_path(config_dir)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ConfigError(f"setup state {p} is unreadable: {type(exc).__name__}") from exc
    return data if isinstance(data, dict) else None


def should_run_setup(settings: Any) -> bool:
    """Setup mode: forced, an unfinished wizard run, or no model configured."""
    if settings.setup_mode:
        return True
    st = load_setup_state(settings.config_dir)
    if st is not None and not st.get("finished"):
        return True
    try:
        return not settings.all_models()
    except ConfigError:
        return False  # a broken models file is reported by the normal start-up, not hidden by the wizard


# =================================================================================================
# request bodies (closed: unknown fields are refused, so no dev option can be smuggled in)
# =================================================================================================
class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoginBody(_Body):
    token: str = Field(min_length=1, max_length=512)


class EnrollBody(_Body):
    code: str = Field(min_length=1, max_length=2000)
    gateway_name: str | None = Field(None, max_length=64)


class ModelBody(_Body):
    name: str = Field(min_length=1, max_length=128)


class MaskStoreBody(_Body):
    root: str | None = Field(None, max_length=1024, description="Where to keep masks (default <data_dir>/masks)")


class SSOBody(_Body):
    issuer: str = Field(min_length=1, max_length=512)
    client_id: str = Field(min_length=1, max_length=256)
    client_secret: str = Field(min_length=1, max_length=4096)
    redirect_url: str = Field(min_length=1, max_length=512)
    admin_values: list[str] = Field(default_factory=list, max_length=32)
    viewer_values: list[str] = Field(default_factory=list, max_length=32)
    label: str = Field("SSO", min_length=1, max_length=40)


class AdminBody(_Body):
    sso: SSOBody | None = None


def _parse(model: type[BaseModel], body: Any) -> Any:
    try:
        return model.model_validate(body if body is not None else {})
    except ValidationError as exc:
        errs: list[Any] = list(exc.errors())
        first: dict[str, Any] = dict(errs[0]) if errs else {}
        loc = ".".join(str(p) for p in first.get("loc", ()))
        if first.get("type") == "extra_forbidden":
            raise PGWError(f"'{loc}' is not a setup option (the wizard only offers production settings)",
                           code=ErrorCode.PROTOCOL, details={"param": loc}) from exc
        raise PGWError(f"invalid {loc or 'request'}: {first.get('msg', 'invalid value')}", code=ErrorCode.PROTOCOL,
                       details={"param": loc}) from exc


# =================================================================================================
# controller (framework-independent; tested directly)
# =================================================================================================
@dataclass
class SetupOptions:
    """Injection points of the wizard (tests, air-gapped installs).

    Attributes:
        checkpoint_resolver: ``(hf_repo, revision) -> local checkpoint`` (default: Hugging Face hub).
        enroll_transport: httpx transport for enrollment (tests).
        factory_launcher: ``(factory_env) -> supervisor`` with ``start()``, ``stop()`` and ``status()``
            (default: :class:`FactorySupervisor` running ``pgw factory run``).
        doctor: ``(**kwargs) -> DoctorReport`` (default :func:`pgw.cli.doctor.run_doctor`).
        static_dir: Console static files (default: the built console).
    """

    checkpoint_resolver: CheckpointResolver | None = None
    enroll_transport: Any = None
    factory_launcher: Callable[[dict[str, str]], Any] | None = None
    doctor: Callable[..., Any] | None = None
    static_dir: Path | None = None


class SetupController:
    """Wizard state machine. All public methods are blocking (call them from a worker thread)."""

    def __init__(self, settings: Any, options: SetupOptions | None = None):
        self.settings = settings
        self.options = options or SetupOptions()
        self.config_dir = Path(settings.config_dir)
        self.data_dir = Path(settings.data_dir)
        self._lock = threading.RLock()
        self.state: dict[str, Any] = load_setup_state(self.config_dir) or {
            "version": 1, "started_at": time.time(), "finished": False}
        if self.state.get("finished") and settings.setup_mode:
            # Explicit re-run (PGW_GATEWAY_SETUP_MODE=1) after a finished wizard: start over, keep nothing secret.
            self.state = {"version": 1, "started_at": time.time(), "finished": False}
        model = self.state.get("model")
        if isinstance(model, dict) and model.get("pack_status") == "building":
            model["pack_status"] = "interrupted"  # the gateway restarted while building: offer to start again
        self.pack_job: PackBuildJob | None = None
        self.hint_job: HintBuildJob | None = None
        self.factory: Any = None
        self._save()

    # ---- persistence ---------------------------------------------------------------------------------
    def _save(self) -> None:
        atomic_write_text(state_path(self.config_dir), json.dumps(self.state, indent=1, default=str) + "\n")

    def _require(self, *steps: str) -> None:
        for s in steps:
            if not self.state.get(s):
                raise PGWError(f"finish the '{s}' step first", code=ErrorCode.PROTOCOL, details={"step": s})

    # ---- paths ---------------------------------------------------------------------------------------
    @property
    def secrets_dir(self) -> Path:
        return self.config_dir / "secrets"

    def _pack_dir(self, name: str) -> Path:
        from pgw.enroll.client import _safe_dirname

        return self.data_dir / "packs" / _safe_dirname(name)

    # ---- views ---------------------------------------------------------------------------------------
    def view(self) -> dict[str, Any]:
        """State for the console (no secrets)."""
        with self._lock:
            st: dict[str, Any] = json.loads(json.dumps(self.state, default=str))
        model = st.get("model") or {}
        ring = model.get("ring") or "K32A12C27"
        pack_path = Path(model["pack_path"]) if model.get("pack_status") == "ready" else None
        st["estimate"] = mask_estimate(pack_path, ring, FACTORY_TARGET_SLOTS)
        st["estimate"]["disk_free_bytes"] = disk_free(self.data_dir)
        st["jobs"] = {
            "pack": self.pack_job.status() if self.pack_job is not None else None,
            "hint": self.hint_job.status() if self.hint_job is not None else None,
            "factory": self.factory.status() if self.factory is not None else None,
        }
        st["masks"] = self._masks_available()
        st["paths"] = {"config_dir": str(self.config_dir), "data_dir": str(self.data_dir)}
        st["kms_options"] = [{"id": "aws-kms", "label": "AWS KMS", "available": False},
                             {"id": "gcp-kms", "label": "Google Cloud KMS", "available": False},
                             {"id": "azure-kv", "label": "Azure Key Vault", "available": False}]
        st["steps"] = list(SETUP_STEPS)
        st["setup_mode"] = True
        return st

    # ---- step 1: enrollment --------------------------------------------------------------------------
    def enroll(self, body: EnrollBody) -> dict[str, Any]:
        from pgw.enroll.client import enroll_gateway

        with self._lock:
            res = enroll_gateway(body.code, self.config_dir, gateway_name=body.gateway_name or None,
                                 force=True, packs_dir=self.data_dir / "packs",
                                 transport=self.options.enroll_transport)
            self.state["enroll"] = {**res.public(), "at": time.time()}
            for later in ("model", "verify"):
                self.state.pop(later, None)
            self._save()
            return dict(self.state["enroll"])

    # ---- step 2: model ---------------------------------------------------------------------------------
    def choose_model(self, body: ModelBody) -> dict[str, Any]:
        with self._lock:
            self._require("enroll")
            models = {m["name"]: m for m in self.state["enroll"]["models"]}
            m = models.get(body.name)
            if m is None:
                raise PGWError(f"model {body.name!r} is not part of your enrollment", code=ErrorCode.MODEL_NOT_FOUND)
            if self.pack_job is not None and not self.pack_job.finished:
                raise PGWError("a model is already being prepared; wait for it to finish", code=ErrorCode.PROTOCOL)
            out_dir = self._pack_dir(m["name"])
            self.state["model"] = {"name": m["name"], "fingerprint": m["fingerprint"], "ring": m.get("ring"),
                                   "hf_repo": m.get("hf_repo"), "revision": m.get("revision"),
                                   "pack_path": str(out_dir), "pack_status": "building"}
            self.state.pop("verify", None)
            self._save()
            self.pack_job = PackBuildJob(m, out_dir, self.options.checkpoint_resolver, on_done=self._pack_done)
            self.pack_job.start()
            return dict(self.state["model"])

    def _pack_done(self, status: dict[str, Any]) -> None:
        with self._lock:
            model = self.state.get("model")
            if not isinstance(model, dict):
                return
            model["pack_status"] = "ready" if status["phase"] == "done" else "failed"
            model["error"] = status.get("error")
            self._save()

    # ---- step 3: mask store -----------------------------------------------------------------------------
    def init_maskstore(self, body: MaskStoreBody) -> dict[str, Any]:
        from pgw.maskstore.admin import init_store

        with self._lock:
            root = Path(body.root).expanduser() if body.root else self.data_dir / "masks"
            if not root.is_absolute():
                raise PGWError("the mask store location must be an absolute path", code=ErrorCode.PROTOCOL)
            key_file = self.secrets_dir / "maskstore.kek"
            counter_file = self.secrets_dir / "maskstore.counter"
            if key_file.resolve().is_relative_to(root.resolve()) or counter_file.resolve().is_relative_to(
                    root.resolve()):
                raise PGWError("keep the mask store outside the configuration directory", code=ErrorCode.PROTOCOL)
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.secrets_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            status = init_store(root, key_file, counter="file", counter_file=counter_file)
            self.state["maskstore"] = {"root": str(root), "key_file": str(key_file), "counter_file": str(counter_file),
                                       "store_id": status.get("store_id"), "at": time.time()}
            self.state.pop("verify", None)
            self._save()
            return dict(self.state["maskstore"])

    def maskstore_settings(self) -> dict[str, Any]:
        ms = self.state["maskstore"]
        return {"root": Path(ms["root"]), "key_provider": "file", "key_file": Path(ms["key_file"]),
                "counter": "file", "counter_file": Path(ms["counter_file"])}

    # ---- step 4: admin access ---------------------------------------------------------------------------
    def set_admin(self, body: AdminBody) -> dict[str, Any]:
        from pgw.gateway.config import GatewaySettings

        with self._lock:
            token = "pgw-admin-" + secrets.token_urlsafe(32)
            sso: dict[str, Any] | None = None
            secret_file = self.secrets_dir / "oidc_client_secret"
            if body.sso is not None:
                sso = {"console_oidc_issuer": body.sso.issuer, "console_oidc_client_id": body.sso.client_id,
                       "console_oidc_redirect_url": body.sso.redirect_url,
                       "console_oidc_admin_values": [v for v in body.sso.admin_values if v],
                       "console_oidc_viewer_values": [v for v in body.sso.viewer_values if v],
                       "console_oidc_label": body.sso.label, "console_oidc_client_secret_file": secret_file}
            try:  # validate exactly as the gateway will at start-up
                GatewaySettings(admin_token=token, **(sso or {}))
            except ValidationError as exc:
                errs: list[Any] = list(exc.errors())
                msg = errs[0].get("msg", "invalid") if errs else "invalid"
                raise PGWError(f"single sign-on settings are not valid: {msg}",
                               code=ErrorCode.PROTOCOL) from exc
            token_file = self.secrets_dir / "admin_token"
            write_private(token_file, token + "\n", append=False)
            if body.sso is not None:
                write_private(secret_file, body.sso.client_secret + "\n", append=False)
            self.state["admin"] = {"token_file": str(token_file), "at": time.time(),
                                   "sso": {k: (str(v) if isinstance(v, Path) else v) for k, v in (sso or {}).items()}
                                   or None}
            self._save()
            return {"admin_token": token, "token_file": str(token_file), "sso": body.sso is not None}

    # ---- step 5: verify -----------------------------------------------------------------------------------
    def factory_env(self) -> dict[str, str]:
        en, model = self.state["enroll"], self.state["model"]
        files = en.get("files") or {}
        env = {
            "PGW_FACTORY_PACK_PATH": model["pack_path"],
            "PGW_FACTORY_RING": model.get("ring") or "K32A12C27",
            "PGW_FACTORY_HE_SERVICE_URL": en["service_url"],
            "PGW_FACTORY_HE_TOKEN_FILE": files["token"],
            "PGW_FACTORY_HE_HINT_DIR": str(self.data_dir / "hemask"),
            "PGW_FACTORY_STATE_DIR": str(self.data_dir / "factory"),
            "PGW_FACTORY_LOW_WATER_SLOTS": str(FACTORY_LOW_WATER_SLOTS),
            "PGW_FACTORY_TARGET_SLOTS": str(FACTORY_TARGET_SLOTS),
            "PGW_FACTORY_FIRST_SEGMENT_SLOTS": str(FACTORY_FIRST_SEGMENT_SLOTS),
        }
        if files.get("ca"):
            env["PGW_FACTORY_HE_CA_FILE"] = files["ca"]
        if files.get("client_cert") and files.get("client_key"):
            # the gateway's enrolled client certificate: needed when the service requires mTLS for compute
            env["PGW_FACTORY_HE_CLIENT_CERT_FILE"] = files["client_cert"]
            env["PGW_FACTORY_HE_CLIENT_KEY_FILE"] = files["client_key"]
        ms = self.state["maskstore"]
        env.update({"PGW_MASKSTORE_ROOT": ms["root"], "PGW_MASKSTORE_KEY_PROVIDER": "file",
                    "PGW_MASKSTORE_KEY_FILE": ms["key_file"], "PGW_MASKSTORE_COUNTER": "file",
                    "PGW_MASKSTORE_COUNTER_FILE": ms["counter_file"]})
        return env

    def _start_masks(self) -> None:
        """Hint build, then the supervised factory (idempotent)."""
        model = self.state["model"]
        if self.hint_job is None or (self.hint_job.finished and self.hint_job.status()["phase"] == "failed"):
            def after_hint(status: dict[str, Any]) -> None:
                if status["phase"] == "done":
                    self._start_factory()

            self.hint_job = HintBuildJob(Path(model["pack_path"]), model.get("ring") or "K32A12C27",
                                         self.data_dir / "hemask", on_done=after_hint)
            self.hint_job.start()
        elif self.hint_job.status()["phase"] == "done":
            self._start_factory()

    def _start_factory(self) -> None:
        with self._lock:
            if self.factory is not None:
                return
            env = self.factory_env()
            launcher = self.options.factory_launcher or (lambda e: FactorySupervisor(e))
            self.factory = launcher(env)
            self.factory.start()

    def _masks_available(self) -> int | None:
        ms, model = self.state.get("maskstore"), self.state.get("model")
        if not ms or not model or model.get("pack_status") != "ready":
            return None
        try:
            from pgw.maskstore.admin import store_status
            from pgw.maskstore.config import MaskStoreSettings

            st = store_status(MaskStoreSettings(**self.maskstore_settings()))
        except PGWError:
            return None
        n = 0
        for e in st.get("inventory") or []:
            if isinstance(e, dict) and str(e.get("fingerprint", "")).startswith(model["fingerprint"][:12]):
                n += int(e.get("slots_available", 0) or 0)
        return n

    def verify(self) -> dict[str, Any]:
        """Run the checks and start mask production; returns the plain-language checklist."""
        with self._lock:
            self._require("enroll", "model", "maskstore", "admin")
            if self.state["model"].get("pack_status") != "ready":
                raise PGWError("the model is not ready yet", code=ErrorCode.PROTOCOL)
            self._start_masks()
        checks = self._checks()
        result = {"checks": checks, "ok": not any(c["status"] == "fail" for c in checks), "at": time.time()}
        with self._lock:
            self.state["verify"] = result
            self._save()
        return result

    def _checks(self) -> list[dict[str, Any]]:
        from pgw.maskstore.admin import store_status
        from pgw.maskstore.config import MaskStoreSettings

        en, model = self.state["enroll"], self.state["model"]
        files = en.get("files") or {}
        doctor: Callable[..., Any]
        if self.options.doctor is not None:
            doctor = self.options.doctor
        else:
            from pgw.cli.doctor import run_doctor

            doctor = run_doctor
        report = doctor(role="all", pack=Path(model["pack_path"]), ring=model.get("ring"),
                        service_url=en["service_url"],
                        service_token_file=Path(files["token"]) if files.get("token") else None,
                        service_ca_file=Path(files["ca"]) if files.get("ca") else None, include_gpu=False)
        out: list[dict[str, Any]] = []

        def add(cid: str, label: str, status: str, detail: str) -> None:
            out.append({"id": cid, "label": label, "status": status, "detail": detail})

        def from_doctor(name: str, cid: str, label: str, ok_text: str) -> None:
            c = report.get(name)
            if c is None:
                add(cid, label, "fail", "check did not run")
            elif c.status in ("ok", "skip"):
                add(cid, label, "ok", ok_text)
            else:
                add(cid, label, "fail" if c.status == "fail" else "warn", c.detail)

        pinned = bool(en.get("ca_fingerprint")) and en["service_url"].startswith("https://")
        from_doctor("service", "service", "Compute service reachable" + (" over pinned TLS" if pinned else ""),
                    f"{en['service_url']} answers and serves your model"
                    + (" (certificate authority pinned at enrollment)" if pinned else ""))
        from_doctor("pack", "pack", "Model verified", f"{model['name']} matches the service's fingerprint "
                    f"{model['fingerprint'][:12]}")
        for name, label in (("entropy", "Secure random numbers"), ("selftest", "Exact arithmetic self test")):
            from_doctor(name, name, label, "passed")
        try:
            st = store_status(MaskStoreSettings(**self.maskstore_settings()))
            rb = st.get("rollback_protection") or {}
            if rb.get("counter", "none") == "none":
                add("maskstore", "Mask store protected", "fail", "no monotonic counter configured")
            else:
                add("maskstore", "Mask store protected", "ok",
                    "encrypted with a local key (0600) and protected against snapshot restore by a counter file")
        except PGWError as exc:
            add("maskstore", "Mask store protected", "fail", exc.message)
        avail = self._masks_available() or 0
        hint = self.hint_job.status() if self.hint_job is not None else {"phase": "queued"}
        fac = self.factory.status() if self.factory is not None else None
        if avail > 0:
            add("masks", "Masks available", "ok", f"{avail} tokens of one-time masks ready; the factory keeps "
                f"between {FACTORY_LOW_WATER_SLOTS} and {FACTORY_TARGET_SLOTS} in stock")
        elif hint["phase"] == "failed":
            add("masks", "Masks available", "fail", f"building the encryption hint failed: {hint.get('error')}")
        elif fac is not None and fac.get("running"):
            add("masks", "Masks available", "warn", "the mask factory is running; the first masks arrive shortly")
        else:
            add("masks", "Masks available", "warn", "preparing encrypted mask generation (one-time hint build); "
                "you can finish now, requests wait until the first masks are ready")
        add("freivalds", "Every result checked (Freivalds)", "ok",
            "each answer from the compute service is verified before use; a wrong answer blocks the model")
        add("noise", "Outbound noise monitor", "ok", "on, fail-closed: traffic stops if outgoing data stops "
            "looking random")
        if self.settings.allow_dev_modes:
            add("policy", "Production policy", "fail", "PGW_GATEWAY_ALLOW_DEV_MODES is set; remove it")
        else:
            add("policy", "Production policy", "ok", "no development modes")
        from urllib.parse import urlsplit

        u = urlsplit(en["service_url"])
        add("egress", "Network advice", "info",
            f"allow outbound HTTPS from this gateway only to {u.hostname}:{u.port or 443} (and the model download "
            "from huggingface.co during setup); block everything else")
        return out

    # ---- step 6: finish ------------------------------------------------------------------------------------
    def final_settings(self) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        """``(gateway settings kwargs, gateway.env lines, factory.env lines)``."""
        from pgw.enroll.client import model_entry

        en, model, admin = self.state["enroll"], self.state["model"], self.state["admin"]
        files = {k: Path(v) for k, v in (en.get("files") or {}).items()}
        state_dir = self.data_dir / "gateway"
        entry = model_entry({"name": model["name"], "fingerprint": model["fingerprint"], "ring": model.get("ring")},
                            Path(model["pack_path"]), en["service_url"], files)
        if self.settings.setup_model_options:  # deployment presets (speed settings only; validated below)
            from pgw.gateway.config import GatewayModelConfig

            entry = {**entry, **{k: (str(v) if isinstance(v, Path) else v)
                                 for k, v in self.settings.setup_model_options.items()}}
            try:
                GatewayModelConfig(**entry)
            except ValidationError as exc:
                raise PGWError(f"setup_model_options is not valid: {exc.errors()[0].get('msg')}",
                               code=ErrorCode.CONFIG) from exc
        models_file = self.config_dir / "models.json"
        atomic_write_text(models_file, json.dumps([entry], indent=2) + "\n", mode=0o600)
        factory_env_file = self.config_dir / "factory.env"
        gw: dict[str, Any] = {
            "models_file": models_file,
            "maskstore_root": Path(self.state["maskstore"]["root"]),
            "admin_token_file": Path(admin["token_file"]),
            "api_keys_file": state_dir / "api-keys.json",
            "audit_log_file": state_dir / "audit.jsonl",
            "quota_state_file": state_dir / "quota.json",
            "config_dir": self.config_dir,
            "data_dir": self.data_dir,
            "factory_env_file": factory_env_file,
        }
        for k, v in (admin.get("sso") or {}).items():
            gw[k] = v
        env: dict[str, str] = {}
        for k, v in gw.items():
            env[f"PGW_GATEWAY_{k.upper()}"] = json.dumps(v) if isinstance(v, list) else str(v)
        for k, v in self.maskstore_settings().items():
            env[f"PGW_MASKSTORE_{k.upper()}"] = str(v)
        return gw, env, self.factory_env()

    def finish(self) -> dict[str, Any]:
        """Write the final configuration (validated with the production policy) and mark setup finished."""
        from pgw.gateway.config import GatewaySettings

        with self._lock:
            self._require("enroll", "model", "maskstore", "admin", "verify")
            if not self.state["verify"].get("ok"):
                raise PGWError("the checks found problems; fix them and run the checks again", code=ErrorCode.PROTOCOL)
            gw, env, fenv = self.final_settings()
            try:
                gs = GatewaySettings(**{k: v for k, v in gw.items() if k != "admin_token_file"},
                                     admin_token_file=gw["admin_token_file"], setup_mode=False)
                gs.validate_policy()
            except ValidationError as exc:
                raise PGWError(f"the final configuration is not valid: {exc.errors()[0].get('msg')}",
                               code=ErrorCode.CONFIG) from exc
            if gs.allow_dev_modes:
                raise PGWError("PGW_GATEWAY_ALLOW_DEV_MODES is set in the environment; remove it to finish",
                               code=ErrorCode.CONFIG)
            header = "# Written by the PGW setup wizard. Load with: set -a; . THIS_FILE; set +a\n"
            atomic_write_text(self.config_dir / "gateway.env",
                              header + "".join(f"{k}={v}\n" for k, v in sorted(env.items())))
            atomic_write_text(self.config_dir / "factory.env",
                              header + "".join(f"{k}={v}\n" for k, v in sorted(fenv.items())))
            self.state["finished"] = True
            self.state["finished_at"] = time.time()
            self._save()
            return {"gateway_settings": gs, "env_file": str(self.config_dir / "gateway.env")}

    def close(self) -> None:
        if self.factory is not None:
            with contextlib.suppress(Exception):
                self.factory.stop()


# =================================================================================================
# switchable ASGI app (setup -> normal mode without a restart)
# =================================================================================================
class SwitchableApp:
    """ASGI app that serves the setup app until :meth:`switch` installs the normal gateway app.

    It handles the ASGI lifespan itself: the normal app's lifespan is entered on switch and exited
    at shutdown; the supervised mask factory (if any) is stopped last.
    """

    def __init__(self, app: Any):
        self.current = app
        self._stack: contextlib.AsyncExitStack | None = None
        self._on_close: list[Callable[[], Any]] = []
        self.switched = False

    @property
    def state(self) -> Any:
        return self.current.state

    def on_close(self, fn: Callable[[], Any]) -> None:
        self._on_close.append(fn)

    async def switch(self, app: Any) -> None:
        stack = contextlib.AsyncExitStack()
        await stack.enter_async_context(app.router.lifespan_context(app))
        self._stack = stack
        self.current = app
        self.switched = True

    async def aclose(self) -> None:
        if self._stack is not None:
            stack, self._stack = self._stack, None
            await stack.aclose()
        for fn in self._on_close:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(fn)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    try:
                        await self.aclose()
                    finally:
                        await send({"type": "lifespan.shutdown.complete"})
                    return
        await self.current(scope, receive, send)


# =================================================================================================
# the setup app
# =================================================================================================
def new_setup_token() -> str:
    """Random one-time setup token (printed at start-up; only its digest is kept)."""
    return "pgw-setup-" + secrets.token_urlsafe(18)


def create_setup_app(
    settings: Any,
    *,
    setup_token: str,
    options: SetupOptions | None = None,
    normal_app_factory: Callable[[Any, Any], Awaitable[Any] | Any] | None = None,
) -> SwitchableApp:
    """Build the setup-mode app (wrapped in a :class:`SwitchableApp`).

    Args:
        settings: :class:`pgw.gateway.config.GatewaySettings` of this process.
        setup_token: The one-time token the wizard requires (print it for the administrator).
        options: Injection points (tests).
        normal_app_factory: ``(final GatewaySettings, mask store kwargs) -> normal app``; default
            :func:`pgw.gateway.app.create_app` with the mask store opened from the wizard's settings.
    """
    from pgw.console.routes import _MEDIA_TYPES, _NOT_BUILT, DEFAULT_STATIC_DIR, ConsoleHeaders, _static_file

    opts = options or SetupOptions()
    controller = SetupController(settings, opts)
    sessions = ConsoleSessions(setup_token, None, idle_s=4 * 3600.0, max_age_s=24 * 3600.0, max_sessions=20)
    limiter = LoginRateLimiter(attempts=10)
    root = opts.static_dir if opts.static_dir is not None else DEFAULT_STATIC_DIR
    app = FastAPI(title="PGW gateway setup", docs_url=None, redoc_url=None, openapi_url=None)
    outer = SwitchableApp(app)
    outer.on_close(controller.close)
    app.state.setup = controller
    app.add_middleware(ConsoleHeaders)

    def _err(exc: PGWError) -> JSONResponse:
        msg = "internal error" if exc.code == ErrorCode.INTERNAL else exc.message
        headers = {}
        retry = exc.details.get("retry_after_s")
        if isinstance(retry, int | float):
            headers["Retry-After"] = str(max(1, round(retry)))
        return JSONResponse({"error": {"message": msg, "type": "setup_error", "code": str(exc.code),
                                       "param": exc.details.get("param")}},
                            status_code=exc.http_status, headers=headers)

    @app.exception_handler(PGWError)
    async def _pgw_error(request: Request, exc: PGWError) -> Any:
        return _err(exc)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> Any:
        log.error("setup: unhandled error on %s: %s", request.url.path, type(exc).__name__)
        return _err(PGWError("internal error"))

    def _client(request: Request) -> str:
        from pgw.core.client_ip import client_ip

        return client_ip(request)

    def _session(request: Request) -> Any:
        sid = request.cookies.get(COOKIE_NAME)
        if not sid:
            raise AuthError("sign in with the setup token first")
        return sessions.authorize(sid, request.method, request.headers.get(CSRF_HEADER))

    async def _step(fn: Callable[..., Any], *args: Any) -> Any:
        """Run a wizard step off the event loop. Problems with the admin's input or environment are 400s with
        the plain-language message (a 401 here would read as "session expired" to the console)."""
        try:
            return await asyncio.to_thread(fn, *args)
        except PGWError as exc:
            if exc.code in _INPUT_ERRORS:
                raise PGWError(exc.message, code=ErrorCode.PROTOCOL, details={**exc.details, "reason": str(exc.code)},
                               ) from exc
            raise

    async def _json(request: Request) -> Any:
        raw = await request.body()
        if len(raw) > 64 << 10:
            raise PGWError("request body too large", code=ErrorCode.PAYLOAD_TOO_LARGE)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise PGWError("body must be JSON", code=ErrorCode.PROTOCOL) from exc

    # ---- ops ----------------------------------------------------------------------------------------
    @app.get("/healthz")
    async def healthz() -> Any:
        return Response("ok", media_type="text/plain")

    @app.get("/readyz")
    async def readyz() -> Any:
        return JSONResponse({"ready": False, "setup_mode": True}, status_code=503)

    # ---- sign-in -----------------------------------------------------------------------------------
    @app.get("/console/api/setup/status")
    async def setup_status(request: Request) -> Any:
        sid = request.cookies.get(COOKIE_NAME)
        return {"setup_mode": True, "signed_in": sessions.get(sid, touch=False) is not None if sid else False}

    @app.post("/console/api/setup/login")
    async def setup_login(request: Request) -> Any:
        client = _client(request)
        with limiter.attempt(client) as att:  # only failed sign-ins count; in-flight ones until settled
            body = _parse(LoginBody, await _json(request))
            if sessions.authenticate(body.token) != "admin":
                att.fail()
                log.warning("setup: wrong setup token from %s", client)
                raise AuthError("that setup token was not accepted; copy it from the gateway's log")
        sid, sess = sessions.create("admin")
        log.info("setup: signed in from %s", client)
        resp = JSONResponse({"role": "admin", "csrf": sess.csrf, "setup": True})
        resp.set_cookie(COOKIE_NAME, sid, max_age=int(sessions.max_age_s), path="/", httponly=True,
                        samesite="strict",
                        secure=cookie_secure(request.url.scheme, request.url.hostname, client))
        return resp

    @app.get("/console/api/me")
    async def me(request: Request) -> Any:
        sid = request.cookies.get(COOKIE_NAME)
        s = sessions.get(sid) if sid else None
        if s is None:
            raise AuthError("not signed in")
        import pgw

        return {"role": "admin", "csrf": s.csrf, "expires_at": int(sessions.expires_at(s)), "version": pgw.__version__,
                "auth": "token", "user": None, "setup": True}

    @app.get("/console/api/auth-methods")
    async def auth_methods() -> Any:
        return {"token": False, "oidc": {"enabled": False, "label": "SSO"}, "setup": True}

    @app.post("/console/api/logout", response_model=None)
    async def logout(request: Request) -> Any:
        sid = request.cookies.get(COOKIE_NAME)
        if sid and sessions.get(sid, touch=False) is not None:
            _session(request)  # CSRF
            sessions.delete(sid)
        resp = Response(status_code=204)
        resp.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="strict")
        return resp

    # ---- wizard steps ------------------------------------------------------------------------------
    @app.get("/console/api/setup/state")
    async def get_state(request: Request) -> Any:
        _session(request)
        return await asyncio.to_thread(controller.view)

    @app.post("/console/api/setup/enroll")
    async def post_enroll(request: Request) -> Any:
        _session(request)
        body = _parse(EnrollBody, await _json(request))
        return await _step(controller.enroll, body)

    @app.post("/console/api/setup/model")
    async def post_model(request: Request) -> Any:
        _session(request)
        body = _parse(ModelBody, await _json(request))
        return await _step(controller.choose_model, body)

    @app.post("/console/api/setup/maskstore")
    async def post_maskstore(request: Request) -> Any:
        _session(request)
        body = _parse(MaskStoreBody, await _json(request))
        return await _step(controller.init_maskstore, body)

    @app.post("/console/api/setup/admin")
    async def post_admin(request: Request) -> Any:
        _session(request)
        body = _parse(AdminBody, await _json(request))
        return await _step(controller.set_admin, body)

    @app.post("/console/api/setup/verify")
    async def post_verify(request: Request) -> Any:
        _session(request)
        _parse(_Body, await _json(request))
        return await _step(controller.verify)

    switch_lock = asyncio.Lock()

    @app.post("/console/api/setup/finish")
    async def post_finish(request: Request) -> Any:
        _session(request)
        _parse(_Body, await _json(request))
        async with switch_lock:
            if outer.switched:
                return {"mode": "normal", "message": "setup is already finished"}
            res = await _step(controller.finish)
            notes = []
            if settings.setup_mode:
                notes.append("Remove PGW_GATEWAY_SETUP_MODE from the gateway's environment, otherwise the next "
                             "restart opens the wizard again.")
            env_file, cfg = res["env_file"], settings.config_dir
            restart = (f"Restart the gateway with 'pgw gateway serve' and PGW_GATEWAY_CONFIG_DIR={cfg}: it reads "
                       f"{env_file} by itself (under systemd you can also use EnvironmentFile={env_file}).")
            try:
                new = await _build_normal_app(res["gateway_settings"])
                await outer.switch(new)
            except Exception as exc:
                msg = exc.message if isinstance(exc, PGWError) else type(exc).__name__
                log.error("setup finished, but the gateway could not switch to normal mode in-process: %s", msg)
                return {"mode": "restart_required", "message": f"Setup is saved, but the gateway could not start "
                        f"serving yet ({msg}). {restart}", "notes": notes}
            log.info("setup finished: gateway switched to normal mode")
            return {"mode": "normal", "message": "Setup is complete. The gateway now serves your model. Sign in "
                    "to the console with the admin token you saved. Until the first batch of privacy masks is "
                    "ready (a few minutes after setup), the model shows as 'preparing privacy masks' and then "
                    "becomes ready by itself. After a restart the gateway keeps this "
                    f"configuration: start it with PGW_GATEWAY_CONFIG_DIR={cfg} and it reads {env_file} by itself.",
                    "notes": notes}

    async def _build_normal_app(gs: Any) -> Any:
        ms_kwargs = controller.maskstore_settings()
        if normal_app_factory is not None:
            out = normal_app_factory(gs, ms_kwargs)
            return await out if asyncio.iscoroutine(out) else out
        from pgw.gateway.app import create_app
        from pgw.maskstore.admin import open_store
        from pgw.maskstore.config import MaskStoreSettings

        store = await asyncio.to_thread(open_store, MaskStoreSettings(**ms_kwargs))
        return create_app(gs, mask_store=store)

    @app.api_route("/console/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def api_missing(rest: str) -> Any:
        return JSONResponse({"error": {"message": "the gateway is in setup mode", "type": "setup_error",
                                       "code": str(ErrorCode.UNAVAILABLE), "param": None}}, status_code=503)

    # ---- console static app ---------------------------------------------------------------------------
    def _index() -> Response:
        index = _static_file(root, "index.html")
        if index is None:
            return HTMLResponse(_NOT_BUILT, headers={"Cache-Control": "no-store"})
        return FileResponse(index, media_type=_MEDIA_TYPES[".html"], headers={"Cache-Control": "no-cache"})

    @app.get("/")
    async def home() -> Any:
        return RedirectResponse("/console/setup", status_code=302)

    @app.get("/console")
    async def console_redirect() -> Any:
        return RedirectResponse("/console/", status_code=308)

    @app.get("/console/{path:path}")
    async def console_static(path: str) -> Any:
        f = _static_file(root, path) if path else None
        if f is not None and f.name != "index.html":
            media = _MEDIA_TYPES.get(f.suffix.lower(), "application/octet-stream")
            cache = "public, max-age=31536000, immutable" if path.startswith("assets/") else "no-cache"
            return FileResponse(f, media_type=media, headers={"Cache-Control": cache})
        last = path.rsplit("/", 1)[-1]
        if path.startswith("assets/") or "." in last:
            return Response("not found", status_code=404, media_type="text/plain")
        return _index()

    @app.api_route("/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def everything_else(rest: str) -> Any:
        return JSONResponse({"error": {"message": "the gateway is in setup mode: finish the setup wizard at "
                                       "/console/ first", "type": "setup_error", "code": str(ErrorCode.UNAVAILABLE),
                                       "param": None}}, status_code=503)

    return outer

