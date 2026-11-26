"""Unit tests of the client library pieces that need no provider: options, models, errors, types, request
validation, cache directories and locks, provisioning adapter, factory environment, CLI, lazy imports."""

from __future__ import annotations

import itertools
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pgw.core.errors import ErrorCode, MaskExhaustedError, PGWError
from pgw.core.errors import IntegrityError as CoreIntegrityError
from pgw.sdk import errors as E
from pgw.sdk.keys import SdkKey

ROOT = Path(__file__).resolve().parents[2]


# ---- models and options -------------------------------------------------------------------------------
@pytest.mark.parametrize("name, want", [("qwen2.5-7b-instruct", "qwen2.5-7b-instruct"),
                                        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
                                        (" QWEN2.5-0.5B-INSTRUCT ", "qwen2.5-0.5b-instruct"),
                                        ("qwen2.5-1.5b-instruct", "qwen2.5-1.5b-instruct"),
                                        ("qwen2.5-3b-instruct", "qwen2.5-3b-instruct")])
def test_supported_models(name: str, want: str) -> None:
    from pgw.sdk.options import canonical_model

    assert canonical_model(name) == want


@pytest.mark.parametrize("name", ["gpt-4o", "qwen2.5-32b-instruct", "llama-3-8b", "", "qwen2.5-7b"])
def test_unsupported_models_are_refused_clearly(name: str) -> None:
    from pgw.sdk.options import canonical_model

    with pytest.raises(E.UnsupportedModelError, match="supported") as ei:
        canonical_model(name)
    assert isinstance(ei.value, E.NotFoundError) and ei.value.status_code == 404


def test_options_defaults_env_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.sdk.options import SDKOptions, default_mask_workers

    o = SDKOptions.from_env()
    assert o.mask_scheme == "pks-v3" and o.native and o.masked_head == "auto" and o.prefix_cache == "auto"
    assert o.lane == "auto" and o.factory and o.mask_nice == 10 and o.log_level is None
    assert o.workers == default_mask_workers() and 1 <= o.workers <= 8
    assert default_mask_workers(1) == 1 and default_mask_workers(16) == 4 and default_mask_workers(128) == 8
    monkeypatch.setenv("PGW_MASK_WORKERS", "3")
    monkeypatch.setenv("PGW_MASK_SCHEME", "mlwe")
    o = SDKOptions.from_env(mask_store_gb=2.0)
    assert o.workers == 3 and o.mask_scheme == "mlwe" and o.mask_store_gb == 2.0
    with pytest.raises(TypeError, match="unknown"):
        SDKOptions.from_env(no_such_option=1)
    for bad in ({"mask_scheme": "rot13"}, {"lane": "udp"}, {"mask_workers": 0}, {"mask_nice": 30},
                {"prefix_cache": True, "masked_head": "on"}):
        with pytest.raises(ValueError):
            SDKOptions.from_env(**bad)


# ---- errors --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("code, cls", [
    (ErrorCode.UNAUTHENTICATED, E.AuthenticationError), (ErrorCode.PERMISSION_DENIED, E.PermissionDeniedError),
    (ErrorCode.RATE_LIMITED, E.RateLimitError), (ErrorCode.INTEGRITY, E.IntegrityError),
    (ErrorCode.MODEL_NOT_FOUND, E.NotFoundError), (ErrorCode.MASK_EXHAUSTED, E.NotReadyError),
    (ErrorCode.UNAVAILABLE, E.NotReadyError), (ErrorCode.BACKEND_UNAVAILABLE, E.APIConnectionError),
    (ErrorCode.DEADLINE_EXCEEDED, E.APITimeoutError), (ErrorCode.PROTOCOL, E.BadRequestError),
    (ErrorCode.INTERNAL, E.InternalServerError), (ErrorCode.MASK_REUSE, E.IntegrityError),
])
def test_error_mapping(code: ErrorCode, cls: type[E.PGWAPIError]) -> None:
    e = E.from_pgw_error(PGWError("msg", code=code, details={"model": "m", "blob": object()}), request_id="r1")
    assert type(e) is cls and e.code == str(code) and e.request_id == "r1" and e.details == {"model": "m"}
    assert isinstance(e, E.APIError) and e.body["code"] == str(code)


def test_integrity_errors_are_never_retryable() -> None:
    e = E.from_pgw_error(CoreIntegrityError("Freivalds check failed"))
    assert isinstance(e, E.IntegrityError) and not e.retryable
    assert E.from_pgw_error(MaskExhaustedError("x")).retryable
    assert isinstance(E.from_pgw_error(TimeoutError()), E.APITimeoutError)
    assert isinstance(E.from_pgw_error(ConnectionResetError()), E.APIConnectionError)
    assert isinstance(E.from_pgw_error(ValueError()), E.InternalServerError)


# ---- types ---------------------------------------------------------------------------------------------
def test_types_are_openai_shaped() -> None:
    from pgw.sdk import types as T

    c = T.ChatCompletion.model_validate({
        "id": "chatcmpl-1", "created": 1, "model": "m", "system_fingerprint": "abc",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop",
                     "logprobs": {"content": [{"token": "hi", "logprob": -0.5, "bytes": [104, 105],
                                               "top_logprobs": [{"token": "hi", "logprob": -0.5, "bytes": [104]}]}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})
    assert c.choices[0].message.content == "hi" and c.usage.total_tokens == 4
    assert c.choices[0].logprobs.content[0].top_logprobs[0].logprob == -0.5
    d = c.to_dict()
    assert d["object"] == "chat.completion" and json.loads(c.to_json())["id"] == "chatcmpl-1"
    ml = T.ModelList(data=[T.Model(id="a"), T.Model(id="b")])
    assert [m.id for m in ml] == ["a", "b"] and len(ml) == 2
    try:
        import openai  # noqa: F401
    except ImportError:
        return
    oc = c.to_openai()  # pragma: no cover - only with openai installed
    assert oc.choices[0].message.content == "hi"


# ---- request validation (same rules as the gateway API) -----------------------------------------------
def test_request_validation() -> None:
    from pgw.sdk.client import _build_request, _conversation, _strip_conversation

    msgs = [{"role": "user", "content": "hi"}]
    r = _build_request({"model": "m", "messages": msgs, "max_tokens": 5, "temperature": 0, "seed": 3,
                        "stop": ("a", "b"), "logprobs": True, "top_logprobs": 2, "timeout": 5,
                        "extra_headers": {"x": "y"}, "extra_body": {"ignore_eos": True, "top_k": 4}})
    assert r.requested_max_tokens == 5 and r.seed == 3 and r.stop == ["a", "b"] and r.ignore_eos and r.top_k == 4
    for bad in ({"tools": [{"type": "function"}]}, {"response_format": {"type": "json_object"}}, {"n": 2},
                {"temperature": 3}, {"top_logprobs": 50}, {"messages": []}):
        with pytest.raises(E.BadRequestError):
            _build_request({"model": "m", "messages": msgs, **bad})
    kw = {"model": "m", "messages": msgs, "extra_body": {"conversation_id": "c1", "ignore_eos": True}}
    assert _conversation(kw) == "c1" and "conversation_id" not in _strip_conversation(kw)["extra_body"]
    assert _conversation({"extra_headers": {"X-PGW-Conversation-Id": "c2"}}) == "c2"
    with pytest.raises(E.BadRequestError):
        _conversation({"extra_body": {"conversation_id": "x" * 200}})


def test_message_objects_are_accepted() -> None:
    from pydantic import BaseModel

    from pgw.sdk.client import _build_request

    class Msg(BaseModel):
        role: str
        content: str

    r = _build_request({"model": "m", "messages": [Msg(role="user", content="hi")]})
    assert r.messages[0].content == "hi"


# ---- client construction ---------------------------------------------------------------------------
def test_client_needs_a_valid_key_and_does_no_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw import OpenAI

    monkeypatch.delenv("PGW_API_KEY", raising=False)
    with pytest.raises(E.AuthenticationError, match="PGW_API_KEY"):
        OpenAI(cache_dir=tmp_path / "c", state_dir=tmp_path / "s")
    with pytest.raises(E.AuthenticationError):
        OpenAI(api_key="sk-not-a-pgw-key", cache_dir=tmp_path / "c", state_dir=tmp_path / "s")
    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [7])).encode()
    monkeypatch.setenv("PGW_API_KEY", key)
    c = OpenAI(cache_dir=tmp_path / "c", state_dir=tmp_path / "s", base_url="https://ignored.example.com",
               organization="ignored", timeout=12, max_retries=0, mask_workers=2)
    assert c.options.timeout_s == 12 and c.options.max_retries == 0 and c.options.workers == 2
    assert not (tmp_path / "c").exists() and not (tmp_path / "s").exists()  # nothing until the first request
    c2 = OpenAI(cache_dir=tmp_path / "c", state_dir=tmp_path / "s")
    assert c2._engine is c._engine  # one engine per key and directories in a process
    with pytest.raises(E.UnsupportedModelError):
        c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    c.close()
    c2.close()
    assert c._engine._closed
    with pytest.raises(E.PGWAPIError, match="closed"):
        c.chat.completions.create(model="qwen2.5-7b-instruct", messages=[{"role": "user", "content": "x"}])
    with OpenAI(api_key=key, cache_dir=tmp_path / "c", state_dir=tmp_path / "s") as c3:
        assert c3.api_key_id == c._key.key_id_hex


def test_import_pgw_is_light() -> None:
    code = ("import sys, pgw; heavy = [m for m in ('torch', 'numpy', 'httpx', 'pgw.sdk') if m in sys.modules]; "
            "print(heavy); assert not heavy; from pgw import OpenAI, AsyncOpenAI; print(OpenAI.__module__)")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)  # noqa: S603
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["[]", "pgw.sdk.client"]


def test_library_is_silent_by_default() -> None:
    """No output on stdout or stderr from creating a client and failing a request (logs only when asked)."""
    code = ("import pgw, logging\n"
            "from pgw.sdk.keys import SdkKey\n"
            "k = SdkKey.new('dev', 'http://127.0.0.1:9', None).encode()\n"
            "c = pgw.OpenAI(api_key=k, cache_dir='{d}/c', state_dir='{d}/s', max_retries=0)\n"
            "try:\n    c.models.list()\nexcept pgw.sdk.APIConnectionError:\n    pass\n"
            "logging.getLogger('pgw.sdk.test').warning('not shown')\n"
            "c.close()\n")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
        env.pop("PGW_SDK_LOG", None)
        r = subprocess.run([sys.executable, "-c", code.format(d=d)], capture_output=True, text=True, env=env,  # noqa: S603
                           timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "" and r.stderr == ""


# ---- cache dirs and locks ----------------------------------------------------------------------------
def test_cache_dirs_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.sdk.cache import CacheLayout, default_cache_dir, default_state_dir, ensure_private_dir

    monkeypatch.setenv("PGW_CACHE_DIR", str(tmp_path / "env-cache"))
    monkeypatch.setenv("PGW_STATE_DIR", str(tmp_path / "env-state"))
    assert default_cache_dir() == tmp_path / "env-cache" and default_state_dir() == tmp_path / "env-state"
    monkeypatch.delenv("PGW_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "pgw"
    lay = CacheLayout(tmp_path / "c", tmp_path / "s").ensure()
    for d in (lay.root, lay.state):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o755)
    loose.chmod(0o755)
    ensure_private_dir(loose)
    assert stat.S_IMODE(loose.stat().st_mode) == 0o700
    (tmp_path / "target").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(PGWError, match="symlink"):
        ensure_private_dir(tmp_path / "link")
    assert lay.masks_dir("k").parent == lay.session_dir("k")
    assert lay.session_state_dir("k").is_relative_to(lay.state)


def test_file_lock_excludes_other_processes(tmp_path: Path) -> None:
    from pgw.sdk.cache import HeldLock, file_lock

    lock = tmp_path / "x.lock"
    code = ("import sys, time; sys.path[:0] = sys.argv[2:]; from pgw.sdk.cache import file_lock\n"
            "with file_lock(sys.argv[1]):\n    print('held', flush=True); time.sleep(1.5)\n")
    env = {**os.environ}
    p = subprocess.Popen([sys.executable, "-c", code, str(lock), str(ROOT / "src")], stdout=subprocess.PIPE,  # noqa: S603
                         text=True, env=env)
    assert p.stdout is not None and p.stdout.readline().strip() == "held"
    with pytest.raises(TimeoutError):
        with file_lock(lock, timeout_s=0.3):
            pass
    t0 = time.monotonic()
    with file_lock(lock, timeout_s=10):
        assert time.monotonic() - t0 > 0.3
    p.wait(10)
    a, b = HeldLock(tmp_path / "o.lock"), HeldLock(tmp_path / "o.lock")
    assert a.try_acquire() and a.held and not b.try_acquire()
    a.release()
    assert b.try_acquire()
    b.release()
    with file_lock(lock, shared=True, timeout_s=1), file_lock(lock, shared=True, timeout_s=1):
        pass  # shared locks coexist (separate descriptors)


# ---- provisioning adapter ------------------------------------------------------------------------------
def test_provisioning_adapter(tmp_path: Path) -> None:
    from pgw.sdk.provisioning import Provisioned, adapt

    fp = "ab" * 32
    p = adapt({"fingerprint": fp.upper(), "pack_dir": str(tmp_path), "hint_store": SimpleNamespace(root=tmp_path)},
              model="m")
    assert p == Provisioned(name="m", fingerprint=fp, pack_path=tmp_path, hint_dir=tmp_path)
    p2 = adapt(SimpleNamespace(fingerprint=fp, pack_path=tmp_path, hint_dir=None, hint_key_file=tmp_path / "k",
                               name="x", source="provider"), model="m")
    assert p2.name == "x" and p2.hint_key_file == tmp_path / "k" and p2.hint_dir is None
    for bad in ({"pack_path": tmp_path}, {"fingerprint": fp}, {"fingerprint": "zz", "pack_path": tmp_path}):
        with pytest.raises(PGWError):
            adapt(bad, model="m")


def _fake_session(tmp_path: Path, entry: dict[str, Any]) -> Any:
    from pydantic import SecretStr

    return SimpleNamespace(service_url="https://gpu.example.com", token=lambda: SecretStr("tok"), ca_file=None,
                           client_cert_file=None, client_key_file=None,
                           model_entry=lambda m: entry if m == entry["name"] else None)


def _fake_provision(calls: list[Any], tmp_path: Path) -> Any:
    def provision(model: str, **kw: Any) -> Any:
        calls.append((model, {k: v for k, v in kw.items() if k not in ("session", "progress")}))
        fp = kw["expected_fingerprint"]
        return SimpleNamespace(fingerprint=fp, pack_dir=tmp_path / "pack", hint_store=None, hint_root=None,
                               hint_key_file=None, factory_pack_dir=tmp_path / "factory-pack", close=lambda: None)

    return provision


def test_provider_provisioner_pins_and_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The default provisioner is the real pgw.provision.client.provision, with the library's fingerprint pins,
    hint="local" (the factory builds the hint) and a factory pack."""
    import pgw.provision.client as pc
    from pgw.core.errors import FingerprintMismatchError
    from pgw.sdk.options import SDKOptions
    from pgw.sdk.provisioning import PINNED_MODELS, ProviderProvisioner, default_provisioner

    assert set(PINNED_MODELS) == {"qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct",
                                  "qwen2.5-7b-instruct"}
    calls: list[Any] = []
    monkeypatch.setattr(pc, "provision", _fake_provision(calls, tmp_path))
    pin = PINNED_MODELS["qwen2.5-0.5b-instruct"]
    entry = {"name": "qwen2.5-0.5b-instruct", "fingerprint": pin.fingerprint}
    prov = default_provisioner(SDKOptions())
    assert isinstance(prov, ProviderProvisioner) and prov.hint == "local"
    p = prov("qwen2.5-0.5b-instruct", session=_fake_session(tmp_path, entry), cache_dir=tmp_path / "models",
             masked_head=True)
    assert p.fingerprint == pin.fingerprint and p.factory_pack_path == tmp_path / "factory-pack" and p.hint_dir is None
    (model, kw), = calls
    assert kw == {"cache_dir": tmp_path, "expected_fingerprint": pin.fingerprint, "masked_head": True,
                  "factory_pack": True, "hint": "none"}
    # opt-in provider-built hint
    calls.clear()
    ProviderProvisioner(hint="provider")("qwen2.5-0.5b-instruct", session=_fake_session(tmp_path, entry),
                                         cache_dir=tmp_path / "models")
    assert calls[0][1]["hint"] == "provider"
    # a provider serving another pack under a pinned name is refused before any download
    calls.clear()
    bad = {"name": "qwen2.5-0.5b-instruct", "fingerprint": "ab" * 32}
    with pytest.raises(FingerprintMismatchError, match="pins"):
        prov("qwen2.5-0.5b-instruct", session=_fake_session(tmp_path, bad), cache_dir=tmp_path)
    assert calls == []
    # an explicit pin (custom pack) is honoured
    p = ProviderProvisioner(fingerprints={"qwen2.5-0.5b-instruct": "ab" * 32})(
        "qwen2.5-0.5b-instruct", session=_fake_session(tmp_path, bad), cache_dir=tmp_path)
    assert p.fingerprint == "ab" * 32
    with pytest.raises(ValueError, match="64 hex"):
        SDKOptions(model_fingerprints={"x": "nope"}).validate()


#: The release's pack fingerprints (scripts/pin_model_fingerprints.py; 3B / 7B derived on a Kaggle CPU kernel,
#: results/kaggle/sdk-7b-validation/pins, where 0.5B / 1.5B re-derived to the integration box's values).
RELEASE_FINGERPRINTS = {
    "qwen2.5-0.5b-instruct": "915e55c19364542c4fd8ab5c8f7154806e2511048755e0d10426374b5c8fe9c7",
    "qwen2.5-1.5b-instruct": "81bb7adcea1524171297c0004c0efb1edad9f71dd0535462c9b32ab03ac99348",
    "qwen2.5-3b-instruct": "a8e7bac934d82f6435fe9385bd1eb3cfdf1a68f4ff415829e49a3dd4f23bcd23",
    "qwen2.5-7b-instruct": "d3db0ca858981a66dd262143d7f3d5b0120069068e0a98a806e1eea3f12653fa",
}


def test_every_release_model_has_a_baked_fingerprint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every supported model is pinned by its pack fingerprint (no first-use re-derivation from the checkpoint), the
    pins use the provider image's checkpoint commits, and a provider serving another pack is refused at once."""
    import pgw.provision.client as pc
    from pgw.core.errors import FingerprintMismatchError
    from pgw.sdk.options import SUPPORTED_MODELS
    from pgw.sdk.provisioning import PINNED_MODELS, ProviderProvisioner

    assert {m: p.fingerprint for m, p in PINNED_MODELS.items()} == RELEASE_FINGERPRINTS
    assert set(SUPPORTED_MODELS) == set(PINNED_MODELS)
    pod_boot = Path(__file__).resolve().parents[2] / "deploy" / "runpod" / "pod_boot.py"
    if pod_boot.is_file():
        src = pod_boot.read_text()
        for p in PINNED_MODELS.values():
            assert f'("{p.hf_repo}", "{p.revision}")' in src
    calls: list[Any] = []
    monkeypatch.setattr(pc, "provision", _fake_provision(calls, tmp_path))
    prov = ProviderProvisioner()
    for m in ("qwen2.5-3b-instruct", "qwen2.5-7b-instruct"):
        entry = {"name": m, "fingerprint": RELEASE_FINGERPRINTS[m]}
        prov(m, session=_fake_session(tmp_path, entry), cache_dir=tmp_path)
        assert "source" not in calls[-1][1] and calls[-1][1]["expected_fingerprint"] == RELEASE_FINGERPRINTS[m]
        n = len(calls)
        with pytest.raises(FingerprintMismatchError, match="pins"):  # refused before any download
            prov(m, session=_fake_session(tmp_path, {**entry, "fingerprint": "ab" * 32}), cache_dir=tmp_path)
        assert len(calls) == n


def test_provider_provisioner_public_check_for_unbaked_pins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pin without a baked fingerprint (a future model; every release model has one): the first provisioning
    re-derives the pack from the public checkpoint (source="hf") and the result is remembered; a provider packing
    another revision is refused."""
    import pgw.provision.client as pc
    from pgw.core.errors import FingerprintMismatchError
    from pgw.sdk.provisioning import PINNED_MODELS, ModelPin, ProviderProvisioner

    calls: list[Any] = []
    monkeypatch.setattr(pc, "provision", _fake_provision(calls, tmp_path))
    pin = ModelPin(PINNED_MODELS["qwen2.5-7b-instruct"].hf_repo, PINNED_MODELS["qwen2.5-7b-instruct"].revision)
    assert pin.fingerprint is None
    entry = {"name": "qwen2.5-7b-instruct", "fingerprint": "cd" * 32, "hf_repo": pin.hf_repo, "revision": pin.revision}
    prov = ProviderProvisioner(pins={"qwen2.5-7b-instruct": pin})
    prov("qwen2.5-7b-instruct", session=_fake_session(tmp_path, entry), cache_dir=tmp_path)
    kw = calls[0][1]
    assert kw["source"] == "hf" and kw["hf_repo"] == pin.hf_repo and kw["hf_revision"] == pin.revision
    assert kw["expected_fingerprint"] == "cd" * 32
    calls.clear()
    prov("qwen2.5-7b-instruct", session=_fake_session(tmp_path, entry), cache_dir=tmp_path)
    assert "source" not in calls[0][1]  # checked once, remembered in the cache
    other = {**entry, "revision": "0" * 40}
    with pytest.raises(FingerprintMismatchError, match="revision"):
        prov("qwen2.5-7b-instruct", session=_fake_session(tmp_path, other), cache_dir=tmp_path)


def test_local_pack_provisioner_checks_the_fingerprint(sdk_pack_dir: Path, tmp_path: Path) -> None:
    from pgw.core.errors import FingerprintMismatchError
    from pgw.modelpack import ModelPack
    from pgw.sdk.provisioning import LocalPackProvisioner

    fp = ModelPack.open(sdk_pack_dir).fingerprint
    session = SimpleNamespace(model_entry=lambda m: {"name": m, "fingerprint": fp})
    prov = LocalPackProvisioner({"qwen2.5-0.5b-instruct": sdk_pack_dir})
    p = prov("qwen2.5-0.5b-instruct", session=session, cache_dir=tmp_path)
    assert p.fingerprint == fp and p.source == "local"
    bad = SimpleNamespace(model_entry=lambda m: {"name": m, "fingerprint": "0" * 64})
    with pytest.raises(FingerprintMismatchError):
        prov("qwen2.5-0.5b-instruct", session=bad, cache_dir=tmp_path)
    with pytest.raises(PGWError, match="no local pack"):
        prov("qwen2.5-7b-instruct", session=session, cache_dir=tmp_path)


# ---- factory environment ---------------------------------------------------------------------------
def _plan(tmp_path: Path, scheme: str) -> Any:
    from pgw.sdk.factory import FactoryPlan

    return FactoryPlan(model="m", pack_path=tmp_path / "pack", ring="K32A12C27", state_dir=tmp_path / "f",
                       store_root=tmp_path / "masks", kek_file=tmp_path / "kek", counter_file=tmp_path / "ctr",
                       scheme=scheme, check_field="p31", masked_head=True, workers=2, threads=1, nice=10,
                       target_slots=1000, low_water_slots=500, segment_slots=256, first_segment_slots=64,
                       service_url="https://gpu.example.com", token_file=tmp_path / "token", ca_file=tmp_path / "ca",
                       client_cert_file=tmp_path / "crt", client_key_file=tmp_path / "key",
                       hint_dir=tmp_path / "hint", hint_key_file=tmp_path / "hk")


def test_factory_env_pks_v3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.factory.config import FactorySettings
    from pgw.hemask.pks import PKS_V3_ACK
    from pgw.sdk.factory import _child_env, factory_env

    env = factory_env(_plan(tmp_path, "pks-v3"))
    assert env["PGW_FACTORY_HE_SCHEME"] == "pks-v3" and env["PGW_FACTORY_HE_PKS_ACK"] == PKS_V3_ACK
    assert env["PGW_FACTORY_NICE"] == "10" and env["PGW_FACTORY_WORKERS"] == "2"
    assert env["PGW_FACTORY_CHECK_FIELD"] == "p31" and env["PGW_FACTORY_MASKED_HEAD"] == "true"
    assert env["PGW_MASKSTORE_COUNTER"] == "file" and env["PGW_FACTORY_DEVICE"] == "cpu"
    for k, v in env.items():  # paths and settings only: never a token value
        assert "secret" not in k.lower() or v.startswith("/")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = FactorySettings()  # the child's settings validate (acknowledgement accepted)
    assert s.he_scheme == "pks-v3" and s.effective_check_field == "p31" and s.workers == 2
    monkeypatch.setenv("PGW_GATEWAY_ADMIN_TOKEN", "not-for-the-child")
    monkeypatch.setenv("PGW_API_KEY", "pgw_live_not_for_the_child")
    child = _child_env(_plan(tmp_path, "pks-v3"))
    assert "PGW_GATEWAY_ADMIN_TOKEN" not in child and "PGW_API_KEY" not in child
    assert child["PGW_SDK_PARENT_PID"] == str(os.getpid())


def test_factory_env_plain_has_no_provider(tmp_path: Path) -> None:
    from pgw.sdk.factory import factory_env

    env = factory_env(_plan(tmp_path, "plain"))
    assert not any(k.startswith("PGW_FACTORY_HE_") for k in env)


def test_factory_supervisor_backs_off_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A child that keeps failing is restarted with back-off and its error line is reported."""
    from pgw.sdk import factory as F

    plan = _plan(tmp_path, "plain")
    fp = F.FactoryProcess(plan, poll_s=0.05)
    spawned: list[float] = []

    class Dead:
        returncode = 1
        pid = 1

        def poll(self) -> int:
            return 1

    def spawn() -> None:
        spawned.append(time.monotonic())
        with fp.log_file.open("a") as fh:
            fh.write('{"level": "INFO"}\nerror [config]: something is wrong\n')
        fp._proc = Dead()  # type: ignore[assignment]
        fp.starts += 1

    monkeypatch.setattr(fp, "_spawn", spawn)
    fp.start()
    deadline = time.monotonic() + 5
    while fp.failures < 4 and time.monotonic() < deadline:
        time.sleep(0.02)
    fp.stop()
    assert fp.failures >= 3 and fp.last_error == "error [config]: something is wrong"
    gaps = [b - a for a, b in itertools.pairwise(spawned)]
    assert gaps[-1] > gaps[0] * 2  # exponential back-off
    st = fp.state()
    assert st["consecutive_failures"] >= 3 and st["owner"] is False  # lock released by stop()


def test_factory_owner_lock_is_shared_between_supervisors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.sdk import factory as F

    a, b = F.FactoryProcess(_plan(tmp_path, "plain"), poll_s=0.05), F.FactoryProcess(_plan(tmp_path, "plain"),
                                                                                       poll_s=0.05)
    started: list[str] = []
    ev = threading.Event()

    class Alive:
        returncode = None
        pid = 1

        def poll(self) -> None:
            return None

        def send_signal(self, sig: int) -> None:
            ev.set()

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def spawner(name: str, fp: Any) -> Any:
        def spawn() -> None:
            started.append(name)
            fp._proc = Alive()

        return spawn

    monkeypatch.setattr(a, "_spawn", spawner("a", a))
    monkeypatch.setattr(b, "_spawn", spawner("b", b))
    a.start()
    b.start()
    time.sleep(0.2)
    assert started == ["a"] and a.owner and not b.owner
    a.stop()  # b takes over within a poll
    deadline = time.monotonic() + 3
    while "b" not in started and time.monotonic() < deadline:
        time.sleep(0.02)
    b.stop()
    assert started == ["a", "b"] and ev.is_set()


# ---- CLI -----------------------------------------------------------------------------------------------
def test_cli_status_and_doctor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from pgw.cli.main import cli

    runner = CliRunner()
    r = runner.invoke(cli, ["sdk", "status", "--cache-dir", str(tmp_path / "c"), "--state-dir", str(tmp_path / "s")])
    assert r.exit_code == 0 and "no enrolled API key" in r.output
    sess = tmp_path / "c" / "sessions" / "0123456789abcdef"
    sess.mkdir(parents=True)
    (sess / "enrollment.json").write_text(json.dumps({"service_url": "https://p", "tenant": "t", "enrolled_at": 1,
                                                      "models": [{"name": "qwen2.5-7b-instruct"}]}))
    (sess / "status.json").write_text(json.dumps({"models": {"qwen2.5-7b-instruct": {
        "state": "ready", "masks_available": 12, "lane": "open", "masked_head": True, "verified_requests": 3,
        "integrity_failures": 0, "last_integrity_check_ok_at": time.time() - 5}}}))
    (sess / "factory" / "abcd").mkdir(parents=True)
    (sess / "factory" / "abcd" / "factory-state.json").write_text(json.dumps({"provider_integrity_failures": 0}))
    r = runner.invoke(cli, ["sdk", "status", "--cache-dir", str(tmp_path / "c"), "--state-dir", str(tmp_path / "s")])
    assert r.exit_code == 0, r.output
    assert "qwen2.5-7b-instruct: ready" in r.output and "masks=12" in r.output and "3 verified" in r.output
    assert "factory abcd: halted=False" in r.output
    r = runner.invoke(cli, ["sdk", "status", "--json", "--no-masks", "--cache-dir", str(tmp_path / "c"),
                            "--state-dir", str(tmp_path / "s")])
    assert json.loads(r.output)["sessions"][0]["key_id"] == "0123456789abcdef"
    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [1])).encode()
    monkeypatch.setenv("PGW_API_KEY", key)
    r = runner.invoke(cli, ["sdk", "doctor", "--json", "--cache-dir", str(tmp_path / "c"),
                            "--state-dir", str(tmp_path / "s")])
    rep = json.loads(r.output)
    assert key not in r.output and any(c["check"] == "api key" and c["status"] == "ok" for c in rep["checks"])
    monkeypatch.setenv("PGW_API_KEY", key[:-6])
    r = runner.invoke(cli, ["sdk", "doctor", "--cache-dir", str(tmp_path / "c"), "--state-dir", str(tmp_path / "s")])
    assert r.exit_code == 1 and "[fail] api key" in r.output and key[:-6] not in r.output


# ---- per-request timeout, shielded start-up, request reports -------------------------------------------
def test_create_timeout_bounds_the_request_and_reports_without_content(tmp_path: Path) -> None:
    import asyncio as aio
    import time

    from pgw import OpenAI
    from pgw.sdk.client import _request_timeout

    assert _request_timeout({}) is None and _request_timeout({"timeout": 2}) == 2.0
    assert _request_timeout({"timeout": SimpleNamespace(connect=1.0, read=5.0, write=None, pool=None)}) == 5.0
    with pytest.raises(E.BadRequestError):
        _request_timeout({"timeout": 0})
    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [3])).encode()
    c = OpenAI(api_key=key, cache_dir=tmp_path / "c", state_dir=tmp_path / "s", timeout=300)
    eng = c._engine

    async def slow(req: Any, *, conversation: Any = None, meta: Any = None, timeout_s: Any = None) -> Any:
        meta["model"] = "qwen2.5-0.5b-instruct"
        meta["prompt_tokens"] = 5
        await aio.sleep(30)
        yield None, None

    eng._chat_events = slow
    msgs = [{"role": "user", "content": "secret prompt"}]
    t0 = time.monotonic()
    with pytest.raises(E.APITimeoutError):
        c.chat.completions.create(model="qwen2.5-0.5b-instruct", messages=msgs, timeout=0.5)
    with pytest.raises(E.APITimeoutError):
        list(c.chat.completions.create(model="qwen2.5-0.5b-instruct", messages=msgs, stream=True, timeout=0.5))
    assert time.monotonic() - t0 < 10
    deadline = time.monotonic() + 5
    while len(eng.reporter._q) < 2 and time.monotonic() < deadline:  # the cancelled stream reports asynchronously
        time.sleep(0.05)
    recs = list(eng.reporter._q)
    assert [r["status"] for r in recs] == ["error", "error"] and [r["stream"] for r in recs] == [False, True]
    assert recs[0]["error_code"] == "deadline_exceeded" and recs[1]["error_code"] in ("deadline_exceeded", "cancelled")
    assert all(r["checks_ok"] is False for r in recs)
    assert "secret" not in json.dumps(recs) and set(recs[0]) == {
        "ts", "model", "status", "error_code", "prompt_tokens", "completion_tokens", "latency_ms", "ttft_ms",
        "stream", "checks_ok"}
    eng.reporter.enabled = False  # nothing to send it to
    c.close()


def test_a_short_timeout_does_not_cancel_model_start_up(tmp_path: Path) -> None:
    import asyncio as aio

    from pgw.sdk.engine import Engine
    from pgw.sdk.options import SDKOptions

    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [4]))
    eng = Engine(key, SDKOptions(cache_dir=tmp_path / "c", state_dir=tmp_path / "s", usage_reports=False))
    starts: list[str] = []

    async def start(h: Any) -> None:
        starts.append(h.name)
        await aio.sleep(1.0)
        h.state = "ready"

    eng._start_model = start  # type: ignore[method-assign]

    async def run() -> str:
        with pytest.raises(TimeoutError):
            await aio.wait_for(eng.ensure_model("qwen2.5-0.5b-instruct"), timeout=0.1)  # the caller gives up
        h = await eng.ensure_model("qwen2.5-0.5b-instruct")  # the same start-up, not a new one
        return h.state

    try:
        assert eng.call(run(), timeout_s=30) == "ready" and starts == ["qwen2.5-0.5b-instruct"]
    finally:
        eng.close()


def test_fused2_tables_live_in_modelpack() -> None:
    """The service derives fused2 lane tables without the engine; the engine re-exports the same objects."""
    import pgw.engine.fused2 as ef
    import pgw.modelpack.fused2_weights as mf

    for n in ("F2Linear", "Fused2Weights", "migration_multipliers", "select_gateway_columns", "hf_float_linears"):
        assert getattr(ef, n) is getattr(mf, n)


# ---- mask stock under bursts --------------------------------------------------------------------------
def _mask_engine(tmp_path: Path, **opts: Any) -> Any:
    from pgw.sdk.engine import Engine, ModelHandle
    from pgw.sdk.options import SDKOptions

    key = SdkKey.new("live", "https://gpu.example.com", bytes(32 * [5]))
    eng = Engine(key, SDKOptions(cache_dir=tmp_path / "c", state_dir=tmp_path / "s", usage_reports=False, **opts))
    h = ModelHandle("qwen2.5-0.5b-instruct", state="ready")
    h.masks_seen = True
    h.factory = SimpleNamespace(owner=True, failures=0, last_error=None, state=lambda: {"halted": False},
                                plan=SimpleNamespace(target_slots=100, segment_slots=16))
    return eng, h


def test_requests_waiting_for_masks_are_served_in_arrival_order(tmp_path: Path) -> None:
    """A burst larger than the stock: a large request that waits first is served before the smaller ones that
    arrive later and would each fit the stock left (no starvation); every request is served in arrival order."""
    import asyncio as aio

    eng, h = _mask_engine(tmp_path, mask_wait_s=30.0)
    stock = {"n": 0}
    served: list[str] = []

    class _RT:
        state = "ready"

        async def open_sessions(self, prompts: Any, params: Any, **kw: Any) -> Any:
            need = len(prompts[0])
            if stock["n"] < need:
                raise MaskExhaustedError("no masks", details={"entitled": need})
            stock["n"] -= need
            served.append(kw["request_id"])
            return [SimpleNamespace()]

    rt = _RT()

    async def run() -> None:
        async def refill() -> None:
            for _ in range(60):
                await aio.sleep(0.02)
                stock["n"] += 1

        filler = aio.create_task(refill())
        dl = time.monotonic() + 30
        big = aio.create_task(eng._open(h, rt, [0] * 10, None, "big", None, dl))
        await aio.sleep(0.05)
        small = []
        for i in range(8):  # later, smaller requests, one at a time
            small.append(aio.create_task(eng._open(h, rt, [0], None, f"small{i}", None, dl)))
            await aio.sleep(0.03)
        await aio.gather(big, *small)
        await filler

    try:
        eng.call(run(), timeout_s=60)
    finally:
        eng.close()
    assert served == ["big"] + [f"small{i}" for i in range(8)], served
    assert h.mask_waiters == []


def test_a_request_larger_than_the_stock_fails_at_once(tmp_path: Path) -> None:
    eng, h = _mask_engine(tmp_path, mask_wait_s=30.0)

    class _RT:
        state = "ready"

        async def open_sessions(self, prompts: Any, params: Any, **kw: Any) -> Any:
            raise MaskExhaustedError("no masks", details={"entitled": 500})

    t0 = time.monotonic()
    try:
        with pytest.raises(E.BadRequestError, match="mask_store_gb"):
            eng.call(eng._open(h, _RT(), [0], None, "r", None, time.monotonic() + 30), timeout_s=30)
    finally:
        eng.close()
    assert time.monotonic() - t0 < 5 and h.mask_waiters == []
