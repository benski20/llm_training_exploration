"""End to end on CPU: the client library against a real ``pgw service`` process on loopback.

The provider runs the real service (TLS, TCP+TLS trip lane, masked LM head, accounts with SDK enrollment and the
admin API); the client enrolls with an SDK key created through the admin API, provisions from the local pack, runs the in-process gateway
with the default stack (native trip / decode drivers, p31 checks, masked head, incremental reservations, prefix
cache) and its own background mask factory, and must give bit-identical answers to the existing gateway path
(the plaintext reference of ``pgw gateway`` with the same model settings) for the same seed and prompt.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.sdk.conftest import MODEL, Provider

pytestmark = pytest.mark.timeout(1200)

PROMPT = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello there, who are you?"}]
GREEDY = {"temperature": 0, "max_tokens": 10, "logprobs": True, "top_logprobs": 3, "extra_body": {"ignore_eos": True}}
SAMPLED = {"temperature": 0.9, "top_p": 0.95, "seed": 1234, "max_tokens": 10, "logprobs": True, "top_logprobs": 2,
           "extra_body": {"ignore_eos": True}}


def _client(provider: Provider, key: str, root: Path, **kw: Any) -> Any:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {"cache_dir": root / "cache", "state_dir": root / "state", "mask_scheme": "plain",
                            "mask_workers": 1, "mask_target_slots": 400, "mask_low_water_slots": 200,
                            "mask_segment_slots": 128, "timeout": 300, "ready_timeout_s": 600}
    opts.update(kw)
    return OpenAI(api_key=key, provisioner=LocalPackProvisioner({MODEL: provider.pack_dir}), **opts)


def _reference(pack_dir: Path, body: dict[str, Any], head_mode: str = "provider_int8",
               messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The existing gateway (``pgw gateway`` app, plaintext reference mode) with the client's model settings."""
    from pgw.gateway import ApiKeyStore, GatewayModelConfig, GatewaySettings, create_app

    keys = ApiKeyStore(None)
    key = keys.create("ref")[1]
    model = GatewayModelConfig(name=MODEL, pack_path=pack_dir, mode="plain", head_mode=head_mode,
                               float_ops="native")
    settings = GatewaySettings(models=[model], admin_token="ref-admin-token-0123456789", allow_dev_modes=True,
                               default_max_tokens=8)
    app = create_app(settings, key_store=keys)
    req = {k: v for k, v in body.items() if k != "extra_body"}
    req.update(body.get("extra_body") or {})

    async def run() -> dict[str, Any]:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ref.test",
                                         timeout=300) as c:
                r = await c.post("/v1/chat/completions", json={"model": MODEL, "messages": messages or PROMPT, **req},
                                 headers={"Authorization": f"Bearer {key}"})
                assert r.status_code == 200, r.text
                return dict(r.json())

    return asyncio.run(run())


@pytest.fixture(scope="module")
def state(provider: Provider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("sdk-client")
    key = provider.new_key()
    client = _client(provider, key, root)
    st: dict[str, Any] = {"root": root, "key": key, "client": client}
    try:
        yield st
    finally:
        st["client"].close()


def _greedy(state: dict[str, Any]) -> Any:
    """The client's greedy answer to PROMPT (the module's first request). Computed here when the test that makes
    it ran elsewhere (pytest-xdist spreads a module's tests over workers, each with its own provider and cache):
    every test that depends on it is self-sufficient."""
    if "greedy" not in state:
        state["greedy"] = state["client"].chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
    return state["greedy"]


def _lp(choice: Any) -> list[tuple[str, float, list[tuple[str, float]]]]:
    return [(c.token, c.logprob, [(t.token, t.logprob) for t in c.top_logprobs]) for c in choice.logprobs.content]


def test_models_list(state: dict[str, Any]) -> None:
    ids = [m.id for m in state["client"].models.list()]
    assert ids == [MODEL]


def test_chat_completion_is_bit_identical_to_the_gateway_path(state: dict[str, Any], provider: Provider) -> None:
    c = state["client"]
    t0 = time.monotonic()
    r = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
    state["first_s"] = time.monotonic() - t0
    assert r.object == "chat.completion" and r.model == MODEL
    assert r.choices[0].finish_reason == "length"
    assert r.usage is not None and r.usage.completion_tokens == 10
    ref = _reference(provider.pack_dir, GREEDY)
    rc = ref["choices"][0]
    assert r.choices[0].message.content == rc["message"]["content"]
    assert r.to_dict()["choices"][0]["logprobs"]["content"] == rc["logprobs"]["content"]  # bit for bit, top-3 alternatives included
    st = c.status()["models"][MODEL]
    assert st["state"] == "ready" and st["masked_head"] is True
    assert st["lane"] == "open", st  # the provider's TCP+TLS trip lane
    assert st["native"] == {"trip_engine": True, "trip_native": True, "decode_native": True, "prime_check": True}
    assert st["verified_requests"] >= 1 and st["integrity_failures"] == 0
    state["greedy"] = r


def test_seeded_sampling_and_streaming_are_bit_identical(state: dict[str, Any], provider: Provider) -> None:
    c = state["client"]
    r = c.chat.completions.create(model=MODEL, messages=PROMPT, **SAMPLED)
    ref = _reference(provider.pack_dir, SAMPLED)
    assert r.choices[0].message.content == ref["choices"][0]["message"]["content"]
    assert r.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
    chunks = list(c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True,
                                            stream_options={"include_usage": True}, **SAMPLED))
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
    assert text == r.choices[0].message.content
    lps = [t for ch in chunks if ch.choices and ch.choices[0].logprobs for t in _lp(ch.choices[0])]
    assert lps == _lp(r.choices[0])
    assert chunks[-1].choices == [] and chunks[-1].usage is not None
    assert [ch.choices[0].finish_reason for ch in chunks if ch.choices][-1] == "length"


def test_conversation_prefix_reuse_is_exact(state: dict[str, Any], provider: Provider) -> None:
    """prefix_cache=True (the gateway's own LM head): turn 2 reuses turn 1's KV cache and gives the bits of a
    recompute and of the existing gateway path."""
    c = _client(provider, state["key"], state["root"] / "prefix", prefix_cache=True)
    try:
        kw = {**GREEDY, "extra_body": {"ignore_eos": True, "conversation_id": "conv-1"}}
        r1 = c.chat.completions.create(model=MODEL, messages=PROMPT, **kw)
        st = c.status()["models"][MODEL]
        assert st["masked_head"] is False and st["native"]["decode_native"] is True
        ref = _reference(provider.pack_dir, GREEDY, head_mode="gateway")
        assert r1.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
        turn2 = [*PROMPT, {"role": "assistant", "content": r1.choices[0].message.content},
                 {"role": "user", "content": "And then?"}]
        r2 = c.chat.completions.create(model=MODEL, messages=turn2, **kw)
        assert r2.usage.prompt_tokens_details is not None and r2.usage.prompt_tokens_details.cached_tokens > 0
        fresh = c.chat.completions.create(model=MODEL, messages=turn2, **{**GREEDY, "extra_body": {
            "ignore_eos": True, "conversation_id": "conv-other"}})
        assert fresh.usage.prompt_tokens_details.cached_tokens == 0
        assert r2.to_dict()["choices"][0] == fresh.to_dict()["choices"][0]  # reuse gives the bits of a recompute
    finally:
        c.close()


def test_async_client(state: dict[str, Any], provider: Provider) -> None:
    from pgw import AsyncOpenAI
    from pgw.sdk import LocalPackProvisioner

    root = state["root"]

    async def run() -> tuple[Any, list[Any]]:
        async with AsyncOpenAI(api_key=state["key"], cache_dir=root / "cache", state_dir=root / "state",
                               mask_scheme="plain", mask_workers=1, mask_target_slots=400,
                               provisioner=LocalPackProvisioner({MODEL: provider.pack_dir})) as ac:
            r = await ac.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
            s = await ac.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, **GREEDY)
            chunks = [ch async for ch in s]
            return r, chunks

    r, chunks = asyncio.run(run())
    assert r.to_dict()["choices"][0] == _greedy(state).to_dict()["choices"][0]
    assert "".join(ch.choices[0].delta.content or "" for ch in chunks) == _greedy(state).choices[0].message.content


def test_concurrent_requests_from_threads(state: dict[str, Any]) -> None:
    from concurrent.futures import ThreadPoolExecutor

    c = state["client"]
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY), range(4)))
    want = _greedy(state).to_dict()["choices"][0]
    assert all(o.to_dict()["choices"][0] == want for o in outs)


def test_status_cli_shows_the_session(state: dict[str, Any]) -> None:
    from pgw.sdk.cli import status_report

    _greedy(state)
    rep = status_report(state["root"] / "cache", state["root"] / "state")
    (s,) = rep["sessions"]
    m = s["status"]["models"][MODEL]
    assert m["state"] == "ready" and m["last_integrity_check_ok_at"] is not None
    assert s["enrollment"]["service_url"].startswith("https://127.0.0.1")
    assert "mask_stock" in s and "error" not in s["mask_stock"], s.get("mask_stock")
    text = json.dumps(rep)
    assert state["key"] not in text  # never the key
    token = (state["root"] / "cache" / "sessions" / s["key_id"] / "secrets" / "service_token").read_text().strip()
    assert token not in text  # never the service token


def test_restart_reuses_the_cache(state: dict[str, Any], provider: Provider) -> None:
    root = state["root"]
    want = _greedy(state).to_dict()["choices"][0]
    state["client"].close()
    sess = next((root / "cache" / "sessions").iterdir())
    meta0 = json.loads((sess / "enrollment.json").read_text())
    tokens0 = (sess / "secrets" / "service_token").read_text()
    c = _client(provider, state["key"], root)
    state["client"] = c
    t0 = time.monotonic()
    r = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
    restart_s = time.monotonic() - t0
    assert r.to_dict()["choices"][0] == want
    meta1 = json.loads((sess / "enrollment.json").read_text())
    assert meta1["enrolled_at"] == meta0["enrolled_at"]  # no second enrollment
    assert (sess / "secrets" / "service_token").read_text() == tokens0
    assert c.status()["models"][MODEL]["masks_available"] is not None
    print(f"\nfirst request (enroll + provision + connect + first masks) {state.get('first_s', 0):.1f}s, "
          f"after restart {restart_s:.1f}s")


def test_integrity_failure_raises_and_blocks(provider: Provider, tmp_path: Path) -> None:
    """Self-contained (its own key, cache and client): a tampering provider gives IntegrityError, the model stays
    blocked after the provider recovers and after a restart (persisted halt, review S11), until cleared."""
    from click.testing import CliRunner

    from pgw.cli.main import cli
    from pgw.sdk import IntegrityError

    key = provider.new_key()
    c = _client(provider, key, tmp_path)
    try:
        good = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY).to_dict()["choices"][0]
        provider.tamper_file.touch()
        try:
            with pytest.raises(IntegrityError):
                c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
            with pytest.raises(IntegrityError):  # blocked: never silently retried into a success
                list(c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, **GREEDY))
        finally:
            provider.tamper_file.unlink()
        with pytest.raises(IntegrityError):  # still blocked after the provider "recovered"
            c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        st = c.status()["models"][MODEL]
        assert st["integrity_failures"] >= 1 and st["last_integrity_failure_at"] is not None
        assert st["integrity_halt"]["halted"]
    finally:
        c.close()
    c = _client(provider, key, tmp_path)  # a restart does not re-open the model
    try:
        with pytest.raises(IntegrityError, match="clear-halt"):
            c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
    finally:
        c.close()
    r = CliRunner().invoke(cli, ["sdk", "clear-halt", "--model", MODEL, "--yes", "--state-dir", str(tmp_path / "state")])
    assert r.exit_code == 0 and "cleared" in r.output, r.output
    c = _client(provider, key, tmp_path)
    try:
        assert c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY).to_dict()["choices"][0] == good
        provider.tamper_file.touch()
        try:
            with pytest.raises(IntegrityError):
                c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        finally:
            provider.tamper_file.unlink()
        c.reset_integrity_breaker(MODEL)  # the in-process reset clears the persisted halt too
        assert c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY).to_dict()["choices"][0] == good
    finally:
        c.close()


def test_revoked_and_unknown_keys_are_refused(provider: Provider, tmp_path: Path) -> None:
    from pgw.sdk import AuthenticationError
    from pgw.sdk.keys import SdkKey

    unknown = SdkKey.new("test", provider.url, provider.ca_sha256).encode()
    c = _client(provider, unknown, tmp_path / "u")
    with pytest.raises(AuthenticationError):
        c.models.list()
    c.close()
    key = provider.new_key()
    provider.revoke(key)
    c = _client(provider, key, tmp_path / "r")
    with pytest.raises(AuthenticationError):
        c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
    c.close()


def test_wrong_ca_pin_is_refused_before_anything_is_sent(provider: Provider, tmp_path: Path) -> None:
    from pgw.sdk import IntegrityError
    from pgw.sdk.keys import SdkKey

    bad = SdkKey.new("test", provider.url, b"\x42" * 32).encode()
    c = _client(provider, bad, tmp_path / "pin")
    with pytest.raises(IntegrityError, match="does not match"):
        c.models.list()
    c.close()
    assert not (tmp_path / "pin" / "cache" / "sessions").exists() or not any(
        (tmp_path / "pin" / "cache" / "sessions").glob("*/secrets/service_token"))


def test_pks_v3_factory_default_scheme(provider: Provider, tmp_path: Path) -> None:
    """The default mask scheme: pks-v3 masks from the provider (encrypted; the provider learns nothing about them),
    self-checked by the factory, p31 segments; the answer is bit-identical to the plaintext reference."""
    key = provider.new_key()
    # masked_head off: the tiny model's 260-token head is not a pks-v3 shape (out_features % 32); real Qwen2.5
    # vocabularies (151936) are
    c = _client(provider, key, tmp_path, mask_scheme="pks-v3", mask_target_slots=200, mask_low_water_slots=100,
                masked_head="off")
    try:
        t0 = time.monotonic()
        r = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        dt_s = time.monotonic() - t0
        ref = _reference(provider.pack_dir, GREEDY, head_mode="gateway")
        assert r.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
        st = c.status()["models"][MODEL]
        assert st["factory"]["scheme"] == "pks-v3" and st["factory"]["owner"] is True
        assert not st["factory"]["halted"] and not st["factory"].get("provider_integrity_failures")
        hint_dirs = list((tmp_path / "cache" / "models").glob("*/hint/*pks-v3*"))
        assert hint_dirs, "the pks-v3 hint was built locally"
        print(f"\npks-v3: first answer after {dt_s:.1f}s (hint build + first encrypted segment)")
    finally:
        c.close()


def test_a_failing_factory_is_reported_not_waited_out(provider: Provider, tmp_path: Path) -> None:
    """pks-v3 with the masked head on the tiny model: the factory refuses the head's shape. The request fails with
    NotReadyError naming the factory's error after a few restarts instead of hanging for mask_wait_s."""
    from pgw.sdk import NotReadyError

    c = _client(provider, provider.new_key(), tmp_path, mask_scheme="pks-v3", mask_wait_s=600)
    try:
        t0 = time.monotonic()
        with pytest.raises(NotReadyError, match=r"mask factory is failing: .*pks-v3 needs"):
            c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        assert time.monotonic() - t0 < 300
        f = c.status()["models"][MODEL]["factory"]
        assert f["consecutive_failures"] >= 3 and "pks-v3" in f["last_child_error"]
    finally:
        c.close()


def test_back_pressure_without_masks(provider: Provider, tmp_path: Path) -> None:
    """No factory and an empty stock: the request waits mask_wait_s for masks, then fails as not ready (nothing was
    sent to the provider)."""
    from pgw.sdk import NotReadyError

    c = _client(provider, provider.new_key(), tmp_path, factory=False, mask_wait_s=1.5)
    try:
        t0 = time.monotonic()
        with pytest.raises(NotReadyError, match="factory is off"):
            c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        assert 1.0 < time.monotonic() - t0 < 60
    finally:
        c.close()


_MP_CODE = """
import json, sys
sys.path[:0] = [sys.argv[4], sys.argv[5]]
from pgw import OpenAI
from pgw.sdk import LocalPackProvisioner
root = sys.argv[2]
c = OpenAI(api_key=sys.argv[1], cache_dir=root + "/cache", state_dir=root + "/state", mask_scheme="plain",
           mask_workers=1, mask_target_slots=400, mask_low_water_slots=200, mask_segment_slots=128,
           provisioner=LocalPackProvisioner({"qwen2.5-0.5b-instruct": sys.argv[3]}))
r = c.chat.completions.create(model="qwen2.5-0.5b-instruct", messages=[{"role": "user", "content": "Hi"}],
                              temperature=0, max_tokens=6, logprobs=True, extra_body={"ignore_eos": True})
f = c.status()["models"]["qwen2.5-0.5b-instruct"]["factory"]
c.close()
print(json.dumps({"lp": r.to_dict()["choices"][0]["logprobs"]["content"], "owner": f["owner"]}))
"""


def test_processes_share_session_store_and_factory(provider: Provider, tmp_path: Path) -> None:
    """Two processes with the same key and cache directory: one enrollment, one mask store, one factory owner."""
    import os
    import subprocess
    import sys

    from tests.sdk.conftest import ROOT

    key = provider.new_key("mp-test")
    args = [sys.executable, "-c", _MP_CODE, key, str(tmp_path), str(provider.pack_dir), str(ROOT / "src"), str(ROOT)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    procs = [subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)  # noqa: S603
             for _ in range(2)]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, err[-3000:]
        assert err == "", err[-3000:]  # silent library
        outs.append(json.loads(out))
    assert outs[0]["lp"] == outs[1]["lp"]
    assert sorted(o["owner"] for o in outs) == [False, True] or all(o["owner"] for o in outs)  # never two at once
    sessions = list((tmp_path / "cache" / "sessions").iterdir())
    assert len(sessions) == 1
    lines = (Path(provider.log.parent) / "secrets" / "tokens").read_text().splitlines()
    assert sum(ln.startswith(provider.tenant(key) + ":") for ln in lines) == 1  # enrolled once for both processes
    meta = json.loads((sessions[0] / "enrollment.json").read_text())
    assert meta["key_id"] == sessions[0].name
