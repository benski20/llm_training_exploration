"""The client library's ``schedule = "fused"`` end to end on CPU (opt-in, default off).

A real ``pgw service`` process (TLS, TCP+TLS trip lane) that also serves the head-fold fused2 lane entry derived from
its own pack (``PGW_SERVICE_FUSED2_FROM_PACK=8``, ``PGW_SERVICE_FUSED2_HEAD=on``) and fused2 maskgen
(``PGW_SERVICE_FUSED2_MASKGEN=1``); the client enrolls, sees the fused schedule offered, and runs fused2 with the
LM head folded into the last trip and layer 0's qkv local (2L trips per pass) with its own background factory
writing K56 pair-mask segments (plain, and the encrypted pks-v3 top-digit plan). Answers must be bit-identical to
the fused plaintext reference of the gateway (``mode = "plain"``, the same schedule): exactness. Also: trips per
token, the provider sees only masked rows (no row twice), every trip checked, tampering caught (fail closed),
the schedule fixed for the client's life, ``auto``'s decision recorded, refusals.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.sdk.conftest import MODEL, Provider, start_provider, stop_provider

pytestmark = pytest.mark.timeout(1800)

fast = pytest.importorskip("pgw_fast")
if not hasattr(fast, "TripChannel"):
    pytest.skip("pgw_fast without the lane client", allow_module_level=True)

PROMPT = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello there, who are you?"}]
GREEDY = {"temperature": 0, "max_tokens": 8, "logprobs": True, "top_logprobs": 3, "extra_body": {"ignore_eos": True}}
SAMPLED = {"temperature": 0.9, "top_p": 0.95, "seed": 99, "max_tokens": 8, "logprobs": True, "top_logprobs": 2,
           "extra_body": {"ignore_eos": True}}
FUSED_ENV = {"PGW_SERVICE_FUSED2_FROM_PACK": "8", "PGW_SERVICE_FUSED2_HEAD": "on", "PGW_SERVICE_FUSED2_MASKGEN": "1"}


@pytest.fixture(scope="module")
def fprovider(sdk_pack_dir: Path, pki: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Iterator[Provider]:
    p = start_provider(tmp_path_factory.mktemp("sdk-fused-provider"), sdk_pack_dir, pki, extra_env=FUSED_ENV)
    try:
        yield p
    finally:
        stop_provider(p)


def _client(provider: Provider, key: str, root: Path, **kw: Any) -> Any:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {"cache_dir": root / "cache", "state_dir": root / "state", "mask_scheme": "plain",
                            "mask_workers": 1, "mask_target_slots": 300, "mask_low_water_slots": 150,
                            "mask_segment_slots": 64, "timeout": 600, "ready_timeout_s": 900, "schedule": "fused"}
    opts.update(kw)
    return OpenAI(api_key=key, provisioner=LocalPackProvisioner({MODEL: provider.pack_dir}), **opts)


def fused_reference(pack_dir: Path, body: dict[str, Any], messages: list[dict[str, Any]] | None = None
                    ) -> dict[str, Any]:
    """The gateway's fused plaintext reference (``mode = "plain"``: exact integers, no provider, no masks)."""
    from pgw.gateway import ApiKeyStore, GatewayModelConfig, GatewaySettings, create_app

    keys = ApiKeyStore(None)
    key = keys.create("ref")[1]
    model = GatewayModelConfig(name=MODEL, pack_path=pack_dir, mode="plain", schedule="fused2", ring="K56A12C53",
                               head_mode="provider_int8", fused2_local_qkv0=True, float_ops="native")
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


def _lpc(r: Any) -> Any:
    return r.to_dict()["choices"][0]["logprobs"]["content"]


@pytest.fixture(scope="module")
def plain_client(fprovider: Provider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    c = _client(fprovider, fprovider.new_key("fused-plain"), tmp_path_factory.mktemp("sdk-fused-client"))
    try:
        yield c
    finally:
        c.close()


def test_enrollment_offers_the_fused_schedule(plain_client: Any, fprovider: Provider) -> None:
    plain_client.models.list()
    s = plain_client._engine.session  # type: ignore[attr-defined]  # the session the client enrolled
    e = s.model_entry(MODEL)
    assert e is not None and e["schedules"] == ["standard", "fused"] and e["fused_ring"] == "K56A12C53"


def test_fused_answers_are_bit_identical_to_the_fused_reference(plain_client: Any, fprovider: Provider) -> None:
    r = plain_client.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
    ref = fused_reference(fprovider.pack_dir, GREEDY)
    assert r.choices[0].message.content == ref["choices"][0]["message"]["content"]
    assert _lpc(r) == ref["choices"][0]["logprobs"]["content"]  # bit for bit, top-3 alternatives included
    r2 = plain_client.chat.completions.create(model=MODEL, messages=PROMPT, **SAMPLED)
    ref2 = fused_reference(fprovider.pack_dir, SAMPLED)
    assert _lpc(r2) == ref2["choices"][0]["logprobs"]["content"]
    st = plain_client.status()["models"][MODEL]
    assert st["schedule"] == "fused" and st["ring"] == "K56A12C53", st
    assert st["schedule_decision"]["requested"] == "fused"
    assert st["verified_requests"] >= 2 and st["integrity_failures"] == 0
    f2 = st.get("fused2") or {}
    assert f2.get("head_fold") is True and f2.get("local_qkv0") is True, st


def test_fused_trips_per_token_and_checks(plain_client: Any) -> None:
    eng = plain_client._engine  # type: ignore[attr-defined]
    h = eng.models[MODEL]
    rt = h.runtime
    be = rt.backend
    prov = be.provider
    L = int(rt.model.num_layers)

    def run(n: int) -> tuple[int, int, int]:
        c0, t0, k0 = prov.calls, be.f2.trips, be.f2.checks
        r = plain_client.chat.completions.create(model=MODEL, messages=PROMPT, temperature=0, max_tokens=n,
                                                 extra_body={"ignore_eos": True})
        assert r.usage.completion_tokens == n
        return prov.calls - c0, be.f2.trips - t0, be.f2.checks - k0

    c2, _, _ = run(2)
    c6, t6, k6 = run(6)
    # each decode pass after the prompt pass: exactly 2L lane calls (layer 0's qkv local, the head folded into the
    # last trip); the prompt pass takes the same calls in both requests (its rows in lane multi frames)
    assert c6 - c2 == 2 * L * 4, (c2, c6, L)
    assert c2 - 2 * L <= 2 * L * 2, c2  # the prompt pass: at most 2 multi calls per trip at this prompt length
    # every trip piece is checked: two checks per pair piece, one per single piece; nothing unchecked
    assert k6 >= t6 >= 2 * L * 5
    st = plain_client.status()["models"][MODEL]
    assert st["fused2"]["native_decode"] is True and st["fused2"].get("native_passes", 0) >= 5


def test_tamper_is_caught_fail_closed(fprovider: Provider, tmp_path_factory: pytest.TempPathFactory) -> None:
    """The provider changes one element of every fused2 lane result: the fused2 Freivalds / range checks raise
    before any output is used; the client reports an integrity failure (no answer, no retry with other masks)."""
    from pgw.sdk.errors import IntegrityError, PGWAPIError

    root = tmp_path_factory.mktemp("sdk-fused-tamper")
    flag = fprovider.tamper_file
    c = _client(fprovider, fprovider.new_key("fused-tamper"), root)
    try:
        c.chat.completions.create(model=MODEL, messages=PROMPT, temperature=0, max_tokens=2)
        flag.write_text("1")
        with pytest.raises((IntegrityError, PGWAPIError)) as ei:
            c.chat.completions.create(model=MODEL, messages=PROMPT, temperature=0, max_tokens=2)
        assert "integrity" in str(ei.value).lower() or isinstance(ei.value, IntegrityError)
        st = c.status()["models"][MODEL]
        assert st["integrity_failures"] >= 1
    finally:
        flag.unlink(missing_ok=True)
        c.close()


def test_schedule_options_and_refusals(fprovider: Provider, provider_std: Provider,
                                       tmp_path_factory: pytest.TempPathFactory) -> None:
    from pgw.sdk.errors import NotFoundError
    from pgw.sdk.options import SDKOptions

    assert SDKOptions().schedule == "standard"  # default off
    for bad in ({"schedule": "fused", "prefix_cache": True}, {"schedule": "fused", "lane": "stream"},
                {"schedule": "auto", "masked_head": "off"}, {"schedule": "fused", "factory_weights": "stream"},
                {"schedule": "bogus"}):
        with pytest.raises(ValueError):
            SDKOptions(**bad).validate()
    # a provider without the fused entry: "fused" is refused (never a silent switch of reference); "auto" falls
    # back to standard with the reason recorded
    root = tmp_path_factory.mktemp("sdk-fused-refuse")
    c = _client(provider_std, provider_std.new_key("fused-refuse"), root)
    try:
        with pytest.raises(NotFoundError):
            c.chat.completions.create(model=MODEL, messages=PROMPT, temperature=0, max_tokens=2)
    finally:
        c.close()
    c = _client(provider_std, provider_std.new_key("fused-auto-std"), tmp_path_factory.mktemp("sdk-auto-std"),
                schedule="auto")
    try:
        c.chat.completions.create(model=MODEL, messages=PROMPT, temperature=0, max_tokens=2)
        d = c.status()["models"][MODEL]["schedule_decision"]
        assert d["schedule"] == "standard" and "does not serve" in d["reason"]
    finally:
        c.close()


def test_auto_measures_the_link_and_fixes_the_schedule(fprovider: Provider,
                                                       tmp_path_factory: pytest.TempPathFactory) -> None:
    """``auto`` on loopback (RTT well below the threshold): standard, with the measured link recorded; with a
    forced high RTT (``PGW_SDK_LINK_RTT_MS``): fused. The decision is made once and kept for the client's life."""
    c = _client(fprovider, fprovider.new_key("fused-auto"), tmp_path_factory.mktemp("sdk-auto"), schedule="auto")
    try:
        c.chat.completions.create(model=MODEL, messages=PROMPT, temperature=0, max_tokens=2)
        d = c.status()["models"][MODEL]["schedule_decision"]
        assert d["schedule"] == "standard" and d["link"]["rtt_ms"] < 3.0, d
        assert d["link"]["up_mbps"] and d["link"]["down_mbps"], d
        assert d["est_ms_standard"] is not None and d["est_ms_fused"] is not None
    finally:
        c.close()
    os.environ["PGW_SDK_LINK_RTT_MS"] = "20"
    try:
        c = _client(fprovider, fprovider.new_key("fused-auto2"), tmp_path_factory.mktemp("sdk-auto2"),
                    schedule="auto")
        try:
            r1 = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
            d = c.status()["models"][MODEL]["schedule_decision"]
            assert d["schedule"] == "fused", d
            os.environ["PGW_SDK_LINK_RTT_MS"] = "0.1"  # the link changes: the client keeps its schedule
            r2 = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
            assert c.status()["models"][MODEL]["schedule"] == "fused"
            assert _lpc(r1) == _lpc(r2)
        finally:
            c.close()
    finally:
        os.environ.pop("PGW_SDK_LINK_RTT_MS", None)


@pytest.fixture(scope="module")
def provider_std(sdk_pack_dir: Path, pki: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
                 ) -> Iterator[Provider]:
    p = start_provider(tmp_path_factory.mktemp("sdk-std-provider"), sdk_pack_dir, pki)
    try:
        yield p
    finally:
        stop_provider(p)


@pytest.fixture(scope="module")
def pks_provider(pki: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Iterator[Provider]:
    """A tiny checkpoint whose vocabulary is a multiple of 32 (pks-v3 needs ``out % 32 == 0``; the test model's 260
    is not, every real model's is), packed and served with the fused entry."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from pgw.modelpack import pack_checkpoint
    from tests.conftest import TINY_ARCH, _save_byte_tokenizer

    d = tmp_path_factory.mktemp("tiny-v288")
    torch.manual_seed(3)
    model = Qwen2ForCausalLM(Qwen2Config(**{**TINY_ARCH, "vocab_size": 288}))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    model.eval().save_pretrained(d, safe_serialization=True)
    _save_byte_tokenizer(d)
    pack = tmp_path_factory.mktemp("sdk-pack-v288") / "pack"
    pack_checkpoint(d, pack, name=MODEL)
    p = start_provider(tmp_path_factory.mktemp("sdk-fused-pks-provider"), pack, pki, extra_env=FUSED_ENV)
    try:
        yield p
    finally:
        stop_provider(p)


def test_fused_with_the_encrypted_pks_v3_factory(pks_provider: Provider, tmp_path_factory: pytest.TempPathFactory
                                                ) -> None:
    """The default mask scheme (pks-v3) on the fused schedule: the factory builds the hint over the fused2 linears
    and makes K56 masks by the top-digit plan (ordinary K32 maskgen items on the provider); the answer equals the
    fused reference bit for bit."""
    c = _client(pks_provider, pks_provider.new_key("fused-pks"), tmp_path_factory.mktemp("sdk-fused-pks"),
                mask_scheme="pks-v3", mask_segment_slots=256, mask_target_slots=256, mask_low_water_slots=128)
    try:
        r = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        ref = fused_reference(pks_provider.pack_dir, GREEDY)
        assert _lpc(r) == ref["choices"][0]["logprobs"]["content"]
        eng = c._engine  # type: ignore[attr-defined]
        logs = list(Path(eng.layout.root).rglob("factory.log"))
        text = "".join(p.read_text(errors="replace") for p in logs)
        assert "top-digit" in text, text[-3000:]
        assert c.status()["models"][MODEL]["schedule"] == "fused"
    finally:
        c.close()


def test_status_json_is_public(plain_client: Any) -> None:
    st = plain_client.status()
    json.dumps(st)  # serializable, no tensors


def test_strict_speculation_on_the_fused_schedule(fprovider: Provider, tmp_path_factory: pytest.TempPathFactory
                                                  ) -> None:
    """Strict speculation with the fused schedule: fixed ``spec_rows`` rows per verify pass (masked filler rows),
    the folded head's logits for every verified row, dummy passes after the answer; the answer is bit-identical
    to the fused reference without speculation, and every decode pass carries exactly ``spec_rows`` rows in 2L
    lane calls whatever the drafts."""
    msgs = [{"role": "user", "content": "Repeat: the cat sat on the mat. the cat sat on the mat. the cat sat on"}]
    body = {"temperature": 0, "max_tokens": 10, "logprobs": True, "top_logprobs": 2, "extra_body": {"ignore_eos": True}}
    c = _client(fprovider, fprovider.new_key("fused-spec"), tmp_path_factory.mktemp("sdk-fused-spec"),
                speculation="lookup", spec_rows=3, mask_target_slots=600, mask_low_water_slots=300)
    try:
        r = c.chat.completions.create(model=MODEL, messages=msgs, **body)
        ref = fused_reference(fprovider.pack_dir, body, msgs)
        assert _lpc(r) == ref["choices"][0]["logprobs"]["content"]
        rt = c._engine.models[MODEL].runtime  # type: ignore[attr-defined]
        prov = rt.backend.provider
        prov.record = []
        r2 = c.chat.completions.create(model=MODEL, messages=msgs, **body)
        assert _lpc(r2) == _lpc(r)
        L = int(rt.model.num_layers)
        rows = [n for _, n, _ in prov.record]
        # after the prompt pass: only 3-row items (spec_rows), 2L per pass
        decode = rows[2 * L * 3:]  # skip the prompt pass's multi pieces generously
        assert decode and set(decode[-2 * L:]) == {3}, rows
        st = c.status()["models"][MODEL]
        assert st["schedule"] == "fused" and st["integrity_failures"] == 0
        assert rt.generator.spec_rows == 3 and rt.generator.spec_strict
    finally:
        c.close()
