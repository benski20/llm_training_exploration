"""The masked LM head wired end to end (opt-in, ``head_mode = "provider_int8"``): the real compute service
(``masked_head = "also"``, the int8 head derived from its own pack), masks for the derived fingerprint from the
real factory, the gateway with the masked head (float head never loaded) against the plaintext reference (the
``plain`` dev mode with the same setting): bit-identical logprobs. Also: the derived fingerprint is checked at
connect, a tampered head result is refused, admin status reports the head, defaults stay off."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import torch

from pgw.core import DEFAULT_RING, FingerprintMismatchError
from pgw.core.errors import ConfigError
from pgw.engine.masked_head import MaskedHeadPack, head_fingerprint, head_linear_id, quantize_head
from pgw.gateway import ApiKeyStore, create_app
from tests.gateway.conftest import ADMIN, MODEL, InProcessService, auth, gateway, masked_settings, plain_settings

CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello there"}], "temperature": 0,
        "max_tokens": 8, "ignore_eos": True, "logprobs": True, "top_logprobs": 5}
HEAD = {"head_mode": "provider_int8"}


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


@pytest.fixture
def key(keys: ApiKeyStore) -> str:
    return keys.create("app")[1]


def _service_settings(tiny_pack_dir: Path, **kw: Any) -> Any:
    from pgw.service import ServiceSettings

    base: dict[str, Any] = {"packs": [tiny_pack_dir], "device": "cpu", "verify_packs": "full",
                            "allow_unauthenticated": True, "batch_window_ms": 1.0, "warmup": False}
    base.update(kw)
    return ServiceSettings(**base)


def _head_store(make_store: Any, gw_pack: Any, slots: int) -> Any:
    from tests.gateway.conftest import fill_store

    store = make_store(0)
    fill_store(store, MaskedHeadPack(gw_pack), DEFAULT_RING, slots)
    return store


async def _reference(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str, req: dict[str, Any],
                     **model_kw: Any) -> dict[str, Any]:
    plain = create_app(plain_settings(tiny_pack_dir, tmp_path, model_kw=model_kw), key_store=keys)
    async with gateway(plain) as c:
        r = await c.post("/v1/chat/completions", json=req, headers=auth(key))
        assert r.status_code == 200, r.text
        return dict(r.json())


async def test_end_to_end_real_service_exact(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                             keys: ApiKeyStore, key: str) -> None:
    from tests.service.conftest import running_app

    store = _head_store(make_store, gw_pack, 120)
    hfp = head_fingerprint(gw_pack.fingerprint)
    async with running_app(_service_settings(tiny_pack_dir, masked_head="also")) as svc:
        reg = svc.state.pgw.registry
        assert hfp in reg and gw_pack.fingerprint in reg
        head_entry = reg.get(hfp)
        lid = head_linear_id(gw_pack.arch.num_layers).key
        # the service derived the int8 head itself, from its own pack; layer kernels are shared
        w8, _ = quantize_head(gw_pack.load_plain("lm_head"))
        from pgw.core import ringmm_reference

        x = torch.randint(0, DEFAULT_RING.modulus, (3, int(w8.shape[1])), dtype=torch.int64)
        got = head_entry.kernel(lid).run(x, DEFAULT_RING)
        assert torch.equal(got, torch.from_numpy(ringmm_reference(x, w8, DEFAULT_RING.k)))
        assert head_entry.kernel("0.qkv") is reg.get(gw_pack.fingerprint).kernel("0.qkv")
        app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw=HEAD), mask_store=store,
                         transports={MODEL: httpx.ASGITransport(app=svc)}, key_store=keys)
        async with gateway(app) as c:
            rm = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
            assert rm.status_code == 200, rm.text
            st = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
            rt = app.state.pgw.runtime.get(MODEL)
            head_rows = int(rt.model._head.shape[0])
    assert st["head_mode"] == "provider_int8" and st["fingerprint"] == hfp
    assert st["masked_head"]["base_fingerprint"] == gw_pack.fingerprint
    assert st["masked_head"]["linear"] == lid
    assert st["speed"]["lm_head"]["device"] == "provider_int8"
    assert st["backend"]["integrity_failures"] == 0
    if not gw_pack.arch.tie_word_embeddings:
        assert head_rows == 0  # the float head was never loaded
    ref = await _reference(tiny_pack_dir, tmp_path, keys, key, CHAT, **HEAD)
    a, b = rm.json()["choices"][0], ref["choices"][0]
    assert a["message"] == b["message"]
    assert a["logprobs"] == b["logprobs"]  # bit for bit, including the top-5 alternatives
    # a new pinned reference: (almost surely) not the float head's logprobs
    flt = await _reference(tiny_pack_dir, tmp_path, keys, key, CHAT)
    assert flt["choices"][0]["logprobs"] != b["logprobs"]


async def test_head_item_on_the_wire_and_strict_speculation(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                            make_store: Any, keys: ApiKeyStore, key: str) -> None:
    """Strict speculation (R = 2, masked fillers): exact, and the provider sees the head as one R-row item."""
    store = _head_store(make_store, gw_pack, 200)
    hp = MaskedHeadPack(gw_pack)
    svc = InProcessService(hp)
    spec = {**HEAD, "spec_rows": 2, "spec_strict_passes": True, "spec_lookup": 2}
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw=spec), mask_store=store,
                     transports={MODEL: svc.transport}, key_store=keys)
    req = {**CHAT, "messages": [{"role": "user", "content": "ab ab ab ab ab ab"}]}
    async with gateway(app) as c:
        rm = await c.post("/v1/chat/completions", json=req, headers=auth(key))
        assert rm.status_code == 200, rm.text
    ref = await _reference(tiny_pack_dir, tmp_path, keys, key, req, **HEAD)
    assert rm.json()["choices"][0]["logprobs"] == ref["choices"][0]["logprobs"]
    lid = hp.head_id.key
    received = list(svc.received)  # one snapshot: a dummy pass may still be on its way when the app stops
    heads = [int(x.shape[0]) for k, x in received if k == lid]
    qkv = [int(x.shape[0]) for k, x in received if k == "0.qkv"]
    # the prompt pass (its layers' rows may be split into pipelined blocks) ends in one 1-row head call; every
    # decode pass, real or dummy, sends R = 2 rows to every linear, the head included (its last call). A pass cut
    # short by the shutdown has its first layer's call but not its head call yet: at most one such pass.
    decode = [n for n in qkv if n == 2]
    assert heads[0] == 1 and all(h == 2 for h in heads[1:]) and len(heads) - 1 >= 4
    assert len(decode) - (len(heads) - 1) in (0, 1), (heads, decode)


async def test_service_without_masked_head_is_refused(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                      make_store: Any, keys: ApiKeyStore) -> None:
    """The gateway derives the head fingerprint itself: a service serving only the base pack is refused."""
    svc = InProcessService(gw_pack)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw=HEAD), mask_store=make_store(0),
                     transports={MODEL: svc.transport}, key_store=keys)
    with pytest.raises(FingerprintMismatchError):
        async with gateway(app):
            pass


async def test_tampered_head_result_fails_closed(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                 make_store: Any, keys: ApiKeyStore, key: str) -> None:
    store = _head_store(make_store, gw_pack, 120)
    hp = MaskedHeadPack(gw_pack)
    lid = hp.head_id.key

    def tamper(linear: str, y: torch.Tensor) -> torch.Tensor:
        if linear == lid:
            y = y.clone()
            y[0, 0] += 1
        return y

    svc = InProcessService(hp, tamper=tamper)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw={**HEAD, "freivalds_rate": 1.0}),
                     mask_store=store, transports={MODEL: svc.transport}, key_store=keys)
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code >= 500 and r.status_code != 200
        st = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
        assert st["backend"]["integrity_failures"] >= 1


def test_service_registry_modes(tiny_pack_dir: Path, gw_pack: Any) -> None:
    from pgw.service import ModelRegistry

    hfp = head_fingerprint(gw_pack.fingerprint)
    off = ModelRegistry("cpu", rings=("K32A12C27",))
    off.load(tiny_pack_dir)
    assert [m.fingerprint for m in off.models()] == [gw_pack.fingerprint]  # default: no head entry
    only = ModelRegistry("cpu", rings=("K32A12C27",), masked_head="only")
    m = only.load(tiny_pack_dir)
    assert m.fingerprint == hfp and m.base_fingerprint == gw_pack.fingerprint and m.masked_head
    assert [x.fingerprint for x in only.models()] == [hfp]
    assert m.linear_dims[head_linear_id(gw_pack.arch.num_layers).key] == (gw_pack.arch.vocab_size,
                                                                           gw_pack.arch.hidden_size)
    also = ModelRegistry("cpu", rings=("K32A12C27",), masked_head="also")
    also.load(tiny_pack_dir)
    assert {x.fingerprint for x in also.models()} == {gw_pack.fingerprint, hfp}
    # shared kernels counted once
    assert sum(int(d["linears"]) for d in also.device_summary()) == 4 * gw_pack.arch.num_layers + 1
    also.unload(gw_pack.fingerprint)  # the derived entry goes with its base
    assert len(also) == 0
    with pytest.raises(ConfigError):
        ModelRegistry("cpu", masked_head="yes")


def test_fingerprint_domain_separation(gw_pack: Any) -> None:
    hp = MaskedHeadPack(gw_pack)
    assert hp.fingerprint == head_fingerprint(gw_pack.fingerprint) != gw_pack.fingerprint
    assert hp.base_fingerprint == gw_pack.fingerprint
    assert head_fingerprint(hp.fingerprint) != hp.fingerprint  # no fixed point / no double wrap
    with pytest.raises(ConfigError):
        MaskedHeadPack(hp)
    # masks made for one never serve the other: the store is keyed by fingerprint
    assert len({gw_pack.fingerprint, hp.fingerprint, head_fingerprint("0" * 64)}) == 3


def test_head_mode_config_defaults_and_validation(tiny_pack_dir: Path, tmp_path: Path) -> None:
    from pydantic import ValidationError

    from pgw.factory.config import FactorySettings
    from pgw.gateway import GatewayModelConfig
    from pgw.gateway.config import SETUP_MODEL_OPTION_KEYS
    from pgw.service import ServiceSettings

    assert GatewayModelConfig.model_fields["head_mode"].default == "gateway"
    assert ServiceSettings.model_fields["masked_head"].default == "off"
    assert FactorySettings.model_fields["masked_head"].default is False
    assert "head_mode" in SETUP_MODEL_OPTION_KEYS
    with pytest.raises(ValidationError, match="provider_int8"):
        GatewayModelConfig(name="m", pack_path=tiny_pack_dir, mode="float", head_mode="provider_int8")
