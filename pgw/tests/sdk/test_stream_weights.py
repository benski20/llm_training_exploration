"""``factory_weights="stream"`` end to end (tiny model, real provider on loopback, pks-v3, the default provisioner with
an explicit pin): no int8 weights anywhere in the client's cache or state, the factory's check material pre-drawn
from fingerprint-checked weight passes over the provider's provisioning endpoint, answers bit-identical to the
reference, and a wrong provider product is still caught online (the p31 check vectors come from the stock)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from tests.sdk.conftest import MODEL, Provider
from tests.sdk.test_e2e_cpu import GREEDY, PROMPT, _reference
from tests.sdk.test_prepare import _opts

pytestmark = pytest.mark.timeout(1800)


def _client(key: str, opts: Any) -> Any:
    from pgw import OpenAI

    return OpenAI(api_key=key, **{k: getattr(opts, k) for k in (
        "cache_dir", "state_dir", "mask_scheme", "masked_head", "model_fingerprints", "mask_workers",
        "mask_target_slots", "mask_low_water_slots", "factory_weights", "check_stock_segments")})


def _no_weights_at_rest(root: Path, provider: Provider) -> None:
    """No factory pack, and no file anywhere under the client's cache and state holds a linear's int8 bytes."""
    from pgw.modelpack import ModelPack

    assert not list((root / "cache" / "models").glob("*/factory-pack*"))
    with ModelPack.open(provider.pack_dir) as full:
        probes = [bytes(full.load_linear(e.id).w8.numpy().tobytes()[:4096]) for e in full.manifest.linears[:3]]
    for f in [*(root / "cache").rglob("*"), *(root / "state").rglob("*")]:
        if f.is_file() and f.stat().st_size >= 4096:
            data = f.read_bytes()
            assert not any(p in data for p in probes), f


def test_stream_mode_bit_identical_no_weights_at_rest_and_tamper_caught(provider: Provider, tmp_path: Path) -> None:
    from pgw.sdk import IntegrityError
    from pgw.sdk.cli import status_report

    key = provider.new_key()
    opts = _opts(provider, tmp_path, factory_weights="stream", check_stock_segments=4)
    c = _client(key, opts)
    try:
        t0 = time.monotonic()
        out = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        first_s = time.monotonic() - t0
        ref = _reference(provider.pack_dir, GREEDY, head_mode="gateway")
        assert out.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
        st = c.status()["models"][MODEL]
        f = st["factory"]
        assert f["scheme"] == "pks-v3" and f["factory_weights"] == "stream" and not f["halted"]
        cs = f["check_stock"]
        assert cs["passes"] >= 1 and cs["taken_segments"] >= 1 and cs["taken_private_sets"] >= 1
        assert cs["last_pass_weights_bytes"] > 0 and not cs.get("last_pass_error")
        assert st["verified_requests"] >= 1 and st["integrity_failures"] == 0
        assert "+stream" in st["provisioned"]["source"]
        _no_weights_at_rest(tmp_path, provider)
        # a second request uses further pre-drawn sets; same bits
        again = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        assert again.to_dict()["choices"][0] == out.to_dict()["choices"][0]
        rep = status_report(tmp_path / "cache", tmp_path / "state")
        text = json.dumps(rep)
        assert "check_stock" in text and key not in text
        # a wrong product from the provider: the online p31 check (u from the stock's seed, uW from a pass) catches it
        provider.tamper_file.touch()
        try:
            with pytest.raises(IntegrityError):
                c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        finally:
            provider.tamper_file.unlink()
        print(f"\nfactory_weights=stream: first answer after {first_s:.1f}s; stock {cs}")
    finally:
        c.close()


def test_prepare_stream_keeps_no_factory_pack(provider: Provider, tmp_path: Path) -> None:
    from pgw.sdk.prepare import prepare

    key = provider.new_key()
    opts = _opts(provider, tmp_path, factory_weights="stream")
    r = prepare(MODEL, api_key=key, options=opts)
    assert r["factory_weights"] == "stream" and not r["factory_pack"] and r["hint_dir_bytes"] > 0
    assert not list((tmp_path / "cache" / "models").glob("*/factory-pack*"))
    # switching an existing "pack" cache to "stream" deletes its factory pack
    opts2 = _opts(provider, tmp_path / "b")
    r2 = prepare(MODEL, api_key=key, options=opts2)
    assert Path(r2["factory_pack"]).is_dir()
    prepare(MODEL, api_key=key, options=_opts(provider, tmp_path / "b", factory_weights="stream"))
    assert not Path(r2["factory_pack"]).exists()


def test_options_refuse_stream_without_pks() -> None:
    from pgw.sdk.options import SDKOptions

    with pytest.raises(ValueError, match="pks-v3"):
        SDKOptions.from_env(factory_weights="stream", mask_scheme="plain")
    with pytest.raises(ValueError, match="factory_weights"):
        SDKOptions.from_env(factory_weights="disk")
    assert SDKOptions.from_env().factory_weights == "pack"


def test_factory_env_for_stream() -> None:
    from pgw.factory.config import FactorySettings
    from pgw.sdk.factory import FactoryPlan, factory_env

    plan = FactoryPlan(model=MODEL, pack_path=Path("/c/pack"), ring="K32A12C27", state_dir=Path("/s"),
                       store_root=Path("/m"), kek_file=Path("/k"), counter_file=Path("/n"), scheme="pks-v3",
                       check_field="p31", masked_head=True, workers=2, threads=1, nice=10, target_slots=1024,
                       low_water_slots=512, segment_slots=1024, first_segment_slots=64, service_url="https://p",
                       factory_weights="stream", pack_fingerprint="a" * 64, check_stock_segments=64,
                       check_stock_private=8)
    env = factory_env(plan)
    assert env["PGW_FACTORY_FACTORY_WEIGHTS"] == "stream" and env["PGW_FACTORY_WEIGHTS_SOURCE"] == "provider"
    import os

    saved = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        s = FactorySettings()
        assert s.factory_weights == "stream" and s.check_stock_segments == 64 and s.stock_low_water == (16, 4)
        assert s.pack_fingerprint == "a" * 64
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
