"""``pgw sdk prepare`` against a real provider on loopback (tiny model): the one-time setup ahead of the first request,
picked up by a later client, bit-identical answers, and a prepared hint is never used unless its metadata verifies.

The default provisioner (the provider's real provisioning endpoints) with an explicit fingerprint pin for the tiny
model; pks-v3 with the masked head off (the tiny model's 260-token head is not a pks-v3 shape)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from tests.sdk.conftest import MODEL, Provider
from tests.sdk.test_e2e_cpu import GREEDY, PROMPT, _reference

pytestmark = pytest.mark.timeout(1800)


def _opts(provider: Provider, root: Path, **kw: Any) -> Any:
    from pgw.modelpack import ModelPack
    from pgw.sdk.options import SDKOptions

    with ModelPack.open(provider.pack_dir) as p:
        fp = p.fingerprint
    base: dict[str, Any] = {"cache_dir": root / "cache", "state_dir": root / "state", "mask_scheme": "pks-v3",
                            "masked_head": "off", "model_fingerprints": {MODEL: fp}, "mask_workers": 1,
                            "mask_target_slots": 200, "mask_low_water_slots": 100, "timeout_s": 600.0}
    base.update(kw)
    return SDKOptions.from_env(**base)


def _provisioned(key: str, opts: Any) -> Any:
    """What a client's engine gets from the default provisioner (no factory started)."""
    from pgw.sdk.engine import Engine
    from pgw.sdk.keys import parse_key

    eng = Engine(parse_key(key), opts)

    async def run() -> Any:
        s = await eng.ensure_session()
        return await asyncio.to_thread(eng.provisioner, MODEL, session=s, cache_dir=eng.layout.models_dir,
                                       masked_head=False)

    try:
        return eng.call(run())
    finally:
        eng.close()


def test_prepare_then_first_request_uses_the_prepared_hint(provider: Provider, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from pgw import OpenAI
    from pgw.cli.main import cli
    from pgw.sdk.prepare import prepare

    key = provider.new_key()
    opts = _opts(provider, tmp_path)
    lines: list[str] = []
    r = prepare(MODEL, api_key=key, options=opts, progress=lines.append)
    assert r["hint_dir"] and r["hint_dir_bytes"] > 0 and r["factory_pack_bytes"] > 0
    assert any(x.startswith("hint") for x in lines) and any(x.startswith("weights") for x in lines)
    store = next(p for p in Path(r["hint_dir"]).glob("*pks-v3*") if p.is_dir())
    meta = json.loads((store / "meta.json").read_text())
    assert meta["source"] == "local" and "mac" in meta
    assert not list(Path(r["hint_dir"]).glob("*.partial"))
    # the CLI again: everything is in place, nothing is rebuilt (same seeds)
    res = CliRunner().invoke(cli, ["sdk", "prepare", "--model", MODEL, "--json", "--masked-head", "off", "--pin",
                                   f"{MODEL}={opts.model_fingerprints[MODEL]}"],
                             env={"PGW_API_KEY": key, "PGW_CACHE_DIR": str(tmp_path / "cache"),
                                  "PGW_STATE_DIR": str(tmp_path / "state"), "PGW_MASK_SCHEME": "pks-v3"},
                             catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert json.loads((store / "meta.json").read_text())["a_seed"] == meta["a_seed"]
    # a client with the same cache hands the prepared store to its factory: no second hint is built
    prov = _provisioned(key, opts)
    assert prov.hint_dir == Path(r["hint_dir"]) and "prepared-hint" in prov.source
    c = OpenAI(api_key=key, **{k: getattr(opts, k) for k in (
        "cache_dir", "state_dir", "mask_scheme", "masked_head", "model_fingerprints", "mask_workers",
        "mask_target_slots", "mask_low_water_slots")})
    try:
        t0 = time.monotonic()
        out = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
        ref = _reference(provider.pack_dir, GREEDY, head_mode="gateway")
        assert out.to_dict()["choices"][0]["logprobs"]["content"] == ref["choices"][0]["logprobs"]["content"]
        st = c.status()["models"][MODEL]
        assert st["factory"]["scheme"] == "pks-v3" and not st["factory"]["halted"]
        assert not list((tmp_path / "cache" / "models").glob("*/hint/*pks-v3*")), "no second hint was built"
        print(f"\nprepared pks-v3: first answer after {time.monotonic() - t0:.1f}s")
    finally:
        c.close()


def test_a_prepared_hint_is_used_only_if_its_metadata_verifies(provider: Provider, tmp_path: Path) -> None:
    from pgw.core.errors import IntegrityError
    from pgw.sdk.prepare import prepare

    key = provider.new_key()
    opts = _opts(provider, tmp_path)
    r = prepare(MODEL, api_key=key, options=opts)
    store = next(p for p in Path(r["hint_dir"]).glob("*pks-v3*") if p.is_dir())
    meta_p = store / "meta.json"
    good = meta_p.read_text()
    meta = json.loads(good)
    k = next(iter(meta["row_norms"]))
    meta["row_norms"][k] = 1.0  # a smaller row norm would weaken the factory's noise refusal
    meta_p.write_text(json.dumps(meta))
    with pytest.raises(IntegrityError):
        _provisioned(key, opts)
    meta_p.write_text(good)
    key_file = tmp_path / "cache" / "keys" / "pks-hint.key"
    key_file.rename(key_file.with_suffix(".moved"))  # a copied cache without its hint key: nothing is trusted
    prov = _provisioned(key, opts)
    assert prov.hint_dir is None and "prepared-hint" not in prov.source


def test_pks_segments_are_whole_groups() -> None:
    from pgw.sdk.factory import segment_slots
    from pgw.sdk.options import SDKOptions

    assert SDKOptions().mask_segment_slots == 1024
    got = [segment_slots("pks-v3", s) for s in (16, 64, 255, 256, 430, 1024, 1500)]
    assert got == [16, 64, 255, 256, 256, 1024, 1280]
    assert segment_slots("plain", 430) == 430 and segment_slots("mlwe", 430) == 430
