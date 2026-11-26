"""The fused, resumable local hint build (``provision(hint="local", factory_pack=True, resume=True)``, the core of
``pgw sdk prepare``) against a real compute service on loopback: bit-identical to a separate build, resumable after
an interruption, and never trusting a tampered partial state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pgw.modelpack import ModelPack
from pgw.provision.client import provision
from tests.provision.test_provision_e2e import _hint_files, _local_reference, _session, service  # noqa: F401


def _det_seeds(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Deterministic client seeds (both seed sources of a build); call the result to restart the sequence."""
    import pgw.core.prng
    import pgw.hemask.pks_backend

    state = {"i": 0}

    def seed() -> bytes:
        state["i"] += 1
        return hashlib.sha256(b"seed%d" % state["i"]).digest()

    monkeypatch.setattr(pgw.core.prng, "new_seed", seed)
    monkeypatch.setattr(pgw.hemask.pks_backend, "new_seed", seed)
    return lambda: state.update(i=0)


def test_fused_local_build_equals_separate_build(service: dict[str, Any], tmp_path: Path,  # noqa: F811
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """hint='local' with the factory pack builds each linear's hint while its W8 is in memory (one pass); the hint
    files equal a separate local build's with the same seeds, and the factory pack verifies."""
    reset = _det_seeds(monkeypatch)
    pa = provision(service["fp"], session=_session(service), cache_dir=tmp_path / "a", hint="local",
                   masked_head=False, factory_pack=True)
    reset()
    pb = provision(service["fp"], session=_session(service), cache_dir=tmp_path / "b", hint="local",
                   masked_head=False)
    try:
        assert pa.hint_store.a_seed == pb.hint_store.a_seed and pa.hint_store.ks_seed == pb.hint_store.ks_seed
        assert _hint_files(pa.hint_store) == _hint_files(pb.hint_store)
        assert pa.hint_store.meta["digests"] == pb.hint_store.meta["digests"]
        assert pa.hint_store.meta["row_norms"] == pb.hint_store.meta["row_norms"]

        assert pa.timings.get("hint_fused", 0) > 0
        with ModelPack.open(pa.factory_pack_dir, verify="full", expected_fingerprint=service["fp"]) as full:
            assert full.fingerprint == service["fp"]
        ref = _local_reference(service["pack_dir"], pa, tmp_path)  # the plain PKSHintStore.build
        got = _hint_files(pa.hint_store)
        assert all(got[k] == v for k, v in _hint_files(ref).items())
    finally:
        pa.close()
        pb.close()


def _interrupt_after(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    from pgw.hemask.pks_backend import HintBuild

    orig = HintBuild.add
    calls = {"n": 0}

    def add(self: Any, item: Any, **kw: Any) -> None:
        if not self.done(item.key):
            calls["n"] += 1
            if calls["n"] > n:
                raise KeyboardInterrupt("simulated interruption")
        orig(self, item, **kw)

    monkeypatch.setattr(HintBuild, "add", add)


def _partial(cache: Path, fp: str) -> tuple[Path, dict[str, Any]]:
    d = next((cache / "models" / fp[:16] / "hints").glob("*.partial"))
    return d, json.loads((d / "partial.json").read_text())


def _interrupted(svc: dict[str, Any], cache: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    kw: dict[str, Any] = {"session": _session(svc), "cache_dir": cache, "hint": "local", "masked_head": False,
                          "factory_pack": True, "resume": True}
    with monkeypatch.context() as mp:
        _interrupt_after(mp, 2)
        with pytest.raises(KeyboardInterrupt):
            provision(svc["fp"], **kw)
    return kw


def test_prepare_resumes_after_interruption(service: dict[str, Any], tmp_path: Path,  # noqa: F811
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "c"
    kw = _interrupted(service, cache, monkeypatch)
    d, st = _partial(cache, service["fp"])
    assert len(st["done"]) == 2 and "mac" in st
    mdir = cache / "models" / service["fp"][:16]
    assert (mdir / "factory-pack.partial" / "state.json").is_file()
    total_w = sum(t.nbytes for t in ModelPack.open(service["pack_dir"]).manifest.tensors.values() if t.dtype == "int8")
    pm = provision(service["fp"], **kw)
    try:
        assert pm.hint_store.a_seed.hex() == st["a_seed"]  # the authenticated seeds are reused
        assert 0 < pm.sizes["weights_streamed"] < total_w  # finished linears are not downloaded again
        ref = _local_reference(service["pack_dir"], pm, tmp_path)
        assert _hint_files(pm.hint_store) == _hint_files(ref)
        assert not d.exists() and not (mdir / "factory-pack.partial").exists()
    finally:
        pm.close()


@pytest.mark.parametrize("tamper", ["state", "file"])
def test_prepare_resume_never_trusts_a_tampered_partial(service: dict[str, Any], tmp_path: Path,  # noqa: F811
                                                        monkeypatch: pytest.MonkeyPatch, tamper: str) -> None:
    """An edited partial.json (fails its HMAC) is discarded with its seeds; a finished linear whose file changed is
    rebuilt. Either way the final store equals a reference build with its seeds."""
    cache = tmp_path / "c"
    kw = _interrupted(service, cache, monkeypatch)
    d, st = _partial(cache, service["fp"])
    if tamper == "state":
        k = next(iter(st["done"]))
        st["done"][k]["norm"] = 1.0  # a smaller row norm would weaken the noise refusal
        (d / "partial.json").write_text(json.dumps(st))
    else:
        p = d / f"{next(iter(st['done']))}.npy"
        raw = bytearray(p.read_bytes())
        raw[-1] ^= 1
        p.write_bytes(bytes(raw))
    pm = provision(service["fp"], **kw)
    try:
        assert (pm.hint_store.a_seed.hex() == st["a_seed"]) == (tamper == "file")
        ref = _local_reference(service["pack_dir"], pm, tmp_path)
        assert _hint_files(pm.hint_store) == _hint_files(ref)
        assert pm.hint_store.meta["row_norms"] == ref.meta["row_norms"]
    finally:
        pm.close()
