"""GPU mask service (customer-cloud deployment): start-up bit-exact check, refusals, scrubbing, delivery.

Runs on the CPU here (``mask_service_allow_cpu``; the device path is exercised with the torch emulation of the
GPU PRNG). The CUDA path is the same :class:`MaskGenerator` code with ``cuda_int8``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from pgw.core import RING_K32_A8, RING_K40_A12, RING_K48_A20, IntegrityError, center, ringmm_reference
from pgw.core.errors import ConfigError
from pgw.core.freivalds import P31, prime_check_vectors
from pgw.core.prng import SeedExpander
from pgw.factory import ExportSink, FactorySettings, MaskGenerator, StoreSink
from pgw.factory import mask_service as ms
from pgw.factory.courier import OutboxServer, pull_once

from .conftest import FakePack

maskstore = pytest.importorskip("pgw.maskstore")


def _settings(tmp_path: Path, **kw: Any) -> FactorySettings:
    base: dict[str, Any] = {"pack_path": tmp_path, "mask_service": True, "mask_service_allow_cpu": True}
    base.update(kw)
    return FactorySettings(**base)


# ---- exactness -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ring", [RING_K32_A8, RING_K40_A12, RING_K48_A20], ids=lambda r: r.name)
def test_startup_check_passes_bit_exact(ring: Any) -> None:
    gen = MaskGenerator(FakePack(layers=3), ring, backend="cpu_int8")
    rep = ms.startup_check(gen, rows=5, linears=0)
    assert rep.ok and not rep.mismatches
    assert len(rep.linears_checked) == 12 and rep.rows_per_linear == 5
    assert rep.slots == gen.chunk_slots_for(None) + 3
    assert rep.wr_elements_compared > 0 and rep.uw_elements_compared > 0


def test_startup_check_p31_and_device_prng_emulation() -> None:
    gen = MaskGenerator(FakePack(), "K32A12C27", backend="cpu_int8", check_field="p31", gpu_prng=True,
                        gpu_prng_impl="torch")
    rep = ms.startup_check(gen, rows=4, linears=4)
    assert rep.ok and rep.check_field == "p31" and rep.gpu_prng is True
    assert rep.checks == gen.checks_for() and len(rep.linears_checked) == 4


def test_reference_matches_python_ints() -> None:
    torch.manual_seed(0)
    w = torch.randint(-128, 128, (700, 37), dtype=torch.int8)  # > one reference block of rows
    r = SeedExpander(b"\x01" * 32).ring_uniform((3, 37), 48, "r", "x")
    assert ms.reference_wr_rows(r.clone(), w, 48).astype(np.int64).tolist() == ringmm_reference(r, w, 48).tolist()
    u = SeedExpander(b"\x02" * 32).ring_uniform((2, 700), 40, "u", "x")
    ref = ringmm_reference(u, w.t().contiguous(), 40)
    assert ms.reference_uw(u, w, 40, None).astype(np.int64).tolist() == ref.tolist()
    up = prime_check_vectors(b"\x03" * 32, 1, 700, "x")
    want = [[int(sum(int(up[0, i]) * int(w[i, j]) for i in range(700)) % P31) for j in range(37)]]
    assert ms.reference_uw(up, w, 32, P31).astype(np.int64).tolist() == want


def _corrupting(gen: MaskGenerator, *, what: str, linear_index: int = -1) -> list[Any]:
    """Make ``gen.generate`` corrupt one element of one linear's W r (or u W); returns the yielded items."""
    seen: list[Any] = []
    target = gen.specs[linear_index].id.key
    orig = gen.generate

    def gen_bad(*a: Any, **kw: Any) -> Any:
        for item in orig(*a, **kw):
            if item.id.key == target:
                if what == "wr":
                    item.wr[-1, 3] ^= 1 << 7
                elif what == "uw" and item.uw is not None:
                    item.uw[0, 1] ^= 1
            seen.append(item)
            yield item

    gen.generate = gen_bad  # type: ignore[method-assign]
    return seen


@pytest.mark.parametrize("what", ["wr", "uw"])
def test_startup_check_refuses_mismatch_and_wipes(what: str) -> None:
    gen = MaskGenerator(FakePack(), RING_K40_A12, backend="cpu_int8", self_check_rows=0)
    seen = _corrupting(gen, what=what)
    with pytest.raises(IntegrityError) as ei:
        ms.startup_check(gen, rows=3, linears=2)  # last layer's down is among the picked linears
    bad = ei.value.details["mismatches"]
    assert bad and bad[0]["what"] == what and bad[0]["linear"] == gen.specs[-1].id.key
    assert seen and all(int(item.wr.abs().sum()) == 0 for item in seen)  # every W r buffer zeroed


def test_startup_check_wipes_on_success() -> None:
    gen = MaskGenerator(FakePack(), RING_K40_A12, backend="cpu_int8")
    seen = _corrupting(gen, what="none")
    ms.startup_check(gen)
    assert seen and all(int(item.wr.abs().sum()) == 0 for item in seen)


def test_pick_linears_covers_first_and_last_layer() -> None:
    specs = list(range(40))
    got = ms._pick_linears(specs, 8)  # type: ignore[arg-type]
    assert len(got) == 8 and {0, 1, 2, 3, 36, 37, 38, 39} == set(got)
    assert ms._pick_linears(specs, 0) == specs  # type: ignore[arg-type]
    got = ms._pick_linears(specs, 3)  # type: ignore[arg-type]
    assert len(got) == 3 and 0 in got and 39 in got


# ---- scrubbing -------------------------------------------------------------------------------------------------
class _FakeCuda:
    def __init__(self, reserved: int, allocated: int, fail_above: int | None = None):
        self.reserved, self.allocated, self.fail_above = reserved, allocated, fail_above
        self.calls: list[str] = []
        self.allocs: list[int] = []

    def synchronize(self, dev: Any) -> None:
        self.calls.append("sync")

    def memory_reserved(self, dev: Any) -> int:
        return self.reserved

    def memory_allocated(self, dev: Any) -> int:
        return self.allocated

    def empty_cache(self) -> None:
        self.calls.append("empty")

    def alloc(self, n: int) -> bytes:
        if self.fail_above is not None and n > self.fail_above:
            raise RuntimeError("CUDA out of memory (simulated)")
        self.allocs.append(n)
        return b""


def test_scrub_device_memory_zeroes_free_cache_then_releases() -> None:
    fc = _FakeCuda(reserved=300 << 20, allocated=100 << 20)
    assert ms.scrub_device_memory("cuda:0", _cuda=fc, _alloc=fc.alloc) == 200 << 20
    assert sum(fc.allocs) == 200 << 20 and max(fc.allocs) <= 64 << 20
    assert fc.calls[0] == "sync" and fc.calls[-1] == "empty"
    frag = _FakeCuda(reserved=40 << 20, allocated=0, fail_above=4 << 20)
    assert ms.scrub_device_memory("cuda:0", _cuda=frag, _alloc=frag.alloc) == 40 << 20
    assert max(frag.allocs) <= 4 << 20 and frag.calls[-1] == "empty"
    assert ms.scrub_device_memory("cpu") == 0


def test_generator_scrubs_after_each_segment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ms, "scrub_device_memory", lambda dev, **kw: calls.append(str(dev)) or 0)
    store = maskstore.MaskStore.open(tmp_path / "s", maskstore.FileKeyProvider(tmp_path / "kek", create=True))
    try:
        gen = MaskGenerator(FakePack(), RING_K40_A12, backend="cpu_int8")
        gen.generate_into_store(StoreSink(store), 4, chunk_slots=4)
        assert calls == []
        gen.scrub_device = True
        gen.generate_into_store(StoreSink(store), 4, chunk_slots=4)
        assert calls == ["cpu"]
    finally:
        store.close()


# ---- refusal paths ---------------------------------------------------------------------------------------------
def test_settings_refusals(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="he_service_url"):
        _settings(tmp_path, he_service_url="https://svc:8443")
    with pytest.raises(ValidationError, match="self-check"):
        _settings(tmp_path, self_check_rows=0)
    with pytest.raises(ValidationError, match="partner_signing_key_file"):
        _settings(tmp_path, output="export", export_dir=tmp_path / "x", recipient_public_key_file=tmp_path / "k",
                  destination_store_id="abc")
    s = FactorySettings(pack_path=tmp_path)
    assert s.mask_service is False and s.mask_service_allow_cpu is False  # default off


def test_prepare_refuses_cpu_without_opt_in(tmp_path: Path) -> None:
    gen = MaskGenerator(FakePack(), RING_K40_A12, backend="cpu_int8")
    with pytest.raises(ConfigError, match="no GPU"):
        ms.prepare_mask_service(_settings(tmp_path, mask_service_allow_cpu=False), {gen.key: gen})
    assert gen.scrub_device is False


def test_prepare_refuses_on_mismatch_before_any_segment(tmp_path: Path) -> None:
    gen = MaskGenerator(FakePack(), RING_K40_A12, backend="cpu_int8")
    _corrupting(gen, what="uw", linear_index=0)
    with pytest.raises(IntegrityError):
        ms.prepare_mask_service(_settings(tmp_path), {gen.key: gen})


def test_startup_check_refuses_encrypted_generation() -> None:
    gen = MaskGenerator(FakePack(), RING_K40_A12, backend="cpu_int8")
    gen.remote = object()
    with pytest.raises(ConfigError, match="locally"):
        ms.startup_check(gen)


# ---- delivery: same-host store and the export channel to another VM ---------------------------------------------
def _verify_store_segment(store: Any, pack: FakePack, ring: Any, slots: int) -> None:
    with store.reserve(pack.fingerprint, ring, slots) as res:
        for spec in pack.linear_specs():
            left = slots
            while left:
                n = min(left, int(res.rows_until_boundary(spec.id)))
                r, wr = res.take(spec.id, n)
                assert wr.tolist() == ringmm_reference(r, pack.weights[spec.id.key], ring.k).tolist()
                left -= n


@pytest.mark.parametrize("check_field", ["ring", "p31"])
def test_mask_service_into_same_host_store(tmp_path: Path, check_field: str) -> None:
    pack = FakePack()
    store = maskstore.MaskStore.open(tmp_path / "gw-store", maskstore.FileKeyProvider(tmp_path / "kek", create=True))
    try:
        gen = MaskGenerator(pack, "K32A12C27", backend="cpu_int8",
                            check_field=None if check_field == "ring" else check_field)
        reports = ms.prepare_mask_service(_settings(tmp_path, check_field=check_field), {gen.key: gen})
        assert reports[gen.key].ok and gen.scrub_device
        gen.generate_into_store(StoreSink(store), 6, chunk_slots=4)
        hdr = [s for s in store.inventory()]
        assert hdr and hdr[0].slots_available == 6
        _verify_store_segment(store, pack, gen.ring, 6)
    finally:
        store.close()


def _write_token(p: Path) -> Path:
    p.write_text("t0ken-for-tests-0123456789\n")
    return p


@pytest.fixture
def two_vms(tmp_path: Path) -> Any:
    """Gateway store (with recipient key + trusted partner key) and a mask-service export sink bound to it."""
    gw = tmp_path / "gateway"
    gw.mkdir()
    recipient = maskstore.RecipientKeyPair.generate(gw / "recipient.key")
    signer = maskstore.PartnerSigningKey.generate(tmp_path / "ms" / "sign.key")
    overrides = {"root": gw / "store", "key_file": gw / "kek", "key_provider": "file",
                 "recipient_key_file": gw / "recipient.key",
                 "trusted_partner_keys": (signer.public_bytes().hex(),)}
    maskstore.FileKeyProvider(gw / "kek", create=True)
    from pgw.maskstore.admin import open_store

    with open_store(**overrides) as st:
        store_id = st.store_id
    sink = ExportSink(tmp_path / "ms" / "export", recipient.public_bytes(), signing_key=signer,
                      destination_store_id=store_id)
    token = _write_token(tmp_path / "token")
    srv = OutboxServer(sink.outbox, token, "127.0.0.1", 0).start()
    yield sink, srv, token, overrides, gw
    srv.stop()


def test_export_courier_pull_round_trip_exact(two_vms: Any, tmp_path: Path) -> None:
    sink, srv, token, overrides, gw = two_vms
    pack = FakePack()
    gen = MaskGenerator(pack, RING_K40_A12, backend="cpu_int8")
    ms.prepare_mask_service(_settings(tmp_path), {gen.key: gen})
    seg1 = gen.generate_into_store(sink, 5, chunk_slots=4)
    seg2 = gen.generate_into_store(sink, 3, chunk_slots=4)
    (sink.outbox / "stray.pgwm").write_bytes(b"x")  # unsigned: never offered
    res = pull_once(srv.url, token, gw / "inbox", store_overrides=overrides)
    assert sorted(r["segment_id"] for r in res if r.get("ok")) == sorted([seg1, seg2])
    assert [p.name for p in sink.outbox.iterdir()] == ["stray.pgwm"]  # delivered files retired remotely
    assert not list((gw / "inbox").iterdir())  # local copies removed after import
    from pgw.maskstore.admin import open_store

    with open_store(**overrides) as st:
        _verify_store_segment(st, pack, RING_K40_A12, 8)
    assert pull_once(srv.url, token, gw / "inbox", store_overrides=overrides) == []


def test_courier_backpressure_reuse_and_auth(two_vms: Any, tmp_path: Path) -> None:
    sink, srv, token, overrides, gw = two_vms
    gen = MaskGenerator(FakePack(), RING_K40_A12, backend="cpu_int8")
    gen.generate_into_store(sink, 4, chunk_slots=4)
    gen.generate_into_store(sink, 4, chunk_slots=4)
    res = pull_once(srv.url, token, gw / "inbox", store_overrides=overrides, max_available_slots=4)
    assert [bool(r.get("segment_id")) for r in res] == [True, False] and res[1]["skipped"] == "store full"
    # a copy of an imported file offered again (e.g. the remote delete was lost): refused as reuse, retired
    left = sorted(p.name for p in sink.outbox.iterdir())
    assert len(left) == 2  # one segment + its signature still offered
    bad = _write_token(tmp_path / "wrong")
    bad.write_text("wrong-token-0123456789\n")
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as ei:
        pull_once(srv.url, bad, gw / "inbox", store_overrides=overrides)
    assert ei.value.code == 401
    res = pull_once(srv.url, token, gw / "inbox", store_overrides=overrides)
    assert res[0]["ok"] and not list(sink.outbox.iterdir())


def test_courier_refuses_path_escape(two_vms: Any, tmp_path: Path) -> None:
    import urllib.request

    sink, srv, token, _overrides, _gw = two_vms
    (sink.outbox.parent / "secret.pgwm").write_bytes(b"nope")
    tok = token.read_text().strip()
    for name in ("..%2Fsecret.pgwm", "../secret.pgwm", "x.txt"):
        rq = urllib.request.Request(f"{srv.url}/pgw/v1/outbox/{name}",  # noqa: S310 - http to a local test server
                                    headers={"Authorization": f"Bearer {tok}"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with pytest.raises(urllib.error.HTTPError) as ei:
            opener.open(rq, timeout=10)
        assert ei.value.code == 404


def test_reuse_refusal_is_retired(two_vms: Any, tmp_path: Path) -> None:
    sink, srv, token, overrides, gw = two_vms
    gen = MaskGenerator(FakePack(), RING_K40_A12, backend="cpu_int8")
    gen.generate_into_store(sink, 4, chunk_slots=4)
    name = next(p.name for p in sink.outbox.glob("*.pgwm"))
    keep = {p.name: p.read_bytes() for p in sink.outbox.iterdir()}
    assert pull_once(srv.url, token, gw / "inbox", store_overrides=overrides)[0]["ok"]
    for n, b in keep.items():  # the same file shows up again
        (sink.outbox / n).write_bytes(b)
    res = pull_once(srv.url, token, gw / "inbox", store_overrides=overrides)
    assert res[0]["ok"] is False and res[0]["code"] == "mask_reuse" and res[0]["file"] == name
    assert not list(sink.outbox.iterdir())


def test_unmask_through_store_is_exact(tmp_path: Path) -> None:
    """Gateway-side use of a mask-service segment: y = center(W8 x_hat - W r) equals W8 x exactly."""
    from pgw.core import mask_activations

    pack = FakePack()
    store = maskstore.MaskStore.open(tmp_path / "s", maskstore.FileKeyProvider(tmp_path / "kek", create=True))
    try:
        gen = MaskGenerator(pack, RING_K40_A12, backend="cpu_int8")
        ms.prepare_mask_service(_settings(tmp_path), {gen.key: gen})
        gen.generate_into_store(StoreSink(store), 3, chunk_slots=4)
        with store.reserve(pack.fingerprint, RING_K40_A12, 3) as res:
            spec = pack.linear_specs()[0]
            w = pack.weights[spec.id.key]
            r, wr = res.take(spec.id, 3)
            xq = torch.randint(-100, 100, (3, spec.in_features), dtype=torch.int64)
            y_hat = torch.from_numpy(ringmm_reference(mask_activations(xq, r, RING_K40_A12), w, 40))
            assert torch.equal(center(y_hat - wr, RING_K40_A12), xq @ w.to(torch.int64).t())
    finally:
        store.close()


# ---- CLI ---------------------------------------------------------------------------------------------------------
def test_cli_check_only_and_refusals(tiny_pack_dir: Path) -> None:
    import json

    from click.testing import CliRunner

    from pgw.factory.cli import factory_group

    runner = CliRunner()
    base = ["mask-service", "--pack", str(tiny_pack_dir), "--device", "cpu", "--backend", "cpu_int8"]
    res = runner.invoke(factory_group, [*base, "--check-only"])
    assert res.exit_code == 1 and "no GPU" in res.output
    res = runner.invoke(factory_group, [*base, "--allow-cpu", "--check-only", "--check-field", "p31", "--json"])
    assert res.exit_code == 0, res.output
    rep = json.loads(res.output[res.output.index("["):])
    assert rep[0]["ok"] and rep[0]["check_field"] == "p31" and rep[0]["wr_elements_compared"] > 0
    res = runner.invoke(factory_group, [*base, "--he-service", "https://svc.example:8443"])
    assert res.exit_code != 0 and "locally" in res.output
