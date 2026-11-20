"""Tiered mask source (ring + store) as the gateway sees it: fallback, spanning, ring_only, windows."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A12_C27, SeedExpander
from pgw.core.errors import ConfigError, MaskExhaustedError
from pgw.gateway.masks import WindowedReservation
from pgw.maskstore import MaskStore, StaticKeyProvider
from pgw.maskstore.ringbuf import RingFull, RingReservation, SeedRegistry, ring_geometry
from pgw.maskstore.tiered import ChainedReservation, TieredMaskSource

from ..maskstore.conftest import ToyModel, make_model, write_segment

RING = RING_K32_A12_C27
SEG = 16


def _cfg(model: Any, slabs: int = 4, **kw: Any) -> SimpleNamespace:
    specs = model.specs if hasattr(model, "specs") else model.linear_specs()
    g = ring_geometry(specs, RING, 1 << 30, segment_slots=SEG)
    base = {"capacity_bytes": slabs * g.slab_bytes, "segment_slots": SEG, "prefer": "ring", "ring_only_wait_s": 0.3,
            "allow_unlocked": False, "max_session_fraction": 0.5, "ring_low_fraction": 0.5}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def model() -> ToyModel:
    return make_model()


@pytest.fixture
def store(tmp_path: Path) -> Any:
    s = MaskStore.open(tmp_path / "store", StaticKeyProvider(os.urandom(32)), low_water_slots=0, fsync=False,
                       min_free_bytes=0, allow_no_external_counter=True, min_soundness_bits=0)
    yield s
    s.close()


def _source(store: Any, model: ToyModel, **kw: Any) -> TieredMaskSource:
    src = TieredMaskSource(store, _cfg(model, **kw), min_soundness_bits=0, seeds=SeedRegistry())
    src.attach_model("toy", model.fingerprint, RING, model.specs)
    return src


def _publish(src: TieredMaskSource, model: ToyModel, n: int = 1) -> dict[str, bytes]:
    mr = src.ring_for(model.fingerprint, RING)
    assert mr is not None
    out = {}
    for _ in range(n):
        seed = os.urandom(32)
        out[write_segment(mr, model, SEG, ring=RING, seed=seed, chunk_slots=4)] = seed
    return out


def _check_rows(res: Any, seeds: dict[str, bytes], model: ToyModel) -> None:
    slots = [(e.segment_id, e.start + i) for e in res.extents for i in range(e.count)]
    for s in model.specs:
        key, pos = s.id.key, 0
        while res.remaining(key):
            n = min(5, res.rows_until_boundary(key))
            r, wr = res.take(key, n)
            for i in range(n):
                sid, slot = slots[pos + i]
                exp = SeedExpander(seeds[sid]).ring_uniform_rows(s.in_features, RING.k, slot, 1, "r", key)
                assert torch.equal(r[i : i + 1], exp)
            assert torch.equal(wr, model.wr(key, r, RING))
            pos += n


def test_ring_short_falls_back_to_store(store: Any, model: ToyModel) -> None:
    src = _source(store, model)
    try:
        seeds = _publish(src, model)
        for _ in range(2):
            seed = os.urandom(32)
            seeds[write_segment(store, model, 40, ring=RING, seed=seed)] = seed
        inv = src.inventory(model.fingerprint, RING)[0]
        assert inv.slots_available == SEG + 80
        a = src.reserve(model.fingerprint, RING, 10)
        assert isinstance(a, RingReservation)
        b = src.reserve(model.fingerprint, RING, 30)  # ring has 6 left: the store serves it whole
        assert src.fallback_reservations == 1 and not isinstance(b, ChainedReservation)
        other = make_model(fingerprint="cd" * 32, seed=1)
        src.attach_model("other", other.fingerprint, RING, other.specs)  # a second model: its own series
        m = src.metrics()
        key = (model.fingerprint, RING.name)
        assert m["pgw_masks_ring_fallback_reservations_total"][key] == 1
        assert m["pgw_masks_ring_fallback_reservations_total"][(other.fingerprint, RING.name)] == 0
        assert m["pgw_masks_available_slots"][key] == SEG + 80 - 40
        assert m["pgw_masks_ring_capacity_slots"][key] == 4 * SEG
        assert m["pgw_masks_ring_locked_bytes"][()] > 0
        for res in (a, b):
            _check_rows(res, seeds, model)
            res.close()
        st = src.status()
        assert st["ring"] and "seed" not in str(st["ring"]).replace("seed_hash", "")
    finally:
        src.close()


def test_spanning_reservation_larger_than_either_tier(store: Any, model: ToyModel) -> None:
    """Blocking issue 7: a request that fits only across both tiers is served by a chain."""
    src = _source(store, model)
    try:
        seeds = _publish(src, model, 2)  # ring: 32
        seed = os.urandom(32)
        seeds[write_segment(store, model, 20, ring=RING, seed=seed)] = seed  # store: 20
        res = src.reserve(model.fingerprint, RING, 45)
        assert isinstance(res, ChainedReservation) and res.slots == 45
        assert src.spanning_reservations == 1
        _check_rows(res, seeds, model)
        res.close()
        assert all(p.closed for p in res.parts)
        with pytest.raises(MaskExhaustedError) as ei:
            src.reserve(model.fingerprint, RING, 100)
        assert ei.value.details["available"] == src.available_slots(model.fingerprint, RING)
    finally:
        src.close()


def test_spanning_failure_closes_ring_part(store: Any, model: ToyModel, monkeypatch: Any) -> None:
    src = _source(store, model)
    try:
        _publish(src, model, 2)
        write_segment(store, model, 20, ring=RING)
        orig = store.reserve
        calls = []

        def flaky(*a: Any, **kw: Any) -> Any:
            calls.append(a)
            if len(calls) == 2:  # the store part of the chain fails
                raise RuntimeError("disk gone")
            return orig(*a, **kw)

        monkeypatch.setattr(store, "reserve", flaky)
        with pytest.raises(RuntimeError):
            src.reserve(model.fingerprint, RING, 45)
        mr = src.ring_for(model.fingerprint, RING)
        assert mr is not None and mr.counts()["open_reservations"] == 0  # the ring part was closed (burned)
    finally:
        src.close()


def test_ring_only_waits_then_exhausts_and_extensions_fail_fast(model: ToyModel) -> None:
    src = _source(None, model)
    try:
        t0 = time.monotonic()
        with pytest.raises(MaskExhaustedError):
            src.reserve(model.fingerprint, RING, 4)
        assert 0.25 <= time.monotonic() - t0 < 3
        t0 = time.monotonic()
        with pytest.raises(MaskExhaustedError):
            src.reserve_extension(model.fingerprint, RING, 4)
        assert time.monotonic() - t0 < 0.2  # no wait inside a batch step
        mr = src.ring_for(model.fingerprint, RING)
        threading.Timer(0.1, lambda: write_segment(mr, model, SEG, ring=RING, chunk_slots=4)).start()
        res = src.reserve(model.fingerprint, RING, 4)  # the publish wakes the waiting reserve
        res.close()
        with pytest.raises(ConfigError):  # ring_only: a session must fit well below capacity
            src.attach_model("toy", model.fingerprint, RING, model.specs, max_session_slots=4 * SEG)
    finally:
        src.close()


def _step(win: WindowedReservation, keys: list[str], rows: int = 1) -> None:
    """One token step as the engine does it: ensure, then take, linear by linear."""
    for k in keys:
        win.ensure(k, rows)
        left = rows
        while left:
            n, _, _, _ = win.take_checked(k, left, False)
            left -= n


def test_long_session_does_not_pin_the_ring(model: ToyModel) -> None:
    """Blocking issue 6: a long session plus many short ones; drained slabs keep coming back."""
    src = _source(None, model, slabs=4, ring_only_wait_s=0.0)
    mr = src.ring_for(model.fingerprint, RING)
    assert mr is not None
    keys = [s.id.key for s in model.specs]
    stop = threading.Event()

    def factory() -> None:
        while not stop.is_set():
            try:
                write_segment(mr, model, SEG, ring=RING, chunk_slots=4)
            except RingFull:
                time.sleep(0.002)

    t = threading.Thread(target=factory, daemon=True)
    t.start()
    try:
        while mr.available < 2 * SEG:
            time.sleep(0.01)
        fp = model.fingerprint
        win = WindowedReservation(src.reserve(fp, RING, 8), lambda n: src.reserve_extension(fp, RING, n),
                                  cap=10_000, window=8)
        wipes0 = mr.counts()["wipes"]
        for step in range(300):
            for _ in range(10):  # ensure() may fail fast while the factory catches up: retry the step
                try:
                    _step(win, keys)
                    break
                except MaskExhaustedError:
                    time.sleep(0.005)
            else:
                pytest.fail("ring never recovered")
            if step % 5 == 0:  # a short session comes and goes
                try:
                    short = src.reserve(fp, RING, 3)
                    _step(WindowedReservation(short, lambda n: src.reserve_extension(fp, RING, n), cap=3, window=3),
                          keys)
                    short.close()
                except MaskExhaustedError:
                    pass
            assert sum(not p.closed for p in win.parts) <= 2  # only the parts still being read are open
        assert win.retired >= 300 // 8 - 2
        assert mr.counts()["wipes"] - wipes0 >= 300 // SEG
        win.close()
    finally:
        stop.set()
        t.join(5)
        src.close()


def test_retire_only_at_step_start(model: ToyModel) -> None:
    """Exhausted parts close at the first linear's ensure(), never mid-step (check vectors stay live)."""
    src = _source(None, model, ring_only_wait_s=0.0)
    try:
        _publish(src, model, 2)
        fp = model.fingerprint
        keys = [s.id.key for s in model.specs]
        win = WindowedReservation(src.reserve(fp, RING, 2), lambda n: src.reserve_extension(fp, RING, n),
                                  cap=100, window=2)
        _step(win, keys, 2)  # part 0 fully used
        first = win.parts[0]
        win.ensure(keys[1], 1)  # a later linear: no retirement
        assert not first.closed
        win.ensure(keys[0], 1)  # start of the next step
        assert first.closed and win.retired == 1
        win.close()
    finally:
        src.close()


def test_masked_generation_through_tiers_is_bit_identical(tiny_pack: Any) -> None:
    """A masked generation over a window chain that crosses ring parts and store parts gives exactly the
    plain backend's tokens."""
    import tempfile

    from pgw.engine import (
        ChatTokenizer,
        Generator,
        LocalComputeClient,
        MaskedBackend,
        PlainBackend,
        SamplingParams,
        TransformerModel,
    )
    from pgw.factory import MaskGenerator
    from pgw.factory.sinks import RingSink

    with tempfile.TemporaryDirectory() as td:
        store = MaskStore.open(Path(td) / "s", StaticKeyProvider(os.urandom(32)), low_water_slots=0, fsync=False,
                               min_free_bytes=0, allow_no_external_counter=True, min_soundness_bits=0)
        src = TieredMaskSource(store, _cfg(tiny_pack, slabs=4), min_soundness_bits=0, seeds=SeedRegistry())
        try:
            src.attach_model("tiny", tiny_pack.fingerprint, RING, tiny_pack.linear_specs())
            gen = MaskGenerator(tiny_pack, RING)
            sink = RingSink(src, store=store)
            for _ in range(2):
                gen.generate_into_store(sink, SEG, chunk_slots=8, store_chunk_slots=4)
            for _ in range(2):
                gen.generate_into_store(store, 40, chunk_slots=8, store_chunk_slots=4)
            prompt = [256, 72, 105, 33, 10]
            params = SamplingParams(max_tokens=30, temperature=0, ignore_eos=True)
            fp = tiny_pack.fingerprint
            first = src.reserve(fp, RING, len(prompt) + 4)
            win = WindowedReservation(first, lambda n: src.reserve_extension(fp, RING, n), cap=len(prompt) + 30,
                                      window=4)
            tok = ChatTokenizer.from_pack(tiny_pack)
            backend = MaskedBackend(tiny_pack, None, LocalComputeClient(tiny_pack), RING, fp)
            masked = Generator(TransformerModel(tiny_pack, backend), tok).generate(prompt, params, session=win)
            plain = Generator(TransformerModel(tiny_pack, PlainBackend(tiny_pack, RING)), tok).generate(prompt, params)
            assert masked.token_ids == plain.token_ids
            kinds = [isinstance(p, RingReservation) for p in win.parts]
            assert kinds[0] and not all(kinds)  # the chain crossed from ring parts into store parts
            assert src.fallback_reservations >= 1 and win.retired >= 1
            win.close()
            backend.close()
        finally:
            src.close()
            store.close()


# ---- gateway settings and the runtime hook ---------------------------------------------------------
def test_mask_ring_settings_validation(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from pgw.gateway import GatewaySettings

    assert GatewaySettings().mask_ring.enabled is False  # off by default
    ok = GatewaySettings(mask_ring={"enabled": True, "capacity_bytes": 1 << 30, "embed_factory": True})
    assert ok.mask_ring.prefer == "ring" and ok.mask_ring.swap_policy == "refuse"
    bad = [
        {"enabled": True},  # no capacity
        {"enabled": True, "capacity_bytes": 1 << 30, "ring_only": True},  # nothing fills the ring
        {"enabled": True, "capacity_bytes": 1 << 30, "allow_unlocked": True},  # dev only
    ]
    for mr in bad:
        with pytest.raises(ValidationError):
            GatewaySettings(mask_ring=mr)
    with pytest.raises(ValidationError):
        GatewaySettings(mask_ring={"enabled": True, "capacity_bytes": 1 << 30, "embed_factory": True},
                        factory_env_file=tmp_path / "f.env")


def test_harden_process_in_subprocess() -> None:
    import subprocess
    import sys

    code = ("from pgw.maskstore.ringbuf import harden_process\nimport ctypes\nharden_process()\n"
            "print(ctypes.CDLL(None).prctl(3, 0, 0, 0, 0))")
    src = str(Path(__file__).resolve().parents[2] / "src")
    out = subprocess.run(  # noqa: S603 - our own interpreter
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": src}, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0"  # no longer dumpable


@pytest.mark.parametrize("mode", ["embed_factory", "factory_process"])
async def test_runtime_ring_only_embedded_factory(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                  monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """The gateway hook end to end: ring_only + an embedded factory (v1 threads, or v2 supervised child on
    shared sealed slabs) serve a masked chat; stop wipes (and reaps the child)."""
    import asyncio

    from pgw.engine import SamplingParams
    from pgw.gateway.runtime import InferenceRuntime
    from pgw.maskstore import tiered

    from .conftest import MODEL, InProcessService, masked_settings

    monkeypatch.setattr(tiered, "harden_process", lambda: None)  # never make the test process non-dumpable
    monkeypatch.setattr(tiered, "swap_status", lambda: {"active": [], "unprotected": []})
    monkeypatch.setenv("PGW_FACTORY_DEVICE", "cpu")
    g = ring_geometry(gw_pack.linear_specs(), RING, 1 << 30, segment_slots=64)
    settings = masked_settings(
        tiny_pack_dir, tmp_path, data_dir=tmp_path / "data", model_kw={"max_context": 64},
        mask_ring={"enabled": True, "ring_only": True, mode: True, "capacity_bytes": 6 * g.slab_bytes,
                   "segment_slots": 64, "run_dir": str(tmp_path / "run")},
    )
    svc = InProcessService(gw_pack)
    rt_all = InferenceRuntime(settings, transports={MODEL: svc.transport})
    assert isinstance(rt_all.mask_store, TieredMaskSource) and rt_all.mask_store.store is None
    await rt_all.start()
    try:
        fac = rt_all.embedded_factory
        assert fac is not None and fac.running, getattr(fac, "error", None)
        mr = next(iter(rt_all.mask_store.rings.values()))
        for _ in range(600):
            if mr.available >= 64:
                break
            await asyncio.sleep(0.05)
        rt = rt_all.get(MODEL)
        params = SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)
        await rt.chat([{"role": "user", "content": "hi"}], params)
        assert mr.counts()["reservations"] >= 1 and mr.shared == (mode == "factory_process")
        assert "ring" in rt_all.masks_status()["store"]
    finally:
        await rt_all.stop()
    assert mr.closed
    if mode == "factory_process":
        assert fac.exit_code is not None and not fac.running  # the child was stopped and reaped


def test_windows_reserved_ahead_in_background(model: ToyModel) -> None:
    src = _source(None, model, ring_only_wait_s=0.0)
    try:
        _publish(src, model, 3)
        fp = model.fingerprint
        keys = [s.id.key for s in model.specs]
        win = WindowedReservation(src.reserve(fp, RING, 4), lambda n: src.reserve_extension(fp, RING, n),
                                  cap=40, window=4, ahead=True)
        for _ in range(30):
            _step(win, keys)
            t = win._ahead_thread
            if t is not None:
                t.join(5)
        assert win.ahead_reservations >= 5
        assert sum(int(p.slots) for p in win.parts) <= 40  # the cap holds with background extensions
        win.close()
        assert all(p.closed for p in win.parts)
    finally:
        src.close()


def test_sync_extension_counts_background_slots_against_the_cap() -> None:
    """ensure()'s synchronous extension never overshoots cap while a background one is in flight."""
    from pgw.gateway.masks import WindowedReservation

    class Part:
        def __init__(self, n: int):
            self.slots = n
            self.left = n
            self.linear_keys = ("a",)

        def remaining(self, lid: Any) -> int:
            return self.left

        def close(self) -> None:
            pass

    asked: list[int] = []

    def reserve(n: int) -> Part:
        asked.append(n)
        return Part(n)

    win = WindowedReservation(Part(4), reserve, cap=10, window=4)
    win._parts[0].left = 0
    win._pending = 4  # a background extension of 4 slots is in flight (simulated)
    win.ensure("a", 2)
    assert asked == [2]  # room = 10 - 4 reserved - 4 pending: never more than the cap in total
    win._parts[-1].left = 0
    from pgw.core.errors import MaskExhaustedError

    with pytest.raises(MaskExhaustedError):
        win.ensure("a", 1)  # 4 + 2 + 4 pending = cap
