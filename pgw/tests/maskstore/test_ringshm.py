"""Mask ring v2 (``pgw.maskstore.ringshm``): per-grant sealed memfds, private socketpair, child crash."""

from __future__ import annotations

import os
import signal
import socket
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A12_C27, SeedExpander, ringmm_reference
from pgw.core.errors import ConfigError, MaskStoreError
from pgw.factory import MaskGenerator
from pgw.maskstore.ringbuf import FdMapping, MaskRing, SeedRegistry, SlabState, ring_geometry
from pgw.maskstore.ringshm import RemoteRingSink, RingBroker, _Channel
from pgw.maskstore.tiered import TieredMaskSource

from ..factory.conftest import FakePack

RING = RING_K32_A12_C27
SEG = 16


@pytest.fixture
def fake_pack() -> FakePack:
    return FakePack()


def _source(pack: FakePack, slabs: int = 4) -> TieredMaskSource:
    g = ring_geometry(pack.linear_specs(), RING, 1 << 30, segment_slots=SEG)
    cfg = SimpleNamespace(capacity_bytes=slabs * g.slab_bytes, segment_slots=SEG, factory_process=True,
                          allow_unlocked=False)
    src = TieredMaskSource(None, cfg, min_soundness_bits=0, seeds=SeedRegistry())
    src.attach_model("fake", pack.fingerprint, RING, pack.linear_specs())
    return src


def _check(res: Any, pack: FakePack) -> None:
    """W r handed out equals W . r for the r the reservation regenerated from the slab seed."""
    for s in pack.linear_specs():
        key = s.id.key
        while res.remaining(key):
            _, r, wr, _ = res.take_checked(key, 5, checks=False)
            w = pack.weights[key].to(torch.int64)
            assert torch.equal(wr.to(torch.int64), torch.from_numpy(ringmm_reference(r, w, RING.k)))
            assert r.any()


def _ring(src: TieredMaskSource, pack: FakePack) -> MaskRing:
    r = src.ring_for(pack.fingerprint, RING)
    assert r is not None and r.shared
    return r


def test_in_process_protocol_masks_are_exact(fake_pack: FakePack) -> None:
    src = _source(fake_pack)
    gw, ch = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    broker = RingBroker(src)
    broker.attach(gw)
    chan = _Channel(ch)
    try:
        hello = chan.wait_hello()
        assert [m["fp"] for m in hello["models"]] == [fake_pack.fingerprint]
        assert set(hello) == {"t", "incarnation", "models"}  # no arena fd: one memfd per granted slab
        sink = RemoteRingSink(chan)
        gen = MaskGenerator(fake_pack, RING)
        seed = os.urandom(32)
        sid = gen.generate_into_store(sink, SEG, checks=2, chunk_slots=5, seed=seed, store_chunk_slots=4)
        ring = _ring(src, fake_pack)
        assert ring.counts()["published_segments"] == 1 and broker.published == 1
        res = ring.reserve(SEG)
        assert res.extents[0].segment_id == sid
        key = fake_pack.linear_specs()[0].id.key
        r, _ = res.take(key, 3)
        assert torch.equal(r, SeedExpander(seed).ring_uniform_rows(fake_pack.linear_specs()[0].in_features, RING.k,
                                                                   0, 3, "r", key))
        _check(res, fake_pack)
        res.close()
        assert ring.wait_idle()
        assert sink.levels(fake_pack.fingerprint, RING) == (0, ring.capacity_slots, ring.n_slabs, SEG)
        # a refused publish surfaces as an error in the child (duplicate seed)
        with pytest.raises(Exception, match="seed"):
            gen.generate_into_store(sink, SEG, checks=2, chunk_slots=5, seed=seed, store_chunk_slots=4)
        assert ring.wait_idle() and ring.counts()["free_slabs"] == ring.n_slabs
    finally:
        broker._stopping = True
        gw.shutdown(socket.SHUT_RDWR)
        ch.close()
        gw.close()
        src.close()


def test_publish_refused_while_writable_and_sealed_after(fake_pack: FakePack) -> None:
    src = _source(fake_pack)
    ring = _ring(src, fake_pack)
    sig = [[s.id.key, s.out_features, s.in_features] for s in fake_pack.linear_specs()]
    try:
        idx, gen, fd = ring.grant_shared(fake_pack.fingerprint, RING, sig, slots=4, checks=2, chunk_slots=4)
        m = FdMapping(os.dup(fd), ring.geometry.slab_bytes, writable=True)
        mv = m.view(0, 32)
        mv[:] = os.urandom(32)
        mv.release()
        with pytest.raises(MaskStoreError, match="writable"):  # the "child" still maps it: seal refused
            ring.publish_shared(idx, gen, slots=4, checks=2, chunk_slots=4, factory_id="t")
        m.close()
        assert ring.wait_idle() and ring.slab_states()[idx] == SlabState.FREE
        # a well-behaved fill: unmap first, then publish -> sealed; nobody can map it writable again
        idx, gen, fd = ring.grant_shared(fake_pack.fingerprint, RING, sig, slots=4, checks=2, chunk_slots=4)
        dup = os.dup(fd)
        m = FdMapping(dup, ring.geometry.slab_bytes, writable=True)
        mv = m.view(0, 32)
        mv[:] = os.urandom(32)
        mv.release()
        m.close(wipe=False)
        ring.publish_shared(idx, gen, slots=4, checks=2, chunk_slots=4, factory_id="t")
        with pytest.raises(ConfigError):
            FdMapping(dup, ring.geometry.slab_bytes, writable=True)
        os.close(dup)
        # an all-zero seed is refused
        idx, gen, _ = ring.grant_shared(fake_pack.fingerprint, RING, sig, slots=4, checks=2, chunk_slots=4)
        with pytest.raises(Exception, match="zero"):
            ring.publish_shared(idx, gen, slots=4, checks=2, chunk_slots=4, factory_id="t")
        with pytest.raises(ConfigError):
            ring.grant_shared(fake_pack.fingerprint, RING, sig[::-1], slots=4, checks=2, chunk_slots=4)
    finally:
        src.close()


_CHILD = textwrap.dedent('''
    import os, sys
    sys.path[:0] = sys.argv[1].split(os.pathsep)
    from pathlib import Path
    from factory.conftest import FakePack
    from pgw.factory import MaskGenerator
    from pgw.factory.config import FactorySettings
    from pgw.maskstore.ringshm import serve
    state = Path(sys.argv[2])
    fd = int(sys.argv[sys.argv.index("--ring-fd") + 1])
    def build(hello):
        pack = FakePack()
        st = FactorySettings(pack_path=state, state_dir=state, poll_interval_s=30.0, checks=2, chunk_slots=8,
                             segment_slots=16)
        gen = MaskGenerator(pack, "K32A12C27")
        return st, {gen.key: gen}
    sys.exit(serve(fd, build))
''')


def test_child_process_fills_ring_and_crash_is_reaped(tmp_path: Path, fake_pack: FakePack) -> None:
    src = _source(fake_pack, slabs=4)
    ring = _ring(src, fake_pack)
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    paths = os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "src")])
    broker = RingBroker(src, [sys.executable, str(script), paths, str(tmp_path / "state")])
    broker.start()
    try:
        assert broker.ready.wait(60), "child never became ready"
        assert broker.child_dumpable == 0  # PR_SET_DUMPABLE=0 in the child
        deadline = time.monotonic() + 60
        while ring.available < ring.capacity_slots and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ring.available == ring.capacity_slots
        res = ring.reserve(SEG + 5)
        _check(res, fake_pack)
        res.close()  # one slab drains -> FREE -> the child refills it
        held = ring.reserve(SEG)  # pinned through the crash
        pid = broker.pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 20
        while broker.running and time.monotonic() < deadline:
            time.sleep(0.05)
        deadline = time.monotonic() + 20
        while broker.exit_code is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert broker.exit_code == -signal.SIGKILL  # reaped by the gateway
        assert ring.wait_idle()
        assert SlabState.FILLING not in ring.slab_states()  # granted slabs of the dead child were dropped
        _check(held, fake_pack)  # published (sealed) slabs keep serving
        held.close()
    finally:
        broker.stop()
        src.close()
