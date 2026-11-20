"""Chat-speed WP-A4b: the TripEngine inside :class:`MaskedBackend` (native trip kernels, deferred check,
synchronous trips over the trip lane) is bit-identical to today's path and to the plain reference.

* masked (TripEngine) == plain, logits bit for bit, on every engine path: split-hop async, synchronous
  ``forward_batch``, the synchronous decode loop, 4-way prefill blocks, deferred check on and off, and
  several rings; tiny Qwen2 and Llama packs and a 2-layer synthetic pack with Qwen2.5-7B shapes;
* the provider receives byte-identical ``x_hat`` items with and without the TripEngine (same masks);
* over a real trip lane (reference lane server, UDS): same logits, decode trips go over the lane;
* the per-linear self-check falls back to today's kernels on a disagreement.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import RING_K40_A12, RING_K48_A20, RingParams, native
from pgw.core.ring import DEFAULT_RING
from pgw.core.types import LINEAR_ORDER, LinearId, LinearKind
from pgw.engine import (
    InMemoryPack,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SequenceInput,
    TransformerModel,
)
from pgw.engine import backends as B
from pgw.modelpack.loader import LinearWeights
from pgw.modelpack.manifest import ArchConfig
from tests.engine.test_model_hooks import scenario_run

pytestmark = pytest.mark.skipif(not native.native_available("trip_prepare_into"),
                                reason="pgw_fast trip kernels not built")


def _plain_ref(pack: Any, ring: RingParams = DEFAULT_RING) -> dict[str, torch.Tensor]:
    model = TransformerModel(pack, PlainBackend(pack, ring))
    model.pipeline_min_rows = 16
    try:
        return scenario_run(model, pack, "sync", False)
    finally:
        model.close()


def _same(ref: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> None:
    assert ref.keys() == got.keys()
    for k in ref:
        assert torch.equal(ref[k], got[k]), k


def _masked(pack: Any, ring: RingParams = DEFAULT_RING, client: Any = None, **kw: Any) -> TransformerModel:
    be = MaskedBackend(pack, None, client or LocalComputeClient(pack), ring, pack.fingerprint, **kw)
    be.defer_min_out = 0  # the tiny packs' linears are all small: defer every check here
    model = TransformerModel(pack, be)
    model.pipeline_min_rows = 16
    return model


@pytest.fixture(scope="module")
def refs(qwen_pack: InMemoryPack, llama_pack: InMemoryPack) -> dict[str, dict[str, torch.Tensor]]:
    return {"qwen": _plain_ref(qwen_pack), "llama": _plain_ref(llama_pack)}


# ---- defaults ------------------------------------------------------------------------------------------
def test_defaults_are_today(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("PGW_TRIP_ENGINE", "PGW_DEFERRED_CHECK", "PGW_SYNC_TRIPS"):
        monkeypatch.delenv(v, raising=False)
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint)
    assert not be.trip_engine and not be.deferred_check and not be.sync_trips
    assert not be.sync_trip_ok(1) and not be.local_linear(LinearId(0, LinearKind.QKV))
    monkeypatch.setenv("PGW_TRIP_ENGINE", "1")
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint)
    assert be.trip_engine and be.deferred_check and not be.sync_trips  # no lane client: no sync trips
    monkeypatch.setenv("PGW_DEFERRED_CHECK", "0")
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint)
    assert be.trip_engine and not be.deferred_check


def test_lane_client_turns_it_on(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.wire.lane import LaneComputeClient

    for v in ("PGW_TRIP_ENGINE", "PGW_DEFERRED_CHECK", "PGW_SYNC_TRIPS"):
        monkeypatch.delenv(v, raising=False)
    lane = LaneComputeClient(LocalComputeClient(qwen_pack))
    be = MaskedBackend(qwen_pack, None, lane, DEFAULT_RING, qwen_pack.fingerprint)
    assert be.trip_engine and be.deferred_check and be.sync_trips
    assert be.sync_trip_ok(1) and be.sync_trip_ok(8) and not be.sync_trip_ok(9) and not be.sync_trip_ok(0)
    be = MaskedBackend(qwen_pack, None, lane, DEFAULT_RING, qwen_pack.fingerprint, sync_trips=False)
    assert not be.sync_trip_ok(1)

    class Wrapped:  # the gateway's instrumented client delegates unknown attributes
        def __init__(self, inner: Any):
            self.inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

    assert MaskedBackend(qwen_pack, None, Wrapped(lane), DEFAULT_RING, qwen_pack.fingerprint).trip_engine


# ---- bit identity --------------------------------------------------------------------------------------
CASES = [
    ("async", {}),
    ("sync", {}),
    ("async", {"deferred_check": False}),
    ("sync", {"deferred_check": False}),
    ("async", {"pipeline_blocks": 4}),
    ("async", {"sync_decode": True}),
]


@pytest.mark.parametrize(("mode", "opts"), CASES, ids=lambda v: v if isinstance(v, str) else "-".join(v) or "default")
@pytest.mark.parametrize("which", ["qwen", "llama"])
def test_trip_engine_equals_plain(qwen_pack: InMemoryPack, llama_pack: InMemoryPack,
                                  refs: dict[str, dict[str, torch.Tensor]], which: str, mode: str,
                                  opts: dict[str, Any]) -> None:
    pack = qwen_pack if which == "qwen" else llama_pack
    opts = dict(opts)
    model_attrs = {k: opts.pop(k) for k in ("pipeline_blocks", "sync_decode") if k in opts}
    model = _masked(pack, trip_engine=True, sync_trips=True, **opts)
    for k, v in model_attrs.items():
        setattr(model, k, v)
    be = model.backend
    try:
        _same(refs[which], scenario_run(model, pack, mode, True))
        st = be.stats.snapshot()
        assert be.trip_engine and st["integrity_failures"] == 0
        assert (st["deferred_checks"] > 0) == be.deferred_check
        assert be.pending_checks == 0  # the last linear is checked synchronously: nothing left over
    finally:
        model.close()


@pytest.mark.parametrize("ring", [RING_K40_A12, RING_K48_A20], ids=lambda r: r.name)
def test_trip_engine_other_rings(qwen_pack: InMemoryPack, ring: RingParams) -> None:
    ref = _plain_ref(qwen_pack, ring)
    for mode in ("async", "sync"):
        model = _masked(qwen_pack, ring, trip_engine=True)
        try:
            got = scenario_run_ring(model, qwen_pack, mode, ring)
            _same(ref, got)
        finally:
            model.close()


def scenario_run_ring(model: TransformerModel, pack: Any, mode: str, ring: RingParams) -> dict[str, torch.Tensor]:
    """``scenario_run`` with reservations of ``ring`` (its helper uses the default ring)."""
    import tests.engine.test_model_hooks as H

    old = H._res
    H._res = lambda p, slots, seed: LocalMaskReservation(p, ring, slots, seed=bytes([seed]) * 32)  # type: ignore[assignment]
    try:
        return scenario_run(model, pack, mode, True)
    finally:
        H._res = old  # type: ignore[assignment]


def test_provider_bytes_identical(qwen_pack: InMemoryPack) -> None:
    """Same masks (fixed seeds) -> the provider receives exactly the same x_hat items with the TripEngine."""
    def run(trip: bool) -> list[tuple[str, tuple[int, ...], bytes]]:
        client = LocalComputeClient(qwen_pack, record=True)
        model = _masked(qwen_pack, client=client, trip_engine=trip)
        try:
            scenario_run(model, qwen_pack, "sync", True)
        finally:
            model.close()
        return [(lid, tuple(x.shape), x.numpy().tobytes()) for lid, x in client.received]

    assert run(True) == run(False)


def test_self_check_failure_falls_back(qwen_pack: InMemoryPack, refs: dict[str, dict[str, torch.Tensor]],
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def disagree(ring: RingParams, n: int, m: int, wnorm: float) -> bool:
        calls.append((n, m))
        return False

    monkeypatch.setattr(B, "_trip_kernels_agree", disagree)
    model = _masked(qwen_pack, trip_engine=True, sync_trips=True)
    be = model.backend
    try:
        _same(refs["qwen"], scenario_run(model, qwen_pack, "async", True))
        assert calls and not be.trip_engine and not be.deferred_check and not be.sync_trips
        assert be.stats.deferred_checks == 0
    finally:
        model.close()


def test_self_check_passes_on_every_ring(qwen_pack: InMemoryPack) -> None:
    for name in ("K32A12C27", "K40A12", "K48A20", "K32A8", "K24A12C21", "K56A20C40", "K16A7"):
        ring = RingParams.from_name(name)
        for n, m, wn in ((64, 96, 900.0), (3584, 4608, 4100.0), (18944, 3584, 9000.0), (7, 5, 1.0)):
            if n > ring.max_in_features():
                continue  # the backend refuses such a linear for this ring
            assert B._trip_kernels_agree(ring, n, m, wn), (name, n, m)


def test_warm_trip_engine_at_startup(qwen_pack: InMemoryPack) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       trip_engine=True)
    be.warm_weight_norms()
    shapes = {(s.in_features, s.out_features) for s in qwen_pack.linear_specs()}
    assert {(n, m) for n, m, _ in be._trip_checked} == shapes and all(be._trip_checked.values())


# ---- the trip lane ------------------------------------------------------------------------------------
class _LaneLocal(LocalComputeClient):
    """In-process service with a reference lane server (UDS) computing with the same weights."""

    def __init__(self, pack: Any, sock_dir: str):
        super().__init__(pack, record=True)
        from pgw.wire import LinearInfo
        from pgw.wire.lane import LaneStubServer

        self.config = SimpleNamespace(base_url="http://127.0.0.1:9", request_timeout_s=10.0)
        linears = tuple(LinearInfo(s.id.key, s.out_features, s.in_features) for s in pack.linear_specs())
        self.lane = LaneStubServer(self._lane_compute, fingerprint=pack.fingerprint, rings=(DEFAULT_RING.name,),
                                   linears=linears, transport="uds", path=str(Path(sock_dir) / "lane.sock"))
        self.regular: list[tuple[str, int]] = []

    def _lane_compute(self, li: Any, rows: int, ring: str, x_hat: bytes) -> bytes:
        from pgw.wire.messages import ComputeItem, ComputeRequest

        req = ComputeRequest(uuid.uuid4().hex, self.pack.fingerprint, ring, (ComputeItem(li.id, rows, x_hat),))
        return bytes(LocalComputeClient.compute_sync(self, req).items[0].y_hat)

    def compute_sync(self, request: Any) -> Any:
        self.regular.append((len(request.items), max(int(it.rows) for it in request.items)))
        return super().compute_sync(request)

    def _ticket(self, fp: str, ring: str, transports: Any, lane_versions: Any) -> Any:
        from pgw.wire.messages import LaneOpenRequest

        return self.lane.issue_ticket(LaneOpenRequest(uuid.uuid4().hex, fp, ring, tuple(transports),
                                                      tuple(lane_versions)))

    def lane_open_sync(self, fp: str, ring: str, *, transports: Any = ("uds",), lane_versions: Any = (1,)) -> Any:
        return self._ticket(fp, ring, transports, lane_versions)

    async def lane_open(self, fp: str, ring: str, *, transports: Any = ("uds",), lane_versions: Any = (1,)) -> Any:
        return self._ticket(fp, ring, transports, lane_versions)

    def close(self) -> None:
        self.lane.close()
        super().close()


@pytest.mark.parametrize("native_channel", [None, False], ids=["native", "python"])
def test_over_the_lane_equals_plain(qwen_pack: InMemoryPack, refs: dict[str, dict[str, torch.Tensor]],
                                    native_channel: bool | None) -> None:
    from pgw.wire.lane import LaneComputeClient, LaneConfig, native_channel_class

    if native_channel is None and native_channel_class() is None:
        pytest.skip("pgw_fast lane channel not built")
    with tempfile.TemporaryDirectory(prefix="pgwl") as d:
        local = _LaneLocal(qwen_pack, d)
        lane = LaneComputeClient(local, LaneConfig(native=native_channel))
        model = _masked(qwen_pack, client=lane)  # lane client: TripEngine, deferred check and sync trips on
        model.sync_decode = True
        be = model.backend
        try:
            assert be.trip_engine and be.deferred_check and be.sync_trips
            _same(refs["qwen"], scenario_run(model, qwen_pack, "async", True))
            assert lane.lane_stats.trips > 0
            lane_rows = [h["rows"] for h, _ in local.lane.trips]
            assert lane_rows and max(lane_rows) <= 8
            # the regular path only carried what the lane does not take: several items, or a prefill
            assert local.regular and all(items > 1 or rows > 8 for items, rows in local.regular)
            assert be.stats.integrity_failures == 0 and be.pending_checks == 0
        finally:
            model.close()
            lane.close()


# ---- 2-layer synthetic pack with Qwen2.5-7B shapes --------------------------------------------------------
def _synthetic_7b(layers: int = 2, vocab: int = 64) -> InMemoryPack:
    arch = ArchConfig(arch="qwen2", hidden_size=3584, intermediate_size=18944, num_layers=layers, num_heads=28,
                      num_kv_heads=4, head_dim=128, vocab_size=vocab, rms_norm_eps=1e-6, rope_theta=1e6,
                      max_position_embeddings=64, tie_word_embeddings=True)
    g = torch.Generator().manual_seed(7)
    shapes = {LinearKind.QKV: (4608, 3584), LinearKind.O: (3584, 3584), LinearKind.GATE_UP: (37888, 3584),
              LinearKind.DOWN: (3584, 18944)}
    linears = {}
    for i in range(layers):
        for kind, (m, n) in shapes.items():
            lid = LinearId(i, kind)
            w8 = torch.randint(-127, 128, (m, n), generator=g, dtype=torch.int8)
            scale = (torch.rand(m, generator=g) * 2e-3 + 5e-4).to(torch.float32)
            bias = (torch.randn(m, generator=g) * 0.02) if kind == LinearKind.QKV else None
            linears[lid.key] = LinearWeights(lid, w8, scale, bias)
    plain = {"embed_tokens": torch.randn(vocab, 3584, generator=g) * 0.05, "final_norm": torch.ones(3584)}
    for i in range(layers):
        plain[f"layers.{i}.input_norm"] = torch.rand(3584, generator=g) + 0.5
        plain[f"layers.{i}.post_attn_norm"] = torch.rand(3584, generator=g) + 0.5
    return InMemoryPack("synthetic-7b-shapes", arch, linears, plain)


def _seq7b(model: TransformerModel, res: Any, mode: str) -> list[torch.Tensor]:
    cache = model.new_cache(16)

    def run(seqs: list[SequenceInput]) -> list[Any]:
        return model.forward_batch(seqs) if mode == "sync" else asyncio.run(model.aforward_batch(seqs))

    out = [run([SequenceInput([3, 1, 4], cache, res, all_logits=True)])[0]]
    for t in (5, 9):
        out.append(run([SequenceInput([t], cache, res)])[0])
    return out


@pytest.mark.slow
def test_synthetic_7b_shapes_equal_plain() -> None:
    pack = _synthetic_7b()
    plain = TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))
    ref = _seq7b(plain, None, "sync")
    plain.close()
    client = LocalComputeClient(pack, backend="cpu_f32")
    for mode in ("sync", "async"):
        res = LocalMaskReservation(pack, DEFAULT_RING, 8, seed=bytes([3]) * 32, weights=client._weights,
                                   backend="cpu_f32")
        model = _masked(pack, client=client, trip_engine=True)
        model.sync_decode = mode == "async"
        model.backend.sync_trips = True
        model.backend.defer_min_out = 16384  # the default: at 7B shapes only gate_up (m = 37888) defers
        assert [model.backend._defer_ok(LinearId(0, k)) for k in LINEAR_ORDER] == [False, False, True, False]
        try:
            got = _seq7b(model, res, mode)
            for a, b in zip(ref, got, strict=True):
                assert torch.equal(a, b), mode
            assert model.backend.stats.deferred_checks > 0 and model.backend.pending_checks == 0
            # the constant-time window at the 7B shapes is the documented W = 2 (public)
            assert native.fast.trip_width_window(18944, DEFAULT_RING.a, DEFAULT_RING.cap,
                                                 model.backend._meta.get(LinearId(0, LinearKind.DOWN)).wnorm) <= 2
        finally:
            model.close()
            res.close()


def test_trip_engine_payload_equals_old_kernel_7b_shapes() -> None:
    """Backend level, every 7B linear shape at 1 and 3 rows: TripEngine payloads == today's kernel's."""
    rng = np.random.default_rng(1)
    ring = DEFAULT_RING
    for n, m in ((3584, 4608), (3584, 3584), (3584, 37888), (18944, 3584)):
        meta = B._MaskedMeta(torch.ones(m), None, n, m, float(np.sqrt(127.0 * 127.0 * n * 0.3)))
        for rows in (1, 3):
            x = (rng.standard_normal((rows, n)) * 2).astype(np.float32)
            r = rng.integers(0, 1 << ring.k, size=(rows, n), dtype=np.int64)
            old, _, sx_old = native.fast.quantize_mask_pack(x, r, ring.k, ring.a, ring.cap, meta.wnorm, False)
            buf = np.empty(rows * n * 4, dtype=np.uint8)
            sx = np.empty(rows)
            native.fast.trip_prepare_into(x, r, ring.k, ring.a, ring.cap, meta.wnorm, buf, sx)
            assert buf.tobytes() == bytes(old)
            assert np.array_equal(sx.view(np.uint64), np.asarray(sx_old).view(np.uint64))
