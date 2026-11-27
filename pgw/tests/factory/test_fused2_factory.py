"""The fused2 mask factory in production wiring (opt-in): scheduler refill, mask-service check, encrypted digit split.

* ``schedule = "fused2"``: :func:`pgw.factory.runtime.build_scheduler` runs the fused2 generator, keeps the fused2
  inventory (fused2 lane fingerprint, K56 ring) between low water and target, segments are fused2 pair-mask
  segments a fused2 reservation takes; the mask-service start-up check; defaults off;
* encrypted digit split (:mod:`pgw.factory.fused2_he`): K56 products as independent uniform digits on the existing
  encrypted factories (mlwe v2 at K40 / 14-bit digits; pks-v3 at K32 / 8-bit digits) equal the plain products and
  the plain generator's segments bit for bit; every provider request is an ordinary maskgen item of public shape
  (ciphertexts only, the standard wire); the service resolves fused2 linears (``PGW_SERVICE_FUSED2_MASKGEN``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import ConfigError
from pgw.core.prng import SeedExpander
from pgw.core.ring import RingParams
from pgw.factory.fused2_gen import Fused2MaskGenerator, reference_wr
from pgw.factory.fused2_he import (
    DigitSplitProducts,
    Fused2HEPack,
    TopDigitProducts,
    digit_bound_ok,
    digit_plan,
    he_key,
    recombine_digits,
    split_digits,
    split_plan,
)

RING = RingParams(56, 12, 53)


@pytest.fixture(scope="module")
def pack(tiny_pack_dir: Path) -> Any:
    from pgw.modelpack import ModelPack

    return ModelPack.open(tiny_pack_dir)


@pytest.fixture(scope="module")
def f2w(pack: Any) -> Any:
    from pgw.engine.fused2 import Fused2Weights, migration_multipliers

    return Fused2Weights(pack, None, multipliers=migration_multipliers(pack, 8))


def _settings(tiny_pack_dir: Path, tmp_path: Path, **kw: Any) -> Any:
    from pgw.factory.config import FactorySettings

    base = {"pack_path": tiny_pack_dir, "ring": RING.name, "schedule": "fused2", "device": "cpu",
            "segment_slots": 32, "low_water_slots": 40, "target_slots": 96, "state_dir": tmp_path / "state"}
    return FactorySettings(**{**base, **kw})


# ---- scheduler ------------------------------------------------------------------------------------------------
def test_scheduler_refills_fused2_inventory(tiny_pack_dir: Path, tmp_path: Path, f2w: Any) -> None:
    from pgw.factory.runtime import build_scheduler
    from pgw.maskstore import MaskStore, StaticKeyProvider
    from pgw.service.fused2_lane import fused2_fingerprint

    store = MaskStore.open(tmp_path / "store", StaticKeyProvider(os.urandom(32)), low_water_slots=0)
    try:
        s = _settings(tiny_pack_dir, tmp_path)
        with build_scheduler(s, store) as sched:
            fp = fused2_fingerprint(f2w, RING)
            assert list(sched.generators) == [(fp, RING.name)]  # the fused2 lane fingerprint, the K56 ring
            plan = sched.plan()
            assert plan and plan[0].fingerprint == fp and plan[0].available == 0
            ids = sched.run_once()
            assert ids and store.available_slots(fp, RING) >= 96
            assert sched.run_once() == []  # at target: nothing to do
            st = sched.status()
            assert st["models"][0]["fingerprint"] == fp and st["models"][0]["available_slots"] >= 96
            # the segments are fused2 pair-mask segments: a fused2 reservation takes them, a standard one refuses
            with store.reserve(fp, RING, 70, schedule="fused2") as res:
                assert res.schedule == "fused2"
            assert store.available_slots(fp, RING) < s.low_water_slots
            assert sched.run_once()  # below low water again: refilled
            assert store.available_slots(fp, RING) >= 96
    finally:
        store.close()


def test_scheduler_throttle_and_stop(tiny_pack_dir: Path, tmp_path: Path, f2w: Any) -> None:
    from pgw.factory.governor import FactoryStopped, Throttle
    from pgw.maskstore import MaskStore, StaticKeyProvider

    store = MaskStore.open(tmp_path / "store", StaticKeyProvider(os.urandom(32)), low_water_slots=0)
    try:
        gen = Fused2MaskGenerator(f2w, RING)
        thr = Throttle()
        calls: list[float] = []
        real = thr.account
        thr.account = lambda work, slots: (calls.append(slots), real(work, slots))  # type: ignore[method-assign]
        gen.generate_into_store(store, 8, chunk_slots=4, throttle=thr, checks=None)
        assert abs(sum(calls) - 8) < 1e-9  # slot-equivalents add up to the segment's slots
        thr.stop_event.set()
        with pytest.raises(FactoryStopped):
            gen.generate_into_store(store, 8, throttle=thr)
        assert store.available_slots(gen.fingerprint, RING) == 8  # the stopped segment was aborted
    finally:
        store.close()


def test_mask_service_check_and_defaults(tiny_pack_dir: Path, tmp_path: Path) -> None:
    from pydantic import ValidationError

    from pgw.factory.config import FactorySettings
    from pgw.factory.runtime import build_generators, open_packs

    s = _settings(tiny_pack_dir, tmp_path, mask_service=True, mask_service_allow_cpu=True)
    gens = build_generators(s, open_packs(s))
    (g,) = gens.values()
    rep = g.mask_service_check.to_dict()
    assert rep["ok"] and rep["check_field"] == "p61" and rep["wr_elements_compared"] > 0
    # defaults: the standard schedule; fused2 refuses settings it does not implement
    assert FactorySettings(pack_path=tiny_pack_dir).schedule == "standard"
    for bad in ({"ring": "K56A12"}, {"masked_head": True}, {"skip_linear_kinds": "qkv"},
                {"check_field": "p31"}):
        with pytest.raises(ValidationError):
            _settings(tiny_pack_dir, tmp_path, **bad)


# ---- digit split: arithmetic ---------------------------------------------------------------------------------
class _ExactHE:
    """A stand-in encrypted backend returning the exact ``W x mod 2^K_he`` (isolates the digit arithmetic)."""

    def __init__(self, w: dict[str, torch.Tensor], k_he: int):
        self.w, self.k_he = w, k_he
        self.seen: list[tuple[str, int, int]] = []

    def matmul(self, key: str, x: torch.Tensor) -> torch.Tensor:
        assert int(x.min()) >= 0 and int(x.max()) < 1 << self.k_he  # a valid K_he mask row
        self.seen.append((key, int(x.shape[0]), int(x.shape[1])))
        return torch.from_numpy(reference_wr(x, self.w[key], self.k_he).view(np.int64))


@pytest.mark.parametrize(("k_he", "b"), [(40, 14), (32, 8)])
def test_digit_split_recombination_is_exact(k_he: int, b: int) -> None:
    g = torch.Generator().manual_seed(k_he)
    n = 4096 if b == 14 else 2048
    w = {"x-f2": torch.randint(-128, 128, (37, n), dtype=torch.int8, generator=g)}
    w["x-f2"][0] = -128  # extreme rows: the bound is tight-ish, the centering must still be exact
    w["x-f2"][1] = 127
    r = SeedExpander(os.urandom(32)).ring_uniform_rows(n, 56, 0, 9, "r", "x")
    r[0] = (1 << 56) - 1  # every digit at its maximum
    r[1] = 0
    he = _ExactHE(w, k_he)
    dsp = DigitSplitProducts(he, k=56, k_he=k_he, digit_bits=b, shapes={"x/f2": (37, n)})
    got = dsp.matmul("x/f2", r.clone())
    assert np.array_equal(got.numpy().view(np.uint64), reference_wr(r, w["x-f2"], 56))
    assert dsp.digits == -(-56 // b) and len(he.seen) == dsp.digits
    assert all(s == ("x-f2", 9, n) for s in he.seen)  # every digit request has the public shape of a mask
    ds = split_digits(r, 56, b)
    assert all(int(d.max()) < 1 << b for d in ds)
    assert torch.equal(sum(d << (b * i) for i, d in enumerate(ds)), r)  # r = sum 2^(b i) d_i
    # the bound: refused where 128 * 2^b * n would exceed 2^(K_he - 1)
    assert digit_bound_ok(29568, 14, 40) and digit_bound_ok(29568, 8, 32) and not digit_bound_ok(1 << 19, 14, 40)
    with pytest.raises(ConfigError):
        DigitSplitProducts(he, k=56, k_he=k_he, digit_bits=b, shapes={"x/f2": (37, 1 << 20)})
    # recombination of hand-made centered values
    ys = [torch.tensor([[(1 << k_he) - 1]]), torch.tensor([[1]])]  # -1, then +1 * 2^b
    assert int(recombine_digits(ys, 56, k_he, b)[0, 0]) == (1 << b) - 1


@pytest.mark.parametrize(("k_he", "n", "products"), [(32, 5120, 3), (32, 27648, 4), (32, 29568, 4), (40, 29568, 2),
                                                     (32, 96, 3)])
def test_top_digit_plan_is_exact(k_he: int, n: int, products: int) -> None:
    """The top-digit plan: balanced exact low digits plus one top digit mod 2^K_he; ``W r mod 2^56`` bit for bit at
    the bound's widest digits, with extreme weight rows and extreme masks."""
    plan = digit_plan(n, 56, k_he)
    assert len(plan) == products and plan[-1] == (56 - k_he, k_he, False)
    assert sum(w for _s, w, ex in plan if ex) == 56 - k_he
    b = plan[0][1]
    assert 128 * (1 << (b - 1)) * n < 1 << (k_he - 1)  # exact balanced digits
    assert b == 56 - k_he or 128 * (1 << b) * n >= 1 << (k_he - 1)  # and the widest such (or all low bits at once)
    g = torch.Generator().manual_seed(n)
    w = {"x-f2": torch.randint(-128, 128, (19, n), dtype=torch.int8, generator=g)}
    w["x-f2"][0] = -128
    w["x-f2"][1] = 127
    rows = SeedExpander(os.urandom(32)).ring_uniform_rows(n, 56, 0, 6, "r", "x")
    rows[0] = (1 << 56) - 1
    rows[1] = 0
    rows[2] = (1 << 55) | ((1 << 23) + (1 << (b - 1)))  # balanced carries into the top digit
    rows[3, : n // 2] = (1 << 56) - 1
    for stack in (False, True):
        he = _ExactHE(w, k_he)
        top = TopDigitProducts(he, k=56, k_he=k_he, shapes={"x/f2": (19, n)}, stack=stack)
        got = top.matmul("x/f2", rows.clone())
        assert np.array_equal(got.numpy().view(np.uint64), reference_wr(rows, w["x-f2"], 56))
        assert top.stats["digit_products"] == products
        # stacked: one call of products x 6 mask rows; else one call of 6 rows per plane
        assert he.seen == ([("x-f2", 6 * products, n)] if stack else [("x-f2", 6, n)] * products)
    ds = split_plan(rows, plan, k_he)
    assert all(int(d.min()) >= 0 and int(d.max()) < 1 << k_he for d in ds)  # every plane a valid K_he mask row
    rec = torch.zeros_like(rows)
    for (sh, wd, ex), d in zip(plan, ds, strict=True):
        dd = torch.where(d >= 1 << (wd - 1), d - (1 << k_he), d) if ex else d
        rec = (rec + (dd << sh)) & ((1 << 56) - 1)
    assert torch.equal(rec, rows)  # r = sum 2^s d mod 2^56


# ---- digit split: the real encrypted factories ------------------------------------------------------------------
class _V2Provider:
    """The compute service's v2 maskgen product over the service's own kernels for the fused2 linears
    (:class:`pgw.service.fused2_lane.Fused2MaskgenModel`): ``W8 b mod 2^64`` on ciphertexts."""

    def __init__(self, model: Any):
        self.model = model
        self.requests: list[tuple[str, tuple[int, ...]]] = []

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        self.requests.append((linear, tuple(b.shape)))
        return np.asarray(self.model.kernel(linear).mulmod64(np.ascontiguousarray(b)), dtype=np.uint64)


@pytest.fixture(scope="module")
def maskgen_model(f2w: Any, monkeypatch_module: Any = None) -> Any:
    from pgw.service.fused2_lane import Fused2LaneEntry, Fused2MaskgenModel

    return Fused2MaskgenModel(Fused2LaneEntry.from_weights(f2w, RING, device="cpu", max_rows=8))


def _plain_and_he_segments(f2w: Any, he: Any, tmp_path: Path) -> tuple[Any, Any, Any]:
    from pgw.maskstore import MaskStore, StaticKeyProvider

    seed = os.urandom(32)
    stores = []
    for name, h in (("plain", None), ("he", he)):
        st = MaskStore.open(tmp_path / name, StaticKeyProvider(os.urandom(32)), low_water_slots=0)
        gen = Fused2MaskGenerator(f2w, RING, he=h, self_check_rows=-1, chunk_rows=5)
        gen.generate_into_store(st, 7, seed=seed)
        stores.append((st, gen))
    return stores[0], stores[1], seed


def _assert_same_segments(a: Any, b: Any, f2w: Any) -> None:
    (sa, ga), (sb, gb) = a, b
    fp = ga.fingerprint
    assert fp == gb.fingerprint
    with sa.reserve(fp, RING, 7, schedule="fused2") as ra, sb.reserve(fp, RING, 7, schedule="fused2") as rb:
        for lin in f2w.linears():
            na, r1, w1, c1 = ra.take_checked(lin.key, 7, True)
            nb, r2, w2, c2 = rb.take_checked(lin.key, 7, True)
            assert na == nb == 7 and torch.equal(r1, r2)
            assert torch.equal(w1, w2)  # W r mod 2^56 bit for bit
            assert np.array_equal(np.asarray(c1[1]), np.asarray(c2[1])) and np.array_equal(np.asarray(c1[2]),
                                                                                         np.asarray(c2[2]))
            assert np.array_equal(w1.numpy().view(np.uint64), reference_wr(r1, lin.w8, 56))
    sa.close()
    sb.close()


def test_digit_split_v2_equals_plain_bit_for_bit(f2w: Any, maskgen_model: Any, tmp_path: Path) -> None:
    from pgw.hemask.backend import HEMaskBackend, HintStore
    from pgw.hemask.lwe import HEParams
    from pgw.service.fused2_lane import fused2_fingerprint

    fp = fused2_fingerprint(f2w, RING)
    adapter = Fused2HEPack(f2w, fp)
    q_bits = HEParams(k=40, rank=16).q
    hints = HintStore.build(tmp_path / "hints", adapter, 16, q_bits=q_bits)
    prov = _V2Provider(maskgen_model)
    be = HEMaskBackend(adapter, RingParams.from_name("K40A12"), hints, prov, groups_per_call=1, device=None)
    shapes = {lin.key: (lin.m, lin.n) for lin in f2w.linears()}
    he = DigitSplitProducts(be, k=56, k_he=40, digit_bits=14, shapes=shapes, scheme="mlwe")
    plain, enc, _ = _plain_and_he_segments(f2w, he, tmp_path)
    _assert_same_segments(plain, enc, f2w)
    # the provider's view: per linear and chunk, 4 ordinary maskgen products (one per 14-bit digit plane), each
    # (n, 256 x groups) ciphertext columns of a linear of public shape; nothing else
    n_lin = len(f2w.linears())
    chunks = 2  # 7 slots in chunks of 5
    assert len(prov.requests) == n_lin * chunks * 4
    dims = maskgen_model.linear_dims
    for name, shape in prov.requests:
        assert name in dims and shape[0] == dims[name][1] and shape[1] % 256 == 0
    assert he.stats["digit_products"] == n_lin * chunks * 4


def test_digit_split_pks_v3_equals_plain_bit_for_bit(f2w: Any, maskgen_model: Any, tmp_path: Path) -> None:
    pytest.importorskip("pgw_ntt")
    from pgw.hemask.pks_backend import InProcessPKSProvider, PKSHintStore, PKSMaskBackend, RemotePKS
    from pgw.service.fused2_lane import fused2_fingerprint

    fp = fused2_fingerprint(f2w, RING)
    adapter = Fused2HEPack(f2w, fp)
    try:
        store = PKSHintStore.build(tmp_path / "pks", adapter)
    except ConfigError as exc:  # pragma: no cover - shapes
        pytest.skip(str(exc))
    kern = {he_key(lin.key): maskgen_model.kernel(he_key(lin.key)) for lin in f2w.linears()}
    inproc = InProcessPKSProvider(adapter, kernels=kern)
    seen: list[int] = []

    def post(body: bytes, rid: str, decode: Any) -> Any:
        seen.append(len(body))
        return inproc(body, rid, decode)

    remote = RemotePKS(post, fp, {k: kk.out_features for k, kk in kern.items()}, store.a_seed)
    be = PKSMaskBackend(adapter, RingParams.from_name("K32A12C27"), store, remote, groups_per_call=1)
    shapes = {lin.key: (lin.m, lin.n) for lin in f2w.linears()}
    he = DigitSplitProducts(be, k=56, k_he=32, digit_bits=8, shapes=shapes, scheme="pks-v3")
    plain, enc, _ = _plain_and_he_segments(f2w, he, tmp_path)
    _assert_same_segments(plain, enc, f2w)
    assert len(seen) == len(f2w.linears()) * 2 * 7  # 7 digit planes per chunk, each one ordinary v3 request


def test_service_resolves_fused2_maskgen(f2w: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.hemask.lwe import matmul_mod64
    from pgw.service.fused2_lane import Fused2LaneEntry, fused2_maskgen_models

    e = Fused2LaneEntry.from_weights(f2w, RING, device="cpu", max_rows=8)
    assert fused2_maskgen_models({e.fingerprint: e}) == {}  # off by default
    monkeypatch.setenv("PGW_SERVICE_FUSED2_MASKGEN", "1")
    (m,) = fused2_maskgen_models({e.fingerprint: e}).values()
    assert set(m.linear_dims) == {he_key(lin.key) for lin in f2w.linears()}
    lin = f2w.gu[0]
    b = np.frombuffer(os.urandom(8 * lin.n * 256), dtype=np.uint64).reshape(lin.n, 256)
    got = np.asarray(m.kernel(he_key(lin.key)).mulmod64(b), dtype=np.uint64)
    assert np.array_equal(got, matmul_mod64(lin.w8.numpy(), b))


def test_mask_service_cli_check_only_fused2(tiny_pack_dir: Path, f2w: Any) -> None:
    import json

    from click.testing import CliRunner

    from pgw.factory.cli import factory_group
    from pgw.service.fused2_lane import fused2_fingerprint

    r = CliRunner().invoke(factory_group, ["mask-service", "--pack", str(tiny_pack_dir), "--ring", RING.name,
                                           "--device", "cpu", "--schedule", "fused2", "--check-only", "--allow-cpu",
                                           "--json"])
    assert r.exit_code == 0, r.output
    (rep,) = json.loads(r.output)
    assert rep["ok"] and rep["fingerprint"] == fused2_fingerprint(f2w, RING) and rep["check_field"] == "p61"


def test_encrypted_fused2_checks_every_row_with_enough_vectors(f2w: Any) -> None:
    """With an encrypted backend (an untrusted provider chooses its errors) the fused2 factory checks every row
    (a sampled check lets a tampered row through) with private vectors for 40 bits (K56A12C53: 1 bit each)."""
    from pgw.core.freivalds import private_check_vectors_for

    class _He:
        def matmul(self, key: str, r: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not called
            raise AssertionError

    gen = Fused2MaskGenerator(f2w, RING, he=_He(), self_check_rows=2)
    assert gen.self_check_rows == -1
    lin = next(iter(f2w.linears()))
    us, _usw = gen._private(lin.key)
    assert us.shape[0] == private_check_vectors_for(gen.ring, lin.n) >= 2
    plain = Fused2MaskGenerator(f2w, RING, self_check_rows=2)
    assert plain.self_check_rows == 2 and plain._private(lin.key)[0].shape[0] == 2


def test_top_digits_pks_v3_equals_plain_bit_for_bit(f2w: Any, maskgen_model: Any, tmp_path: Path) -> None:
    """The top-digit plan on the real pks-v3 factory: the same segments as plain generation with 3-4 ordinary v3
    requests per linear and chunk instead of 7."""
    pytest.importorskip("pgw_ntt")
    from pgw.hemask.pks_backend import InProcessPKSProvider, PKSHintStore, PKSMaskBackend, RemotePKS
    from pgw.service.fused2_lane import fused2_fingerprint

    fp = fused2_fingerprint(f2w, RING)
    adapter = Fused2HEPack(f2w, fp)
    try:
        store = PKSHintStore.build(tmp_path / "pks", adapter)
    except ConfigError as exc:  # pragma: no cover - shapes
        pytest.skip(str(exc))
    kern = {he_key(lin.key): maskgen_model.kernel(he_key(lin.key)) for lin in f2w.linears()}
    inproc = InProcessPKSProvider(adapter, kernels=kern)
    seen: list[int] = []

    def post(body: bytes, rid: str, decode: Any) -> Any:
        seen.append(len(body))
        return inproc(body, rid, decode)

    remote = RemotePKS(post, fp, {k: kk.out_features for k, kk in kern.items()}, store.a_seed)
    be = PKSMaskBackend(adapter, RingParams.from_name("K32A12C27"), store, remote, groups_per_call=1)
    shapes = {lin.key: (lin.m, lin.n) for lin in f2w.linears()}
    he = TopDigitProducts(be, k=56, k_he=32, shapes=shapes, scheme="pks-v3")
    plain, enc, _ = _plain_and_he_segments(f2w, he, tmp_path)
    _assert_same_segments(plain, enc, f2w)
    assert all(len(p) in (3, 4) for p in he.plans.values())
    # stacked planes: chunks of 5 and 2 slots x 3-4 planes fit one 256-slot group: ONE ordinary v3 request per linear
    # and chunk (the digit split above sends 7)
    assert len(seen) == 2 * len(he.plans)
