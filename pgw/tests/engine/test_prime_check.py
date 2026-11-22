"""Prime-field Freivalds check (opt-in segment format ``check_field = "p31"``, SPEC 5.1).

* the native kernels are exact (input and output halves against Python integers, extreme values, every ISA
  build) and catch single-entry, structured and adversarial (within the range cap) tampering;
* a model on prime-format reservations gives the plain reference's logits bit for bit;
* tampered results are refused (IntegrityError, never an output);
* defaults stay off (backend, factory, header bytes), a gateway without ``prime_check`` refuses prime
  segments, legacy ring consumers never receive prime vectors, and both formats run side by side;
* factory -> store -> reservation round trip of the new segment format.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import native
from pgw.core.errors import ConfigError, IntegrityError, MaskStoreError
from pgw.core.freivalds import (
    P31,
    check_soundness_bits,
    is_prime_token,
    prime_check,
    prime_check_vectors,
    prime_output_half,
    prime_precompute,
    prime_soundness_bits,
    soundness_bits,
)
from pgw.core.ring import DEFAULT_RING, RingParams
from pgw.engine import InMemoryPack, LocalComputeClient, MaskedBackend, TransformerModel
from pgw.engine.backends import LinearCall, PlainBackend
from pgw.engine.local import LocalMaskReservation
from pgw.engine.model import SequenceInput

pytestmark = pytest.mark.skipif(not native.native_available("trip_prime_output"),
                                reason="pgw_fast prime kernels not built")
F = native.fast if native.native_available("trip_prime_output") else None
RING = RingParams.from_name("K32A12C27")


# ---- soundness bounds ----------------------------------------------------------------------------------
def test_soundness_bounds_at_7b_shapes() -> None:
    for n in (3584, 18944):
        ring_bits = soundness_bits(RING, n, 7)
        prime_bits = prime_soundness_bits(RING, n, 1)
        assert ring_bits == 21 and 30.99 < prime_bits < 31  # one prime vector beats seven ring vectors
        assert prime_soundness_bits(RING, n, 2) > 61.99
        assert check_soundness_bits(RING, n, 1, "p31") == prime_bits
        assert check_soundness_bits(RING, n, 7, None) == ring_bits
    # 2B >= p: refused (0 bits): an error of exactly p would vanish mod p
    assert prime_soundness_bits(RingParams.from_name("K32A8"), 152064, 1) == 0.0
    assert prime_soundness_bits(RingParams.from_name("K40A12"), 3584, 1) == 0.0  # > 32-bit rings: no


# ---- native kernels: exactness ---------------------------------------------------------------------------
def _setup(n: int, m: int, rows: int, seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    w8 = torch.from_numpy(rng.integers(-127, 128, (m, n), dtype=np.int8))
    u = prime_check_vectors(bytes([seed]) * 32, 1, m, "0.qkv")
    uw = prime_precompute(u, w8)
    x = rng.standard_normal((rows, n)).astype(np.float32)
    r = rng.integers(0, 2**32, (rows, n), dtype=np.int64)
    wn = float(np.sqrt((w8.numpy().astype(np.int64) ** 2).sum(1).max()))
    return {"w8": w8, "u": u.numpy().astype(np.uint32), "uw": uw.numpy().astype(np.uint32), "x": x, "r": r, "wn": wn}


def _prepare(d: dict[str, Any], threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rows, n = d["x"].shape
    buf = np.empty(rows * n * 4, np.uint8)
    sx = np.empty(rows)
    xh = np.empty((rows, n), np.int64)
    rhs = np.empty((rows, 1), np.uint64)
    F.trip_prepare_into(d["x"], d["r"], 32, RING.a, RING.cap, d["wn"], buf, sx, xh, d["uw"], rhs, threads, field=P31)
    xq = ((xh - d["r"] + 2**31) % 2**32) - 2**31
    # same payload bytes as without the check
    buf2 = np.empty_like(buf)
    F.trip_prepare_into(d["x"], d["r"], 32, RING.a, RING.cap, d["wn"], buf2, np.empty(rows), None, None, None, threads)
    assert bytes(buf) == bytes(buf2)
    return xq, rhs


def _reply(yq: np.ndarray, wr: np.ndarray) -> bytes:
    return ((yq + wr.astype(np.int64)) % 2**32).astype("<u4").tobytes()


@pytest.mark.parametrize("threads", [1, 0])
def test_input_half_exact(threads: int) -> None:
    d = _setup(3584, 64, 40 if threads == 0 else 3)
    xq, rhs = _prepare(d, threads)
    ref = np.array([[sum(int(a) * int(b) for a, b in zip(d["uw"][0], row, strict=True)) % P31] for row in xq])
    assert np.array_equal(ref.astype(np.uint64), rhs)


def test_output_half_extremes() -> None:
    rows, m = 4, 5000
    rng = np.random.default_rng(3)
    u = rng.integers(0, P31, (1, m), dtype=np.int64).astype(np.uint32)
    yq = np.stack([np.full(m, -(2**31)), np.full(m, 2**31 - 1), rng.integers(-(2**31), 2**31, m),
                   np.zeros(m, np.int64)]).astype(np.int64)
    wr = rng.integers(0, 2**32, (rows, m), dtype=np.int64)
    rhs = np.array([[sum(int(a) * int(b) for a, b in zip(u[0], row, strict=True)) % P31] for row in yq], np.uint64)
    for w in (wr, wr.astype(np.uint32), wr.astype(np.uint64)):
        assert F.trip_prime_output(_reply(yq, wr), w, u, rhs, 32, 2**31) == (0, 0)
    assert np.array_equal(prime_output_half(yq, u.astype(np.int64)).astype(np.uint64), rhs)
    # the range check is part of the verdict
    assert F.trip_prime_output(_reply(yq, wr), wr, u, rhs, 32, 2**27)[0] == 2


@pytest.mark.parametrize("isa", ["scalar", "avx2", "auto"])
def test_isa_builds_agree(isa: str) -> None:
    """The AVX-512, AVX2 and portable builds give the same verdicts (subprocess: the ISA is chosen once)."""
    code = (
        "import numpy as np, pgw_fast as f\n"
        "rng=np.random.default_rng(5); m=7001; p=2**31-1\n"
        "u=rng.integers(0,p,(2,m),dtype=np.int64).astype(np.uint32)\n"
        "yq=rng.integers(-2**27,2**27,(3,m)); wr=rng.integers(0,2**32,(3,m),dtype=np.int64)\n"
        "rhs=np.array([[sum(int(a)*int(b) for a,b in zip(uu,row))%p for uu in u] for row in yq],np.uint64)\n"
        "y=((yq+wr)%2**32).astype('<u4').tobytes()\n"
        "assert f.trip_prime_output(y,wr,u,rhs,32,2**27)==(0,0)\n"
        "b=bytearray(y); b[4*(m+9)]^=1\n"
        "assert f.trip_prime_output(bytes(b),wr,u,rhs,32,2**27)==(1,1)\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    if isa != "auto":
        env["PGW_PRIME_ISA"] = isa
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)  # noqa: S603
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


# ---- native kernels: soundness (tampering) --------------------------------------------------------------
def test_kernel_catches_tampering() -> None:
    n, m, rows = 3584, 4608, 2
    d = _setup(n, m, rows, seed=7)
    xq, rhs = _prepare(d)
    w = d["w8"].numpy().astype(np.int64)
    yq = xq @ w.T
    bound = 1 << 27
    assert np.abs(yq).max() <= bound
    wr = np.random.default_rng(8).integers(0, 2**32, (rows, m), dtype=np.int64).astype(np.uint32)
    u = d["u"]
    assert F.trip_prime_output(_reply(yq, wr), wr, u, rhs, 32, bound) == (0, 0)
    cases: list[np.ndarray] = []
    for j, e in ((0, 1), (m - 1, -1), (17, 2**20), (100, 2**26)):  # single entries, small and large
        t = yq.copy()
        t[1, j] += e
        cases.append(t)
    t = yq.copy()
    t[0] += 1  # structured: the same offset on every entry
    cases.append(t)
    t = yq.copy()
    t[0] += 3 * w[:, 5]  # structured: the product of a different input (x + 3 e_5)
    cases.append(t)
    t = yq.copy()
    t[1] *= -1  # sign flip of a whole row (|yq| unchanged: within the cap)
    cases.append(t)
    t = yq.copy()
    t[0, ::2] += 2**25  # a power-of-two pattern (the ring check's weak direction), within the cap
    cases.append(t)
    rng = np.random.default_rng(9)
    for _ in range(20):  # adversarial within the cap: random wrong outputs anywhere in [-B, B]
        t = yq.copy()
        idx = rng.integers(0, m, 3)
        t[0, idx] = rng.integers(-bound, bound + 1, 3)
        if np.array_equal(t, yq):
            continue
        cases.append(t)
    for t in cases:
        assert np.abs(t).max() <= bound
        st, bad = F.trip_prime_output(_reply(t, wr), wr, u, rhs, 32, bound)
        assert st == 1 and bad == int((t != yq).any(axis=1).sum())
        assert not prime_check(xq, t, u.astype(np.int64), d["uw"].astype(np.int64)).all()
    # an error of exactly p per entry would pass mod p: it is outside the cap, so the range check refuses it
    t = yq.copy()
    t[0, 3] += P31
    assert F.trip_prime_output(_reply(t, wr), wr, u, rhs, 32, bound)[0] == 2
    # all-zero vectors verify nothing
    assert F.trip_prime_output(_reply(yq, wr), wr, np.zeros_like(u), rhs, 32, bound)[0] == 3


# ---- model level ---------------------------------------------------------------------------------------
def _ids(n: int, s: int) -> list[int]:
    return [(s * 7 + i * 13) % 200 + 3 for i in range(n)]


def _run(pack: Any, backend: Any, sess: Any) -> list[torch.Tensor]:
    model = TransformerModel(pack, backend)
    try:
        out = []
        cache = model.new_cache(64)
        res = sess(32, 1)
        out.append(model.forward_batch([SequenceInput(_ids(9, 1), cache, res, all_logits=True)])[0])
        for i in range(3):
            out.append(model.forward_batch([SequenceInput([11 + i], cache, res)])[0])
        c2 = model.new_cache(64)
        res2 = sess(32, 2)
        out.append(model.forward_batch([SequenceInput(_ids(5, 2), c2, res2), SequenceInput([7], cache, res)])[0])
        return [t if isinstance(t, torch.Tensor) else torch.stack([t[i] for i in range(len(t))]) for t in out]
    finally:
        model.close()


def _prime_res(pack: Any) -> Any:
    return lambda slots, s: LocalMaskReservation(pack, DEFAULT_RING, slots, seed=bytes([s]) * 32, checks=1,
                                                 check_field="p31")


def _ring_res(pack: Any) -> Any:
    return lambda slots, s: LocalMaskReservation(pack, DEFAULT_RING, slots, seed=bytes([s + 50]) * 32, checks=7)


@pytest.mark.parametrize("fast", [False, True])
def test_prime_model_exact(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch, fast: bool) -> None:
    monkeypatch.setenv("PGW_TRIP_FAST", "1" if fast else "0")
    ref = _run(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING), lambda n, s: None)
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       prime_check=True, min_soundness_bits=20)
    assert be.trip_engine and be.prime_check
    be.defer_min_out = 0  # every ring check would be deferred: the prime one still is not
    got = _run(qwen_pack, be, _prime_res(qwen_pack))
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    assert be.stats.integrity_failures == 0 and be.stats.freivalds_checked > 0
    assert be.pending_checks == 0 and be.stats.deferred_checks == 0  # the prime check is never deferred


def test_both_formats_interoperate(qwen_pack: InMemoryPack) -> None:
    """One backend with prime_check on serves prime and ring reservations (old segments keep the old check)."""
    ref = _run(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING), lambda n, s: None)
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       prime_check=True, min_soundness_bits=20)
    mixed = lambda slots, s: (_prime_res(qwen_pack) if s == 1 else _ring_res(qwen_pack))(slots, s)  # noqa: E731
    got = _run(qwen_pack, be, mixed)
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    # and the ring format alone, on the same backend
    got = _run(qwen_pack, be, _ring_res(qwen_pack))
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("kind", ["single", "offset", "consistent"])
def test_prime_model_tamper_refused(qwen_pack: InMemoryPack, kind: str) -> None:
    calls = [0]
    w_first: dict[str, torch.Tensor] = {}

    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        calls[0] += 1
        if calls[0] != 3:
            return y
        y = y.clone()
        if kind == "single":
            y[0, 0] += 1
        elif kind == "offset":
            y[0] += 12345
        else:  # W (x + 2 e_0): a well-formed product of the wrong input
            w = w_first.setdefault(lid, qwen_pack.load_linear(__import__("pgw.core.types", fromlist=["LinearId"])
                                                                  .LinearId.parse(lid)).w8.to(torch.int64))
            y[0] += 2 * w[:, 0]
        return y

    failures: list[Any] = []
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack, tamper=tamper), DEFAULT_RING,
                       qwen_pack.fingerprint, prime_check=True, on_integrity_failure=failures.append)
    with pytest.raises(IntegrityError):
        _run(qwen_pack, be, _prime_res(qwen_pack))
    assert len(failures) == 1 and be.stats.integrity_failures == 1


# ---- defaults and fail-closed behaviour ------------------------------------------------------------------
def test_defaults_off(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.factory.config import FactorySettings
    from pgw.gateway.config import GatewaySettings

    monkeypatch.delenv("PGW_PRIME_CHECK", raising=False)
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint)
    assert not be.prime_check
    assert FactorySettings.model_fields["check_field"].default == "ring"
    assert GatewaySettings.model_fields["prime_check"].default is False


def test_gateway_without_prime_check_refuses(qwen_pack: InMemoryPack) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       trip_engine=True, min_soundness_bits=20)
    spec = qwen_pack.linear_specs()[0]
    res = _prime_res(qwen_pack)(4, 3)
    with pytest.raises(MaskStoreError, match="prime"):
        be.forward_many([LinearCall(spec.id, torch.randn(1, spec.in_features), res)])
    assert res.remaining(spec.id) == 4  # refused before any row was taken


def test_ring_consumers_never_get_prime_vectors(qwen_pack: InMemoryPack) -> None:
    res = _prime_res(qwen_pack)(4, 4)
    spec = qwen_pack.linear_specs()[0]
    with pytest.raises(ConfigError):
        res.checks(spec.id)
    tok, u, uw = res.checks_compact(spec.id)
    assert is_prime_token(tok) and u.dtype == np.uint32 and int(u.max()) < P31 and int(uw.max()) < P31
    # a non-TripEngine path refuses prime extents (fail closed) instead of running the ring check on them
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       prime_check=True)
    be.trip_engine = False
    with pytest.raises(ConfigError, match="prime"):
        be.forward_many([LinearCall(spec.id, torch.randn(1, spec.in_features), res)])


# ---- segment format: factory -> store -> reservation ------------------------------------------------------
def test_store_round_trip(qwen_pack: InMemoryPack) -> None:
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter
    from pgw.maskstore.format import SegmentHeader, encode_header

    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        store = MaskStore.open(work / "store", FileKeyProvider(work / "kek", create=True), fsync=False,
                               counter=FileMonotonicCounter(work / "ctr", fsync=False))
        try:
            gen = MaskGenerator(qwen_pack, DEFAULT_RING, check_field="p31")
            assert gen.checks_for() == 1
            gen.generate_into_store(store, 8, chunk_slots=8, factory_id="t")
            MaskGenerator(qwen_pack, DEFAULT_RING).generate_into_store(store, 8, chunk_slots=8, factory_id="t")
            hdrs = {}
            for p in sorted((work / "store").rglob("*.pgwm")):
                from pgw.maskstore.format import read_header

                with p.open("rb") as f:
                    h, raw = read_header(f)
                hdrs[h.check_field] = (h, raw)
            assert set(hdrs) == {None, "p31"}
            h0, raw0 = hdrs[None]
            assert b"check_field" not in raw0 and encode_header(h0) == raw0  # old headers: same bytes
            h1, raw1 = hdrs["p31"]
            assert b'"check_field":"p31"' in raw1 and h1.checks == 1
            with pytest.raises(ValueError):
                SegmentHeader.model_validate({**h1.model_dump(mode="json"), "check_field": "p61"})
            spec = qwen_pack.linear_specs()[0]
            seen = set()
            for _ in range(2):
                res = store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 8)
                cf = res.check_field(spec.id)
                seen.add(cf)
                tok, u, uw = res.checks_compact(spec.id)
                if cf == "p31":
                    assert is_prime_token(tok)
                    want = prime_precompute(torch.from_numpy(u.astype(np.int64)), qwen_pack.load_linear(spec.id).w8)
                    assert np.array_equal(want.numpy(), uw.astype(np.int64))
                    with pytest.raises(ConfigError):
                        res.checks(spec.id)
                    with pytest.raises(ConfigError):
                        res.checks_u64(spec.id)
                    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING,
                                       qwen_pack.fingerprint, prime_check=True, min_soundness_bits=20)
                else:
                    assert not is_prime_token(tok)
                    res.checks(spec.id)
                    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING,
                                       qwen_pack.fingerprint, trip_engine=True, min_soundness_bits=20)
                x = torch.randn(2, spec.in_features)
                ref = PlainBackend(qwen_pack, DEFAULT_RING).forward_many([LinearCall(spec.id, x, None)])[0]
                assert torch.equal(be.forward_many([LinearCall(spec.id, x, res)])[0], ref)
                res.close()
            assert seen == {None, "p31"}
        finally:
            store.close()
