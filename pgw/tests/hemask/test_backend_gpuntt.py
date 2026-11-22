"""HEMaskBackend with the GPU NTT engine (``encrypt``/``decrypt="gpu_ntt"``): start-up cross-check against the CPU
paths, exact ``W8 r`` end to end, CPU fallback on a failed self-test, a mismatch or an error in a GPU call.

The engine's eager kernels run here on CPU tensors (``device="cpu"``); the Triton kernels are covered by
``test_gpuntt.py`` (interpreter, and CUDA when present).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pgw.hemask import gpuntt as G
from pgw.hemask import ntt
from pgw.hemask.backend import HEMaskBackend
from pgw.hemask.lwe import DEGREE

from .test_native_sampling import _Hints, _Pack, _Provider, _ref_wr, _ring_for


def _weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {"a": rng.integers(-127, 128, size=(40, 96)).astype(np.int8),
            "b": np.where(rng.random((24, 160)) < 0.5, -127, 127).astype(np.int8)}


@pytest.mark.parametrize("q_bits", [56, 64])
@pytest.mark.parametrize("modes", [("gpu_ntt", "gpu_ntt"), ("ntt_pre", "gpu_ntt"), ("gpu_ntt", "auto")])
def test_backend_gpu_ntt_exact(q_bits: int, modes: tuple[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    enc, dec = modes
    rng = np.random.default_rng(q_bits + 7)
    weights = _weights(rng)
    ring = _ring_for(q_bits)
    provider = _Provider(weights, q_bits)
    be = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, groups_per_call=2,
                       max_inflight=2, groups_per_pass=3, encrypt=enc, decrypt=dec, device="cpu")
    assert be._gpu is not None and be._gpu_enc == (enc == "gpu_ntt") and be._gpu_dec == (dec == "gpu_ntt")
    assert be.engine.startswith("gpu_ntt:cpu")
    for key, rows in (("a", 3 * DEGREE + 3), ("b", 77)):
        w = weights[key]
        r = rng.integers(0, 1 << ring.k, size=(rows, w.shape[1]), dtype=np.int64)
        got = be.matmul(key, torch.from_numpy(r))
        assert np.array_equal(got.numpy(), _ref_wr(w.astype(np.int64), r, ring.k))
    assert (be.decrypt_paths.get("gpu_ntt", 0) > 0) == (dec == "gpu_ntt")
    assert (be.decrypt_paths.get("gpu_ntt_encrypt", 0) > 0) == (enc == "gpu_ntt")

    # an error in a GPU call (e.g. out of device memory): that call on the CPU, same bits, CPU from then on
    def _boom(*a: object, **k: object) -> None:
        raise RuntimeError("device out of memory")

    monkeypatch.setattr(be._gpu, "decrypt_into", _boom)
    monkeypatch.setattr(be._gpu, "encrypt", _boom)
    r = rng.integers(0, 1 << ring.k, size=(77, 160), dtype=np.int64)
    got = be.matmul("b", torch.from_numpy(r))
    assert np.array_equal(got.numpy(), _ref_wr(weights["b"].astype(np.int64), r, ring.k))
    assert not be._gpu_enc and not be._gpu_dec

    # pipelined jobs (the factory's submit path) give the same bits
    be2 = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, groups_per_call=2,
                        max_inflight=2, groups_per_pass=3, encrypt=enc, decrypt=dec, device="cpu",
                        pipeline_depth=4)
    jobs = []
    for key, rows in (("a", 2 * DEGREE + 5), ("b", 300)):
        r = rng.integers(0, 1 << ring.k, size=(rows, weights[key].shape[1]), dtype=np.int64)
        jobs.append((key, r, be2.submit(key, torch.from_numpy(r))))
    for key, r, job in jobs:
        assert np.array_equal(job.result().numpy(), _ref_wr(weights[key].astype(np.int64), r, ring.k))
    assert be2._gpu_dec == (dec == "gpu_ntt")

    # a failed self-test: the CPU paths from the start
    monkeypatch.setattr(G, "_TESTED", {})
    monkeypatch.setattr(G, "self_test", lambda *a, **k: False)
    fb = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, encrypt=enc, decrypt=dec,
                       device="cpu")
    assert fb._gpu is None and not fb._gpu_enc and not fb._gpu_dec
    assert fb.encrypt_mode == (("ntt_pre" if ntt.available() else "dense") if enc == "gpu_ntt" else enc)
    assert fb.decrypt_mode == "auto"


@pytest.mark.parametrize("what", ["decrypt", "encrypt"])
def test_backend_gpu_ntt_cross_check_refuses_mismatch(what: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine that differs from the CPU path in one bit is not used."""
    rng = np.random.default_rng(3)
    weights = {"a": rng.integers(-127, 128, size=(40, 96)).astype(np.int8)}
    ring = _ring_for(56)
    if what == "decrypt":
        orig = G.GpuNTTHE.decrypt_into

        def _off(self, key, hint, items, out, delta_bits, k):  # type: ignore[no-untyped-def]
            orig(self, key, hint, items, out, delta_bits, k)
            out[0, 0] ^= 1

        monkeypatch.setattr(G.GpuNTTHE, "decrypt_into", _off)
    else:
        orig_e = G.GpuNTTHE.encrypt

        def _off_e(self, r, *, _seeds=None):  # type: ignore[no-untyped-def]
            cts = orig_e(self, r, _seeds=_seeds)
            cts[0].b = cts[0].b.copy()
            cts[0].b[0, 0] ^= np.uint64(1)
            return cts

        monkeypatch.setattr(G.GpuNTTHE, "encrypt", _off_e)
    be = HEMaskBackend(_Pack(weights), ring, _Hints(weights, 56), _Provider(weights, 56), encrypt="gpu_ntt",
                       decrypt="gpu_ntt", device="cpu")
    assert be._gpu is None and not be._gpu_dec and not be._gpu_enc


def test_factory_setting_accepts_gpu_ntt(tmp_path: Path) -> None:
    from pgw.factory.config import FactorySettings

    s = FactorySettings(pack_path=tmp_path)
    assert s.he_encrypt == "dense" and s.he_decrypt == "auto"  # off by default
    s = FactorySettings(pack_path=tmp_path, he_encrypt="gpu_ntt", he_decrypt="gpu_ntt")
    assert (s.he_encrypt, s.he_decrypt) == ("gpu_ntt", "gpu_ntt")


def test_os_thread_name(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from pgw.core.proc_tuning import os_thread_name

    monkeypatch.delenv("PGW_THREAD_NAMES", raising=False)
    assert os_thread_name("x") is False  # off by default
    monkeypatch.setenv("PGW_THREAD_NAMES", "1")
    got: list[str] = []

    def body() -> None:
        os_thread_name()
        got.append(Path(f"/proc/self/task/{threading.get_native_id()}/comm").read_text().strip())

    t = threading.Thread(target=body, name="pgw-hemask-decrypt_0")
    t.start()
    t.join()
    assert got == ["he-decrypt_0"]
