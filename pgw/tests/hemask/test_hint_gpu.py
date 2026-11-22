"""The hint's GEMMs through :class:`pgw.hemask.hint_gpu.GpuHintPlanes` give the CPU path's bits.

The class runs on any torch device; here on the CPU (``torch._int_mm`` there too), which checks the byte-plane
arithmetic (signed planes, row sums, wrapping int64 recombination). ``pytest -m gpu`` also runs it on CUDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.hemask.hint_gpu import GpuHintPlanes, cpu_planes

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("m", "n"), [(24, 64), (130, 896), (32, 4864)])
def test_planes_equal_cpu(device: str, m: int, n: int) -> None:
    rng = np.random.default_rng(m * n)
    w = rng.integers(-128, 128, size=(m, n), dtype=np.int8)
    w[0] = -128
    a = rng.integers(0, 2**63, size=(n, 96), dtype=np.uint64) << np.uint64(1)
    a[:, 3] = np.uint64(2**64 - 1)
    g = GpuHintPlanes(device)
    assert np.array_equal(g(w, a), cpu_planes(w, a))
    assert np.array_equal(g(w, a[:, :32]), cpu_planes(w, a[:, :32]))  # weights cached, other columns


@pytest.mark.parametrize("device", DEVICES)
def test_self_check_passes(device: str) -> None:
    assert GpuHintPlanes(device).self_check([(64, 128), (300, 896)], cols=64)


def test_hint_store_bits_unchanged(tiny_pack: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.hemask.backend as backend

    seed = bytes(range(32))
    monkeypatch.setattr(backend, "new_seed", lambda: seed)
    a = backend.HintStore.build(tmp_path / "cpu", tiny_pack, 8, q_bits=56)
    b = backend.HintStore.build(tmp_path / "dev", tiny_pack, 8, q_bits=56, planes_fn=GpuHintPlanes("cpu"))
    files = sorted(p.name for p in a.path.glob("*.npy"))
    assert files and files == sorted(p.name for p in b.path.glob("*.npy"))
    for f in files:
        assert np.array_equal(np.load(a.path / f), np.load(b.path / f)), f
    assert a.meta["row_norms"] == b.meta["row_norms"]


def test_setup_job_uses_the_device_only_when_asked(tiny_pack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.gateway.setup_jobs import HintBuildJob

    monkeypatch.delenv("PGW_HINT_DEVICE", raising=False)
    assert HintBuildJob._gpu_planes(tiny_pack) is None
    monkeypatch.setenv("PGW_HINT_DEVICE", "cpu")  # "cpu" means the default CPU path
    assert HintBuildJob._gpu_planes(tiny_pack) is None
    monkeypatch.setenv("PGW_HINT_DEVICE", "cuda")
    got = HintBuildJob._gpu_planes(tiny_pack)
    assert (got is not None) == torch.cuda.is_available()
