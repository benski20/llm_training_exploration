"""North-star mask path: encrypted (MLWE) factory -> in-memory ring -> gateway takes, no disk.

The provider stand-in only multiplies the ciphertext words it is sent (``W8 b mod 2^64``); the factory
decrypts ``W r`` and writes it into a :class:`pgw.maskstore.ringbuf.MaskRing`. Rows taken from the
ring (sequential ``take_checked``, and the batched native claim when built) must equal those of the
local (plaintext) factory for the same seed, and the pads must be the normative ones.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import SeedExpander
from pgw.core.ring import RingParams
from pgw.factory import MaskGenerator
from pgw.factory.sinks import RingSink
from pgw.hemask import matmul_mod64
from pgw.hemask.backend import HEMaskBackend, HintStore
from pgw.maskstore import ringbuf, take_checked_many
from pgw.maskstore.ringbuf import MaskRing, SeedRegistry, ring_geometry


class _Provider:
    def __init__(self, pack: Any):
        self.w = {lw.id.key: lw.w8.numpy() for lw in pack.iter_linears()}
        self.seen: list[np.ndarray] = []

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        self.seen.append(b.copy())
        return matmul_mod64(self.w[linear], b)


def _ring(pack: Any, ring: RingParams, slots: int) -> MaskRing:
    specs = list(pack.linear_specs())
    geo = ring_geometry(specs, ring, 1 << 30, segment_slots=slots)
    return MaskRing(pack.fingerprint, ring, specs, capacity_bytes=4 * geo.slab_bytes, segment_slots=slots,
                    allow_unlocked=True, min_soundness_bits=0.0, seeds=SeedRegistry())


def _local_rows(pack: Any, ring: RingParams, seed: bytes, slots: int) -> dict[str, torch.Tensor]:
    out: dict[str, list[torch.Tensor]] = {}
    for g in MaskGenerator(pack, ring, device="cpu").generate(seed, slots, checks=2, chunk_slots=slots):
        out.setdefault(g.id.key, []).append(g.wr.to(torch.int64))
    return {k: torch.cat(v) for k, v in out.items()}


@pytest.mark.parametrize("batched", [False, True])
def test_encrypted_factory_into_ring_is_exact(tiny_pack: Any, tmp_path: Path, batched: bool,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.maskstore import store as store_mod

    if batched and not store_mod._native_claim:
        pytest.skip("pgw_fast.claim_rows_many not built")
    monkeypatch.setattr(ringbuf.RingReservation, "batched_claim", batched)
    ring = RingParams.from_name("K32A12C27")
    slots = 24
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    provider = _Provider(tiny_pack)
    he = MaskGenerator(tiny_pack, ring, device="cpu",
                       remote=HEMaskBackend(tiny_pack, ring, hints, provider, groups_per_call=2))
    mr = _ring(tiny_pack, ring, slots)
    try:
        seed = os.urandom(32)
        he.generate_into_store(RingSink({mr.key: mr}), slots, checks=2, chunk_slots=8, seed=seed)
        ref = _local_rows(tiny_pack, ring, seed, slots)
        res = mr.reserve(slots)
        exp = SeedExpander(seed)
        keys = [s.id.key for s in tiny_pack.linear_specs()]
        got: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {k: [] for k in keys}
        while any(res.remaining(k) for k in keys):
            live = [k for k in keys if res.remaining(k)]
            if batched:
                outs = take_checked_many([(res, k, 5, True) for k in live])
            else:
                outs = [res.take_checked(k, 5, True) for k in live]
            for k, o in zip(live, outs, strict=True):
                got[k].append((o[1], o[2].to(torch.int64)))
        for s in tiny_pack.linear_specs():
            k = s.id.key
            r_all = torch.cat([g[0] for g in got[k]])
            wr_all = torch.cat([g[1] for g in got[k]])
            assert torch.equal(wr_all, ref[k]), k
            assert torch.equal(r_all, exp.ring_uniform_rows(s.in_features, ring.k, 0, slots, "r", k)), k
        res.close()
        # the provider saw only q-bit ciphertext words; no seed, pad or W r left the factory
        assert provider.seen and all(int(b.max()) < (1 << 56) for b in provider.seen)
    finally:
        mr.close()
