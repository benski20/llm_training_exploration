"""Client CPU and bytes per mask slot of the pks-v3 factory, by component, at real Qwen2.5 shapes.

Where ``benchmarks/pks_client_slot_bench.py`` times the HE parts only, this runs every per-slot and per-segment
step the factory and the store do for one mask slot, each with the function the production path calls:

factory, per slot (one row of every linear, the masked LM head included)
  * ``r``           the mask rows' ChaCha20 keystream (``pgw_fast.mask_rows_with_key``, as ``_CpuRing``)
  * ``encrypt``     upload encryption + reply keys (``Customer.encrypt``), ``ksk_pack`` their wire form
  * ``decrypt``     the reply decryption with the compact hint (``Customer.decrypt``; stand-in random reply)
  * ``widen``       ``PKSMaskBackend.matmul``'s uint32 -> int64 copy of ``W8 r`` (``PGW_PKS_COMPACT=0``, the path
                    before 2026-11-26; the compact default hands the decryption's uint32 buffer on: 0)
  * ``self_check``  the every-row private Freivalds check, 40-bit vectors (``generator.check_flags``)
  * ``seal``        range check + pack + AES-256-GCM + ``pwrite`` (``seal_ring_chunks``) and the segment's fsync
factory, per segment (amortised over ``--slots``)
  * ``uw``          the segment's online check vectors ``u W8 mod p`` (p31, ``prime_vecmat``; ``uw_f64``: the
                    float64 reference ``prime_precompute``, the path before 2026-11-26) or ``u W8 mod 2^32``
                    (``--check-field ring``, ``ring_vecmat``)
factory, per worker start and after every aborted segment (``priv_vw``, total seconds; amortised per slot over
one segment): the private ``v W8`` sets of the self-check (``ring_vecmat``)
use time (the gateway process)
  * ``open``        AES-256-GCM open of the slot's chunk bytes (``chunk_open_into``) + unpack
  * ``retire``      crypto-erase + zero-fill + unlink of the consumed segment (``shred_file``)

CPU seconds (``time.process_time``: all threads), best of ``--reps``, for one layer of each kind plus the masked
LM head, times the layer count. The shapes are the real ones of ``--model``; weights are random int8 (no kernel
here depends on their values). ``PGW_VECMAT_NATIVE=0`` runs the check-vector products as before this change.

    PYTHONPATH=src:. python benchmarks/pks_slot_breakdown.py --model 0.5b --slots 1024
    PGW_VECMAT_NATIVE=0 PYTHONPATH=src:. python benchmarks/pks_slot_breakdown.py --model 0.5b --uw-f64
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

#: layers and (out, in) per linear kind incl. the masked LM head
SHAPES: dict[str, tuple[int, dict[str, tuple[int, int]]]] = {
    "0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864),
                  "lm_head": (151936, 896)}),
    "7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944),
                "lm_head": (152064, 3584)}),
    "32b": (64, {"qkv": (7168, 5120), "o": (5120, 5120), "gate_up": (55296, 5120), "down": (5120, 27648),
                 "lm_head": (152064, 5120)}),
    "72b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (59136, 8192), "down": (8192, 29568),
                 "lm_head": (152064, 8192)}),
}
#: per-slot factory parts, per-segment parts, use-time parts (``priv_vw`` is reported separately)
FACTORY = ("r", "encrypt", "ksk_pack", "decrypt", "widen", "self_check", "seal", "uw")
USE_TIME = ("open", "retire")


def _best(fn: Any, reps: int) -> float:
    best = float("inf")
    for _ in range(max(1, reps)):
        c0 = time.process_time()
        fn()
        best = min(best, time.process_time() - c0)
    return best


class _Ctx:
    """Shared, once-per-run objects."""

    def __init__(self, check_field: str, uw_f64: bool):
        import torch

        from pgw.core.prng import SeedExpander, new_seed
        from pgw.core.settings import ring_from_setting
        from pgw.factory.kernels import ring_vecmat
        from pgw.hemask import pks
        from pgw.hemask.pks_native import Customer

        self.ring = ring_from_setting("K32A12C27")
        self.rng = np.random.default_rng(0)
        self.cust = Customer(pks.ks_a(new_seed()), threads=1)
        self.a_seed = new_seed()
        self.check_field = check_field
        self.uw_f64 = uw_f64
        # warm-up: torch's int8 GEMM probe and thread pools must not count against the first linear
        ring_vecmat(SeedExpander(new_seed()).ring_uniform((2, 64), 32, "v"), torch.ones((64, 64), dtype=torch.int8),
                    self.ring)


def measure_kind(ctx: _Ctx, kind: str, m: int, n: int, slots: int, reps: int, tmpdir: str) -> dict[str, float]:
    """CPU seconds of every part for one linear of shape ``(m, n)`` and a segment of ``slots`` slots."""
    import torch

    from pgw.core import native
    from pgw.core.freivalds import prime_check_vectors, prime_precompute, private_check_vectors_for
    from pgw.core.prng import SeedExpander, new_seed
    from pgw.factory.generator import check_flags
    from pgw.factory.kernels import prime_vecmat, ring_vecmat
    from pgw.hemask.pks import NR, PACK
    from pgw.hemask.pks_native import GroupSecret
    from pgw.hemask.wire import V3_Y_BITS
    from pgw.maskstore.format import chunk_aad, chunk_nonce
    from pgw.maskstore.segment_io import shred_file
    from pgw.service.pks_provider import pack_ksk

    ring, rng, cust = ctx.ring, ctx.rng, ctx.cust
    lid = f"0.{kind}"
    res: dict[str, float] = {}
    st: dict[str, Any] = {}
    w = torch.from_numpy(rng.integers(-128, 128, (m, n), dtype=np.int8))
    hint = rng.integers(0, 256, (m // PACK, 2, NR, 3), dtype=np.uint8)
    hint[..., 2] &= 0x7F
    key = bytearray(SeedExpander(new_seed()).stream_key("r", lid))

    res["r"] = _best(lambda: st.__setitem__("r", native.fast.mask_rows_with_key(key, n, 32, 0, slots)), reps)
    r = st["r"]
    res["encrypt"] = _best(lambda: st.__setitem__("enc", cust.encrypt(ctx.a_seed, lid, r)), reps)
    _, beta, secrets = st.pop("enc")
    res["ksk_pack"] = _best(lambda: pack_ksk(beta), reps)
    y = rng.integers(0, 256, len(secrets) * (m // PACK) * NR * V3_Y_BITS // 8, dtype=np.uint8).tobytes()
    best = float("inf")
    for _ in range(reps):
        sec = [GroupSecret(s.t.copy(), s.t_reply.copy(), s.trhat.copy(), s.rows) for s in secrets]
        c0 = time.process_time()
        st["wr32"] = cust.decrypt(hint, y, sec)
        best = min(best, time.process_time() - c0)
    res["decrypt"] = best
    wr32 = st.pop("wr32")

    def widen() -> None:
        o = np.empty((slots, m), dtype=np.int64)
        o[:] = wr32
        st["wr64"] = o

    compact = os.environ.get("PGW_PKS_COMPACT", "1") != "0"
    if compact:
        res["widen"] = 0.0
        wr64 = wr32.view(np.int32)  # what the generator hands on (the compact int32 form)
    else:
        res["widen"] = _best(widen, reps)
        wr64 = st["wr64"]
    c_priv = private_check_vectors_for(ring, n)
    v = SeedExpander(new_seed()).ring_uniform((c_priv, m), 32, "v")

    def priv() -> None:  # as MaskGenerator._private_check
        vw = ring_vecmat(v, w, ring)
        st["cu"] = np.ascontiguousarray(v.t().numpy().astype(np.uint32))
        st["cw"] = np.ascontiguousarray(vw.t().numpy().astype(np.uint32))

    res["priv_vw"] = _best(priv, max(1, reps // 2))
    cu, cw = st["cu"], st["cw"]
    res["self_check"] = _best(lambda: check_flags(r, wr64, cu, cw, ring.mask), reps)
    seed = new_seed()
    if ctx.check_field == "p31":
        res["uw"] = _best(lambda: prime_vecmat(prime_check_vectors(seed, 1, m, lid), w), max(1, reps // 2))
        if ctx.uw_f64:
            res["uw_f64"] = _best(lambda: prime_precompute(prime_check_vectors(seed, 1, m, lid), w), 1)
    else:
        res["uw"] = _best(lambda: ring_vecmat(SeedExpander(seed).ring_uniform((2, m), 32, "u", lid), w, ring),
                          max(1, reps // 2))
    # seal: chunks of ~1 MiB of this linear, as the store's writer
    nb = 4
    cs = max(1, (1 << 20) // (m * nb))
    path = Path(tmpdir) / f"seg-{kind}.bin"
    dek = bytearray(os.urandom(32))
    prefix, digest = os.urandom(4), os.urandom(32)
    jobs = []
    off = 0
    for i, s0 in enumerate(range(0, slots, cs)):
        rows = min(cs, slots - s0)
        jobs.append((i, off, s0, rows))
        off += rows * m * nb + 16

    def seal() -> None:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, off)
            assert native.fast.seal_ring_chunks(fd, dek, prefix, digest, wr64, 32, jobs, 0)
            os.fsync(fd)
        finally:
            os.close(fd)

    res["seal"] = _best(seal, reps)

    def open_all() -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            for i, o, _, rows in jobs:
                buf = bytearray(rows * m * nb)
                native.fast.chunk_open_into(bytes(dek), chunk_nonce(prefix, i), chunk_aad(digest, i), fd, o,
                                            rows * m * nb + 16, buf)
                native.fast.unpack_rows_u32(buf, 0, rows, m, 32)
        finally:
            os.close(fd)

    res["open"] = _best(open_all, reps)
    best = float("inf")
    for _ in range(max(1, reps // 2)):
        p2 = path.with_suffix(".ret")
        p2.write_bytes(path.read_bytes())
        c0 = time.process_time()
        shred_file(p2)
        best = min(best, time.process_time() - c0)
    res["retire"] = best
    path.unlink()
    for s in secrets:
        s.wipe()
    res["c_priv"] = c_priv
    return res


def measure(model: str, slots: int, reps: int, check_field: str, uw_f64: bool, tmpdir: str,
            kinds: list[str] | None = None) -> dict[str, Any]:
    layers, shapes = SHAPES[model]
    ctx = _Ctx(check_field, uw_f64)
    out: dict[str, Any] = {"model": model, "slots": slots, "check_field": check_field,
                           "vecmat_native": os.environ.get("PGW_VECMAT_NATIVE", "1"), "per_kind": {}}
    for kind, (m, n) in shapes.items():
        if kinds and kind not in kinds:
            continue
        mult = 1 if kind == "lm_head" else layers
        res = measure_kind(ctx, kind, m, n, slots, reps, tmpdir)
        row = {k: round(v * 1e3 * mult / slots, 4) for k, v in res.items() if k not in ("c_priv", "priv_vw")}
        row["priv_vw_s"] = round(res["priv_vw"] * mult, 3)
        row["c_priv"] = int(res["c_priv"])
        out["per_kind"][kind] = row
        print(json.dumps({kind: row}), flush=True)
    tot: dict[str, float] = {}
    for d in out["per_kind"].values():
        for k, v in d.items():
            if k != "c_priv":
                tot[k] = round(tot.get(k, 0.0) + v, 4)
    out["total"] = tot
    out["factory_ms_per_slot"] = round(sum(tot.get(k, 0.0) for k in FACTORY), 3)
    out["use_time_ms_per_slot"] = round(sum(tot.get(k, 0.0) for k in USE_TIME), 3)
    out["bytes_per_slot"] = 4 * sum(m * (1 if k == "lm_head" else layers) for k, (m, _) in shapes.items())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="0.5b", choices=sorted(SHAPES))
    ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--check-field", default="p31", choices=["p31", "ring"])
    ap.add_argument("--uw-f64", action="store_true", help="also time the float64 reference of the p31 uW")
    ap.add_argument("--kinds", nargs="*", default=None)
    ap.add_argument("--tmpdir", default=None, help="a directory on the store's disk (fsync cost)")
    ap.add_argument("--threads", type=int, default=1, help="kernel threads (CPU seconds are reported)")
    a = ap.parse_args()
    os.environ["PGW_PKS_THREADS"] = str(a.threads)
    import torch

    torch.set_num_threads(a.threads)
    with tempfile.TemporaryDirectory(dir=a.tmpdir) as td:
        res = measure(a.model, a.slots, a.reps, a.check_field, a.uw_f64, td, a.kinds)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
