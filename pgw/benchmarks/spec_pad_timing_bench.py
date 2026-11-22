#!/usr/bin/env python3
"""Gateway turnaround of a padded speculation call: real rows vs padding rows vs a dummy call (L3).

The provider sees when each request of a pass arrives, so the gateway's local work between a response and
the next request (finish of one linear, float step, prepare of the next) must not depend on how many rows
of the pass are real. This measures prepare and finish of one remote linear (``0.gate_up`` and ``0.down`` of
the pack: the linears a ``local_linears=attention`` gateway sends) for R rows split as

* ``real``: R real rows (``meta[PAD_ROWS] = 0``),
* ``1+pad``: 1 real row and R - 1 padding rows,
* ``dummy``: a dummy call (all R rows padding; strict dummy passes),

interleaved, median of ``--reps``. Masks come from a reservation that expands ``r`` with the store's
ChaCha20 stream reader and copies precomputed ``W r`` rows (the store's work per take, without disk I/O).

Usage::

    PYTHONPATH=src python benchmarks/spec_pad_timing_bench.py --pack P [--rs 2,4,8] [--reps 60]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class _PrecomputedReservation:
    """``take`` = ChaCha20 ``r`` rows (as the store derives them) + a copy of precomputed ``W r`` rows."""

    def __init__(self, inner: Any, pack: Any, lids: list[Any], rows: int):
        import torch

        from pgw.core.prng import SeedExpander

        self.inner = inner
        self.key = os.urandom(32)
        self.SeedExpander = SeedExpander
        self.wr: dict[str, torch.Tensor] = {}
        self.pos: dict[str, int] = {}
        self.rows = rows
        self.ring = inner.ring
        self.n = {lid.key: int(inner._spec(lid).in_features) for lid in lids}
        for lid in lids:  # W r for this reservation's own r rows (int64 is exact: |W r| < 2^63 before the mod)
            r = SeedExpander.ring_uniform_rows_with_key(self.key, self.n[lid.key], self.ring.k, 0, rows)
            w = pack.load_linear(lid).w8.to(torch.int64)
            self.wr[lid.key] = (r @ w.t()) & self.ring.mask
            self.pos[lid.key] = 0

    def remaining(self, lid: Any) -> int:
        return 1 << 30

    def rows_until_boundary(self, lid: Any) -> int:
        return 1 << 30

    def take(self, lid: Any, rows: int) -> Any:
        k = lid.key
        p = self.pos[k] % (self.rows - rows + 1)
        self.pos[k] += rows
        r = self.SeedExpander.ring_uniform_rows_with_key(self.key, self.n[k], self.ring.k, p, rows)
        return r, self.wr[k][p:p + rows].clone()

    def checks(self, lid: Any) -> Any:
        return self.inner.checks(lid)

    def close(self) -> None:
        self.inner.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--rs", default="2,4,8")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch

    from pgw.core.ring import RingParams
    from pgw.core.types import LinearId, LinearKind
    from pgw.engine import LinearCall, LocalComputeClient, LocalMaskReservation, MaskedBackend
    from pgw.engine.backends import DUMMY, PAD_ROWS
    from pgw.modelpack import ModelPack

    ring = RingParams.from_name(args.ring)
    pack = ModelPack.open(args.pack, verify="fingerprint")
    lids = [LinearId(0, LinearKind.GATE_UP), LinearId(0, LinearKind.DOWN)]
    client = LocalComputeClient(pack)
    be = MaskedBackend(pack, None, client, ring, pack.fingerprint)
    res = _PrecomputedReservation(LocalMaskReservation(pack, ring, 64), pack, lids, 64)
    torch.manual_seed(0)
    out: dict[str, Any] = {"pack": str(args.pack), "ring": ring.name, "reps": args.reps, "rows": {}}
    for r in (int(x) for x in args.rs.split(",")):
        times: dict[str, dict[str, list[float]]] = {}
        for _ in range(args.reps):
            for lid in lids:
                n = res.n[lid.key]
                x = torch.randn(r, n) * 2
                for kind, call in (("real", LinearCall(lid, x, res, {PAD_ROWS: 0})),
                                   ("1+pad", LinearCall(lid, x[:1], res, {PAD_ROWS: r - 1})),
                                   ("dummy", LinearCall(lid, x, res, {DUMMY: True}))):
                    t0 = time.perf_counter()
                    prep = be.prepare_calls([call])
                    t1 = time.perf_counter()
                    resp = client.compute_sync(prep.request)
                    t2 = time.perf_counter()
                    be.finish_prepared(prep, resp, t2 - t1)
                    t3 = time.perf_counter()
                    d = times.setdefault(f"{lid.kind.value}", {}).setdefault(kind, [])
                    d.append((t1 - t0) * 1e6 + (t3 - t2) * 1e6)
        row = {lk: {k: round(statistics.median(v), 1) for k, v in kinds.items()} for lk, kinds in times.items()}
        out["rows"][str(r)] = row
        print(f"R={r} gateway prepare+finish us (median): {json.dumps(row)}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
