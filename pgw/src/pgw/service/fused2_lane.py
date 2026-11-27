"""fused2 on the service trip lane: pair items (opt-in, default off).

A fused2 weight set (``research/plans/fused2-two-trips-per-layer.md``) is served as its own lane model whose
table holds one **item** per provider trip of a pass, in trip order::

    0.qkv/f2                     single  (n = n_qkv,      m = m_qkv)
    0.o+0.gate_up/f2             pair    (n = n_o + m_o,  m = m_o + m_gu)
    0.down+1.qkv/f2              pair    (n = n_d + m_d,  m = m_d + m_qkv)
    ...
    (L-1).down                   single

A **pair item** is the op ``(x_hat, R_hat) -> (H_hat, Z_hat)`` with ``H_hat = R_hat + hmul * (W1 x_hat)`` and
``Z_hat = W2' H_hat`` (mod ``2^K``). It rides the lane unchanged: the table entry's ``n`` / ``m`` are the
concatenated widths, the request body is ``pack(x_hat) || pack(R_hat)`` (``rows * n1`` then ``rows * m1``
elements of ``K/8`` bytes) and the OK reply ``pack(H_hat) || pack(Z_hat)``. Same headers, tickets, admission,
deadlines, UDS / shared-memory / TLS transports and fallback statuses as every lane call; the provider still sees
only uniformly masked ring elements of public shapes (exactly what the socket harness carried).

Weights are public: ``w8`` of ``o`` / ``down``, their multipliers ``hmul`` and the folded ``W2'`` (``qkv/f2``,
``gate_up/f2``), from :meth:`pgw.modelpack.fused2_weights.Fused2Weights.save` (``provider_only=True``) or derived
here from the provider's own pack (:meth:`Fused2LaneEntry.from_pack`: the int8 fold with pack-time migration
multipliers).
The entry's lane fingerprint (:func:`fused2_fingerprint`) binds the base pack, the ring and every item's weights;
a gateway pins it like a pack fingerprint.

Execution: :class:`pgw.core.ring64.PairOp` per item (Triton kernels; on CUDA one graph per (item, rows) that the
native lane replays with no Python, see :func:`capture_fused2_graphs`), self-checked at start-up against the
Python-int reference (:func:`pgw.core.ring64.self_check`) and each captured graph through the native serving path
against the eager op.

Opt-in: ``PGW_SERVICE_FUSED2_WEIGHTS=/path/a.pt[,/path/b.pt]`` (provider-only weight files) or
``PGW_SERVICE_FUSED2_FROM_PACK=<migration bits>`` (derive from every loaded pack), with
``PGW_SERVICE_FUSED2_RING`` (default ``K56A12C53``). Without them nothing changes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, ProtocolError
from pgw.core.ring import RingParams
from pgw.core.ring64 import PairOp

__all__ = [
    "ENV_FROM_PACK",
    "ENV_RING",
    "ENV_WEIGHTS",
    "Fused2Item",
    "Fused2LaneEntry",
    "fused2_fingerprint",
    "fused2_items",
    "load_fused2_entries",
]

log = logging.getLogger(__name__)

ENV_WEIGHTS = "PGW_SERVICE_FUSED2_WEIGHTS"
ENV_FROM_PACK = "PGW_SERVICE_FUSED2_FROM_PACK"
ENV_RING = "PGW_SERVICE_FUSED2_RING"
#: With ``PGW_SERVICE_FUSED2_FROM_PACK``: ``off`` (default) the 2L + 1 item table (head on the gateway), ``on`` the
#: head-fold table only (item table v2: the client library's ``schedule = "fused"``), ``also`` both entries per pack.
ENV_HEAD = "PGW_SERVICE_FUSED2_HEAD"
DEFAULT_RING = "K56A12C53"
KIND_PAIR = "f2pair"
KIND_SINGLE = "f2single"


@dataclass(frozen=True)
class Fused2Item:
    """One provider trip of a fused2 pass (public)."""

    name: str  # "0.qkv/f2", "3.o+3.gate_up/f2", "(L-1).down"
    k1: str
    k2: str | None
    n1: int
    m1: int
    m2: int

    @property
    def kind(self) -> str:
        return KIND_PAIR if self.k2 is not None else KIND_SINGLE

    @property
    def n(self) -> int:
        """Lane table input width (``n1 + m1`` for a pair: ``x_hat`` then ``R_hat``)."""
        return self.n1 + (self.m1 if self.k2 is not None else 0)

    @property
    def m(self) -> int:
        """Lane table output width (``m1 + m2`` for a pair: ``H_hat`` then ``Z_hat``)."""
        return self.m1 + self.m2


def fused2_items(weights: Any) -> list[Fused2Item]:
    """The trip items of a :class:`pgw.modelpack.fused2_weights.Fused2Weights` in pass order (gateway and service
    agree)."""
    w = weights
    out = [Fused2Item(w.qkv[0].key, w.qkv[0].key, None, w.qkv[0].n, w.qkv[0].m, 0)]
    for i in range(w.num_layers):
        o, gu = w.o[i], w.gu[i]
        out.append(Fused2Item(f"{o.key}+{gu.key}", o.key, gu.key, o.n, o.m, gu.m))
        d = w.down[i]
        if i + 1 < w.num_layers:
            q = w.qkv[i + 1]
            out.append(Fused2Item(f"{d.key}+{q.key}", d.key, q.key, d.n, d.m, q.m))
        else:
            out.append(Fused2Item(d.key, d.key, None, d.n, d.m, 0))
    hd = getattr(w, "head", None)
    if hd is not None:  # head_fold (item table v2): the last down's logit rows as a (down, folded head) pair
        d = w.down[-1]
        out.append(Fused2Item(f"{d.key}+{hd.key}", d.key, hd.key, d.n, d.m, hd.m))
    return out


#: Version of the fused2 item table a lane ticket offers (``fused2_items`` field): 1 = the 2026-11-23 table (2L + 1
#: items); 2 = with the folded-head pair item ``(L-1).down+L.lm_head/f2`` appended (``head_fold``). A gateway checks
#: the whole table by name and the version, so a version-1 service is refused by a head-fold gateway (fail closed).
ITEMS_V1, ITEMS_V2 = 1, 2


def items_version(weights: Any) -> int:
    """The item-table version of a fused2 weight set (2 with the folded head)."""
    return ITEMS_V2 if getattr(weights, "head", None) is not None else ITEMS_V1


def fused2_fingerprint(weights: Any, ring: RingParams | str) -> str:
    """Lane fingerprint of a fused2 weight set: SHA-256 over the base pack fingerprint, the ring, and every
    item's public weights (``w8`` bytes and ``hmul``) in table order. 64 hex characters."""
    rp = ring if isinstance(ring, RingParams) else RingParams.from_name(ring)
    h = hashlib.sha256(b"pgw-fused2-lane-v1\0")
    h.update(str(getattr(weights, "fingerprint", "")).encode() + b"\0" + rp.name.encode() + b"\0")
    by_key = {lin.key: lin for lin in weights.all_linears()}
    for it in fused2_items(weights):
        h.update(it.name.encode() + b"\0")
        for key in (it.k1, it.k2):
            if key is None:
                continue
            lin = by_key[key]
            h.update(np.asarray(lin.w8.shape, dtype="<i8").tobytes())
            h.update(hashlib.sha256(lin.w8.contiguous().numpy().tobytes()).digest())
            hm = lin.hmul if it.k2 is not None and key == it.k1 else None
            h.update(b"\0" if hm is None else hashlib.sha256(hm.to(torch.int64).numpy().astype("<i8").tobytes()).digest())
    return h.hexdigest()


@dataclass
class Fused2LaneEntry:
    """A fused2 weight set served on the lane: its fingerprint, ring, item table and one op per item."""

    fingerprint: str
    ring: RingParams
    items: list[Fused2Item]
    ops: list[PairOp]
    device: torch.device
    base_fingerprint: str = ""
    lock: Any = field(default_factory=threading.RLock)
    #: The item table's version (:data:`ITEMS_V1` / :data:`ITEMS_V2`), offered in the lane ticket.
    items_version: int = 1
    calls: int = 0

    @classmethod
    def from_weights(cls, weights: Any, ring: RingParams | str = DEFAULT_RING, *, device: str | torch.device = "cpu",
                     max_rows: int = 8, variant: str | None = None) -> Fused2LaneEntry:
        rp = ring if isinstance(ring, RingParams) else RingParams.from_name(ring)
        if rp.k % 8 or rp.k > 64:
            raise ConfigError(f"fused2 lane: ring {rp.name} is not served (K must be a multiple of 8, <= 64)")
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:  # one spelling per device (locks and executors are per device)
            dev = torch.device("cuda", torch.cuda.current_device())
        by_key = {lin.key: lin for lin in weights.all_linears()}
        items = fused2_items(weights)
        ops = []
        for it in items:
            l1 = by_key[it.k1]
            if it.k2 is None:
                ops.append(PairOp(rp.k, l1.w8, device=dev, max_rows=max_rows, variant=variant))
            else:
                ops.append(PairOp(rp.k, l1.w8, by_key[it.k2].w8, l1.hmul, device=dev, max_rows=max_rows,
                                  variant=variant))
        return cls(fused2_fingerprint(weights, rp), rp, items, ops, dev, str(getattr(weights, "fingerprint", "")),
                   items_version=items_version(weights))

    @classmethod
    def from_pack(cls, pack: Any, migrate_bits: int = 8, ring: RingParams | str = DEFAULT_RING, *,
                  head: bool = False, **kw: Any) -> Fused2LaneEntry:
        """Derive the public fused2 weights from the provider's own pack (int8 fold, pack-time migration
        multipliers :func:`pgw.modelpack.fused2_weights.migration_multipliers`; no calibration text, no float
        checkpoint). Pure table derivation from :mod:`pgw.modelpack` (the service never imports the engine).
        ``head``: the folded LM head as the last trip's second stage (item table v2, the client library's
        ``schedule = "fused"``); the last down then takes migration multipliers too (the final norm's gain)."""
        from pgw.modelpack.fused2_weights import Fused2Weights, migration_multipliers

        mults = migration_multipliers(pack, int(migrate_bits), head=head) if migrate_bits else None
        # sigma=False: the pair op and the fingerprint use w8 and hmul only (no SVDs on the provider)
        return cls.from_weights(Fused2Weights(pack, None, multipliers=mults, sigma=False, head=head), ring, **kw)

    def table(self) -> list[tuple[int, int, str]]:
        """``[(n, m, kind)]`` for :meth:`LaneService.set_model`."""
        return [(it.n, it.m, it.kind) for it in self.items]

    def names(self) -> list[str]:
        return [it.name for it in self.items]

    def run(self, index: int, rows: int, body: bytes | bytearray | memoryview) -> bytes:
        """Eager call of item ``index`` (the lane's executor for calls without a graph)."""
        if not 0 <= index < len(self.ops):
            raise ProtocolError(f"fused2 lane: unknown item {index}")
        with self.lock:
            self.calls += 1
            return self.ops[index].run(rows, body)

    def self_check(self) -> list[str]:
        """Kernels vs the Python-int reference on small random and extreme pairs (start-up gate)."""
        from pgw.core.ring64 import self_check

        return self_check(self.device, k=self.ring.k)


def load_fused2_entries(state: Any, *, device: str | torch.device | None = None, max_rows: int = 8
                        ) -> dict[str, Fused2LaneEntry]:
    """Entries the environment asks for (empty without ``PGW_SERVICE_FUSED2_WEIGHTS`` / ``..._FROM_PACK``).

    Every entry passes :meth:`Fused2LaneEntry.self_check` or is not served (logged)."""
    files = [p for p in os.environ.get(ENV_WEIGHTS, "").split(",") if p.strip()]
    from_pack = os.environ.get(ENV_FROM_PACK, "").strip()
    if not files and not from_pack:
        return {}
    ring = RingParams.from_name(os.environ.get(ENV_RING, DEFAULT_RING))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[str, Fused2LaneEntry] = {}
    t0 = time.perf_counter()
    entries: list[Fused2LaneEntry] = []
    from pgw.modelpack.fused2_weights import Fused2Weights

    for path in files:
        entries.append(Fused2LaneEntry.from_weights(Fused2Weights.load(path.strip()), ring, device=device,
                                                    max_rows=max_rows))
    if from_pack:
        from pgw.modelpack import ModelPack

        head_mode = os.environ.get(ENV_HEAD, "off").strip().lower() or "off"
        if head_mode not in ("off", "on", "also"):
            raise ConfigError(f"{ENV_HEAD} must be off, on or also, got {head_mode!r}")
        heads = {"off": (False,), "on": (True,), "also": (False, True)}[head_mode]
        for m in state.registry.models():
            if m.pack_path is None or m.masked_head:
                continue
            pack = ModelPack.open(m.pack_path, verify="fingerprint")
            try:
                for hd in heads:
                    entries.append(Fused2LaneEntry.from_pack(pack, int(from_pack), ring, device=device,
                                                             max_rows=max_rows, head=hd))
            finally:
                pack.close()
    for e in entries:
        fails = e.self_check()
        if fails:
            log.error("fused2 lane entry %s not served: kernel self-check failed (%s)", e.fingerprint[:12], fails[:3])
            continue
        out[e.fingerprint] = e
        log.info("fused2 lane entry %s (base %s, %s, %d items) on %s", e.fingerprint[:12], e.base_fingerprint[:12],
                 e.ring.name, len(e.items), e.device)
    log.info("fused2 lane: %d entries in %.1f s", len(out), time.perf_counter() - t0)
    return out


#: Opt-in: serve encrypted maskgen (v1/v2/v3) for the fused2 linears of every lane entry under its lane fingerprint.
ENV_MASKGEN = "PGW_SERVICE_FUSED2_MASKGEN"


class Fused2MaskgenModel:
    """The fused2 linears of a lane entry as a maskgen model (``linear_dims`` / ``kernel``), for the customer's
    encrypted fused2 factory (:mod:`pgw.factory.fused2_he`: K56 masks as digits on the existing schemes).

    Linear names are the file-safe fused2 keys (``"3.gate_up-f2"``); each kernel is an ordinary
    :class:`pgw.service.kernels.ComputeKernel` over the entry's public int8 weight (the same matrix the pair op
    holds), built on first use. The provider computes exactly what a standard maskgen request asks
    (``W8 b mod 2^64`` on MLWE ciphertexts / the pks-v3 item): the requests carry ciphertexts only."""

    def __init__(self, entry: Fused2LaneEntry):
        self.fingerprint = entry.fingerprint
        self.device = entry.device
        self._w: dict[str, torch.Tensor] = {}
        for it, op in zip(entry.items, entry.ops, strict=True):
            self._w[it.k1.replace("/", "-")] = op.w1
            if it.k2 is not None:
                self._w[it.k2.replace("/", "-")] = op.w2  # type: ignore[assignment]
        self._kern: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def linear_dims(self) -> dict[str, tuple[int, int]]:
        return {k: (int(w.shape[0]), int(w.shape[1])) for k, w in self._w.items()}

    def kernel(self, linear: str) -> Any:
        from types import SimpleNamespace

        from pgw.service.kernels import ComputeKernel

        with self._lock:
            k = self._kern.get(linear)
            if k is None:
                w = self._w.get(linear)
                if w is None:
                    raise ProtocolError(f"unknown fused2 linear {linear!r}")
                spec = SimpleNamespace(id=SimpleNamespace(key=linear, kind=SimpleNamespace(value="f2")),
                                       out_features=int(w.shape[0]), in_features=int(w.shape[1]))
                k = self._kern[linear] = ComputeKernel(spec, w.cpu(), device=str(self.device))  # type: ignore[arg-type]
            return k


def fused2_maskgen_models(entries: dict[str, Fused2LaneEntry]) -> dict[str, Fused2MaskgenModel]:
    """``{lane fingerprint: Fused2MaskgenModel}`` when ``PGW_SERVICE_FUSED2_MASKGEN`` is on (else empty)."""
    if os.environ.get(ENV_MASKGEN, "0").strip().lower() not in ("1", "true", "yes", "on"):
        return {}
    return {fp: Fused2MaskgenModel(e) for fp, e in entries.items()}


def capture_fused2_graphs(mgr: Any, entry: Fused2LaneEntry, slot: int, rows_list: list[int], rep: Any
                          ) -> None:  # pragma: no cover - GPU only
    """Per (item, rows) one CUDA graph ``[H2D, unpack, gemv, epilogue, (gemv, epilogue), D2H]`` registered with
    the native lane, then replayed once through the native serving path on random and extreme bodies and
    compared byte for byte with the eager op (itself self-checked against the Python-int reference). Any
    mismatch sets ``rep.disabled`` (the caller turns graph replay off for the whole lane)."""
    from pgw.core.packing import pack_ring
    from pgw.core.ring64 import _rand_ring
    from pgw.service.gpu_exec import trip_stream
    from pgw.service.lane import _as_ptr

    dev = entry.device
    rmax = max(rows_list)
    in_max = max(op.in_bytes(rmax) for op in entry.ops)
    out_max = max(op.out_bytes(rmax) for op in entry.ops)
    gen = np.random.default_rng(0xF2)
    k = entry.ring.k
    with torch.cuda.device(dev):
        stream = trip_stream(dev)
        pool = torch.cuda.graph_pool_handle()
        h_in = torch.zeros(in_max, dtype=torch.uint8, pin_memory=True)
        h_out = torch.zeros(out_max, dtype=torch.uint8, pin_memory=True)
        mgr._graphs.extend([h_in, h_out, stream])
        for li, op in enumerate(entry.ops):
            for rows in rows_list:
                with mgr._f2_lock(entry):
                    g = op.capture(rows, h_in, h_out, stream=stream, pool=pool)
                exec_ptr = _as_ptr(g.raw_cuda_graph_exec())
                mgr._graphs.extend([g, op])
                mgr.svc.register_graph(slot, li, 0, rows, int(dev.index or 0), exec_ptr, int(stream.cuda_stream),
                                       int(h_in.data_ptr()), op.in_bytes(rows), int(h_out.data_ptr()),
                                       op.out_bytes(rows))
                rep.captured += 1
                for case in range(2):
                    if case == 0:
                        x = _rand_ring(gen, (rows, op.in_elems), k)
                    else:
                        x = torch.from_numpy(np.full((rows * op.in_elems,), (1 << k) - 1, dtype=np.uint64)
                                             .view(np.int64).reshape(rows, op.in_elems))
                    # the body layout is x rows then R rows: random bytes of the right length serve either way
                    body = pack_ring(x, k)
                    got = bytes(mgr.svc.run_graph(slot, li, 0, rows, body))
                    with mgr._f2_lock(entry):
                        want = entry.run(li, rows, body)
                    if got != want:
                        rep.disabled = f"fused2 graph replay differs from the eager op ({entry.items[li].name}, rows={rows})"
                        return
                rep.checked += 1


def serve(argv: list[str] | None = None) -> None:  # pragma: no cover - process entry point (e2e harness)
    """``python -m pgw.service.fused2_lane serve``: one fused2 weight set on the production native lane (Unix socket,
    ``PGW_SERVICE_LANE_SHM=1`` for the shared-memory transport, CUDA graphs per (item, rows) on CUDA), with no HTTP
    front end: it writes one lane ticket (JSON, mode 0600) to ``--ticket-file`` for the co-located gateway, then
    serves until killed. Opt-in harness for the fused2 e2e run (``benchmarks/fused2_e2e.py --transport lane``).
    """
    import argparse
    import asyncio
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from pgw.modelpack.fused2_weights import Fused2Weights
    from pgw.service import ModelRegistry, ServiceMetrics, ServiceSettings
    from pgw.service.lane import LaneManager

    ap = argparse.ArgumentParser(description="fused2 pair items on the native trip lane (opt-in harness)")
    ap.add_argument("serve", choices=["serve"])
    ap.add_argument("--weights", required=True, help="Fused2Weights.save(provider_only=True) file")
    ap.add_argument("--ring", default=DEFAULT_RING)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--ticket-file", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=8)
    args = ap.parse_args(argv)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    t0 = time.perf_counter()
    entry = Fused2LaneEntry.from_weights(Fused2Weights.load(args.weights), args.ring, device=dev,
                                         max_rows=args.max_rows)
    fails = entry.self_check()
    if fails:
        raise SystemExit(f"fused2 kernels failed the self-check: {fails[:3]}")
    state = SimpleNamespace(registry=ModelRegistry("cpu"), metrics=ServiceMetrics(enabled=False),
                            handler=SimpleNamespace(lane_admission=None, draining=False),
                            fused2={entry.fingerprint: entry})
    s = ServiceSettings(device="cpu", allow_unauthenticated=True, lane="uds", lane_socket=Path(args.socket),
                        lane_max_rows=args.max_rows)
    mgr = LaneManager(s, state)
    loop = asyncio.new_event_loop()
    if not mgr._start_sync(loop):
        raise SystemExit(f"lane unavailable: {mgr.unavailable_reason}")
    rep = mgr.graph_report
    t = mgr.open_ticket(None, entry.fingerprint)
    t["graphs"] = {"captured": rep.captured, "checked": rep.checked, "disabled": rep.disabled,
                   "skipped": rep.skipped, "s": round(rep.seconds, 1)}
    t["startup_s"] = round(time.perf_counter() - t0, 1)
    old = os.umask(0o177)
    try:
        Path(args.ticket_file).write_text(json.dumps(t))
    finally:
        os.umask(old)
    print(f"fused2 lane: {len(entry.items)} items, {entry.ring.name}, {dev}, graphs {rep.captured} "
          f"(checked {rep.checked}, off: {rep.disabled}), {args.socket}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":  # pragma: no cover
    serve()
