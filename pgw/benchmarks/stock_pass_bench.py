"""Weight-free factory (``factory_weights="stream"``): disk and network of one check-stock pass, measured on a
real model
over the provider's real provisioning endpoint (loopback, in-process service), and extrapolated to 7B / 32B / 72B.

What it does (CPU only; no cloud):

1. packs the public checkpoint (a local Hugging Face snapshot) with the pinned parameters and checks the pin, then
   derives the weight-free gateway pack (what a client keeps);
2. serves the full pack's provisioning endpoint (``/pgw/v1/provision``) on loopback;
3. runs one :class:`pgw.factory.checkstock.CheckStock` pass with the SDK's settings (masked head, p31, one check vector,
   ``K32A12C27``, 40-bit private vectors) over that endpoint, counting every byte received;
4. reports the stock on disk (``du``), the bytes received, the pass time, and the pack-mode factory pack it replaces;
   then the same quantities per model from the real shapes (the formulas the measurement just validated).

    PYTHONPATH=src:. python benchmarks/stock_pass_bench.py --hf-dir <HF snapshot of Qwen2.5-0.5B-Instruct> \\
        --work /tmp/stock-bench [--segments 64 --low-water 16 --private 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Any

MODELS = {  # layers, hidden, {kind: (m, n)}; head (vocab, hidden) is the masked head (derived, not downloaded)
    "7B": (28, 3584, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "32B": (64, 5120, {"qkv": (7168, 5120), "o": (5120, 5120), "gate_up": (55296, 5120), "down": (5120, 27648)}),
    "72B": (80, 8192, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (59136, 8192), "down": (8192, 29568)}),
}
HEAD_VOCAB = 152064
KSK_BYTES = 16 * 8192 * (24 + 24 + 30 + 30) // 8  # one pks-v3 reply key per linear and 256-slot group
PRIVATE_VECTORS = 14  # 40 bits at capped K32 widths (pgw.core.freivalds.private_check_vectors_for)
SEGMENT_SLOTS = 1024  # the SDK's segment size


def pks_traffic_per_slot(shapes: list[tuple[int, int]]) -> float:
    """pks-v3 bytes up + down per mask slot (benchmarks/hemask_pks_model.py: 7 B per input feature, the reply keys,
    4.5 B per output feature), every linear including the masked head."""
    sm, sn = sum(m for m, _ in shapes), sum(n for _, n in shapes)
    return sn * 7 + len(shapes) * KSK_BYTES / 256 + sm * 4.5


def item_bytes(sn: int, words: int) -> int:
    """One stock item on disk: header (measured ~1.2 KB at 0.5B; grows with the linear list), the 32-byte seed and
    ``words`` uint32 words per input feature, 16 bytes of GCM tag per record."""
    return 1200 + 32 + 4 * words * sn


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class _Service:
    def __init__(self, settings: Any):
        self.settings, self.loop, self.stop_event = settings, asyncio.new_event_loop(), None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        from pgw.service.server import serve_async

        asyncio.set_event_loop(self.loop)

        async def main() -> None:
            self.stop_event = asyncio.Event()
            await serve_async(self.settings, shutdown_event=self.stop_event)

        self.loop.run_until_complete(main())

    def start(self, url: str) -> None:
        import httpx

        self.thread.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 300:
            try:
                if httpx.get(f"{url}/readyz", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise TimeoutError("service not ready")

    def stop(self) -> None:
        if self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(timeout=60)


def measure(a: argparse.Namespace) -> dict[str, Any]:
    from pgw.core.freivalds import private_check_vectors_for
    from pgw.core.ring import RingParams
    from pgw.factory.checkstock import CheckStock, StockLinear
    from pgw.maskstore import FileKeyProvider
    from pgw.modelpack import ModelPack, pack_checkpoint
    from pgw.modelpack.masked_head import MaskedHeadPack
    from pgw.provision.client import _install_from_full_pack, weight_source
    from pgw.provision.session import ProvisionSession
    from pgw.service.auth import hash_token
    from pgw.service.config import ServiceSettings

    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, mode=0o700)
    full_dir = work / "full"
    t0 = time.perf_counter()
    pack_checkpoint(a.hf_dir, full_dir, name=a.name, revision=a.revision, plain_dtype="source")
    full = ModelPack.open(full_dir, verify="full")
    if a.pin and full.fingerprint != a.pin:
        raise SystemExit(f"pack fingerprint {full.fingerprint[:12]} != pin {a.pin[:12]}")
    mdir = work / "client"
    mdir.mkdir(mode=0o700)
    _install_from_full_pack(full, mdir)
    from pgw.provision.pack import GatewayPack

    gp = GatewayPack.open_gateway(mdir / "pack", expected_fingerprint=full.fingerprint, verify="full")
    print(f"packed {a.name} ({full.fingerprint[:12]}) and its gateway pack in {time.perf_counter() - t0:.0f} s")
    factory_pack_w8 = sum(_du_file(full_dir / f) for f in full.manifest.linear_files())
    full.close()

    token = "stock-bench-token-0123456789abcdef"  # noqa: S105 - loopback test credential
    tok = work / "tokens.txt"
    tok.write_text(f"bench:{hash_token(token)}\n")
    port = _free_port()
    svc = _Service(ServiceSettings(host="127.0.0.1", port=port, metrics_port=None, packs=[full_dir], device="cpu",
                                   tokens_file=tok, batch_window_ms=1.0, rings=["K32A12C27"], drain_timeout_s=2.0,
                                   provision=True, warmup=False))
    url = f"http://127.0.0.1:{port}"
    svc.start(url)
    received = [0]

    class Counting(ProvisionSession):
        def request(self, method: str, path: str, **kw: Any) -> Any:
            r = super().request(method, path, **kw)
            received[0] += len(r.content) + sum(len(k) + len(v) + 4 for k, v in r.headers.items())
            return r

    try:
        ring = RingParams.from_name("K32A12C27")
        hp = MaskedHeadPack(gp)
        lins = [StockLinear(s.id.key, s.out_features, s.in_features, private_check_vectors_for(ring, s.in_features))
                for s in hp.linear_specs()]
        kp = FileKeyProvider(work / "masks.kek", create=True)
        stock = CheckStock(work / "check-stock", kp, fingerprint=hp.fingerprint, ring=ring, check_field="p31",
                           checks=1, linears=lins,
                           source=lambda: weight_source(gp, session=Counting(url, token=token, retries=0)),
                           target_segments=a.segments, low_water_segments=a.low_water, target_private=a.private,
                           low_water_private=max(1, a.private // 2), background=False)
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t1 = time.perf_counter()
        out = stock.refill()
        wall = time.perf_counter() - t1
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)  # includes the in-process service's work
        disk = _du(stock.root)
        seg_files = sorted(stock.root.glob("seg-*.pgwc"))
        prv_files = sorted(stock.root.glob("priv-*.pgwc"))
        sn = sum(lin.in_features for lin in lins)
        shapes = [(lin.out_features, lin.in_features) for lin in lins]
        res = {
            "model": a.name, "fingerprint": hp.fingerprint, "segments": out["segments"],
            "private_sets": out["private_sets"], "private_vectors": sorted({lin.private_vectors for lin in lins}),
            "pass_wall_s": round(wall, 2), "pass_cpu_s_incl_service": round(cpu, 2),
            # weights_bytes counts the streamed layer W8 only: the masked head's W8 is derived from the gateway pack
            "w8_downloaded_bytes": out["weights_bytes"], "head_w8_derived_bytes": lins[-1].out_features
            * lins[-1].in_features, "network_bytes": received[0], "stock_disk_bytes": disk,
            "pass_wall_s_per_gb_w8": round(wall / (out["weights_bytes"] / 1e9), 1),
            "segment_item_bytes": seg_files[0].stat().st_size, "private_item_bytes": prv_files[0].stat().st_size,
            "factory_pack_w8_bytes": factory_pack_w8, "gateway_pack_bytes": _du(mdir / "pack"),
            "sum_in_features": sn, "pks_traffic_per_segment": pks_traffic_per_slot(shapes) * SEGMENT_SLOTS,
        }
    finally:
        svc.stop()
    return res


def _du_file(p: Path) -> int:
    return p.stat().st_size if p.is_file() else 0


def extrapolate(m: dict[str, Any] | None, segments: int, low_water: int, private: int) -> list[dict[str, Any]]:
    per_pass = segments - low_water
    rows = []
    hdr_over = 1.0 if m is None else m["network_bytes"] / max(1, m["w8_downloaded_bytes"])
    for name, (layers, hidden, kinds) in MODELS.items():
        shapes = [mn for _ in range(layers) for mn in kinds.values()] + [(HEAD_VOCAB, hidden)]
        sn = sum(n for _, n in shapes)
        w8 = sum(mm * n for mm, n in shapes[:-1])  # the head is derived from the plain head the gateway keeps
        seg = pks_traffic_per_slot(shapes) * SEGMENT_SLOTS
        stock = segments * item_bytes(sn, 1) + private * item_bytes(sn, PRIVATE_VECTORS)
        rows.append({"model": name, "w8_per_pass_gb": w8 * hdr_over / 1e9, "pks_per_segment_gb": seg / 1e9,
                     "network_pct_default": 100 * w8 * hdr_over / per_pass / seg,
                     "network_pct_64_per_pass": 100 * w8 * hdr_over / 64 / seg, "stock_gb": stock / 1e9,
                     "factory_pack_gb": w8 / 1e9})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", help="local Hugging Face snapshot of the checkpoint (omit: extrapolation only)")
    ap.add_argument("--work", default=None, help="scratch directory (deleted at the end)")
    ap.add_argument("--name", default="qwen2.5-0.5b-instruct")
    ap.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    ap.add_argument("--pin", default="d773a502fb976d45070332cbd4222ee30bc971d14c7f9c56a317805c230c0b44")
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--low-water", type=int, default=16)
    ap.add_argument("--private", type=int, default=8)
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    m = None
    if a.hf_dir:
        if not a.work:
            raise SystemExit("--work is required with --hf-dir")
        try:
            m = measure(a)
        finally:
            if not a.keep and a.work:
                shutil.rmtree(a.work, ignore_errors=True)
        seg = m["pks_traffic_per_segment"]
        print(json.dumps(m, indent=1))
        print(f"\n0.5B measured: one pass = {m['segments']} segment sets + {m['private_sets']} private sets; "
              f"{m['network_bytes'] / 1e6:.1f} MB received ({m['w8_downloaded_bytes'] / 1e6:.1f} MB of W8, the head "
              f"derived locally); stock {m['stock_disk_bytes'] / 1e6:.1f} MB on disk instead of the "
              f"{m['factory_pack_w8_bytes'] / 1e6:.1f} MB factory pack W8; pass {m['pass_wall_s']:.1f} s wall")
        per_pass = a.segments - a.low_water
        print(f"  network per segment: +{100 * m['network_bytes'] / per_pass / seg:.2f} % of pks-v3's "
              f"{seg / 1e9:.2f} GB per {SEGMENT_SLOTS}-slot segment ({per_pass} segments per pass); "
              f"+{100 * m['network_bytes'] / 64 / seg:.2f} % at 64 per pass")
    rows = extrapolate(m, a.segments, a.low_water, a.private)
    print(f"\n| model | W8 per pass | pks-v3 per 1024-slot segment | network, {a.segments - a.low_water} segments per "
          f"pass (default) | network, 64 per pass | stock on disk ({a.segments} + {a.private} private) | factory pack "
          "W8 it replaces |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['w8_per_pass_gb']:.2f} GB | {r['pks_per_segment_gb']:.1f} GB | "
              f"+{r['network_pct_default']:.2f} % | +{r['network_pct_64_per_pass']:.2f} % | {r['stock_gb']:.2f} GB | "
              f"{r['factory_pack_gb']:.1f} GB |")
    if a.json:
        Path(a.json).write_text(json.dumps({"measured": m, "extrapolated": rows}, indent=1))
    os._exit(0)  # the in-process service's threads


if __name__ == "__main__":
    main()
