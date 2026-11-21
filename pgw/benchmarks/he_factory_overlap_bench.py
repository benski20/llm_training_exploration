#!/usr/bin/env python3
"""Encrypted mask factory throughput, serial stages vs overlapped stages (``he_pipeline_depth``).

The real :class:`pgw.factory.MaskGenerator` with the production :func:`pgw.factory.runtime.build_he_backend`
(``RemoteMulmod64`` over HTTP, ``he_encrypt``/``he_groups_per_call``/``he_max_inflight``/``he_pipeline_depth``
from :class:`pgw.factory.FactorySettings`), on a packed synthetic Qwen2 checkpoint of a preset's shapes. Every
layer gets the same weights, so the one-time hint ``W8 A`` is computed once per weight shape and copied (or
hard-linked) for the other layers; the factory, the wire and the service do the full per-layer work.

Provider (``--provider``):

* ``service``: a real ``pgw service serve`` subprocess on loopback HTTP (``--service-device``; on Kaggle the
  T4). The factory and the service share the host's CPUs, as in the Modal demo's single container.
* ``inproc``: :meth:`pgw.service.kernels.ComputeKernel.mulmod64` in this process (``--service-device``),
  optionally with ``--delay-ms`` of sleep per request (a stand-in for wire time; CPU stub for local runs).

For each ``depth:inflight`` config (``--configs 0:2 2:2 4:4``) it runs a warm-up segment, then ``--segments``
segments of ``--slots`` slots and reports slots/s (full model; ``--layers`` below the preset's count are
scaled) with the per-stage seconds. Every config generates one segment from a fixed seed whose masks are
hashed: all configs must give the same digest, equal to the local plaintext factory's (``--reference``).

    python benchmarks/he_factory_overlap_bench.py --provider inproc --layers 2 --delay-ms 20 --configs 0:2 4:4
    python benchmarks/he_factory_overlap_bench.py --provider service --service-device cuda --layers 24 \\
        --configs 0:2 2:2 4:2 4:4 --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEED = bytes(range(7, 39))
TOKEN = "bench-token-0123456789abcdef"  # noqa: S105 - throw-away loopback credential

PRESETS: dict[str, dict[str, int]] = {
    "qwen2.5-0.5b": {"hidden_size": 896, "intermediate_size": 4864, "num_attention_heads": 14,
                     "num_key_value_heads": 2, "layers": 24},
    "qwen2.5-7b": {"hidden_size": 3584, "intermediate_size": 18944, "num_attention_heads": 28,
                   "num_key_value_heads": 4, "layers": 28},
}


def build_pack(work: Path, preset: str, layers: int) -> Path:
    """Synthetic Qwen2 checkpoint of ``preset``'s shapes with ``layers`` identical layers, packed."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from pgw.modelpack import pack_checkpoint

    p = PRESETS[preset]
    ck, pk = work / f"ckpt-{preset}-{layers}", work / f"pack-{preset}-{layers}"
    if (pk / "manifest.json").exists():
        return pk
    sys.path.insert(0, str(ROOT))
    from tests.conftest import _save_byte_tokenizer  # type: ignore[import-not-found]

    cfg = Qwen2Config(vocab_size=512, hidden_size=p["hidden_size"], intermediate_size=p["intermediate_size"],
                      num_hidden_layers=layers, num_attention_heads=p["num_attention_heads"],
                      num_key_value_heads=p["num_key_value_heads"], max_position_embeddings=1024,
                      tie_word_embeddings=False, torch_dtype="bfloat16")
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():  # every layer = layer 0, so hints are built once per weight shape
        sd0 = model.model.layers[0].state_dict()
        for lay in model.model.layers[1:]:
            lay.load_state_dict(sd0)
    model.eval().save_pretrained(ck, safe_serialization=True)
    _save_byte_tokenizer(ck)
    t0 = time.perf_counter()
    pack_checkpoint(ck, pk, name=f"synthetic-{preset}-{layers}", overwrite=True)
    shutil.rmtree(ck, ignore_errors=True)
    print(f"packed {pk.name} in {time.perf_counter() - t0:.1f}s", flush=True)
    return pk


def build_hints(root: Path, pack: Any, rank: int, q_bits: int, mode: str) -> Any:
    """:class:`pgw.hemask.backend.HintStore` in the production layout, one hint per distinct weight."""
    from pgw.core.prng import new_seed
    from pgw.hemask.backend import _HINT_VERSION, HintStore
    from pgw.hemask.fastmm import to_planes
    from pgw.hemask.lwe import HEParams, matmul_mod64, public_matrix

    path = HintStore.dir_for(root, pack.fingerprint, rank, q_bits)
    if (path / "meta.json").exists():
        hs = HintStore.open(root, pack.fingerprint, rank, q_bits)
        if hs is not None:
            return hs
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    params = HEParams(k=16, rank=rank, q_bits=q_bits)
    planes = q_bits // 8 - 1
    seed = new_seed()
    specs = list(pack.linear_specs())
    a = public_matrix(seed, max(s.in_features for s in specs), params)
    first: dict[str, Path] = {}
    norms: dict[str, float] = {}
    t0 = time.perf_counter()
    for lw in pack.iter_linears():
        w = lw.w8.numpy() if isinstance(lw.w8, torch.Tensor) else np.asarray(lw.w8)
        key = hashlib.sha256(w.tobytes()).hexdigest() + str(w.shape)
        norms[lw.id.key] = float(np.sqrt((w.astype(np.float64) ** 2).sum(1)).max())
        dst = path / f"{lw.id.key}.npy"
        if key in first:
            (os.link if mode == "link" else shutil.copyfile)(first[key], dst)
            continue
        out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.int8, shape=(w.shape[0], planes, params.dim))
        for c0 in range(0, params.dim, 512):
            out[:, :, c0:c0 + 512] = to_planes(matmul_mod64(w, a[:w.shape[1], c0:c0 + 512]))[:, 1:1 + planes]
        out.flush()
        del out
        first[key] = dst
    meta = {"version": _HINT_VERSION, "fingerprint": pack.fingerprint, "rank": rank, "q_bits": q_bits,
            "a_seed": seed.hex(), "dim": params.dim, "row_norms": norms, "created": time.time()}
    (path / "meta.json").write_text(json.dumps(meta))
    hs = HintStore(path, meta)
    print(f"hints ({mode}, {len(first)} distinct) built in {time.perf_counter() - t0:.1f}s: {hs.nbytes / 1e9:.2f} GB",
          flush=True)
    return hs


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_service(work: Path, pack_dir: Path, ring: str, device: str,
                  extra_env: dict[str, str] | None = None) -> tuple[subprocess.Popen[bytes], str]:
    from pgw.service import hash_token

    tokens = work / "tenants.txt"
    tokens.write_text(f"bench:{hash_token(TOKEN)}\n")
    port = _free_port()
    env = dict(os.environ)
    env.update({
        "PGW_SERVICE_HOST": "127.0.0.1", "PGW_SERVICE_PORT": str(port),
        "PGW_SERVICE_PACKS": json.dumps([str(pack_dir)]), "PGW_SERVICE_RINGS": json.dumps([ring]),
        "PGW_SERVICE_DEVICE": device, "PGW_SERVICE_TOKENS_FILE": str(tokens),
        "PGW_SERVICE_MAX_REQUEST_BYTES": str(1 << 30), "PGW_LOG_LEVEL": "WARNING",
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
    })
    env.update(extra_env or {})
    log = open(work / "service.log", "wb")
    proc = subprocess.Popen([sys.executable, "-m", "pgw.cli", "service", "serve"], env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"service exited: {(work / 'service.log').read_text()[-3000:]}")
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=2) as r:  # noqa: S310 - loopback
                if r.status == 200:
                    return proc, url
        except OSError:
            pass
        time.sleep(1.0)
    proc.kill()
    raise RuntimeError("service not ready")


class InprocProvider:
    """``W8 b mod 2^q`` in this process (``ComputeKernel.mulmod64``), plus an optional fixed delay."""

    def __init__(self, pack: Any, device: str, nbytes: int, delay_ms: float):
        from pgw.service.kernels import ComputeKernel

        self.kern: dict[str, Any] = {}
        shared: dict[str, Any] = {}
        for lw in pack.iter_linears():
            spec = next(s for s in pack.linear_specs() if s.id == lw.id)
            key = hashlib.sha256(lw.w8.numpy().tobytes()).hexdigest()
            if key not in shared:
                shared[key] = ComputeKernel(spec, lw.w8, device=device)
            self.kern[lw.id.key] = shared[key]
        self.nbytes = nbytes
        self.delay_s = delay_ms / 1e3
        self.lock = threading.Lock()  # one device, products in series (the service queues per device too)
        self.seconds = 0.0

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        if self.delay_s:
            time.sleep(self.delay_s)
        with self.lock:
            t0 = time.perf_counter()
            y: np.ndarray = self.kern[linear].mulmod64(b, self.nbytes)
            self.seconds += time.perf_counter() - t0
        return y


def digest(gen: Any, slots: int) -> str:
    """Hash of a segment's ``W r`` and ``u W8`` (independent of the chunk size: items come in canonical order)."""
    h = hashlib.sha256()
    for item in gen.generate(SEED, slots, checks=2):
        if item.uw is not None:  # first chunk of a linear
            h.update(item.id.key.encode())
            h.update(item.uw.contiguous().numpy().tobytes())
        h.update(item.wr.contiguous().numpy().tobytes())
        item.wr.zero_()
    return h.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from pgw.core.prng import new_seed
    from pgw.core.ring import RingParams
    from pgw.factory import MaskGenerator
    from pgw.factory.config import FactorySettings
    from pgw.factory.generator import format_stats
    from pgw.factory.runtime import build_he_backend
    from pgw.hemask.backend import HEMaskBackend
    from pgw.hemask.lwe import HEParams
    from pgw.modelpack import ModelPack

    work = Path(args.workdir or tempfile.mkdtemp(prefix="pgw-heov-"))
    work.mkdir(parents=True, exist_ok=True)
    layers = args.layers or PRESETS[args.preset]["layers"]
    pack_dir = build_pack(work, args.preset, layers)
    pack = ModelPack.open(pack_dir, components=("linears",))
    ring = RingParams.from_name(args.ring)
    q_bits = HEParams(k=ring.k, rank=args.rank).q
    hints = build_hints(work / "hints", pack, args.rank, q_bits, args.hint_mode)
    scale = PRESETS[args.preset]["layers"] / layers
    res: dict[str, Any] = {"preset": args.preset, "layers": layers, "ring": ring.name, "slots": args.slots,
                           "segments": args.segments, "provider": args.provider, "encrypt": args.encrypt,
                           "groups_per_call": args.groups_per_call, "cpus": os.cpu_count(),
                           "torch_threads": torch.get_num_threads(), "hint_gb": round(hints.nbytes / 1e9, 2),
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           "configs": []}
    proc = None
    url = None
    inproc = None
    try:
        if args.provider == "service":
            extra = dict(kv.split("=", 1) for kv in args.service_env)
            res["service_env"] = extra
            proc, url = start_service(work, pack_dir, ring.name, args.service_device, extra)
        else:
            inproc = InprocProvider(pack, args.service_device, HEParams(k=ring.k, rank=args.rank).b_bytes,
                                    args.delay_ms)
        if args.reference:
            t0 = time.perf_counter()
            res["reference_digest"] = digest(MaskGenerator(pack, ring, device=args.device), args.slots)
            print(f"plaintext reference segment in {time.perf_counter() - t0:.1f}s", flush=True)
        for cfg in args.configs:
            depth, inflight = (int(x) for x in cfg.split(":"))
            if url is not None:
                st = FactorySettings(pack_path=pack_dir, he_service_url=url, he_token_file=work / "token",
                                     he_hint_dir=work / "hints", he_rank=args.rank, he_encrypt=args.encrypt,
                                     he_groups_per_call=args.groups_per_call, he_max_inflight=inflight,
                                     he_pipeline_depth=depth)
                (work / "token").write_text(TOKEN)
                be = build_he_backend(st, pack, ring.name)
            else:
                be = HEMaskBackend(pack, ring, hints, inproc, groups_per_call=args.groups_per_call,  # type: ignore[arg-type]
                                   max_inflight=inflight, encrypt=args.encrypt, pipeline_depth=depth)
            gen = MaskGenerator(pack, ring, device=args.device, remote=be)
            for _ in gen.generate(new_seed(), args.warmup_slots, checks=2):  # warm-up (connections, caches)
                pass
            walls = []
            stages: dict[str, float] = {}
            he: dict[str, float] = {}
            for _ in range(args.segments):
                t0 = time.perf_counter()
                for item in gen.generate(new_seed(), args.slots, checks=2):
                    item.wr.zero_()
                walls.append(time.perf_counter() - t0)
                s = gen.last_stats
                for k, v in (("rng", s.rng_seconds), ("gemm", s.gemm_seconds), ("check", s.check_seconds)):
                    stages[k] = stages.get(k, 0.0) + v
                for k, v in s.extra.get("he", {}).items():
                    he[k] = he.get(k, 0.0) + v
                line = format_stats(s)
                print(f"[depth {depth} inflight {inflight}] {line}", flush=True)
            dig = digest(gen, args.slots)
            wall = sum(walls)
            row = {"depth": depth, "inflight": inflight, "wall_s": round(wall, 3),
                   "seg_s": [round(w, 3) for w in walls],
                   "slots_per_s_full_model": round(args.slots * args.segments / wall / scale, 2),
                   "stages": {k: round(v, 3) for k, v in stages.items()},
                   "he": {k: round(v, 3) for k, v in he.items()}, "digest": dig}
            if "reference_digest" in res:
                row["bit_identical_to_plaintext"] = dig == res["reference_digest"]
            res["configs"].append(row)
            print("CONFIG " + json.dumps(row), flush=True)
            del gen, be
        digs = {c["digest"] for c in res["configs"]}
        res["all_configs_identical"] = len(digs) == 1
        base = res["configs"][0]["slots_per_s_full_model"]
        for c in res["configs"]:
            c["speedup_vs_first"] = round(c["slots_per_s_full_model"] / base, 2)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        if not args.keep:
            shutil.rmtree(work / "hints", ignore_errors=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="qwen2.5-0.5b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=0, help="layers to build (0: the preset's)")
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--slots", type=int, default=512, help="slots per segment (the Modal demo's factory: 512)")
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--warmup-slots", type=int, default=256)
    ap.add_argument("--encrypt", default="ntt_pre", choices=("dense", "ntt_pre"))
    ap.add_argument("--groups-per-call", type=int, default=4)
    ap.add_argument("--configs", nargs="+", default=["0:2", "2:2", "4:2", "4:4"],
                    help="depth:inflight pairs (depth 0 = the serial path)")
    ap.add_argument("--provider", choices=("service", "inproc"), default="inproc")
    ap.add_argument("--service-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="inproc: sleep per request")
    ap.add_argument("--service-env", nargs="*", default=[], help="service: extra KEY=VALUE settings")
    ap.add_argument("--device", default="cpu", help="factory device (u W8 checks; the reference GEMMs)")
    ap.add_argument("--hint-mode", choices=("link", "copy"), default="link",
                    help="identical layers' hints: hard links (less disk, warmer page cache) or copies")
    ap.add_argument("--reference", action="store_true", help="also hash the local plaintext factory's segment")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the hints in --workdir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    res = run(args)
    print("RESULT " + json.dumps(res), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
