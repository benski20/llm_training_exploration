"""One exact end-to-end fused2 run (2 provider round trips per layer): gateway here, provider in its own process.

Opt-in harness (default off everywhere). Steps:

1. pack time (public): fold the norm gains into the float checkpoint's linears, calibrate the migrated split fold
   (:func:`pgw.engine.fused2.calibrate_fused2`, public calibration text below) and write the gateway's weights and
   the provider's (``w8`` + ``hmul`` only);
2. start the provider (``python -m pgw.engine.fused2_remote serve``: the pair op on its GPU, masked ring elements
   only) on a 0600 Unix socket or loopback TCP;
3. gateway (this process, CPU): masks precomputed by an in-process plain factory (:meth:`Fused2MaskSource.prefill`,
   seed-expanded ``r_x`` and ``v``, ``W1 r_x`` and ``W2' v``), :class:`Fused2MaskedBackend` (mask, call, unmask,
   range + F_p Freivalds on both stages of every trip), greedy decoding of ``--tokens`` tokens;
4. the plain fused2 reference replays the same forward calls; every logit must be bit-identical.

Prints and writes (``--json``) tokens/s, ms per token, trips per token, bytes per token, slots (mask rows of every
trip key) per token, trip latency percentiles, provider compute time, gateway check time, exactness.

Usage (local CPU, both processes on this box):
    python benchmarks/fused2_e2e.py --hf-dir <HF snapshot dir of Qwen2.5-0.5B-Instruct> \\
        --pack /path/to/pgw/pack --tokens 16 --provider-device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

#: Public calibration text (any public text works; it only picks the pack's public multipliers and gateway columns).
CALIB_TEXT = (
    "The history of science is the study of the development of science and scientific knowledge, including both "
    "the natural and social sciences. Science is a body of empirical, theoretical, and practical knowledge about the "
    "natural world, produced by scientists who emphasize the observation, explanation, and prediction of real-world "
    "phenomena. In the early modern period, the printing press made it possible to share results quickly, and new "
    "instruments such as the telescope and the microscope opened worlds that had been invisible before.\n\n"
    "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n    mid = len(items) // 2\n"
    "    left = merge_sort(items[:mid])\n    right = merge_sort(items[mid:])\n    return merge(left, right)\n\n"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the boiling point of water "
    "at sea level, and why does it change with altitude?<|im_end|>\n<|im_start|>assistant\nAt sea level water boils "
    "at 100 degrees Celsius (212 degrees Fahrenheit). At higher altitudes the air pressure is lower, so water boils "
    "at a lower temperature: roughly one degree Celsius less for every 300 metres of elevation.<|im_end|>\n"
    "Il était une fois une petite ville au bord de la mer, où les pêcheurs partaient chaque matin avant l'aube. "
    "Die Wissenschaft lebt vom Zweifel und von der Bereitschaft, Irrtümer zu korrigieren. 1, 2, 3, 5, 8, 13, 21, 34."
)
PROMPT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain in a few sentences "
          "why the sky is blue.<|im_end|>\n<|im_start|>assistant\n")


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def probe_rows_below_a(pack: Any, w: Any, ring: Any, ids: torch.Tensor) -> float:
    """Fraction of trip rows whose a-priori bound forces fewer than ``A`` activation bits on public text."""
    from pgw.engine.fused2 import Fused2PlainBackend, enable_fused2
    from pgw.engine.model import TransformerModel

    m = TransformerModel(pack, Fused2PlainBackend(ring))
    sched = enable_fused2(m, w, ring)
    with torch.no_grad():
        m.forward(ids, m.new_cache(len(ids) + 1))
    tot = sum(sched.bits_hist.values())
    return 1.0 - sched.bits_hist.get(ring.a, 0) / max(1, tot)


def build_weights(args: argparse.Namespace, pack: Any, tok: Any, ring: Any) -> Any:
    from pgw.engine.fused2 import Fused2Weights, calibrate_fused2, hf_float_linears, migration_multipliers

    t0 = time.time()
    state = Path(args.hf_dir)
    fl = hf_float_linears(state, pack.arch) if not args.fold_int8 else None
    if args.migrate_bits <= 0:
        w = Fused2Weights(pack, fl)
        info: dict[str, Any] = {"calibration": None, "fold": "unmigrated (old reference)"}
    elif args.columns <= 0:  # pure pack-time migration: public weights only, no calibration, no gateway columns
        tries = [args.migrate_bits] + ([b for b in (6, 4) if b < args.migrate_bits] if args.auto_bits else [])
        probes: dict[str, float] = {}
        for b in tries:
            w = Fused2Weights(pack, fl, multipliers=migration_multipliers(pack, b))
            if not args.auto_bits:
                break
            below = probe_rows_below_a(pack, w, ring, tok(CALIB_TEXT, return_tensors="pt").input_ids[0][:128])
            probes[str(b)] = below
            print(f"migration {b} bits: {100 * below:.2f}% of probe rows below {ring.a} activation bits", flush=True)
            if below <= args.max_rows_below_a:
                break
        info = {"calibration": None, "fold": f"migrated {b} bits", "migrate_bits_used": b, "probe_rows_below_a": probes}
    else:
        ids = tok(CALIB_TEXT, return_tensors="pt").input_ids[0][: args.calib_tokens]
        cal = calibrate_fused2(pack, fl, [ids], ring, bits=args.migrate_bits, columns=args.columns,
                               gateway_bits=args.gateway_bits)
        w = Fused2Weights(pack, fl, gateway_columns=cal["gateway_columns"], multipliers=cal["multipliers"],
                          gateway_bits=cal["gateway_bits"])
        info = {"calibration": cal["stats"], "calib_tokens": int(ids.numel()),
                "fold": f"migrated split fold {args.migrate_bits} bits, {args.columns} top gateway columns"}
    info["gateway_cost"] = w.gateway_cost()
    info["build_s"] = time.time() - t0
    return w, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="HF checkpoint dir (config, tokenizer, safetensors)")
    ap.add_argument("--pack", default=None, help="pgw pack of the same checkpoint (default: built in memory)")
    ap.add_argument("--ring", default="K56A12C53")
    ap.add_argument("--migrate-bits", type=int, default=8, help="0: the unmigrated fold (old reference)")
    ap.add_argument("--auto-bits", action="store_true",
                    help="probe 128 public tokens; if too many rows drop activation bits, migrate fewer (8, 6, 4)")
    ap.add_argument("--max-rows-below-a", type=float, default=0.01)
    ap.add_argument("--columns", type=int, default=0, help=">0: calibrated split fold with that many top columns")
    ap.add_argument("--gateway-bits", type=int, default=12)
    ap.add_argument("--calib-tokens", type=int, default=512)
    ap.add_argument("--fold-int8", action="store_true", help="fold the pack's int8 weights (no float checkpoint)")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--provider-device", default="auto")
    ap.add_argument("--address", default=None, help="unix:/path (default: a 0600 socket in a temp dir) or tcp:...")
    ap.add_argument("--provider-cmd-prefix", default="", help="e.g. 'runuser -u pgwprov --' (a separate user)")
    ap.add_argument("--no-replay", action="store_true", help="skip the plain replay (exactness check)")
    ap.add_argument("--replay-calls", type=int, default=0,
                    help="replay only the first N forward calls (prefill = 1; 0: all). Every trip is Freivalds-checked "
                    "against the exact integers anyway; the replay also covers the gateway's own float code")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer

    from pgw.core.ring import RingParams
    from pgw.engine.fused2 import Fused2MaskedBackend, Fused2MaskSource, Fused2PlainBackend, enable_fused2
    from pgw.engine.fused2_remote import RemoteFused2Provider
    from pgw.engine.model import TransformerModel

    ring = RingParams.from_name(args.ring)
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    if args.pack:
        from pgw.modelpack import ModelPack

        pack = ModelPack.open(args.pack)
    else:
        from safetensors.torch import load_file

        from pgw.engine.pack import InMemoryPack, arch_config_from_hf

        sd: dict[str, torch.Tensor] = {}
        for f in sorted(Path(args.hf_dir).glob("*.safetensors")):
            sd.update(load_file(str(f)))
        cfg = json.loads((Path(args.hf_dir) / "config.json").read_text())
        pack = InMemoryPack.from_state_dict(sd, arch_config_from_hf(cfg))
        del sd
    w, info = build_weights(args, pack, tok, ring)
    print("weights", json.dumps(info), flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="pgw-f2-"))
    os.chmod(tmp, 0o711)  # noqa: S103  # traversable: the provider's user reads its public weights file
    prov_file = tmp / "provider_weights.pt"
    w.save(prov_file, provider_only=True)
    os.chmod(prov_file, 0o644)  # public weights; the provider's user reads them
    address = args.address or f"unix:{tmp / 'f2.sock'}"
    cmd = [*args.provider_cmd_prefix.split(), sys.executable, "-m", "pgw.engine.fused2_remote", "serve",
           "--weights", str(prov_file), "--k", str(ring.k), "--address", address, "--device", args.provider_device]
    t0 = time.time()
    prov = subprocess.Popen(cmd)  # noqa: S603
    try:
        dims = {lin.key: (lin.m, lin.n) for lin in w.linears()}
        client = RemoteFused2Provider(address, ring.k, dims)
        connect_s = time.time() - t0
        src = Fused2MaskSource(w, ring)
        ids = tok(args.prompt, return_tensors="pt").input_ids[0]
        slots = int(ids.numel()) + args.tokens
        t_chk = time.time()
        for lin in w.linears():
            src.check_vectors(lin)  # u W8 mod p of every linear (factory precompute), off the measured path
        check_setup_s = time.time() - t_chk
        prefill_s = src.prefill(slots)
        print(f"provider up in {connect_s:.1f}s; masks for {slots} rows (slots) in {prefill_s:.1f}s", flush=True)
        be = Fused2MaskedBackend(ring, client, src)
        model = TransformerModel(pack, be)
        sched = enable_fused2(model, w, ring)
        cache = model.new_cache(slots + 1)
        calls: list[torch.Tensor] = [ids]
        t_first = time.perf_counter()
        logits = [model.forward(ids, cache)]
        ttft = time.perf_counter() - t_first
        trips0, bytes0 = be.f2.trips, client.bytes_up + client.bytes_down
        n_trip_ms0 = len(client.trip_ms)
        step_s: list[float] = []
        out_ids: list[int] = []
        for _ in range(args.tokens):
            nxt = int(logits[-1].argmax())
            out_ids.append(nxt)
            x = torch.tensor([nxt])
            calls.append(x)
            t1 = time.perf_counter()
            logits.append(model.forward(x, cache))
            step_s.append(time.perf_counter() - t1)
        dec_trips = be.f2.trips - trips0
        dec_bytes = client.bytes_up + client.bytes_down - bytes0
        trip_ms = client.trip_ms[n_trip_ms0:]
        comp_ms = client.compute_ms[n_trip_ms0:]
        res: dict[str, Any] = {
            "model": args.hf_dir, "ring": args.ring, "migrate_bits": args.migrate_bits, "columns": args.columns,
            "weights": info, "prompt_tokens": int(ids.numel()), "tokens": args.tokens,
            "text": tok.decode(out_ids), "ttft_s": ttft,
            "decode_tok_s": len(step_s) / sum(step_s) if step_s else None,
            "ms_per_token_p50": 1e3 * statistics.median(step_s) if step_s else None,
            "trips_per_token": dec_trips / max(1, len(step_s)),
            "expected_trips_per_token": 2 * w.num_layers + 1,
            "bytes_per_token": dec_bytes / max(1, len(step_s)),
            "trip_ms_p50": _pct(trip_ms, 0.5), "trip_ms_p90": _pct(trip_ms, 0.9),
            "provider_compute_ms_p50": _pct(comp_ms, 0.5),
            "trip_ms_sum_per_token": sum(trip_ms) / max(1, len(step_s)),
            "checks": be.f2.checks, "all_checks_passed": True, "slots_used": slots, "slots_per_token": 1.0,
            "mask_prefill_s": prefill_s, "check_vectors_setup_s": check_setup_s,
            "mask_stock_hits": src.stock_hits, "mask_stock_misses": src.stock_misses,
            "factory_macs_per_slot": src.macs / max(1, slots),
            "act_bits_hist": {str(k): v for k, v in sorted(sched.bits_hist.items())},
        }
        client.close()
        if not args.no_replay:
            t2 = time.time()
            plain = TransformerModel(pack, Fused2PlainBackend(ring))
            enable_fused2(plain, w, ring)
            pc = plain.new_cache(slots + 1)
            n_rep = len(calls) if args.replay_calls <= 0 else min(len(calls), args.replay_calls)
            ref = [plain.forward(c, pc) for c in calls[:n_rep]]
            same = [bool(torch.equal(a, b)) for a, b in zip(ref, logits[:n_rep], strict=True)]
            res["exact"] = all(same)
            res["exact_positions"] = f"{sum(same)}/{len(same)}"
            res["max_abs_logit_diff"] = max(float((a - b).abs().max()) for a, b in zip(ref, logits, strict=False))
            res["replay_s"] = time.time() - t2
        print(json.dumps(res, indent=1), flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=1))
        if not args.no_replay and not res["exact"]:
            raise SystemExit("fused2 e2e: masked != plain")
        assert math.isfinite(res["trips_per_token"])
    finally:
        prov.terminate()
        prov.wait(timeout=30)


if __name__ == "__main__":
    main()
