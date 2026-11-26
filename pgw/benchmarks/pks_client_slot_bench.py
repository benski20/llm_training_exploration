"""Client CPU per mask slot of the pks-v3 factory path, by segment size (the customer side only).

For every linear shape of a Qwen2.5 model (default 0.5B; real shapes, random weights are not needed) it runs the
customer's per-segment work exactly as :meth:`pgw.hemask.pks_backend.PKSMaskBackend.matmul` does it, with a stand-in
provider that returns a random reply of the right size (the customer's cost does not depend on the reply's values):

* ``r``        the mask rows' keystream (``SeedExpander.ring_uniform_rows``, as the factory draws them),
* ``encrypt``  upload encryption + reply keys (:meth:`pgw.hemask.pks_native.Customer.encrypt`),
* ``ksk_pack`` the reply keys' wire form,
* ``decrypt``  the reply decryption with the compact hint (:meth:`Customer.decrypt`),

for segments of ``--slots`` (a 256-slot group is the unit of encryption: a segment of 64 slots pays for a whole
group). Prints CPU-ms per slot per part and the extrapolation to 7B / 32B / 72B from ns per input- and output-slot,
and cores at 25 slots/s. ``--threads``: the kernels' threads (CPU-seconds are reported, so 1 is the robust figure).

    PYTHONPATH=src python benchmarks/pks_client_slot_bench.py --slots 64 256 1024
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

SHAPES = {
    "0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "32b": (64, {"qkv": (7168, 5120), "o": (5120, 5120), "gate_up": (55296, 5120), "down": (5120, 27648)}),
    "72b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (59136, 8192), "down": (8192, 29568)}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.5b", choices=sorted(SHAPES))
    ap.add_argument("--slots", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    os.environ["PGW_PKS_THREADS"] = str(args.threads)

    from pgw.core.prng import SeedExpander, new_seed
    from pgw.hemask import pks
    from pgw.hemask.pks import NR, PACK, D
    from pgw.hemask.pks_native import Customer
    from pgw.hemask.wire import V3_Y_BITS
    from pgw.service.pks_provider import pack_ksk

    layers, kinds = SHAPES[args.model]
    rng = np.random.default_rng(0)
    cust = Customer(pks.ks_a(new_seed()), threads=args.threads)
    a_seed = new_seed()
    # one reply key + its wire form (per linear and 256-slot group, whatever the widths): timed on their own
    best_k = best_p = float("inf")
    for _ in range(3):
        c0 = time.process_time()
        _, beta1, sec1 = cust.encrypt(a_seed, "0.x", np.zeros((1, 16), dtype=np.int64))
        c1 = time.process_time()
        pack_ksk(beta1)
        c2 = time.process_time()
        best_k, best_p = min(best_k, c1 - c0), min(best_p, c2 - c1)
        for s1 in sec1:
            s1.wipe()
    print(json.dumps({"reply_key_ms": round(best_k * 1e3, 3), "ksk_pack_ms": round(best_p * 1e3, 3),
                      "per": "linear and group (reply key incl. a 16-input upload)"}), flush=True)
    results: dict[int, dict[str, float]] = {}
    for slots in args.slots:
        tot = {"r": 0.0, "encrypt": 0.0, "ksk_pack": 0.0, "decrypt": 0.0}
        for m, n in kinds.values():
            hint = rng.integers(0, 256, (m // PACK, 2, NR, 3), dtype=np.uint8)
            hint[..., 2] &= 0x7F  # residues below the ~2^24 primes
            best = {k: float("inf") for k in tot}
            for _ in range(args.reps):
                c0 = time.process_time()
                r = SeedExpander(new_seed()).ring_uniform_rows(n, 32, 0, slots, "r", "0.x").numpy()
                c1 = time.process_time()
                b, beta, secrets = cust.encrypt(a_seed, "0.x", r)
                c2 = time.process_time()
                ksk = pack_ksk(beta)
                c3 = time.process_time()
                groups = len(secrets)
                y = rng.integers(0, 256, groups * (m // PACK) * NR * V3_Y_BITS // 8, dtype=np.uint8).tobytes()
                c4 = time.process_time()
                out = cust.decrypt(hint, y, secrets)
                c5 = time.process_time()
                del b, ksk, out
                for k, v in (("r", c1 - c0), ("encrypt", c2 - c1), ("ksk_pack", c3 - c2), ("decrypt", c5 - c4)):
                    best[k] = min(best[k], v)
            for k in tot:
                tot[k] += layers * best[k] / slots
        results[slots] = {k: round(v * 1e3, 3) for k, v in tot.items()}
        results[slots]["total_ms"] = round(sum(tot.values()) * 1e3, 3)
        print(json.dumps({"model": args.model, "slots": slots, "groups": -(-slots // D), "threads": args.threads,
                          "cpu_ms_per_slot": results[slots]}), flush=True)
    # extrapolation: r and the upload scale with sum n, decryption with sum m, reply keys and their wire form with
    # the number of linears (per 256-slot group)
    ext = {}
    l0, k0 = SHAPES[args.model]
    sn0 = l0 * sum(n for _, n in k0.values())
    sm0 = l0 * sum(m for m, _ in k0.values())
    nl0 = l0 * len(k0)
    for slots, res in results.items():
        groups = -(-slots // D)
        keys0 = nl0 * groups * best_k * 1e3 / slots  # ms per slot of reply keys in "encrypt"
        upload = max(0.0, res["encrypt"] - keys0)
        row = {}
        for name, (layers, kinds) in SHAPES.items():
            sn = layers * sum(n for _, n in kinds.values())
            sm = layers * sum(m for m, _ in kinds.values())
            nl = layers * len(kinds)
            ms = ((res["r"] + upload) * sn / sn0 + res["decrypt"] * sm / sm0
                  + (keys0 + res["ksk_pack"]) * nl / nl0)
            row[name] = {"cpu_ms_per_slot": round(ms, 2), "cores_at_25": round(ms * 25 / 1e3, 2),
                         "parts_ms": {"r": round(res["r"] * sn / sn0, 2), "upload": round(upload * sn / sn0, 2),
                                      "reply_keys": round(keys0 * nl / nl0, 2),
                                      "ksk_pack": round(res["ksk_pack"] * nl / nl0, 2),
                                      "decrypt": round(res["decrypt"] * sm / sm0, 2)}}
        ext[slots] = row
    print(json.dumps({"extrapolated_from": args.model, "by_segment_slots": ext}, indent=1))


if __name__ == "__main__":
    main()
