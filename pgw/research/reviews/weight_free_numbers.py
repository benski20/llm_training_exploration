"""Numbers for ``research/reviews/weight-free-client-checks.md``.

Run: ``PYTHONPATH=src python research/reviews/weight_free_numbers.py [--measure]``.

Same models as the pks-v3 packet and ``pks_v4_numbers.py``: ``pgw.hemask.mlwe_ring.core_svp_beta`` (primal uSVP,
2016 estimate) and ``PKSParams.noise_terms`` (24-sigma / Hoeffding tails, budget ``q1 / 2^33``). Shapes are the
real Qwen2.5 ones (fused ``gate_up``, as the packs store them) plus the masked LM head (152064 x hidden). Per-slot
traffic of today's pks-v3 comes from ``benchmarks/hemask_pks_model.py`` (same formulas).

Everything that is a probability is computed exactly from a closed form, and the closed forms are checked by Monte
Carlo at small moduli (section 5). ``--measure`` adds two measurements on this box: SHA-256 throughput, and the
row and column norms of the Qwen2.5-0.5B int8 pack if it is present (``PGW_W8_PACK`` or the SDK demo cache).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import time
from pathlib import Path

from pgw.hemask.mlwe_ring import core_svp_beta
from pgw.hemask.pks import NU, PS, Q1, PKSParams

P31 = (1 << 31) - 1
SLACK = 1.0 + 2.0 ** -33  # point probability of u64 mod p, relative to 1/p (SPEC 5.1)
SIG = math.sqrt(21 / 2)
HEAD_VOCAB = 152064
HINT_ROW_BYTES = 256 * 6  # stored pks-v3 hint: 256 coefficients x 6 bytes per output row
UP_BYTES_PER_FEATURE_GROUP = 256 * 7  # 4096 coefficients x 7 bytes per 16 features = 7 B per feature and slot
KSK_BYTES = 16 * 8192 * (24 + 24 + 30 + 30) // 8  # one reply key (per linear and 256-slot group)
MAX_ROW_L2 = 40931  # check_noise(margin 1) accepts row norms up to this (adversarial review, noise section)

MODELS = {  # layers, hidden, {kind: (m, n)} (benchmarks/hemask_pks_model.py), plus the masked head
    "7B": (28, 3584, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "32B": (64, 5120, {"qkv": (7168, 5120), "o": (5120, 5120), "gate_up": (55296, 5120), "down": (5120, 27648)}),
    "72B": (80, 8192, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (59136, 8192), "down": (8192, 29568)}),
}
# pks-v3 per-slot traffic (MB up / MB down), from benchmarks/hemask_pks_model.py
TRAFFIC_MB_PER_SLOT = {"7B": (6.6, 6.3), "32B": (21.0, 20.9), "72B": (32.5, 30.9)}
CLIENT_MS_PER_SLOT = {"7B": 22.0, "32B": 72.0, "72B": 110.7}  # pks-v3 customer CPU, tuned AVX-512 IFMA kernels


def shapes(name: str) -> list[tuple[int, int]]:
    layers, hidden, kinds = MODELS[name]
    return [mn for _ in range(layers) for mn in kinds.values()] + [(HEAD_VOCAB, hidden)]


def lg(x: float) -> str:
    return "0" if x == 0 else f"2^{math.log2(x):.2f}"


def gb(x: float) -> str:
    return f"{x / 1e9:.2f} GB"


def mb(x: float) -> str:
    return f"{x / 1e6:.1f} MB"


# ------------------------------------------------------------------------------------------------------------
# closed forms
# ------------------------------------------------------------------------------------------------------------

def prime_fresh(c: int) -> float:
    """Today (SPEC 5.1): fresh uniform u per segment, c vectors, error independent of u."""
    return (SLACK / P31) ** c


def prime_basis(c: int, k: int) -> float:
    """Candidate B: u_i = B c_i, B (m x k) secret and long-lived, c_i fresh. Error e != 0 independent of (B, c).

    B^T e is (close to) uniform in F_p^k: zero with probability <= (SLACK/p)^k, and then every u_i . e = 0.
    Otherwise each fresh c_i hits the hyperplane with probability <= SLACK/p."""
    z = (SLACK / P31) ** k
    return z + (1 - z) * (SLACK / P31) ** c


def prime_basis_conversation(c: int, k: int, segments: int) -> float:
    """The same fixed error re-tried in ``segments`` segments (fresh c each, same B)."""
    z = (SLACK / P31) ** k
    return z + (1 - z) * (SLACK / P31) ** (c * segments)


def ring_fresh(c: int, t: int, K: int = 32) -> float:
    """Today's private check: c fresh uniform vectors over Z_2^K, error of 2-adic valuation t."""
    return 2.0 ** (-c * (K - t))


def ring_basis(c: int, k: int, t: int, K: int = 32) -> float:
    """Candidate B over Z_2^K: v_i = B c_i with B (m x k) uniform and long-lived, c_i fresh uniform.

    w = B^T delta has k independent coordinates, each uniform on 2^t Z / 2^K Z, so P[val(w) >= s] = 2^-(s-t)k.
    Given val(w) = s < K, each fresh c_i misses with probability 2^-(K-s); w = 0 passes every c_i."""
    total = 2.0 ** (-(K - t) * k)  # w = 0
    for s in range(t, K):
        p_eq = 2.0 ** (-(s - t) * k) - 2.0 ** (-(s + 1 - t) * k)
        total += p_eq * 2.0 ** (-c * (K - s))
    return total


# ------------------------------------------------------------------------------------------------------------
# small linear algebra mod p (python ints; attack demonstrations at small sizes)
# ------------------------------------------------------------------------------------------------------------

def left_kernel_vector(rows: list[list[int]], p: int, rng: random.Random) -> list[int]:
    """A random nonzero x with sum_j x_j rows[i][j] = 0 for every i, i.e. x in ker of the k x m matrix ``rows``."""
    k, m = len(rows), len(rows[0])
    a = [r[:] for r in rows]
    piv_cols: list[int] = []
    r = 0
    for col in range(m):
        pr = next((i for i in range(r, k) if a[i][col] % p), None)
        if pr is None:
            continue
        a[r], a[pr] = a[pr], a[r]
        inv = pow(a[r][col], p - 2, p)
        a[r] = [(x * inv) % p for x in a[r]]
        for i in range(k):
            if i != r and a[i][col]:
                f = a[i][col]
                a[i] = [(x - f * y) % p for x, y in zip(a[i], a[r])]
        piv_cols.append(col)
        r += 1
        if r == k:
            break
    free = [j for j in range(m) if j not in piv_cols]
    x = [0] * m
    for j in free:
        x[j] = rng.randrange(1, p)
    for i, pc in enumerate(piv_cols):
        x[pc] = (-sum(a[i][j] * x[j] for j in free)) % p
    return x


def dot(a: list[int], b: list[int], p: int) -> int:
    return sum(x * y for x, y in zip(a, b)) % p


# ------------------------------------------------------------------------------------------------------------
# sections
# ------------------------------------------------------------------------------------------------------------

def section_sizes() -> None:
    print("[1] what needs W8 per segment (real shapes, incl. the masked head)")
    print("| model | linears | sum m | sum n | W8 layers (factory pack, at rest) | head W8 (derived from the plain head) "
          "| uW per segment (p31, c=1) | private set (14 x n, u64) |")
    print("|---|---|---|---|---|---|---|---|")
    for name in MODELS:
        sh = shapes(name)
        sm, sn, smn = sum(m for m, _ in sh), sum(n for _, n in sh), sum(m * n for m, n in sh)
        head = HEAD_VOCAB * MODELS[name][1]
        print(f"| {name} | {len(sh)} | {sm / 1e6:.2f} M | {sn / 1e6:.3f} M | {gb(smn - head)} | {gb(head)} "
              f"| {mb(4 * sn)} | {mb(8 * 14 * sn)} |")
    print("  today's per-segment factory work on W8: prime_precompute(u, W8) = 2 exact GEMV passes (16-bit limbs)"
          " over all of W8, plus 14 ring GEMVs per linear whenever the private set is redrawn")


def section_security_params() -> None:
    pr = PKSParams()
    q1, P = Q1[0] * Q1[1], PS[0] * PS[1]
    print("\n[2] lattice parameters (unchanged by every candidate; candidate A re-uses the upload instance on W8^T)")
    b_up = core_svp_beta(NU, 56, SIG)
    b_re = core_svp_beta(8192, math.log2(q1 * P), SIG)
    print(f"  upload  (n 4096, log q 56):     beta {b_up}, core-SVP {0.292 * b_up:.1f} / {0.265 * b_up:.1f}")
    print(f"  reply   (n 8192, log q {math.log2(q1 * P):.2f}): beta {b_re}, core-SVP {0.292 * b_re:.1f} / {0.265 * b_re:.1f}")
    print(f"  noise budget {lg(pr.budget())}, refusal threshold {lg(pr.budget(1))}")
    print("  candidate A, transposed rows = columns of W8 (length m); max accepted L2 norm", MAX_ROW_L2)
    for name in MODELS:
        layers, hidden, kinds = MODELS[name]
        worst_layer = max(m for m, _ in kinds.values())
        for label, m in (("widest layer column", worst_layer), ("head column (vocab)", HEAD_VOCAB)):
            l2 = 127 * math.sqrt(m)
            print(f"    {name} {label}: m = {m}, all-127 L2 {l2:.0f}, noise {lg(pr.noise_bound(l2))} "
                  f"-> {'ok' if pr.noise_ok(l2) else 'REFUSED at worst case (needs measured norms)'}; "
                  f"MLWE ring samples m/16 = {m // 16} (v3 packet states <= 1848)")


def section_candidate_a() -> None:
    print("\n[3] candidate A: transposed pks-v3 product (Enc(u) . W8 computed by the provider)")
    print("  limbs: |W8^T u_l| < 2^31 needs centred limbs of b bits with 127 m 2^(b-1) < 2^31")
    print("| model | limb bits (layers / head) | limbs (31-bit u) | transposed hint | per 256-slot group: upload + reply keys "
          "| segments served per group | upload per segment amortised | vs pks-v3 traffic per segment |")
    print("|---|---|---|---|---|---|---|---|")
    for name in MODELS:
        sh = shapes(name)
        layers, hidden, kinds = MODELS[name]
        bits = {}
        for label, m in (("layers", max(m for m, _ in kinds.values())), ("head", HEAD_VOCAB)):
            bits[label] = 1 + math.floor(31 - math.log2(127 * m) - 1e-9)
        limbs = math.ceil(31 / min(bits.values()))
        sm, sn = sum(m for m, _ in sh), sum(n for _, n in sh)
        hint = sn * HINT_ROW_BYTES
        group = sm * UP_BYTES_PER_FEATURE_GROUP + len(sh) * KSK_BYTES
        per_group = 256 // limbs  # one p31 vector per linear and segment; the 14 private vectors are rarer
        up, down = TRAFFIC_MB_PER_SLOT[name]
        seg_traffic = (up + down) * 1e6 * 1024
        print(f"| {name} | {bits['layers']} / {bits['head']} | {limbs} | {gb(hint)} | {gb(group)} | {per_group} "
              f"| {mb(group / per_group)} | +{100 * group / per_group / seg_traffic:.2f} % |")
    print("  anchor needed to verify the decrypted u W8 (section 4 of the note): a secret sketch S = W8 X (m x d)")
    for name in MODELS:
        sm = sum(m for m, _ in shapes(name))
        print(f"    {name}: S costs {mb(4 * sm)} per column; d = 2 gives P[wrong uW accepted] <= {lg(prime_fresh(2))}")
    pr = PKSParams()
    print(f"  without an anchor: an additive shift delta of the decrypted uW makes the gateway accept iff "
          f"delta . xq = 0 mod p, a function of the private activation (soundness error 1 for that leak); budget "
          f"{lg(pr.budget())}")


def section_candidate_b() -> None:
    print("\n[4] candidate B: secret long-lived basis, u = B c (p31) and v = B_v c (Z_2^32)")
    print("  p31 online check, one fixed error, c = 1 vector per linear:")
    print(f"    today (fresh u):         {lg(prime_fresh(1))}")
    for k in (1, 2, 3, 4, 5, 8):
        print(f"    basis k = {k}:             {lg(prime_basis(1, k))}  (same error re-tried in 1000 segments: "
              f"{lg(prime_basis_conversation(1, k, 1000))}, today {lg(prime_fresh(1000)) if prime_fresh(1000) else '2^-31000'})")
    print("  factory private check over Z_2^32, 14 fresh combinations per linear, error valuation t (t <= 29 matters):")
    today = max(ring_fresh(14, t) for t in range(30))
    print(f"    today (14 fresh uniform vectors): worst {lg(today)}")
    for k in (14, 20, 28, 32, 42):
        worst = max(ring_basis(14, k, t) for t in range(30))
        print(f"    basis k_v = {k:2d}: worst over t <= 29 {lg(worst)} ({math.log2(worst / today):+.2f} bits vs today)")
    print("| model | G = B^T W8 (k = 5, u32) | V = B_v^T W8 (k_v = 28, u32) | total | W8 it replaces | factory GEMV work per segment: today -> B |")
    print("|---|---|---|---|---|---|")
    for name in MODELS:
        sh = shapes(name)
        sn, sm, smn = sum(n for _, n in sh), sum(m for m, _ in sh), sum(m * n for m, n in sh)
        g, v = 5 * 4 * sn, 28 * 4 * sn
        print(f"| {name} | {mb(g)} | {mb(v)} | {mb(g + v)} | {gb(smn)} | {2 * smn / 1e9:.1f} G MAC -> "
              f"{(5 * sn + 5 * sm) / 1e6:.1f} M MAC (+ gateway u = B c) |")
    print("  leak attack: k segments' u (or G, which with public W8 determines a kernel vector) give e with B^T e = 0:"
          " accepted by every later segment (demonstrated in [5])")


def section_monte_carlo(seed: int = 7) -> None:
    rng = random.Random(seed)
    print("\n[5] Monte Carlo checks of the closed forms, and the two attacks")
    # (a) prime basis at a small prime
    p, m, trials = 31, 12, 60000
    e = [rng.randrange(p) for _ in range(m)]
    e[0] = 1
    for k in (1, 2, 3):
        acc = 0
        for _ in range(trials):
            b = [[rng.randrange(p) for _ in range(k)] for _ in range(m)]
            c = [rng.randrange(p) for _ in range(k)]
            u = [sum(b[j][i] * c[i] for i in range(k)) % p for j in range(m)]
            acc += dot(u, e, p) == 0
        z = p ** -k
        print(f"  F_{p}, k = {k}: accept {acc / trials:.5f}, closed form {z + (1 - z) / p:.5f} (fresh u: {1 / p:.5f})")
    # (a') same error over T segments, fixed B
    k, T = 2, 3
    acc = 0
    for _ in range(trials):
        b = [[rng.randrange(p) for _ in range(k)] for _ in range(m)]
        ok = True
        for _ in range(T):
            c = [rng.randrange(p) for _ in range(k)]
            u = [sum(b[j][i] * c[i] for i in range(k)) % p for j in range(m)]
            ok &= dot(u, e, p) == 0
        acc += ok
    z = p ** -k
    print(f"  F_{p}, k = {k}, same error in {T} segments: accept {acc / trials:.5f}, closed form "
          f"{z + (1 - z) * p ** -T:.5f} (fresh u: {p ** -T:.6f})")
    # (b) ring basis at K = 8
    K, m, c_n = 8, 10, 2
    for k, t in ((1, 4), (2, 4), (3, 6), (2, 3)):
        acc = 0
        delta = [0] * m
        delta[3] = 1 << t
        delta[5] = (rng.randrange(1 << (K - t)) << t) % (1 << K)
        for _ in range(trials):
            b = [[rng.randrange(1 << K) for _ in range(k)] for _ in range(m)]
            w = [sum(b[j][i] * delta[j] for j in range(m)) % (1 << K) for i in range(k)]
            ok = all(sum(rng.randrange(1 << K) * wi for wi in w) % (1 << K) == 0 for _ in range(c_n))
            acc += ok
        print(f"  Z_2^{K}, k = {k}, {c_n} combinations, valuation {t}: accept {acc / trials:.5f}, closed form "
              f"{ring_basis(c_n, k, t, K):.5f} (fresh: {ring_fresh(c_n, t, K):.5f})")
    # (c) attack 1: leaked u's of k segments -> a forged error accepted forever (real p = 2^31 - 1)
    p, m, k = P31, 64, 5
    b = [[rng.randrange(p) for _ in range(k)] for _ in range(m)]

    def u_of(c: list[int]) -> list[int]:
        return [sum(b[j][i] * c[i] for i in range(k)) % p for j in range(m)]

    leaked = [u_of([rng.randrange(p) for _ in range(k)]) for _ in range(k)]
    e = left_kernel_vector(leaked, p, rng)
    later = sum(dot(u_of([rng.randrange(p) for _ in range(k)]), e, p) == 0 for _ in range(2000))
    print(f"  attack (B): u of {k} past segments leaked -> forged e accepted by {later}/2000 later segments "
          f"(fresh u today: expected 0)")
    # (d) attack 2: unanchored transposed product, additive shift -> data-dependent verdict
    n = 16
    w8 = [[rng.randrange(-127, 128) for _ in range(n)] for _ in range(m)]
    u = [rng.randrange(p) for _ in range(m)]
    uw = [sum(u[i] * w8[i][j] for i in range(m)) % p for j in range(n)]
    j = 5
    uw_bad = uw[:]
    uw_bad[j] = (uw_bad[j] + 1) % p  # provider adds delta = e_j to the decrypted uW
    verdicts = []
    for xj in (0, 0, 3, -7, 0, 1):
        xq = [rng.randrange(-2048, 2048) for _ in range(n)]
        xq[j] = xj
        yq = [sum(w8[i][l] * xq[l] for l in range(n)) for i in range(m)]  # honest provider output
        verdicts.append((xj, dot(u, [y % p for y in yq], p) == dot(uw_bad, [x % p for x in xq], p)))
    print("  attack (A, no anchor): honest output, uW shifted by e_5; accepted iff xq_5 == 0:",
          ", ".join(f"xq_5={x}:{'accept' if a else 'REJECT'}" for x, a in verdicts))


def section_candidate_c(sha_gbps: float | None) -> None:
    print("\n[6] candidate C: no W8 at rest, stream it (verified against the pinned SHA-256) when needed")
    print("| model | W8 per pass | pks-v3 traffic per 1024-slot segment | C1: a pass per segment | "
          "C2: pass per 64 segments, +stock on disk (64 uW sets + 8 private sets, u32) | SHA-256 per pass |")
    print("|---|---|---|---|---|---|")
    for name in MODELS:
        sh = shapes(name)
        sn, smn = sum(n for _, n in sh), sum(m * n for m, n in sh)
        up, down = TRAFFIC_MB_PER_SLOT[name]
        seg = (up + down) * 1e6 * 1024
        stock = 64 * 4 * sn + 8 * 14 * 4 * sn
        head = HEAD_VOCAB * MODELS[name][1]
        smn -= head  # the head W8 is derived from the plain head the gateway keeps anyway
        cpu = CLIENT_MS_PER_SLOT[name] * 1024 / 1e3
        sha = (f"{smn / (sha_gbps * 1e9):.0f} core-s (+{100 * smn / (sha_gbps * 1e9) / cpu:.0f} % of the "
               f"{cpu:.0f} core-s pks-v3 client CPU per segment)") if sha_gbps else "(--measure)"
        print(f"| {name} | {gb(smn)} | {gb(seg)} | +{100 * smn / seg:.0f} % | +{100 * smn / seg / 64:.2f} %, "
              f"{gb(stock)} | {sha} |")


def section_pcs() -> None:
    print("\n[7] anchoring A with a public commitment instead of a secret (rough, Ligero-style, t = 300 queries)")
    for name in MODELS:
        sh = shapes(name)
        smn = sum(m * n for m, n in sh)
        proof = 2 * math.sqrt(300 * smn) * 8
        print(f"  {name}: N = {smn / 1e9:.1f} G entries; one opening ~{mb(proof)}; provider O(N) per opening; "
              f"client re-encodes and hashes all of W8 once at setup to check the commitment")


def measure(pack_dir: Path | None) -> float:
    buf = os.urandom(1 << 26)
    t0 = time.perf_counter()
    reps = 8
    for _ in range(reps):
        hashlib.sha256(buf).digest()
    gbps = reps * len(buf) / (time.perf_counter() - t0) / 1e9
    print(f"\n[m] SHA-256 on this box: {gbps:.2f} GB/s per core")
    if pack_dir is None or not (pack_dir / "linears-00000-of-00001.safetensors").exists():
        print("[m] no Qwen2.5-0.5B pack found: skipping the norm measurement")
        return gbps
    import numpy as np
    import torch
    from safetensors import safe_open

    from pgw.modelpack.masked_head import quantize_head

    pr = PKSParams()
    worst_row, worst_col = (0.0, ""), (0.0, "")
    with safe_open(str(pack_dir / "linears-00000-of-00001.safetensors"), "np") as f:
        for key in f.keys():
            if not key.endswith(".weight"):
                continue
            w = f.get_tensor(key).astype(np.float64)
            r, c = float(np.sqrt((w * w).sum(1)).max()), float(np.sqrt((w * w).sum(0)).max())
            worst_row = max(worst_row, (r, key))
            worst_col = max(worst_col, (c, f"{key} (m = {w.shape[0]})"))
    print(f"[m] 0.5B layers: max row L2 {worst_row[0]:.0f} ({worst_row[1]}), max column L2 {worst_col[0]:.0f} "
          f"({worst_col[1]}); noise at the column max {lg(pr.noise_bound(worst_col[0]))}")
    with safe_open(str(pack_dir / "plain.safetensors"), "pt") as f:
        head = f.get_tensor("embed_tokens")
    w8, _ = quantize_head(head)
    w = w8.to(torch.float64)
    col = float(w.square().sum(0).sqrt().max())
    row = float(w.square().sum(1).sqrt().max())
    print(f"[m] 0.5B head (151936 x 896): max row L2 {row:.0f}, max column L2 {col:.0f} "
          f"(all-127 column {127 * math.sqrt(151936):.0f}); noise at the column max {lg(pr.noise_bound(col))} "
          f"-> {'ok' if pr.noise_ok(col) else 'REFUSED'}")
    return gbps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    a = ap.parse_args()
    sha = None
    if a.measure:
        env = os.environ.get("PGW_W8_PACK")
        default = Path.home() / ".pgw-client-demo/packs/qwen2.5-0.5b-instruct"
        sha = measure(Path(env) if env else (default if default.exists() else None))
    section_sizes()
    section_security_params()
    section_candidate_a()
    section_candidate_b()
    section_monte_carlo()
    section_candidate_c(sha)
    section_pcs()


if __name__ == "__main__":
    main()
