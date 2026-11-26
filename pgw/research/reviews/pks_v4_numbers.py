"""Numbers for ``research/reviews/pks-v4-prime-upload.md`` (run: ``PYTHONPATH=src python research/reviews/pks_v4_numbers.py``).

Same models as the pks-v3 packet: ``pgw.hemask.mlwe_ring.core_svp_beta`` (primal uSVP, 2016 estimate,
Bai-Galbraith; 0.292 beta classical / 0.265 beta quantum) and ``PKSParams.noise_terms`` (24-sigma tails,
Hoeffding at t = 24; budget q1 / 2^33). Every pks-v4 variant is expressed as edits of v3's noise table.
"""

from __future__ import annotations

import math

from pgw.hemask.mlwe_ring import core_svp_beta
from pgw.hemask.pks import NU, PS, Q1, PKSParams, _is_prime

q1, P = Q1[0] * Q1[1], PS[0] * PS[1]
LQ1, LP = math.log2(q1), math.log2(P)
SIG = math.sqrt(21 / 2)  # CBD(21)
KS = "key switch sum A~ e / P"
HS = "mod switch of H times s (Hoeffding)"
W8E = "W8 e (upload noise)"


def sec(n: int, log_q: float, sigma: float) -> str:
    b = core_svp_beta(n, log_q, sigma)
    return f"beta {b}, core-SVP {0.292 * b:.1f} classical / {0.265 * b:.1f} quantum"


def lg(x: float) -> str:
    return f"2^{math.log2(x):.2f}"


def total(d: dict[str, float]) -> str:
    return lg(sum(d.values()))


def main() -> None:
    pr = PKSParams()
    cases = {  # (L2 norm, L1 norm) of the worst row
        "worst case (all-127 rows, n = 29568, 72B)": (127 * math.sqrt(29568), 127 * 29568),
        "Qwen2.5-7B down_proj at real norms (0.37 x all-127, n = 18944)": (0.37 * 127 * math.sqrt(18944), None),
        "Qwen2.5-0.5B q/k/v/o at real norms (n = 896)": (0.37 * 127 * math.sqrt(896), None),
    }
    print(f"q1 = 2^{LQ1:.2f}, P = 2^{LP:.2f}; budget {lg(pr.budget())}, refusal threshold (1-bit margin) {lg(pr.budget(1))}")
    print(f"65537 prime: {_is_prime(65537)}, = 1 mod 16384: {65537 % 16384 == 1}")
    print("\n[v3]")
    print("upload  (n 4096, log q 56):", sec(NU, 56, SIG))
    print("reply   (n 8192, log q %.2f):" % (LQ1 + LP), sec(8192, LQ1 + LP, SIG))
    for name, (l2, _) in cases.items():
        print(f"  noise, {name}: {total(pr.noise_terms(l2))}")

    print("\n[V0: upload modulus = q1 exactly (the section-8 sketch), CBD(21) and CBD(1)]")
    print("upload  CBD(21), log q 47.98:", sec(NU, LQ1, SIG))
    print("upload  CBD(1),  log q 47.98:", sec(NU, LQ1, math.sqrt(0.5)))
    print("sigma that keeps v3's sigma/q at q1: %.4f" % (SIG * q1 / 2 ** 56))
    for eta in (21, 1):
        s = math.sqrt(eta / 2)
        for name, (l2, _) in cases.items():
            d = pr.noise_terms(l2)
            d[W8E] = pr.tail * s * l2  # no 2^56 -> q1 scaling any more
            d.pop(HS)
            d["encoding round(q1 M / 2^32) (Hoeffding)"] = pr.tail * 0.5 * l2
            print(f"  CBD({eta}) {name}: W8 e {lg(d[W8E])}, total {total(d)}")

    print("\n[V1/V2: upload modulus Q = q1 * f, sigma scaled by Q / 2^56 (v3's sigma/q), ModDown by P f]")
    for fname, f in (("65537 (smallest prime = 1 mod 16384)", 65537), ("2^8 (not prime: 2-adic part)", 256)):
        lq = LQ1 + math.log2(f)
        su = SIG * 2 ** lq / 2 ** 56
        print(f" f = {fname}: Q = 2^{lq:.2f}, sigma_up {su:.1f}; upload: {sec(NU, lq, su)}")
        l2, l1 = cases["worst case (all-127 rows, n = 29568, 72B)"]
        d = pr.noise_terms(l2)
        d.pop(HS)
        d["encoding (worst case |W8 eps| / f)"] = l1 / 2 / f
        dh = dict(d)
        dh["encoding (worst case |W8 eps| / f)"] = pr.tail * 0.5 * l2 / f
        print(f"   (a) same KS key (sigma, P): noise {total(d)} (Hoeffding encoding {total(dh)}); "
              f"reply key log q {lq + LP:.2f}: {sec(8192, lq + LP, SIG)}")
        d2 = dict(d)
        d2[KS] *= f
        print(f"   (b) same reply-key q/sigma (sigma_ks x f): KS term {lg(d2[KS])}, noise {total(d2)}; "
              f"reply: {sec(8192, lq + LP, SIG * f)}")
        target = sum(pr.noise_terms(l2).values())
        x = (target - (sum(dh.values()) - dh[KS])) / dh[KS]
        if x > 1:
            print(f"   (c) equal correctness with v3 (sigma_ks x {x:.1f}): reply: {sec(8192, lq + LP, SIG * x)}")
        else:
            print("   (c) equal correctness with v3: impossible even at sigma_ks x 1")

    print("\n[V3: KS operand = unreduced integer H = W8 A (A centred in Q), so C is linear mod every prime]")
    l2, l1 = cases["worst case (all-127 rows, n = 29568, 72B)"]
    # |A~| grows from q1/2 (v3) to |H_int| <= l1 * Q / 2 (worst case) or 24 l2 Q / sqrt(12) (Gaussian tail)
    for nm, g in (("worst case", l1 * 65537), ("24-sigma", 24 * l2 * 65537 / math.sqrt(12) * 2)):
        bits = math.log2(g)  # growth of the KS operand relative to q1/2, absorbed by a larger P
        lq = LQ1 + 16 + LP + bits - 16  # Q = q1 * 65537, P' = P * g / 65537 keeps the v3 KS term
        print(f"  {nm}: P grows by 2^{bits - 16:.1f}; reply key log q {lq:.1f}: {sec(8192, lq, SIG)}")

    rows = 1.54e6  # hint rows at Qwen2.5-7B incl. the masked head (pks-v3-provider-hint.md, section 7)
    coefs = rows * 256
    print("\n[download at 7B]")
    print(f"  stored hint c1: {coefs * 6 / 1e9:.2f} GB")
    print(f"  V0 c1 + [C]_P (60 bits): {coefs * (6 + 60 / 8) / 1e9:.2f} GB")
    print(f"  V1 c1 + [C]_(P 65537) (76 bits): {coefs * (6 + 76 / 8) / 1e9:.2f} GB")
    print(f"  lift quotients for H mod Q (22 bits x 4096 per row): {rows * 4096 * 22 / 8 / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
