"""Symbolic (linear) bound propagation, concretised exactly over the input domain.

Every intermediate vector v is sandwiched between affine functions of the
flattened input x:  A_l x + c_l <= v <= A_u x + c_u. Affine layers are exact;
unstable ReLUs use the standard triangle relaxation (upper chord, lower slope
0 or 1). Bounds are concretised over either
  - a box  {x : |x - x0|_inf <= eps}, or
  - the one-hot domain (each position is exactly one token): the minimum of a
    linear function is the sum over positions of the per-position minimum,
    which is exact, not a relaxation.
Results are intersected with IBP and returned in the same format as
`verify.np_ibp`, so they drop straight into the MILP encoder.
"""
import numpy as np


class Box:
    def __init__(self, x0, eps):
        self.x0, self.eps = x0.reshape(-1), eps
        self.in_bounds = (self.x0 - eps, self.x0 + eps)

    def lower(self, A, c):
        return A @ self.x0 - self.eps * np.abs(A).sum(1) + c


class OneHot:
    def __init__(self, L, V):
        self.L, self.V = L, V
        self.in_bounds = (np.zeros(L * V), np.ones(L * V))

    def lower(self, A, c):
        return A.reshape(A.shape[0], self.L, self.V).min(2).sum(1) + c


def _affine(W, b, Al, cl, Au, cu):
    Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
    return Wp @ Al + Wn @ Au, Wp @ cl + Wn @ cu + b, Wp @ Au + Wn @ Al, Wp @ cu + Wn @ cl + b


def _conc(dom, Al, cl, Au, cu):
    return dom.lower(Al, cl), -dom.lower(-Au, -cu)


def symbolic_bounds(P, dom, ibp=None):
    L, V, d = P["L"], P["V"], P["d"]
    n = L * V
    # embedding is exact: s0 = kron(I, W_in) x + b
    E = np.kron(np.eye(L), P["W_in"])
    Al = Au = E
    cl = cu = np.tile(P["b_in"], L)
    out = {"in": dom.in_bounds, "s": [], "z": []}

    def tighten(key, idx, l, u):
        if ibp is not None:
            il, iu = ibp[key][idx]
            l, u = np.maximum(l, il), np.minimum(u, iu)
        return l, u

    out["s"].append(tighten("s", 0, *_conc(dom, Al, cl, Au, cu)))
    k = 1
    for blk, (M, W1, b1, W2, b2) in enumerate(zip(P["mix"], P["W1"], P["b1"], P["W2"], P["b2"])):
        Mx = np.kron(np.eye(L) + M, np.eye(d))
        Al, cl, Au, cu = _affine(Mx, 0.0, Al, cl, Au, cu)
        out["s"].append(tighten("s", k, *_conc(dom, Al, cl, Au, cu))); k += 1
        K1 = np.kron(np.eye(L), W1)
        zAl, zcl, zAu, zcu = _affine(K1, np.tile(b1, L), Al, cl, Au, cu)
        l, u = tighten("z", blk, *_conc(dom, zAl, zcl, zAu, zcu))
        out["z"].append((l, u))
        # ReLU relaxation, row-wise
        act = l >= 0
        dead = u <= 0
        unst = ~act & ~dead
        lam = np.where(unst, u / np.where(unst, u - l, 1.0), 0.0)
        up_slope = np.where(act, 1.0, lam)
        up_off = np.where(unst, -lam * l, 0.0)
        lo_slope = np.where(act, 1.0, np.where(unst & (u >= -l), 1.0, 0.0))
        aAu, acu = up_slope[:, None] * zAu, up_slope * zcu + up_off
        aAl, acl = lo_slope[:, None] * zAl, lo_slope * zcl
        K2 = np.kron(np.eye(L), W2)
        dAl, dcl, dAu, dcu = _affine(K2, np.tile(b2, L), aAl, acl, aAu, acu)
        Al, cl, Au, cu = Al + dAl, cl + dcl, Au + dAu, cu + dcu
        out["s"].append(tighten("s", k, *_conc(dom, Al, cl, Au, cu))); k += 1
    out["final_sym"] = (Al, cl, Au, cu)
    assert Al.shape[1] == n
    return out
