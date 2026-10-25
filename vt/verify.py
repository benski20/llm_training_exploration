"""Sound verification of local robustness for exported MixerNet weights.

Property: for every x' with ||x' - x||_inf <= eps, argmax f(x') == y.

`verify` first tries interval bound propagation (cheap, incomplete). If IBP
cannot decide, it builds an exact MILP (big-M ReLU encoding, bounds from IBP)
and solves it with HiGHS via scipy. Counterexamples are re-checked with a
plain float64 forward pass.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


def relu(x):
    return np.maximum(x, 0.0)


def np_forward(P, x):
    s = x @ P["W_in"].T + P["b_in"]
    for M, W1, b1, W2, b2 in zip(P["mix"], P["W1"], P["b1"], P["W2"], P["b2"]):
        s = s + M @ s
        s = s + relu(s @ W1.T + b1) @ W2.T + b2
    return P["W_out"] @ s.reshape(-1) + P["b_out"]


def np_ibp(P, x, eps):
    """Bounds on every intermediate vector (flattened position-major)."""
    L = P["L"]
    W = P["W_in"]
    c, r = x @ W.T + P["b_in"], np.full_like(x, eps) @ np.abs(W).T
    out = {"in": (x.reshape(-1) - eps, x.reshape(-1) + eps), "s": [], "z": []}
    out["s"].append((c - r, c + r))
    for M, W1, b1, W2, b2 in zip(P["mix"], P["W1"], P["b1"], P["W2"], P["b2"]):
        IM = np.eye(L) + M
        c, r = IM @ c, np.abs(IM) @ r
        out["s"].append((c - r, c + r))
        zc, zr = c @ W1.T + b1, r @ np.abs(W1).T
        l, u = zc - zr, zc + zr
        out["z"].append((l, u))
        al, au = relu(l), relu(u)
        c = c + ((al + au) / 2) @ W2.T + b2
        r = r + ((au - al) / 2) @ np.abs(W2).T
        out["s"].append((c - r, c + r))
    out["s"] = [(a.reshape(-1), b.reshape(-1)) for a, b in out["s"]]
    out["z"] = [(a.reshape(-1), b.reshape(-1)) for a, b in out["z"]]
    out["final"] = (c.reshape(-1), r.reshape(-1))
    return out


def ibp_margins(P, x, y, eps, bounds=None):
    b = bounds or np_ibp(P, x, eps)
    c, r = b["final"]
    Wd = P["W_out"][y][None, :] - P["W_out"]
    bd = P["b_out"][y] - P["b_out"]
    lb = Wd @ c - np.abs(Wd) @ r + bd
    return np.delete(lb, y)


def count_unstable(bounds):
    return int(sum(((l < 0) & (u > 0)).sum() for l, u in bounds["z"]))


class _MILP:
    def __init__(self):
        self.lb, self.ub, self.integ = [], [], []
        self.rows, self.cols, self.vals = [], [], []
        self.rlb, self.rub = [], []
        self.n_rows = 0

    def vars(self, lb, ub, integer=False):
        start = len(self.lb)
        self.lb.extend(lb); self.ub.extend(ub)
        self.integ.extend([int(integer)] * len(lb))
        return np.arange(start, start + len(lb))

    def eq(self, out_idx, terms, b):
        """out = sum_k A_k @ in_k + b."""
        n = len(out_idx)
        r0 = self.n_rows
        self.rows.append(np.arange(r0, r0 + n)); self.cols.append(out_idx)
        self.vals.append(np.ones(n))
        for A, in_idx in terms:
            A = sp.coo_matrix(A)
            self.rows.append(r0 + A.row); self.cols.append(in_idx[A.col])
            self.vals.append(-A.data)
        self.rlb.extend(b); self.rub.extend(b)
        self.n_rows += n

    def row(self, entries, lo, hi):
        for c, v in entries:
            self.rows.append(np.array([self.n_rows])); self.cols.append(np.array([c]))
            self.vals.append(np.array([v]))
        self.rlb.append(lo); self.rub.append(hi)
        self.n_rows += 1

    def solve(self, obj_idx, obj_coef, time_limit):
        n = len(self.lb)
        c = np.zeros(n); c[obj_idx] = obj_coef
        A = sp.csr_matrix((np.concatenate(self.vals),
                           (np.concatenate(self.rows), np.concatenate(self.cols))),
                          shape=(self.n_rows, n))
        return milp(c, constraints=LinearConstraint(A, self.rlb, self.rub),
                    integrality=np.array(self.integ), bounds=Bounds(self.lb, self.ub),
                    options={"time_limit": time_limit, "disp": False})


def milp_min_margin(P, x, y, j, eps, bounds, time_limit):
    """Minimise logit_y - logit_j over the eps-ball. Returns scipy result,
    the input-variable indices, and the number of binary variables."""
    L, d = P["L"], P["d"]
    m = _MILP()
    xi = m.vars(*bounds["in"])
    s = m.vars(*bounds["s"][0])
    m.eq(s, [(sp.kron(sp.eye(L), P["W_in"]), xi)], np.tile(P["b_in"], L))
    k = 1
    n_bin = 0
    for M, W1, b1, W2, b2, (zl, zu) in zip(P["mix"], P["W1"], P["b1"], P["W2"],
                                           P["b2"], bounds["z"]):
        s2 = m.vars(*bounds["s"][k]); k += 1
        m.eq(s2, [(sp.kron(sp.eye(L) + M, sp.eye(d)), s)], np.zeros(L * d))
        s = s2
        z = m.vars(zl, zu)
        m.eq(z, [(sp.kron(sp.eye(L), W1), s)], np.tile(b1, L))
        a = m.vars(np.zeros_like(zl), np.maximum(zu, 0))
        for i in range(len(zl)):
            l, u = zl[i], zu[i]
            if u <= 0:
                m.ub[a[i]] = 0.0
            elif l >= 0:
                m.row([(a[i], 1.0), (z[i], -1.0)], 0.0, 0.0)
            else:
                dlt = m.vars([0.0], [1.0], integer=True)[0]
                n_bin += 1
                m.row([(a[i], 1.0), (z[i], -1.0)], 0.0, np.inf)              # a >= z
                m.row([(a[i], 1.0), (dlt, -u)], -np.inf, 0.0)                # a <= u*delta
                m.row([(a[i], 1.0), (z[i], -1.0), (dlt, -l)], -np.inf, -l)   # a <= z - l(1-delta)
        s2 = m.vars(*bounds["s"][k]); k += 1
        m.eq(s2, [(sp.eye(L * d), s), (sp.kron(sp.eye(L), W2), a)], np.tile(b2, L))
        s = s2
    coef = P["W_out"][y] - P["W_out"][j]
    const = P["b_out"][y] - P["b_out"][j]
    res = m.solve(s, coef, time_limit)
    return res, const, xi, n_bin


def verify(P, x, y, eps, time_limit=30.0):
    """Returns dict(status in {verified, falsified, unknown}, method, time, ...)."""
    t0 = time.perf_counter()
    x = np.asarray(x, dtype=np.float64)
    logits = np_forward(P, x)
    info = {"unstable": 0, "binaries": 0}
    if int(np.argmax(logits)) != y:
        return {"status": "falsified", "method": "clean", "time": time.perf_counter() - t0, **info}
    bounds = np_ibp(P, x, eps)
    info["unstable"] = count_unstable(bounds)
    if (ibp_margins(P, x, y, eps, bounds) > 0).all():
        return {"status": "verified", "method": "ibp", "time": time.perf_counter() - t0, **info}
    status = "verified"
    for j in range(P["C"]):
        if j == y:
            continue
        res, const, xi, n_bin = milp_min_margin(P, x, y, j, eps, bounds,
                                                max(time_limit - (time.perf_counter() - t0), 1.0))
        info["binaries"] += n_bin
        dual = getattr(res, "mip_dual_bound", None)
        if res.x is not None and res.fun + const <= 0:
            x_adv = res.x[xi].reshape(x.shape)
            margin = np_forward(P, x_adv)
            margin = margin[y] - margin[j]
            info["cex_margin"] = float(margin)
            info["cex_linf"] = float(np.abs(x_adv - x).max())
            if margin <= 1e-6:
                status = "falsified"
                break
            # numerically spurious: margin is ~0 but positive; cannot claim either way
            status = "unknown"
        elif res.status == 0:
            continue  # optimal and margin > 0
        elif dual is not None and dual + const > 0:
            continue
        else:
            status = "unknown"
    return {"status": status, "method": "milp", "time": time.perf_counter() - t0, **info}
