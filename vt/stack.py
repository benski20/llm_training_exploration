"""Stack-state models: exactly verifiable models of nested, typed structure.

Count-state models (vt/model.py uniform, vt/gen.py) summarise the past by token
counts, which cannot match two kinds of bracket (Dyck-2) or JSON-like nesting.
Here the summary is a bounded stack:

* token roles come from the format: openers push themselves, closers pop
  (`roles`: one of "push", "pop", "none" per token);
* the stack keeps at most D symbols; `depth` counts the true depth (up to L),
  so overflow is visible to the model;
* everything learned is a function of (current token, stack top, depth), so a
  dynamic program over (stack contents, depth, spec state) decides the
  model's behaviour on all V^L strings exactly, in integer arithmetic.

StackNet is a classifier: margin = sum_t readout . h(x_t, top, depth) plus an
end-of-sequence step that sees the final (top, depth). StackGen is a
generator: next-token scores = relu(f(top, depth, remaining)), normalised.
"""
import math
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMPTY = -1


# --- spec: Dyck-k with bounded nesting depth -------------------------------------

class DyckKSpec:
    """k bracket types over alphabet like "()[]{}" (opener, closer pairs); a
    string is valid iff it is well nested, every closer matches the most recent
    unmatched opener, and nesting never exceeds max_depth."""
    n_classes = 2

    def __init__(self, alphabet="()[]", max_depth=4):
        if len(alphabet) % 2:
            raise ValueError("alphabet must be opener/closer pairs")
        self.alphabet, self.max_depth = alphabet, max_depth
        self.V = len(alphabet)
        self.name = f"dyck{self.V // 2}_d{max_depth}"
        self.roles = ["push" if i % 2 == 0 else "pop" for i in range(self.V)]

    def canonical(self):
        return ("dyckk", self.alphabet, self.max_depth)

    def dfa(self, L):
        def step(q, v):
            if q == "dead":
                return q
            if v % 2 == 0:
                return "dead" if len(q) >= self.max_depth else q + (v,)
            return q[:-1] if q and q[-1] == v - 1 else "dead"
        return (), step, lambda q: int(q == ())

    def label(self, seq):
        init, step, lab = self.dfa(len(seq))
        q = init
        for v in seq:
            q = step(q, int(v))
        return lab(q)

    def label_batch(self, seqs):
        return np.array([self.label(s) for s in seqs], dtype=np.int64)

    def sample(self, n, L, rng):
        from .counterspec import uniform_accepted
        k = n // 3
        uni = rng.integers(0, self.V, size=(k, L))
        pos = uniform_accepted(self, L, n - k, rng)
        near = pos[:k].copy()
        near[np.arange(k), rng.integers(L, size=k)] = rng.integers(0, self.V, size=k)
        return np.concatenate([uni, pos[k:], near])


# --- stack dynamics shared by models and verifiers -------------------------------

def stack_step(stack, depth, v, roles, D):
    """Apply token v's role. Returns (stack', depth')."""
    if roles[v] == "push":
        return (stack + (v,))[-D:] if len(stack) < D else stack, depth + 1
    if roles[v] == "pop":
        return stack[:-1], max(0, depth - 1)
    return stack, depth


def _run_stacks(x, roles, D):
    """For each sequence and position: (top before, depth before); plus final."""
    B, L = x.shape
    tops = np.full((B, L + 1), EMPTY, dtype=np.int64)
    deps = np.zeros((B, L + 1), dtype=np.int64)
    for b in range(B):
        st, dep = (), 0
        for t in range(L):
            tops[b, t] = st[-1] if st else EMPTY
            deps[b, t] = dep
            st, dep = stack_step(st, dep, int(x[b, t]), roles, D)
        tops[b, L] = st[-1] if st else EMPTY
        deps[b, L] = dep
    return torch.from_numpy(tops), torch.from_numpy(deps)


# --- classifier -------------------------------------------------------------------

class StackNet(nn.Module):
    def __init__(self, vocab, roles, D=4, d_model=8, d_hidden=32):
        super().__init__()
        V, d, h = vocab, d_model, d_hidden
        self.V, self.roles, self.D, self.d, self.h = V, list(roles), D, d, h
        self.E = nn.Parameter(torch.randn(V + 1, d) / math.sqrt(d))      # row V = end-of-sequence
        self.T = nn.Parameter(torch.randn(V + 1, d) / math.sqrt(d))      # stack top; row V = empty
        self.u = nn.Parameter(torch.randn(d) / math.sqrt(d))             # depth direction
        self.W1 = nn.Parameter(torch.randn(h, d) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(h))
        self.W2 = nn.Parameter(torch.randn(d, h) / math.sqrt(h))
        self.b2 = nn.Parameter(torch.zeros(d))
        self.r_pos = nn.Parameter(torch.randn(2, d) / math.sqrt(d))      # per-position readout
        self.r_end = nn.Parameter(torch.randn(2, d) / math.sqrt(d))      # end-of-sequence readout
        self.b_out = nn.Parameter(torch.zeros(2))

    def block(self, tok, top, dep):
        s = self.E[tok] + self.T[torch.where(top < 0, self.V, top)] + dep.unsqueeze(-1) * self.u
        return s + F.relu(s @ self.W1.T + self.b1) @ self.W2.T + self.b2

    def forward(self, x):
        tops, deps = _run_stacks(x.numpy(), self.roles, self.D)
        B, L = x.shape
        h = self.block(x, tops[:, :L], deps[:, :L].float())
        h_end = self.block(torch.full((B,), self.V), tops[:, L], deps[:, L].float())
        return h.sum(1) @ self.r_pos.T + h_end @ self.r_end.T + self.b_out

    @torch.no_grad()
    def export(self):
        f = lambda t: t.detach().double().numpy()
        return {"kind": "stacknet", "V": self.V, "D": self.D, "roles": list(self.roles),
                **{k: f(getattr(self, k)) for k in
                   ("E", "T", "u", "W1", "b1", "W2", "b2", "r_pos", "r_end", "b_out")}}


def _ints(P, keys):
    K = 0
    for k in keys:
        for v in np.asarray(P[k], dtype=np.float64).ravel():
            if not np.isfinite(v):
                raise ValueError("non-finite weight")
            K = max(K, Fraction(float(v)).denominator.bit_length() - 1)
    S = 1 << K
    conv = lambda a: np.vectorize(lambda v: int(Fraction(float(v)) * S), otypes=[object])(
        np.asarray(a, dtype=np.float64))
    return S, {k: conv(P[k]) for k in keys}


def verify_stacknet(P, spec, L, tau=1e-4):
    """Exact worst-case margin over all V^L strings (DP over stack, depth, spec state)."""
    t0 = time.perf_counter()
    V, D, roles = P["V"], P["D"], P["roles"]
    S, W = _ints(P, ("E", "T", "u", "W1", "b1", "W2", "b2", "r_pos", "r_end", "b_out"))
    cache = {}

    def h(tok, top, dep):
        key = (tok, top, dep)
        if key not in cache:
            s = W["E"][tok] + W["T"][V if top == EMPTY else top] + dep * W["u"]     # scale S
            z = W["W1"].dot(s) + W["b1"] * S                                        # S^2
            r = np.array([zi if zi > 0 else 0 for zi in z], dtype=object)
            cache[key] = s * S * S + W["W2"].dot(r) + W["b2"] * S * S              # S^3
        return cache[key]
    init, step, label = spec.dfa(L)
    dr = {(y, j): W["r_pos"][y] - W["r_pos"][j] for y in (0, 1) for j in (0, 1) if y != j}
    frontier = {((), 0, init): {p: 0 for p in dr}}
    for t in range(L):
        nxt = {}
        for (st, dep, q), vals in frontier.items():
            for v in range(V):
                hv = h(v, st[-1] if st else EMPTY, dep)
                st2, dep2 = stack_step(st, dep, v, roles, D)
                key = (st2, dep2, step(q, v))
                slot = nxt.setdefault(key, {})
                for p, w in dr.items():
                    val = vals[p] + w.dot(hv)
                    if p not in slot or val < slot[p]:
                        slot[p] = val
        frontier = nxt
    worst = None
    for (st, dep, q), vals in frontier.items():
        y = label(q)
        he = h(V, st[-1] if st else EMPTY, dep)
        m = vals[(y, 1 - y)] + (W["r_end"][y] - W["r_end"][1 - y]).dot(he) \
            + (W["b_out"][y] - W["b_out"][1 - y]) * S ** 3                          # S^4
        if worst is None or m < worst:
            worst = m
    wq = Fraction(worst, S ** 4)
    return {"status": "proved" if wq > Fraction(tau) else "counterexample", "method": "exact-stack-dp",
            "min_margin": str(wq), "min_margin_float": float(wq), "time": time.perf_counter() - t0}


# --- generator --------------------------------------------------------------------

class StackGen(nn.Module):
    """Next-token scores relu(f(top, depth, remaining)), normalised (exact zeros)."""

    def __init__(self, vocab, roles, D=4, d_model=8, d_hidden=32):
        super().__init__()
        V, d, h = vocab, d_model, d_hidden
        self.V, self.roles, self.D = V, list(roles), D
        self.T = nn.Parameter(torch.randn(V + 1, d) / math.sqrt(d))
        self.u = nn.Parameter(torch.randn(d) / math.sqrt(d))       # depth
        self.ur = nn.Parameter(torch.randn(d) / math.sqrt(d))      # remaining tokens
        self.W1 = nn.Parameter(torch.randn(h, d) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(h))
        self.Wo = nn.Parameter(torch.randn(V, h) / math.sqrt(h))
        self.bo = nn.Parameter(torch.ones(V))

    def logits(self, top, dep, rem):
        s = self.T[torch.where(top < 0, self.V, top)] + dep.unsqueeze(-1) * self.u + rem.unsqueeze(-1) * self.ur
        return F.relu(s @ self.W1.T + self.b1) @ self.Wo.T + self.bo

    def log_prob(self, x, eps):
        tops, deps = _run_stacks(x.numpy(), self.roles, self.D)
        B, L = x.shape
        rem = (L - torch.arange(L)).float().expand(B, L)
        sc = F.relu(self.logits(tops[:, :L], deps[:, :L].float(), rem)) + eps
        p = sc / sc.sum(-1, keepdim=True)
        return torch.log(p.gather(-1, x[..., None]).squeeze(-1)).sum(-1)

    @torch.no_grad()
    def sample(self, n, L, generator=None):
        out = np.zeros((n, L), dtype=np.int64)
        for i in range(n):
            st, dep = (), 0
            for t in range(L):
                top = torch.tensor([st[-1] if st else EMPTY])
                sc = F.relu(self.logits(top, torch.tensor([float(dep)]), torch.tensor([float(L - t)])))[0]
                if sc.sum() <= 0:
                    raise RuntimeError("dead end")
                v = int(torch.multinomial(sc / sc.sum(), 1, generator=generator))
                out[i, t] = v
                st, dep = stack_step(st, dep, v, self.roles, self.D)
        return out

    @torch.no_grad()
    def export(self):
        f = lambda t: t.detach().double().numpy()
        return {"kind": "stackgen", "V": self.V, "D": self.D, "roles": list(self.roles),
                **{k: f(getattr(self, k)) for k in ("T", "u", "ur", "W1", "b1", "Wo", "bo")}}


def _gen_graph(V, roles, D, L, spec):
    """Product states (stack, depth, q) per layer, transitions and feasibility."""
    init, step, label = spec.dfa(L)
    layers = [[((), 0, init)]]
    for _ in range(L):
        seen = {}
        for (st, dep, q) in layers[-1]:
            for v in range(V):
                st2, dep2 = stack_step(st, dep, v, roles, D)
                seen.setdefault((st2, dep2, step(q, v)), None)
        layers.append(list(seen))
    good = [None] * (L + 1)
    good[L] = {s for s in layers[L] if label(s[2]) == 1}
    for t in range(L - 1, -1, -1):
        good[t] = set()
        for s in layers[t]:
            st, dep, q = s
            if any((*stack_step(st, dep, v, roles, D), step(q, v)) in good[t + 1] for v in range(V)):
                good[t].add(s)
    return layers, good, step, label


def verify_stackgen(G, spec, L):
    """Exact support certificate: sound, complete, live at length L."""
    t0 = time.perf_counter()
    V, D, roles = G["V"], G["D"], G["roles"]
    S, W = _ints(G, ("T", "u", "ur", "W1", "b1", "Wo", "bo"))
    layers, good, step, label = _gen_graph(V, roles, D, L, spec)

    def scores(st, dep, rem):
        s = W["T"][V if not st else st[-1]] + dep * W["u"] + rem * W["ur"]       # S
        z = W["W1"].dot(s) + W["b1"] * S                                          # S^2
        r = np.array([zi if zi > 0 else 0 for zi in z], dtype=object)
        return [max(int(o), 0) for o in W["Wo"].dot(r) + W["bo"] * S * S]        # S^3
    reach = {((), 0, spec.dfa(L)[0])}
    sound = complete = live = True
    for t in range(L):
        nxt = set()
        for (st, dep, q) in reach:
            sc = scores(st, dep, L - t)
            if sum(sc) == 0:
                live = False
                continue
            for v in range(V):
                st2, dep2 = stack_step(st, dep, v, roles, D)
                key = (st2, dep2, step(q, v))
                ok = key in good[t + 1]
                if sc[v] > 0:
                    nxt.add(key)
                    if not ok:
                        sound = False
                elif ok:
                    complete = False
        reach = nxt
    status = "proved" if (sound and complete and live) else "failed"
    return {"status": status, "sound": sound, "complete": complete, "live": live,
            "method": "exact-stack-gen-dp", "time": time.perf_counter() - t0}


def train_stackgen(spec, L, lengths=(), D=None, d_hidden=32, steps=3000, lam=10.0, margin=0.5,
                   lr=1e-2, eps=1e-3, n_data=20000, seed=0, log=print):
    """NLL on spec-uniform data + hinge over every live (state, token) pair."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    D = D or spec.max_depth
    lengths = sorted(set(lengths) | {L})
    m = StackGen(spec.V, spec.roles, D, d_hidden=d_hidden)
    from .counterspec import uniform_accepted
    data = {Lk: torch.from_numpy(uniform_accepted(spec, Lk, n_data, rng)) for Lk in lengths}
    pairs = []          # (top, depth, remaining, feasible-mask) over all live states and lengths
    for Lk in lengths:
        layers, good, step, label = _gen_graph(spec.V, spec.roles, D, Lk, spec)
        for t in range(Lk):
            for (st, dep, q) in good[t]:
                feas = [((*stack_step(st, dep, v, spec.roles, D), step(q, v)) in good[t + 1])
                        for v in range(spec.V)]
                pairs.append((st[-1] if st else EMPTY, dep, Lk - t, feas))
    top = torch.tensor([p[0] for p in pairs]); dep = torch.tensor([float(p[1]) for p in pairs])
    rem = torch.tensor([float(p[2]) for p in pairs]); feas = torch.tensor([p[3] for p in pairs])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for i in range(steps):
        Lk = lengths[i % len(lengths)]
        xb = data[Lk][torch.randint(len(data[Lk]), (64,))]
        nll = -m.log_prob(xb, eps).mean() / Lk
        z = m.logits(top, dep, rem)
        hinge = F.relu(torch.where(feas, margin - z, margin + z)).mean()
        loss = nll + lam * hinge
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if i % 500 == 0 or i == steps - 1:
            log({"step": i, "nll_per_token": nll.item(), "hinge": hinge.item(), "live_pairs": len(pairs)})
    return m
