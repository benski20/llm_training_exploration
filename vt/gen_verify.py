"""Exact certificates for count-state generators (vt/gen.py).

The generator's next-token distribution depends only on (last w tokens, counts
of older tokens). Pairing that with the spec's DFA gives a finite product
graph per length L whose nodes are (counts, recent, q). On it we decide,
exactly (integer arithmetic at scale 2^K, as in vt/exact.py):

  sound      every string the generator can emit (probability > 0) is in the spec,
  complete   every spec string of length L has probability > 0,
  live       no reachable prefix has all next-token scores zero (sampling never stalls),

and compute the exact probability mass on spec violations (0 when sound).
The same graph drives training (`state_graph`), because it is enumerable: the
loss can cover every (state, token) pair, not just sampled data.
"""
import time
from fractions import Fraction

import numpy as np


def _dfa_layers(spec, L, V):
    """reach[t]: DFA states reachable after t tokens; coreach[r]: states from which
    an accepting state is reachable in exactly r more tokens."""
    init, step, label = spec.dfa(L)
    reach = [{init}]
    for _ in range(L):
        reach.append({step(q, v) for q in reach[-1] for v in range(V)})
    co = [{q for q in reach[L] if label(q) == 1}]
    for r in range(1, L + 1):
        co.append({q for q in reach[L - r] if any(step(q, v) in co[r - 1] for v in range(V))})
    return init, step, label, reach, co


def state_graph(V, w, L, spec):
    """Layers of product states and their transitions.

    Returns a list of layers; layer t holds the states after t tokens as tuples
    (counts, recent, q), plus for each state and token v: the next state's index
    and whether v keeps the spec satisfiable ("feasible")."""
    init, step, label, reach, co = _dfa_layers(spec, L, V)
    layers = [[((0,) * V, (), init)]]
    trans, feas = [], []
    for t in range(L):
        index, nxt, tr, fe = {}, [], [], []
        for (counts, recent, q) in layers[t]:
            row_t, row_f = [], []
            for v in range(V):
                r2, c2 = recent + (v,), counts
                if len(r2) > w:
                    old, r2 = r2[0], r2[1:]
                    c2 = tuple(c + (u == old) for u, c in enumerate(counts))
                key = (c2, r2, step(q, v))
                if key not in index:
                    index[key] = len(nxt)
                    nxt.append(key)
                row_t.append(index[key])
                row_f.append(key[2] in co[L - t - 1])
            tr.append(row_t)
            fe.append(row_f)
        layers.append(nxt)
        trans.append(np.array(tr, dtype=np.int64))
        feas.append(np.array(fe, dtype=bool))
    labels = np.array([label(q) for (_, _, q) in layers[L]])
    return {"layers": layers, "trans": trans, "feasible": feas, "final_label": labels}


class _ExactGen:
    def __init__(self, G):
        from .exact import StructureError
        arrays = [G[k] for k in ("E", "kern", "W1", "b1", "Wo", "bo", "z0", "u", "u0")]
        K = 0
        for a in arrays:
            for f in np.asarray(a, dtype=np.float64).ravel():
                if not np.isfinite(f):
                    raise StructureError("non-finite weight")
                K = max(K, Fraction(float(f)).denominator.bit_length() - 1)
        S = 1 << K
        ints = lambda a: np.vectorize(lambda f: int(Fraction(float(f)) * S), otypes=[object])(
            np.asarray(a, dtype=np.float64))
        self.S, self.w = S, G["w"]
        self.E, self.k = ints(G["E"]), ints(G["kern"])
        self.W1, self.b1 = ints(G["W1"]), ints(G["b1"])
        self.Wo, self.bo, self.z0 = ints(G["Wo"]), ints(G["bo"]), ints(G["z0"])
        self.u, self.u0 = ints(G["u"]), ints(G["u0"])

    def scores(self, state, r):
        """Exact nonnegative next-token scores (common scale) after this prefix
        state, with r tokens still to generate (including the next one)."""
        counts, recent, _ = state
        if not recent:                                  # empty prefix
            return [max(int(z), 0) for z in self.z0 + r * self.u0]
        S, E, k, w = self.S, self.E, self.k, self.w
        s = (S + k[0]) * E[recent[-1]] + r * S * self.u          # scale S^2
        for i in range(1, min(w, len(recent))):
            s = s + k[i] * E[recent[-1 - i]]
        for u, c in enumerate(counts):
            if c:
                s = s + (k[w] * c) * E[u]
        z = self.W1.dot(s) + self.b1 * S * S             # S^3
        r = np.array([zi if zi > 0 else 0 for zi in z], dtype=object)
        out = self.Wo.dot(r) + self.bo * S ** 3           # S^4
        return [max(int(o), 0) for o in out]


def verify_generator(G, spec, L, exact_mass=True):
    """Exact support certificate for a count-state generator at length L."""
    t0 = time.perf_counter()
    V = G["V"]
    g = state_graph(V, G["w"], L, spec)
    ex = _ExactGen(G)
    mass = {0: Fraction(1)} if exact_mass else None
    reachable = {0}
    issues = {"unsound": None, "incomplete": None, "dead_end": None}
    back = [{0: None}]
    for t in range(L):
        nxt_reach, nxt_mass, bp = set(), {}, {}
        for i in sorted(reachable):
            sc = ex.scores(g["layers"][t][i], L - t)
            tot = sum(sc)
            if tot == 0:
                issues["dead_end"] = issues["dead_end"] or (t, i)
                continue
            for v in range(V):
                j = int(g["trans"][t][i, v])
                fz = bool(g["feasible"][t][i, v])
                if sc[v] > 0:
                    nxt_reach.add(j)
                    bp.setdefault(j, (i, v))
                    if exact_mass:
                        nxt_mass[j] = nxt_mass.get(j, 0) + mass[i] * Fraction(sc[v], tot)
                elif fz and issues["incomplete"] is None:
                    issues["incomplete"] = (t, i, v)      # a valid continuation has zero probability
        reachable, back = nxt_reach, back + [bp]
        if exact_mass:
            mass = nxt_mass
    bad = [j for j in reachable if g["final_label"][j] == 0]
    if bad:
        issues["unsound"] = bad[0]
    p_violation = sum((mass[j] for j in bad), Fraction(0)) if exact_mass else None

    def path(j):                                         # reconstruct an emitted string
        seq = []
        for t in range(L, 0, -1):
            i, v = back[t][j]
            seq.append(v)
            j = i
        return seq[::-1]
    out = {"method": "exact-gen-dp", "length": L, "states": sum(len(x) for x in g["layers"]),
           "sound": issues["unsound"] is None, "complete": issues["incomplete"] is None,
           "live": issues["dead_end"] is None,
           "p_violation": str(p_violation) if exact_mass else None,
           "time": None}
    if issues["unsound"] is not None:
        out["violating_output"] = path(issues["unsound"])
    out["status"] = "proved" if (out["sound"] and out["complete"] and out["live"]) else "failed"
    out["time"] = time.perf_counter() - t0
    return out


# --- artifacts and certificates ------------------------------------------------

GEN_KEYS = ("E", "kern", "W1", "b1", "Wo", "bo", "z0", "u", "u0")


def save_generator(G, path):
    import io
    from .artifact import sha256_file
    buf = io.BytesIO()
    np.savez(buf, meta=np.array([G["V"], G["d"], G["h"], G["w"]], dtype=np.int64),
             **{k: np.asarray(G[k], dtype=np.float64) for k in GEN_KEYS})
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return sha256_file(path)


def read_generator(path):
    """Read once; return (G, sha256 of those bytes). Same zip/shape guards as classifiers."""
    import hashlib
    import io
    import zipfile
    from .artifact import MAX_ARTIFACT_BYTES, ArtifactError
    with open(path, "rb") as f:
        buf = f.read(MAX_ARTIFACT_BYTES + 1)
    if len(buf) > MAX_ARTIFACT_BYTES:
        raise ArtifactError("generator artifact too large")
    try:
        infos = zipfile.ZipFile(io.BytesIO(buf)).infolist()
    except zipfile.BadZipFile as e:
        raise ArtifactError(f"not an npz artifact: {e}") from None
    if any(i.compress_type != zipfile.ZIP_STORED for i in infos) or \
            sum(i.file_size for i in infos) > MAX_ARTIFACT_BYTES:
        raise ArtifactError("compressed or oversized members")
    z = np.load(io.BytesIO(buf), allow_pickle=False)
    try:
        V, d, h, w = (int(v) for v in z["meta"])
        G = {"kind": "countgen", "V": V, "d": d, "h": h, "w": w}
        for k in GEN_KEYS:
            G[k] = z[k].astype(np.float64)
    except (KeyError, ValueError) as e:
        raise ArtifactError(f"bad generator artifact: {e}") from None
    want = {"E": (V, d), "kern": (w + 1,), "W1": (h, d), "b1": (h,), "Wo": (V, h), "bo": (V,),
            "z0": (V,), "u": (d,), "u0": (V,)}
    for k, shp in want.items():
        if G[k].shape != shp or not np.isfinite(G[k]).all():
            raise ArtifactError(f"{k}: bad shape {G[k].shape} or non-finite values")
    return G, hashlib.sha256(buf).hexdigest()


def load_generator_model(G):
    """A CountGen torch module with the artifact's exact weights (for sampling)."""
    import torch
    from .gen import CountGen
    m = CountGen(G["V"], G["d"], G["h"], G["w"])
    with torch.no_grad():
        for k in GEN_KEYS:
            getattr(m, k).copy_(torch.as_tensor(G[k]))
    return m.double()
