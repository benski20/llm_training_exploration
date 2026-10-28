"""Convert a pretrained Qwen2 model into a glass-box model.

Each dense MLP is replaced by an exactly sparse feature layer: top-k of ReLU features (the rest
exactly 0), a linear decoder, and a train-only readout that must reconstruct the layer's input
from its active features (the Glassbox-2.4 objective). With every discrete decision of a forward
pass fixed (RMS-norm scales, attention patterns, active feature sets) the whole network is linear,
so `explain` decomposes any output logit exactly into contributions of each input token and of
each layer's biases. Attention is untouched: its pattern is read off the forward pass.

    m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
    convert(m, n_features=8192, k=64)          # sparse layers start untrained: distill them first
    parts = explain(m, ids, t, token)          # sum(parts.values()) == the raw logit

Exact attribution also works without converting: a dense SwiGLU MLP is linear in its input once its
SiLU gate values are fixed, so `explain` runs on the unmodified model too.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMLP(nn.Module):
    """Drop-in replacement for a transformer MLP: exactly k features active per token."""

    def __init__(self, d, n_features, k, original=None):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d, n_features)
        self.dec = nn.Linear(n_features, d, bias=False)
        self.rec = nn.Linear(n_features, d, bias=False)          # train-only; not in the forward pass
        nn.init.normal_(self.dec.weight, 0, 0.01)
        self.original = original                               # kept for distillation / teacher passes
        self.use_original = False
        self.trace = None

    def forward(self, x):
        if self.use_original:
            return self.original(x)
        pre = self.enc(x)
        with torch.no_grad():                                  # the active set is a discrete decision
            top = pre.topk(self.k, -1).indices
            mask = torch.zeros_like(pre, dtype=torch.bool).scatter_(-1, top, True) & (pre > 0)
        f = pre * mask
        self.trace = {"x": x, "mask": mask, "acts": f}
        return self.dec(f)

    def recon_loss(self):
        x = self.trace["x"].detach().float()
        return F.mse_loss(self.rec(self.trace["acts"]).float(), x) / (x.var() + 1e-6)


def layers_of(model):
    return model.model.layers


def convert(model, n_features=8192, k=64):
    """Replace every MLP by a SparseMLP (keeping the original inside for distillation)."""
    d = model.config.hidden_size
    for layer in layers_of(model):
        orig = layer.mlp
        sp = SparseMLP(d, n_features, k, original=orig)
        sp.to(next(orig.parameters()).device)
        layer.mlp = sp
    return model


def sparse_layers(model):
    return [layer.mlp for layer in layers_of(model)]


def set_original(model, flag):
    for sp in sparse_layers(model):
        sp.use_original = flag


class KHead(nn.Module):
    """Self-knowledge: predicts, from the final hidden state, whether the top prediction is right."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, h):
        return self.lin(h).squeeze(-1)


class ScaleRecorder:
    """Records the RMS-norm scale of every norm's input during a forward pass (a decision)."""

    def __init__(self, model):
        self.scales, self.handles = {}, []
        for i, layer in enumerate(layers_of(model)):
            for name, mod in (("in", layer.input_layernorm), ("post", layer.post_attention_layernorm)):
                self.handles.append(mod.register_forward_hook(self._hook((name, i), mod)))
        self.handles.append(model.model.norm.register_forward_hook(self._hook(("final", 0), model.model.norm)))

    def _hook(self, key, mod):
        def f(m, inp, out):
            x = inp[0].float()
            self.scales[key] = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + m.variance_epsilon)
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


class InputRecorder:
    """Records the input of every dense MLP during a forward pass: its SiLU gate values are the
    MLP's decisions (with them fixed, a SwiGLU MLP is linear in its input)."""

    def __init__(self, model):
        self.x, self.handles = {}, []
        for i, layer in enumerate(layers_of(model)):
            if not isinstance(layer.mlp, SparseMLP):
                self.handles.append(layer.mlp.register_forward_hook(self._hook(i)))

    def _hook(self, i):
        def f(m, inp, out):
            self.x[i] = inp[0]
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


class PatternRecorder:
    """Recomputes every layer's attention pattern in float32 from the inputs its attention module
    receives (a decision, like the RMS scales). Works with any attention implementation: float16
    models must use sdpa, since eager attention overflows on Qwen2.5's raw q.k (up to ~2.6e5 > 65504),
    and sdpa does not return its pattern. Single unpadded sequences (B = 1), causal attention.

    Only each layer's input is kept (T x d); `A[i]` computes layer i's (H, T, T) pattern on demand, so
    memory holds one layer's pattern at a time (all of them at once is ~13 GB for a 7B model at 2k tokens)."""

    def __init__(self, model):
        self.inputs, self.handles = {}, []
        cfg = model.config
        self.H, self.KV = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.mods = {}
        for i, layer in enumerate(layers_of(model)):
            self.mods[i] = layer.self_attn
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._hook(i), with_kwargs=True))
        self.A = _LazyPatterns(self)

    def _hook(self, i):
        def f(mod, args, kwargs):
            x = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
            cos, sin = kwargs["position_embeddings"]
            self.inputs[i] = (x.detach(), cos.detach(), sin.detach())
        return f

    def pattern(self, i):
        mod = self.mods[i]
        x, cos, sin = self.inputs[i]
        x = x.float()
        rope = sys.modules[type(mod).__module__].apply_rotary_pos_emb
        B, T, _ = x.shape
        q = F.linear(x, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
        k = F.linear(x, mod.k_proj.weight.float(), None if mod.k_proj.bias is None else mod.k_proj.bias.float())
        q = q.view(B, T, self.H, self.hd).transpose(1, 2)
        k = k.view(B, T, self.KV, self.hd).transpose(1, 2)
        q, k = rope(q, k, cos.float(), sin.float())
        k = k.repeat_interleave(self.H // self.KV, dim=1)
        sc = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        sc = sc.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        return sc.softmax(-1)[0]                                                   # (H, T, T)

    def remove(self):
        for h in self.handles:
            h.remove()


class _LazyPatterns:
    def __init__(self, rec):
        self.rec = rec

    def __getitem__(self, i):
        return self.rec.pattern(i)

    def __iter__(self):
        return iter(sorted(self.rec.inputs))

    def __len__(self):
        return len(self.rec.inputs)


def _decompose(model, ids, groups=None, merge_bias=False):
    """One forward pass, then the exact linear decomposition of the residual stream at every position
    (the model's decisions fixed): comp (S, T, d) with one component per token group plus one per
    layer's biases (or a single merged bias component). Returns comp, names, final RMS scales, logits."""
    rec, inp, pat = ScaleRecorder(model), InputRecorder(model), PatternRecorder(model)
    try:
        out = model(ids)
    finally:
        rec.remove()
        inp.remove()
        pat.remove()
    sc = rec.scales
    T = ids.shape[1]
    cfg = model.config
    L, d = cfg.num_hidden_layers, cfg.hidden_size
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = pat.hd
    names = list(dict.fromkeys(groups)) if groups else [f"token@{j}" for j in range(T)]
    src = [names.index(g) for g in groups] if groups else list(range(T))
    G = len(names)
    S = G + (1 if merge_bias else L)
    emb = model.model.embed_tokens.weight
    comp = torch.zeros(S, T, d, dtype=torch.float32, device=ids.device)
    comp[torch.tensor(src), torch.arange(T)] = emb[ids[0]].float()
    for i, layer in enumerate(layers_of(model)):
        b = G if merge_bias else G + i
        at = layer.self_attn
        comp = comp.to(at.v_proj.weight.device)                 # models split across GPUs (device_map)
        z = comp * sc[("in", i)][0] * layer.input_layernorm.weight.float()
        v = z @ at.v_proj.weight.float().T
        if at.v_proj.bias is not None:
            v[b] += at.v_proj.bias.float()
        v = v.view(S, T, KV, hd).repeat_interleave(H // KV, dim=2).transpose(1, 2)    # (S, H, T, hd)
        A = pat.A[i]                                                                # (H, T, T)
        o = torch.einsum("htu,shud->sthd", A, v).reshape(S, T, d)
        comp = comp + o @ at.o_proj.weight.float().T
        sp = layer.mlp
        z2 = comp * sc[("post", i)][0] * layer.post_attention_layernorm.weight.float()
        if isinstance(sp, SparseMLP):
            mask = sp.trace["mask"][0]
            pre = z2 @ sp.enc.weight.float().T
            pre[b] += sp.enc.bias.float()
            comp = comp + (pre * mask) @ sp.dec.weight.float().T
        else:                                                   # SwiGLU with its gate values fixed
            gate = sp.act_fn(F.linear(inp.x[i][0].float(), sp.gate_proj.weight.float()))   # (T, d_ff)
            comp = comp + ((z2 @ sp.up_proj.weight.float().T) * gate) @ sp.down_proj.weight.float().T
    names = names + (["bias"] if merge_bias else [f"bias@{i}" for i in range(L)])
    dev = model.lm_head.weight.device
    return comp.to(dev), names, sc[("final", 0)][0].to(dev), out.logits[0].to(dev)


@torch.no_grad()
def explain(model, ids, t, token, groups=None):
    """Exact decomposition of the raw logit of `token` at position t of ids (1, T):
    {"token@j": contribution of input token j, "bias@i": of layer i's biases}. Sums to the logit.
    Works for converted layers (fixed active sets) and dense SwiGLU MLPs (fixed gate values) alike.
    `groups` (a name per input position) merges tokens into one component per name."""
    comp, names, fs, logits = _decompose(model, ids, groups)
    zf = comp[:, t] * fs[t] * model.model.norm.weight.float().to(comp.device)   # tied lm_head can sit off the norm's GPU
    contrib = zf @ model.lm_head.weight[token].float()
    return dict(zip(names, contrib.tolist())), float(logits[t, token])


@torch.no_grad()
def explain_all(model, ids, groups, start=0):
    """Provenance of every written token in one pass: for each position t >= start, the exact split of
    the logit of the token actually at t+1 into each group's contribution plus one merged bias term.
    Returns (names, contrib (S, n), logits (n,)) with n = T - 1 - start; columns sum to the logits."""
    comp, names, fs, logits = _decompose(model, ids, groups, merge_bias=True)
    T = ids.shape[1]
    pos = torch.arange(start, T - 1, device=comp.device)
    tgt = ids[0, start + 1:].to(comp.device)
    zf = comp[:, pos] * fs[pos] * model.model.norm.weight.float().to(comp.device)   # (S, n, d)
    contrib = torch.einsum("snd,nd->sn", zf, model.lm_head.weight[tgt].float())
    return names, contrib, logits[pos, tgt].float()


@torch.no_grad()
def layer_contributions(model, ids, t, token):
    """Exact decomposition of the same logit by what wrote it into the residual stream:
    {"embed", "attn@i", "mlp@i"} (the final RMS scale fixed). Sums to the logit."""
    writes, hs = {}, []
    for i, layer in enumerate(layers_of(model)):
        hs.append(layer.self_attn.register_forward_hook(
            lambda m, a, o, i=i: writes.__setitem__(f"attn@{i}", o[0][0, t].float())))
        hs.append(layer.mlp.register_forward_hook(
            lambda m, a, o, i=i: writes.__setitem__(f"mlp@{i}", o[0, t].float())))
    rec = ScaleRecorder(model)
    try:
        out = model(ids)
    finally:
        rec.remove()
        for h in hs:
            h.remove()
    w = model.lm_head.weight[token].float()
    w = rec.scales[("final", 0)][0, t].to(w.device) * model.model.norm.weight.float().to(w.device) * w
    parts = {"embed": float(model.model.embed_tokens.weight[ids[0, t]].float().to(w.device) @ w)}
    for i in range(model.config.num_hidden_layers):
        parts[f"attn@{i}"] = float(writes[f"attn@{i}"].to(w.device) @ w)
        parts[f"mlp@{i}"] = float(writes[f"mlp@{i}"].to(w.device) @ w)
    return parts, float(out.logits[0, t, token])


def active_features(model, position):
    """{layer: (feature ids, activations)} at a position of the last forward (batch item 0)."""
    res = {}
    for i, sp in enumerate(sparse_layers(model)):
        a = sp.trace["acts"][0, position]
        nz = a.nonzero().squeeze(-1)
        res[i] = (nz.tolist(), a[nz].float().tolist())
    return res


@torch.no_grad()
def lookback_ratio(model, ids, groups, start):
    """Lookback Lens's context signal (Chuang et al. 2024), as a baseline for exact provenance: for
    each position t >= start, the attention mass on document tokens over the mass on document plus
    already-written response tokens, averaged over all layers and heads."""
    pat = PatternRecorder(model)
    try:
        model(ids)
    finally:
        pat.remove()
    g = groups
    doc = torch.tensor([x == "document" for x in g])
    gen = torch.tensor([x == "response" for x in g])
    out = []
    for i in sorted(pat.A):
        A = pat.A[i].float()                                                  # (H, T, T)
        d, r = doc.to(A.device), gen.to(A.device)
        md = (A[:, start:-1] * d).sum(-1)
        mr = (A[:, start:-1] * r).sum(-1)
        out.append((md / (md + mr + 1e-9)).mean(0).cpu())
    return torch.stack(out).mean(0)                                           # (n,)
