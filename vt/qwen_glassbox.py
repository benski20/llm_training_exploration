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
"""
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


@torch.no_grad()
def explain(model, ids, t, token):
    """Exact decomposition of the raw logit of `token` at position t of ids (1, T):
    {"token@j": contribution of input token j, "bias@i": of layer i's biases}. Sums to the logit."""
    rec = ScaleRecorder(model)
    try:
        out = model(ids, output_attentions=True)
    finally:
        rec.remove()
    sc = rec.scales
    T = ids.shape[1]
    cfg = model.config
    L, d = cfg.num_hidden_layers, cfg.hidden_size
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = d // H
    S = T + L
    emb = model.model.embed_tokens.weight
    comp = torch.zeros(S, T, d, dtype=torch.float32, device=ids.device)
    comp[torch.arange(T), torch.arange(T)] = emb[ids[0]].float()
    for i, layer in enumerate(layers_of(model)):
        at = layer.self_attn
        z = comp * sc[("in", i)][0] * layer.input_layernorm.weight.float()
        v = z @ at.v_proj.weight.float().T
        if at.v_proj.bias is not None:
            v[T + i] += at.v_proj.bias.float()
        v = v.view(S, T, KV, hd).repeat_interleave(H // KV, dim=2).transpose(1, 2)    # (S, H, T, hd)
        A = out.attentions[i][0].float()                                            # (H, T, T)
        o = torch.einsum("htu,shud->sthd", A, v).reshape(S, T, d)
        comp = comp + o @ at.o_proj.weight.float().T
        sp = layer.mlp
        z2 = comp * sc[("post", i)][0] * layer.post_attention_layernorm.weight.float()
        mask = sp.trace["mask"][0]
        pre = z2 @ sp.enc.weight.float().T
        pre[T + i] += sp.enc.bias.float()
        comp = comp + (pre * mask) @ sp.dec.weight.float().T
    zf = comp[:, t] * sc[("final", 0)][0, t] * model.model.norm.weight.float()
    contrib = zf @ model.lm_head.weight[token].float()
    names = [f"token@{j}" for j in range(T)] + [f"bias@{i}" for i in range(L)]
    return dict(zip(names, contrib.tolist())), float(out.logits[0, t, token])


def active_features(model, position):
    """{layer: (feature ids, activations)} at a position of the last forward (batch item 0)."""
    res = {}
    for i, sp in enumerate(sparse_layers(model)):
        a = sp.trace["acts"][0, position]
        nz = a.nonzero().squeeze(-1)
        res[i] = (nz.tolist(), a[nz].float().tolist())
    return res
