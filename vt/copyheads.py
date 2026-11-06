"""The reader's copying (induction) heads, read inside the grounding check's own forward pass.

A copying head at the position before reply token t attends to the evidence token that followed an earlier
occurrence of the current token: that evidence token is its candidate for t. So, per reply token:

    match     attention on evidence positions holding exactly token t        (copied from the evidence)
    conflict  attention on evidence positions holding a different token      (the evidence says something else
                                                                              here: a swapped number or name)

The heads are found once per reader, with no labels: the standard induction score on a random token sequence
repeated twice (find_copy_heads). Their attention is computed from the layer's own projections inside a hook, only
for the chosen heads and only for the reply's positions, so the model keeps its fast attention kernel
(docs/copy_heads.md; prompted by "Sparks of Weight-Space Programming", MIT CSAIL, 2026).
"""
import sys

import numpy as np


class HeadAttention:
    """Attention rows of chosen heads, at chosen query positions, recomputed from each layer's q/k projections in a
    forward pre-hook: the forward pass itself is unchanged. Works for Llama-style attention (q_proj, k_proj,
    optional q_norm/k_norm, rotary embeddings passed to the attention module)."""

    def __init__(self, model, heads):
        self.model, self.by_layer = model, {}
        for l, h in heads:
            self.by_layer.setdefault(int(l), []).append(int(h))
        self.rows, self.hooks, self.queries = {}, [], None

    def __enter__(self):
        for l, hs in self.by_layer.items():
            attn = self.model.model.layers[l].self_attn
            self.hooks.append(attn.register_forward_pre_hook(
                lambda mod, args, kwargs, l=l, hs=hs: self._capture(mod, args, kwargs, l, hs), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _capture(self, attn, args, kwargs, l, hs):
        import torch
        x = kwargs.get("hidden_states", args[0] if args else None)[0]          # T x hidden (batch of one)
        cos, sin = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
        cfg = attn.config
        d = getattr(attn, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        kv = sorted({h // groups for h in hs})
        with torch.no_grad():
            def project(lin, idx, norm):
                w = lin.weight.view(-1, d, lin.weight.shape[-1])[idx]          # heads x d x hidden
                y = torch.einsum("th,ndh->ntd", x, w)
                if lin.bias is not None:
                    y = y + lin.bias.view(-1, d)[idx][:, None]
                return norm(y) if norm is not None else y
            q = project(attn.q_proj, hs, getattr(attn, "q_norm", None))[None]   # 1 x H x T x d
            k = project(attn.k_proj, kv, getattr(attn, "k_norm", None))[None]
            rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb
            q, _ = rope(q, q, cos, sin)
            k, _ = rope(k, k, cos, sin)
            k = k[0, [kv.index(h // groups) for h in hs]]                        # H x T x d, each head's own k
            qpos = torch.as_tensor(self.queries)
            s = torch.einsum("hqd,htd->hqt", q[0][:, qpos].float(), k.float()) * attn.scaling
            T = k.shape[1]
            s = s.masked_fill(torch.arange(T)[None, None, :] > qpos[None, :, None], float("-inf"))
            self.rows[l] = torch.softmax(s, -1)                                 # H x Q x T

    def at(self, queries):
        """Capture these query positions in the next forward pass (inside the `with` block)."""
        self.queries, self.rows = list(queries), {}
        return self

    def collect(self):
        """heads x queries x positions (float32, CPU), in the order the heads were given by layer."""
        import torch
        return torch.cat([self.rows[l] for l in self.by_layer]).cpu()

    def run(self, ids, queries):
        """One forward pass over ids (a list of token ids); returns heads x queries x positions, float32."""
        import torch
        self.at(queries)
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            self.model(input_ids=torch.tensor([ids], device=dev), logits_to_keep=1)
        return self.collect()


def find_copy_heads(model, top=16, n=64, batch=2, seed=0):
    """The reader's strongest copying heads, with no labels: mean attention from each token of the second copy of
    a random sequence to the token after its first occurrence. Returns [(layer, head, score)], strongest first."""
    import torch
    cfg = model.config
    heads = [(l, h) for l in range(cfg.num_hidden_layers) for h in range(cfg.num_attention_heads)]
    g = torch.Generator().manual_seed(seed)
    scores = np.zeros(len(heads))
    for _ in range(batch):
        r = torch.randint(1000, min(cfg.vocab_size, 100000), (n,), generator=g).tolist()
        q = list(range(n + 1, 2 * n))
        with HeadAttention(model, heads) as ha:
            att = ha.run(r + r, q)                                               # all heads x Q x T
        idx = torch.arange(len(q))
        scores += att[:, idx, torch.as_tensor(q) - n + 1].mean(1).numpy() / batch
    order = np.argsort(-scores)[:top]
    return [(heads[i][0], heads[i][1], round(float(scores[i]), 3)) for i in order]


def token_copy(att, prefix_ids, reply_ids, start=1):
    """Per reply token: copying-head attention on evidence (prefix positions from `start`) holding that token
    (match) and holding another token (conflict), averaged over the heads. att: heads x reply x positions, the
    rows taken at the positions that predict each reply token."""
    a = att.mean(0).numpy()[:, start:len(prefix_ids)]
    same = np.array(prefix_ids[start:])[None, :] == np.array(reply_ids)[:, None]
    total, match = a.sum(1), (a * same).sum(1)
    return match, total - match
