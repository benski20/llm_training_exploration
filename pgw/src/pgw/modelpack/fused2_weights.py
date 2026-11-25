"""Public fused2 weight tables: pack-time derivation shared by the provider service and the gateway engine.

The fused2 reference (``pgw.engine.fused2``; design in ``research/plans/fused2-two-trips-per-layer.md``) is a pinned
model built from public weights only: ``o`` and ``down`` are the pack's int8 linears (with optional residual-grid
multipliers ``hmul``), ``gate_up`` and ``qkv`` are int8 folds of the following RMSNorm gain into the weight. This
module holds that derivation (:class:`Fused2Weights`, :func:`migration_multipliers`, :class:`F2Linear`) with no
engine imports, so the compute service can derive its lane tables (``pgw.service.fused2_lane``) without importing
the gateway engine, exactly as the masked LM head's derivation lives in :mod:`pgw.modelpack.masked_head`.
``pgw.engine.fused2`` re-exports every name; the code moved unchanged, so the tables are bit-identical.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from pgw.core.errors import ConfigError
from pgw.core.ring import quantize_weight_rows
from pgw.core.types import LinearId, LinearKind

__all__ = ["F2Linear", "Fused2Weights", "hf_float_linears", "migration_multipliers", "select_gateway_columns"]

#: Relative margin on the float64 spectral norm (SVD backward error is ~n * 2^-52 relative).
_SIGMA_MARGIN = 1.0 + 1e-6

# ---- public weights ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class F2Linear:
    """One public int8 linear of the fused2 reference.

    Attributes:
        key: Name, e.g. ``"3.o"``, ``"3.gate_up/f2"`` (folded), ``"4.qkv/f2"``.
        w8: int8 ``(m, n)``.
        scale: float64 ``(m,)`` row scales (the float32 scales widened).
        bias: float64 ``(m,)`` or None.
        wnorm: ``max_i ||w8_i||_2`` (``max_i hmul_i ||w8_i||_2`` with ``hmul``).
        sigma: Upper bound on the spectral norm of ``w8`` (of ``diag(hmul) w8`` with ``hmul``; 0 when
            not needed).
        hmul: Residual-grid multipliers of a trip's first linear (o, down; int64 ``(m,)``) or None. The
            provider returns ``H = R + hmul * (w8 x)``, so the residual lives on the finer grid
            ``scale / hmul`` (:attr:`grid`): a per-channel migration of the next RMSNorm gain out of the
            folded weight into the residual's fixed-point grid (see :func:`calibrate_fused2`).
        sigma_in: On a folded linear with gateway columns: upper bound on the spectral norm of the
            previous first linear's ``diag(hmul) w8`` restricted to the provider's (non-gateway) channels,
            for the tighter bound ``|Z| <= wnorm (||R_keep|| + sigma_in ||aq||)`` (0: not set).
        gcols: Gateway columns of a folded linear (int64 ``(g,)``, sorted) or None. ``w8`` is zero on
            them (the provider's product skips them); the gateway adds ``gw @ H[:, gcols]`` itself from
            the verified ``H`` it gets back in the same trip (see :func:`gateway_product`).
        gw: int64 ``(m, g)`` integer weights of the gateway columns (``gbits``-bit, per-row scales).
        gscale: float64 ``(m,)`` row scales of ``gw``.
        gbits: Bits of ``gw`` (0 without gateway columns).
    """

    key: str
    w8: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None
    wnorm: float
    sigma: float = 0.0
    gcols: torch.Tensor | None = None
    gw: torch.Tensor | None = None
    gscale: torch.Tensor | None = None
    gbits: int = 0
    hmul: torch.Tensor | None = None
    sigma_in: float = 0.0

    @property
    def grid(self) -> torch.Tensor:
        """float64 ``(m,)`` residual grid of a first linear's output channels: ``scale / hmul``."""
        return self.scale if self.hmul is None else self.scale / self.hmul.to(torch.float64)

    @property
    def keep(self) -> torch.Tensor | None:
        """bool ``(n,)`` mask of the provider's input columns of a split fold (None: all)."""
        if self.gcols is None:
            return None
        k = torch.ones(self.n, dtype=torch.bool)
        k[self.gcols] = False
        return k

    @property
    def w8t(self) -> torch.Tensor:
        """``(n, m)`` view of ``w8`` (the exact digit product reads ``w8`` itself)."""
        return self.w8.t()

    @property
    def m(self) -> int:
        return int(self.w8.shape[0])

    @property
    def n(self) -> int:
        return int(self.w8.shape[1])


def _make_linear(key: str, w8: torch.Tensor, sw: torch.Tensor, bias: torch.Tensor | None,
                 sigma: bool = False, hmul: torch.Tensor | None = None) -> F2Linear:
    w8 = w8.to(torch.int8).contiguous()
    w64 = w8.to(torch.float64)
    if hmul is not None:
        hmul = torch.as_tensor(hmul, dtype=torch.int64).contiguous()
        if hmul.shape != (w8.shape[0],) or int(hmul.min()) < 1 or int(hmul.max()) >= 1 << 20:
            raise ConfigError(f"fused2 {key}: multipliers must be integers in [1, 2^20), one per row")
        if int(hmul.max()) == 1:
            hmul = None
        else:
            w64 = w64 * hmul.to(torch.float64)[:, None]
    wnorm = float(w64.pow(2).sum(dim=1).sqrt().max())
    sig = float(torch.linalg.matrix_norm(w64, ord=2)) * _SIGMA_MARGIN if sigma else 0.0
    return F2Linear(key, w8, sw.to(torch.float32).to(torch.float64), None if bias is None
                    else bias.to(torch.float32).to(torch.float64), wnorm, sig, hmul=hmul)


def _sigma_rows(lin: F2Linear, keep: torch.Tensor) -> float:
    """Spectral norm bound of ``diag(hmul) w8`` restricted to the rows in ``keep``."""
    w64 = lin.w8.to(torch.float64)
    if lin.hmul is not None:
        w64 = w64 * lin.hmul.to(torch.float64)[:, None]
    return float(torch.linalg.matrix_norm(w64[keep], ord=2)) * _SIGMA_MARGIN


def _quantize_rows_bits(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax quantization of float64 ``w`` to signed ``bits``-bit integers (int64, float32 scales
    widened to float64), the same rule as :func:`quantize_weight_rows` with ``qmax = 2^(bits-1) - 1``."""
    qmax = (1 << (bits - 1)) - 1
    amax = w.abs().amax(dim=1)
    sw = torch.where(amax > 0, amax / qmax, torch.ones_like(amax)).to(torch.float32).to(torch.float64)
    q = torch.round(w / sw[:, None]).clamp_(-qmax, qmax).to(torch.int64)
    return q, sw


def _fold(key: str, w: torch.Tensor, d: torch.Tensor, bias: torch.Tensor | None,
          gcols: torch.Tensor | None = None, gbits: int = 16) -> F2Linear:
    """int8 per-row quantization of ``w diag(d)`` (float64 product, then :func:`quantize_weight_rows`).

    With ``gcols`` those columns are cut out of the int8 matrix before its row scales are taken (they are
    zero in ``w8``) and quantized on their own, per row, to ``gbits``-bit integers for the gateway.
    """
    wf = w.to(torch.float64) * d.to(torch.float64)[None, :]
    if gcols is None or not len(gcols):
        w8, sw = quantize_weight_rows(wf)
        return _make_linear(key, w8, sw, bias)
    gcols = torch.as_tensor(gcols, dtype=torch.int64).unique(sorted=True)
    main = wf.clone()
    main[:, gcols] = 0.0
    w8, sw = quantize_weight_rows(main)
    gw, gs = _quantize_rows_bits(wf[:, gcols], gbits)
    lin = _make_linear(key, w8, sw, bias)
    return F2Linear(lin.key, lin.w8, lin.scale, lin.bias, lin.wnorm, lin.sigma, gcols, gw.contiguous(), gs, gbits)


def select_gateway_columns(w: torch.Tensor, g: torch.Tensor, count: int) -> torch.Tensor:
    """Pack-time choice of a folded linear's gateway columns, from public weights only.

    The ``count`` columns with the largest absmax of ``w diag(g)`` (``g`` the RMSNorm gain): the columns
    that set the int8 row scales of the folded matrix (Qwen's gain outliers) and so coarsen all others.
    """
    if count <= 0:
        return torch.zeros(0, dtype=torch.int64)
    score = (w.to(torch.float64).abs() * g.to(torch.float64).abs()[None, :]).amax(dim=0)
    return score.topk(min(count, int(score.numel()))).indices.sort().values


class _LazyFloat(Mapping[str, torch.Tensor]):
    def __init__(self, keys: list[str], load: Any):
        self._keys, self._load = keys, load

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._keys:
            raise KeyError(key)
        return self._load(key)

    def __iter__(self) -> Any:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def hf_float_linears(state_dict: Mapping[str, torch.Tensor] | Any, arch: Any) -> Mapping[str, torch.Tensor]:
    """Float weights of the folded linears (qkv and gate_up of every layer) from an HF state dict.

    ``state_dict`` is a mapping or a safetensors file path; either way each linear is materialized only
    when :class:`Fused2Weights` asks for it (low memory)."""
    from pgw.modelpack.arch import hf_linear_parts

    if not isinstance(state_dict, Mapping):
        from pathlib import Path

        from safetensors import safe_open

        root = Path(state_dict)
        files = sorted(root.glob("*.safetensors")) if root.is_dir() else [root]  # a file, or a sharded dir
        where: dict[str, str] = {}
        for fp in files:
            with safe_open(str(fp), framework="pt") as f:
                where.update({k: str(fp) for k in f.keys()})

        def get(name: str) -> torch.Tensor:
            with safe_open(where[name], framework="pt") as f:
                return f.get_tensor(name)
    else:
        def get(name: str) -> torch.Tensor:
            return state_dict[name]

    kinds = (LinearKind.QKV, LinearKind.GATE_UP)
    keys = [f"{layer}.{kind.value}" for layer in range(arch.num_layers) for kind in kinds]

    def load(key: str) -> torch.Tensor:
        layer, kind = key.split(".")
        parts = hf_linear_parts(arch.arch, int(layer), LinearKind(kind))
        return torch.cat([get(wn).detach().float() for _, wn, _ in parts], dim=0)

    return _LazyFloat(keys, load)


class Fused2Weights:
    """Public weights of the fused2 reference, derived from a pack (pack time; no secret).

    ``o`` and ``down`` are the pack's int8 linears unchanged. ``gu[l]`` folds ``post_attn_norm[l] *
    sw_o[l]`` into gate_up; ``qkv[l]`` folds ``input_norm[l] * sw_down[l-1]`` (``input_norm[0]`` for layer
    0) into qkv. ``float_linears`` (from :func:`hf_float_linears`) folds the original float weights;
    without it the pack's dequantized int8 weights are folded (a second rounding; measured slightly worse).

    ``gateway_columns`` (default 0: none, the original reference) splits the fold: per folded linear,
    that many columns (chosen by :func:`select_gateway_columns`, or an explicit ``{key: columns}``
    mapping with keys like ``"3.gate_up/f2"``) are taken out of the provider's int8 matrix and multiplied
    by the gateway itself, as ``gateway_bits``-bit integers, with the verified ``H`` of the same trip.
    Trips, wire bytes and masks are unchanged; the gateway does ``m * g`` extra integer MACs per row.

    ``multipliers`` (``{"3.o": hmul, "3.down": hmul}``, int ``>= 1``; default none) puts the residual of
    the trips started by that first linear on the finer grid ``sw / hmul`` (see :attr:`F2Linear.hmul`);
    the following fold then uses ``g * sw / hmul``. :func:`calibrate_fused2` chooses both maps. The last
    layer's down (a single trip) takes no multipliers.

    ``sigma=False`` skips the spectral-norm bounds ``sigma`` of the trips' first linears (one dense SVD per ``o`` and
    ``down``: about 27 s each at Qwen2.5-32B widths on a 4-vCPU box, the whole derivation's cost) and leaves them 0.
    Only the gateway's quantization (and the plaintext reference) reads ``sigma``; the provider's pair op, the
    lane fingerprint (``w8`` and ``hmul``) and the mask factory do not, so they derive their tables without it.
    """

    def __init__(self, pack: Any, float_linears: Mapping[str, torch.Tensor] | None = None, *,
                 gateway_columns: int | Mapping[str, Sequence[int]] = 0, gateway_bits: int = 16,
                 multipliers: Mapping[str, Any] | None = None, sigma: bool = True):
        a = pack.arch
        self.num_layers = int(a.num_layers)
        self.fingerprint = str(getattr(pack, "fingerprint", ""))
        self.folded_from = "float" if float_linears is not None else "int8"
        if not 2 <= int(gateway_bits) <= 32:
            raise ConfigError("gateway_bits must be in [2, 32]")
        self.gateway_bits = int(gateway_bits)
        self.gateway_columns = gateway_columns if isinstance(gateway_columns, int) else dict(gateway_columns)
        mults = dict(multipliers or {})
        if f"{self.num_layers - 1}.down" in mults:
            raise ConfigError("fused2: the last layer's down (a single trip) takes no multipliers")
        self.migrated = bool(mults)
        self.o: list[F2Linear] = []
        self.down: list[F2Linear] = []
        self.gu: list[F2Linear] = []
        self.qkv: list[F2Linear] = []
        g_in = [pack.load_plain(f"layers.{i}.input_norm", dtype=torch.float32) for i in range(self.num_layers)]
        g_post = [pack.load_plain(f"layers.{i}.post_attn_norm", dtype=torch.float32) for i in range(self.num_layers)]

        def lin(layer: int, kind: LinearKind) -> Any:
            return pack.load_linear(LinearId(layer, kind), device="cpu")

        def float_w(layer: int, kind: LinearKind) -> torch.Tensor:
            if float_linears is not None:
                return float_linears[f"{layer}.{kind.value}"]
            lw = lin(layer, kind)
            return lw.w8.to(torch.float64) * lw.scale.to(torch.float64)[:, None]

        for i in range(self.num_layers):
            lo, ld = lin(i, LinearKind.O), lin(i, LinearKind.DOWN)
            self.o.append(_make_linear(f"{i}.o", lo.w8, lo.scale, lo.bias, sigma=sigma, hmul=mults.get(f"{i}.o")))
            self.down.append(_make_linear(f"{i}.down", ld.w8, ld.scale, ld.bias, sigma=sigma,
                                          hmul=mults.get(f"{i}.down")))
        def cols(key: str, wf: torch.Tensor, g: torch.Tensor) -> torch.Tensor | None:
            gc = self.gateway_columns
            if isinstance(gc, int):
                return select_gateway_columns(wf, g, gc) if gc > 0 else None
            sel = gc.get(key)
            return None if sel is None else torch.as_tensor(list(sel), dtype=torch.int64)

        for i in range(self.num_layers):
            lq, lg = lin(i, LinearKind.QKV), lin(i, LinearKind.GATE_UP)
            c_in = g_in[i].to(torch.float64) * (self.down[i - 1].grid if i else 1.0)
            wq = float_w(i, LinearKind.QKV)
            key = f"{i}.qkv/f2"
            self.qkv.append(_fold(key, wq, c_in, lq.bias, cols(key, wq, g_in[i]), self.gateway_bits))
            del wq
            c_post = g_post[i].to(torch.float64) * self.o[i].grid
            wg = float_w(i, LinearKind.GATE_UP)
            key = f"{i}.gate_up/f2"
            self.gu.append(_fold(key, wg, c_post, lg.bias, cols(key, wg, g_post[i]), self.gateway_bits))
            del wg
        # tighter Z bounds for split folds: only the provider's channels of H enter Z
        for i in range(self.num_layers):
            pairs = [(self.o[i], self.gu, i)] + ([(self.down[i], self.qkv, i + 1)] if i + 1 < self.num_layers else [])
            for l1, folds, j in pairs:
                keep = folds[j].keep
                if keep is not None:
                    folds[j] = dataclasses.replace(folds[j], sigma_in=_sigma_rows(l1, keep))

    _META = ("num_layers", "fingerprint", "folded_from", "gateway_bits", "migrated")

    def save(self, path: str | os.PathLike[str], *, provider_only: bool = False) -> None:
        """Write the public weights (tensors and numbers only; ``torch.load(weights_only=True)`` reads it).

        ``provider_only``: just what the provider's pair op needs (``w8`` and ``hmul`` per key).
        """
        def one(lin: F2Linear) -> dict[str, Any]:
            if provider_only:
                return {"key": lin.key, "w8": lin.w8, "hmul": lin.hmul}
            return {f.name: getattr(lin, f.name) for f in dataclasses.fields(F2Linear)}

        blob: dict[str, Any] = {name: getattr(self, name) for name in self._META}
        blob["provider_only"] = provider_only
        for part in ("qkv", "o", "gu", "down"):
            blob[part] = [one(lin) for lin in getattr(self, part)]
        torch.save(blob, os.fspath(path))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Fused2Weights:
        """Read :meth:`save`'s file. A provider-only file has ``w8`` / ``hmul`` and zero scales and norms
        (enough for :class:`Fused2Provider`, not for a gateway)."""
        blob = torch.load(os.fspath(path), weights_only=True)
        self = cls.__new__(cls)
        for name in cls._META:
            setattr(self, name, blob[name])
        self.gateway_columns = 0
        for part in ("qkv", "o", "gu", "down"):
            lins = []
            for d in blob[part]:
                if blob.get("provider_only"):
                    m = int(d["w8"].shape[0])
                    d = {"key": d["key"], "w8": d["w8"], "hmul": d["hmul"],
                         "scale": torch.zeros(m, dtype=torch.float64), "bias": None, "wnorm": 0.0}
                lins.append(F2Linear(**d))
            setattr(self, part, lins)
        return self

    def linears(self) -> list[F2Linear]:
        out: list[F2Linear] = []
        for i in range(self.num_layers):
            out += [self.qkv[i], self.o[i], self.gu[i], self.down[i]]
        return out

    def gateway_cost(self) -> dict[str, int]:
        """Gateway side of the split fold: columns, public weight bytes it holds, integer MACs per token."""
        lins = [lin for lin in self.qkv + self.gu if lin.gw is not None]
        return {
            "columns": sum(int(lin.gw.shape[1]) for lin in lins),  # type: ignore[union-attr]
            "weight_bytes": sum(int(lin.gw.numel()) * ((lin.gbits + 7) // 8) for lin in lins),  # type: ignore[union-attr]
            "macs_per_token": sum(int(lin.gw.numel()) for lin in lins),  # type: ignore[union-attr]
        }


def migration_multipliers(pack: Any, bits: int = 8, *, sw: Mapping[str, torch.Tensor] | None = None
                          ) -> dict[str, torch.Tensor]:
    """Pure pack-time migration (no calibration data, no gateway columns): public integer multipliers.

    Per trip's first linear ``l1`` (every ``o``, every ``down`` but the last) and the gain ``g`` of the norm
    folded after it, with ``c_j = |g_j| * sw1_j`` and ``kappa = max_j c_j / 2^bits``:
    ``hmul_j = max(1, floor(c_j / kappa))``. The folded matrix then sees column scales ``c_j / hmul_j`` within
    a factor 2 of each other above ``kappa`` (its per-row int8 grid is as fine as today's int8 weights for
    those columns), and the residual of the trip lives on the ``bits``-finer grid ``sw1 / hmul``. Costs about
    ``bits`` more bits in ``H`` and ``Z`` (Qwen2.5-0.5B, 8 bits: ``|H|`` 41.5, ``|Z|`` 48.6 bits; ring K56C53).
    """
    nl = int(pack.arch.num_layers)
    out: dict[str, torch.Tensor] = {}
    for i in range(nl):
        pairs = [(LinearId(i, LinearKind.O), f"layers.{i}.post_attn_norm")]
        if i + 1 < nl:
            pairs.append((LinearId(i, LinearKind.DOWN), f"layers.{i + 1}.input_norm"))
        for lid, gname in pairs:
            g = pack.load_plain(gname, dtype=torch.float32).to(torch.float64).abs()
            key = f"{i}.{'o' if lid.kind == LinearKind.O else 'down'}"
            s1 = (sw[key] if sw is not None else pack.load_linear(lid, device="cpu").scale).to(torch.float32)
            c = g * s1.to(torch.float64)
            kappa = float(c.max()) / 2.0 ** bits
            out[key] = (torch.floor(c / kappa).clamp(min=1.0) if kappa > 0 else torch.ones_like(c)).to(torch.int64)
    return out
