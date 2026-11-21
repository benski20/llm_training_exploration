"""Fused exact ring GEMV ``y = x @ W8^T mod 2^K`` for few rows (``K <= 32``) as one Triton kernel.

The digit-GEMM path (:func:`pgw.core.ringmm.offset_byte_core`) splits ``x`` into ``K/8`` byte
planes, pads them to a ``torch._int_mm`` shape and recombines the plane products: about a dozen
kernel launches per call, which dominates decode-sized calls (1-16 rows). For ``K <= 32`` the ring
product needs no digits at all: arithmetic modulo ``2^32`` is exactly what 32-bit integer
multiply-add does, and ``Z_{2^K}`` is a quotient of ``Z_{2^32}``. So one kernel streams the int8
weight once and computes, per output element::

    y[r, j] = sum_i  u32(x[r, i]) * u32(sext(W8[j, i]))   (mod 2^32),   then  & (2^K - 1)

Exactness argument:

* ``x`` enters as the low 32 bits of each ring element (``x mod 2^32``; ``x mod 2^K`` is
  determined by it for ``K <= 32``), reinterpreted as ``uint32``.
* ``W8`` is sign-extended to int32 and reinterpreted as ``uint32``: the same residue mod ``2^32``.
* All products, partial sums, the in-register reduction and the cross-program ``atomic_add``
  (split-K) are **unsigned** 32-bit operations. Unsigned overflow wraps by definition (LLVM
  ``mul``/``add`` on ``i32``, PTX ``mul.lo.u32``/``add.u32``/``red.add.u32``, numpy ``uint32`` in the
  interpreter), there is no signed-overflow UB and no float anywhere. Wrapping addition is
  associative and commutative, so any reduction order (tile shape, warp shuffles, split-K
  atomics) gives the same bits.
* The result is reduced mod ``2^K`` (a mask) and returned as int32 bit patterns: the compact form
  of :func:`pgw.service.gpu_exec.to_compact` (the low 32 bits of the ring element).

Three kernel variants, one launch each, all bit-identical:

``sum`` (CUDA cores, the one-row kernel)
    Each tile is multiplied and reduced across lanes inside the loop. Memory-bound for one row.
``stream`` (CUDA cores, 1-8 rows, :data:`MULTIROW_MAX_ROWS`; ``GemvConfig.stream``)
    The memory-bound few-row variant: each weight tile is loaded and sign-extended once and
    multiplied into one ``(BLOCK_M, BLOCK_N)`` accumulator per row (``x`` is a few KB per row and
    stays resident in L1/L2); each accumulator is reduced once at the end, so the loop has no
    cross-lane traffic. Costs ``rows`` integer multiply-adds per weight byte: on a T4 that stays
    under the memory time up to about 8 rows. It serves 2-8 rows where the tensor-core variant
    is unavailable (T4 with current Triton) and a startup timing gate shows it beats the
    digit-GEMM path (``auto``), or everywhere it passed the self-check (``on``).
``dot`` (tensor cores, several rows; ``GemvConfig.dot``)
    Each ``x`` tile is split in registers into four balanced signed base-256 digits
    ``x = sum_i d_i 256^i (mod 2^32)``, ``d_i in [-128, 127]`` (peeled with wrapping unsigned subtract
    and logical shift), each digit plane is multiplied with the int8 weight tile by ``tl.dot``
    (int32 accumulation, exact because ``128 * 128 * n < 2^31``; the host caps the inner range per
    program), and the four products are recombined with unsigned shifts and adds mod ``2^32``.

Masked ``x`` entries are 0 (all-zero digits in ``dot``), so padding columns contribute nothing in
any variant.

**Launch configurations** come from a per-device table (:data:`DEVICE_TABLES`, keyed by
``torch.cuda.get_device_name``; measured with ``benchmarks/gpu_kernel_bench.py --gemv-sweep``,
which can also write a JSON table that ``PGW_SERVICE_TRITON_GEMV_TABLE`` loads) and otherwise
from the heuristic of :func:`gemv_config`. A table rule is used only after its configuration
passed the self-check on the device (:func:`self_check` checks every rule and every heuristic
variant; a failing rule is dropped, a failing few-row variant is switched off).

Triton is optional (it ships with CUDA builds of torch); :func:`triton_available` says whether it
imports. Set ``TRITON_INTERPRET=1`` before triton is first imported to run the kernels on CPU tensors
(tests; ``tests/conftest.py`` does this on hosts without CUDA).
Callers that serve traffic gate the kernel on :func:`self_check`, which compares it with the
digit-GEMM path on the target device (random and extreme operands, every supported row count,
K and launch configuration).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import threading
import types
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pgw.core.errors import ShapeError

__all__ = [
    "DEVICE_TABLES",
    "MULTIROW_MAX_ROWS",
    "MULTIROW_MODES",
    "TRITON_GEMV_KS",
    "TRITON_GEMV_MAX_ROWS",
    "GemvConfig",
    "TableRule",
    "configure",
    "device_name",
    "device_rules",
    "gemv_config",
    "load_table",
    "low32",
    "max_rows",
    "multirow_rows",
    "ring_gemv",
    "self_check",
    "stream_config",
    "table_to_json",
    "triton_available",
]

log = logging.getLogger(__name__)

#: Largest row count the fused kernel serves (decode, concurrent sessions, speculative verify).
TRITON_GEMV_MAX_ROWS = 16
#: Ring widths the fused kernel serves (``K <= 32``; multiples of 8 as :class:`RingParams` requires).
TRITON_GEMV_KS = (16, 24, 32)
#: Most rows of the memory-bound ``stream`` variant (one accumulator tile per row).
MULTIROW_MAX_ROWS = 8
#: ``PGW_SERVICE_TRITON_GEMV_MULTIROW``: when the ``stream`` variant serves 2-8 rows.
#: ``auto`` (default): only on devices without the tensor-core variant, for the row counts where a
#: startup timing gate shows it faster than the digit-GEMM path. ``on``: wherever it passed the
#: self-check. ``off``: never (the previous behaviour).
MULTIROW_MODES = ("auto", "on", "off")
ENV_MULTIROW = "PGW_SERVICE_TRITON_GEMV_MULTIROW"
ENV_TABLE = "PGW_SERVICE_TRITON_GEMV_TABLE"


@functools.cache
def triton_available() -> bool:
    """True if ``triton`` imports (it ships with CUDA builds of torch)."""
    try:
        import triton
        import triton.language  # noqa: F401
    except Exception as exc:  # ImportError, or a broken install
        log.info("triton unavailable (%s: %s); fused ring GEMV disabled", type(exc).__name__, str(exc)[:200])
        return False
    return True


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


@functools.cache
def _kernel(interpret: bool, variant: str) -> Any:
    """The jitted kernel of ``variant`` (``sum``, ``dot`` or ``stream``), lazily: importing this
    module must not need triton.

    ``triton.jit`` picks the interpreter or the compiler from ``TRITON_INTERPRET`` when it wraps the
    function, so the wrapper is cached per mode.
    """
    del interpret  # cache key only
    import triton

    from pgw.core import _triton_gemv_kernel as kmod

    fn = {"sum": kmod.ring_gemv_kernel, "dot": kmod.ring_gemv_dot_kernel, "stream": kmod.ring_gemv_rows_kernel}[variant]
    if variant == "stream":
        # the device helpers are jitted here too (same mode as the kernel), bound through a private copy
        # of the module globals so kernels of different modes never share a helper wrapper
        fn = _rebind(fn, {h: triton.jit(getattr(kmod, h)) for h in kmod.ROWS_HELPERS})
    return triton.jit(fn)


def _rebind(fn: Any, extra: dict[str, Any]) -> Any:
    """A copy of ``fn`` whose globals are its module's globals plus ``extra`` (same code and source)."""
    g = dict(fn.__globals__)
    g.update(extra)
    new = types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__annotations__ = dict(fn.__annotations__)  # tl.constexpr parameters
    new.__doc__ = fn.__doc__
    new.__qualname__ = fn.__qualname__
    new.__module__ = fn.__module__
    return new


@dataclass(frozen=True)
class GemvConfig:
    """Launch configuration: tile sizes, split-K factor, warps and kernel variant.

    ``dot=False, stream=False``: CUDA-core ``sum`` kernel (``uint32`` multiply-adds of ``x`` with the
    sign-extended weight, reduced across lanes per tile).
    ``stream=True``: CUDA-core few-row kernel (one accumulator tile per row, reduced once at the end);
    needs ``block_r <= 8``.
    ``dot=True``: tensor-core kernel (four balanced signed byte digits of ``x``, int8 ``tl.dot`` with
    int32 accumulation, recombined mod 2^32); needs ``block_r, block_m >= 16`` and ``block_n >= 32``.
    Tile sizes are powers of two (``tl.arange``).
    """

    block_r: int
    block_m: int
    block_n: int
    split_k: int = 1
    num_warps: int = 4
    dot: bool = False
    stream: bool = False

    @property
    def variant(self) -> str:
        """Kernel variant: ``sum``, ``stream`` or ``dot``."""
        return "dot" if self.dot else "stream" if self.stream else "sum"

    def validate(self) -> None:
        """Raise :class:`ShapeError` for a configuration no kernel variant can launch."""
        # the stream variant takes any block_r <= 8 (one accumulator per row, launched with exactly b)
        for name in ("block_m", "block_n", "num_warps") + (() if self.stream else ("block_r",)):
            v = int(getattr(self, name))
            if v < 1 or v & (v - 1):
                raise ShapeError(f"GemvConfig.{name}={v} must be a positive power of two")
        if self.split_k < 1:
            raise ShapeError(f"GemvConfig.split_k={self.split_k} must be >= 1")
        if self.dot and self.stream:
            raise ShapeError("GemvConfig: dot and stream are exclusive")
        if self.dot and (self.block_r < 16 or self.block_m < 16 or self.block_n < 32):
            raise ShapeError(f"ring_gemv: tensor-core tiles need block_r, block_m >= 16 and block_n >= 32, got {self}")
        if self.stream and self.block_r > MULTIROW_MAX_ROWS:
            raise ShapeError(f"ring_gemv: the stream variant serves at most {MULTIROW_MAX_ROWS} rows, got {self}")


@dataclass(frozen=True)
class TableRule:
    """One entry of a device's launch table: ``rows`` rows, ``m`` and padded ``nw`` in the closed ranges.

    The first matching rule of a device wins (:func:`gemv_config`). ``rows`` is exact (a rule's
    ``config.block_r`` must cover it).
    """

    rows: int
    config: GemvConfig
    m: tuple[int, int] = (1, 1 << 31)
    nw: tuple[int, int] = (1, 1 << 31)
    note: str = field(default="", compare=False)

    def matches(self, b: int, m: int, nw: int) -> bool:
        return b == self.rows and self.m[0] <= m <= self.m[1] and self.nw[0] <= nw <= self.nw[1]

    def validate(self) -> None:
        self.config.validate()
        if not 1 <= self.rows <= self.config.block_r:
            raise ShapeError(f"table rule for {self.rows} rows has block_r={self.config.block_r}")
        if self.rows > TRITON_GEMV_MAX_ROWS:
            raise ShapeError(f"table rule for {self.rows} rows > {TRITON_GEMV_MAX_ROWS}")
        for lo, hi in (self.m, self.nw):
            if not 1 <= lo <= hi:
                raise ShapeError(f"table rule range ({lo}, {hi}) is empty")


def _exact(rows: int, m: int, nw: int, cfg: GemvConfig, note: str) -> TableRule:
    return TableRule(rows, cfg, (m, m), (nw, nw), note)


#: Measured launch configurations per GPU model (``torch.cuda.get_device_name``). Tesla T4: the
#: config sweeps of ``benchmarks/results/kaggle/triton_gemv`` (four runs; device time per call at the
#: Qwen2.5-7B shapes, heuristic -> table): qkv 0.115 -> ~0.108 ms, o 0.069 -> ~0.060 ms, gate_up
#: 0.70 -> ~0.62 ms, down 0.37 -> ~0.30 ms, about 0.17 ms per layer (~4.7 ms per 7B token).
#: The wide (gate_up-like) and deep (down-like) rules also cover other models' shapes in range.
DEVICE_TABLES: dict[str, tuple[TableRule, ...]] = {
    "Tesla T4": (
        _exact(1, 4608, 3584, GemvConfig(1, 16, 512, 8), "7B qkv"),
        _exact(1, 3584, 3584, GemvConfig(1, 8, 1024, 1), "7B o"),
        TableRule(1, GemvConfig(1, 32, 1024, 8), (16384, 1 << 31), (2048, 8192), "wide (7B gate_up)"),
        TableRule(1, GemvConfig(1, 32, 1024, 8), (2048, 8192), (8192, 1 << 31), "deep (7B down)"),
    ),
}

_GUARD = threading.Lock()
#: ``128 * 128 * n < 2^31`` keeps each tensor-core digit dot exact in int32
_DOT_MAX_INNER = 131_040

_SELF_CHECK: dict[str, bool] = {}
#: devices where the tensor-core variant failed its check: it is never used there
_NO_DOT: set[str] = set()
#: table rules that passed the self-check, per device key
_RULES: dict[str, tuple[TableRule, ...]] = {}
#: most rows the stream variant serves per device key (0 or absent: not used)
_MULTIROW: dict[str, int] = {}
#: rules added at run time (:func:`configure`, a JSON table), keyed by device name; tried first
_EXTRA_TABLES: dict[str, tuple[TableRule, ...]] = {}
_SETTINGS: dict[str, Any] = {}


def configure(*, multirow: str | None = None, table: str | os.PathLike[str] | dict[str, Any] | None = None,
              rules: dict[str, Iterable[TableRule]] | None = None) -> None:
    """Set the few-row mode and extra table rules programmatically (before the device self-check).

    Args:
        multirow: ``auto``, ``on`` or ``off`` (overrides ``PGW_SERVICE_TRITON_GEMV_MULTIROW``).
        table: A JSON table (path or parsed dict, format of :func:`table_to_json`) added in front of
            the built-in rules (overrides ``PGW_SERVICE_TRITON_GEMV_TABLE``).
        rules: Extra rules per device name, tried before the built-in ones.

    Devices that already ran :func:`self_check` keep their verdict.
    """
    with _GUARD:
        if multirow is not None:
            if multirow not in MULTIROW_MODES:
                raise ValueError(f"multirow must be one of {MULTIROW_MODES}, got {multirow!r}")
            _SETTINGS["multirow"] = multirow
        if table is not None:
            for name, rs in load_table(table).items():
                _EXTRA_TABLES[name] = tuple(rs) + _EXTRA_TABLES.get(name, ())
        for name, rs in (rules or {}).items():
            _EXTRA_TABLES[name] = tuple(rs) + _EXTRA_TABLES.get(name, ())


def _multirow_mode() -> str:
    mode = _SETTINGS.get("multirow") or os.environ.get(ENV_MULTIROW, "auto").strip().lower() or "auto"
    if mode not in MULTIROW_MODES:
        log.warning("ignoring %s=%r (not one of %s); using auto", ENV_MULTIROW, mode, MULTIROW_MODES)
        return "auto"
    return str(mode)


def table_to_json(rules: dict[str, Iterable[TableRule]]) -> dict[str, Any]:
    """JSON form of per-device rules: ``{"devices": {name: [{"rows", "m", "nw", "config", "note"}]}}``."""
    return {"devices": {name: [{"rows": r.rows, "m": list(r.m), "nw": list(r.nw), "config": asdict(r.config),
                                "note": r.note} for r in rs] for name, rs in rules.items()}}


def load_table(src: str | os.PathLike[str] | dict[str, Any]) -> dict[str, tuple[TableRule, ...]]:
    """Parse a JSON table (path or dict). Never raises: a malformed file or top-level structure loads as
    no rules, a malformed device entry as no rules for that device, and malformed rules are skipped,
    each with a warning."""
    try:
        if isinstance(src, dict):
            data = src
        else:
            with open(src) as f:
                data = json.load(f)
        devices = dict(data.get("devices", {}))
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        log.warning("triton GEMV table %s not loaded: %s: %s", src, type(exc).__name__, exc)
        return {}
    out: dict[str, tuple[TableRule, ...]] = {}
    for name, raw in devices.items():
        try:
            if isinstance(raw, (str, bytes, dict)):
                raise TypeError(f"expected a list of rules, got {type(raw).__name__}")
            entries = list(raw)
        except (TypeError, ValueError) as exc:
            log.warning("triton GEMV table: skipping the rules for %s: %s", name, exc)
            continue
        rules: list[TableRule] = []
        for e in entries:
            try:
                c = dict(e["config"])
                cfg = GemvConfig(int(c["block_r"]), int(c["block_m"]), int(c["block_n"]), int(c.get("split_k", 1)),
                                 int(c.get("num_warps", 4)), bool(c.get("dot", False)), bool(c.get("stream", False)))
                rule = TableRule(int(e["rows"]), cfg, (int(e["m"][0]), int(e["m"][1])),
                                 (int(e["nw"][0]), int(e["nw"][1])), str(e.get("note", "")))
                rule.validate()
            except (KeyError, TypeError, ValueError, IndexError, ShapeError) as exc:
                log.warning("triton GEMV table: skipping rule %r for %s: %s", e, name, exc)
                continue
            rules.append(rule)
        out[str(name)] = tuple(rules)
    return out


def _env_table() -> dict[str, tuple[TableRule, ...]]:
    path = os.environ.get(ENV_TABLE, "").strip()
    return load_table(path) if path else {}


def device_name(device: torch.device | str) -> str:
    """Table key of ``device``: the CUDA device name (``"Tesla T4"``), else the device type."""
    dev = torch.device(device)
    if dev.type == "cuda":
        try:
            return str(torch.cuda.get_device_name(dev.index if dev.index is not None else 0))
        except Exception:  # pragma: no cover - broken driver
            return "cuda"
    return dev.type


def device_rules(device: torch.device | str) -> tuple[TableRule, ...]:
    """Candidate table rules for ``device`` (run-time and env tables first, then built-in)."""
    name = device_name(device)
    return _EXTRA_TABLES.get(name, ()) + _env_table().get(name, ()) + DEVICE_TABLES.get(name, ())


def _key(device: torch.device | str) -> str:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        try:
            return f"cuda:{torch.cuda.current_device()}"
        except Exception:  # pragma: no cover - no CUDA
            return "cuda:0"
    return str(dev)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _pow2_at_least(v: int) -> int:
    return 1 << max(0, (v - 1).bit_length())


@functools.cache
def _sm_count(device_index: int) -> int:
    try:
        return int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    except Exception:
        return 40


def _stream_tiles(b: int) -> tuple[int, int]:
    """``(block_m, block_n)`` of the stream heuristic: 2048-element tiles for 1-2 rows, 1024 above (keeps
    the ``b`` accumulator tiles within ~128 registers per thread at 4 warps; no spills on sm_75)."""
    return (16, 128) if b <= 2 else (16, 64)


def gemv_config(b: int, m: int, nw: int, device: torch.device | None = None) -> GemvConfig:
    """Launch configuration for ``b`` rows, ``m`` outputs and ``nw`` weight columns on ``device``.

    A matching rule of the device's table that passed the self-check comes first. Otherwise one row
    uses the CUDA-core kernel (``BLOCK_N = 1024``); 2-8 rows use the stream variant where it is
    enabled (:func:`multirow_rows`); other row counts use the tensor-core kernel (rows padded to 16)
    where it compiled and passed the self-check. Narrow outputs (few programs) split the inner
    dimension across programs that combine with wrapping atomic adds (aiming at about 2048
    programs, at most 8 splits). Heuristic tuned on a T4 with the 7B shapes.
    """
    key = _key(device) if device is not None else None
    if key is not None:
        for rule in _RULES.get(key, ()):
            # a stream rule for several rows applies only where the stream variant is on for them
            if rule.matches(b, m, nw) and not (rule.config.stream and 1 < b and b > _MULTIROW.get(key, 0)):
                return rule.config
    if b <= 1:
        block_r, block_m, block_n, dot = 1, (32 if m >= 16384 else 16), 1024, False
    elif key is not None and b <= _MULTIROW.get(key, 0):
        return stream_config(b, m, nw, device)
    elif key is not None and key in _NO_DOT:
        block_r, block_m, block_n, dot = _pow2_at_least(b), 8, 512, False  # correct but compute-bound
    else:
        block_r, block_m, block_n, dot = max(16, _pow2_at_least(b)), 64, 128, True
    return GemvConfig(block_r, block_m, block_n, _split(m, nw, block_m, device), 4, dot)


def _split(m: int, nw: int, block_m: int, device: torch.device | None) -> int:
    """Split-K factor: about ``max(2048, 16 * SMs)`` programs, at most 8 splits, >= 512 columns each."""
    sms = _sm_count(device.index or 0) if device is not None and device.type == "cuda" else 40
    programs = _cdiv(m, block_m)
    target = max(2048, 16 * sms)
    return max(1, min(8, _cdiv(target, programs), max(1, nw // 512)))


def stream_config(b: int, m: int, nw: int, device: torch.device | None = None) -> GemvConfig:
    """Heuristic configuration of the stream variant for ``1 <= b <= 8`` rows (enabled or not)."""
    if not 1 <= b <= MULTIROW_MAX_ROWS:
        raise ShapeError(f"the stream variant serves 1..{MULTIROW_MAX_ROWS} rows, got {b}")
    block_m, block_n = _stream_tiles(b)
    return GemvConfig(b, block_m, block_n, _split(m, nw, block_m, device), 4, stream=True)


def low32(x: torch.Tensor) -> torch.Tensor:
    """The low 32 bits of int64 ``x`` as an int32 tensor of the same shape (exact, no range cast)."""
    if x.dtype != torch.int64:
        raise ShapeError(f"expected int64 ring elements, got {x.dtype}")
    if sys.byteorder == "little" and x.is_contiguous():
        return x.view(torch.int32).view(*x.shape, 2)[..., 0]
    return (((x & 0xFFFFFFFF) + (1 << 31)) & 0xFFFFFFFF).sub_(1 << 31).to(torch.int32)


def ring_gemv(
    x32: torch.Tensor,
    w: torch.Tensor,
    k: int,
    *,
    n: int | None = None,
    m: int | None = None,
    out: torch.Tensor | None = None,
    config: GemvConfig | None = None,
) -> torch.Tensor:
    """``x @ w[:m, :n]^T mod 2^K`` as int32 bit patterns ``(b, m)`` with one kernel launch.

    Args:
        x32: int32 ``(b, n)`` low 32 bits of the ring elements (:func:`low32`), unit stride in ``n``;
            ``1 <= b <= TRITON_GEMV_MAX_ROWS`` (larger ``b`` works but is not tuned).
        w: int8 ``(mw, nw)`` weight with unit column stride, ``mw >= m`` and ``nw >= n``. Columns past
            ``n`` (zero padding of :func:`pgw.core.ringmm.pad_weight_cuda`) are streamed but meet zero
            ``x``, so their content does not matter; whole padded rows keep loads aligned.
        k: Ring bits, ``K <= 32``.
        n: Inner dimension (default ``x32.shape[1]``).
        m: Output features (default ``w.shape[0]``).
        out: Optional int32 ``(b, m)`` output with unit column stride.
        config: Launch configuration (default :func:`gemv_config`).

    Returns:
        int32 ``(b, m)``: ``y mod 2^K`` (for ``K = 32`` the two's-complement bits of ``y mod 2^32``).
    """
    if not 1 <= k <= 32:
        raise ShapeError(f"fused ring GEMV needs K <= 32, got {k}")
    if x32.dtype != torch.int32 or x32.ndim != 2 or w.dtype != torch.int8 or w.ndim != 2:
        raise ShapeError(f"ring_gemv needs int32 x (b, n) and int8 w (m, n), got {x32.dtype} {tuple(x32.shape)}, "
                         f"{w.dtype} {tuple(w.shape)}")
    b = int(x32.shape[0])
    n = int(x32.shape[1]) if n is None else int(n)
    m = int(w.shape[0]) if m is None else int(m)
    mw, nw = (int(s) for s in w.shape)
    if n > int(x32.shape[1]) or n > nw or m > mw:
        raise ShapeError(f"ring_gemv: n={n}, m={m} exceed x {tuple(x32.shape)} / w {tuple(w.shape)}")
    if x32.stride(1) != 1 and n > 1:
        x32 = x32.contiguous()
    if w.stride(1) != 1 and nw > 1:
        raise ShapeError("ring_gemv: weight needs unit column stride")
    if x32.device != w.device:
        raise ShapeError(f"ring_gemv: x on {x32.device}, w on {w.device}")
    cfg = config or gemv_config(b, m, nw, w.device)
    if cfg.block_r < b:
        raise ShapeError(f"ring_gemv: block_r={cfg.block_r} < rows={b}")
    if out is not None and (out.dtype != torch.int32 or tuple(out.shape) != (b, m) or (out.stride(1) != 1 and m > 1)):
        raise ShapeError(f"ring_gemv: bad out {out.dtype} {tuple(out.shape)}")
    cfg.validate()
    # inner range per split-K program, a multiple of BLOCK_N (keeps loads aligned)
    per = -(-_cdiv(max(1, nw), max(1, int(cfg.split_k))) // cfg.block_n) * cfg.block_n
    if cfg.dot and per > _DOT_MAX_INNER:  # keep every int32 dot accumulator exact (no wrap at all)
        per = _DOT_MAX_INNER // cfg.block_n * cfg.block_n
    split = -(-max(1, nw) // per)  # the programs actually launched along K (the atomics need a zeroed out)
    if out is None:
        out = (torch.zeros if split > 1 else torch.empty)((b, m), dtype=torch.int32, device=w.device)
    elif split > 1:
        out.zero_()
    if b == 0 or m == 0:
        return out
    kmask = 0 if k == 32 else (1 << k) - 1
    grid = (-(-m // cfg.block_m), split)
    # split-K partial sums are added mod 2^32 by the atomics and masked afterwards
    kmask_launch = -1 if (kmask == 0 or split > 1) else kmask
    with _launch_guard(w.device):
        _kernel(_interpret(), cfg.variant)[grid](
            x32, w, out, b, n, m, nw, per, kmask_launch,
            x32.stride(0), w.stride(0), out.stride(0),
            # the stream variant keeps one accumulator per row: launched with exactly b of them
            BLOCK_R=b if cfg.stream else cfg.block_r, BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n,
            ATOMIC=split > 1, num_warps=cfg.num_warps,
        )
    if split > 1 and kmask:
        out.bitwise_and_(kmask)
    return out


class _launch_guard:  # noqa: N801 - context manager used like a function
    """Selects the weight's CUDA device for the launch (no-op on CPU)."""

    def __init__(self, device: torch.device):
        self._dev = device
        self._ctx: Any = None

    def __enter__(self) -> None:
        if self._dev.type == "cuda":
            self._ctx = torch.cuda.device(self._dev)
            self._ctx.__enter__()

    def __exit__(self, *exc: object) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*exc)


# ---- self-check -----------------------------------------------------------------------------
def _reference(x: torch.Tensor, w8: torch.Tensor, k: int) -> torch.Tensor:
    """Existing digit-GEMM path on ``w8``'s device (int64 ``[0, 2^K)``), the self-check's oracle.

    CUDA: :class:`pgw.core.ringmm.PreparedWeight` (``cuda_int8``, the service's eager path). CPU: the
    Python-int reference for small operands, else the float digit GEMM (``cpu_f64``, always exact).
    On CUDA the check therefore proves "bit-identical to today's path", not independent correctness:
    a digit-path bug would be shared, so the digit path keeps its own tests against the Python-int
    reference (``ringmm_reference``).
    """
    from pgw.core.ringmm import PreparedWeight, ringmm_reference

    if w8.device.type == "cuda":
        return PreparedWeight(w8, device=w8.device, backend="cuda_int8").matmul(x.to(w8.device), k).cpu()
    if int(x.shape[0]) * w8.numel() <= 1 << 16:
        return torch.from_numpy(ringmm_reference(x, w8, k))
    return PreparedWeight(w8.cpu(), device="cpu", backend="cpu_f64").matmul(x.cpu(), k)


Case = tuple[str, torch.Tensor, torch.Tensor, int, "GemvConfig | None"]


def _check_cases(device: torch.device, quick: bool) -> list[Case]:
    """Base cases (heuristic configs): random operands at every row count and K, and extreme operands."""
    gen = torch.Generator().manual_seed(20261120)
    cases: list[Case] = []
    shapes = [(1, 1), (37, 129), (64, 4097)] if quick else [(1, 1), (37, 129), (300, 1000), (64, 4097), (96, 18944)]
    rows_all = [1, 2, 3, 5, 8, 9, 16] if not quick else [1, 3, 8, 16]
    for k in TRITON_GEMV_KS:
        top = (1 << k) - 1
        for m, n in shapes:
            for b in rows_all:
                x = torch.randint(0, 1 << k, (b, n), dtype=torch.int64, generator=gen)
                w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
                cases.append((f"random K={k} b={b} m={m} n={n}", x, w, k, None))
        m, n = shapes[-1]
        for wv in (-128, 127):
            for xv in (top, 1 << (k - 1), (1 << (k - 1)) - 1):
                b = TRITON_GEMV_MAX_ROWS
                x = torch.full((b, n), xv, dtype=torch.int64)
                w = torch.full((m, n), wv, dtype=torch.int8)
                cases.append((f"extreme K={k} W={wv} x={xv} m={m} n={n}", x, w, k, None))
    return cases


def _config_cases(entries: Iterable[tuple[int, GemvConfig, tuple[int, int] | None]], quick: bool,
                  seed: int = 7) -> list[Case]:
    """Cases for explicit configurations: ``(rows, config, shape or None)``.

    Every configuration runs on a small odd shape that exercises masked edge tiles and (for
    ``split_k > 1``) the split-K atomics: random operands for every K, and all-extreme operands
    (every partial sum wraps). With a shape (a table rule's own range), also a random case there.
    """
    gen = torch.Generator().manual_seed(seed)
    cases: list[Case] = []
    for b, cfg, shape in entries:
        m = 3 * cfg.block_m + 1
        n = 2 * cfg.block_n * cfg.split_k + 17
        for k in (32,) if quick else TRITON_GEMV_KS:
            x = torch.randint(0, 1 << k, (b, n), dtype=torch.int64, generator=gen)
            w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
            cases.append((f"random K={k} b={b} m={m} n={n} {cfg}", x, w, k, cfg))
        k = 32
        for wv, xv in ((-128, (1 << k) - 1), (127, (1 << (k - 1)) - 1), (-128, 1 << (k - 1))):
            x = torch.full((b, n), xv, dtype=torch.int64)
            w = torch.full((m, n), wv, dtype=torch.int8)
            cases.append((f"extreme b={b} W={wv} x={xv} m={m} n={n} {cfg}", x, w, k, cfg))
        if shape is not None:
            m, n = shape
            x = torch.randint(0, 1 << 32, (b, n), dtype=torch.int64, generator=gen)
            w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
            cases.append((f"rule shape K=32 b={b} m={m} n={n} {cfg}", x, w, 32, cfg))
    return cases


def _run_cases(dev: torch.device, cases: Iterable[Case]) -> list[str]:
    from pgw.core.ringmm import pad_weight_cuda

    bad: list[str] = []
    for name, x, w8, k, cfg in cases:
        if cfg is None and int(x.shape[0]) > max_rows(dev):
            continue  # not served on this device
        want = _reference(x, w8.to(dev), k).to(torch.int64)
        wpad = pad_weight_cuda(w8, dev) if dev.type == "cuda" else w8
        got = ring_gemv(low32(x).to(dev), wpad, k, n=int(x.shape[1]), m=int(w8.shape[0]), config=cfg)
        got64 = got.cpu().to(torch.int64) & ((1 << k) - 1)
        if not torch.equal(got64, want):
            bad.append(name)
    return bad


def heuristic_variants(*, stream: bool) -> list[tuple[int, GemvConfig]]:
    """Every compiled variant the heuristic can launch: one-row ``sum`` tiles (both ``block_m``) and,
    with ``stream``, the stream tiles for 2..8 rows, each without and with split-K."""
    out: list[tuple[int, GemvConfig]] = []
    for bm in (16, 32):
        out += [(1, GemvConfig(1, bm, 1024, 1)), (1, GemvConfig(1, bm, 1024, 3))]
    if stream:
        for b in range(2, MULTIROW_MAX_ROWS + 1):
            bm, bn = _stream_tiles(b)
            out += [(b, GemvConfig(b, bm, bn, 1, 4, stream=True)), (b, GemvConfig(b, bm, bn, 3, 4, stream=True))]
    return out


def _rule_shape(rule: TableRule) -> tuple[int, int] | None:
    """A shape inside the rule's range (its lower corner), if small enough to check quickly."""
    m, nw = rule.m[0], rule.nw[0]
    return (m, nw) if m * nw <= 1 << 26 else None


def check_against_reference(device: torch.device | str, *, quick: bool = False) -> list[str]:
    """Run the fused kernel on ``device`` for the base self-check cases; returns the failing case names."""
    dev = torch.device(device)
    return _run_cases(dev, _check_cases(dev, quick))


def check_configs(device: torch.device | str, entries: Iterable[tuple[int, GemvConfig, tuple[int, int] | None]],
                  *, quick: bool = False) -> list[str]:
    """Compare explicit configurations with the digit-GEMM path on ``device``; returns failing case names."""
    dev = torch.device(device)
    return _run_cases(dev, _config_cases(list(entries), quick))


def max_rows(device: torch.device | str) -> int:
    """Most rows the fused kernel serves on ``device``: :data:`TRITON_GEMV_MAX_ROWS` with the tensor-core
    variant; without it, the rows of the stream variant (:func:`multirow_rows`) or 1 (the ``sum``
    kernel is slower than the digit-GEMM path for several rows)."""
    key = _key(device)
    if key in _NO_DOT:
        return max(1, _MULTIROW.get(key, 0))
    return TRITON_GEMV_MAX_ROWS


def multirow_rows(device: torch.device | str) -> int:
    """Most rows the stream variant serves on ``device`` (0: not used); set by :func:`self_check`."""
    return _MULTIROW.get(_key(device), 0)


def _run_check(key: str, device: torch.device | str) -> tuple[bool, str]:
    try:
        bad = check_against_reference(device)
        if not bad:
            bad = check_configs(device, [(b, c, None) for b, c in heuristic_variants(stream=False)])
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
    return not bad, f"{len(bad)} failing cases, e.g. {bad[:3]}"


def _checked_rules(key: str, device: torch.device | str) -> tuple[TableRule, ...]:
    """The device's table rules whose configuration passes the self-check (others are logged and dropped)."""
    ok: list[TableRule] = []
    try:
        candidates = device_rules(device)
    except Exception as exc:  # e.g. a table file that changed under us; load_table itself never raises
        log.error("fused ring GEMV: table rules for %s not loaded (%s: %s); using the heuristic", key,
                  type(exc).__name__, str(exc)[:200])
        return ()
    for rule in candidates:
        try:
            rule.validate()
            if rule.config.dot and key in _NO_DOT:
                raise ShapeError("tensor-core variant unavailable on this device")
            bad = check_configs(device, [(rule.rows, rule.config, _rule_shape(rule))])
        except Exception as exc:
            bad = [f"{type(exc).__name__}: {str(exc)[:200]}"]
        if bad:
            log.error("fused ring GEMV: table rule %s (%s) failed its self-check on %s (%s); using the heuristic",
                      rule.config, rule.note, key, bad[:2])
            continue
        ok.append(rule)
    return tuple(ok)


def _cuda_ms(fn: Callable[[], object], iters: int) -> float:  # pragma: no cover - GPU only
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):  # best of three short batches: robust to a noisy neighbour
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, float(start.elapsed_time(end)) / iters)
    return best


#: shapes of the startup timing gate (a wide and a deep linear, 64 MB of int8 each)
GATE_SHAPES = ((16384, 4096), (4096, 16384))
#: the stream variant must beat the digit-GEMM path by this factor at a row count to serve it
GATE_MARGIN = 1.03


def _timing_gate(device: torch.device) -> int:  # pragma: no cover - GPU only
    """Largest ``R`` in 2..8 such that the stream variant beats the digit-GEMM path (device time, K=32)
    at every row count ``2..R`` on :data:`GATE_SHAPES`; 0 if it loses at 2 rows. Public random data."""
    from pgw.core.ringmm import offset_byte_consts, offset_byte_core, offset_byte_rows, pad_weight_cuda

    gen = torch.Generator(device=device).manual_seed(1)
    k = 32
    sh, pm = offset_byte_consts(k, device)
    ws = []
    for m, n in GATE_SHAPES:
        wpad = pad_weight_cuda(torch.randint(-128, 128, (m, n), dtype=torch.int8, device=device, generator=gen), device)
        ws.append((m, n, wpad, wpad.to(torch.int64).sum(1)))
    best = 0
    timings = []
    for b in range(2, MULTIROW_MAX_ROWS + 1):
        t_digit = t_stream = 0.0
        for m, n, wpad, rowsum in ws:
            x = torch.randint(0, 1 << 32, (b, n), dtype=torch.int64, device=device, generator=gen)
            x32 = low32(x)
            p = offset_byte_rows(k, b)
            t_digit += _cuda_ms(lambda x=x, w=wpad, r=rowsum, p=p: offset_byte_core(x, k, w, r, sh, pm, p), 10)
            cfg = _stream_choice(_key(device), b, m, int(wpad.shape[1]), device)
            t_stream += _cuda_ms(lambda x32=x32, w=wpad, m=m, n=n, c=cfg: ring_gemv(x32, w, k, n=n, m=m, config=c), 10)
        timings.append(f"{b}:{t_stream:.3f}/{t_digit:.3f}")
        if t_stream * GATE_MARGIN > t_digit:
            break
        best = b
    del ws
    log.info("fused ring GEMV timing gate on %s (rows: stream/digit ms) %s -> stream variant up to %d rows",
             device, " ".join(timings), best)
    return best


def _stream_choice(key: str, b: int, m: int, nw: int, device: torch.device) -> GemvConfig:
    """The stream configuration :func:`gemv_config` will use once the variant is on: a checked stream
    rule of the device table, else the heuristic."""
    for rule in _RULES.get(key, ()):
        if rule.matches(b, m, nw):
            return rule.config if rule.config.stream else stream_config(b, m, nw, device)
    return stream_config(b, m, nw, device)


def _can_time(device: torch.device) -> bool:
    """The ``auto`` timing gate needs CUDA events."""
    return device.type == "cuda"


def _enable_multirow(key: str, device: torch.device | str) -> int:
    """Self-check (and in ``auto`` mode time) the stream variant; returns the rows it serves (0 = off)."""
    dev = torch.device(device)
    mode = _multirow_mode()
    if mode == "off" or (mode == "auto" and (key not in _NO_DOT or not _can_time(dev))):
        return 0
    entries = [(b, c, None) for b, c in heuristic_variants(stream=True) if c.stream]
    entries += [(r.rows, r.config, _rule_shape(r)) for r in _RULES.get(key, ()) if r.config.stream]
    try:
        bad = check_configs(dev, entries)
    except Exception as exc:
        bad = [f"{type(exc).__name__}: {str(exc)[:300]}"]
    if bad:
        log.error("fused ring GEMV stream variant failed its self-check on %s (%d cases, e.g. %s); several rows "
                  "use the digit-GEMM path", key, len(bad), bad[:3])
        return 0
    if mode == "on":
        return MULTIROW_MAX_ROWS
    try:
        return max(0, min(MULTIROW_MAX_ROWS, int(_timing_gate(dev))))
    except Exception as exc:
        log.error("fused ring GEMV timing gate failed on %s (%s: %s); stream variant off", key,
                  type(exc).__name__, str(exc)[:200])
        return 0


#: Oldest CUDA compute capability with an int8 tensor-core ``tl.dot`` in Triton (MMA v2 is sm_80+).
DOT_MIN_CAPABILITY = (8, 0)


def _dot_unsupported(device: torch.device | str) -> str | None:
    """Why the tensor-core variant cannot compile on ``device`` (``None``: try it).

    Triton lowers ``tl.dot`` to tensor cores (MMA) only from sm_80. On older GPUs (T4 = sm_75) it
    takes the FMA fallback, which up-casts the operands with ``arith.extf``: invalid for int8, so the
    compile fails in ``TritonGPUAccelerateMatmul`` ("PassManager::run failed"; reproduced offline with
    Triton 3.9 for sm_75, and on Kaggle and Modal T4s). An fp16 digit variant does compile there but
    also runs on the FMA path (no ``mma`` in the PTX), so it would be no faster than the CUDA-core
    kernels. Deciding from the capability up front makes the choice deterministic and skips a
    failing compile at every start.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    try:
        cc = tuple(torch.cuda.get_device_capability(dev.index if dev.index is not None else 0))
    except Exception:  # pragma: no cover - broken driver: let the self-check decide
        return None
    if cc < DOT_MIN_CAPABILITY:
        return (f"compute capability {cc[0]}.{cc[1]} < {DOT_MIN_CAPABILITY[0]}.{DOT_MIN_CAPABILITY[1]}: "
                "Triton has no int8 tensor-core dot there")
    return None


def self_check(device: torch.device | str) -> bool:
    """True if triton imports and the fused kernel matches the digit-GEMM path bit for bit on ``device``.

    Checks the heuristic configurations of every variant, then each table rule of the device (a rule
    that fails is dropped), then the stream variant for 2-8 rows (per :data:`MULTIROW_MODES`; a
    failure switches it off). Cached per device. Any exception (compile failure, unsupported GPU)
    counts as a failure of the part that raised it. On GPUs older than :data:`DOT_MIN_CAPABILITY`
    the tensor-core variant is not tried (:func:`_dot_unsupported`): the CUDA-core kernels are checked
    directly.
    """
    key = _key(device)
    with _GUARD:
        if key in _SELF_CHECK:
            return _SELF_CHECK[key]
        ok = False
        if triton_available():
            no_dot = _dot_unsupported(device)
            if no_dot is not None:
                log.info("fused ring GEMV on %s: tensor-core variant skipped (%s); CUDA-core kernels", key, no_dot)
                _NO_DOT.add(key)
            ok, why = _run_check(key, device)
            if not ok and key not in _NO_DOT:
                # retry with the CUDA-core kernel alone (one-row calls only)
                log.warning("fused ring GEMV (tensor-core variant) failed its self-check on %s (%s); retrying "
                            "with the CUDA-core variant for one-row calls", key, why)
                _NO_DOT.add(key)
                ok, why = _run_check(key, device)
            if ok:
                _RULES[key] = _checked_rules(key, device)
                rows = _enable_multirow(key, device)
                if rows:
                    _MULTIROW[key] = rows
                log.info("fused ring GEMV self-check passed on %s (up to %d rows; %d table rules; stream variant "
                         "up to %d rows)", key, max_rows(key), len(_RULES[key]), rows)
            else:
                log.error("fused ring GEMV self-check FAILED on %s (%s); using the digit-GEMM path", key, why)
        _SELF_CHECK[key] = ok
        return ok


def _reset(device: torch.device | str) -> None:
    """Forget the self-check verdict of ``device`` (tests)."""
    key = _key(device)
    with _GUARD:
        for d in (_SELF_CHECK, _RULES, _MULTIROW):
            d.pop(key, None)
        _NO_DOT.discard(key)
