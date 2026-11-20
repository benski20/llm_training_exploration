"""Loading model packs (all components use this; nobody reads pack files directly).

Tensors are read lazily through memory-mapped safetensors handles, so opening a pack is cheap and
each component touches only what it needs:

* compute service and mask factory: ``components=("linears",)``, then :meth:`ModelPack.load_linear`,
  :meth:`ModelPack.iter_linears` or :meth:`ModelPack.iter_layers` (streaming by layer).
* gateway: ``components=("plain", "tokenizer")``, then :meth:`ModelPack.load_plain` /
  :meth:`ModelPack.load_plain_tensors` and :attr:`ModelPack.tokenizer_dir`.

A pack directory may also be a *subset* produced by :meth:`ModelPack.export` (for example only
the linear shards for the service). The manifest, and therefore the fingerprint, is always complete.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from pgw.core.errors import ConfigError, FingerprintMismatchError, ShapeError
from pgw.core.fingerprint import require_same_fingerprint, sha256_file, short_fingerprint, validate_fingerprint
from pgw.core.ring import RingParams
from pgw.core.types import LinearId, LinearKind, LinearSpec
from pgw.modelpack.manifest import MANIFEST_FILE, TOKENIZER_DIR, ArchConfig, LinearEntry, Manifest, TensorEntry
from pgw.modelpack.st_io import hash_tensor_ranges, read_safetensors_header, tensor_sha256, torch_dtype

__all__ = ["COMPONENTS", "Component", "LinearMeta", "LinearWeights", "ModelPack", "VerifyMode", "row_norm_sq_max"]

log = logging.getLogger(__name__)

VerifyMode = Literal["none", "fingerprint", "full"]
Component = Literal["linears", "plain", "tokenizer"]
COMPONENTS: tuple[Component, ...] = ("linears", "plain", "tokenizer")


@dataclass(frozen=True)
class LinearWeights:
    """Public quantized weights of one masked linear group.

    Attributes:
        id: Linear id.
        w8: int8 ``(m, n)``.
        scale: float32 ``(m,)`` per-row scale ``sw``.
        bias: float32 ``(m,)`` or ``None``.
    """

    id: LinearId
    w8: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None

    @property
    def out_features(self) -> int:
        """``m``."""
        return int(self.w8.shape[0])

    @property
    def in_features(self) -> int:
        """``n``."""
        return int(self.w8.shape[1])

    @property
    def spec(self) -> LinearSpec:
        """As a :class:`~pgw.core.types.LinearSpec`."""
        return LinearSpec(self.id, self.out_features, self.in_features, self.bias is not None)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Float weight ``w8 * scale[:, None]`` (quality baselines / debugging)."""
        return (self.w8.to(torch.float64) * self.scale.to(torch.float64)[:, None]).to(dtype)


@dataclass(frozen=True)
class LinearMeta:
    """Everything about one linear group except its int8 weight (what a masked gateway needs).

    Attributes:
        id: Linear id.
        scale: float32 ``(m,)`` per-row scale ``sw``.
        bias: float32 ``(m,)`` or ``None``.
        out_features: ``m``.
        in_features: ``n``.
        row_norm_sq_max: ``max_j ||W8_j||^2`` as an exact integer from the manifest, or ``None`` for
            packs written before the field existed.
    """

    id: LinearId
    scale: torch.Tensor
    bias: torch.Tensor | None
    out_features: int
    in_features: int
    row_norm_sq_max: int | None = None


#: Elements per block of :func:`row_norm_sq_max` (bounds the int32/int64 temporaries to ~64 MB).
_NORM_BLOCK_ELEMS = 1 << 23


def row_norm_sq_max(w8: torch.Tensor) -> int:
    """``max_j sum_i W8[j, i]^2`` of an int8 ``(m, n)`` weight, exactly (int64 sums of int32 squares).

    Rows are processed in blocks so a large linear (e.g. 7B ``down``) never needs an int64 copy of
    the whole matrix. ``math.sqrt(float(result))`` equals the float64 ``max_j ||W8_j||_2`` the
    engine computed before (each row's float64 sum of squares is exact below ``2^53`` and ``sqrt``
    is correctly rounded and monotone).
    """
    if w8.dtype != torch.int8 or w8.ndim != 2:
        raise ShapeError(f"row_norm_sq_max expects a 2-D int8 weight, got {w8.dtype}{list(w8.shape)}")
    m, n = int(w8.shape[0]), int(w8.shape[1])
    if m == 0 or n == 0:
        return 0
    step = max(1, _NORM_BLOCK_ELEMS // n)
    best = 0
    for r0 in range(0, m, step):
        blk = w8[r0 : r0 + step].to(torch.int32)
        blk.mul_(blk)  # |w| <= 128: squares fit int32
        best = max(best, int(blk.sum(dim=1, dtype=torch.int64).max()))
        del blk
    return best


def _check_scale_bias(lid: str, scale: torch.Tensor, bias: torch.Tensor | None) -> None:
    """Scales must be finite and > 0 (``sw = max|row|/127``, 1.0 for zero rows); biases finite."""
    if not bool((torch.isfinite(scale) & (scale > 0)).all()):
        raise ConfigError(f"linear {lid}: weight scales must be finite and positive", details={"linear": lid})
    if bias is not None and not bool(torch.isfinite(bias).all()):
        raise ConfigError(f"linear {lid}: bias contains NaN or Inf", details={"linear": lid})


def _normalize_components(components: Iterable[str] | None) -> tuple[Component, ...]:
    if components is None:
        return COMPONENTS
    out: list[Component] = []
    for c in components:
        if c not in COMPONENTS:
            raise ConfigError(f"unknown pack component {c!r}; expected one of {COMPONENTS}")
        if c not in out:
            out.append(c)
    return tuple(out)


class ModelPack:
    """A model pack on disk.

    ``verify`` modes:
        ``"none"``: parse the manifest only.
        ``"fingerprint"``: check the manifest's fingerprint against its content (cheap, default).
        ``"full"``: additionally hash every file/tensor and compare to the manifest (slow, use at
        install time and in ``pgw doctor``).

    Loading is lazy (memory-mapped safetensors); components load only what they need:
    the service and factory only linears, the gateway only plain tensors and the tokenizer.

    Thread-safe for concurrent reads. Use as a context manager (or call :meth:`close`) to release
    file mappings early.
    """

    def __init__(self, root: Path, manifest: Manifest, components: Sequence[Component] = COMPONENTS):
        """Use :meth:`open`."""
        self.root = root
        self.manifest = manifest
        self.components: tuple[Component, ...] = tuple(components)
        self._handles: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._entries: dict[str, LinearEntry] = {e.id: e for e in manifest.linears}

    # ---- opening -----------------------------------------------------------------------------
    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        verify: VerifyMode = "fingerprint",
        expected_fingerprint: str | None = None,
        components: Iterable[Component] | None = None,
        workers: int | None = None,
    ) -> ModelPack:
        """Open a pack directory, verifying as requested.

        Args:
            path: Pack directory.
            verify: ``"none"``, ``"fingerprint"`` (default) or ``"full"``.
            expected_fingerprint: If given, the pack must have exactly this fingerprint (checked
                in every verify mode).
            components: Parts that must be present (and, with ``verify="full"``, are hashed):
                any of ``"linears"``, ``"plain"``, ``"tokenizer"``. Default: all three.
            workers: Threads used for hashing with ``verify="full"`` (default: min(4, CPUs)).

        Raises:
            FingerprintMismatchError: content or ``expected_fingerprint`` mismatch.
            ConfigError: missing/invalid manifest, inconsistent structure or missing files.
        """
        root = Path(path)
        if not root.is_dir():
            raise ConfigError(f"model pack directory {root} does not exist", details={"path": str(root)})
        if verify not in ("none", "fingerprint", "full"):
            raise ConfigError(f"invalid verify mode {verify!r}")
        comps = _normalize_components(components)
        manifest = Manifest.from_file(root / MANIFEST_FILE)
        if verify == "none":
            if expected_fingerprint is not None:
                require_same_fingerprint(
                    validate_fingerprint(expected_fingerprint),
                    validate_fingerprint(manifest.fingerprint or ""),
                    context="expected pack",
                )
        else:
            manifest.verify_fingerprint(
                validate_fingerprint(expected_fingerprint) if expected_fingerprint is not None else None
            )
            manifest.check_consistency()
        pack = cls(root, manifest, comps)
        pack._check_present()
        if verify == "full":
            pack.verify_files(workers=workers)
        log.info(
            "opened model pack %s fp=%s components=%s verify=%s",
            manifest.name, short_fingerprint(manifest.fingerprint or ""), ",".join(comps), verify,
        )
        return pack

    def _check_present(self) -> None:
        missing: list[str] = []
        for f in self._files_for(self.components, tokenizer=False):
            if not (self.root / f).is_file():
                missing.append(f)
        if "tokenizer" in self.components:
            for t in self.manifest.tokenizer:
                if not (self.tokenizer_dir / t.file).is_file():
                    missing.append(f"{TOKENIZER_DIR}/{t.file}")
        if missing:
            raise ConfigError(
                f"model pack {self.root.name} is missing {len(missing)} file(s) for components "
                f"{list(self.components)}: {missing[:5]}",
                details={"missing": missing[:20]},
            )

    def _files_for(self, components: Iterable[Component], tokenizer: bool = True) -> list[str]:
        files: list[str] = []
        for c in components:
            if c == "linears":
                files += self.manifest.linear_files()
            elif c == "plain":
                files += self.manifest.plain_files()
            elif c == "tokenizer" and tokenizer:
                files += [f"{TOKENIZER_DIR}/{t.file}" for t in self.manifest.tokenizer]
        return list(dict.fromkeys(files))

    def close(self) -> None:
        """Drop cached file handles (memory maps are released when tensors are freed)."""
        with self._lock:
            self._handles.clear()

    def __enter__(self) -> ModelPack:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ModelPack({self.name!r}, fp={short_fingerprint(self.fingerprint)}, root={self.root})"

    # ---- metadata --------------------------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """Verified model fingerprint."""
        fp = self.manifest.fingerprint
        if fp is None:  # pragma: no cover - open() rejects this
            raise FingerprintMismatchError("model pack manifest has no fingerprint")
        return fp

    @property
    def short_fingerprint(self) -> str:
        """Display form of the fingerprint (12 hex chars)."""
        return short_fingerprint(self.fingerprint)

    @property
    def name(self) -> str:
        """Served model name."""
        return self.manifest.name

    @property
    def arch(self) -> ArchConfig:
        """Architecture config."""
        return self.manifest.arch

    @property
    def num_layers(self) -> int:
        """Number of decoder layers."""
        return self.manifest.arch.num_layers

    def linear_specs(self) -> list[LinearSpec]:
        """All linear group specs in canonical order."""
        return self.manifest.linear_specs()

    def linear_entry(self, lid: LinearId | str) -> LinearEntry:
        """Manifest entry (shape, parts) of one linear group."""
        try:
            return self._entries[str(lid)]
        except KeyError as exc:
            raise ConfigError(f"model pack {self.name} has no linear {str(lid)!r}") from exc

    def max_in_features(self) -> int:
        """Largest ``n`` of any linear group (drives ring headroom)."""
        return max(e.in_features for e in self.manifest.linears)

    def check_ring(self, params: RingParams) -> None:
        """Raise :class:`~pgw.core.errors.RingOverflowError` if ``params`` lacks headroom for this pack."""
        params.check_in_features(self.max_in_features())

    def supported_rings(self, candidates: Iterable[RingParams] | None = None) -> list[str]:
        """Names of the ring presets (or ``candidates``) with enough headroom for every linear."""
        from pgw.core.ring import RING_K32_A8, RING_K40_A12, RING_K48_A20

        out = []
        n = self.max_in_features()
        for p in candidates or (RING_K32_A8, RING_K40_A12, RING_K48_A20):
            if n <= p.max_in_features():
                out.append(p.name)
        return out

    @property
    def tokenizer_dir(self) -> Path:
        """Directory with tokenizer files (``<pack>/tokenizer``)."""
        return self.root / TOKENIZER_DIR

    # ---- tensors ---------------------------------------------------------------------------
    def _handle(self, file: str) -> Any:
        with self._lock:
            h = self._handles.get(file)
            if h is None:
                from safetensors import safe_open

                path = self.root / file
                if not path.is_file():
                    raise ConfigError(f"model pack {self.root.name} is missing {file}", details={"file": file})
                h = safe_open(str(path), framework="pt", device="cpu")
                self._handles[file] = h
            return h

    def _tensor_entry(self, name: str) -> TensorEntry:
        try:
            return self.manifest.tensors[name]
        except KeyError as exc:
            raise ConfigError(f"model pack {self.name} has no tensor {name!r}") from exc

    def load_tensor(
        self, name: str, *, device: torch.device | str = "cpu", check_hash: bool = False
    ) -> torch.Tensor:
        """Load one tensor by pack name (memory-mapped on CPU; copied when ``device`` is not CPU).

        Args:
            check_hash: Re-hash the bytes and compare with the manifest.

        Raises:
            ShapeError: on-disk dtype/shape disagree with the manifest.
            FingerprintMismatchError: ``check_hash`` and the bytes differ from the manifest.
        """
        e = self._tensor_entry(name)
        try:
            t: torch.Tensor = self._handle(e.file).get_tensor(name)
        except Exception as exc:
            if isinstance(exc, ConfigError):
                raise
            raise ConfigError(f"cannot read tensor {name!r} from {e.file}: {exc}") from exc
        if t.dtype != torch_dtype(e.dtype) or tuple(t.shape) != tuple(e.shape):
            raise ShapeError(
                f"{name}: file has {t.dtype}{list(t.shape)}, manifest says {e.dtype}{list(e.shape)}",
                details={"tensor": name},
            )
        if check_hash:
            require_same_fingerprint(e.sha256, tensor_sha256(t), context=f"tensor {name}")
        dev = torch.device(device)
        return t if dev.type == "cpu" else t.to(dev, non_blocking=False)

    def load_linear(
        self, lid: LinearId | str, device: torch.device | str = "cpu", *, check_hash: bool = False
    ) -> LinearWeights:
        """Load one linear group's int8 weight, scale and bias.

        The (small) scale and bias vectors are always validated: scales must be finite and
        positive and biases finite (SPEC section 5). With ``check_hash`` the weight is also checked
        to contain no ``-128``.

        Raises:
            ConfigError: invalid scale/bias values or a ``-128`` weight (with ``check_hash``).
            ShapeError / FingerprintMismatchError: see :meth:`load_tensor`.
        """
        e = self.linear_entry(lid)
        w8 = self.load_tensor(e.weight, device=device, check_hash=check_hash)
        sw = self.load_tensor(e.scale, device=device, check_hash=check_hash)
        b = self.load_tensor(e.bias, device=device, check_hash=check_hash) if e.bias is not None else None
        _check_scale_bias(e.id, sw, b)
        if check_hash and bool((w8 == -128).any()):
            raise ConfigError(f"linear {e.id}: int8 weight contains -128 (packs use [-127, 127])")
        return LinearWeights(e.linear_id, w8, sw, b)

    def load_linear_meta(self, lid: LinearId | str, *, check_hash: bool = False) -> LinearMeta:
        """Scale, bias and shape of one linear group without reading its int8 weight.

        The masked gateway never needs ``W8``; this keeps it from reading (and holding) every weight
        matrix of the pack just to learn the scales. Scale and bias are validated as in
        :meth:`load_linear`.
        """
        e = self.linear_entry(lid)
        sw = self.load_tensor(e.scale, check_hash=check_hash)
        b = self.load_tensor(e.bias, check_hash=check_hash) if e.bias is not None else None
        _check_scale_bias(e.id, sw, b)
        if tuple(sw.shape) != (e.out_features,):
            raise ShapeError(f"linear {e.id}: scale has shape {tuple(sw.shape)}, expected ({e.out_features},)")
        return LinearMeta(e.linear_id, sw, b, int(e.out_features), int(e.in_features), e.row_norm_sq_max)

    def verify_row_norms(self) -> int:
        """Recompute ``row_norm_sq_max`` from the int8 weights and compare with the manifest.

        Reads every linear weight (seconds for a 7B pack); for install-time checks (``pgw doctor``),
        not for startup. Packs without the field are skipped. Returns the number of linears checked.

        Raises:
            FingerprintMismatchError: a stored value differs from the weights.
        """
        checked = 0
        for e in self.manifest.linears:
            if e.row_norm_sq_max is None:
                continue
            got = row_norm_sq_max(self.load_tensor(e.weight))
            if got != e.row_norm_sq_max:
                raise FingerprintMismatchError(
                    f"linear {e.id}: manifest row_norm_sq_max {e.row_norm_sq_max} != {got} computed from the weight",
                    details={"linear": e.id},
                )
            checked += 1
        return checked

    def iter_linears(
        self,
        device: torch.device | str = "cpu",
        *,
        layers: Iterable[int] | None = None,
        kinds: Iterable[LinearKind | str] | None = None,
    ) -> Iterator[LinearWeights]:
        """Yield every linear group in canonical order (streams; does not keep them resident).

        Args:
            layers: Restrict to these layer indices.
            kinds: Restrict to these linear kinds.
        """
        lay = set(layers) if layers is not None else None
        kin = {LinearKind(k) for k in kinds} if kinds is not None else None
        for e in self.manifest.linears:
            lid = e.linear_id
            if (lay is None or lid.layer in lay) and (kin is None or lid.kind in kin):
                yield self.load_linear(lid, device)

    def load_layer(self, layer: int, device: torch.device | str = "cpu") -> dict[LinearKind, LinearWeights]:
        """All four linear groups of one layer."""
        if not 0 <= layer < self.num_layers:
            raise ConfigError(f"layer {layer} out of range [0, {self.num_layers})")
        return {lw.id.kind: lw for lw in self.iter_linears(device, layers=[layer])}

    def iter_layers(
        self, device: torch.device | str = "cpu", *, layers: Iterable[int] | None = None
    ) -> Iterator[tuple[int, dict[LinearKind, LinearWeights]]]:
        """Stream ``(layer, {kind: LinearWeights})`` one layer at a time (weight streaming)."""
        for i in layers if layers is not None else range(self.num_layers):
            yield i, self.load_layer(i, device)

    def load_plain(
        self, name: str, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Load a plain tensor (``embed_tokens``, ``final_norm``, ``lm_head`` (falls back to
        ``embed_tokens`` when tied), ``layers.<i>.input_norm``, ``layers.<i>.post_attn_norm``)."""
        if name == "lm_head" and self.arch.tie_word_embeddings:
            name = "embed_tokens"
        if name not in self.manifest.plain_tensor_names():
            raise ConfigError(f"unknown plain tensor {name!r}")
        return self.load_tensor(name, device=device).to(dtype)

    def load_plain_tensors(
        self, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """All plain tensors the gateway needs, keyed by pack name. ``lm_head`` is always present
        (it aliases ``embed_tokens`` when tied, without a second copy)."""
        out: dict[str, torch.Tensor] = {}
        for n in self.manifest.plain_tensor_names():
            out[n] = self.load_tensor(n, device=device).to(dtype)
        if "lm_head" not in out:
            out["lm_head"] = out["embed_tokens"]
        return out

    # ---- verification ------------------------------------------------------------------------
    def verify_files(self, components: Iterable[Component] | None = None, *, workers: int | None = None) -> None:
        """Hash every tensor/tokenizer file and compare with the manifest (``verify="full"``).

        Also checks that each safetensors file holds exactly the tensors the manifest assigns to
        it with matching dtype and shape.

        Args:
            components: Parts to verify (default: those this pack was opened with).
            workers: Hashing threads (default: min(4, CPUs)).

        Raises:
            FingerprintMismatchError: any tensor or tokenizer file hash differs.
            ConfigError: a file is missing or its layout differs from the manifest.
        """
        comps = _normalize_components(components) if components is not None else self.components
        by_file: dict[str, list[str]] = {}
        wanted: set[str] = set()
        weights: set[str] = set()
        if "linears" in comps:
            for e in self.manifest.linears:
                wanted.update(t for t in (e.weight, e.scale, e.bias) if t is not None)
                weights.add(e.weight)
        if "plain" in comps:
            wanted.update(self.manifest.plain_tensor_names())
        for name in sorted(wanted):
            by_file.setdefault(self.manifest.tensors[name].file, []).append(name)

        def verify_file(file: str) -> list[str]:
            entries, _ = read_safetensors_header(self.root / file)
            expected = {n for n, t in self.manifest.tensors.items() if t.file == file}
            if set(entries) != expected:
                extra = sorted(set(entries) - expected)[:3]
                miss = sorted(expected - set(entries))[:3]
                raise ConfigError(f"{file}: tensor set differs from manifest (extra {extra}, missing {miss})")
            for n in by_file[file]:
                me, he = self.manifest.tensors[n], entries[n]
                if he.dtype != me.dtype or he.shape != tuple(me.shape):
                    raise ConfigError(
                        f"{file}: {n} is {he.dtype}{list(he.shape)}, manifest says {me.dtype}{list(me.shape)}"
                    )
            digests = hash_tensor_ranges(
                self.root / file, [entries[n] for n in by_file[file]], forbid_int8_min=weights
            )
            return [n for n in by_file[file] if digests[n] != self.manifest.tensors[n].sha256]

        bad: list[str] = []
        nw = workers or min(4, os.cpu_count() or 1)
        files = sorted(by_file)
        if nw > 1 and len(files) > 1:
            with ThreadPoolExecutor(max_workers=nw, thread_name_prefix="pgw-pack-verify") as ex:
                for res in ex.map(verify_file, files):
                    bad += res
        else:
            for f in files:
                bad += verify_file(f)
        if "tokenizer" in comps:
            for t in self.manifest.tokenizer:
                p = self.tokenizer_dir / t.file
                if not p.is_file():
                    raise ConfigError(f"missing tokenizer file {t.file}", details={"file": t.file})
                if sha256_file(p) != t.sha256:
                    bad.append(f"{TOKENIZER_DIR}/{t.file}")
        if bad:
            raise FingerprintMismatchError(
                f"model pack {self.name}: {len(bad)} tensor(s)/file(s) do not match the manifest: {bad[:5]}",
                details={"mismatched": bad[:50]},
            )
        if "linears" in comps:  # value checks on the (now hash-verified) scale/bias vectors
            for e in self.manifest.linears:
                sw = self.load_tensor(e.scale)
                b = self.load_tensor(e.bias) if e.bias is not None else None
                _check_scale_bias(e.id, sw, b)
        log.info("verified %d file(s) of model pack %s (%s)", len(files), self.name, ",".join(comps))

    def size_bytes(self, part: Literal["linears", "plain", "all"] = "all") -> int:
        """On-disk size of the selected tensors."""
        if part not in ("linears", "plain", "all"):
            raise ConfigError(f"invalid part {part!r}")
        names: set[str] = set()
        if part in ("linears", "all"):
            for e in self.manifest.linears:
                names.update(t for t in (e.weight, e.scale, e.bias) if t is not None)
        if part in ("plain", "all"):
            names.update(self.manifest.plain_tensor_names())
        return sum(self.manifest.tensors[n].nbytes for n in names)

    # ---- distribution ------------------------------------------------------------------------
    def export(
        self, out_dir: str | Path, components: Iterable[Component], *, link: bool = True, overwrite: bool = False
    ) -> Path:
        """Write a subset pack (same manifest and fingerprint) containing only ``components``.

        Typical use: ship only the linear shards to the compute service, or only plain tensors and
        tokenizer to a gateway that never runs the ``plain`` backend. Files are hard-linked when
        possible (``link=True``), otherwise copied. Hard links share bytes with this pack: never
        edit files of either copy in place. The output is built in a temporary sibling directory
        and renamed into place atomically, like :func:`~pgw.modelpack.pack_checkpoint`.

        Returns:
            The output directory.

        Raises:
            ConfigError: unknown/unavailable components, ``out_dir`` exists without ``overwrite``,
                or ``out_dir`` overlaps this pack's directory.
        """
        from pgw.modelpack.packer import _AtomicDir

        comps = _normalize_components(components)
        missing = [c for c in comps if c not in self.components]
        if missing:
            raise ConfigError(f"cannot export components {missing}: this pack was opened without them")
        out = Path(out_dir)
        ro, oo = self.root.resolve(), out.resolve()
        if ro == oo or ro.is_relative_to(oo) or oo.is_relative_to(ro):
            raise ConfigError(f"export target {out} overlaps the source pack directory {self.root}")
        files = [MANIFEST_FILE, *self._files_for(comps)]

        def build(tmp: Path) -> None:
            for f in files:
                src, dst = self.root / f, tmp / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                if link:
                    try:
                        os.link(src, dst)
                        continue
                    except OSError:
                        pass
                shutil.copyfile(src, dst)

        ad = _AtomicDir(out, overwrite)
        try:
            build(ad.tmp)
            ad.commit()
        except BaseException:
            ad.abort()
            raise
        log.info("exported %s components %s to %s", self.name, ",".join(comps), out.name)
        return out

    def content_digest(self) -> str:
        """SHA-256 over the manifest bytes on disk (detects edits to non-semantic fields too)."""
        return hashlib.sha256((self.root / MANIFEST_FILE).read_bytes()).hexdigest()
