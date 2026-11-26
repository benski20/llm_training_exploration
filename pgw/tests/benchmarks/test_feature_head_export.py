"""The draft head exporter (``benchmarks/feature_head_train.py``) and the gateway's loader agree on real-scale inputs.

The 7B export bug: the head is trained on normalized inputs (``[f / fs ; e / es]``) and the exporter folded those
input scales into ``W_in`` before int8 per-row quantization. The target's final-norm features have outlier
dimensions (per-dimension std spread over two to three orders of magnitude at 7B), so the folded columns of one
row span that range, and one int8 scale per row rounds the outlier columns (the dimensions that carry most of
the feature) to 0 or +-1. With one scale over both halves (the first 7B export) the whole feature half rounded to
0 and the head drafted from the token alone (alpha 0.19 whatever its size). Format v2 stores ``W_in`` in the
normalized space plus the input scales; the loader folds them for the float kernel and scales the input for the
native int8 kernel.

Here a head with 7B-like feature statistics is exported and loaded with ``FeatureHeadDrafter.from_file``; the
loaded head's proposals must match the float head's, on the float kernel and on the native int8 kernel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"

H, D, V, VFULL, N = 512, 64, 384, 1000, 2048


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pgw_test_feature_head_train", BENCH / "feature_head_train.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fht = _load()


def _case() -> dict:
    g = torch.Generator().manual_seed(7)
    # 7B-like final-norm feature statistics: most dimensions with std ~1-5, a few outliers with std in the hundreds
    fs = torch.exp(torch.randn(H, generator=g) * 0.5) * 2.0
    fs[torch.randperm(H, generator=g)[:8]] *= 150.0
    feats = torch.randn(N, H, generator=g) * fs
    embed = torch.randn(VFULL, H, generator=g) * 0.015
    toks = torch.randint(0, VFULL, (N,), generator=g)
    vocab = list(range(0, VFULL, VFULL // V))[:V]
    hv = torch.randn(V, H, generator=g) * 0.02
    torch.manual_seed(11)
    head = fht.Head(H, D, fs, float(embed.pow(2).mean().sqrt()))
    with torch.no_grad():  # a trained head's weights are O(1) in the normalized space, not tiny
        for lin in (head.w_in, head.w1, head.w2, head.w_out):
            lin.weight.mul_(4.0)
        _, ref = head.step(feats, embed[toks], hv)
    return {"head": head, "feats": feats, "embed": embed, "toks": toks, "vocab": vocab, "hv": hv, "ref": ref}


@pytest.fixture(scope="module")
def case() -> dict:
    return _case()


def _agree(lv: torch.Tensor, ref: torch.Tensor) -> float:
    return float((lv.argmax(-1) == ref.argmax(-1)).float().mean())


def _drafter(case: dict, blob: dict, path: Path, kernel: str = "float", ring: object = None) -> object:
    from pgw.engine.generate import FeatureHeadDrafter

    torch.save(blob, path)
    return FeatureHeadDrafter.from_file(path, case["embed"], kernel=kernel, ring=ring)


def _old_blob(case: dict, one_scale: bool) -> dict:
    """The earlier exports: input scales folded into ``W_in``, then one int8 scale per row over both halves
    (``one_scale``, the first 7B export) or per half (the first fix)."""
    head, hv = case["head"], case["hv"]
    w_in = head.w_in_eff()
    ws = {"w1": head.w1.weight, "w2": head.w2.weight, "w_out": head.w_out_eff()}
    ws.update({"w_in": w_in} if one_scale else {"w_in_f": w_in[:, :H], "w_in_e": w_in[:, H:]})
    q = {k: fht.q8(v) for k, v in ws.items()}
    q["u"] = fht.q8(hv @ (q["w_out"][0].float() * q["w_out"][1][:, None]))
    return {"format": "pgw-feature-head-v1", "target": "test", "hidden": H, "d": D, "vocab": case["vocab"],
            "int8": {k: {"w8": a, "scale": b} for k, (a, b) in q.items()},
            "bias": {"b_in": head.w_in.bias.detach().float(), "b1": head.w1.bias.detach().float(),
                     "b2": head.w2.bias.detach().float()}}


def test_export_v2_matches_the_float_head(case: dict, tmp_path: Path) -> None:
    blob = fht.export(case["head"], case["hv"], case["vocab"], "test", {}, keep_float=False)
    assert blob["format"] == "pgw-feature-head-v2" and "float" not in blob
    dr = _drafter(case, blob, tmp_path / "v2.pt")
    _, lv = dr._step(case["feats"], case["toks"].tolist())
    agree = _agree(lv, case["ref"])
    assert agree >= 0.95, agree
    # the benchmark's own dequantization (used for the alpha it reports) is the loader's float head
    w = fht.dequant_head(blob)
    assert torch.allclose(w["w_in"], dr.w_in, rtol=1e-5, atol=1e-7)


def test_the_old_exports_lose_the_feature(case: dict, tmp_path: Path) -> None:
    """The bug, reproduced: the first export drafts from the token alone, the folded per-half export still
    loses the outlier dimensions; v2 is several times closer to the float head than either."""
    _, ref_fp = case["head"].step(case["feats"], case["embed"][case["toks"]], case["hv"])
    errs = {}
    agree = {}
    for name, blob in (("one_scale", _old_blob(case, True)), ("folded", _old_blob(case, False)),
                       ("v2", fht.export(case["head"], case["hv"], case["vocab"], "test", {}, keep_float=False))):
        dr = _drafter(case, blob, tmp_path / f"{name}.pt")
        _, lv = dr._step(case["feats"], case["toks"].tolist())
        agree[name] = _agree(lv, case["ref"])
        errs[name] = float((lv - case["ref"]).norm() / case["ref"].norm())
    assert agree["one_scale"] < 0.5, agree
    assert agree["v2"] > agree["folded"] > agree["one_scale"], agree
    assert errs["v2"] * 3 < errs["folded"], errs


def test_native_int8_kernel_reads_v2(case: dict, tmp_path: Path) -> None:
    from pgw.core.ring import RingParams
    from pgw.engine.backends import NativeInt8DraftBackend

    if not NativeInt8DraftBackend.available():
        pytest.skip("pgw_fast (native local-linear kernel) not built")
    blob = fht.export(case["head"], case["hv"], case["vocab"], "test", {}, keep_float=False)
    dr = _drafter(case, blob, tmp_path / "v2n.pt", kernel="native_int8", ring=RingParams.from_name("K40A12"))
    assert dr.kernel == "native_int8"
    _, lv = dr._step(case["feats"], case["toks"].tolist())
    agree = _agree(lv, case["ref"])
    assert agree >= 0.93, agree


def test_v1_files_still_load(case: dict, tmp_path: Path) -> None:
    dr = _drafter(case, _old_blob(case, False), tmp_path / "v1.pt")
    fp, lv = dr._step(case["feats"][:4], case["toks"][:4].tolist())
    assert fp.shape == (4, H) and lv.shape == (4, V)
