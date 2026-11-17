"""Unit tests of the sizing calculator ``docs/tools/pgw_sizing.py`` against pgw.core / pgw.modelpack."""

from __future__ import annotations

import json
import subprocess
import sys
from types import ModuleType

import pytest

from tests.docs.conftest import DOCS

TOOL = DOCS / "tools" / "pgw_sizing.py"


@pytest.mark.parametrize("ring", ["K32A8", "K40A12", "K48A20"])
def test_ring_math_matches_core(ring: str, sizing: ModuleType) -> None:
    from pgw.core.freivalds import soundness_bits
    from pgw.core.ring import RingParams

    ours = sizing.RINGS[ring]
    core = RingParams.from_name(ring)
    assert ours.max_in_features() == core.max_in_features()
    assert ours.bytes_per_element == core.k // 8
    for n in (64, 896, 4096, 4864, 14336, 18944, 28672):
        for c in (1, 2, 4, 8):
            assert ours.soundness_bits(n, c) == soundness_bits(core, n, c), (ring, n, c)


@pytest.mark.parametrize("model", ["qwen2.5-0.5b", "qwen2.5-7b", "llama-3-8b", "llama-3-70b"])
def test_linear_shapes_match_modelpack(model: str, sizing: ModuleType) -> None:
    from pgw.core.types import LinearId, LinearKind
    from pgw.modelpack.arch import expected_linear_shape
    from pgw.modelpack.manifest import ArchConfig

    s = sizing.MODELS[model]
    arch = ArchConfig(
        arch="qwen2" if model.startswith("qwen") else "llama", hidden_size=s.hidden,
        intermediate_size=s.intermediate, num_layers=s.layers, num_heads=s.heads, num_kv_heads=s.kv_heads,
        head_dim=s.hd, vocab_size=32000, rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    for kind in LinearKind:
        m, n, _parts = expected_linear_shape(arch, LinearId(0, kind))
        assert s.linears()[kind.value] == (m, n), (model, kind)


def test_known_parameter_counts(sizing: ModuleType) -> None:
    # Masked linears (no embeddings/LM head): published totals minus embedding tables.
    assert sizing.MODELS["llama-3-8b"].linear_params == 6_979_321_856
    assert abs(sizing.MODELS["llama-3-70b"].linear_params - 68.5e9) < 0.1e9
    assert sizing.MODELS["llama-3-8b"].max_n == 14336


def test_wire_and_mask_bytes_match_core_packing(sizing: ModuleType) -> None:
    """Bytes per element equal pgw.core.pack_ring's encoding size."""
    import numpy as np

    from pgw.core.packing import pack_ring
    from pgw.core.ring import RingParams

    for ring in ("K32A8", "K40A12", "K48A20"):
        p = RingParams.from_name(ring)
        buf = pack_ring(np.zeros((3, 7), dtype=np.int64), p)
        assert len(buf) == 21 * sizing.RINGS[ring].bytes_per_element


def test_size_labels_and_values(sizing: ModuleType) -> None:
    rep = sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K48A20"], tokens_per_day=1e6, rtt_ms=20)
    f = rep.figures
    assert f["rounds_per_token"].value == 128
    assert f["latency_floor_per_token"].kind == "estimate"
    assert f["latency_floor_per_token"].value == pytest.approx(2.56)
    assert f["factory_t4_s_per_slot"].kind == "derived" and f["factory_t4_s_per_slot"].value == 0.0066
    assert f["factory_t4_gpus_needed"].value == pytest.approx(1e6 * 0.0066 / 86400)
    assert f["wire_per_day"].value == pytest.approx(13.37, rel=1e-3)
    assert f["freivalds_bits_worst_linear"].value == 12
    assert {fig.kind for fig in f.values()} <= {"analytic", "derived", "estimate"}
    text = rep.format()
    assert "[estimate]" in text and "[analytic]" in text and "llama-3-8b" in text
    json.dumps(rep.to_dict())


def test_factory_rate_fallbacks(sizing: ModuleType) -> None:
    k40 = sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"]).figures["factory_t4_s_per_slot"]
    assert k40.kind == "estimate" and k40.value == pytest.approx(0.0066 * 5 / 6)
    k32 = sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K32A8"]).figures["factory_t4_s_per_slot"]
    assert k32.kind == "derived" and k32.value == 0.0043
    q = sizing.size(sizing.MODELS["qwen2.5-7b"], sizing.RINGS["K40A12"]).figures["factory_t4_s_per_slot"]
    assert q.kind == "estimate" and 0.003 < q.value < 0.006


def test_segment_overhead_is_small(sizing: ModuleType) -> None:
    f = sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"]).figures
    assert f["mask_storage_per_slot"].value / f["wire_down_per_token"].value - 1 < 0.001
    tiny = sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"], segment_slots=1).figures
    assert tiny["mask_storage_per_slot"].value > f["mask_storage_per_slot"].value


def test_invalid_inputs(sizing: ModuleType) -> None:
    with pytest.raises(ValueError):
        sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"], rtt_ms=-1)
    with pytest.raises(ValueError):
        sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"], segment_slots=0)
    with pytest.raises(ValueError, match="checks"):
        sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"], checks=sizing.MAX_CHECKS + 1)
    with pytest.raises(ValueError, match="finite"):
        sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"], tokens_per_day=float("inf"))
    huge = sizing.ModelShape("huge", 8192, 200_000, 2, 64, 8)
    with pytest.raises(ValueError, match="cannot decode"):
        sizing.size(huge, sizing.RINGS["K32A8"])


def test_cli_text_json_and_errors(tmp_path) -> None:
    r = subprocess.run([sys.executable, str(TOOL), "--model", "llama-3-8b", "--ring", "K48A20", "--json"],  # noqa: S603
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["ring"] == "K48A20" and d["figures"]["wire_total_per_token"]["value"] == pytest.approx(13.37, rel=1e-3)
    r = subprocess.run([sys.executable, str(TOOL), "--hidden", "896", "--intermediate", "4864", "--layers", "24",  # noqa: S603
                        "--heads", "14", "--kv-heads", "2"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "custom" in r.stdout
    r = subprocess.run([sys.executable, str(TOOL), "--hidden", "896"], capture_output=True, text=True, timeout=60)  # noqa: S603
    assert r.returncode == 2 and "--intermediate" in r.stderr
    bad = tmp_path / "m.json"
    bad.write_text("{}")
    r = subprocess.run([sys.executable, str(TOOL), "--model", "llama-3-8b", "--measurements", str(bad)],  # noqa: S603
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and r.stderr.startswith("error:")


def test_tool_is_stdlib_only() -> None:
    """Operators run it without installing pgw: it must import only the standard library."""
    r = subprocess.run([sys.executable, "-I", "-S", str(TOOL), "--model", "qwen2.5-0.5b"],  # noqa: S603
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    src = TOOL.read_text(encoding="utf-8")
    for mod in ("numpy", "torch", "pgw", "pydantic"):
        assert f"import {mod}" not in src and f"from {mod}" not in src


def test_documented_invocations_run(docs: dict[str, str]) -> None:
    import re
    import shlex

    cmds = set(re.findall(r"python docs/tools/pgw_sizing\.py[^`\n]*", "\n".join(docs.values())))
    assert len(cmds) >= 2
    for c in cmds:
        argv = shlex.split(c)[2:]
        r = subprocess.run([sys.executable, str(TOOL), *argv], capture_output=True, text=True, timeout=60)  # noqa: S603
        assert r.returncode == 0, (c, r.stderr)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((64, 128, 2, 0, 1), "heads must be a positive integer"),
        ((-64, 128, 2, 4, 1), "hidden must be a positive integer"),
        ((64, 128, 0, 4, 1), "layers must be a positive integer"),
        ((64, 128, 2, 4, 3), "multiple of kv_heads"),
        ((66, 128, 2, 4, 1), "not divisible by heads"),
    ],
)
def test_invalid_shapes_raise_value_error(args: tuple[int, ...], match: str, sizing: ModuleType) -> None:
    with pytest.raises(ValueError, match=match):
        sizing.ModelShape("bad", *args)


def test_max_checks_matches_factory(sizing: ModuleType) -> None:
    from pgw.factory.config import FactorySettings

    meta = FactorySettings.model_fields["checks"].metadata
    assert any(getattr(m, "le", None) == sizing.MAX_CHECKS for m in meta)


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (["--hidden", "64", "--intermediate", "128", "--layers", "2", "--heads", "0", "--kv-heads", "1"], "heads"),
        (["--model", "llama-3-8b", "--layers", "3"], "cannot be combined"),
        (["--model", "llama-3-8b", "--checks", "50"], "checks"),
        (["--model", "llama-3-8b", "--rtt-ms", "nan"], "finite"),
    ],
)
def test_cli_rejects_bad_input_without_traceback(argv: list[str], needle: str) -> None:
    r = subprocess.run([sys.executable, str(TOOL), *argv], capture_output=True, text=True, timeout=60)  # noqa: S603
    assert r.returncode == 2, r.stderr
    assert r.stderr.startswith("error:") and needle in r.stderr and "Traceback" not in r.stderr
