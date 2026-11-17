"""Cross-language vectors (``core_v1.json``) shared by ``pgw.core`` and ``rust/pgw-core-rs``.

* The JSON must be exactly what ``rust/tools/gen_vectors.py`` produces from the current Python core,
  so any behaviour change in ``pgw.core`` that would break the Rust implementation (or any other
  independent implementation) fails here first.
* If the optional Rust extension ``pgw_core_rs`` is importable, it is checked against the vectors.
* ``PGW_TEST_RUST=1`` additionally runs ``cargo test`` (needs cargo and crates.io access).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import torch

from pgw.core import RingParams, SeedExpander, freivalds_check, unmask_outputs

PGW_ROOT = Path(__file__).resolve().parents[2]
VECTOR_FILE = Path(__file__).with_name("core_v1.json")
GEN_SCRIPT = PGW_ROOT / "rust" / "tools" / "gen_vectors.py"
RUST_DIR = PGW_ROOT / "rust"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pgw_gen_vectors", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vectors() -> dict:
    return json.loads(VECTOR_FILE.read_text(encoding="utf-8"))


def _labels(raw: list[dict]) -> list[str | int | bytes]:
    out: list[str | int | bytes] = []
    for lab in raw:
        if "s" in lab:
            out.append(lab["s"])
        elif "i" in lab:
            out.append(int(lab["i"]))
        else:
            out.append(bytes.fromhex(lab["b"]))
    return out


def test_vector_file_matches_python_core() -> None:
    gen = _load_generator()
    assert VECTOR_FILE.read_text(encoding="utf-8") == gen.render(gen.build()), (
        "tests/vectors/core_v1.json is stale: pgw.core changed behaviour (protocol break?). "
        "If intended, run `python rust/tools/gen_vectors.py` and update SPEC/INTERFACE_NOTES."
    )


def test_vectors_contain_frozen_known_answers(vectors: dict) -> None:
    # The same KAT as tests/core/test_prng.py, reached through the vector file.
    kat = bytes(range(32)).hex()
    derived = {
        tuple(map(str, _labels(c["labels"]))): c["derived"] for c in vectors["prng"]["derive_seed"] if c["seed"] == kat
    }
    assert derived[("segment", "1")] == "00bc5de3b96a347f37eb7113018cc757222c86488560b4a81e1d4df28c199af7"
    rings = {p["name"] for p in vectors["ring"]["params"]}
    assert {"K32A8", "K40A12", "K48A20"} <= rings
    assert {c["ring"] for c in vectors["linear"]} >= {"K32A8", "K40A12", "K48A20"}


def test_linear_vectors_are_self_consistent(vectors: dict) -> None:
    for c in vectors["linear"]:
        p = RingParams.from_name(c["ring"])
        rows, n, m = c["rows"], c["n"], c["m"]
        r = SeedExpander(bytes.fromhex(c["seed"])).ring_uniform_rows(
            n, p.k, c["mask_row_start"], rows, "r", c["linear"]
        )
        assert r.reshape(-1).tolist() == c["r"]
        y_hat = torch.tensor(c["y_hat"]).reshape(rows, m)
        wr = torch.tensor(c["wr"]).reshape(rows, m)
        assert unmask_outputs(y_hat, wr, p).reshape(-1).tolist() == c["yq"]
        x_hat = torch.tensor(c["x_hat"]).reshape(rows, n)
        u = torch.tensor(c["u"]).reshape(c["checks"], m)
        uw = torch.tensor(c["uw"]).reshape(c["checks"], n)
        assert bool(freivalds_check(x_hat, y_hat, u, uw, p).all())
        bad = torch.tensor(c["tampered_y_hat"]).reshape(rows, m)
        assert freivalds_check(x_hat, bad, u, uw, p).tolist() == c["tampered_verdicts"]
        assert not all(c["tampered_verdicts"])


def test_rust_extension_matches_vectors(vectors: dict) -> None:
    ext = pytest.importorskip("pgw_core_rs", reason="optional Rust extension not built (feature 'python')")
    for c in vectors["prng"]["streams"]:
        want = c["bytes"]
        got = ext.random_bytes(bytes.fromhex(c["seed"]), len(want) // 2, _labels(c["labels"]))
        assert got.hex() == want
    for c in vectors["prng"]["ring_uniform_rows"]:
        got = ext.ring_uniform_rows(
            bytes.fromhex(c["seed"]), c["cols"], c["k"], c["row_start"], c["row_count"], _labels(c["labels"])
        )
        assert got == c["values"]
    for c in vectors["linear"]:
        rows, n, m = c["rows"], c["n"], c["m"]
        assert ext.mask_activations(c["xq"], c["r"], c["ring"]) == c["x_hat"]
        assert ext.ringmm(c["x_hat"], rows, n, c["w8"], m, c["ring"]) == c["y_hat"]
        assert ext.unmask_outputs(c["y_hat"], c["wr"], c["ring"]) == c["yq"]
        verdicts = ext.freivalds_check(
            c["x_hat"], c["tampered_y_hat"], rows, n, m, c["u"], c["uw"], c["checks"], c["ring"]
        )
        assert verdicts == c["tampered_verdicts"]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("PGW_TEST_RUST") != "1", reason="set PGW_TEST_RUST=1 to run cargo test")
def test_cargo_test_passes() -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo not installed")
    proc = subprocess.run(  # noqa: S603 - fixed argv, cargo from PATH
        [cargo, "test", "--quiet"], cwd=RUST_DIR, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
