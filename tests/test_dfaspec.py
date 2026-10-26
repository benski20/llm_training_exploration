"""User-defined DFA specs: parsing, sampling, CLI end to end with embedded spec."""
import itertools
import json
import os

import numpy as np
import pytest

from vt.cli import main
from vt.dfaspec import DFASpec

SPECS_DIR = os.path.join(os.path.dirname(__file__), "..", "specs")


def test_parse_label_and_sampler():
    s = DFASpec.load(os.path.join(SPECS_DIR, "no_bb.json"))
    for seq in itertools.product(range(2), repeat=8):
        assert s.label(seq) == int("11" not in "".join(map(str, seq)))
    x = s.sample(3000, 10, np.random.default_rng(0))
    y = s.label_batch(x)
    assert all(s.label(r) == v for r, v in zip(x, y))
    assert 0.2 < y.mean() < 0.9
    init, step, label = s.dfa(10)
    for seq in x[:200]:
        q = init
        for v in seq:
            q = step(q, int(v))
        assert label(q) == s.label(seq)


def test_rejects_incomplete_dfa():
    with pytest.raises(ValueError):
        DFASpec({"alphabet": "ab", "start": "s", "accept": ["s"], "delta": {"s": {"a": "s"}}})


def test_prove_check_predict_with_spec_file(tmp_path):
    out = str(tmp_path / "r")
    spec = os.path.join(SPECS_DIR, "no_bb.json")
    assert main(["prove", "--spec-file", spec, "--out", out, "--length", "12",
                 "--also-lengths", "4,8", "--float", "--uniform", "--window", "2",
                 "--rounds", "10", "--threads", "1", "--workers", "1", "--n-train", "4000"]) == 0
    cert = json.load(open(f"{out}/certificate.json"))
    assert cert["spec"] == "dfa:no_bb" and cert["method"] == "exact-dp" and "spec_dfa" in cert
    # the DFA is embedded, but the operator must pin it: a certificate can't define "correct"
    assert main(["check", f"{out}/model.npz", f"{out}/certificate.json"]) == 1
    assert main(["check", f"{out}/model.npz", f"{out}/certificate.json", "--spec-file", spec,
                 "--exhaustive"]) == 0
    assert main(["predict", f"{out}/model.npz", "--spec-file", spec, "abababababab", "abbaaaaaaaaa"]) == 0
