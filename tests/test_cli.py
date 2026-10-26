"""Artifact round-trip, certificate binding and the CLI."""
import json

import numpy as np

from test_global import hand_built_dyck
from vt.artifact import load_artifact, save_artifact
from vt.cli import main
from vt.verify import np_forward


def test_artifact_roundtrip(tmp_path):
    P = hand_built_dyck(8)
    save_artifact(P, tmp_path / "m.npz")
    Q = load_artifact(tmp_path / "m.npz")
    x = np.eye(2)[[0, 1, 0, 0, 1, 1, 0, 1]]
    assert np.array_equal(np_forward(P, x), np_forward(Q, x))


def test_verify_check_predict(tmp_path, capsys):
    art, cert = str(tmp_path / "m.npz"), str(tmp_path / "c.json")
    save_artifact(hand_built_dyck(12), art)
    assert main(["verify", art, "--certificate", cert]) == 0
    assert json.load(open(cert))["status"] == "proved"
    assert main(["check", art, cert, "--exhaustive"]) == 0
    assert main(["predict", art, "(()())()(())", "())(()()()()"]) == 0
    out = capsys.readouterr().out
    assert "(()())()(())\t1" in out and "())(()()()()\t0" in out


def test_check_rejects_tampered_or_wrong_model(tmp_path):
    art, cert = str(tmp_path / "m.npz"), str(tmp_path / "c.json")
    P = hand_built_dyck(12)
    save_artifact(P, art)
    main(["verify", art, "--certificate", cert])
    P["W_out"][1, -1] = 0.0          # break the model, keep the old certificate
    save_artifact(P, art)
    assert main(["check", art, cert]) == 1
    # a fresh certificate for the broken model must not claim a proof
    assert main(["verify", art, "--certificate", cert]) == 1
    assert json.load(open(cert))["status"] == "counterexample"
