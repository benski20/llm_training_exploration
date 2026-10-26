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


def test_package_api():
    import vt
    assert callable(vt.train_until_proved) and vt.CegisConfig().L == 16
    P = hand_built_dyck(8)
    assert vt.verify_global(P, vt.DyckSpec())["status"] == "proved"


def test_prove_multi_length_end_to_end(tmp_path):
    out = str(tmp_path / "run")
    # force the MILP path for L=10 (2^10 > 512); shorter lengths are enumerated
    code = main(["prove", "--out", out, "--length", "10", "--also-lengths", "4,6", "--float",
                 "--shared", "--rounds", "8", "--threads", "1", "--workers", "1",
                 "--n-train", "4000", "--enumerate-up-to", "512"])
    assert code == 0
    for L, cert in [(4, "certificate_L4.json"), (6, "certificate_L6.json"), (10, "certificate.json")]:
        c = json.load(open(f"{out}/{cert}"))
        assert c["status"] == "proved" and c["length"] == L
        assert c["method"] == ("milp" if L == 10 else "enumeration")
        art = f"{out}/model.npz" if L == 10 else f"{out}/model_L{L}.npz"
        assert main(["check", art, f"{out}/{cert}", "--exhaustive", "--workers", "1"]) == 0
