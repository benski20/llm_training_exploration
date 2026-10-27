"""Glass-box chat service: hash-pinned packages, per-token reports, hardened endpoint."""
import json
import os
import threading
import urllib.error
import urllib.request

import pytest
import torch

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from vt.chat_serve import PackageError, export_package, load_package, make_chat_server, reply  # noqa: E402
from vt.glassbox import GBConfig, GlassBoxLM  # noqa: E402


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    d = tmp_path_factory.mktemp("chat")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["User: hello\nAssistant: hi there. I don't know."] * 50,
                            trainers.BpeTrainer(vocab_size=300, special_tokens=["<eos>"],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(d / "tokenizer.json"))
    cfg = dict(vocab_size=tok.get_vocab_size(), block_size=96, n_layer=2, n_head=2, n_embd=32, mem_layer=1,
               mem_keys=8, mem_topk=4, fixed_embed=False, mlp_ratio=1, track_evidence=False, sharpness=True)
    torch.manual_seed(0)
    torch.save({"model": GlassBoxLM(GBConfig(**cfg)).state_dict()}, d / "ckpt.pt")
    export_package(str(d / "ckpt.pt"), str(d / "tokenizer.json"), str(d / "pkg"), cfg)
    return str(d / "pkg")


def test_reply_reports_every_token(package):
    model, tok, man = load_package(package)
    r = reply(model, tok, "hello", max_new_tokens=8)
    assert r["tokens"] and len(r["tokens"]) <= 8
    for t in r["tokens"]:
        assert 0 <= t["confidence"] <= 1
        assert abs(t["from_prompt"] + t["from_output"] + t["from_memory"] - 1) < 1e-3
    assert r["min_confidence"] == min(t["confidence"] for t in r["tokens"])


def test_tampered_package_is_refused(package, tmp_path):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(package, bad)
    with open(bad / "config.json", "a") as f:
        f.write(" ")
    with pytest.raises(PackageError):
        load_package(str(bad))


def test_chat_endpoint(package):
    model, tok, man = load_package(package)
    srv = make_chat_server(model, tok, man, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    def post(body):
        req = urllib.request.Request(base + "/chat", data=json.dumps(body).encode(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
    try:
        assert json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())["package_sha256"] == \
            man["package_sha256"]
        code, r = post({"message": "hello", "max_new_tokens": 4})
        assert code == 200 and "tokens" in r and r["package_sha256"] == man["package_sha256"]
        assert post({"message": ""})[0] == 400
        assert post({"message": "hi", "max_new_tokens": 10_000})[0] == 400
        assert post({"message": "hi", "history": [["a"]]})[0] == 400
    finally:
        srv.shutdown()
