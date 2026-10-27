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

from vt.chat_serve import (GlassboxResponder, PackageError, export_package, load_package,  # noqa: E402
                           make_chat_server, reply)
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
    srv = make_chat_server(GlassboxResponder(model, tok, man), port=0)
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


def test_hf_backend_on_a_retrofitted_checkpoint(tmp_path):
    """Pin and serve the kind of directory kaggle/verifiable_chat/train.py writes."""
    transformers = pytest.importorskip("transformers")
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from vt.chat_serve import HFResponder, export_hf_package, load_responder
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(["system: hi user: hello assistant: I don't know."] * 30,
                           trainers.BpeTrainer(vocab_size=280, special_tokens=["<eos>"],
                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", pad_token="<eos>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}assistant:{% endif %}")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    d = tmp_path / "model"
    m.save_pretrained(d); tok.save_pretrained(d)
    torch.save({"lin.weight": torch.zeros(1, 32), "lin.bias": torch.zeros(1)}, d / "k_head.pt")
    man = export_hf_package(str(d))
    r = load_responder(str(d))
    assert isinstance(r, HFResponder) and r.manifest["package_sha256"] == man["package_sha256"]
    out = r.reply("hello", max_new_tokens=4)
    assert all(t["confidence"] == 0.5 and 0 <= t["token_probability"] <= 1 for t in out["tokens"])
    with open(d / "config.json", "a") as f:
        f.write(" ")
    with pytest.raises(PackageError):
        load_responder(str(d))


def test_glassbox3_package_reports_certificates_uncertainty_and_unsure(package, tmp_path):
    from vt.chat_serve import export_glassbox2_package, load_responder
    from vt.glassbox3 import G3Config, Glassbox3
    tok_path = os.path.join(package, "tokenizer.json")
    V = Tokenizer.from_file(tok_path).get_vocab_size()
    cfg = dict(vocab_size=V + 1, block_size=64, n_layer=2, n_head=2, n_embd=32, n_features=64, k_active=4,
               mem_layer=1, mem_keys=4, mem_topk=2, unsure_id=V)
    torch.manual_seed(0)
    m = Glassbox3(G3Config(**cfg))
    with torch.no_grad():
        m.c_unsure.fill_(50.0)                 # make <unsure> win wherever u > 0
    export_glassbox2_package(m, cfg, tok_path, str(tmp_path / "g3"))
    r = load_responder(str(tmp_path / "g3")).reply("hello", max_new_tokens=6)
    assert r["tokens"] and r["unsure"] and r["abstained"]
    assert r["tokens"][0]["token"] == "<unsure>"
    for t in r["tokens"]:
        assert 0 <= t["confidence"] <= 1 and t["u"] >= 0 and len(t["features"]) == 3
        assert t["memory_free"] in (True, False)
    assert r["certified_memory_free"] == all(t["memory_free"] for t in r["tokens"])


def test_correctness_backend_withholds_below_threshold(tmp_path):
    """A pinned HF model plus a correctness head: reply carries P(correct) and evidence, and is
    withheld below the package's threshold; a tampered head is refused."""
    pytest.importorskip("transformers")
    import numpy as np
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from vt.chat_serve import CorrectnessResponder, export_correctness_package, load_responder
    from vt.correctness import CorrectnessHead, Probe
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2))
    d = tmp_path / "model"
    m.save_pretrained(d); tok.save_pretrained(d)
    rng = np.random.default_rng(0)
    probe = Probe().fit(rng.normal(size=(60, 32)).astype(np.float32), rng.integers(0, 2, 60).astype(np.float32))
    stack = Probe().fit(rng.normal(size=(60, 5)).astype(np.float32), rng.integers(0, 2, 60).astype(np.float32))
    head = CorrectnessHead([1], probe, stack, use_consistency=False, use_grounding=False)
    torch.save({"probe+output": head.state_dict()}, tmp_path / "heads.pt")
    man = export_correctness_package(str(d), str(tmp_path / "heads.pt"), variant="probe+output", threshold=1.0)
    r = load_responder(str(d))
    assert isinstance(r, CorrectnessResponder) and r.manifest["package_sha256"] == man["package_sha256"]
    out = r.reply("Who wrote Hamlet?")
    assert 0 <= out["p_correct"] <= 1 and out["answer_withheld"] and out["text"].startswith("I'm not sure")
    assert "probe_logit" in out["evidence"] and "entity_p" in out["evidence"]
    assert not r.reply("Who wrote Hamlet?", threshold=0.0)["answer_withheld"]
    r.head.answer = lambda *a_: {"reply": "That country does not exist.", "p_correct": 0.1, "evidence": {}}
    ref = r.reply("Who ruled Zanthoria?")                    # a refusal is passed through, not judged
    assert ref["abstained"] and not ref["answer_withheld"] and ref["p_correct"] is None
    with open(d / "correctness_heads.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(PackageError):
        load_responder(str(d))
