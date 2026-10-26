"""GPT track: ternary layers, margin loss and the exact MLP-block verifier."""
import importlib.util
import os

import numpy as np
import torch

from vt.lm import GPT, GPTConfig

_spec = importlib.util.spec_from_file_location(
    "verify_lm_blocks", os.path.join(os.path.dirname(__file__), "..", "experiments", "verify_lm_blocks.py"))
vlb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vlb)


def tiny(**kw):
    return GPT(GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=16, **kw))


def test_forward_backward_and_ternary_values():
    torch.manual_seed(0)
    m = tiny(ternary=True, margin_weight=1.0)
    x = torch.randint(0, 256, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 256)
    (loss + m.margin_loss()).backward()
    q = m.blocks[0].mlp.fc.qweight().detach()
    assert len(torch.unique(q)) <= 3
    assert 0.0 <= m.stable_fraction() <= 1.0


def test_margin_loss_increases_stability():
    torch.manual_seed(0)
    m = tiny(margin_eps=0.05)
    x = torch.randint(0, 256, (8, 16))
    m(x)
    before = m.stable_fraction()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(50):
        m(x)
        loss = m.margin_loss()
        opt.zero_grad(); loss.backward(); opt.step()
    m(x)
    assert m.stable_fraction() > before + 0.1


def test_mlp_worst_case_is_sound_and_exact():
    rng = np.random.default_rng(0)
    W1, b1, W2 = rng.standard_normal((12, 6)), rng.standard_normal(12) * 0.1, rng.standard_normal((6, 12))
    x0, eps, c = rng.standard_normal(6), 0.1, rng.standard_normal(6)
    r = vlb.mlp_worst_case(W1, b1, W2, x0, eps, c, 30)
    assert r["status"] == 0
    f = lambda x: c @ (W2 @ np.maximum(W1 @ x + b1, 0))
    samples = [f(x0 + eps * rng.choice([-1, 1], 6)) for _ in range(3000)]
    samples += [f(x0 + rng.uniform(-eps, eps, 6)) for _ in range(3000)]
    assert max(samples) <= r["max"] + 1e-6
    # every maximum of a piecewise-linear function on a box is attained at a
    # vertex of some linear region; corners sampled above usually find it
    assert r["max"] - max(samples) < 0.05 * abs(r["max"]) + 1e-6


def test_ddp_two_process_training_runs(tmp_path):
    """The multi-process (torchrun / DDP, gloo on CPU) training path works end to end."""
    import json
    import shutil
    import subprocess
    import sys
    if shutil.which("torchrun") is None:
        import pytest
        pytest.skip("torchrun not available")
    data = tmp_path / "d.txt"
    data.write_bytes(b"to be or not to be, that is the question. " * 400)
    root = os.path.join(os.path.dirname(__file__), "..")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run(["torchrun", "--standalone", "--nproc_per_node", "2",
                    os.path.join(root, "experiments", "train_lm.py"), "--data", str(data),
                    "--out", str(tmp_path / "run"), "--preset", "tiny", "--block-size", "32",
                    "--batch-size", "4", "--grad-accum", "2", "--steps", "6", "--eval-every", "3",
                    "--eval-iters", "1", "--ternary", "--margin-weight", "0.5"],
                   check=True, env=env, cwd=root, capture_output=True, timeout=600)
    m = json.load(open(tmp_path / "run" / "metrics.json"))
    assert m["final"]["step"] == 6 and m["final"]["loss"] < 6.0
