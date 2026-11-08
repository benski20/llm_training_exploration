"""vt/omni.py without a model: thresholds, the guarantee floor, training, profiles, adaptation, calibration."""
import numpy as np

from vt.omni import (OmniVerifier, cp_threshold, feature_dicts, make_item, min_clean_for_guarantee, sentence_spans,
                     subset, train_artifact)


def synthetic(n=400, d=16, L=3, domains=("a", "b", "c", "d"), seed=0):
    """stored-feature arrays where a direction in the hidden states (plus a per-domain offset) marks wrong replies."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    dom = np.array([domains[i % len(domains)] for i in range(n)])
    offs = {g: rng.normal(0, 3, d) for g in domains}                         # covariate shift between domains
    direction = rng.normal(0, 1, d)
    H = rng.normal(0, 1, (n, L, d)) + y[:, None, None] * direction + np.stack([offs[g] for g in dom])[:, None]
    nu = 3
    F = {"ids": np.arange(n), "r_mean": H.astype(np.float16), "r_last": H.astype(np.float16),
         "j_state": H.astype(np.float16), "r_sc": rng.normal(0, 1, (n, 10)).astype(np.float32),
         "j_logit": (rng.normal(0, 1, n) - 1.5 * y).astype(np.float32),      # the judge says "No" more to wrong
         "u_mean": np.repeat(H, nu, 0).astype(np.float16), "u_sc": rng.normal(0, 1, (n * nu, 10)).astype(np.float32),
         "u_row": np.repeat(np.arange(n), nu), "u_idx": np.tile(np.arange(nu), n),
         "u_lab": np.repeat(y, nu), "meta": {"reader": "none", "layers": [1, 2, 3], "max_tokens": 64}}
    rows = {i: {"domain": dom[i], "label": int(y[i])} for i in range(n)}
    return F, rows, dom, y


def auc(s, y):
    from vt.correctness import auroc
    return auroc(np.asarray(s), np.asarray(y))


def test_guarantee_floor_and_threshold():
    assert min_clean_for_guarantee(0.10, 0.95) == 29
    rng = np.random.default_rng(1)
    fails = 0
    for _ in range(200):                                     # the bound may fail in at most ~5% of calibration draws
        t = cp_threshold(rng.normal(size=300), 0.10, 0.95)
        fails += (rng.normal(size=20000) >= t).mean() > 0.10
    assert fails <= 0.08 * 200
    clean = rng.normal(size=10)
    assert cp_threshold(clean, 0.10, 0.95) > clean.max()    # too few to promise anything: flag nothing seen


def test_units_and_items():
    text = "First claim. Second claim!\nStep 3: x = 2.5 so y = 5."
    spans = sentence_spans(text)
    assert [text[a:b].strip() for a, b in spans] == ["First claim.", "Second claim!", "Step 3: x = 2.5 so y = 5."]
    it = make_item("A reply.", "Q?", documents=["doc one"])
    assert it["messages"][0]["role"] == "system" and it["messages"][-1]["content"].startswith("doc one")


def test_train_profile_adapt_calibrate(tmp_path):
    F, rows, dom, y = synthetic()
    held = "d"
    art = train_artifact(subset(F, dom != held), rows, layer_idx=[0, 1, 2], pca=8)
    v = OmniVerifier(art)
    fd = feature_dicts(subset(F, dom == held))
    yh = y[dom == held]
    s_uni = v.score(v.universal_profile(), fd)
    prof = v.profile(fd, "d")
    s_prof = v.score(prof, fd)
    assert auc(s_prof, yh) > 0.8 and auc(s_prof, yh) >= auc(s_uni, yh) - 0.02
    prof = v.adapt(prof, fd[:40], yh[:40])
    prof = v.calibrate(prof, fd[:60], yh[:60], fpr=0.10, confidence=0.95)
    assert prof["threshold"] is not None and prof["card"]["min_clean_needed"] == 29
    OmniVerifier.save(art, tmp_path / "h.npz")
    v2 = OmniVerifier.load(tmp_path / "h.npz")
    assert np.allclose(v2.score(prof, fd), v.score(prof, fd), atol=1e-2)       # bases stored in float16
    OmniVerifier.save_profile(prof, tmp_path / "p.json")
    assert OmniVerifier.load_profile(tmp_path / "p.json")["name"] == "d"


def test_flags_and_extras_enter_the_score():
    F, rows, dom, y = synthetic()
    art = train_artifact(subset(F, dom != "d"), rows, layer_idx=[0, 1, 2], pca=8)
    v = OmniVerifier(art)
    fd = feature_dicts(subset(F, dom == "d"))
    prof = v.profile(fd, "d")
    base = v.score(prof, fd)
    fd[0]["flags"] = 2
    assert v.score(prof, fd)[0] == base[0] + 2 * v.FLAG_WEIGHT
    for k, f in enumerate(fd):                                  # an external judge that knows the label
        f["judges"] = [float(y[dom == "d"][k]) * 2 - 1]
    prof2 = v.profile(fd, "d")
    assert auc(v.score(prof2, fd), y[dom == "d"]) > auc(v.score(prof, fd), y[dom == "d"])
