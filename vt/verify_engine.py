"""The verification engine behind /v1/verify: one or more scorers read a reply against its documents, and a
per-customer false-alarm promise (vt/promise.py) turns their scores into a verdict.

Scorers (higher score = more likely to hold something the documents do not support):
    lettuce     LettuceDetect-large (KRLabsOrg, ModernBERT fine-tuned on RAGTruth token labels, MIT): every token
                of the reply gets a probability of being unsupported; the reply's score is the highest. The tokens
                above the threshold become character spans, so the customer sees which words are unsupported.
                Long documents are read in 4,096-token chunks, a token's probability its highest over chunks.
    internals   the frozen-reader grounding check (vt/grounding.py): a probe on Qwen3-4B's neurons plus how much
                the documents raise the words written. Trained without human labels; per-sentence verdicts.

With two scorers the engine reads one combined score: each scorer's value on a fixed scale (LettuceDetect's logit
divided by 2.92, the internals total by 0.94: their spreads over clean HaluBench answers), summed. On 300 HaluBench
answers outside LettuceDetect's training domain this ranks better than LettuceDetect alone (AUROC 0.746 against
0.682; difference +0.063, 95% CI [+0.025, +0.104]) and level with the internals (0.724), while flagging when either
scorer passes a threshold at half the budget wastes it (results/verify_engine_choice.json). The scales are fixed,
not fitted on the customer's replies, so the calibration's guarantee is exact.

Without a calibration each scorer uses its own default cut (LettuceDetect's published 0.5; the grounding check's
own threshold) and a reply is flagged when any passes. With one, the reply is flagged when its score (or the
combined score) passes the calibrated threshold, and the promise says at most that share of good replies like the
calibration set are flagged, with that confidence.
"""
import json
import os
import threading
import time

import numpy as np

from vt import promise

LETTUCE = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
LETTUCE_REV = "22296c700ef0ba4ab3e5c9afffa0185caaf61e52"       # the commit measured (also the container)
GOOD, FLAGGED = "replies known to be supported by their documents", "unsupported"


def spans_from_probs(answer, offsets, probs, threshold):
    """Character spans of the answer whose tokens score above the threshold (adjacent tokens merged)."""
    out = []
    for (s, e), p in zip(offsets, probs):
        if p <= threshold or s == e:
            continue
        if out and s <= out[-1]["end"] + 1:
            out[-1]["end"] = e
            out[-1]["confidence"] = max(out[-1]["confidence"], p)
        else:
            out.append({"start": int(s), "end": int(e), "confidence": p})
    for sp in out:
        sp["text"] = answer[sp["start"]:sp["end"]]
        sp["confidence"] = round(float(sp["confidence"]), 4)
    return out


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


class LettuceScorer:
    name, default_threshold, spread = "lettuce", 0.5, 2.92

    def z(self, value):
        return _logit(value) / self.spread

    def __init__(self, model=LETTUCE, revision=LETTUCE_REV, threads=None, detector=None):
        if detector is None:
            import torch
            from lettucedetect.models.inference import HallucinationDetector
            if threads:
                torch.set_num_threads(threads)
            kw = {"revision": revision} if revision else {}
            detector = HallucinationDetector(method="transformer", model_path=model, **kw)
        self.det = detector
        self.tok = getattr(getattr(detector, "detector", detector), "tokenizer", None)

    def score(self, item):
        toks = self.det.predict(context=item["documents"] or [""], question=item["question"] or None,
                                answer=item["reply"], output_format="tokens")
        probs = [float(t["prob"]) for t in toks]
        offsets = []
        if self.tok is not None:
            offsets = self.tok(item["reply"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            if len(offsets) != len(probs):                     # truncated answer: keep what aligns
                offsets = offsets[:len(probs)]
        return max(probs, default=0.0), {"probs": probs, "offsets": offsets}

    def detail(self, item, extra, threshold):
        return {"spans": spans_from_probs(item["reply"], extra["offsets"], extra["probs"], threshold)}

    def tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"]) if self.tok is not None else len(text) // 4


class InternalsScorer:
    name, spread = "internals", 0.94

    def z(self, value):
        return value / self.spread

    def __init__(self, grounding):
        self.g = grounding
        self.default_threshold = grounding.calibration[0] if grounding.calibration is not None else 2.0

    def score(self, item):
        res = self.g.check([{"role": "user", "content": item["question"]}], item["reply"],
                           documents=item["documents"])
        return float(res["total"]), res

    def detail(self, item, extra, threshold):
        return {"sentences": extra.get("sentences", [])}

    def tokens(self, text):
        return len(self.g.tok(text, add_special_tokens=False).input_ids)


class Engine:
    """Scores a reply with every scorer, under one lock (the models are not shared across requests)."""

    def __init__(self, scorers):
        self.scorers = list(scorers)
        self.lock = threading.Lock()

    @property
    def names(self):
        return [s.name for s in self.scorers]

    @property
    def keys(self):
        """The scores a calibration thresholds: the one scorer's, or the combined score."""
        return self.names if len(self.scorers) == 1 else ["combined"]

    def scores(self, item):
        with self.lock:
            raw = {s.name: s.score(item) for s in self.scorers}
        values = {k: v for k, (v, _) in raw.items()}
        if len(self.scorers) > 1:
            values["combined"] = sum(s.z(values[s.name]) for s in self.scorers)
        return values, raw

    def verify(self, item, calibration=None):
        t = time.time()
        values, raw = self.scores(item)
        if calibration:
            fired = promise.fired(calibration, values)
        else:
            fired = [s.name for s in self.scorers if values[s.name] > s.default_threshold]
        out = {"verdict": FLAGGED if fired else "supported", "fired": fired,
               "scores": {k: round(v, 4) for k, v in values.items()}}
        for s in self.scorers:
            thr = (calibration or {}).get("thresholds", {}).get(s.name, s.default_threshold)
            out.update(s.detail(item, raw[s.name][1], thr))
        if calibration:
            out["promise"] = calibration["promise"]
        else:
            out["note"] = ("default thresholds: send replies known to be supported to /v1/verify/calibrate for a "
                           "false-alarm promise on your own traffic")
        s0 = self.scorers[0]
        out["usage"] = {"tokens": s0.tokens(item["reply"]) + sum(s0.tokens(d) for d in item["documents"])}
        out["ms"] = round(1000 * (time.time() - t), 1)
        return out

    def calibrate(self, items, fpr=0.05, confidence=0.95):
        per = [self.scores(it)[0] for it in items]
        s = {k: np.array([p[k] for p in per]) for k in self.keys}
        return promise.promise_from_scores(s, fpr, confidence, good=GOOD, flagged_as=FLAGGED,
                                           extra={"scorers": self.names})

    def audit(self, items, calibration):
        flags = [bool(promise.fired(calibration, self.scores(it)[0])) for it in items]
        return promise.audit_from_flags(flags, calibration["fpr"], calibration["confidence"], good=GOOD)


class Service:
    """Per-API-key calibrations on disk (one JSON file per key), the engine shared."""

    calibration_field = "replies"

    def __init__(self, engine, calibration=None, calibration_dir=None, info=None):
        self.e, self.default, self.dir = engine, calibration, calibration_dir
        self.info = dict(info or {}, scorers=engine.names, per_key_calibration=bool(calibration_dir),
                         calibrated=calibration is not None)

    def calibration_for(self, key_name):
        if self.dir:
            path = os.path.join(self.dir, f"{key_name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        return self.default

    def _check_scorers(self, cal):
        if cal and cal.get("scorers", self.e.names) != self.e.names:
            raise ValueError(f"calibration was made with scorers {cal.get('scorers')}, the server runs "
                             f"{self.e.names}: recalibrate")

    def verify(self, item, key_name):
        cal = self.calibration_for(key_name)
        self._check_scorers(cal)
        return self.e.verify(item, cal)

    def audit(self, items, key_name):
        cal = self.calibration_for(key_name)
        if not cal:
            raise ValueError("not calibrated: nothing to audit")
        self._check_scorers(cal)
        return self.e.audit(items, cal)

    def calibrate(self, items, fpr, confidence, key_name):
        cal = self.e.calibrate(items, fpr, confidence)
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, f"{key_name}.json"), "w") as f:
                json.dump(cal, f, indent=1)
        else:
            self.default = cal
            self.info["calibrated"] = True
        return cal
