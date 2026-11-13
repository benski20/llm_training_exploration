"""A false-alarm promise a customer can set and audit, for any detector with one or more scores.

Calibration takes items the customer knows are fine (human-written texts for the AI-text detector; replies known
to be supported by their documents for the verifier) and sets one threshold per score so that, flagging an item
when any score passes its threshold, at most `fpr` of new items like these are flagged, with `confidence`. Each
threshold is the one-sided Clopper-Pearson threshold at fpr/k and confidence 1 - (1 - confidence)/k over the
k scores (the union bound). No examples of the bad case are needed: the promise is about the good items.

The audit checks the promise later on new known-good items: the share flagged, its exact 95% interval and the
one-sided binomial p-value against the promised rate.
"""
import math
import time

import numpy as np


def min_items(fpr, confidence, k=1):
    """The fewest known-good items for which a threshold can promise fpr at this confidence (union over k scores)."""
    return math.ceil(math.log((1 - confidence) / k) / math.log(1 - fpr / k))


def guaranteed_threshold(score, fpr, confidence):
    """The lowest threshold whose false-alarm rate over items like `score` is at most fpr with this confidence."""
    from scipy.stats import beta
    neg = np.sort(np.asarray(score, float))
    n = len(neg)
    for j in range(n):
        above = n - j - 1                                         # items scoring above neg[j]
        if beta.ppf(confidence, above + 1, n - above) <= fpr:
            return float(neg[j])
    return float(neg[-1]) if n else float("inf")


def promise_from_scores(scores, fpr=0.05, confidence=0.95, good="human texts", flagged_as="AI", extra=None):
    """scores: {name: array over the known-good items}. Returns the calibration: thresholds and the promise."""
    names = list(scores)
    n, k = len(scores[names[0]]), len(names)
    need = min_items(fpr, confidence, k)
    if n < need:
        raise ValueError(f"{n} {good}: at least {need} are needed to promise {fpr:.0%} false alarms with "
                         f"{confidence:.0%} confidence (more items, more caught at the same promise)")
    thr = {name: guaranteed_threshold(scores[name], fpr / k, 1 - (1 - confidence) / k) for name in names}
    how = f"union of {k} one-sided Clopper-Pearson bounds" if k > 1 else "one-sided Clopper-Pearson bound"
    cal = {"fpr": fpr, "confidence": confidence, "items": n, "thresholds": thr,
           "promise": f"on new {good} like these, at most {fpr:.0%} are flagged as {flagged_as}, with "
                      f"{confidence:.0%} confidence ({how} over {n} {good})",
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cal.update(extra or {})
    return cal


def fired(calibration, scores):
    """Which scores pass their calibrated threshold, for one item: scores is {name: value}."""
    return [name for name, t in calibration["thresholds"].items() if name in scores and scores[name] > t]


def audit_from_flags(flagged, fpr, confidence=0.95, good="human texts"):
    """Did the promise hold on new known-good items? `flagged`: one bool per item from the calibrated detector.
    "exceeded" when the one-sided binomial p-value against the promised rate is below 1 - confidence."""
    from scipy.stats import beta, binom
    flagged = np.asarray(flagged, bool)
    n, k = len(flagged), int(flagged.sum())
    if not n:
        raise ValueError(f"no {good} to audit")
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    p = float(binom.sf(k - 1, n, fpr))
    return {"items": n, "flagged": k, "flagged_share": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "promised": fpr, "p_value": round(p, 4), "status": "exceeded" if p < 1 - confidence else "consistent",
            "flagged_indices": np.flatnonzero(flagged).tolist(),
            "note": f"consistent means these {good} give no evidence that more than the promised share is "
                    "flagged; with few items the interval is wide"}
