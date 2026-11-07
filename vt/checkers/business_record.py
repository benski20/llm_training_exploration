"""Checker for replies that describe a structured business record (Yelp-style JSON: name, address, hours,
attributes, stars, reviews), written by hand from the record's schema: no training, no labels.

Invariants:
  numbers     every number the reply states (hours, address numbers, ratings, counts) is in the record, allowing
              24h <-> 12h clock conversion and rounding of ratings
  attributes  a claimed amenity (outdoor seating, Wi-Fi, reservations, delivery, take-out, parking, kids, groups,
              alcohol, TV, dogs, wheelchair access...) does not contradict the record's value (False / 'no' / 'none')
  days        "every day" / "seven days a week" only if the record lists hours for all seven days
"""
import ast
import json
import re

ATTRS = {   # phrase pattern -> attribute keys (any of them); a claim contradicts if every present key is falsy
    r"outdoor (?:seating|patio|dining)|patio": ["OutdoorSeating"],
    r"wi-?fi|wireless internet|internet access": ["WiFi"],
    r"reservations?": ["RestaurantsReservations"],
    r"deliver(?:y|s|ies)": ["RestaurantsDelivery"],
    r"take-?out|takeaway|to-go|to go": ["RestaurantsTakeOut"],
    r"wheelchair|accessible": ["WheelchairAccessible"],
    r"good for (?:kids|children)|kid-friendly|family-friendly|child-friendly": ["GoodForKids"],
    r"good for groups|large groups|groups": ["RestaurantsGoodForGroups"],
    r"\btvs?\b|televisions?": ["HasTV"],
    r"dogs?\b|pet-friendly": ["DogsAllowed"],
    r"full bar|serves? (?:alcohol|beer|wine|cocktails)|alcoholic": ["Alcohol"],
    r"credit cards?": ["BusinessAcceptsCreditCards"],
    r"bike parking": ["BikeParking"],
    r"table service|waiter service": ["RestaurantsTableService"],
    r"caters?|catering": ["Caters"],
    r"parking": ["BusinessParking"],
}
NEG = re.compile(r"\b(?:no|not|n't|without|lack|lacks|doesn't|does not|unavailable|neither|nor)\b", re.I)
NUM = re.compile(r"(?<![\w.\-])(\d{1,5}(?:\.\d+)?)(?![\w.]*\d)(?!-\w)")   # not part of a name (COVID-19, I-5)


def parse_record(doc):
    a, b = doc.find("{"), doc.rfind("}")                    # the record, without any prompt text around it
    if a < 0 or b <= a:
        return None
    doc = doc[a:b + 1]
    for f in (json.loads, ast.literal_eval):
        try:
            v = f(doc)
            if isinstance(v, dict):
                return v
        except (ValueError, SyntaxError):
            pass
    return None


def _falsy(v):
    if v is None:
        return None                                  # unknown: no claim can contradict it
    if isinstance(v, dict):
        vals = [x for x in v.values() if x is not None]
        return None if not vals else not any(vals)
    if isinstance(v, str):
        s = v.strip().strip("u'\"").lower()
        return s in ("no", "none", "false", "") if s not in ("", ) else None
    return not bool(v)


def _record_numbers(rec):
    txt = json.dumps(rec)
    nums = set()
    for m in re.findall(r"\d+(?:\.\d+)?", txt):
        x = float(m)
        nums.add(x)
    for h in re.findall(r"(\d{1,2}):(\d{1,2})", txt):         # 24h hours -> also 12h
        hh, mm = int(h[0]), int(h[1])
        nums.update({hh, hh % 12 or 12, mm})
    return nums


def check(reply, doc):
    rec = parse_record(doc)
    if rec is None:
        return []
    out = []
    nums = _record_numbers(rec)
    attrs = rec.get("attributes") or {}
    hours = rec.get("hours") or {}
    for s in re.split(r"(?<=[.!?])\s+|\n+", reply):
        if not s.strip():
            continue
        for m in NUM.finditer(s):
            x = float(m.group(1))
            if x in nums or round(x) in nums or any(abs(x - n) <= 0.05 for n in nums if isinstance(n, float)):
                continue
            ctx = s[max(0, m.start() - 12):m.end() + 12].lower()
            if re.search(r"\b(?:am|pm|a\.m|p\.m)\b", ctx) and x <= 12:
                continue                                  # clock time already covered by 12h/24h; a miss is a miss
            out.append({"claim": s.strip()[:160], "why": f"number {m.group(1)} is not in the record"})
        neg = bool(NEG.search(s))
        for pat, keys in ATTRS.items():
            if not re.search(pat, s, re.I):
                continue
            vals = [attrs[k] for k in keys if k in attrs]
            if not vals:
                continue
            f = [_falsy(v) for v in vals]
            if neg or any(x is None for x in f):
                continue
            if all(f):
                out.append({"claim": s.strip()[:160], "why": f"record says {keys[0]} is {vals[0]!r}"})
        if re.search(r"open (?:seven days|7 days|every day|daily)|(?:seven|7) days a week", s, re.I) and hours \
                and len(hours) < 7:
            out.append({"claim": s.strip()[:160], "why": f"record lists hours for {len(hours)} days"})
    return out
