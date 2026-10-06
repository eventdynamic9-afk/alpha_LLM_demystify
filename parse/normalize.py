"""Normalization rules (§9.5).

* vague terms map through codebook bins ("short-term" = <= 21 trading days, "medium" = 22-126,
  "long" > 126); a term without a bin -> AMBIGUOUS
* claims about unobservable mechanisms -> C5 (UNVERIFIABLE)
* claims about inputs outside the formula's universe (e.g. "uses analyst revisions") are C1.1
  DEPENDS_ON claims (REFUTED by the verifier)
* reference names, codebook terms and library ids are canonicalized so predicates can be compared
"""
from __future__ import annotations

import re

from configs import codebook

TYPE_OF = {p: v["type"] for p, v in codebook()["predicates"].items()}


def horizon_bin_of_days(d: int | None) -> str | None:
    if d is None:
        return None
    b = codebook()["horizon_bins"]
    if d <= b["very_short"][1]:
        return "very_short"
    if d <= b["short"][1]:
        return "short"
    if d <= b["medium"][1]:
        return "medium"
    return "long"


def _canon_ref(ref: str) -> tuple[str | None, bool]:
    cb = codebook()
    if ref in cb["reference_signals"] or re.fullmatch(r"(alpha101|gtja191|alpha158)_\w+", ref or ""):
        return ref, False
    low = (ref or "").strip().lower()
    if low in cb["terms"]:
        return low, False
    for t in sorted(cb["terms"], key=len, reverse=True):
        if t in low:
            return t, False
    for name in cb["reference_signals"]:
        if name.lower() == low:
            return name, False
    return ref, True


def _sign(x) -> str:
    s = str(x).strip().lower()
    return "-" if s in ("-", "neg", "negative", "-1", "down", "lower") else "+"


def normalize_claim(c: dict) -> dict:
    """Canonicalize predicate arguments; set ``ambiguous`` when mapping would require guessing."""
    c = dict(c)
    p = str(c.get("predicate", "")).upper()
    c["predicate"] = p
    if p not in TYPE_OF:
        c["ambiguous"] = True
        c["type"] = c.get("type", "C1")
        return c
    c["type"] = TYPE_OF[p]
    a = dict(c.get("args") or {})
    amb = bool(c.get("ambiguous", False))
    if p in ("SIGN", "MONO"):
        a["direction"] = _sign(a.get("direction", "+"))
        a["input"] = str(a.get("input", "")).strip().lower().replace("$", "")
    elif p == "DEPENDS_ON":
        a["input"] = str(a.get("input", "")).strip().lower().replace("$", "")
    elif p == "LOOKBACK":
        try:
            a["window"] = int(round(float(a.get("window"))))
        except (TypeError, ValueError):
            amb = True
    elif p == "HORIZON":
        b = str(a.get("bin", "")).lower().replace("-", "_").replace(" ", "_").replace("_term", "")
        if b not in codebook()["horizon_bins"]:
            amb = True
        a["bin"] = b
    elif p in ("RESEMBLES", "EXPOSED", "INDEPENDENT", "BETTER_THAN"):
        ref, unknown = _canon_ref(str(a.get("ref") or a.get("factor") or ""))
        a["ref"] = ref
        a.pop("factor", None)
        if p in ("RESEMBLES", "EXPOSED"):
            a["sign"] = _sign(a.get("sign", "+"))
        amb = amb or unknown
    elif p == "IDENTITY":
        lid = str(a.get("library_id", "")).lower()
        m = re.fullmatch(r"(?:worldquant\s*)?alpha\s*#?\s*(\d+)", lid)
        if m:
            lid = f"alpha101_{int(m.group(1)):03d}"
        a["library_id"] = lid
    elif p == "PRED_SIGN":
        a["sign"] = _sign(a.get("sign", "+"))
        h = a.get("horizon")
        try:
            a["horizon"] = int(h) if h not in (None, "", "null") else None
        except (TypeError, ValueError):
            a["horizon"] = None
    elif p == "TURNOVER":
        a["level"] = "low" if str(a.get("level", "low")).lower().startswith("l") else "high"
    elif p == "XSEC":
        v = a.get("value", True)
        a["value"] = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
    c["args"] = a
    c["ambiguous"] = amb
    c.setdefault("hedge", "absolute")
    c.setdefault("polarity", "affirm")
    c.setdefault("scope", None)
    c.setdefault("horizon", None)
    return c


def predicate_key(c: dict) -> tuple:
    """Normalized predicate identity used by the parser-validation matching rule (§9.3)."""
    p = c["predicate"]
    a = c.get("args") or {}
    keys = {"DEPENDS_ON": ("input",), "SIGN": ("input", "direction"), "MONO": ("input", "direction"),
            "LOOKBACK": ("window",), "HORIZON": ("bin",), "XSEC": ("value",), "INVARIANT": ("transform", "input"),
            "RANGE": ("low", "high"), "STRUCT": ("pattern",), "RESEMBLES": ("ref", "sign"), "INDEPENDENT": ("ref",),
            "EXPOSED": ("ref", "sign"), "TURNOVER": ("level",), "REGIME": ("regime", "effect"),
            "PRED_SIGN": ("sign",), "NOVEL": (), "BETTER_THAN": ("ref",), "PERF": ("metric",), "THEORY": (),
            "IDENTITY": ("library_id",), "VARIANT_OF": ("template",)}.get(p, ())
    return (p,) + tuple(str(a.get(k)) for k in keys) + (c.get("polarity", "affirm"),)
