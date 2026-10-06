"""Normalization rules (§9.5).

* vague terms map through codebook bins ("short-term" = <= 21 trading days, "medium" = 22-126,
  "long" > 126); a term without a bin -> AMBIGUOUS
* claims about unobservable mechanisms -> C5 (UNVERIFIABLE)
* claims about inputs outside the formula's universe (e.g. "uses analyst revisions") are C1.1
  DEPENDS_ON claims (REFUTED by the verifier)
* reference names, codebook terms and library ids are canonicalized so predicates can be compared;
  only exact matches (after case / hyphen / whitespace / plural normalization) are accepted, and a
  direction, sign or level that is missing or unrecognized is never defaulted: the claim is AMBIGUOUS
  (§10.1 "cannot be mapped to a predicate without guessing")
"""
from __future__ import annotations

import functools
import re

from configs import codebook

TYPE_OF = {p: v["type"] for p, v in codebook()["predicates"].items()}

# direction / sign vocabulary (ASCII and the protocol's Unicode notation: U+2212 minus, en dash, arrows)
_NEG = {"-", "−", "–", "—", "↓", "neg", "negative", "negatively", "-1", "-1.0", "down", "lower",
        "decreasing", "decrease", "decreases", "falling", "inverse", "opposite"}
_POS = {"+", "↑", "pos", "positive", "positively", "1", "+1", "1.0", "+1.0", "up", "higher", "increasing",
        "increase", "increases", "rising", "same"}
_TURNOVER = {"low": "low", "lower": "low", "slow": "low", "slow-moving": "low", "slow moving": "low",
             "persistent": "low", "stable": "low", "high": "high", "higher": "high", "fast": "high",
             "fast-moving": "high", "fast moving": "high", "rapid": "high", "rapidly changing": "high"}
# generic head nouns / articles that never change which codebook term is meant
_FILLER_PRE = ("the ", "a ", "an ")
_FILLER_POST = (" effect", " effects", " factor", " factors", " anomaly", " anomalies", " signal", " signals",
                " premium", " tilt", " exposure", " strategy", " style")


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


def norm_term(s: str) -> str:
    """Case-, hyphen- and whitespace-insensitive form of a codebook term ("Low-Volatility" -> "low volatility")."""
    s = re.sub(r"[‐‑‒–—-]", " ", str(s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%]", " ", s)).strip()


@functools.lru_cache(maxsize=1)
def _term_index() -> dict[str, str]:
    return {norm_term(t): t for t in codebook()["terms"]}


def _lookup_term(key: str) -> str | None:
    idx = _term_index()
    cands = [key]
    if key.endswith("ies"):
        cands.append(key[:-3] + "y")
    if key.endswith("s"):
        cands.append(key[:-1])
    for k in cands:
        if k in idx:
            return idx[k]
    return None


def _canon_ref(ref: str, allow_sets: bool = False) -> tuple[str | None, bool]:
    """Codebook term / reference name / library id -> (canonical ref, ambiguous).  Exact matches only
    (after :func:`norm_term`, simple plurals and generic filler words); anything else is AMBIGUOUS.
    ``allow_sets`` (BETTER_THAN) also accepts the codebook ``best_of_sets`` words (§10.4), exactly."""
    cb = codebook()
    raw = (ref or "").strip()
    if allow_sets:
        for name, words in (cb.get("best_of_sets") or {}).items():
            if raw.lower() == name or raw.lower() in [w.lower() for w in words]:
                return raw.lower(), False
    if raw in cb["reference_signals"] or re.fullmatch(r"(alpha101|gtja191|alpha158)_\w+", raw):
        return raw, False
    for name in cb["reference_signals"]:
        if name.lower() == raw.lower():
            return name, False
    key = norm_term(raw)
    if not key:
        return raw or None, True
    t = _lookup_term(key)
    if t is None:
        stripped = key
        for p in _FILLER_PRE:
            if stripped.startswith(p):
                stripped = stripped[len(p):]
        for p in _FILLER_POST:
            if stripped.endswith(p):
                stripped = stripped[: -len(p)]
        t = _lookup_term(stripped.strip()) if stripped != key else None
    return (t, False) if t is not None else (raw, True)


def _sign(x) -> str | None:
    """'+' / '-' for a recognized direction (ASCII, U+2212, en dash, arrows, words); None otherwise."""
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in _NEG:
        return "-"
    if s in _POS:
        return "+"
    return None


@functools.lru_cache(maxsize=1)
def _library_keys() -> dict[str, str]:
    from pools.library import library

    return {k.lower(): k for k in library()}


_BARE_LIBRARY = re.compile(r"(?:worldquant\s*)?(?:alpha\s*(?:101|158)|(?:gtja|guotai\s*junan)(?:[\s_-]*191)?)"
                           r"(?:\s*(?:alphas?|factors?|library))?")


def canonical_library_id(raw) -> tuple[str, bool]:
    """Library id -> (canonical key, ambiguous): ``alpha101_012``, ``gtja191_005``, ``alpha158_KMID``.

    The library prefix is case-insensitive; the Alpha158 feature name is resolved against the library's
    own (case-sensitive) keys.  A bare library name ("Alpha 101", "Alpha158") names no formula -> AMBIGUOUS.
    Ids from libraries outside the reference set are returned unchanged (the verifier reports them)."""
    s = re.sub(r"\s+", " ", str(raw or "")).strip()
    low = s.lower()
    if not low:
        return "", True
    keys = _library_keys()
    if low in keys:
        return keys[low], False
    if _BARE_LIBRARY.fullmatch(low):
        return s, True
    m = re.fullmatch(r"(?:qlib\s*)?alpha\s*158[\s_#:-]*([a-z]+\d*)", low)
    if m:
        name = m.group(1)
        return keys.get(f"alpha158_{name}", f"alpha158_{name.upper()}"), False
    m = (re.fullmatch(r"(?:worldquant\s*)?alpha\s*101[\s_#:-]*(?:alpha\s*)?#?\s*(\d{1,3})", low)
         or re.fullmatch(r"(?:worldquant\s*)?alpha\s*#\s*(\d{1,3})", low)
         or re.fullmatch(r"worldquant\s*alpha\s*(\d{1,3})", low)
         or re.fullmatch(r"alpha\s*(\d{1,3})", low))
    if m:
        return f"alpha101_{int(m.group(1)):03d}", False
    m = re.fullmatch(r"(?:gtja|guotai\s*junan)(?:[\s_-]*191)?[\s_#:-]*(?:alpha\s*)?#?\s*(\d{1,3})", low)
    if m:
        return f"gtja191_{int(m.group(1)):03d}", False
    return s, False


def _number(x) -> float | None:
    try:
        return float(str(x).strip().replace("−", "-").replace("–", "-"))
    except (TypeError, ValueError):
        return None


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

    def signed(key: str) -> None:
        nonlocal amb
        s = _sign(a.get(key))
        if s is None:
            amb = True
            a[key] = None if a.get(key) is None else str(a.get(key)).strip()
        else:
            a[key] = s

    if p in ("SIGN", "MONO"):
        signed("direction")
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
        ref, unknown = _canon_ref(str(a.get("ref") or a.get("factor") or ""), allow_sets=p == "BETTER_THAN")
        a["ref"] = ref
        a.pop("factor", None)
        if p in ("RESEMBLES", "EXPOSED"):
            signed("sign")
        amb = amb or unknown
    elif p == "IDENTITY":
        lid, unknown = canonical_library_id(a.get("library_id"))
        a["library_id"] = lid
        amb = amb or unknown
    elif p == "PRED_SIGN":
        signed("sign")
        h = a.get("horizon")
        try:
            a["horizon"] = int(h) if h not in (None, "", "null") else None
        except (TypeError, ValueError):
            a["horizon"] = None
    elif p == "TURNOVER":
        lvl = _TURNOVER.get(str(a.get("level") or "").strip().lower())
        if lvl is None:
            amb = True
            a["level"] = a.get("level")
        else:
            a["level"] = lvl
    elif p == "RANGE":
        for k in ("low", "high"):
            v = _number(a.get(k))
            if v is None:
                amb = True
            else:
                a[k] = v
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


_KEYS = {"DEPENDS_ON": ("input",), "SIGN": ("input", "direction"), "MONO": ("input", "direction"),
         "LOOKBACK": ("window",), "HORIZON": ("bin",), "XSEC": ("value",), "INVARIANT": ("transform", "input"),
         "RANGE": ("low", "high"), "STRUCT": ("pattern",), "RESEMBLES": ("ref", "sign"), "INDEPENDENT": ("ref",),
         "EXPOSED": ("ref", "sign"), "TURNOVER": ("level",), "REGIME": ("regime", "effect"),
         "PRED_SIGN": ("sign",), "NOVEL": (), "BETTER_THAN": ("ref",), "PERF": ("metric",), "THEORY": (),
         "IDENTITY": ("library_id",), "VARIANT_OF": ("template",)}


def predicate_key(c: dict, with_polarity: bool = True) -> tuple:
    """Normalized predicate identity.  ``with_polarity=False`` is the §9.3 matching key (polarity is a
    slot, scored separately); the default keeps deny/affirm distinct for ensemble agreement, B5 and SP
    invariance."""
    p = c["predicate"]
    a = c.get("args") or {}
    key = (p,) + tuple(str(a.get(k)) for k in _KEYS.get(p, ()))
    return key + (c.get("polarity", "affirm"),) if with_polarity else key


def match_key(c: dict) -> tuple:
    """§9.3 matching rule key: normalized predicate without the polarity slot."""
    return predicate_key(c, with_polarity=False)
