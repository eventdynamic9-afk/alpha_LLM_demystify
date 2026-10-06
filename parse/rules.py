"""Rule-based claim extraction and slot filling (§9.2 option E1).

Lexicons for fields, operators, anomaly names and regimes come from the codebook; regular expressions
catch numbers, windows, identities and direction cues.  High precision for C1.1 (dependence), C1.4
(lookback/horizon) and C6 (identity); lower recall for paraphrased C2/C3 claims — which is why the
frozen LLM parser (E2) is primary and this layer fills slots (hedge, polarity, scope, horizon).
"""
from __future__ import annotations

import re

from configs import codebook

from .normalize import TYPE_OF, horizon_bin_of_days

_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_NEG = re.compile(r"\b(not|no|never|doesn't|does not|isn't|is not|cannot|can't|without|lacks?)\b", re.I)

_DIR_UP = r"(raises|increases|boosts|lifts|pushes up|higher|larger|greater|more positive|scores? higher)"
_DIR_DOWN = r"(lowers|decreases|reduces|depresses|pushes down|lower|smaller|more negative|scores? lower)"


def sentences(text: str) -> list[tuple[int, int, str]]:
    out = []
    for m in _SENT.finditer(text):
        s = m.group().strip()
        if s:
            start = m.start() + (len(m.group()) - len(m.group().lstrip()))
            out.append((start, start + len(s), s))
    return out


def _field_mentions(s: str) -> list[str]:
    cb = codebook()
    low = s.lower()
    found = []
    for f, words in cb["field_lexicon"].items():
        for w in sorted(words, key=len, reverse=True):
            if re.search(rf"\b{re.escape(w)}\b", low):
                found.append(f)
                break
    if "dollar volume" in low and "volume" in found and "amount" in found:
        found.remove("volume")
    return found


def slot_hedge(s: str) -> str:
    low = s.lower()
    cb = codebook()["hedge_lexicon"]
    if any(re.search(rf"\b{re.escape(w)}\b", low) for w in cb["possible"]):
        return "possible"
    if any(re.search(rf"\b{re.escape(w)}\b", low) for w in cb["typical"]):
        return "typical"
    return "absolute"


def slot_polarity(s: str, cue: str | None = None) -> str:
    if cue:
        i = s.lower().find(cue.lower())
        window = s[max(0, i - 30): i + len(cue)] if i >= 0 else s
        return "deny" if _NEG.search(window) else "affirm"
    return "deny" if _NEG.search(s) else "affirm"


def slot_scope(s: str) -> str | None:
    low = s.lower()
    for k in codebook().get("scopes", {}):
        if re.search(rf"\b(in|among|for|within)\s+(the\s+)?{re.escape(k)}", low):
            return k
    return None


def _days(s: str) -> int | None:
    m = re.search(r"(\d+)\s*[- ]?\s*(?:trading\s+)?days?", s.lower())
    if m:
        return int(m.group(1))
    words = {"next day": 1, "one day": 1, "tomorrow": 1, "next week": 5, "a week": 5, "next month": 21,
             "a month": 21, "one month": 21, "a quarter": 63, "a year": 252}
    for w, d in words.items():
        if w in s.lower():
            return d
    return None


def _term(s: str) -> tuple[str, str] | None:
    """Longest codebook term mentioned in the sentence -> (term, sign)."""
    low = s.lower()
    terms = codebook()["terms"]
    for t in sorted(terms, key=len, reverse=True):
        if re.search(rf"\b{re.escape(t)}\b", low):
            return t, "+"
    return None


def _claim(rid, k, start, end, text, predicate, args, s, cue=None, ambiguous=False, horizon=None):
    return {"claim_id": f"{rid}-c{k}", "span": [start, end], "text": text, "type": TYPE_OF[predicate],
            "predicate": predicate, "args": args, "scope": slot_scope(s), "hedge": slot_hedge(s),
            "polarity": slot_polarity(s, cue), "horizon": horizon, "ambiguous": ambiguous}


def extract_claims(text: str, rationale_id: str = "r") -> list[dict]:
    cb = codebook()
    claims: list[dict] = []
    k = 0

    def add(*a, **kw):
        nonlocal k
        claims.append(_claim(rationale_id, k, *a, **kw))
        k += 1

    for start, end, s in sentences(text):
        low = s.lower()
        # ---------------- C6 identity / provenance
        hit_identity = False
        for lib, pats in cb["identity_patterns"].items():
            for pat in pats:
                for m in re.finditer(pat, s, re.I):
                    tok = m.group(1)
                    lid = f"{lib}_{int(tok):03d}" if tok.isdigit() else f"{lib}_{tok.upper()}"
                    add(start, end, s, "IDENTITY", {"library_id": lid}, s, m.group(0))
                    hit_identity = True
        for name, spec in cb["variant_templates"].items():
            for w in spec.get("words", []):
                if re.search(rf"\b{re.escape(w)}\b", low) and re.search(r"\b(variant|version|similar|like|resembl|akin|form of|type of|based on)", low):
                    add(start, end, s, "VARIANT_OF", {"template": name}, s, w)
                    break
        if hit_identity:
            continue
        # ---------------- C5 theory
        for cue in cb["theory_cues"]:
            if cue in low:
                add(start, end, s, "THEORY", {"mechanism": cue}, s, cue)
                break
        # ---------------- out-of-panel inputs
        for key, words in cb["out_of_universe_inputs"].items():
            if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words) and re.search(
                    r"\b(use|uses|using|based on|driven by|depends? on|incorporat|combin|relies on|draws on)", low):
                add(start, end, s, "DEPENDS_ON", {"input": key}, s)
        # ---------------- C1 dependence
        if re.search(r"\b(driven by|based on|uses|using|depends? on|combines|built from|constructed from|inputs? (are|is)|relies on|computed from)\b", low):
            for f in _field_mentions(s):
                add(start, end, s, "DEPENDS_ON", {"input": "ret_1d" if f == "returns" else f}, s)
        # ---------------- C1 direction / monotonicity
        m = re.search(r"\b(a |an )?(higher|rising|larger|greater|increasing|lower|falling|smaller|decreasing) "
                      r"([a-z\- ]{2,40}?)( today)? (" + _DIR_UP + "|" + _DIR_DOWN + r") (the )?(factor|score|signal|value|ranking)", low)
        if m:
            inc_input = m.group(2) in ("higher", "rising", "larger", "greater", "increasing")
            up = bool(re.match(_DIR_UP, m.group(5)))
            direction = "+" if inc_input == up else "-"
            noun = m.group(3).strip()
            fields = _field_mentions(noun)
            inp = ("ret_5d" if "return" in noun else fields[0]) if (fields or "return" in noun) else noun
            if "recent return" in noun or "past return" in noun:
                inp = "ret_5d"
            add(start, end, s, "SIGN", {"input": inp, "direction": direction}, s, m.group(0))
        m = re.search(r"stocks (that|which|with) (rose|gained|outperformed|fell|declined|underperformed)[a-z ]*? (score|rank|get|receive)s? (higher|lower)", low)
        if m:
            rose = m.group(2) in ("rose", "gained", "outperformed")
            higher = m.group(4) == "higher"
            d = _days(s) or 5
            add(start, end, s, "SIGN", {"input": f"ret_{d}d", "direction": "+" if rose == higher else "-"}, s, m.group(0))
        m = re.search(r"\b(increasing|decreasing|monotonic(?:ally)? (?:increasing|decreasing)) in ([a-z\- ]{2,30})", low)
        if m:
            fields = _field_mentions(m.group(2))
            inp = "abn_vol" if "abnormal volume" in m.group(2) else (fields[0] if fields else m.group(2).strip())
            add(start, end, s, "MONO", {"input": inp, "direction": "+" if "increasing" in m.group(1) else "-"}, s, m.group(0))
        # ---------------- C1 lookback / horizon
        for m in re.finditer(r"(\d+)\s*-?\s*(?:trading\s+)?day (window|lookback|period|average|moving average|span|history|horizon)", low):
            if "predict" in low and m.group(2) == "horizon":
                continue
            add(start, end, s, "LOOKBACK", {"window": int(m.group(1))}, s, m.group(0))
        m = re.search(r"\bover the (past|last|previous) (\d+) (trading )?days\b", low)
        if m:
            add(start, end, s, "LOOKBACK", {"window": int(m.group(2))}, s, m.group(0))
        if re.search(r"\b(signal|factor|indicator|effect|reversal|momentum|horizon|measure)\b", low) and "predict" not in low:
            for bin_, words in cb["horizon_words"].items():
                if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words if "-" in w or " " in w):
                    add(start, end, s, "HORIZON", {"bin": bin_}, s)
                    break
        # ---------------- C1 cross-section / invariance / range / structure
        if re.search(r"\b(ranks? stocks against each other|cross-sectional(ly)?|relative to (other stocks|peers|the cross-section))\b", low):
            add(start, end, s, "XSEC", {"value": True}, s)
        elif re.search(r"\b(purely time-series|relative to its own history|time-series (signal|measure))\b", low):
            add(start, end, s, "XSEC", {"value": False}, s)
        if re.search(r"\b(scale-free|unit-?less|scale invariant|scale-invariant|dimensionless|independent of the price level)\b", low):
            add(start, end, s, "INVARIANT", {"transform": "scale", "input": "price"}, s)
        m = re.search(r"\b(?:bounded|ranges?|lies|between)\b[^0-9\-]*(-?\d+(?:\.\d+)?)\s*(?:and|to)\s*(-?\d+(?:\.\d+)?)", low)
        if m:
            add(start, end, s, "RANGE", {"low": float(m.group(1)), "high": float(m.group(2))}, s, m.group(0))
        if re.search(r"ratio of (a |the )?short[- a-z]* to (a |the )?long[- a-z]* (moving )?average", low):
            add(start, end, s, "STRUCT", {"pattern": "ratio(MA_s, MA_l)"}, s)
        if re.search(r"\bz-?score\b", low):
            add(start, end, s, "STRUCT", {"pattern": "zscore"}, s)
        if re.search(r"\b(rolling )?correlation between\b", low):
            add(start, end, s, "STRUCT", {"pattern": "corr"}, s)
        if re.search(r"\brate of change\b", low):
            add(start, end, s, "STRUCT", {"pattern": "roc"}, s)
        # ---------------- C2 behavioral
        t = _term(s)
        if re.search(r"\b(independent of|uncorrelated with|orthogonal to|unrelated to)\b", low) and t:
            add(start, end, s, "INDEPENDENT", {"ref": t[0]}, s)
            t = None
        if t and re.search(r"\b(tilt|exposure|loads? on|loading)\b", low):
            add(start, end, s, "EXPOSED", {"ref": t[0], "sign": "+"}, s)
        elif t and re.search(r"\b(captur|resembl|similar|proxy|reflect|akin|form of|version of|variant of|like|measures|is a)\w*", low):
            add(start, end, s, "RESEMBLES", {"ref": t[0], "sign": t[1]}, s, t[0])
        if re.search(r"\b(low turnover|slow-moving|slow moving|persistent|stable rankings)\b", low):
            add(start, end, s, "TURNOVER", {"level": "low"}, s)
        elif re.search(r"\b(high turnover|fast-moving|fast moving|rapidly changing)\b", low):
            add(start, end, s, "TURNOVER", {"level": "high"}, s)
        m = re.search(r"\b(works? best|stronger|more effective|performs? better|more pronounced|weaker|less effective|fails?|breaks down)\b[^.]*\b(in|during|when)\b([^.]*)", low)
        if m:
            reg_text = m.group(3)
            effect = "weaker" if m.group(1) in ("weaker", "less effective", "fail", "fails", "breaks down") else "stronger"
            for reg, spec in cb["regimes"].items():
                if any(w in reg_text for w in spec["words"]):
                    add(start, end, s, "REGIME", {"regime": reg, "effect": effect}, s)
                    break
        m = re.search(r"\b(high|higher|large|larger) (factor )?values? (predict|signal|forecast|indicate|are associated with) (higher|lower|positive|negative|stronger|weaker) (future |subsequent |next[- a-z]* )?returns", low)
        if m:
            sign = "+" if m.group(4) in ("higher", "positive", "stronger") else "-"
            d = _days(s[m.start():]) or None
            add(start, end, s, "PRED_SIGN", {"sign": sign, "horizon": d}, s, m.group(0),
                horizon=horizon_bin_of_days(d) if d else None)
        else:
            m = re.search(r"\b(positively|negatively) (related|correlated|associated) (to|with) (future|subsequent|next[- a-z]*) returns", low)
            if m:
                d = _days(s)
                add(start, end, s, "PRED_SIGN", {"sign": "+" if m.group(1) == "positively" else "-", "horizon": d}, s, m.group(0))
        # ---------------- C3 originality / C4 performance
        if re.search(r"\b(novel|unique|new signal|not (captured|explained|spanned) by|distinct from existing|not a known)\b", low):
            cue = re.search(r"\b(novel|unique|new signal|not (captured|explained|spanned) by|distinct from existing|not a known)\b", low).group(0)
            pol = "affirm" if cue.startswith("not ") else slot_polarity(s, cue)
            c = _claim(rationale_id, k, start, end, s, "NOVEL", {"library": "all"}, s, cue)
            c["polarity"] = pol
            claims.append(c)
            k += 1
        m = re.search(r"\b(improves on|outperforms|better than|superior to)\b", low)
        if m and t is None:
            tt = _term(s[m.end():])
            if tt:
                add(start, end, s, "BETTER_THAN", {"ref": tt[0], "metric": "IC"}, s, m.group(0))
        if re.search(r"\b(high ic|strong predictive power|statistically significant|significant alpha|robust (performance|predictive))\b", low):
            add(start, end, s, "PERF", {"metric": "IC", "level": "high"}, s)
        if re.search(r"\b(stable across (years|time|periods)|consistent over time)\b", low):
            add(start, end, s, "PERF", {"metric": "stability", "level": "stable"}, s)
        if re.search(r"\b(high sharpe|sharpe ratio)\b", low):
            add(start, end, s, "PERF", {"metric": "sharpe", "level": "high"}, s)
    return claims


def fill_slots(claim: dict, text: str) -> dict:
    """Fill missing hedge / polarity / scope / horizon slots from the claim's span (E1 slot layer)."""
    s0, s1 = claim.get("span", [0, 0])
    s = text[s0:s1] if 0 <= s0 < s1 <= len(text) else claim.get("text", "")
    claim.setdefault("hedge", slot_hedge(s))
    claim.setdefault("polarity", slot_polarity(s))
    if claim.get("scope") is None:
        claim["scope"] = slot_scope(s)
    if claim.get("horizon") is None and claim.get("predicate") == "PRED_SIGN":
        d = (claim.get("args") or {}).get("horizon")
        claim["horizon"] = horizon_bin_of_days(int(d)) if d else None
    return claim
