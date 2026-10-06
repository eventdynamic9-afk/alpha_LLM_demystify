"""Claim-extraction ensemble (§9.2 option E3): two LLM parsers from two families plus the rule layer.

Claims extracted by both LLM parsers (same normalized predicate, overlapping spans) are accepted; a
claim found by only one parser is accepted when the rule layer corroborates it, otherwise it is
queued for human adjudication.  The first-run configuration (E2 + rules for slots + human sample)
uses :func:`merge_parser_and_rules`.
"""
from __future__ import annotations

from .normalize import normalize_claim, predicate_key
from .rules import extract_claims


def spans_overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _match(c, pool) -> int | None:
    for i, d in enumerate(pool):
        if predicate_key(c) == predicate_key(d) and spans_overlap(c["span"], d["span"]):
            return i
    return None


def ensemble(claims_a: list[dict], claims_b: list[dict], text: str, rationale_id: str) -> dict:
    rules = [normalize_claim(c) for c in extract_claims(text, rationale_id)]
    accepted, adjudicate = [], []
    used_b = set()
    for c in claims_a:
        j = _match(c, claims_b)
        if j is not None:
            used_b.add(j)
            accepted.append({**c, "agreement": "both"})
        elif _match(c, rules) is not None:
            accepted.append({**c, "agreement": "a+rules"})
        else:
            adjudicate.append({**c, "agreement": "a_only"})
    for j, c in enumerate(claims_b):
        if j in used_b:
            continue
        if _match(c, rules) is not None:
            accepted.append({**c, "agreement": "b+rules"})
        else:
            adjudicate.append({**c, "agreement": "b_only"})
    for k, c in enumerate(accepted + adjudicate):
        c["claim_id"] = f"{rationale_id}-c{k}"
    return {"accepted": accepted, "adjudicate": adjudicate}


def merge_parser_and_rules(llm_claims: list[dict], text: str, rationale_id: str) -> list[dict]:
    """First-run E2 + E1: keep every LLM claim; add rule-layer identity/lookback claims the LLM missed
    (deterministic, high-precision C1.1 / C1.4 / C6 patterns)."""
    out = list(llm_claims)
    for r in (normalize_claim(c) for c in extract_claims(text, rationale_id)):
        if r["predicate"] in ("IDENTITY", "LOOKBACK", "DEPENDS_ON") and _match(r, out) is None:
            out.append({**r, "parser": "rules"})
    for k, c in enumerate(out):
        c["claim_id"] = f"{rationale_id}-c{k}"
    return out
