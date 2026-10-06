"""Frozen LLM claim parser (§9.2 option E2): a family not among the narrators, temperature 0, JSON
schema, Claimify-style instruction to extract only unambiguous claims and mark the rest AMBIGUOUS.

Every parser call goes through the response cache keyed by (model, prompt hash, seed) and is logged
with the same fields as a narrator call (§4.3); spans are recomputed from the copied claim text
(§9.1 span slot) and missing hedge / polarity / scope slots are filled by the rule layer (E1) before
any static default applies.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from configs import fill, prompt, template_sha256
from pools.records import now_iso

from .normalize import normalize_claim
from .rules import fill_slots, sentences
from .schema import SCHEMA_TEXT, validate_output

PARSER_VERSION = "parser-v1"
MAX_TOKENS = 2000
SEED = 0


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def build_parser_messages(rationale_id: str, text: str) -> list[dict]:
    user = fill(prompt("parser_instruction"), schema=SCHEMA_TEXT, rationale_id=rationale_id, rationale_text=text)
    return [{"role": "user", "content": user}]


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _valid(span, n: int) -> bool:
    return (isinstance(span, (list, tuple)) and len(span) == 2 and all(isinstance(x, int) for x in span)
            and 0 <= span[0] < span[1] <= n)


def locate_span(text: str, quote: str, hint=None, min_coverage: float = 0.8) -> tuple[list[int], str]:
    """Character offsets of a claim's copied text in the rationale and how they were found.

    Sources: ``llm`` (parser offsets agree with the copied text), ``exact``, ``normalized`` (case- and
    whitespace-insensitive), ``fuzzy`` (the sentence covering >= ``min_coverage`` of the quote),
    ``llm_offsets`` (no quote; offsets kept unverified) or ``unlocated`` (span [0, 0])."""
    q = (quote or "").strip()
    h0 = hint[0] if _valid(hint, len(text)) else None
    if h0 is not None and q and _squash(text[hint[0]:hint[1]]) == _squash(q):
        return [hint[0], hint[1]], "llm"
    if not q:
        return ([hint[0], hint[1]], "llm_offsets") if h0 is not None else ([0, 0], "unlocated")
    near = (lambda i: abs(i - h0)) if h0 is not None else (lambda i: i)
    hits = [m.start() for m in re.finditer(re.escape(q), text)]
    if hits:
        s0 = min(hits, key=near)
        return [s0, s0 + len(q)], "exact"
    pat = r"\s+".join(re.escape(w) for w in q.split())
    hits2 = [(m.start(), m.end()) for m in re.finditer(pat, text, re.I)]
    if hits2:
        return list(min(hits2, key=lambda se: near(se[0]))), "normalized"
    best = None
    ql = q.lower()
    for s0, s1, s in sentences(text):
        sm = SequenceMatcher(None, s.lower(), ql, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 3]
        cov = sum(b.size for b in blocks) / len(ql)
        better = best is None or cov > best[0] or (cov == best[0] and near(s0) < near(best[1]))
        if blocks and cov >= min_coverage and better:
            best = (cov, s0 + blocks[0].a, s0 + blocks[-1].a + blocks[-1].size)
    if best is not None:
        return [best[1], best[2]], "fuzzy"
    return [0, 0], "unlocated"


def _call(client, msgs: list[dict], temperature: float, log, cache, rationale_id: str) -> tuple[str, bool]:
    """One parser call through the cache (never regenerated silently, §4.3); returns (text, cached)."""
    from narrate.logger import prompt_hash

    ph = prompt_hash(msgs)
    if cache is not None:
        hit = cache.get(client.model_id, ph, SEED, temperature)
        if hit is not None:
            return hit.get("text", ""), True
    r = client.complete(msgs, temperature=temperature, max_tokens=MAX_TOKENS, seed=SEED)
    resp = {"text": r.text, "tool_calls": r.tool_calls, "usage": r.usage, "latency_s": r.latency_s,
            "fingerprint": r.fingerprint, "model": r.model, "finish_reason": r.finish_reason,
            "reasoning": getattr(r, "reasoning", None)}
    if cache is not None:
        cache.put(client.model_id, ph, SEED, temperature, resp)
    if log:
        prov = (client.provenance() if hasattr(client, "provenance")
                else {"model_string": getattr(client, "model_string", None)})
        log(role="parser", model=client.model_id, model_string=getattr(client, "model_string", None), prompt_hash=ph,
            template_sha256={"parser_instruction": template_sha256(prompt("parser_instruction"))},
            request={"messages": msgs, "tools": None}, response=resp,
            params={"temperature": temperature, "max_tokens": MAX_TOKENS, "seed": SEED,
                    "reasoning_setting": getattr(client, "reasoning_setting", None), **(getattr(r, "params", None) or {})},
            provenance=prov, parser_version=PARSER_VERSION, rationale_id=rationale_id, timestamp=now_iso())
    return r.text, False


def llm_parse(client, rationale_id: str, text: str, temperature: float = 0.0, log=None, cache=None) -> dict:
    """Returns {"claims": [...], "errors": [...], "raw": str, "cached": bool}; claims are span-located,
    slot-filled (E1) and normalized."""
    msgs = build_parser_messages(rationale_id, text)
    raw, cached = _call(client, msgs, temperature, log, cache, rationale_id)
    obj = _extract_json(raw)
    if obj is None or not isinstance(obj, dict):
        return {"claims": [], "errors": ["unparsable JSON"], "raw": raw, "cached": cached}
    obj.setdefault("rationale_id", rationale_id)
    for i, c in enumerate(obj.get("claims", [])):
        c.setdefault("claim_id", f"{rationale_id}-c{i}")
        c.setdefault("text", "")
        c.setdefault("args", {})
        c.setdefault("ambiguous", False)
        c.setdefault("type", "C1")
        span, src = locate_span(text, c.get("text") or "", c.get("span"))
        c["span"], c["span_source"] = span, src
        if src != "unlocated" and text[span[0]:span[1]] != c["text"]:
            if c["text"]:
                c["quote"] = c["text"]                # the parser's copy, kept when it differs from the span
            c["text"] = text[span[0]:span[1]]
        fill_slots(c, text)                       # E1 hedge / polarity / scope / horizon before any default
    errors = validate_output(obj)
    claims = []
    for c in obj.get("claims", []):
        c = normalize_claim(c)
        c["parser"] = f"{PARSER_VERSION}:{client.model_id}"
        claims.append(c)
    return {"claims": claims, "errors": errors, "raw": raw, "cached": cached}
