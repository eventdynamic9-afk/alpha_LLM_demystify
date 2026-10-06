"""Frozen LLM claim parser (§9.2 option E2): a family not among the narrators, temperature 0, JSON
schema, Claimify-style instruction to extract only unambiguous claims and mark the rest AMBIGUOUS."""
from __future__ import annotations

import json
import re

from configs import fill, prompt

from .normalize import normalize_claim
from .rules import fill_slots
from .schema import SCHEMA_TEXT, validate_output

PARSER_VERSION = "parser-v1"


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


def llm_parse(client, rationale_id: str, text: str, temperature: float = 0.0, log=None, cache=None) -> dict:
    """Returns {"claims": [...], "errors": [...], "raw": str}; claims are normalized and slot-filled."""
    msgs = build_parser_messages(rationale_id, text)
    r = client.complete(msgs, temperature=temperature, max_tokens=2000, seed=0)
    if log:
        log(role="parser", model=client.model_id, request=msgs, response=r.text, parser_version=PARSER_VERSION)
    obj = _extract_json(r.text)
    if obj is None:
        return {"claims": [], "errors": ["unparsable JSON"], "raw": r.text}
    obj.setdefault("rationale_id", rationale_id)
    for i, c in enumerate(obj.get("claims", [])):
        c.setdefault("claim_id", f"{rationale_id}-c{i}")
        c.setdefault("span", [0, 0])
        c.setdefault("text", "")
        c.setdefault("args", {})
        c.setdefault("ambiguous", False)
        c.setdefault("hedge", "absolute")
        c.setdefault("polarity", "affirm")
        c.setdefault("type", "C1")
    errors = validate_output(obj)
    claims = []
    for c in obj.get("claims", []):
        c = normalize_claim(fill_slots(c, text))
        c["parser"] = f"{PARSER_VERSION}:{client.model_id}"
        claims.append(c)
    return {"claims": claims, "errors": errors, "raw": r.text}
