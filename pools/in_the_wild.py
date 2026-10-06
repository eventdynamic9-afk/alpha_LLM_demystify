"""P4 in-the-wild importers (§7.1): audit what published frameworks emit, run with default settings.

* AlphaAgent (github.com/RndmVariableQ/AlphaAgent): hypothesis / description / expression triples.
* RD-Agent(Q) factor scenario outputs: factor name, description, formulation.
* Alpha-R1 semantic factor profiles keyed by Alpha101 id; AlphaLogics market-logic explanations.
* Generic JSONL with ``expression`` and ``rationale`` fields.

Expressions are parsed leniently (Qlib -> Alpha101 -> GTJA); unparsable or out-of-panel expressions are
kept in a rejection log with the reason, never silently dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

from dsl import ParseError, try_parse_any

from .library import get as lib_get
from .records import FormulaRecord

_KEYS = {
    "alphaagent": {"expr": ("factor_expression", "expression", "formulation"),
                   "rationale": ("factor_description", "description"), "hypothesis": ("hypothesis",)},
    "rdagent": {"expr": ("factor_formulation", "formulation", "expression"),
                "rationale": ("factor_description", "description"), "hypothesis": ("hypothesis",)},
    "generic": {"expr": ("expression", "formula", "dsl"), "rationale": ("rationale", "description", "explanation"),
                "hypothesis": ("hypothesis",)},
}


def _first(d: dict, keys) -> str | None:
    for k in keys:
        if d.get(k):
            return str(d[k])
    return None


def _iter_records(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("factors") or data.get("results") or list(data.values())
        yield from data
    except json.JSONDecodeError:
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)


def import_framework_outputs(path: str | Path, framework: str = "generic", prefix: str = "P4") -> tuple[list[FormulaRecord], list[dict]]:
    keys = _KEYS.get(framework, _KEYS["generic"])
    out, rejected = [], []
    for i, raw in enumerate(_iter_records(Path(path))):
        expr = _first(raw, keys["expr"])
        if not expr:
            rejected.append({"index": i, "reason": "no expression field"})
            continue
        try:
            res = try_parse_any(expr, strict=False)
        except ParseError as exc:
            rejected.append({"index": i, "expression": expr, "reason": str(exc)})
            continue
        rec = FormulaRecord.from_node(f"{prefix}-{framework}-{i:04d}", "A", "P4", res.node, presented=expr,
                                      notation=res.dialect, provided_rationale=_first(raw, keys["rationale"]),
                                      hypothesis=_first(raw, keys["hypothesis"]),
                                      meta={"framework": framework, "parse_repairs": res.repairs})
        out.append(rec)
    return out, rejected


def import_alpha_r1_profiles(path: str | Path) -> tuple[list[FormulaRecord], list[dict]]:
    """Alpha-R1 profiles: {"alpha_id": "alpha101_012" | "Alpha#12", "profile": "..."}."""
    out, rejected = [], []
    for i, raw in enumerate(_iter_records(Path(path))):
        aid = str(raw.get("alpha_id", ""))
        if aid.lower().startswith("alpha#"):
            aid = f"alpha101_{int(aid[6:]):03d}"
        lf = lib_get(aid)
        if lf is None:
            rejected.append({"index": i, "alpha_id": aid, "reason": "not in the OHLCV library"})
            continue
        out.append(FormulaRecord.from_node(f"P4-alphar1-{i:04d}", "A", "P4", lf.node, presented=lf.source_text,
                                           notation="alpha101", provided_rationale=raw.get("profile"),
                                           base_id=lf.short_id, meta={"framework": "alpha-r1"}))
    return out, rejected
