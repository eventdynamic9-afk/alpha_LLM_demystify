"""Claim parser output schema (Appendix B) and validation."""
from __future__ import annotations

import json

import jsonschema

PREDICATES = ["DEPENDS_ON", "SIGN", "MONO", "LOOKBACK", "HORIZON", "XSEC", "INVARIANT", "RANGE", "STRUCT",
              "RESEMBLES", "INDEPENDENT", "EXPOSED", "TURNOVER", "REGIME", "PRED_SIGN", "NOVEL", "BETTER_THAN",
              "PERF", "THEORY", "IDENTITY", "VARIANT_OF"]
TYPES = ["C1", "C2", "C3", "C4", "C5", "C6"]

CLAIM_SCHEMA = {
    "type": "object",
    "required": ["rationale_id", "claims"],
    "properties": {
        "rationale_id": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim_id", "span", "text", "type", "predicate", "args", "hedge", "polarity", "ambiguous"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "span": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "text": {"type": "string"},
                    "type": {"enum": TYPES},
                    "predicate": {"enum": PREDICATES},
                    "args": {"type": "object"},
                    "scope": {"type": ["string", "null"]},
                    "hedge": {"enum": ["absolute", "typical", "possible"]},
                    "polarity": {"enum": ["affirm", "deny"]},
                    "horizon": {"type": ["string", "null"]},
                    "ambiguous": {"type": "boolean"},
                },
            },
        },
    },
}

SCHEMA_TEXT = json.dumps({
    "rationale_id": "string",
    "claims": [{"claim_id": "string", "span": ["start_char", "end_char"], "text": "string",
                "type": "C1|C2|C3|C4|C5|C6", "predicate": "SIGN", "args": {"input": "ret_5d", "direction": "-"},
                "scope": "string or null", "hedge": "absolute|typical|possible", "polarity": "affirm|deny",
                "horizon": "string or null", "ambiguous": False}]}, indent=2)


def validate_output(obj: dict) -> list[str]:
    v = jsonschema.Draft7Validator(CLAIM_SCHEMA)
    return [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in v.iter_errors(obj)]
