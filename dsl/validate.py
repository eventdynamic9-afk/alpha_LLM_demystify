"""AST validator — layer 2 of look-ahead impossibility (§6.3).

Layer 1 is the grammar itself (there is no negative lag, no forward shift and no centered window
in the DSL).  This validator additionally rejects any tree in which a node could reference a time
index after t: negative or non-integer lags/windows, unknown operators or fields, wrong arity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ast import Node, walk
from .fields import FIELDS
from .operators import OPS


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)


def time_offsets(node: Node) -> tuple[int, int]:
    """(min_offset, max_offset) of time indices read relative to t (offset 0 = t, k = t-k).

    By construction ``min_offset`` is 0 for every valid tree; a negative value would be look-ahead.
    """
    if node.is_const:
        return (0, 0)
    if node.is_field:
        return (0, 0)
    s = OPS[node.op]
    lo, hi = 0, 0
    child = [time_offsets(c) for c in node.children]
    lo = min((c[0] for c in child), default=0)
    hi = max((c[1] for c in child), default=0)
    if s.kind == "ts":
        p = int(node.params[0])
        if node.op == "Ref":
            return lo + p, hi + p
        if node.op == "Delta":
            return lo, hi + p
        return lo, hi + p - 1
    return lo, hi


def validate(node: Node) -> ValidationReport:
    errors: list[str] = []
    for n in walk(node):
        if n.is_field:
            if n.name not in FIELDS:
                errors.append(f"unknown field {n.name!r}")
            continue
        if n.is_const:
            if not isinstance(n.value, float):
                errors.append(f"constant {n.value!r} is not a float")
            continue
        if n.op not in OPS:
            errors.append(f"unknown operator {n.op!r}")
            continue
        s = OPS[n.op]
        if len(n.children) != s.n_children:
            errors.append(f"{n.op} expects {s.n_children} children, got {len(n.children)}")
        if len(n.params) != len(s.params):
            errors.append(f"{n.op} expects parameters {s.params}, got {n.params}")
            continue
        for pname, kind, val in zip(s.params, s.param_kinds, n.params):
            if kind in ("lag", "lag1", "window"):
                if not isinstance(val, int) or isinstance(val, bool):
                    errors.append(f"{n.op}.{pname} must be an integer, got {val!r}")
                    continue
                if kind == "lag" and val < 0:
                    errors.append(f"{n.op}.{pname}={val}: negative lag (look-ahead)")
                if kind == "lag1" and val < 1:
                    errors.append(f"{n.op}.{pname}={val}: lag must be >= 1")
                if kind == "window" and val < s.min_window:
                    errors.append(f"{n.op}.{pname}={val}: window must be >= {s.min_window}")
            elif kind == "level":
                if not (0.0 < float(val) < 1.0):
                    errors.append(f"{n.op}.{pname}={val}: quantile level must be in (0, 1)")
    if not errors:
        lo, _ = time_offsets(node)
        if lo < 0:
            errors.append(f"tree reads time index t+{-lo} (look-ahead)")
    return ValidationReport(not errors, errors)
