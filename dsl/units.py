"""Dimensional-consistency check (§6.2 unit rule).

Units are products of base dimensions (``price``, ``shares``) with real exponents; ``Log`` of a
dimensional quantity produces a ``log(<unit>)`` tag.  Constants are unit-polymorphic.  Adding a
volume term to a log-price term (the example criticised by Alpha Jungle) is reported as an issue;
dimensionally inconsistent formulas are allowed (they occur in GP output) but flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ast import Node
from .fields import FIELDS

DIMENSIONLESS: tuple = ()
ANY = ("__any__",)


def _norm(d: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in d.items() if abs(v) > 1e-12))


def _mul(a: tuple, b: tuple, sign: float = 1.0) -> tuple:
    if a == ANY:
        return b if sign > 0 else _pow(b, -1.0)
    if b == ANY:
        return a
    d = dict(a)
    for k, v in b:
        d[k] = d.get(k, 0.0) + sign * v
    return _norm(d)


def _pow(a: tuple, p: float) -> tuple:
    if a == ANY:
        return ANY
    return _norm({k: v * p for k, v in a})


def _is_log(u: tuple) -> bool:
    return len(u) == 1 and u[0][0].startswith("log(")


def unit_str(u: tuple) -> str:
    if u == ANY:
        return "any"
    if not u:
        return "1"
    return "*".join(k if v == 1 else f"{k}^{v:g}" for k, v in u)


@dataclass
class UnitReport:
    unit: tuple
    issues: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not self.issues

    @property
    def dimensionless(self) -> bool:
        return self.unit in (DIMENSIONLESS, ANY)


def _same(op: str, us: list[tuple], issues: list[str]) -> tuple:
    concrete = [u for u in us if u != ANY]
    if not concrete:
        return ANY
    first = concrete[0]
    for u in concrete[1:]:
        if u != first:
            issues.append(f"{op}: incompatible units {unit_str(first)} and {unit_str(u)}")
    return first


def analyze_units(node: Node) -> UnitReport:
    issues: list[str] = []

    def go(n: Node) -> tuple:
        if n.is_const:
            return ANY
        if n.is_field:
            return FIELDS[n.name].unit
        us = [go(c) for c in n.children]
        op = n.op
        if op == "Sub" and len(us) == 2 and _is_log(us[0]) and us[0] == us[1]:
            return DIMENSIONLESS
        if op in ("Add", "Sub", "Greater", "Less"):
            return _same(op, us, issues)
        if op in ("Gt", "Ge", "Lt", "Le", "Eq", "Ne"):
            _same(op, us, issues)
            return DIMENSIONLESS
        if op in ("And", "Or", "Not"):
            return DIMENSIONLESS
        if op == "If":
            return _same(op, us[1:], issues)
        if op == "Mul":
            if any(_is_log(u) for u in us) and all(u != ANY for u in us):
                issues.append("Mul: product involving a log-unit")
            return _mul(us[0], us[1])
        if op == "Div":
            if any(_is_log(u) for u in us) and all(u != ANY for u in us) and us[0] != us[1]:
                issues.append("Div: ratio involving a log-unit")
            return _mul(us[0], us[1], -1.0)
        if op == "Log":
            u = us[0]
            if u in (DIMENSIONLESS, ANY):
                return u
            return (("log(" + unit_str(u) + ")", 1.0),)
        if op in ("Power", "SignedPower"):
            if _is_log(us[0]) and float(n.params[0]) != 1.0:
                issues.append(f"{op}: power of a log-unit")
                return us[0]
            return _pow(us[0], float(n.params[0]))
        if op in ("Sign", "CSRank", "TsRank", "Corr", "Rsquare", "CSZScore", "CSScale", "IdxMax", "IdxMin"):
            return DIMENSIONLESS
        if op == "Var":
            return _pow(us[0], 2.0)
        if op == "Cov":
            return _mul(us[0], us[1])
        # Ref, Delta, Mean, Sum, Std, Max, Min, Med, Quantile, WMA, Slope, Resi, Abs, Neg
        return us[0]

    u = go(node)
    return UnitReport(u, issues)
