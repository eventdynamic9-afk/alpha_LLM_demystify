"""Static interval (output-range) analysis used for RANGE claims and for value-sign information in
the monotonicity analysis (§6.2 "output range", §10.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ast import Node
from .fields import FIELDS

INF = math.inf


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    lo_open: bool = False
    hi_open: bool = False

    @property
    def strictly_positive(self) -> bool:
        return self.lo > 0 or (self.lo == 0 and self.lo_open)

    @property
    def strictly_negative(self) -> bool:
        return self.hi < 0 or (self.hi == 0 and self.hi_open)

    @property
    def nonneg(self) -> bool:
        return self.lo >= 0

    @property
    def nonpos(self) -> bool:
        return self.hi <= 0

    def sign(self) -> int:
        """+1 if >= 0 (non-negative), -1 if <= 0, 0 if mixed/unknown."""
        if self.nonneg:
            return 1
        if self.nonpos:
            return -1
        return 0

    def within(self, lo: float, hi: float) -> bool:
        return self.lo >= lo and self.hi <= hi

    def disjoint(self, lo: float, hi: float) -> bool:
        return self.hi < lo or self.lo > hi

    def as_list(self) -> list[float]:
        return [self.lo, self.hi]


REAL = Interval(-INF, INF)


def _mul_end(a: float, b: float) -> float:
    if (a == 0 and math.isinf(b)) or (b == 0 and math.isinf(a)):
        return 0.0
    return a * b


def _mul(x: Interval, y: Interval) -> Interval:
    c = [_mul_end(x.lo, y.lo), _mul_end(x.lo, y.hi), _mul_end(x.hi, y.lo), _mul_end(x.hi, y.hi)]
    lo, hi = min(c), max(c)
    lo_open = (x.strictly_positive and y.strictly_positive) or (x.strictly_negative and y.strictly_negative)
    hi_open = (x.strictly_positive and y.strictly_negative) or (x.strictly_negative and y.strictly_positive)
    return Interval(lo, hi, lo_open and lo == 0, hi_open and hi == 0)


def _recip(y: Interval) -> Interval | None:
    # x / 0 is NaN in the DSL, so a closed bound at 0 is excluded from the domain
    if y.lo == 0 and not y.lo_open and y.hi > 0:
        y = Interval(0.0, y.hi, True, y.hi_open)
    if y.hi == 0 and not y.hi_open and y.lo < 0:
        y = Interval(y.lo, 0.0, y.lo_open, True)
    if y.strictly_positive:
        lo = 1.0 / y.hi if y.hi < INF else 0.0
        hi = 1.0 / y.lo if y.lo > 0 else INF
        return Interval(lo, hi, lo_open=(y.hi == INF), hi_open=False)
    if y.strictly_negative:
        r = _recip(Interval(-y.hi, -y.lo, y.hi_open, y.lo_open))
        return Interval(-r.hi, -r.lo, r.hi_open, r.lo_open)
    return None


def interval(node: Node) -> Interval:
    if node.is_const:
        return Interval(node.value, node.value)
    if node.is_field:
        f = FIELDS[node.name]
        return Interval(f.lo, f.hi, f.lo_open)
    ks = [interval(c) for c in node.children]
    op = node.op
    rule = {"Add": "add", "Sub": "sub"}.get(op)
    if op == "Add":
        a, b = ks
        return Interval(a.lo + b.lo, a.hi + b.hi, a.lo_open or b.lo_open, a.hi_open or b.hi_open)
    if op == "Sub":
        a, b = ks
        return Interval(a.lo - b.hi, a.hi - b.lo, a.lo_open or b.hi_open, a.hi_open or b.lo_open)
    if op == "Neg":
        a = ks[0]
        return Interval(-a.hi, -a.lo, a.hi_open, a.lo_open)
    if op == "Mul":
        return _mul(ks[0], ks[1])
    if op == "Div":
        r = _recip(ks[1])
        return REAL if r is None else _mul(ks[0], r)
    if op == "Abs":
        a = ks[0]
        if a.nonneg:
            return a
        if a.nonpos:
            return Interval(-a.hi, -a.lo, a.hi_open, a.lo_open)
        return Interval(0.0, max(-a.lo, a.hi))
    if op == "Sign":
        a = ks[0]
        if a.strictly_positive:
            return Interval(1.0, 1.0)
        if a.strictly_negative:
            return Interval(-1.0, -1.0)
        return Interval(0.0 if a.nonneg else -1.0, 0.0 if a.nonpos else 1.0)
    if op == "Log":
        a = ks[0]
        lo = math.log(a.lo) if a.lo > 0 else -INF
        hi = math.log(a.hi) if 0 < a.hi < INF else (INF if a.hi == INF else -INF)
        return Interval(lo, hi)
    if op in ("Power", "SignedPower"):
        a, p = ks[0], float(node.params[0])
        if op == "Power" and a.nonneg and p > 0:
            hi = a.hi ** p if a.hi < INF else INF
            return Interval(a.lo ** p, hi, a.lo_open and a.lo == 0)
        if op == "Power" and a.strictly_positive and p < 0:
            lo = a.hi ** p if a.hi < INF else 0.0
            hi = a.lo ** p if a.lo > 0 else INF
            return Interval(lo, hi, lo_open=(a.hi == INF))
        if op == "Power" and float(p).is_integer() and int(p) % 2 == 0 and p > 0:
            m = max(abs(a.lo), abs(a.hi))
            return Interval(0.0, m ** p if m < INF else INF)
        if op == "SignedPower" and p > 0:
            def sp(v: float) -> float:
                return math.copysign(abs(v) ** p, v) if not math.isinf(v) else v
            return Interval(sp(a.lo), sp(a.hi), a.lo_open, a.hi_open)
        return REAL
    if op in ("Greater", "Less"):
        a, b = ks
        f = max if op == "Greater" else min
        return Interval(f(a.lo, b.lo), f(a.hi, b.hi))
    if op in ("Gt", "Ge", "Lt", "Le", "Eq", "Ne", "And", "Or", "Not"):
        return Interval(0.0, 1.0)
    if op == "If":
        _, a, b = ks
        return Interval(min(a.lo, b.lo), max(a.hi, b.hi))
    if op in ("Ref", "Mean", "Max", "Min", "Med", "Quantile", "WMA"):
        a = ks[0]
        return Interval(a.lo, a.hi, a.lo_open, a.hi_open)
    if op == "Sum":
        a, n = ks[0], int(node.params[0])
        return Interval(_mul_end(a.lo, n), _mul_end(a.hi, n), a.lo_open, a.hi_open)
    if op == "Delta":
        a = ks[0]
        w = a.hi - a.lo
        return Interval(-w, w) if not math.isinf(w) else REAL
    if op in ("Std", "Var"):
        a = ks[0]
        w = a.hi - a.lo
        if math.isinf(w):
            return Interval(0.0, INF)
        return Interval(0.0, (w / 2 * math.sqrt(2)) if op == "Std" else (w * w / 2))
    if op in ("IdxMax", "IdxMin"):
        return Interval(1.0, float(node.params[0]))
    if op in ("TsRank", "CSRank"):
        return Interval(0.0, 1.0, lo_open=True)
    if op == "Rsquare":
        return Interval(0.0, 1.0)
    if op == "Corr":
        return Interval(-1.0, 1.0)
    if op == "CSScale":
        a = abs(float(node.params[0]))
        return Interval(-a, a)
    if op in ("Slope", "Resi", "Cov", "CSZScore"):
        return REAL
    del rule
    return REAL
