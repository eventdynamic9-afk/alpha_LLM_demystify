"""Monotonicity abstract interpretation (§10.2).

The analysis propagates a sign lattice {↑ (+1), ↓ (-1), ± (AMB)} from each *time-indexed leaf*
``(field, lag)`` to the root, using the per-argument monotonicity codes of the operator table.
``Ref`` shifts lags, windowed operators spread a child's map over the window, ``Delta``/``TsRank``/
``Slope``/``Resi`` assign signs per lag, and products/ratios use the value-sign of the other operand
from :mod:`dsl.ranges`.  Absent keys mean "no dependence" (0).

A definite ↑/↓ decides SIGN/MONO claims exactly; ± sends the claim to the nudge test.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .ast import Node
from .operators import BOOLEAN_OPS
from .ranges import interval

POS, NEG, AMB = 1, -1, 2
Key = Tuple[str, int]
SignMap = Dict[Key, int]

_SYMBOL = {POS: "+", NEG: "-", AMB: "±", 0: "0"}


def symbol(s: int) -> str:
    return _SYMBOL[s]


def join(a: int, b: int) -> int:
    if a == 0:
        return b
    if b == 0:
        return a
    return a if a == b else AMB


def negate(s: int) -> int:
    return -s if s in (POS, NEG) else s


def scale(m: SignMap, s: int) -> SignMap:
    if s == POS:
        return dict(m)
    if s == NEG:
        return {k: negate(v) for k, v in m.items()}
    if s == 0:
        return {}
    return {k: AMB for k in m}


def merge(*maps: SignMap) -> SignMap:
    out: SignMap = {}
    for m in maps:
        for k, v in m.items():
            out[k] = join(out.get(k, 0), v)
    return out


def shift(m: SignMap, d: int) -> SignMap:
    return {(f, lag + d): v for (f, lag), v in m.items()}


def _window(m: SignMap, n: int, signs=None) -> SignMap:
    parts = []
    for k in range(n):
        s = POS if signs is None else signs[k]
        if s != 0:
            parts.append(scale(shift(m, k), s))
    return merge(*parts)


def _ols_weights(n: int, which: str) -> np.ndarray:
    """Exact derivative of Slope / last Resi w.r.t. x at lag k (k = 0 newest)."""
    j = np.arange(1, n + 1, dtype=float)       # time index, oldest = 1
    tbar = j.mean()
    stt = ((j - tbar) ** 2).sum()
    if which == "slope":
        w = (j - tbar) / stt
    else:  # residual at the last point
        w = -1.0 / n - (n - tbar) * (j - tbar) / stt
        w[-1] += 1.0
    return w[::-1]                               # index by lag k


def _value_sign(node: Node) -> int:
    iv = interval(node)
    if iv.nonneg:
        return POS
    if iv.nonpos:
        return NEG
    return AMB


def _is_bool(node: Node) -> bool:
    return node.op in BOOLEAN_OPS


def sign_map(node: Node) -> SignMap:
    if node.is_const:
        return {}
    if node.is_field:
        return {(node.name, 0): POS}
    op = node.op
    kids = node.children
    ms = [sign_map(c) for c in kids]
    if op == "Add":
        return merge(ms[0], ms[1])
    if op == "Sub":
        return merge(ms[0], scale(ms[1], NEG))
    if op == "Neg":
        return scale(ms[0], NEG)
    if op == "Mul":
        return merge(scale(ms[0], _value_sign(kids[1])), scale(ms[1], _value_sign(kids[0])))
    if op == "Div":
        den_iv = interval(kids[1])
        # division by zero is NaN (undefined), so a non-negative denominator is positive on the domain
        if den_iv.nonneg:
            s_num = POS
        elif den_iv.nonpos:
            s_num = NEG
        else:
            s_num = AMB
        # d/dy (x / y) = -x / y^2: sign is -sign(x) when y does not cross zero
        s_den = AMB if s_num == AMB else negate(_value_sign(kids[0]))
        return merge(scale(ms[0], s_num), scale(ms[1], s_den))
    if op == "Abs":
        return scale(ms[0], _value_sign(kids[0]))
    if op in ("Sign", "Log", "CSRank", "CSZScore", "CSScale"):
        return ms[0]
    if op == "Power":
        p = float(node.params[0])
        iv = interval(kids[0])
        if p == 0:
            return {}
        if p > 0:
            if iv.nonneg or (p.is_integer() and int(p) % 2 == 1):
                return ms[0]
            if p.is_integer() and int(p) % 2 == 0 and iv.nonpos:
                return scale(ms[0], NEG)
            return scale(ms[0], AMB)
        if iv.strictly_positive:
            return scale(ms[0], NEG)
        return scale(ms[0], AMB)
    if op == "SignedPower":
        p = float(node.params[0])
        return ms[0] if p >= 0 else scale(ms[0], AMB)
    if op in ("Greater", "Less"):
        return merge(ms[0], ms[1])
    if op in ("Gt", "Ge"):
        return merge(ms[0], scale(ms[1], NEG))
    if op in ("Lt", "Le"):
        return merge(scale(ms[0], NEG), ms[1])
    if op in ("Eq", "Ne"):
        return scale(merge(ms[0], ms[1]), AMB)
    if op in ("And", "Or"):
        return merge(*(m if _is_bool(k) else scale(m, AMB) for m, k in zip(ms, kids)))
    if op == "Not":
        return scale(ms[0], NEG if _is_bool(kids[0]) else AMB)
    if op == "If":
        return merge(scale(ms[0], AMB), ms[1], ms[2])
    if op == "Ref":
        return shift(ms[0], int(node.params[0]))
    if op == "Delta":
        d = int(node.params[0])
        return merge(ms[0], scale(shift(ms[0], d), NEG))
    if op in ("Mean", "Sum", "WMA", "Max", "Min", "Med", "Quantile"):
        return _window(ms[0], int(node.params[0]))
    if op in ("Std", "Var", "IdxMax", "IdxMin", "Rsquare"):
        return scale(_window(ms[0], int(node.params[0])), AMB)
    if op == "TsRank":
        n = int(node.params[0])
        return _window(ms[0], n, [POS] + [NEG] * (n - 1))
    if op in ("Slope", "Resi"):
        n = int(node.params[0])
        w = _ols_weights(n, "slope" if op == "Slope" else "resi")
        signs = [POS if v > 1e-12 else NEG if v < -1e-12 else 0 for v in w]
        return _window(ms[0], n, signs)
    if op in ("Corr", "Cov"):
        n = int(node.params[0])
        return scale(merge(_window(ms[0], n), _window(ms[1], n)), AMB)
    raise KeyError(f"no monotonicity rule for {op}")


def field_direction(node: Node, fieldname: str, lags: tuple[int, ...] | None = None) -> int:
    """Direction of the formula w.r.t. ``fieldname`` raised at the given lags (all lags if None).

    Returns POS, NEG, AMB, or 0 (no dependence).
    """
    m = sign_map(node)
    out = 0
    for (f, lag), s in m.items():
        if f != fieldname:
            continue
        if lags is not None and lag not in lags:
            continue
        out = join(out, s)
    return out


def current_value_direction(node: Node, fieldname: str) -> int:
    return field_direction(node, fieldname, (0,))
