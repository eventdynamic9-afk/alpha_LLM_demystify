"""Symbolic normalization and canonical hashing (§6.5).

Rewrites (iterated to a fixed point): constant folding; ``Sub``/``Delta`` expanded into n-ary
``Add`` with ``Neg`` terms; flattening and commutative ordering of ``Add``/``Mul``/``Greater``/
``Less``/``Eq``/``Ne``/``And``/``Or`` and of the symmetric ``Corr``/``Cov`` arguments; double
negation removal and sign extraction from products; ``Lt``/``Le`` mirrored to ``Gt``/``Ge``;
``Ref`` composition (``Ref(Ref(x,a),b) -> Ref(x,a+b)``, ``Ref(x,0) -> x``) and ``Ref`` pushed
through pointwise and time-series operators down to the leaves (it is *not* pushed through
cross-sectional operators because the universe changes over time).

The canonical form is a normal form for equality testing only; numerical equivalence on data
(``verify.identity.numerically_equivalent``) remains the final arbiter.
"""
from __future__ import annotations

import hashlib
import math

from .ast import C, Node
from .operators import OPS, POINT, TS
from .serialize import to_qlib

_COMMUTATIVE_NARY = {"Add", "Mul"}
_COMMUTATIVE_BIN = {"Greater", "Less", "Eq", "Ne", "And", "Or", "Corr", "Cov"}


def _key(n: Node) -> str:
    return to_qlib(n)


def scalar_eval(op: str, vals: list[float], params: tuple) -> float | None:
    """Evaluate a pointwise operator on constants (constant folding)."""
    try:
        with_nan = any(math.isnan(v) for v in vals)
        if with_nan:
            return math.nan
        if op == "Add":
            return sum(vals)
        if op == "Sub":
            return vals[0] - vals[1]
        if op == "Mul":
            out = 1.0
            for v in vals:
                out *= v
            return out
        if op == "Div":
            return math.nan if vals[1] == 0 else vals[0] / vals[1]
        if op == "Neg":
            return -vals[0]
        if op == "Abs":
            return abs(vals[0])
        if op == "Sign":
            return float((vals[0] > 0) - (vals[0] < 0))
        if op == "Log":
            return math.log(vals[0]) if vals[0] > 0 else math.nan
        if op == "Power":
            x, p = vals[0], float(params[0])
            if x < 0 and not float(p).is_integer():
                return math.nan
            if x == 0 and p < 0:
                return math.nan
            return x ** p
        if op == "SignedPower":
            x, p = vals[0], float(params[0])
            return math.copysign(abs(x) ** p, x) if x != 0 else 0.0
        if op == "Greater":
            return max(vals)
        if op == "Less":
            return min(vals)
        cmp = {"Gt": lambda a, b: a > b, "Ge": lambda a, b: a >= b, "Lt": lambda a, b: a < b,
               "Le": lambda a, b: a <= b, "Eq": lambda a, b: a == b, "Ne": lambda a, b: a != b,
               "And": lambda a, b: a != 0 and b != 0, "Or": lambda a, b: a != 0 or b != 0}
        if op in cmp:
            return float(cmp[op](vals[0], vals[1]))
        if op == "Not":
            return float(vals[0] == 0)
        if op == "If":
            return vals[1] if vals[0] != 0 else vals[2]
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.nan
    return None


def _neg(n: Node) -> Node:
    if n.is_const:
        return C(-n.value)
    if n.op == "Neg":
        return n.children[0]
    return Node("Neg", (n,))


def _push_ref(x: Node, d: int) -> Node:
    """Ref(x, d) with the lag pushed down through pointwise and time-series operators."""
    if d == 0:
        return x
    if x.is_const:
        return x
    if x.is_field:
        return Node("Ref", (x,), (d,))
    s = OPS[x.op]
    if x.op == "Ref":
        return _push_ref(x.children[0], d + int(x.params[0]))
    if s.kind == POINT or s.kind == TS:
        return Node(x.op, tuple(_push_ref(c, d) for c in x.children), x.params)
    return Node("Ref", (x,), (d,))  # cross-sectional: stop


def _step(n: Node) -> Node:
    if n.is_leaf:
        return n
    kids = tuple(_step(c) for c in n.children)
    op, params = n.op, n.params

    # expansions to the normal vocabulary
    if op == "Sub":
        return _step_add((kids[0], _neg(kids[1])))
    if op == "Delta":
        return _step_add((kids[0], _neg(_push_ref(kids[0], int(params[0])))))
    if op == "Lt":
        op, kids = "Gt", (kids[1], kids[0])
    elif op == "Le":
        op, kids = "Ge", (kids[1], kids[0])

    # constant folding of pointwise operators
    if OPS[op].kind == POINT and all(k.is_const for k in kids):
        v = scalar_eval(op, [k.value for k in kids], params)
        if v is not None:
            return C(v)

    if op == "Ref":
        return _push_ref(kids[0], int(params[0]))
    if op == "Neg":
        k = kids[0]
        if k.op == "Add":
            return _step_add(tuple(_neg(c) for c in k.children))
        return _neg(k)
    if op == "Add":
        return _step_add(kids)
    if op == "Mul":
        return _step_mul(kids)
    if op == "Div":
        num, den = kids
        if den.is_const and den.value == 1.0:
            return num
        sign = 1
        if num.op == "Neg":
            num, sign = num.children[0], -sign
        if den.op == "Neg":
            den, sign = den.children[0], -sign
        if den.is_const and den.value < 0:
            den, sign = C(-den.value), -sign
        out = Node("Div", (num, den))
        return out if sign > 0 else Node("Neg", (out,))
    if op == "Power" and float(params[0]) == 1.0:
        return kids[0]
    if op == "Abs":
        k = kids[0]
        if k.op in ("Abs", "Neg"):
            return Node("Abs", (k.children[0],)) if k.op == "Neg" else k
    if op == "Sign" and kids[0].op == "Neg":
        return Node("Neg", (Node("Sign", (kids[0].children[0],)),))
    if op in _COMMUTATIVE_BIN:
        kids = tuple(sorted(kids, key=_key))
    return Node(op, kids, params)


def _flatten(op: str, kids: tuple[Node, ...]) -> list[Node]:
    out: list[Node] = []
    for k in kids:
        if k.op == op:
            out.extend(_flatten(op, k.children))
        else:
            out.append(k)
    return out


def _nary(op: str, terms: list[Node]) -> Node:
    """Build a left-nested binary chain from sorted terms (keeps the binary operator table)."""
    node = terms[0]
    for t in terms[1:]:
        node = Node(op, (node, t))
    return node


def _step_add(kids: tuple[Node, ...]) -> Node:
    terms = _flatten("Add", kids)
    const = 0.0
    rest: list[Node] = []
    for t in terms:
        if t.is_const:
            const += t.value
        else:
            rest.append(t)
    # cancel x + (-x)
    keys = [_key(t) for t in rest]
    keep = [True] * len(rest)
    for i, t in enumerate(rest):
        if not keep[i]:
            continue
        target = _key(_neg(t))
        for j in range(i + 1, len(rest)):
            if keep[j] and keys[j] == target:
                keep[i] = keep[j] = False
                break
    rest = [t for t, k in zip(rest, keep) if k]
    if const != 0.0 or not rest:
        rest.append(C(const))
    if len(rest) == 1:
        return rest[0]
    rest.sort(key=_key)
    return _nary("Add", rest)


def _step_mul(kids: tuple[Node, ...]) -> Node:
    terms = _flatten("Mul", kids)
    coef = 1.0
    sign = 1
    rest: list[Node] = []
    for t in terms:
        while t.op == "Neg":
            t, sign = t.children[0], -sign
        if t.is_const:
            coef *= t.value
        else:
            rest.append(t)
    if coef < 0:
        coef, sign = -coef, -sign
    if coef == 0.0:
        return C(0.0)
    if coef != 1.0 or not rest:
        rest.append(C(coef))
    rest.sort(key=_key)
    node = rest[0] if len(rest) == 1 else _nary("Mul", rest)
    return node if sign > 0 else Node("Neg", (node,))


def canonicalize(node: Node, max_iter: int = 20) -> Node:
    cur = node
    for _ in range(max_iter):
        nxt = _step(cur)
        if nxt == cur:
            return cur
        cur = nxt
    return cur


def canonical_string(node: Node) -> str:
    return to_qlib(canonicalize(node))


def canonical_hash(node: Node) -> str:
    return "sha256:" + hashlib.sha256(canonical_string(node).encode("utf-8")).hexdigest()


def canonical_equal(a: Node, b: Node) -> bool:
    return canonical_string(a) == canonical_string(b)
