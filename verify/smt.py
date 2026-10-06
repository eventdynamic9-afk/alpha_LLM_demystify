"""Optional formal check (§10.2): prove or refute sign claims on small formula fragments with Z3.

Supported fragment: pointwise operators (+ - * / neg, abs, sign, max/min, comparisons, if, integer
powers) and short time-series operators expanded into lagged leaves (Ref, Delta, Mean/Sum/WMA/Max/Min
with n <= 10).  Domain constraints: prices > 0, volume/amount >= 0, every denominator != 0.

``prove_monotone`` returns "proved" (no counterexample exists), "counterexample" (a concrete input
where raising the leaf moves the formula the other way) or "unknown" (outside the fragment / timeout).
"""
from __future__ import annotations

from dsl import Node
from dsl.fields import FIELDS

MAX_WINDOW = 10


class _Unsupported(Exception):
    pass


def _encode(node: Node, z3, var, side: list, lag: int = 0):
    if node.is_const:
        return z3.RealVal(repr(node.value))
    if node.is_field:
        return var(node.name, lag)
    op, k, p = node.op, node.children, node.params

    def enc(c, extra=0):
        return _encode(c, z3, var, side, lag + extra)

    if op == "Add":
        return enc(k[0]) + enc(k[1])
    if op == "Sub":
        return enc(k[0]) - enc(k[1])
    if op == "Mul":
        return enc(k[0]) * enc(k[1])
    if op == "Div":
        den = enc(k[1])
        side.append(den != 0)
        return enc(k[0]) / den
    if op == "Neg":
        return -enc(k[0])
    if op == "Abs":
        x = enc(k[0])
        return z3.If(x >= 0, x, -x)
    if op == "Sign":
        x = enc(k[0])
        return z3.If(x > 0, z3.RealVal(1), z3.If(x < 0, z3.RealVal(-1), z3.RealVal(0)))
    if op in ("Greater", "Less"):
        a, b = enc(k[0]), enc(k[1])
        return z3.If(a >= b, a, b) if op == "Greater" else z3.If(a <= b, a, b)
    if op in ("Gt", "Ge", "Lt", "Le", "Eq", "Ne"):
        a, b = enc(k[0]), enc(k[1])
        c = {"Gt": a > b, "Ge": a >= b, "Lt": a < b, "Le": a <= b, "Eq": a == b, "Ne": a != b}[op]
        return z3.If(c, z3.RealVal(1), z3.RealVal(0))
    if op == "If":
        return z3.If(enc(k[0]) != 0, enc(k[1]), enc(k[2]))
    if op == "Power":
        e = float(p[0])
        if not e.is_integer() or not (1 <= e <= 5):
            raise _Unsupported("non-integer or large power")
        x = enc(k[0])
        out = x
        for _ in range(int(e) - 1):
            out = out * x
        return out
    if op == "Ref":
        return enc(k[0], int(p[0]))
    if op == "Delta":
        return enc(k[0]) - enc(k[0], int(p[0]))
    if op in ("Mean", "Sum", "WMA", "Max", "Min"):
        n = int(p[0])
        if n > MAX_WINDOW:
            raise _Unsupported("window too long to expand")
        terms = [enc(k[0], j) for j in range(n)]
        if op == "Sum":
            return sum(terms[1:], terms[0])
        if op == "Mean":
            return sum(terms[1:], terms[0]) / n
        if op == "WMA":
            tot = n * (n + 1) / 2
            return sum(((n - j) / tot) * t for j, t in enumerate(terms))
        out = terms[0]
        for t in terms[1:]:
            out = z3.If(t > out, t, out) if op == "Max" else z3.If(t < out, t, out)
        return out
    raise _Unsupported(op)


def prove_monotone(node: Node, fieldname: str, lag: int, direction: str, timeout_ms: int = 5000) -> str:
    try:
        import z3
    except ImportError:  # pragma: no cover
        return "unknown"
    vars_a: dict = {}
    vars_b: dict = {}
    h = z3.Real("h")

    def mk(store, suffix):
        def var(name, lg):
            key = (name, lg)
            if key not in store:
                store[key] = z3.Real(f"{name}_{lg}_{suffix}")
            return store[key]
        return var

    side_a: list = []
    side_b: list = []
    try:
        fa = _encode(node, z3, mk(vars_a, "a"), side_a)
        fb = _encode(node, z3, mk(vars_b, "b"), side_b)
    except _Unsupported:
        return "unknown"
    if (fieldname, lag) not in vars_a:
        return "unknown"
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    for key, va in vars_a.items():
        vb = vars_b.setdefault(key, z3.Real(f"{key[0]}_{key[1]}_b"))
        f = FIELDS[key[0]]
        s.add(va > 0 if f.lo_open else va >= 0)
        if key == (fieldname, lag):
            s.add(vb == va + h)
        else:
            s.add(vb == va)
    s.add(h > 0)
    for c in side_a + side_b:
        s.add(c)
    s.add(fb < fa if direction == "+" else fb > fa)
    r = s.check()
    if r == z3.unsat:
        return "proved"
    if r == z3.sat:
        return "counterexample"
    return "unknown"
