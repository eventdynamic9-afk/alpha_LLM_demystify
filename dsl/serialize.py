"""Serializers from the typed tree to surface notations (§6.1): Qlib DSL (extended or strict),
Alpha101 notation, plain math (LaTeX), field-anonymized Qlib with legend, and a program form with
named intermediates.  All verification runs on the tree; these only produce what narrators see."""
from __future__ import annotations

import random

from .ast import Node, walk
from .operators import OPS

_INFIX_PREC = {"Or": 2, "And": 3, "Gt": 4, "Ge": 4, "Lt": 4, "Le": 4, "Eq": 4, "Ne": 4, "Add": 5, "Sub": 5,
               "Mul": 6, "Div": 6}
_ASSOC = {"Add", "Mul", "And", "Or"}


def fmt_num(v: float) -> str:
    v = float(v)
    if v == 0:
        return "0"
    if v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _param_str(p) -> str:
    return str(p) if isinstance(p, int) else fmt_num(p)


# ------------------------------------------------------------------------------------------ qlib
def to_qlib(node: Node, strict: bool = False, legend: dict[str, str] | None = None) -> str:
    """Qlib expression string.

    ``strict=True`` emits only operators that exist in Qlib (``Rank`` for TsRank, ``-1*x`` for
    negation); cross-sectional operators have no Qlib equivalent and raise ``ValueError``.
    ``legend`` maps field names to anonymous labels (SP field anonymization).
    """
    return _q(node, strict, legend, 0)


def _q(n: Node, strict: bool, legend, parent_prec: int, right: bool = False) -> str:
    if n.is_field:
        return legend[n.name] if legend else f"${n.name}"
    if n.is_const:
        s = fmt_num(n.value)
        return f"({s})" if n.value < 0 and parent_prec > 0 else s
    s = OPS[n.op]
    if n.op in _INFIX_PREC:
        prec = _INFIX_PREC[n.op]
        if n.op in ("And", "Or"):
            # Python (hence Qlib) binds & and | tighter than comparisons: always parenthesize operands
            a = _q(n.children[0], strict, legend, 6)
            b = _q(n.children[1], strict, legend, 6, True)
            txt = f"{a} {'&' if n.op == 'And' else '|'} {b}"
        elif prec == 4:
            # comparisons are non-associative (Python would chain a > b > c)
            a = _q(n.children[0], strict, legend, 5)
            b = _q(n.children[1], strict, legend, 5, True)
            txt = f"{a} {s.infix} {b}"
        else:
            a = _q(n.children[0], strict, legend, prec)
            b = _q(n.children[1], strict, legend, prec + (0 if n.op in _ASSOC else 1), True)
            txt = f"{a}{s.infix}{b}"
        return f"({txt})" if prec < parent_prec or (prec == parent_prec and right) else txt
    if n.op == "Neg":
        inner = _q(n.children[0], strict, legend, 7)
        if strict:
            return f"(-1*{inner})"
        return f"(-{inner})" if parent_prec > 0 else f"-{inner}"
    if n.op == "Power":
        if strict:
            return f"Power({_q(n.children[0], strict, legend, 0)}, {fmt_num(n.params[0])})"
        base = _q(n.children[0], strict, legend, 9)
        return f"{base}^{fmt_num(n.params[0])}" if parent_prec <= 8 else f"({base}^{fmt_num(n.params[0])})"
    name = s.qlib if s.qlib else n.op
    if n.op == "TsRank":
        name = "Rank" if strict else "TsRank"
    if strict and s.qlib is None:
        raise ValueError(f"{n.op} has no equivalent in the Qlib expression language")
    args = [_q(c, strict, legend, 0) for c in n.children] + [_param_str(p) for p in n.params]
    if n.op == "CSScale" and n.params and float(n.params[0]) == 1.0:
        args = args[:-1]
    return f"{name}({', '.join(args)})"


# -------------------------------------------------------------------------------------- alpha101
_A101 = {"CSRank": "rank", "TsRank": "ts_rank", "Ref": "delay", "Delta": "delta", "Corr": "correlation",
         "Cov": "covariance", "Std": "stddev", "Sum": "sum", "Max": "ts_max", "Min": "ts_min",
         "IdxMax": "ts_argmax", "IdxMin": "ts_argmin", "WMA": "decay_linear", "CSScale": "scale",
         "SignedPower": "signedpower", "Abs": "abs", "Log": "log", "Sign": "sign", "Quantile": "ts_quantile",
         "Med": "ts_median", "Slope": "ts_slope", "Rsquare": "ts_rsquare", "Resi": "ts_resi",
         "CSZScore": "zscore"}


def _is_returns(n: Node) -> bool:
    return (n.op == "Sub" and n.children[1].is_const and n.children[1].value == 1.0
            and n.children[0].op == "Div" and n.children[0].children[0].is_field
            and n.children[0].children[0].name == "close" and n.children[0].children[1].op == "Ref"
            and n.children[0].children[1].params == (1,) and n.children[0].children[1].children[0].is_field
            and n.children[0].children[1].children[0].name == "close")


def _is_adv(n: Node) -> int | None:
    if n.op == "Mean" and n.children[0].op == "Mul":
        names = sorted(c.name for c in n.children[0].children if c.is_field)
        if names == ["volume", "vwap"]:
            return int(n.params[0])
    return None


def to_alpha101(node: Node) -> str:
    return _a(node, 0)


def _a(n: Node, parent_prec: int, right: bool = False) -> str:
    if n.is_field:
        return n.name
    if n.is_const:
        s = fmt_num(n.value)
        return f"({s})" if n.value < 0 and parent_prec > 0 else s
    if _is_returns(n):
        return "returns"
    adv = _is_adv(n)
    if adv is not None:
        return f"adv{adv}"
    s = OPS[n.op]
    if n.op in _INFIX_PREC:
        prec = _INFIX_PREC[n.op]
        a = _a(n.children[0], prec)
        b = _a(n.children[1], prec + (0 if n.op in _ASSOC else 1), True)
        sym = s.infix
        txt = f"{a} {sym} {b}"
        return f"({txt})"
    if n.op == "Neg":
        return f"(-1 * {_a(n.children[0], 6)})"
    if n.op == "Not":
        return f"!({_a(n.children[0], 0)})"
    if n.op == "If":
        c, x, y = (_a(ch, 1) for ch in n.children)
        return f"({c} ? {x} : {y})"
    if n.op == "Power":
        return f"({_a(n.children[0], 9)} ^ {fmt_num(n.params[0])})"
    if n.op == "Mean":
        w = int(n.params[0])
        return f"(sum({_a(n.children[0], 0)}, {w}) / {w})"
    if n.op == "Var":
        return f"(stddev({_a(n.children[0], 0)}, {int(n.params[0])}) ^ 2)"
    if n.op in ("Greater", "Less"):
        x, y = n.children
        if x.is_const or y.is_const:
            cmp = ">" if n.op == "Greater" else "<"
            ax, ay = _a(x, 1), _a(y, 1)
            return f"(({ax} {cmp} {ay}) ? {ax} : {ay})"
        return f"{'max' if n.op == 'Greater' else 'min'}({_a(x, 0)}, {_a(y, 0)})"
    name = _A101[n.op]
    args = [_a(c, 0) for c in n.children] + [_param_str(p) for p in n.params]
    if n.op == "CSScale" and float(n.params[0]) == 1.0:
        args = args[:-1]
    return f"{name}({', '.join(args)})"


# ------------------------------------------------------------------------------------------ math
_MATH_NAME = {"Mean": "mean", "Sum": "sum", "Std": "std", "Var": "var", "Max": "max", "Min": "min",
              "IdxMax": "argmax", "IdxMin": "argmin", "TsRank": "tsrank", "Quantile": "quantile",
              "Med": "median", "WMA": "wma", "Slope": "slope", "Rsquare": "R^2", "Resi": "resid",
              "Corr": "corr", "Cov": "cov", "Delta": "\\Delta"}


def to_math(node: Node) -> str:
    """Plain-math (LaTeX) rendering, e.g. ``-\\operatorname{corr}_{10}(\\text{open}, \\text{volume})``."""
    return _m(node, 0)


def _m(n: Node, parent_prec: int) -> str:
    if n.is_field:
        return f"\\text{{{n.name}}}"
    if n.is_const:
        s = fmt_num(n.value)
        return f"({s})" if n.value < 0 and parent_prec > 0 else s
    op = n.op
    if op == "Div":
        return f"\\frac{{{_m(n.children[0], 0)}}}{{{_m(n.children[1], 0)}}}"
    if op in ("Add", "Sub", "Mul"):
        prec = _INFIX_PREC[op]
        sym = {"Add": " + ", "Sub": " - ", "Mul": " \\cdot "}[op]
        a = _m(n.children[0], prec)
        b = _m(n.children[1], prec + (0 if op in _ASSOC else 1))
        txt = f"{a}{sym}{b}"
        return f"\\left({txt}\\right)" if prec < parent_prec or (op == "Sub" and parent_prec == prec + 1) else txt
    if op in ("Gt", "Ge", "Lt", "Le", "Eq", "Ne"):
        sym = {"Gt": ">", "Ge": "\\ge", "Lt": "<", "Le": "\\le", "Eq": "=", "Ne": "\\ne"}[op]
        return f"\\mathbb{{1}}[{_m(n.children[0], 0)} {sym} {_m(n.children[1], 0)}]"
    if op in ("And", "Or"):
        sym = "\\wedge" if op == "And" else "\\vee"
        return f"\\mathbb{{1}}[{_m(n.children[0], 0)} {sym} {_m(n.children[1], 0)}]"
    if op == "Not":
        return f"\\mathbb{{1}}[\\neg {_m(n.children[0], 0)}]"
    if op == "Neg":
        return f"-{_m(n.children[0], 7)}"
    if op == "Abs":
        return f"\\left|{_m(n.children[0], 0)}\\right|"
    if op == "Sign":
        return f"\\operatorname{{sign}}({_m(n.children[0], 0)})"
    if op == "Log":
        return f"\\ln({_m(n.children[0], 0)})"
    if op == "Power":
        return f"{_m(n.children[0], 9)}^{{{fmt_num(n.params[0])}}}"
    if op == "SignedPower":
        x = _m(n.children[0], 0)
        return f"\\operatorname{{sign}}({x})\\left|{x}\\right|^{{{fmt_num(n.params[0])}}}"
    if op == "Greater":
        return f"\\max({_m(n.children[0], 0)}, {_m(n.children[1], 0)})"
    if op == "Less":
        return f"\\min({_m(n.children[0], 0)}, {_m(n.children[1], 0)})"
    if op == "If":
        c, x, y = (_m(ch, 0) for ch in n.children)
        return f"\\begin{{cases}} {x} & \\text{{if }} {c} \\ne 0 \\\\ {y} & \\text{{otherwise}} \\end{{cases}}"
    if op == "Ref":
        return f"\\operatorname{{lag}}_{{{n.params[0]}}}({_m(n.children[0], 0)})"
    if op == "CSRank":
        return f"\\operatorname{{rank}}_{{\\mathrm{{cs}}}}({_m(n.children[0], 0)})"
    if op == "CSZScore":
        return f"\\operatorname{{z}}_{{\\mathrm{{cs}}}}({_m(n.children[0], 0)})"
    if op == "CSScale":
        return f"\\operatorname{{scale}}_{{\\mathrm{{cs}}}}({_m(n.children[0], 0)})"
    name = _MATH_NAME[op]
    sub = ",".join(_param_str(p) for p in n.params)
    args = ", ".join(_m(c, 0) for c in n.children)
    opname = name if name.startswith("\\") or "^" in name else f"\\operatorname{{{name}}}"
    return f"{opname}_{{{sub}}}({args})"


# ------------------------------------------------------------------------------- anonymization
def anonymize_legend(node: Node, seed: int = 0, all_fields: bool = False) -> dict[str, str]:
    """Field -> anonymous label map (``close -> x3``); order randomized by ``seed`` (SP-iv)."""
    from .fields import FIELDS

    names = sorted(FIELDS) if all_fields else sorted({n.name for n in walk(node) if n.is_field})
    rng = random.Random(seed)
    labels = [f"x{i + 1}" for i in range(len(names))]
    rng.shuffle(labels)
    return dict(zip(names, labels))


def to_anonymized(node: Node, seed: int = 0) -> tuple[str, dict[str, str]]:
    legend = anonymize_legend(node, seed)
    return _q(node, False, legend, 0), legend


# --------------------------------------------------------------------------- named intermediates
def to_program(node: Node, min_size: int = 3, notation: str = "qlib", names: str = "ABCDEFGHIJKLMNOP") -> str:
    """Rename/inline intermediates (SP-ii): every distinct time-series/cross-sectional subtree with at
    least ``min_size`` nodes becomes a named intermediate, outermost formula written as ``factor = ...``."""
    from .ast import size

    order: list[Node] = []

    def visit(n: Node) -> None:
        for c in n.children:
            visit(c)
        if not n.is_leaf and n is not node and OPS[n.op].kind in ("ts", "xs") and size(n) >= min_size:
            if n not in order:
                order.append(n)

    visit(node)
    order = order[: len(names)]
    label = {n: names[i] for i, n in enumerate(order)}
    ser = to_qlib if notation == "qlib" else to_alpha101

    def render(n: Node) -> str:
        def repl(x: Node, is_root: bool) -> Node:
            if not is_root and x in label:
                return Node("$", (), (f"__var_{label[x]}",))
            if x.children:
                return Node(x.op, tuple(repl(c, False) for c in x.children), x.params)
            return x

        txt = ser(repl(n, True))
        for v in names:
            txt = txt.replace(f"$__var_{v}", v).replace(f"__var_{v}", v)
        return txt

    lines = [f"{label[n]} = {render(n)}" for n in order]
    lines.append(f"factor = {render(node)}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ gtja
_GTJA = {"CSRank": "RANK", "TsRank": "TSRANK", "Ref": "DELAY", "Delta": "DELTA", "Corr": "CORR",
         "Cov": "COVIANCE", "Std": "STD", "Sum": "SUM", "Mean": "MEAN", "Max": "TSMAX", "Min": "TSMIN",
         "Greater": "MAX", "Less": "MIN", "Abs": "ABS", "Log": "LOG", "Sign": "SIGN", "WMA": "DECAYLINEAR"}


def to_gtja(node: Node) -> str:
    """GTJA-191 notation (upper case; MAX/MIN elementwise, TSMAX/TSMIN rolling).  Operators without a
    GTJA name raise ValueError (callers fall back to another notation)."""
    return _g(node)


def _g(n: Node) -> str:
    if n.is_field:
        return n.name.upper()
    if n.is_const:
        s = fmt_num(n.value)
        return f"({s})" if n.value < 0 else s
    if _is_returns(n):
        return "RET"
    s = OPS[n.op]
    if n.op in _INFIX_PREC:
        return f"({_g(n.children[0])} {s.infix} {_g(n.children[1])})"
    if n.op == "Neg":
        return f"(-1 * {_g(n.children[0])})"
    if n.op == "If":
        c, x, y = (_g(ch) for ch in n.children)
        return f"({c} ? {x} : {y})"
    if n.op == "Power":
        return f"({_g(n.children[0])} ^ {fmt_num(n.params[0])})"
    if n.op not in _GTJA:
        raise ValueError(f"{n.op} has no GTJA-191 name")
    args = [_g(c) for c in n.children] + [_param_str(p) for p in n.params]
    return f"{_GTJA[n.op]}({', '.join(args)})"


def to_notation(node: Node, notation: str) -> str:
    """Serialize in 'qlib' | 'alpha101' | 'gtja' | 'math' (gtja falls back to alpha101)."""
    if notation == "qlib":
        return to_qlib(node)
    if notation == "alpha101":
        return to_alpha101(node)
    if notation == "math":
        return to_math(node)
    if notation == "gtja":
        try:
            return to_gtja(node)
        except ValueError:
            return to_alpha101(node)
    raise ValueError(notation)
