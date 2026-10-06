"""Parsers from surface notations into the typed tree (§6.1).

Dialects
--------
``qlib``     Qlib expression language (``Mean($close, 5)``) extended with ``CSRank``/``CSZScore``/
             ``CSScale``.  ``Rank(x, n)`` is Qlib's rolling *time-series* rank -> ``TsRank``; a
             one-argument ``Rank(x)`` (not valid Qlib) is read as cross-sectional ``CSRank``.
``alpha101`` Kakushadze (2016) notation: ``rank``, ``ts_rank``, ``delay``, ``correlation``,
             ``decay_linear``, ``returns``, ``adv20``, ternary ``c ? a : b``; non-integer windows are
             floored as in the paper; ``min``/``max`` with a constant second argument are ts_min/ts_max.
``gtja``     Guotai Junan (2017) "191" notation: upper-case names, ``MAX``/``MIN`` elementwise,
             ``TSMAX``/``TSMIN`` rolling, ``COUNT``, ``HIGHDAY``/``LOWDAY``, ``=`` equality,
             Matlab-style ``.*`` and ``./``.

All dialects accept multi-line programs with intermediates (``A = Mean($close, 5); factor = A/$close``),
which is the "rename/inline intermediates" semantics-preserving surface form (§7.2 SP-ii).
Precedence (low -> high): ternary, ``||``, ``&&``, comparisons, ``+ -``, ``* /``, unary, power.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .ast import C, F, Node
from .fields import FIELDS
from .operators import BOOLEAN_OPS, OPS


class ParseError(ValueError):
    pass


class UnsupportedOperator(ParseError):
    pass


_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<field>\$[A-Za-z_]\w*)
  | (?P<ident>[A-Za-z_]\w*)
  | (?P<op>\*\*|&&|\|\||>=|<=|==|!=|\.\*|\./|[-+*/^><=&|!?:,();])
    """,
    re.VERBOSE,
)


@dataclass
class Tok:
    kind: str
    text: str
    pos: int


def tokenize(src: str) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    while i < len(src):
        m = _TOKEN.match(src, i)
        if not m:
            raise ParseError(f"unexpected character {src[i]!r} at {i}")
        kind = m.lastgroup
        if kind != "ws":
            out.append(Tok(kind, m.group(), i))
        i = m.end()
    out.append(Tok("eof", "", len(src)))
    return out


# --------------------------------------------------------------------------- dialect function maps
# value: canonical op name, or a callable(args, parser) -> Node for macros / disambiguation
QLIB_FUNCS = {n.lower(): n for n in OPS if n not in ("Neg",)}
QLIB_FUNCS.update({"rank": "__qlib_rank__", "tsrank": "TsRank", "max": "__ts_or_elem_max__",
                   "min": "__ts_or_elem_min__", "neg": "Neg"})

ALPHA101_FUNCS = {
    "abs": "Abs", "log": "Log", "sign": "Sign", "rank": "CSRank", "delay": "Ref", "correlation": "Corr",
    "corr": "Corr", "covariance": "Cov", "cov": "Cov", "scale": "CSScale", "delta": "Delta",
    "signedpower": "SignedPower", "decay_linear": "WMA", "ts_min": "Min", "ts_max": "Max",
    "min": "__ts_or_elem_min__", "max": "__ts_or_elem_max__", "ts_argmax": "IdxMax", "ts_argmin": "IdxMin",
    "ts_rank": "TsRank", "sum": "Sum", "ts_sum": "Sum", "stddev": "Std", "ts_stddev": "Std",
    "ts_mean": "Mean", "mean": "Mean", "sma": "Mean", "ts_quantile": "Quantile", "ts_median": "Med",
    "ts_slope": "Slope", "ts_rsquare": "Rsquare", "ts_resi": "Resi", "zscore": "CSZScore",
    "power": "Power",
}
ALPHA101_UNSUPPORTED = {"product", "indneutralize", "ts_product"}

GTJA_FUNCS = {
    "rank": "CSRank", "tsrank": "TsRank", "delay": "Ref", "delta": "Delta", "corr": "Corr",
    "coviance": "Cov", "covariance": "Cov", "cov": "Cov", "std": "Std", "sum": "Sum", "mean": "Mean",
    "tsmax": "Max", "tsmin": "Min", "max": "Greater", "min": "Less", "abs": "Abs", "log": "Log",
    "sign": "Sign", "decaylinear": "WMA", "count": "__count__", "highday": "__highday__",
    "lowday": "__lowday__",
}
GTJA_UNSUPPORTED = {"sma", "wma", "sumif", "regbeta", "regresi", "prod", "filter", "sequence", "sumac",
                    "smean", "ema"}

ALPHA101_FIELDS = {"open", "close", "high", "low", "volume", "vwap", "amount"}
GTJA_FIELDS = {"open", "close", "high", "low", "volume", "vwap", "amount"}


@dataclass
class Dialect:
    name: str
    funcs: dict
    unsupported: set = field(default_factory=set)
    eq_single: bool = False         # '=' is equality (GTJA)
    floor_windows: bool = False     # Alpha101: non-integer windows floored
    caret_power: bool = True


DIALECTS = {
    "qlib": Dialect("qlib", QLIB_FUNCS, {"ema", "skew", "kurt", "mad", "mask", "count", "changeinstrument",
                                         "rolling", "cumsum", "pairrolling"}),
    "alpha101": Dialect("alpha101", ALPHA101_FUNCS, ALPHA101_UNSUPPORTED, floor_windows=True),
    "gtja": Dialect("gtja", GTJA_FUNCS, GTJA_UNSUPPORTED, eq_single=True),
}


def returns_node() -> Node:
    """Daily close-to-close return macro: close / Ref(close, 1) - 1."""
    return Node("Sub", (Node("Div", (F("close"), Node("Ref", (F("close"),), (1,)))), C(1.0)))


def adv_node(d: int) -> Node:
    """Alpha101 adv{d}: average daily dollar volume over d days = Mean(volume * vwap, d)."""
    return Node("Mean", (Node("Mul", (F("volume"), F("vwap"))),), (d,))


_BINARY = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div", ".*": "Mul", "./": "Div", ">": "Gt", "<": "Lt",
           ">=": "Ge", "<=": "Le", "==": "Eq", "!=": "Ne", "&&": "And", "&": "And", "||": "Or", "|": "Or"}
_PREC = {"||": 2, "|": 2, "&&": 3, "&": 3, "==": 4, "!=": 4, "=": 4, ">": 4, "<": 4, ">=": 4, "<=": 4,
         "+": 5, "-": 5, "*": 6, "/": 6, ".*": 6, "./": 6}
_UNARY_PREC = 7
_POWER_PREC = 8


class _Parser:
    def __init__(self, tokens: list[Tok], dialect: Dialect, env: dict[str, Node], strict: bool):
        self.toks = tokens
        self.i = 0
        self.d = dialect
        self.env = env
        self.strict = strict
        self.repairs: list[str] = []

    # ------------------------------------------------------------------ token helpers
    def peek(self) -> Tok:
        return self.toks[self.i]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> Tok:
        t = self.next()
        if t.text != text:
            raise ParseError(f"expected {text!r} at {t.pos}, got {t.text!r}")
        return t

    # ------------------------------------------------------------------ pratt
    def expr(self, min_prec: int = 0) -> Node:
        left = self.unary()
        while True:
            t = self.peek()
            if t.kind != "op":
                break
            sym = t.text
            if sym == "?":
                if min_prec > 1:
                    break
                self.next()
                a = self.expr(0)
                self.expect(":")
                b = self.expr(1)
                left = Node("If", (left, a, b))
                continue
            if sym == "=" and not self.d.eq_single:
                break
            prec = _PREC.get(sym)
            if prec is None or prec < min_prec:
                break
            self.next()
            right = self.expr(prec + 1)
            opname = "Eq" if sym == "=" else _BINARY[sym]
            left = Node(opname, (left, right))
        return left

    def unary(self) -> Node:
        t = self.peek()
        if t.kind == "op" and t.text in ("-", "+", "!"):
            self.next()
            operand = self.expr(_UNARY_PREC)
            if t.text == "+":
                return operand
            if t.text == "!":
                return Node("Not", (operand,))
            if operand.is_const:
                return C(-operand.value)
            return Node("Neg", (operand,))
        return self.power()

    def power(self) -> Node:
        base = self.primary()
        t = self.peek()
        if t.kind == "op" and t.text in ("^", "**"):
            self.next()
            exponent = self.unary_for_power()
            exp_val = _const_value(exponent)
            if exp_val is None:
                raise UnsupportedOperator("power with a non-constant exponent is not supported")
            return Node("Power", (base,), (exp_val,))
        return base

    def unary_for_power(self) -> Node:
        t = self.peek()
        if t.kind == "op" and t.text in ("-", "+"):
            self.next()
            e = self.unary_for_power()
            if t.text == "-":
                v = _const_value(e)
                return C(-v) if v is not None else Node("Neg", (e,))
            return e
        return self.power()

    def primary(self) -> Node:
        t = self.next()
        if t.kind == "num":
            return C(float(t.text))
        if t.kind == "field":
            name = t.text[1:].lower()
            return self.field_node(name, t)
        if t.kind == "op" and t.text == "(":
            e = self.expr(0)
            self.expect(")")
            return e
        if t.kind == "ident":
            if self.peek().text == "(":
                return self.call(t)
            return self.identifier(t)
        raise ParseError(f"unexpected token {t.text!r} at {t.pos}")

    # ------------------------------------------------------------------ leaves
    def field_node(self, name: str, t: Tok) -> Node:
        if name in FIELDS:
            return F(name)
        if name in ("returns", "ret"):
            return returns_node()
        raise ParseError(f"unknown field {name!r} at {t.pos}")

    def identifier(self, t: Tok) -> Node:
        raw = t.text
        if raw in self.env:
            return self.env[raw]
        low = raw.lower()
        if self.d.name in ("alpha101", "gtja"):
            if low in ALPHA101_FIELDS | GTJA_FIELDS:
                return F(low)
            if low in ("returns", "ret"):
                return returns_node()
            m = re.fullmatch(r"adv(\d+)", low)
            if m:
                return adv_node(int(m.group(1)))
            if low in ("cap", "industry", "sector", "subindustry", "banchmarkindexclose", "banchmarkindexopen"):
                raise UnsupportedOperator(f"field {raw!r} is outside the OHLCV(+VWAP) panel")
        if self.d.name == "qlib" and not self.strict and low in FIELDS:
            self.repairs.append(f"bare field {raw} -> ${low}")
            return F(low)
        raise ParseError(f"unknown identifier {raw!r} at {t.pos}")

    # ------------------------------------------------------------------ calls
    def call(self, t: Tok) -> Node:
        name = t.text
        low = name.lower()
        self.expect("(")
        args: list[Node] = []
        if self.peek().text != ")":
            args.append(self.expr(0))
            while self.peek().text == ",":
                self.next()
                args.append(self.expr(0))
        self.expect(")")
        if low in self.d.unsupported:
            raise UnsupportedOperator(f"operator {name!r} is not in the supported operator set")
        target = self.d.funcs.get(low)
        if target is None:
            raise ParseError(f"unknown function {name!r} at {t.pos}")
        if target == "__qlib_rank__":
            target = "TsRank" if len(args) == 2 else "CSRank"
            if target == "CSRank":
                self.repairs.append("Rank(x) with one argument read as cross-sectional CSRank")
        elif target in ("__ts_or_elem_max__", "__ts_or_elem_min__"):
            is_ts = len(args) == 2 and _const_value(args[1]) is not None and self.d.name != "gtja"
            if self.d.name == "qlib" and not is_ts:
                if self.strict:
                    raise ParseError(f"Qlib {name}(x, n) needs a constant window; use Greater/Less for elementwise")
                self.repairs.append(f"{name}(x, y) read as elementwise {'Greater' if 'max' in target else 'Less'}")
            if is_ts:
                target = "Max" if "max" in target else "Min"
            else:
                target = "Greater" if "max" in target else "Less"
        elif target == "__count__":
            if len(args) != 2:
                raise ParseError("COUNT(cond, n) takes two arguments")
            cond = args[0]
            if cond.op not in BOOLEAN_OPS:
                cond = Node("Ne", (cond, C(0.0)))
            return Node("Sum", (cond,), (self.window(args[1]),))
        elif target in ("__highday__", "__lowday__"):
            n = self.window(args[1])
            idx = Node("IdxMax" if target == "__highday__" else "IdxMin", (args[0],), (n,))
            return Node("Sub", (C(float(n)), idx))
        return self.build(target, args, name)

    def window(self, arg: Node) -> int:
        v = _const_value(arg)
        if v is None:
            raise ParseError("window/lag parameters must be constants")
        if self.d.floor_windows:
            return int(math.floor(v))
        if abs(v - round(v)) > 1e-9:
            if self.strict:
                raise ParseError(f"non-integer window {v}")
            self.repairs.append(f"window {v} floored")
            return int(math.floor(v))
        return int(round(v))

    def build(self, opname: str, args: list[Node], raw: str) -> Node:
        s = OPS[opname]
        n_params = len(s.params)
        n_kids = s.n_children
        if len(args) == n_kids + n_params - len(s.defaults):
            args = args + [C(v) for v in s.defaults]
        if len(args) != n_kids + n_params:
            raise ParseError(f"{raw} expects {n_kids} expression(s) and {n_params} parameter(s), got {len(args)}")
        kids = tuple(args[:n_kids])
        params = []
        for pname, kind, a in zip(s.params, s.param_kinds, args[n_kids:]):
            if kind in ("lag", "lag1", "window"):
                params.append(self.window(a))
            else:
                v = _const_value(a)
                if v is None:
                    raise ParseError(f"parameter {pname} of {raw} must be a constant")
                params.append(float(v))
        return Node(opname, kids, tuple(params))


def _const_value(n: Node) -> float | None:
    if n.is_const:
        return n.value
    if n.op == "Neg" and n.children[0].is_const:
        return -n.children[0].value
    return None


_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+?)\s*$", re.S)


def _split_statements(src: str) -> list[str]:
    parts = []
    for line in re.split(r"[;\n]", src):
        if line.strip():
            parts.append(line.strip())
    return parts


@dataclass
class ParseResult:
    node: Node
    dialect: str
    repairs: list[str]


def parse_full(src: str, dialect: str = "qlib", strict: bool = True) -> ParseResult:
    if dialect not in DIALECTS:
        raise ValueError(f"unknown dialect {dialect!r}")
    d = DIALECTS[dialect]
    env: dict[str, Node] = {}
    statements = _split_statements(src) if (";" in src or "\n" in src.strip()) else [src.strip()]
    result: Node | None = None
    repairs: list[str] = []
    for k, stmt in enumerate(statements):
        m = _ASSIGN.match(stmt) if not d.eq_single else None
        if m and m.group(1).lower() not in d.funcs:
            name, body = m.group(1), m.group(2)
            p = _Parser(tokenize(body), d, env, strict)
            node = p.expr(0)
            if p.peek().kind != "eof":
                raise ParseError(f"trailing input in statement {stmt!r}")
            env[name] = node
            repairs += p.repairs
            result = node
        else:
            p = _Parser(tokenize(stmt), d, env, strict)
            node = p.expr(0)
            if p.peek().kind != "eof":
                t = p.peek()
                raise ParseError(f"trailing input {t.text!r} at {t.pos}")
            repairs += p.repairs
            result = node
    if result is None:
        raise ParseError("empty formula")
    return ParseResult(result, dialect, repairs)


def parse(src: str, dialect: str = "qlib", strict: bool = True) -> Node:
    """Parse a surface form into a :class:`Node` (raises :class:`ParseError`)."""
    return parse_full(src, dialect, strict).node


def try_parse_any(src: str, strict: bool = False) -> ParseResult:
    """Lenient multi-dialect parse used for LLM-authored formulas (P1/P2, B4 reconstruction)."""
    errors = []
    order = ["qlib", "alpha101", "gtja"] if "$" in src else ["alpha101", "qlib", "gtja"]
    if re.search(r"\b[A-Z]{3,}\(", src) and "$" not in src:
        order = ["gtja", "alpha101", "qlib"]
    for d in order:
        try:
            return parse_full(src, d, strict=strict)
        except ParseError as e:
            errors.append(f"{d}: {e}")
    raise ParseError("; ".join(errors))
