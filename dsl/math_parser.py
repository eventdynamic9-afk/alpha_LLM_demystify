"""Parser for the plain-math (LaTeX) rendering produced by :func:`dsl.serialize.to_math`.

Used to validate SP notation variants (§7.2): the text a narrator sees is parsed back into a tree
and checked for numerical equivalence with the base formula, so a serializer bug cannot slip into
the SP pool unnoticed.  The grammar covers exactly what ``to_math`` emits.
"""
from __future__ import annotations

import re

from .ast import C, F, Node
from .parser import ParseError

_TOKEN = re.compile(r"""
    (?P<num>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)
  | (?P<cmd>\\[A-Za-z]+|\\\\|\\\|)
  | (?P<sym>[-+(){}\[\],^_&|<>=])
  | (?P<word>[A-Za-z][A-Za-z0-9]*)
""", re.X)

_NAMES = {"mean": "Mean", "sum": "Sum", "std": "Std", "var": "Var", "max": "Max", "min": "Min",
          "argmax": "IdxMax", "argmin": "IdxMin", "tsrank": "TsRank", "quantile": "Quantile", "median": "Med",
          "wma": "WMA", "slope": "Slope", "resid": "Resi", "corr": "Corr", "cov": "Cov", "lag": "Ref"}
_CS = {"rank": "CSRank", "z": "CSZScore", "scale": "CSScale"}


def _tokens(src: str) -> list[str]:
    out, i = [], 0
    while i < len(src):
        if src[i].isspace():
            i += 1
            continue
        m = _TOKEN.match(src, i)
        if not m:
            raise ParseError(f"math: unexpected character {src[i]!r} at {i}")
        out.append(m.group(0))
        i = m.end()
    return out


class _P:
    def __init__(self, src: str):
        self.t = _tokens(src)
        self.i = 0

    def peek(self, k: int = 0) -> str | None:
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def eat(self, tok: str | None = None) -> str:
        cur = self.peek()
        if cur is None or (tok is not None and cur != tok):
            raise ParseError(f"math: expected {tok!r}, got {cur!r}")
        self.i += 1
        return cur

    def group(self) -> str:
        """Raw text of a {...} group (for names and parameters)."""
        self.eat("{")
        depth, parts = 1, []
        while True:
            tok = self.eat()
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth -= 1
                if depth == 0:
                    return "".join(parts)
            parts.append(tok)

    # expr := term (('+' | '-') term)*
    def expr(self) -> Node:
        n = self.term()
        while self.peek() in ("+", "-"):
            op = "Add" if self.eat() == "+" else "Sub"
            n = Node(op, (n, self.term()))
        return n

    # term := unary (('\cdot' | juxtaposition) unary)*
    def term(self) -> Node:
        n = self.unary()
        while True:
            if self.peek() == "\\cdot":
                self.eat()
                n = Node("Mul", (n, self.unary()))
            elif self.peek() == "\\left" and self.peek(1) == "|":      # sign(x)|x|^p = SignedPower(x, p)
                m = self.unary()
                left, sgn = (n.children[0], n.children[1]) if n.op == "Mul" else (None, n)
                if (sgn.op == "Sign" and m.op == "Power" and m.children[0].op == "Abs"
                        and m.children[0].children[0] == sgn.children[0]):
                    sp = Node("SignedPower", (sgn.children[0],), m.params)
                    n = sp if left is None else Node("Mul", (left, sp))
                else:
                    n = Node("Mul", (n, m))
            else:
                return n

    def unary(self) -> Node:
        if self.peek() == "-":
            self.eat()
            x = self.unary()
            if x.is_const:
                return C(-x.value)
            return Node("Neg", (x,))
        return self.power()

    def power(self) -> Node:
        n = self.atom()
        if self.peek() == "^":
            self.eat()
            n = Node("Power", (n,), (float(self.group()),))
        return n

    def args(self) -> list[Node]:
        self.eat("(")
        out = [self.expr()]
        while self.peek() == ",":
            self.eat()
            out.append(self.expr())
        self.eat(")")
        return out

    def atom(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise ParseError("math: unexpected end")
        if re.fullmatch(r"[\d.].*", tok):
            return C(float(self.eat()))
        if tok == "(":
            self.eat()
            n = self.expr()
            self.eat(")")
            return n
        if tok == "\\left":
            self.eat()
            if self.peek() == "(":
                self.eat()
                n = self.expr()
                self.eat("\\right")
                self.eat(")")
                return n
            self.eat("|")
            n = self.expr()
            self.eat("\\right")
            self.eat("|")
            return Node("Abs", (n,))
        if tok == "\\text":
            self.eat()
            return F(self.group())
        if tok == "\\frac":
            self.eat()
            self.eat("{")
            a = self.expr()
            self.eat("}")
            self.eat("{")
            b = self.expr()
            self.eat("}")
            return Node("Div", (a, b))
        if tok == "\\ln":
            self.eat()
            return Node("Log", tuple(self.args()))
        if tok in ("\\max", "\\min"):
            self.eat()
            a = self.args()
            return Node("Greater" if tok == "\\max" else "Less", tuple(a))
        if tok == "\\mathbb":
            self.eat()
            self.group()                                    # {1}
            self.eat("[")
            if self.peek() == "\\neg":
                self.eat()
                x = self.expr()
                self.eat("]")
                return Node("Not", (x,))
            a = self.expr()
            rel = self.eat()
            if rel == ">" and self.peek() == "=":
                self.eat()
                rel = "\\ge"
            ops = {">": "Gt", "<": "Lt", "=": "Eq", "\\ge": "Ge", "\\le": "Le", "\\ne": "Ne", "\\wedge": "And",
                   "\\vee": "Or"}
            if rel not in ops:
                raise ParseError(f"math: unknown relation {rel!r}")
            b = self.expr()
            self.eat("]")
            return Node(ops[rel], (a, b))
        if tok == "\\begin":
            self.eat()
            self.group()                                    # {cases}
            x = self.expr()
            self.eat("&")
            self.eat("\\text")
            self.group()                                    # {if }
            c = self.expr()
            self.eat("\\ne")
            zero = self.atom()
            if not (zero.is_const and zero.value == 0):
                raise ParseError("math: cases condition must be '!= 0'")
            self.eat("\\\\")
            y = self.expr()
            self.eat("&")
            self.eat("\\text")
            self.group()                                    # {otherwise}
            self.eat("\\end")
            self.group()
            return Node("If", (c, x, y))
        if tok == "\\Delta":
            self.eat()
            self.eat("_")
            d = int(self.group())
            return Node("Delta", tuple(self.args()), (d,))
        if tok == "R":                                      # R^2_{n}(x)
            self.eat()
            self.eat("^")
            self.eat("2")
            self.eat("_")
            n = int(self.group())
            return Node("Rsquare", tuple(self.args()), (n,))
        if tok == "\\operatorname":
            self.eat()
            name = self.group()
            if name == "sign" and self.peek() == "(":
                return Node("Sign", tuple(self.args()))
            sub = None
            if self.peek() == "_":
                self.eat()
                sub = self.group()
            if (sub or "").startswith("\\mathrm{cs}"):
                rest = sub.split(",", 1)
                op = _CS[name]
                (x,) = self.args()
                if op == "CSScale":
                    a = float(rest[1]) if len(rest) > 1 else 1.0
                    return Node(op, (x,), (a,))
                return Node(op, (x,))
            if name not in _NAMES:
                raise ParseError(f"math: unknown operator {name!r}")
            op = _NAMES[name]
            params = []
            for p in (sub or "").split(","):
                if p:
                    params.append(int(p) if re.fullmatch(r"\d+", p) else float(p))
            return Node(op, tuple(self.args()), tuple(params))
        raise ParseError(f"math: unexpected token {tok!r}")


def parse_math(src: str) -> Node:
    p = _P(src)
    n = p.expr()
    if p.peek() is not None:
        raise ParseError(f"math: trailing tokens from {p.peek()!r}")
    return n
