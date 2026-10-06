"""Random typed-tree sampling from the DSL grammar (used by P3b random-grammar pools, GP
initialization, the 500-formula executor-agreement test and property-based tests)."""
from __future__ import annotations

import random

from .ast import C, F, Node
from .operators import OPS

DEFAULT_FIELDS = ("open", "high", "low", "close", "vwap", "volume")
DEFAULT_WINDOWS = (2, 3, 5, 10, 20, 30, 60)
DEFAULT_LAGS = (1, 2, 3, 5, 10, 20)
DEFAULT_CONSTS = (0.5, 1.0, 2.0, 0.001, 10.0, -1.0)

# Operator weights: emphasise the operators that dominate published alpha libraries.
DEFAULT_WEIGHTS = {
    "Add": 3, "Sub": 4, "Mul": 3, "Div": 5, "Neg": 1, "Abs": 1, "Sign": 1, "Log": 1, "Power": 0.5,
    "SignedPower": 0.3, "Greater": 0.5, "Less": 0.5, "Gt": 0.4, "Lt": 0.4, "If": 0.3,
    "Ref": 3, "Delta": 3, "Mean": 4, "Sum": 2, "Std": 3, "Var": 0.5, "Max": 2, "Min": 2, "IdxMax": 0.7,
    "IdxMin": 0.7, "TsRank": 2, "Quantile": 0.5, "Med": 0.5, "WMA": 1, "Slope": 0.7, "Rsquare": 0.5,
    "Resi": 0.5, "Corr": 3, "Cov": 1, "CSRank": 4, "CSZScore": 1, "CSScale": 0.5,
}


def _params(name: str, rng: random.Random, windows, lags) -> tuple:
    s = OPS[name]
    out = []
    for kind in s.param_kinds:
        if kind == "lag":
            out.append(rng.choice(lags))
        elif kind == "lag1":
            out.append(rng.choice(lags))
        elif kind == "window":
            out.append(max(rng.choice(windows), s.min_window))
        elif kind == "level":
            out.append(rng.choice((0.2, 0.5, 0.8)))
        else:  # float: exponents / scale
            out.append(1.0 if name == "CSScale" else rng.choice((0.5, 2.0, 3.0)))
    return tuple(out)


def random_tree(rng: random.Random, max_depth: int = 4, p_leaf: float = 0.25, fields=DEFAULT_FIELDS,
                weights: dict | None = None, windows=DEFAULT_WINDOWS, lags=DEFAULT_LAGS,
                consts=DEFAULT_CONSTS, p_const: float = 0.15) -> Node:
    weights = weights or DEFAULT_WEIGHTS
    names = [n for n in weights if weights[n] > 0]
    w = [weights[n] for n in names]

    def leaf(allow_const: bool) -> Node:
        if allow_const and rng.random() < p_const:
            return C(rng.choice(consts))
        return F(rng.choice(fields))

    def gen(d: int, allow_const: bool) -> Node:
        if d <= 1 or (d < max_depth and rng.random() < p_leaf):
            return leaf(allow_const)
        name = rng.choices(names, w)[0]
        s = OPS[name]
        if s.n_children == 1:
            kids = (gen(d - 1, False),)
        elif s.n_children == 2:
            a = gen(d - 1, False)
            b = gen(d - 1, s.kind == "point" and name not in ("Corr", "Cov"))
            kids = (a, b) if rng.random() < 0.5 or b.is_const else (b, a)
            if all(k.is_const for k in kids):
                kids = (gen(d - 1, False), kids[1])
        else:  # If
            kids = (Node(rng.choice(("Gt", "Lt")), (gen(d - 1, False), gen(d - 1, True))),
                    gen(d - 1, False), gen(d - 1, True))
        return Node(name, kids, _params(name, rng, windows, lags))

    return gen(max_depth, False)
