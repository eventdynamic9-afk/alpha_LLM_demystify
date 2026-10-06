"""Typed operator specification (§6.2).

Every operator carries: arity, parameter names/kinds (windows, lags, levels), time-series vs
cross-sectional vs pointwise nature, monotonicity per argument, output-range rule, warm-up rule,
unit rule, NaN behaviour and its names in each notation.  The static analysers in
:mod:`dsl.monotonicity`, :mod:`dsl.ranges` and :mod:`dsl.units` interpret the declarative codes
below; both executors implement exactly the semantics written in ``semantics``.

Monotonicity codes (per expression child):
  "+"        non-decreasing                    "-"     non-increasing
  "±"        non-monotone                       "mul"   sign follows the other factor's sign
  "div_num"  sign of the denominator            "div_den" minus the sign of the numerator
  "abs"      sign of the child's value range    "pow"   depends on exponent and domain
  "bool"     monotone if the child is boolean-valued, else non-monotone
  "lagsum"   non-decreasing in every lag of the window      "lagamb" non-monotone in every lag
  "tsrank"   + in the current value, - in past values        "delta" + current, - lag d
  "slope"/"resi" exact sign of the OLS weight per lag         "shift" lag shift by d
"""
from __future__ import annotations

from dataclasses import dataclass, field

POINT, TS, XS = "point", "ts", "xs"


@dataclass(frozen=True)
class OpSpec:
    name: str
    n_children: int
    params: tuple = ()                  # parameter names
    param_kinds: tuple = ()             # lag (int >= 0), lag1 (int >= 1), window (int >= min_window), float, level (0<q<1)
    kind: str = POINT
    mono: tuple = ()                    # one code per child (see module docstring)
    range_rule: str = "real"
    unit_rule: str = "same"
    commutative: bool = False
    min_window: int = 1
    nonmonotone: bool = False           # counted in complexity descriptors (§6.6)
    boolean: bool = False               # output is 0/1
    infix: str | None = None
    precedence: int = 0
    qlib: str | None = None             # name in the Qlib expression language (None = DSL extension)
    alpha101: str | None = None
    gtja: str | None = None
    description: str = ""
    semantics: str = ""
    defaults: tuple = field(default=())  # default values for trailing params


def _ts(name, mono, rng, unit, desc, sem, qlib=None, alpha=None, gtja=None, min_window=1, nonmono=False,
        params=("n",), kinds=("window",), n_children=1):
    return OpSpec(name, n_children, params, kinds, TS, mono, rng, unit, min_window=min_window,
                  nonmonotone=nonmono, qlib=qlib if qlib is not None else name, alpha101=alpha, gtja=gtja,
                  description=desc, semantics=sem)


_WIN_NAN = "NaN unless all n window values are non-NaN (warm-up n-1)"

OPS: dict[str, OpSpec] = {}


def _reg(spec: OpSpec) -> None:
    OPS[spec.name] = spec


# ------------------------------------------------------------------------------------- pointwise
_reg(OpSpec("Add", 2, kind=POINT, mono=("+", "+"), range_rule="add", unit_rule="same", commutative=True,
            infix="+", precedence=5, qlib="Add", alpha101="+", gtja="+", description="x + y",
            semantics="elementwise sum; NaN propagates"))
_reg(OpSpec("Sub", 2, kind=POINT, mono=("+", "-"), range_rule="sub", unit_rule="same", infix="-", precedence=5,
            qlib="Sub", alpha101="-", gtja="-", description="x - y", semantics="elementwise difference"))
_reg(OpSpec("Mul", 2, kind=POINT, mono=("mul", "mul"), range_rule="mul", unit_rule="mul", commutative=True,
            infix="*", precedence=6, qlib="Mul", alpha101="*", gtja="*", description="x * y",
            semantics="elementwise product"))
_reg(OpSpec("Div", 2, kind=POINT, mono=("div_num", "div_den"), range_rule="div", unit_rule="div", infix="/",
            precedence=6, qlib="Div", alpha101="/", gtja="/", description="x / y",
            semantics="elementwise ratio; NaN where y == 0"))
_reg(OpSpec("Neg", 1, kind=POINT, mono=("-",), range_rule="neg", unit_rule="same", qlib=None, alpha101="-",
            gtja="-", description="-x", semantics="negation"))
_reg(OpSpec("Abs", 1, kind=POINT, mono=("abs",), range_rule="abs", unit_rule="same", nonmonotone=True,
            qlib="Abs", alpha101="abs", gtja="ABS", description="absolute value |x|", semantics="|x|"))
_reg(OpSpec("Sign", 1, kind=POINT, mono=("+",), range_rule="sign", unit_rule="dimensionless", qlib="Sign",
            alpha101="sign", gtja="SIGN", description="sign of x (-1, 0, +1)", semantics="step function"))
_reg(OpSpec("Log", 1, kind=POINT, mono=("+",), range_rule="log", unit_rule="log", qlib="Log", alpha101="log",
            gtja="LOG", description="natural logarithm", semantics="ln(x); NaN for x <= 0"))
_reg(OpSpec("Power", 1, ("p",), ("float",), POINT, ("pow",), "power", "power", qlib="Power", alpha101="^",
            gtja="^", description="x raised to the constant power p",
            semantics="x**p; NaN for negative x with non-integer p"))
_reg(OpSpec("SignedPower", 1, ("p",), ("float",), POINT, ("spow",), "spower", "power", qlib=None,
            alpha101="signedpower", description="sign(x) * |x|**p", semantics="sign(x)*|x|**p"))
_reg(OpSpec("Greater", 2, kind=POINT, mono=("+", "+"), range_rule="max", unit_rule="same", commutative=True,
            qlib="Greater", alpha101="max", gtja="MAX", description="elementwise maximum of x and y",
            semantics="max(x, y); NaN if either is NaN"))
_reg(OpSpec("Less", 2, kind=POINT, mono=("+", "+"), range_rule="min", unit_rule="same", commutative=True,
            qlib="Less", alpha101="min", gtja="MIN", description="elementwise minimum of x and y",
            semantics="min(x, y); NaN if either is NaN"))
for _n, _sym, _mono, _desc in (("Gt", ">", ("+", "-"), "1 if x > y else 0"),
                               ("Ge", ">=", ("+", "-"), "1 if x >= y else 0"),
                               ("Lt", "<", ("-", "+"), "1 if x < y else 0"),
                               ("Le", "<=", ("-", "+"), "1 if x <= y else 0"),
                               ("Eq", "==", ("±", "±"), "1 if x == y else 0"),
                               ("Ne", "!=", ("±", "±"), "1 if x != y else 0")):
    _reg(OpSpec(_n, 2, kind=POINT, mono=_mono, range_rule="bool", unit_rule="compare",
                commutative=_n in ("Eq", "Ne"), nonmonotone=_n in ("Eq", "Ne"), boolean=True, infix=_sym,
                precedence=4, qlib=_n, alpha101=_sym, gtja=_sym, description=_desc,
                semantics="comparison returning 1.0/0.0; NaN if either input is NaN"))
_reg(OpSpec("And", 2, kind=POINT, mono=("bool", "bool"), range_rule="bool", unit_rule="logic", commutative=True,
            boolean=True, infix="&&", precedence=3, qlib="And", alpha101="&&", gtja="&&",
            description="logical and", semantics="1.0 if both non-zero; NaN propagates"))
_reg(OpSpec("Or", 2, kind=POINT, mono=("bool", "bool"), range_rule="bool", unit_rule="logic", commutative=True,
            boolean=True, infix="||", precedence=2, qlib="Or", alpha101="||", gtja="||",
            description="logical or", semantics="1.0 if either non-zero; NaN propagates"))
_reg(OpSpec("Not", 1, kind=POINT, mono=("notbool",), range_rule="bool", unit_rule="logic", boolean=True,
            qlib="Not", alpha101="!", gtja="!", description="logical not", semantics="1.0 if zero"))
_reg(OpSpec("If", 3, kind=POINT, mono=("±", "+", "+"), range_rule="if", unit_rule="if", nonmonotone=True,
            qlib="If", alpha101="?:", gtja="?:", description="if cond then x else y",
            semantics="where(cond != 0, x, y); NaN if cond is NaN"))

# ----------------------------------------------------------------------------------- time series
_reg(OpSpec("Ref", 1, ("d",), ("lag",), TS, ("shift",), "same", "same", qlib="Ref", alpha101="delay",
            gtja="DELAY", description="value of x d days ago (lag)",
            semantics="x shifted by d rows; first d rows NaN"))
_reg(OpSpec("Delta", 1, ("d",), ("lag1",), TS, ("delta",), "delta", "same", qlib="Delta", alpha101="delta",
            gtja="DELTA", description="x minus its value d days ago", semantics="x - Ref(x, d)"))
_reg(_ts("Mean", ("lagsum",), "same", "same", "moving average of x over the past n days", _WIN_NAN,
         alpha="mean", gtja="MEAN"))
_reg(_ts("Sum", ("lagsum",), "sum", "same", "moving sum of x over the past n days", _WIN_NAN, alpha="sum", gtja="SUM"))
_reg(_ts("Std", ("lagamb",), "nonneg", "same", "rolling sample standard deviation (ddof=1) over n days",
         _WIN_NAN, alpha="stddev", gtja="STD", min_window=2, nonmono=True))
_reg(_ts("Var", ("lagamb",), "nonneg", "square", "rolling sample variance (ddof=1) over n days", _WIN_NAN,
         min_window=2, nonmono=True))
_reg(_ts("Max", ("lagsum",), "same", "same", "rolling maximum of x over the past n days", _WIN_NAN,
         alpha="ts_max", gtja="TSMAX"))
_reg(_ts("Min", ("lagsum",), "same", "same", "rolling minimum of x over the past n days", _WIN_NAN,
         alpha="ts_min", gtja="TSMIN"))
_reg(_ts("IdxMax", ("lagamb",), "idx", "dimensionless",
         "position (1 = oldest, n = today) of the maximum of x within the past n days", _WIN_NAN,
         alpha="ts_argmax", nonmono=True))
_reg(_ts("IdxMin", ("lagamb",), "idx", "dimensionless",
         "position (1 = oldest, n = today) of the minimum of x within the past n days", _WIN_NAN,
         alpha="ts_argmin", nonmono=True))
_reg(_ts("TsRank", ("tsrank",), "unit01", "dimensionless",
         "time-series percentile rank of today's x within the past n days, in (0, 1]",
         "average rank of the last value among the window divided by n (= Qlib Rank(x, n))",
         qlib="Rank", alpha="ts_rank", gtja="TSRANK"))
_reg(OpSpec("Quantile", 1, ("n", "q"), ("window", "level"), TS, ("lagsum",), "same", "same", qlib="Quantile",
            description="rolling q-quantile of x over the past n days (linear interpolation)", semantics=_WIN_NAN))
_reg(_ts("Med", ("lagsum",), "same", "same", "rolling median of x over the past n days", _WIN_NAN))
_reg(_ts("WMA", ("lagsum",), "same", "same",
         "linearly decaying weighted average over n days (weights n..1, newest heaviest, sum to 1)", _WIN_NAN,
         alpha="decay_linear", gtja="DECAYLINEAR"))
_reg(_ts("Slope", ("slope",), "real", "same", "OLS slope of x on time over the past n days", _WIN_NAN,
         min_window=2, nonmono=True))
_reg(_ts("Rsquare", ("lagamb",), "unit01c", "dimensionless", "R-squared of x regressed on time over n days",
         _WIN_NAN + "; NaN if x is constant in the window; n >= 3 (a 2-point fit is identically 1)",
         min_window=3, nonmono=True))
_reg(_ts("Resi", ("resi",), "real", "same", "residual of today's x from its n-day linear time trend",
         _WIN_NAN + "; n >= 3 (a 2-point fit has zero residual)", min_window=3, nonmono=True))
_reg(_ts("Corr", ("lagamb", "lagamb"), "corr", "dimensionless",
         "rolling Pearson correlation of x and y over n days",
         _WIN_NAN + "; NaN if either window is constant", alpha="correlation", gtja="CORR", min_window=2,
         nonmono=True, n_children=2))
_reg(_ts("Cov", ("lagamb", "lagamb"), "real", "product", "rolling sample covariance (ddof=1) of x and y over n days",
         _WIN_NAN, alpha="covariance", gtja="COVIANCE", min_window=2, nonmono=True, n_children=2))

# -------------------------------------------------------------------------------- cross-sectional
_reg(OpSpec("CSRank", 1, kind=XS, mono=("+",), range_rule="unit01", unit_rule="dimensionless", qlib=None,
            alpha101="rank", gtja="RANK", description="cross-sectional percentile rank among stocks today, in (0, 1]",
            semantics="average rank / count over universe members with non-NaN values"))
_reg(OpSpec("CSZScore", 1, kind=XS, mono=("+",), range_rule="real", unit_rule="dimensionless", qlib=None,
            description="cross-sectional z-score among stocks today",
            semantics="(x - mean) / std (ddof=1) over members; NaN if std == 0"))
_reg(OpSpec("CSScale", 1, ("a",), ("float",), XS, ("+",), "scale", "dimensionless", qlib=None, alpha101="scale",
            description="cross-sectional rescaling so that sum of |x| equals a (default 1)",
            semantics="a * x / sum(|x|) over members; NaN if sum is 0", defaults=(1.0,)))

POINTWISE = tuple(n for n, s in OPS.items() if s.kind == POINT)
TIME_SERIES = tuple(n for n, s in OPS.items() if s.kind == TS)
CROSS_SECTIONAL = tuple(n for n, s in OPS.items() if s.kind == XS)
BOOLEAN_OPS = tuple(n for n, s in OPS.items() if s.boolean)


def spec(name: str) -> OpSpec:
    try:
        return OPS[name]
    except KeyError as exc:
        raise KeyError(f"unknown operator {name!r}") from exc


def warmup(node) -> int:
    """Own warm-up contribution of a node: lag d for Ref/Delta, n-1 for windowed operators."""
    if node.is_leaf:
        return 0
    s = OPS[node.op]
    if s.kind != TS:
        return 0
    if node.op in ("Ref", "Delta"):
        return int(node.params[0])
    return int(node.params[0]) - 1


def window_params(node) -> list[int]:
    """Window/lag parameters of a node (used for SA-window perturbations and complexity)."""
    if node.is_leaf or OPS[node.op].kind != TS:
        return []
    return [int(node.params[0])]


def operator_glossary(names, notation: str = "qlib") -> str:
    """One entry per operator used, in the requested notation (§8.1 glossary)."""
    out = []
    for n in names:
        s = OPS[n]
        label = {"qlib": s.qlib or n, "alpha101": s.alpha101 or n, "gtja": s.gtja or n}.get(notation, n)
        if n == "TsRank" and notation == "qlib":
            label = "TsRank"
        if n == "Neg":
            continue
        if s.infix:
            out.append(f"{s.infix}: {s.description}")
        else:
            names_ = {1: ["x"], 2: ["x", "y"], 3: ["cond", "x", "y"]}[s.n_children] if s.n_children else []
            args = ", ".join(names_ + list(s.params))
            out.append(f"{label}({args}): {s.description}")
    return "; ".join(out)
