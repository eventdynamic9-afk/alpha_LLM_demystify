"""Static (exact) verification of mechanistic claims (§10.2): dependency set, effective lookback,
horizon bins, cross-sectional structure, output range, structure patterns and the static part of
direction claims (monotonicity abstract interpretation)."""
from __future__ import annotations

import numpy as np

from dsl import Node, canonicalize, effective_lookback, walk
from dsl.monotonicity import AMB, NEG, POS, field_direction, sign_map
from dsl.operators import OPS, XS
from dsl.ranges import interval

from .inputs import InputSpec
from .verdicts import REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict


def dependency_set(node: Node) -> set[str]:
    """Leaf fields of the AST after constant folding (canonical form)."""
    return {n.name for n in walk(canonicalize(node)) if n.is_field}


def window_parameters(node: Node) -> set[int]:
    return {int(n.params[0]) for n in walk(node) if not n.is_leaf and OPS[n.op].kind == "ts"}


def verify_depends_on(node: Node, spec: InputSpec, ctx=None) -> Verdict:
    if spec.kind == "out_of_universe":
        return Verdict(REFUTED, "static_dependency",
                       {"reason": f"input '{spec.field}' is not in the OHLCV(+VWAP) panel; the formula provably does not use it"})
    if spec.kind == "unknown":
        return Verdict(UNVERIFIABLE, "static_dependency", {"reason": f"unmapped input {spec.raw!r}"})
    deps = dependency_set(node)
    need = set(spec.fields)
    present = need & deps
    if not present:
        return Verdict(REFUTED, "static_dependency", {"dependency_set": sorted(deps), "claimed": sorted(need)})
    ev = {"dependency_set": sorted(deps), "claimed": sorted(need)}
    if ctx is not None:
        from .metamorphic import field_ablation

        mags = {f: field_ablation(node, ctx, f)["magnitude"] for f in present}
        ev["ablation_magnitude"] = mags
        if all(m <= 1e-9 for m in mags.values()):
            return Verdict(REFUTED, "static_dependency+ablation", ev)
    return Verdict(SUPPORTED, "static_dependency", ev)


def lookback_matches(L: int, window: int, convention: str = "L_or_L+1") -> bool:
    """Pre-registered LOOKBACK convention (thresholds.yaml ``lookback.convention``): the claimed n is
    compared with the effective lookback L (lags d + windows n - 1), its span L + 1 (days including
    today), or either."""
    return window in {"L": (L,), "L+1": (L + 1,), "L_or_L+1": (L, L + 1)}[convention]


def uses_window(node: Node, window: int) -> bool:
    """Whether some time-series operator has window/lag parameter ``window`` (evidence only; it does
    not decide LOOKBACK claims, §10.2)."""
    return window in window_parameters(node)


def verify_lookback(node: Node, window: int, convention: str = "L_or_L+1") -> Verdict:
    """§10.2: compare the claimed n with the effective lookback exactly; SUPPORTED iff n matches L under
    the pre-registered convention, REFUTED otherwise.  Window parameters are reported as evidence only."""
    L = effective_lookback(node)
    params = window_parameters(node)
    ev = {"effective_lookback": L, "span_days": L + 1, "convention": convention, "window_params": sorted(params),
          "claimed": window, "claimed_is_window_param": window in params}
    return Verdict(SUPPORTED if lookback_matches(L, window, convention) else REFUTED, "static_lookback", ev)


def horizon_bin_of(lookback: int, bins: dict) -> list[str]:
    return [name for name, (lo, hi) in bins.items() if lo <= lookback <= hi]


def verify_horizon(node: Node, bin_name: str, bins: dict) -> Verdict:
    """§10.2 / §10.6: the effective lookback L itself is binned (<= 5 / <= 21 / 22-126 / > 126 days)."""
    L = effective_lookback(node)
    if bin_name not in bins:
        return Verdict(UNVERIFIABLE, "static_lookback_bins", {"reason": f"unknown horizon bin {bin_name!r}"})
    lo, hi = bins[bin_name]
    ok = lo <= L <= hi
    return Verdict(SUPPORTED if ok else REFUTED, "static_lookback_bins",
                   {"effective_lookback": L, "span_days": L + 1, "claimed_bin": bin_name, "bin": [lo, hi],
                    "bins_of_lookback": horizon_bin_of(L, bins)})


def xsec_coverage(node: Node) -> tuple[int, int]:
    """(# leaf paths passing through a cross-sectional operator, # leaf paths)."""
    def go(n: Node, under: bool) -> tuple[int, int]:
        if n.is_const:
            return (0, 0)
        if n.is_field:
            return (1 if under else 0, 1)
        u = under or OPS[n.op].kind == XS
        a = b = 0
        for c in n.children:
            x, y = go(c, u)
            a, b = a + x, b + y
        return a, b

    return go(node, False)


def verify_xsec(node: Node, value: bool) -> Verdict:
    hit, total = xsec_coverage(node)
    ev = {"leaf_paths_through_xs": hit, "leaf_paths": total}
    if hit == total and total > 0:
        state = True
    elif hit == 0:
        state = False
    else:
        return Verdict(UNRESOLVED, "static_xsec", {**ev, "reason": "only part of the formula is cross-sectional"})
    return Verdict(SUPPORTED if state == bool(value) else REFUTED, "static_xsec", ev)


def verify_range(node: Node, lo: float, hi: float, ctx=None, tol: float = 1e-9) -> Verdict:
    iv = interval(node)
    ev = {"static_range": [iv.lo, iv.hi], "claimed": [lo, hi]}
    if iv.within(lo, hi):
        return Verdict(SUPPORTED, "static_range", ev)
    if iv.disjoint(lo, hi):
        return Verdict(REFUTED, "static_range", ev)
    if ctx is not None:
        v = ctx.signal(node)
        v = v[np.isfinite(v)]
        if v.size:
            ev["empirical_range"] = [float(v.min()), float(v.max())]
            if v.min() < lo - tol or v.max() > hi + tol:
                return Verdict(REFUTED, "static_range+empirical", ev)
    return Verdict(UNRESOLVED, "static_range+empirical", ev)


# ------------------------------------------------------------------------------------- STRUCT
def _is_ma(n: Node) -> bool:
    return n.op in ("Mean", "WMA")


def _match_pattern(n: Node, pattern: str) -> bool:
    p = pattern.replace(" ", "").lower()
    if p == "ratio(ma_s,ma_l)":
        if n.op == "Div" and _is_ma(n.children[0]) and _is_ma(n.children[1]):
            return n.children[0].params[0] < n.children[1].params[0]
        if n.op == "Div" and (n.children[1].is_field and _is_ma(n.children[0])):
            return True          # MA_l / x_t with x_t the 1-day "short" average
        return False
    if p == "diff(ma_s,ma_l)":
        if n.op == "Add" and len(n.children) == 2:
            a, b = n.children
            for x, y in ((a, b), (b, a)):
                if _is_ma(x) and y.op == "Neg" and _is_ma(y.children[0]):
                    return True
        return False
    if p == "zscore":
        if n.op == "Div" and n.children[1].op == "Std" and n.children[0].op == "Add":
            return any(c.op == "Neg" and _is_ma(c.children[0]) for c in n.children[0].children)
        return n.op == "CSZScore"
    if p == "corr":
        return n.op == "Corr"
    if p == "roc":
        return (n.op == "Div" and n.children[1].op == "Ref") or n.op == "Delta" or (
            n.op == "Add" and any(c.op == "Neg" and c.children[0].op == "Ref" for c in n.children))
    if p == "range_position":
        return n.op == "Div" and n.children[1].op in ("Add",) and any(
            c.op == "Max" for c in walk(n.children[1])) and any(c.op == "Min" for c in walk(n.children[1]))
    if p == "rank":
        return n.op == "CSRank"
    if p == "ts_rank":
        return n.op == "TsRank"
    return False


_PATTERN_OPS = {"ratio(ma_s,ma_l)": {"Mean", "WMA"}, "diff(ma_s,ma_l)": {"Mean", "WMA"}, "zscore": {"Std", "CSZScore"},
                "corr": {"Corr"}, "roc": {"Ref", "Delta"}, "range_position": {"Max", "Min"}, "rank": {"CSRank"},
                "ts_rank": {"TsRank"}}


def verify_struct(node: Node, pattern: str) -> Verdict:
    canon = canonicalize(node)
    key = pattern.replace(" ", "").lower()
    if key not in _PATTERN_OPS:
        return Verdict(UNVERIFIABLE, "ast_pattern", {"reason": f"pattern {pattern!r} not in the codebook"})
    if any(_match_pattern(n, pattern) for n in walk(canon)) or any(_match_pattern(n, pattern) for n in walk(node)):
        return Verdict(SUPPORTED, "ast_pattern", {"pattern": pattern})
    ops = {n.op for n in walk(node) if not n.is_leaf}
    if not (_PATTERN_OPS[key] & ops):
        return Verdict(REFUTED, "ast_pattern", {"pattern": pattern, "reason": "required operators absent"})
    return Verdict(UNRESOLVED, "ast_pattern", {"pattern": pattern, "reason": "operators present, pattern not matched"})


# ------------------------------------------------------------------------------------- directions
def input_field(spec: InputSpec) -> str | None:
    """Panel field whose own value a direction claim is about (None for derived quantities)."""
    return spec.field if spec.kind == "field" else "volume" if spec.kind == "abn_vol" else None


def read_lag(node: Node, fieldname: str) -> int | None:
    """Most recent lag at which the formula reads ``fieldname`` (0 = today's value); None if never."""
    lags = [lag for (f, lag) in sign_map(node) if f == fieldname]
    return min(lags) if lags else None


def static_direction(node: Node, spec: InputSpec) -> int | None:
    """Exact direction of the formula w.r.t. raising the input's most recent value that the formula reads
    (POS/NEG): the current value, or the most recent lagged leaf when the field enters only through
    ``Ref`` & co. (§10.2: ``Ref`` preserves).  0 only if the field is absent from the dependency set;
    AMB/None if static analysis cannot decide (-> SMT / nudge test)."""
    f = input_field(spec)
    if f is None:
        return None
    if f not in dependency_set(node):
        return 0
    lag = read_lag(node, f)
    return AMB if lag is None else field_direction(node, f, (lag,))


def direction_symbol(s: int) -> str:
    return {POS: "+", NEG: "-", AMB: "±", 0: "0"}.get(s, "?")
