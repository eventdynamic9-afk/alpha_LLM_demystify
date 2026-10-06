"""Routes a normalized claim to its verification method (§10) and returns a Verdict.

Order for direction claims (SIGN / MONO): exact static monotonicity -> Z3 proof on small fragments ->
nudge test.  Behavioral claims run on the training window with test-window replication.  PERF claims
use the formula's recorded search (``trials``) for the DSR; "best of a set" claims (BETTER_THAN / PERF
naming a codebook ``best_of_sets`` set) go to the Romano-Wolf / RC / SPA route (§10.4).  Claims with
``polarity == "deny"`` are verified affirmatively and inverted.
"""
from __future__ import annotations

import numpy as np

from dsl import Node
from dsl.monotonicity import AMB, NEG, POS

from . import behavioral, identity, metamorphic, nudge, originality, performance, static
from .inputs import normalize_input
from .verdicts import (AMBIGUOUS, REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict, invert)

STATIC_PREDICATES = {"DEPENDS_ON", "LOOKBACK", "HORIZON", "XSEC", "RANGE", "STRUCT", "INVARIANT", "IDENTITY",
                     "VARIANT_OF"}


_NEG_WORDS = ("-", "neg", "negative", "down", "lower", "decreasing", "-1",
              "\u2212", "\u2013", "\u2193", "\u22121", "\u20131")     # U+2212 minus, en dash, down arrow


def _dir(x) -> str:
    """Claimed direction: '-' for minus / en-dash / down-arrow glyphs and words, '+' otherwise (up arrow)."""
    s = str(x).strip().lower()
    return "-" if s in _NEG_WORDS else "+"


def verify_direction(node: Node, ctx, args: dict, scope_mask=None) -> Verdict:
    spec = normalize_input(args.get("input", ""))
    want = _dir(args.get("direction", "+"))
    if spec.kind == "out_of_universe":
        return Verdict(REFUTED, "static_dependency", {"reason": f"input '{spec.field}' is not used by an OHLCV formula"})
    if spec.kind == "unknown":
        return Verdict(UNVERIFIABLE, "monotonicity", {"reason": f"unmapped input {spec.raw!r}"})
    s = static.static_direction(node, spec)
    ev = {"input": spec.raw, "claimed": want}
    fld = static.input_field(spec)
    lag = static.read_lag(node, fld) if fld else None
    if s is not None:
        ev["static"] = static.direction_symbol(s)
        if lag:
            ev["read_lag"] = lag            # field enters only through lagged leaves (Ref & co.)
        if s == 0:
            return Verdict(REFUTED, "monotonicity_static", {**ev, "reason": "formula does not depend on the input"})
        if s in (POS, NEG):
            ok = (s == POS) == (want == "+")
            return Verdict(SUPPORTED if ok else REFUTED, "monotonicity_static", ev)
        if spec.kind == "field":
            from .smt import prove_monotone

            res = prove_monotone(node, spec.field, lag or 0, want)
            ev["smt"] = res
            if res == "proved":
                return Verdict(SUPPORTED, "smt", ev)
            if res == "counterexample":
                opp = prove_monotone(node, spec.field, lag or 0, "-" if want == "+" else "+")
                ev["smt_opposite"] = opp
                if opp == "proved":
                    return Verdict(REFUTED, "smt", ev)
    v = nudge.nudge_test(node, ctx, spec, want, scope_mask=scope_mask, lag=lag or 0)
    v.evidence.update({k: val for k, val in ev.items() if k not in v.evidence})
    return v


def verify_claim(claim: dict, node: Node | None, ctx, signal: np.ndarray | None = None,
                 trials: dict | None = None) -> Verdict:
    """``trials``: the formula's recorded search (``performance.record_trials``): number of trials, the
    variance of the trial Sharpe ratios and the candidate signals, used for DSR and "best of" claims
    (§10.4).  It is never read from the claim's arguments."""
    pred = str(claim.get("predicate", "")).upper()
    args = claim.get("args") or {}
    if claim.get("ambiguous"):
        return Verdict(AMBIGUOUS, "codebook", {"reason": "claim could not be mapped without guessing"})
    if pred == "THEORY":
        return Verdict(UNVERIFIABLE, "unverifiable", {"reason": "C5 theory claims are counted, never scored"})
    if node is None and (pred in STATIC_PREDICATES or pred in ("SIGN", "MONO")):
        return Verdict(UNVERIFIABLE, "static", {"reason": "mechanistic claim without a formula"})
    scope_mask = ctx.scope_mask(claim.get("scope")) if claim.get("scope") else None
    f = signal if signal is not None else (ctx.signal(node) if node is not None else None)
    try:
        v = _route(pred, args, node, ctx, f, scope_mask, trials)
    except KeyError as exc:
        v = Verdict(UNVERIFIABLE, "dispatcher", {"reason": f"missing argument {exc}"})
    if claim.get("scope") and scope_mask is None:
        v.evidence["scope_note"] = f"scope {claim.get('scope')!r} not in codebook; verified unconditionally"
    if str(claim.get("polarity", "affirm")).lower() == "deny":
        v = invert(v)
    return v


def _route(pred: str, a: dict, node, ctx, f, scope_mask, trials: dict | None = None) -> Verdict:
    if pred == "DEPENDS_ON":
        return static.verify_depends_on(node, normalize_input(a["input"]), ctx)
    if pred in ("SIGN", "MONO"):
        return verify_direction(node, ctx, a, scope_mask)
    if pred == "LOOKBACK":
        conv = ctx.thr.get("lookback", {}).get("convention", "L_or_L+1")
        return static.verify_lookback(node, int(a["window"]), conv)
    if pred == "HORIZON":
        return static.verify_horizon(node, str(a["bin"]), ctx.cb["horizon_bins"])
    if pred == "XSEC":
        val = a.get("value", True)
        val = val if isinstance(val, bool) else str(val).lower() in ("true", "1", "yes")
        return static.verify_xsec(node, val)
    if pred == "INVARIANT":
        return metamorphic.verify_invariant(node, ctx, a.get("transform", "scale"), a.get("input", "price"))
    if pred == "RANGE":
        return static.verify_range(node, float(a["low"]), float(a["high"]), ctx)
    if pred == "STRUCT":
        return static.verify_struct(node, str(a["pattern"]))
    if pred == "RESEMBLES":
        return behavioral.verify_resembles(f, ctx, str(a["ref"]), a.get("sign", "+"), scope_mask)
    if pred == "INDEPENDENT":
        return behavioral.verify_independent(f, ctx, str(a["ref"]), scope_mask)
    if pred == "EXPOSED":
        return behavioral.verify_exposed(f, ctx, str(a.get("ref") or a["factor"]), a.get("sign", "+"), scope_mask)
    if pred == "TURNOVER":
        return behavioral.verify_turnover(f, ctx, str(a.get("level", "low")).lower(), scope_mask)
    if pred == "REGIME":
        return behavioral.verify_regime(f, ctx, str(a["regime"]), str(a.get("effect", "stronger")), scope_mask)
    if pred == "PRED_SIGN":
        return behavioral.verify_pred_sign(f, ctx, a.get("sign", "+"), a.get("horizon"), scope_mask)
    if pred == "NOVEL":
        return originality.verify_novel(f, ctx, node)
    if pred == "BETTER_THAN":
        best = performance.best_of_set(ctx, str(a["ref"]))
        if best is not None:                     # "best of" a set -> Romano-Wolf stepdown + RC / SPA (§10.4)
            return performance.verify_best_of(f, ctx, best, str(a.get("metric", "IC")), trials)
        return originality.verify_better_than(f, ctx, str(a["ref"]), str(a.get("metric", "IC")))
    if pred == "PERF":
        level = str(a.get("level", "high"))
        best = performance.best_of_set(ctx, level)
        if best is not None:
            return performance.verify_best_of(f, ctx, best, str(a.get("metric", "IC")), trials)
        return performance.verify_perf(f, ctx, str(a.get("metric", "IC")), level, trials=trials)
    if pred == "IDENTITY":
        return identity.verify_identity(node, ctx, str(a["library_id"]))
    if pred == "VARIANT_OF":
        return identity.verify_variant_of(node, ctx, str(a["template"]))
    return Verdict(UNVERIFIABLE, "dispatcher", {"reason": f"unknown predicate {pred!r}"})


__all__ = ["verify_claim", "verify_direction", "SUPPORTED", "REFUTED", "UNRESOLVED", "AMB"]
