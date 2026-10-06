"""Metamorphic relations (§10.2): price / volume scaling (global and per stock), log-price shift,
stock permutation (cross-sectional equivariance), date shift (time equivariance) and field ablation.
Each relation either holds exactly or yields a violation magnitude; INVARIANT claims are decided here."""
from __future__ import annotations

import numpy as np

from data.panel import Panel
from dsl import Node

from .stats import daily_spearman
from .verdicts import REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict

PRICE = ("open", "high", "low", "close", "vwap")


def _scaled(panel: Panel, fields, factors: np.ndarray) -> Panel:
    new = {f: panel.get(f) * factors for f in fields if f in panel.fields}
    return panel.with_fields(**new)


def _deviation(a: np.ndarray, b: np.ndarray, rel_tol: float) -> dict:
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    nan_mismatch = int((fa != fb).sum())
    if not both.any():
        return {"max_rel_dev": float("nan"), "violation_share": float("nan"), "nan_mismatch": nan_mismatch, "n": 0}
    den = np.maximum(np.maximum(np.abs(a[both]), np.abs(b[both])), 1e-300)
    rel = np.abs(a[both] - b[both]) / den
    rel[(a[both] == b[both])] = 0.0
    viol = (rel > rel_tol).sum() + nan_mismatch
    n = int(both.sum() + nan_mismatch)
    return {"max_rel_dev": float(rel.max()), "violation_share": float(viol / n), "nan_mismatch": nan_mismatch, "n": n}


def relation(node: Node, ctx, name: str, seed: int = 0) -> dict:
    """Run one metamorphic relation; returns deviation statistics."""
    p = ctx.panel
    ex = ctx.executor
    base = ctx.signal(node)
    rng = np.random.default_rng(seed)
    tol = ctx.thr["metamorphic"]["violation_rel_tol"]
    if name == "price_scale_global":
        other = ex.evaluate(node, _scaled(p, PRICE + ("amount",), np.float64(1.7)))
    elif name == "price_scale_per_stock":
        c = rng.uniform(0.5, 2.0, p.N)[None, :]
        other = ex.evaluate(node, _scaled(p, PRICE + ("amount",), c))
    elif name == "volume_scale_global":
        other = ex.evaluate(node, _scaled(p, ("volume", "amount"), np.float64(1.7)))
    elif name == "volume_scale_per_stock":
        c = rng.uniform(0.5, 2.0, p.N)[None, :]
        other = ex.evaluate(node, _scaled(p, ("volume", "amount"), c))
    elif name == "log_price_shift":
        other = ex.evaluate(node, _scaled(p, PRICE + ("amount",), np.float64(np.exp(0.7))))
    elif name == "stock_permutation":
        perm = rng.permutation(p.N)
        q = p.take_instruments(perm)
        out = ex.evaluate(node, q)
        other = np.empty_like(out)
        other[:, perm] = out
    elif name == "date_shift":
        s = min(17, p.T // 4)
        q = p.take_dates(np.arange(s, p.T))
        other = np.full_like(base, np.nan)
        other[s:] = ex.evaluate(node, q)
        b = base.copy()
        b[:s] = np.nan
        warm = np.isfinite(other) & np.isfinite(b)
        return _deviation(np.where(warm, b, np.nan), np.where(warm, other, np.nan), tol)
    else:
        raise ValueError(f"unknown relation {name!r}")
    return _deviation(base, other, tol)


def field_ablation(node: Node, ctx, fieldname: str, window: str = "all") -> dict:
    """Replace a field by its cross-sectional median (members) each day; magnitude = 1 - mean daily
    Spearman(original, ablated).  Used for DEPENDS_ON strength."""
    p = ctx.panel
    x = p.get(fieldname)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(p.member, x, np.nan), axis=1, keepdims=True)
    import warnings

    abl = np.where(np.isfinite(x), np.broadcast_to(med, x.shape), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = ctx.executor.evaluate(node, p.with_fields(**{fieldname: abl}))
    base = ctx.signal(node)
    rows = ctx.rows(window) if window != "all" else None
    if not np.isfinite(other).any():
        return {"magnitude": 1.0, "mean_rank_corr": float("nan")}
    rho = daily_spearman(base, other, rows)
    if not np.isfinite(rho).any():
        same = np.array_equal(np.nan_to_num(base, nan=1e300), np.nan_to_num(other, nan=1e300))
        return {"magnitude": 0.0 if same else 1.0, "mean_rank_corr": float("nan")}
    m = float(np.nanmean(rho))
    return {"magnitude": float(max(0.0, 1.0 - m)), "mean_rank_corr": m}


_TRANSFORMS = {("scale", "price"): "price_scale_per_stock", ("scale", "volume"): "volume_scale_per_stock",
               ("scale", "prices"): "price_scale_per_stock", ("shift", "log_price"): "log_price_shift",
               ("shift", "time"): "date_shift", ("permutation", "stocks"): "stock_permutation"}


def verify_invariant(node: Node, ctx, transform: str, input_: str) -> Verdict:
    key = (str(transform).lower(), str(input_).lower())
    rel = _TRANSFORMS.get(key)
    if rel is None:
        return Verdict(UNVERIFIABLE, "metamorphic", {"reason": f"no relation for {key}"})
    dev = relation(node, ctx, rel)
    cfg = ctx.thr["metamorphic"]
    ev = {"relation": rel, **dev}
    if dev["n"] == 0:
        return Verdict(UNVERIFIABLE, "metamorphic", ev)
    if dev["nan_mismatch"] == 0 and dev["max_rel_dev"] <= cfg["exact_rel_tol"]:
        return Verdict(SUPPORTED, "metamorphic", ev)
    if dev["violation_share"] > cfg["violation_share"]:
        return Verdict(REFUTED, "metamorphic", ev)
    return Verdict(UNRESOLVED, "metamorphic", ev)
