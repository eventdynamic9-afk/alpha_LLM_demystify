"""Nudge (perturbation) tests for direction / monotonicity claims (§10.2).

For N stock-date contexts sampled from the training window (stratified by year and size), the
claimed input of one stock is raised by +delta * sigma_i (sigma_i: that stock's own variability of
the input) while every other field and every other stock is held fixed; the formula is re-evaluated
on the context window and the share of signal changes in the claimed direction is bounded with an
exact Clopper-Pearson interval:

  SUPPORTED  if lower bound >= 0.90      REFUTED if upper bound <= 0.10      else UNRESOLVED

Perturbations (pre-registered):
  field x          x_t *= exp(delta * sd(dlog x))                       (latest value only)
  ret_d            close_{t-d+1+k} *= (1 + delta*sd(r_d))^((k+1)/d)    (path rescaling, k = 0..d-1)
  abn_vol          volume_t *= exp(delta * sd(dlog volume))
  volatility_n     de-meaned last-n log returns scaled by (1 + kappa), closes rebuilt
  range            high_t *= exp(delta*sd/2), low_t /= exp(delta*sd/2), sd = sd(log(high/low))
  intraday         close_t *= exp(delta * sd(log(close/open)))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsl import Node, effective_lookback
from dsl.operators import OPS, XS

from .inputs import InputSpec
from .stats import clopper_pearson
from .verdicts import REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict


def _extra(spec: InputSpec) -> int:
    if spec.kind == "ret":
        return spec.d
    if spec.kind == "volatility":
        return spec.d + 1
    return 1


def _sigma(ctx, spec: InputSpec, rows: np.ndarray) -> np.ndarray:
    p = ctx.panel
    with np.errstate(all="ignore"):
        if spec.kind in ("field", "abn_vol"):
            f = spec.field or "volume"
            x = np.log(np.where(p.get(f) > 0, p.get(f), np.nan))
            d = np.diff(x, axis=0, prepend=np.nan)
            return np.nanstd(np.where(rows[:, None], d, np.nan), axis=0)
        c = p.get("close")
        if spec.kind == "ret":
            r = c / np.vstack([np.full((spec.d, p.N), np.nan), c[:-spec.d]]) - 1
            return np.nanstd(np.where(rows[:, None], r, np.nan), axis=0)
        if spec.kind == "volatility":
            lr = np.diff(np.log(c), axis=0, prepend=np.nan)
            vol = pd.DataFrame(lr).rolling(spec.d, min_periods=spec.d).std().to_numpy()
            return np.nanstd(np.where(rows[:, None], vol, np.nan), axis=0)
        if spec.kind == "range":
            return np.nanstd(np.where(rows[:, None], np.log(p.get("high") / p.get("low")), np.nan), axis=0)
        if spec.kind == "intraday":
            return np.nanstd(np.where(rows[:, None], np.log(c / p.get("open")), np.nan), axis=0)
    raise ValueError(spec.kind)


def _perturb(fields: dict, j: int, e: int, spec: InputSpec, delta: float, sigma: float) -> bool:
    """Apply the perturbation in place to stock column j at last row e; False if not applicable."""
    if not np.isfinite(sigma) or sigma <= 0:
        return False
    if spec.kind in ("field", "abn_vol"):
        f = spec.field or "volume"
        x = fields[f]
        if not np.isfinite(x[e, j]) or x[e, j] <= 0:
            return False
        x[e, j] *= np.exp(delta * sigma)
        return True
    c = fields["close"]
    if spec.kind == "ret":
        d = spec.d
        if e - d + 1 < 0 or not np.all(np.isfinite(c[e - d:e + 1, j])):
            return False
        g = 1.0 + delta * sigma
        for k in range(d):
            c[e - d + 1 + k, j] *= g ** ((k + 1) / d)
        return True
    if spec.kind == "volatility":
        n = spec.d
        seg = c[e - n:e + 1, j]
        if e - n < 0 or not np.all(np.isfinite(seg)) or np.any(seg <= 0):
            return False
        lr = np.diff(np.log(seg))
        cur = lr.std(ddof=1)
        if cur <= 0:
            return False
        kappa = delta * sigma / cur
        lr2 = lr.mean() + (lr - lr.mean()) * (1.0 + kappa)
        c[e - n + 1:e + 1, j] = seg[0] * np.exp(np.cumsum(lr2))
        return True
    if spec.kind == "range":
        h, lo = fields["high"], fields["low"]
        if not (np.isfinite(h[e, j]) and np.isfinite(lo[e, j])):
            return False
        h[e, j] *= np.exp(delta * sigma / 2)
        lo[e, j] /= np.exp(delta * sigma / 2)
        return True
    if spec.kind == "intraday":
        if not np.isfinite(c[e, j]):
            return False
        c[e, j] *= np.exp(delta * sigma)
        return True
    return False


def sample_contexts(ctx, valid: np.ndarray, n: int, seed: int) -> list[tuple[int, int]]:
    """Stratified (year x size tercile) sample of (t, i) cells where ``valid``."""
    rng = np.random.default_rng(seed)
    years = ctx.panel.years()
    try:
        size = ctx.references.signal("SIZE_PROXY")
        pct = pd.DataFrame(np.where(valid, size, np.nan)).rank(axis=1, pct=True).to_numpy()
        terc = np.where(np.isfinite(pct), np.minimum(np.floor(np.nan_to_num(pct) * 3), 2), 1).astype(int)
    except Exception:
        terc = np.ones_like(valid, dtype=int)
    cells: dict[tuple, list] = {}
    ts, js = np.nonzero(valid)
    for t, j in zip(ts, js):
        cells.setdefault((years[t], terc[t, j]), []).append((t, j))
    total = sum(len(v) for v in cells.values())
    if total == 0:
        return []
    out = []
    for key in sorted(cells):
        v = cells[key]
        k = max(1, int(round(n * len(v) / total)))
        pick = rng.choice(len(v), size=min(k, len(v)), replace=False)
        out += [v[i] for i in pick]
    if len(out) > n:
        idx = rng.choice(len(out), size=n, replace=False)
        out = [out[i] for i in sorted(idx)]
    return out


def nudge_test(node: Node, ctx, spec: InputSpec, direction: str, window: str = "train",
               scope_mask: np.ndarray | None = None, delta_sigma: float | None = None,
               n_contexts: int | None = None, bounds: tuple[float, float] | None = None,
               seed: int | None = None) -> Verdict:
    cfg = ctx.thr["nudge"]
    delta = cfg["delta_sigma"] if delta_sigma is None else delta_sigma
    n = ctx.n_contexts() if n_contexts is None else n_contexts
    lo_b, hi_b = bounds if bounds else (cfg["supported_lower"], cfg["refuted_upper"])
    seed = ctx.seed if seed is None else seed
    if spec.kind in ("unknown", "out_of_universe"):
        return Verdict(UNVERIFIABLE, "nudge", {"reason": f"cannot perturb input {spec.raw!r}"})
    p = ctx.panel
    sig = ctx.signal(node)
    rows = ctx.rows(window)
    need = effective_lookback(node) + _extra(spec) + 1
    valid = np.isfinite(sig) & p.member & rows[:, None]
    valid[:need] = False
    if scope_mask is not None:
        valid &= scope_mask
    contexts = sample_contexts(ctx, valid, n, seed)
    if not contexts:
        return Verdict(UNVERIFIABLE, "nudge", {"reason": "no valid contexts"})
    sigma = _sigma(ctx, spec, rows)
    has_xs = any(not m.is_leaf and OPS[m.op].kind == XS for m in _walk(node))
    by_date: dict[int, list[int]] = {}
    for t, j in contexts:
        by_date.setdefault(t, []).append(j)
    diffs = []
    for t, js in by_date.items():
        lo = t - need
        sub = p.take_dates(np.arange(lo, t + 1))
        e = sub.T - 1
        base = ctx.executor.evaluate(node, sub)[e]
        groups = [js] if not has_xs else [[j] for j in js]
        for g in groups:
            fields = {k: v.copy() for k, v in sub.fields.items()}
            applied = [j for j in g if _perturb(fields, j, e, spec, delta, float(sigma[j]))]
            if not applied:
                continue
            pert = ctx.executor.evaluate(node, sub.with_fields(**fields))[e]
            for j in applied:
                if np.isfinite(base[j]) and np.isfinite(pert[j]):
                    diffs.append(pert[j] - base[j])
    d = np.array(diffs)
    if d.size == 0:
        return Verdict(UNVERIFIABLE, "nudge", {"reason": "perturbation not applicable in any context"})
    scale = max(1e-12, float(np.nanmedian(np.abs(sig[np.isfinite(sig)]))) * 1e-12)
    nz = np.abs(d) > scale
    k_dir = int(((d > scale) if direction == "+" else (d < -scale)).sum())
    n_nz = int(nz.sum())
    ev = {"n_contexts": int(d.size), "n_nonzero": n_nz, "share_zero": float(1 - n_nz / d.size), "delta_sigma": delta,
          "input": spec.raw, "kind": spec.kind, "claimed_direction": direction}
    if n_nz == 0:
        return Verdict(REFUTED, "nudge", {**ev, "reason": "signal does not respond to the input"})
    lo, hi = clopper_pearson(k_dir, n_nz, cfg["confidence"])
    ev.update({"share_in_claimed_direction": k_dir / n_nz, "cp_ci": [lo, hi]})
    if lo >= lo_b:
        return Verdict(SUPPORTED, "nudge", ev)
    if hi <= hi_b:
        return Verdict(REFUTED, "nudge", ev)
    return Verdict(UNRESOLVED, "nudge", {**ev, "reason": "property holds only conditionally"})


def _walk(node: Node):
    from dsl import walk

    return walk(node)
