"""Behavioral claims (C2, §10.3): resemblance, independence (TOST), exposure, turnover, regime
dependence and predictive direction.  Primary window = training window; the test window is the
replication — a decided verdict that flips between windows is tagged ``regime_dependent``."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import decile_long_short
from .stats import (bootstrap_mean_ci, daily_spearman, hac_ols, lag1_rank_autocorr, newey_west_mean, nw_lags,
                    tost_equivalent)
from .verdicts import (REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict, aggregate_any_all)

_SIGN = {"+": 1, "-": -1, "pos": 1, "neg": -1, 1: 1, -1: -1}


def _sgn(s) -> int:
    return _SIGN.get(s, 1)


def _masked(f: np.ndarray, scope_mask: np.ndarray | None) -> np.ndarray:
    return f if scope_mask is None else np.where(scope_mask, f, np.nan)


def _windows(ctx) -> list[str]:
    return [w for w in ("train", "test") if ctx.has_window(w)]


def _replicate(ctx, fn) -> Verdict:
    """Run ``fn(window) -> Verdict`` on train (primary) and test (replication)."""
    wins = _windows(ctx)
    if not wins:
        return Verdict(UNVERIFIABLE, "behavioral", {"reason": "no usable window"})
    primary = fn(wins[0])
    if len(wins) > 1:
        rep = fn(wins[1])
        primary.evidence["replication"] = {"window": wins[1], "verdict": rep.verdict, **{
            k: v for k, v in rep.evidence.items() if k in ("rho_bar", "ci95", "t", "beta", "mean")}}
        if primary.decidable and rep.decidable and primary.verdict != rep.verdict:
            primary.regime_dependent = True
    primary.evidence["window"] = wins[0]
    return primary


# --------------------------------------------------------------------------------- resemblance
def _rho_stats(ctx, f, g, rows, level, seed):
    series = daily_spearman(f, g, rows)
    m, lo, hi, b = bootstrap_mean_ci(series, level, ctx.n_boot(), seed)
    return series, m, lo, hi, b


def verify_resembles(f: np.ndarray, ctx, ref: str, sign="+", scope_mask=None) -> Verdict:
    resolved = ctx.references.resolve(ref)
    if not resolved:
        return Verdict(UNVERIFIABLE, "signal_corr", {"reason": f"no reference for {ref!r}"})
    cfg = ctx.thr["behavioral"]
    f = _masked(f, scope_mask)
    out = []
    for name, s in resolved:
        eff = _sgn(sign) * s
        g = ctx.references.signal(name)

        def run(window, g=g, eff=eff, name=name):
            _, m, lo, hi, b = _rho_stats(ctx, eff * f, g, ctx.rows(window), cfg["ci_level"], ctx.seed)
            ev = {"ref": name, "claimed_sign": eff, "rho_bar": m, "ci95": [lo, hi], "block": b,
                  "market": ctx.panel.market}
            if not np.isfinite(lo):
                return Verdict(UNVERIFIABLE, "signal_corr", ev)
            if lo >= cfg["resemblance_floor"]:
                return Verdict(SUPPORTED, "signal_corr", ev)
            if hi <= cfg["resemblance_refute_ceiling"]:
                if hi <= cfg["resemblance_sign_error"]:
                    ev["sign_error"] = True
                return Verdict(REFUTED, "signal_corr", ev)
            return Verdict(UNRESOLVED, "signal_corr", ev)

        out.append(_replicate(ctx, run))
    return aggregate_any_all(out, "signal_corr")


def verify_independent(f: np.ndarray, ctx, ref: str, scope_mask=None) -> Verdict:
    """TOST: SUPPORTED iff the 90% CI of rho_bar is inside (-m, m); REFUTED iff the 95% CI is entirely
    outside [-m, m]; failing to reject rho = 0 is never support."""
    resolved = ctx.references.resolve(ref)
    if not resolved:
        return Verdict(UNVERIFIABLE, "signal_corr_tost", {"reason": f"no reference for {ref!r}"})
    cfg = ctx.thr["behavioral"]
    mgn = cfg["independence_margin"]
    f = _masked(f, scope_mask)
    out = []
    for name, _ in resolved:
        g = ctx.references.signal(name)

        def run(window, g=g, name=name):
            series = daily_spearman(f, g, ctx.rows(window))
            m, lo90, hi90, b = bootstrap_mean_ci(series, cfg["tost_ci_level"], ctx.n_boot(), ctx.seed)
            _, lo95, hi95, _ = bootstrap_mean_ci(series, cfg["ci_level"], ctx.n_boot(), ctx.seed, block=b)
            ev = {"ref": name, "rho_bar": m, "ci90": [lo90, hi90], "ci95": [lo95, hi95], "margin": mgn}
            if not np.isfinite(lo90):
                return Verdict(UNVERIFIABLE, "signal_corr_tost", ev)
            if tost_equivalent((lo90, hi90), mgn):
                return Verdict(SUPPORTED, "signal_corr_tost", ev)
            if lo95 > mgn or hi95 < -mgn:
                return Verdict(REFUTED, "signal_corr_tost", ev)
            return Verdict(UNRESOLVED, "signal_corr_tost", ev)

        out.append(_replicate(ctx, run))
    # a claim of independence fails if dependence on any operationalization is shown
    if any(v.verdict == REFUTED for v in out):
        verdict = REFUTED
    elif all(v.verdict == SUPPORTED for v in out):
        verdict = SUPPORTED
    else:
        verdict = UNRESOLVED
    return Verdict(verdict, "signal_corr_tost", {"operationalizations": [v.to_dict() for v in out]},
                   any(v.regime_dependent for v in out))


# --------------------------------------------------------------------------------- exposure
_REF_TO_FACTOR = {"SIZE_PROXY": ("SMB", -1), "STREV_5d": ("STREV", 1), "MOM_12_1": ("MOM", 1),
                  "VOL_20d": ("LOWVOL", -1), "AMIHUD_21d": ("ILLIQ", 1)}


def verify_exposed(f: np.ndarray, ctx, factor: str, sign="+", scope_mask=None) -> Verdict:
    names = ctx.factors.names()
    targets: list[tuple[str, int]] = []
    if factor in names:
        targets = [(factor, 1)]
    else:
        for ref, s in ctx.references.resolve(factor):
            if ref in _REF_TO_FACTOR and _REF_TO_FACTOR[ref][0] in names:
                fac, fs = _REF_TO_FACTOR[ref]
                targets.append((fac, s * fs))
    if not targets:
        # no factor-return series: decide at the signal level with the resemblance rule
        v = verify_resembles(f, ctx, factor, sign, scope_mask)
        v.method = "exposure_signal_level"
        return v
    cfg = ctx.thr["behavioral"]
    controls = ctx.factors.controls()
    fwd = ctx.fwd(1, "close_t")
    member = ctx.panel.member if scope_mask is None else (ctx.panel.member & scope_mask)
    ls = decile_long_short(f, fwd, member)
    out = []
    for fac, s in targets:
        eff = _sgn(sign) * s

        def run(window, fac=fac, eff=eff):
            rows = ctx.rows(window)
            X = controls.to_numpy()[rows]
            res = hac_ols(ls[rows], X, names=list(controls.columns))
            ev = {"factor": fac, "claimed_sign": eff, "controls": list(controls.columns)}
            if not res.get("ok"):
                return Verdict(UNVERIFIABLE, "exposure", {**ev, "reason": "insufficient observations"})
            beta, t, se = res["params"][fac], res["t"][fac], res["bse"][fac]
            ev.update({"beta": beta, "t": t, "se": se, "n": res["n"]})
            if eff * t >= cfg["exposure_t"] and abs(beta) >= cfg["exposure_beta_floor"]:
                return Verdict(SUPPORTED, "exposure", ev)
            if -eff * t >= cfg["exposure_t"]:
                return Verdict(REFUTED, "exposure", ev)
            z = 1.6448536269514722
            if -cfg["exposure_beta_floor"] < beta - z * se and beta + z * se < cfg["exposure_beta_floor"]:
                return Verdict(REFUTED, "exposure", {**ev, "reason": "TOST: loading below the floor"})
            return Verdict(UNRESOLVED, "exposure", ev)

        out.append(_replicate(ctx, run))
    return aggregate_any_all(out, "exposure")


# --------------------------------------------------------------------------------- turnover
def verify_turnover(f: np.ndarray, ctx, level: str, scope_mask=None) -> Verdict:
    cfg = ctx.thr["behavioral"]
    pct = ctx.references.turnover_percentiles()
    f = _masked(f, scope_mask)

    def run(window):
        ac = lag1_rank_autocorr(f, ctx.rows(window))
        m, lo, hi, b = bootstrap_mean_ci(ac, cfg["ci_level"], ctx.n_boot(), ctx.seed)
        ev = {"rank_autocorr": m, "ci95": [lo, hi], "ref_p75": pct["p75"], "ref_p25": pct["p25"], "level": level}
        if not np.isfinite(lo):
            return Verdict(UNVERIFIABLE, "rank_autocorr", ev)
        slow, fast = lo > pct["p75"], hi < pct["p25"]
        if level == "low":
            return Verdict(SUPPORTED if slow else REFUTED if fast else UNRESOLVED, "rank_autocorr", ev)
        return Verdict(SUPPORTED if fast else REFUTED if slow else UNRESOLVED, "rank_autocorr", ev)

    return _replicate(ctx, run)


# --------------------------------------------------------------------------------- regimes
def regime_indicator(ctx, regime: str) -> np.ndarray:
    mkt = pd.Series(ctx.market_returns())
    spec = ctx.cb.get("regimes", {}).get(regime)
    if spec is None:
        raise KeyError(regime)
    ind = spec["indicator"]
    if ind == "market_realized_vol_20d":
        x = mkt.rolling(20, min_periods=20).std().to_numpy()
    elif ind == "market_return_20d":
        x = mkt.rolling(20, min_periods=20).sum().to_numpy()
    elif ind == "cross_sectional_return_std":
        r = np.where(ctx.panel.member, ctx.daily_returns(), np.nan)
        with np.errstate(all="ignore"):
            x = np.nanstd(r, axis=1)
    else:
        raise KeyError(ind)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if spec.get("rule") == "positive":
        out[ok] = (x[ok] > 0).astype(float)
    elif spec.get("rule") == "negative":
        out[ok] = (x[ok] < 0).astype(float)
    else:
        q = np.nanquantile(x, [1 / 3, 2 / 3])
        out[ok] = ((x[ok] > q[1]) if spec["tercile"] == "top" else (x[ok] <= q[0])).astype(float)
    return out


def resolve_regime(ctx, text: str) -> str | None:
    regs = ctx.cb.get("regimes", {})
    t = str(text).lower()
    if t in regs:
        return t
    for k, v in regs.items():
        if any(w in t for w in v.get("words", [])):
            return k
    return None


def verify_regime(f: np.ndarray, ctx, regime: str, effect: str = "stronger", scope_mask=None) -> Verdict:
    key = resolve_regime(ctx, regime)
    if key is None:
        return Verdict(UNVERIFIABLE, "regime_interaction", {"reason": f"unknown regime {regime!r}"})
    cfg = ctx.thr["behavioral"]
    ind = regime_indicator(ctx, key)
    f = _masked(f, scope_mask)
    ic = daily_spearman(f, ctx.fwd(1, "close_t"))
    want = 1 if effect != "weaker" else -1
    s_ic = np.sign(np.nanmean(ic[ctx.rows("train")])) or 1.0
    results = {}
    for w in _windows(ctx):
        rows = ctx.rows(w)
        res = hac_ols(s_ic * ic[rows], ind[rows], names=["regime"])
        results[w] = res
    tr = results.get("train")
    if not tr or not tr.get("ok"):
        return Verdict(UNVERIFIABLE, "regime_interaction", {"reason": "insufficient observations", "regime": key})
    t_tr = tr["t"]["regime"]
    ev = {"regime": key, "effect": effect, "t_train": t_tr, "beta_train": tr["params"]["regime"]}
    te = results.get("test")
    t_te = te["t"]["regime"] if te and te.get("ok") else None
    ev["t_test"] = t_te
    if want * t_tr >= cfg["regime_t"] and t_te is not None and want * t_te >= cfg["regime_t"]:
        return Verdict(SUPPORTED, "regime_interaction", ev)
    if -want * t_tr >= cfg["regime_t"]:
        return Verdict(REFUTED, "regime_interaction", ev)
    return Verdict(UNRESOLVED, "regime_interaction", ev)


# --------------------------------------------------------------------------------- predictive sign
def ic_series(f: np.ndarray, ctx, h: int = 1, entry: str = "open_t+1") -> np.ndarray:
    return daily_spearman(f, ctx.fwd(h, entry))


def verify_pred_sign(f: np.ndarray, ctx, sign="+", horizon: int | None = None, scope_mask=None) -> Verdict:
    h = int(horizon) if horizon else 1
    cfg = ctx.thr["behavioral"]
    f = _masked(f, scope_mask)
    ic = ic_series(f, ctx, h)

    def run(window):
        x = ic[ctx.rows(window)]
        n = int(np.isfinite(x).sum())
        m, se, t = newey_west_mean(x, max(h - 1, nw_lags(max(n, 1))))
        ev = {"mean": m, "t": t, "h": h, "claimed_sign": _sgn(sign)}
        if not np.isfinite(t):
            return Verdict(UNVERIFIABLE, "rank_ic_sign", ev)
        if _sgn(sign) * t >= cfg["pred_sign_t"]:
            return Verdict(SUPPORTED, "rank_ic_sign", ev)
        if -_sgn(sign) * t >= cfg["pred_sign_t"]:
            return Verdict(REFUTED, "rank_ic_sign", ev)
        return Verdict(UNRESOLVED, "rank_ic_sign", ev)

    return _replicate(ctx, run)
