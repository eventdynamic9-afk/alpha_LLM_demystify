"""Synthetic fixture panels (§6.4): geometric Brownian motion with a market factor, heterogeneous
volatility, persistent volume and *planted* cross-sectional effects (short-term reversal, momentum,
an abnormal-volume effect) so behavioral and performance verifiers have known ground truth.

Also simulates realistic data defects: late listings, delistings, suspensions (NaN prices and zero
volume), point-in-time universe changes and, for ``market="CN"``, daily price limits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel


def business_days(start: str, n: int) -> np.ndarray:
    return pd.bdate_range(start=start, periods=n).values.astype("datetime64[D]")


def synthetic_panel(n_stocks: int = 60, n_days: int = 600, seed: int = 0, start: str = "2014-01-02",
                    market: str = "CN", reversal: float = 0.08, momentum: float = 0.03,
                    volume_effect: float = 0.03, defects: bool = True, universe_frac: float = 0.8) -> Panel:
    """Generate a deterministic panel.

    Planted effects are expressed in units of each stock's own daily volatility: next-day return
    loads ``-reversal`` on the cross-sectional z-score of the past 5-day return, ``+momentum`` on the
    z-score of the 12-1 month return and ``-volume_effect`` on the z-score of abnormal volume.
    """
    rng = np.random.default_rng(seed)
    T, N = n_days, n_stocks
    dates = business_days(start, T)
    insts = np.array([f"S{i:04d}" for i in range(N)])

    sigma = rng.uniform(0.012, 0.035, N)                    # daily idiosyncratic volatility
    beta = rng.uniform(0.6, 1.4, N)
    mkt_vol = 0.01
    m = rng.standard_normal(T) * mkt_vol
    # volatility clustering in the market
    m *= np.exp(0.5 * np.sin(np.arange(T) / 60.0))

    log_vol_mean = rng.uniform(12.0, 16.0, N)
    log_volume = np.empty((T, N))
    ret = np.zeros((T, N))
    logp = np.zeros((T, N))
    logp[0] = np.log(rng.uniform(5.0, 80.0, N))
    log_volume[0] = log_vol_mean + rng.standard_normal(N) * 0.3

    def z(x: np.ndarray) -> np.ndarray:
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        return np.zeros_like(x) if not np.isfinite(sd) or sd == 0 else (x - mu) / sd

    for t in range(1, T):
        eps = rng.standard_normal(N)
        planted = np.zeros(N)
        if t > 5:
            planted -= reversal * z(logp[t - 1] - logp[t - 6])
        if t > 252:
            planted += momentum * z(logp[t - 22] - logp[t - 253])
        if t > 60:
            abn = log_volume[t - 1] - log_volume[max(0, t - 61):t - 1].mean(axis=0)
            planted -= volume_effect * z(abn)
        ret[t] = beta * m[t] + sigma * (eps + planted)
        logp[t] = logp[t - 1] + ret[t]
        # volume: AR(1) in logs plus a response to |return|
        log_volume[t] = (log_vol_mean + 0.7 * (log_volume[t - 1] - log_vol_mean)
                         + 8.0 * np.abs(ret[t]) + rng.standard_normal(N) * 0.25)

    close = np.exp(logp)
    overnight = rng.standard_normal((T, N)) * sigma * 0.4
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * np.exp(overnight)
    hi_ext = np.abs(rng.standard_normal((T, N))) * sigma * 0.6
    lo_ext = np.abs(rng.standard_normal((T, N))) * sigma * 0.6
    high = np.maximum(open_, close) * np.exp(hi_ext)
    low = np.minimum(open_, close) * np.exp(-lo_ext)
    w = rng.uniform(0.2, 0.8, (T, N))
    vwap = low + w * (high - low)
    volume = np.round(np.exp(log_volume))
    amount = vwap * volume

    member = np.ones((T, N), dtype=bool)
    fields = {"open": open_, "high": high, "low": low, "close": close, "vwap": vwap,
              "volume": volume, "amount": amount}

    if defects:
        # late listings and delistings
        n_late = max(1, N // 12)
        for i in rng.choice(N, n_late, replace=False):
            k = int(rng.integers(T // 10, T // 3))
            for f in fields:
                fields[f][:k, i] = np.nan
            member[:k, i] = False
        n_del = max(1, N // 15)
        for i in rng.choice(N, n_del, replace=False):
            k = int(rng.integers(T // 2, T - 5))
            for f in fields:
                fields[f][k:, i] = np.nan
            member[k:, i] = False
        # suspensions: prices NaN, volume 0
        n_susp = max(1, (T * N) // 400)
        rows = rng.integers(0, T, n_susp)
        cols = rng.integers(0, N, n_susp)
        for r, c in zip(rows, cols):
            if np.isfinite(fields["close"][r, c]):
                for f in ("open", "high", "low", "close", "vwap"):
                    fields[f][r, c] = np.nan
                fields["volume"][r, c] = 0.0
                fields["amount"][r, c] = 0.0
        # point-in-time universe: a rotating subset by trailing dollar volume, rebalanced every 63 days
        k_keep = max(2, int(round(universe_frac * N)))
        dv = pd.DataFrame(fields["amount"]).rolling(63, min_periods=20).mean().to_numpy()
        for t0 in range(0, T, 63):
            ref = dv[t0] if t0 > 20 else np.nan_to_num(fields["amount"][t0], nan=0.0)
            ref = np.nan_to_num(ref, nan=-np.inf)
            keep = np.zeros(N, dtype=bool)
            keep[np.argsort(-ref)[:k_keep]] = True
            member[t0:t0 + 63] &= keep[None, :]
        member &= np.isfinite(fields["close"]) | (fields["volume"] == 0)

    meta = {"source": "synthetic", "seed": seed, "planted": {"reversal_5d": -reversal, "momentum_12_1": momentum,
                                                             "abnormal_volume": -volume_effect}}
    if market == "CN":
        meta["price_limit"] = 0.10
    return Panel(dates, insts, fields, member, market, meta)
