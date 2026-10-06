"""Factor returns for return-level behavioral checks (§10.3 EXPOSED, originality).

Self-built daily factors (CN primary; US fallback when external tables are absent): equal-weighted
market (MKT) and decile long-short returns of reference characteristics.  For the US, the Ken French
daily FF5 + MOM + ST-reversal table can be supplied as ``ctx.external_factors``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SELF_BUILT = {             # factor name -> (reference signal, sign: +1 long high values)
    "SMB": ("SIZE_PROXY", -1),
    "STREV": ("STREV_5d", 1),
    "MOM": ("MOM_12_1", 1),
    "LOWVOL": ("VOL_20d", -1),
    "ILLIQ": ("AMIHUD_21d", 1),
}
US_EXTERNAL_CONTROLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "ST_Rev"]


def decile_long_short(signal: np.ndarray, fwd: np.ndarray, member: np.ndarray, q: float = 0.1,
                      min_n: int = 10) -> np.ndarray:
    """Equal-weighted top-minus-bottom quantile forward return per date."""
    s = np.where(member & np.isfinite(signal) & np.isfinite(fwd), signal, np.nan)
    pct = pd.DataFrame(s).rank(axis=1, pct=True).to_numpy()
    n = np.isfinite(s).sum(axis=1)
    top = pct > 1 - q
    bot = pct <= q
    with np.errstate(all="ignore"):
        lt = np.nansum(np.where(top, fwd, 0.0), axis=1) / top.sum(axis=1)
        sb = np.nansum(np.where(bot, fwd, 0.0), axis=1) / bot.sum(axis=1)
    out = lt - sb
    out[(n < min_n) | ~np.isfinite(out)] = np.nan
    return out


def decile_portfolios(signal: np.ndarray, fwd: np.ndarray, member: np.ndarray, k: int = 10) -> np.ndarray:
    """(T, k) equal-weighted quantile-portfolio forward returns (test assets for FGX)."""
    s = np.where(member & np.isfinite(signal) & np.isfinite(fwd), signal, np.nan)
    pct = pd.DataFrame(s).rank(axis=1, pct=True).to_numpy()
    out = np.full((signal.shape[0], k), np.nan)
    for j in range(k):
        m = (pct > j / k) & (pct <= (j + 1) / k)
        with np.errstate(all="ignore"):
            out[:, j] = np.nansum(np.where(m, fwd, 0.0), axis=1) / m.sum(axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


class FactorSet:
    def __init__(self, ctx):
        self.ctx = ctx
        self._self_built = None

    def self_built(self) -> pd.DataFrame:
        if self._self_built is None:
            ctx = self.ctx
            fwd = ctx.fwd(1, "close_t")
            cols = {"MKT": np.r_[ctx.market_returns()[1:], np.nan]}
            for name, (ref, sign) in SELF_BUILT.items():
                cols[name] = sign * decile_long_short(ctx.references.signal(ref), fwd, ctx.panel.member)
            self._self_built = pd.DataFrame(cols, index=pd.DatetimeIndex(ctx.panel.dates))
        return self._self_built

    def controls(self) -> pd.DataFrame:
        """US: FF5 + MOM + ST-reversal (daily) when supplied; otherwise the self-built set."""
        ext = self.ctx.external_factors
        if self.ctx.panel.market == "US" and ext is not None:
            cols = [c for c in US_EXTERNAL_CONTROLS if c in ext.columns]
            # forward-align: factor return realized on t+1 matches the label of a signal at t
            idx = pd.DatetimeIndex(self.ctx.panel.dates)
            return ext[cols].shift(-1).reindex(idx)
        return self.self_built()

    def names(self) -> list[str]:
        return list(self.controls().columns)
