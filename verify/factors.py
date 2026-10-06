"""Factor returns for return-level behavioral checks (§10.3 EXPOSED, originality).

Self-built daily factors (CN primary; US fallback when external tables are absent): equal-weighted
market (MKT) and decile long-short returns of reference characteristics.  For the US, the Ken French
daily FF5 + MOM + ST-reversal table (``python -m data refs`` -> data/processed/us_factor_returns_daily.csv)
is loaded as ``ctx.external_factors``; for CN the Liu-Stambaugh-Yuan CH-3 (CH-4) monthly table is
loaded as ``ctx.external_factors_monthly``.  Reference signals map to factor columns per factor set.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
CH_MONTHLY_CONTROLS = ["mktrf", "SMB", "VMG", "PMO"]           # CH-3 (+ PMO when the CH-4 file is supplied)

# reference signal -> (factor column, sign: +1 if the factor is long high values of the signal)
SELF_BUILT_REF_MAP = {ref: (name, sign) for name, (ref, sign) in SELF_BUILT.items()}
US_EXTERNAL_REF_MAP = {"SIZE_PROXY": ("SMB", -1), "MOM_12_1": ("Mom", 1), "MOM_6_1": ("Mom", 1),
                       "STREV_1d": ("ST_Rev", 1), "STREV_5d": ("ST_Rev", 1), "STREV_21d": ("ST_Rev", 1)}
CH_REF_MAP = {"SIZE_PROXY": ("SMB", -1), "ABNVOL_60d": ("PMO", -1)}   # PMO: pessimistic (low turnover) minus optimistic
# factor names without a same-panel reference signal (terms with one go through the codebook + ref maps)
FACTOR_ALIASES = {"market": ("MKT", "Mkt-RF", "mktrf"), "value": ("HML", "VMG"), "profitability": ("RMW",),
                  "investment": ("CMA",)}


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


def long_short_turnover(signal: np.ndarray, member: np.ndarray, rows: np.ndarray | None = None,
                        q: float = 0.1) -> np.ndarray:
    """Daily turnover sum|w_t - w_{t-1}| / 2 of the equal-weighted top-minus-bottom quantile portfolio."""
    s = np.where(member & np.isfinite(signal), signal, np.nan)
    pct = pd.DataFrame(s).rank(axis=1, pct=True).to_numpy()
    top, bot = pct > 1 - q, pct <= q
    with np.errstate(all="ignore"):
        w = top / top.sum(axis=1, keepdims=True) - bot / bot.sum(axis=1, keepdims=True)
    ok = (top.sum(axis=1) > 0) & (bot.sum(axis=1) > 0)
    w[~ok] = np.nan
    to = np.full(len(w), np.nan)
    to[1:] = np.abs(np.nan_to_num(w[1:]) - np.nan_to_num(w[:-1])).sum(axis=1) / 2.0
    to[1:][~(ok[1:] & ok[:-1])] = np.nan
    if rows is not None:
        to = np.where(rows, to, np.nan)
    return to


def monthly_compound(daily: np.ndarray, dates, min_days: int = 15) -> pd.Series:
    """Compound daily returns to calendar months (PeriodIndex); months with < min_days returns -> NaN."""
    x = pd.Series(daily, index=pd.DatetimeIndex(dates))
    g = x.groupby(x.index.to_period("M"))
    out = g.apply(lambda r: float(np.prod(1.0 + r.dropna()) - 1.0))
    out[g.count() < min_days] = np.nan
    return out


def load_factor_table(path: str | Path) -> pd.DataFrame:
    """Read a date-indexed factor-return CSV in decimal units (Ken French daily table written by
    ``python -m data refs``, or a CH-3/CH-4 table); percent tables are converted."""
    df = pd.read_csv(path, index_col=0)
    idx = df.index.astype(str)
    fmt = "%Y%m" if idx.str.fullmatch(r"\d{6}").all() else "%Y%m%d" if idx.str.fullmatch(r"\d{8}").all() else None
    df.index = pd.to_datetime(idx, format=fmt) if fmt else pd.to_datetime(idx)
    df = df.select_dtypes("number")
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    if df.abs().mean().mean() > 0.05:          # percent -> decimal
        df = df / 100.0
    return df.sort_index()


def default_us_factor_path() -> Path:
    from configs import REPO_ROOT

    return REPO_ROOT / "data" / "processed" / "us_factor_returns_daily.csv"


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
        self._warned = False

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

    # ------------------------------------------------------------------ factor set / mapping
    def external(self) -> bool:
        return self.ctx.panel.market == "US" and self.ctx.external_factors is not None

    def source(self) -> str:
        return "french_daily" if self.external() else "self_built"

    def deviation(self) -> str | None:
        """§10.3 requires the French daily set for the US; using the self-built proxies is a deviation."""
        if self.ctx.panel.market == "US" and not self.external():
            if not self._warned:
                print("WARNING: US EXPOSED/NOVEL use self-built factor proxies: the Ken French FF5 + Mom + ST_Rev "
                      "daily table was not loaded (python -m data refs; --factors)", file=sys.stderr)
                self._warned = True
            return "US Ken French FF5 + Mom + ST_Rev daily table not loaded; self-built proxies used"
        return None

    def ref_map(self) -> dict:
        return US_EXTERNAL_REF_MAP if self.external() else SELF_BUILT_REF_MAP

    def _direct(self, factor: str, names: list[str]) -> list[tuple[str, int]]:
        if factor in names:
            return [(factor, 1)]
        low = {n.lower(): n for n in names}
        if factor.strip().lower() in low:
            return [(low[factor.strip().lower()], 1)]
        hit = [c for c in FACTOR_ALIASES.get(factor.strip().lower(), ()) if c in names]
        return [(hit[0], 1)] if hit else []

    def _mapped(self, resolved, ref_map: dict, names: list[str]) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for ref, s in resolved:
            if ref in ref_map and ref_map[ref][0] in names:
                fac, fs = ref_map[ref]
                if (fac, s * fs) not in out:
                    out.append((fac, s * fs))
        return out

    def targets(self, factor: str, resolved: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Daily factor columns (with sign) that operationalize a claimed factor / codebook term."""
        names = self.names()
        return self._direct(factor, names) or self._mapped(resolved, self.ref_map(), names)

    # ------------------------------------------------------------------ CN: CH-3 monthly
    def has_monthly(self) -> bool:
        return self.ctx.panel.market == "CN" and self.ctx.external_factors_monthly is not None

    def monthly_controls(self) -> pd.DataFrame:
        ext = self.ctx.external_factors_monthly
        cols = [c for c in CH_MONTHLY_CONTROLS if c in ext.columns]
        out = ext[cols].copy()
        out.index = pd.DatetimeIndex(out.index).to_period("M")
        return out

    def monthly_targets(self, factor: str, resolved: list[tuple[str, int]]) -> list[tuple[str, int]]:
        if not self.has_monthly():
            return []
        names = list(self.monthly_controls().columns)
        return self._direct(factor, names) or self._mapped(resolved, CH_REF_MAP, names)
