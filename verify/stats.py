"""Statistical primitives for the verifier.

* daily cross-sectional Spearman series
* stationary block bootstrap (Politis & Romano 1994) with automatic block length
  (Politis & White 2004, corrected by Patton, Politis & White 2009)
* Newey-West HAC t-statistics for means and OLS coefficients
* Clopper-Pearson exact binomial intervals
* TOST equivalence decisions via (1 - 2 alpha) confidence intervals (Lakens 2017)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps


# ------------------------------------------------------------------------------ cross-section
def daily_spearman(f: np.ndarray, g: np.ndarray, rows: np.ndarray | None = None, min_n: int = 10) -> np.ndarray:
    """rho_t = Spearman(f_t, g_t) over stocks with both values finite; NaN when < min_n stocks."""
    if rows is not None:
        f, g = f[rows], g[rows]
    both = np.isfinite(f) & np.isfinite(g)
    fr = pd.DataFrame(np.where(both, f, np.nan)).rank(axis=1).to_numpy()
    gr = pd.DataFrame(np.where(both, g, np.nan)).rank(axis=1).to_numpy()
    return _rowwise_pearson(fr, gr, both, min_n)


def _rowwise_pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_n: int) -> np.ndarray:
    n = mask.sum(axis=1)
    with np.errstate(all="ignore"):
        am = np.nansum(np.where(mask, a, 0.0), axis=1) / n
        bm = np.nansum(np.where(mask, b, 0.0), axis=1) / n
        ac = np.where(mask, a - am[:, None], 0.0)
        bc = np.where(mask, b - bm[:, None], 0.0)
        num = (ac * bc).sum(axis=1)
        den = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
        r = num / den
    r[(n < min_n) | ~np.isfinite(r)] = np.nan
    return r


def daily_pearson(f: np.ndarray, g: np.ndarray, rows=None, min_n: int = 10) -> np.ndarray:
    if rows is not None:
        f, g = f[rows], g[rows]
    both = np.isfinite(f) & np.isfinite(g)
    return _rowwise_pearson(f, g, both, min_n)


def lag1_rank_autocorr(f: np.ndarray, rows=None, min_n: int = 10) -> np.ndarray:
    """Daily Spearman correlation between f_t and f_{t-1} (signal persistence / turnover proxy)."""
    prev = np.vstack([np.full((1, f.shape[1]), np.nan), f[:-1]])
    return daily_spearman(f, prev, rows, min_n)


# ------------------------------------------------------------------------------ block bootstrap
def _flat_top(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def optimal_block_length(x: np.ndarray) -> float:
    """Politis-White (2004) automatic block length for the stationary bootstrap (PPW 2009 correction)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return 1.0
    x = x - x.mean()
    kn = max(5, int(math.ceil(math.log10(n))))
    mmax = int(math.ceil(math.sqrt(n))) + kn
    bmax = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    c = 2.0
    var = x @ x / n
    if var == 0:
        return 1.0
    acv = np.array([x[: n - k] @ x[k:] / n for k in range(mmax + 1)])
    rho = acv / acv[0]
    thresh = c * math.sqrt(math.log10(n) / n)
    mhat = None
    for m in range(1, mmax - kn + 2):
        if np.all(np.abs(rho[m: m + kn]) < thresh):
            mhat = m - 1
            break
    if mhat is None:
        mhat = mmax
    M = min(2 * max(mhat, 1), mmax)
    k = np.arange(-M, M + 1)
    lam = _flat_top(k / M)
    R = acv[np.abs(k)]
    G = np.sum(lam * np.abs(k) * R)
    g0 = np.sum(lam * R)
    D_sb = 2.0 * g0 ** 2
    if D_sb <= 0 or G == 0:
        return 1.0
    b = (2.0 * G ** 2 / D_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), bmax))


def stationary_indices(n: int, block: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_mean_ci(x: np.ndarray, level: float = 0.95, n_boot: int = 1000, seed: int = 0,
                      block: float | None = None) -> tuple[float, float, float, float]:
    """(mean, lo, hi, block length): percentile CI of the mean under the stationary bootstrap."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float(np.mean(x)) if len(x) else float("nan"), float("nan"), float("nan"), float("nan"))
    b = optimal_block_length(x) if block is None else block
    rng = np.random.default_rng(seed)
    idx = stationary_indices(len(x), b, n_boot, rng)
    means = x[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), float(b)


def bootstrap_matrix_means(X: np.ndarray, n_boot: int = 1000, seed: int = 0, block: float | None = None) -> np.ndarray:
    """Bootstrap means of K aligned series (K, T) with common resampled dates -> (n_boot, K)."""
    keep = np.all(np.isfinite(X), axis=0)
    X = X[:, keep]
    if X.shape[1] < 3:
        return np.full((n_boot, X.shape[0]), np.nan)
    if block is None:
        block = float(np.median([optimal_block_length(x) for x in X]))
    idx = stationary_indices(X.shape[1], block, n_boot, np.random.default_rng(seed))
    return np.stack([X[:, i].mean(axis=1) for i in idx])


# ------------------------------------------------------------------------------ HAC inference
def nw_lags(n: int) -> int:
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_mean(x: np.ndarray, lags: int | None = None) -> tuple[float, float, float]:
    """(mean, HAC standard error, t) with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float("nan"),) * 3
    L = nw_lags(n) if lags is None else lags
    mu = x.mean()
    e = x - mu
    s = e @ e / n
    for k in range(1, min(L, n - 1) + 1):
        w = 1.0 - k / (L + 1.0)
        s += 2.0 * w * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(s, 0.0) / n)
    return float(mu), float(se), float(mu / se) if se > 0 else float("nan")


def hac_ols(y: np.ndarray, X: np.ndarray, lags: int | None = None, names: list[str] | None = None) -> dict:
    """OLS with Newey-West covariance (statsmodels). X excludes the constant (added here)."""
    import statsmodels.api as sm

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    if len(y) < X.shape[1] + 5:
        return {"n": int(len(y)), "ok": False}
    L = nw_lags(len(y)) if lags is None else lags
    res = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    names = ["const"] + (names or [f"x{i}" for i in range(X.shape[1])])
    return {"n": int(len(y)), "ok": True, "params": dict(zip(names, res.params.tolist())),
            "bse": dict(zip(names, res.bse.tolist())), "t": dict(zip(names, res.tvalues.tolist())),
            "r2": float(res.rsquared)}


# ------------------------------------------------------------------------------ binomial / TOST
def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    a = 1.0 - level
    lo = 0.0 if k == 0 else sps.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else sps.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = sps.norm.ppf(1 - (1 - level) / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def tost_equivalent(ci90: tuple[float, float], margin: float) -> bool:
    """TOST at alpha=0.05: equivalent iff the 90% CI lies inside (-margin, margin)."""
    return -margin < ci90[0] and ci90[1] < margin
