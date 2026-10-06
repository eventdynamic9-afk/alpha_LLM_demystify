"""E1-vs-E2 agreement (§6.4): max |E1 - E2| <= 1e-8 after identical NaN handling, or daily
cross-sectional rank correlation >= 0.9999 where float order matters."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr

from configs import thresholds


@dataclass
class Agreement:
    ok: bool
    nan_mismatch: int
    max_abs_diff: float
    min_rank_corr: float
    n_compared: int
    rule: str

    def to_dict(self) -> dict:
        return asdict(self)


def min_daily_rank_corr(a: np.ndarray, b: np.ndarray, min_n: int = 5) -> float:
    worst = 1.0
    for t in range(a.shape[0]):
        m = np.isfinite(a[t]) & np.isfinite(b[t])
        if m.sum() < min_n:
            continue
        x, y = a[t, m], b[t, m]
        if np.all(x == x[0]) and np.all(y == y[0]):
            continue
        if np.all(x == x[0]) or np.all(y == y[0]):
            return 0.0
        r = spearmanr(x, y).statistic
        if np.isfinite(r):
            worst = min(worst, float(r))
    return worst


def agreement(a: np.ndarray, b: np.ndarray, abs_tol: float | None = None,
              rank_corr: float | None = None) -> Agreement:
    cfg = thresholds()["executors"]
    abs_tol = cfg["abs_tol"] if abs_tol is None else abs_tol
    rank_corr = cfg["rank_corr_fallback"] if rank_corr is None else rank_corr
    na, nb = np.isnan(a), np.isnan(b)
    mismatch = int((na != nb).sum())
    both = ~na & ~nb
    diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    if mismatch == 0 and diff <= abs_tol:
        return Agreement(True, 0, diff, 1.0, int(both.sum()), "abs_tol")
    rc = min_daily_rank_corr(a, b)
    ok = mismatch == 0 and rc >= rank_corr
    return Agreement(ok, mismatch, diff, rc, int(both.sum()), "rank_corr" if ok else "failed")


def compare_executors(node, panel, e1=None, e2=None) -> Agreement:
    from . import E1Executor, E2Executor

    e1 = e1 or E1Executor()
    e2 = e2 or E2Executor()
    return agreement(e1.evaluate(node, panel), e2.evaluate(node, panel))
