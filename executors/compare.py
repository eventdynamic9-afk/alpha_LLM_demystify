"""E1-vs-E2 agreement (§6.4): max |E1 - E2| <= 1e-8 after identical NaN handling, or daily
cross-sectional rank correlation >= 0.9999 where float order matters."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr

from configs import thresholds

from .semantics import TIE_SENSITIVE


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


# Operators whose output depends on the order of (possibly float-noisy) values: ties, argmax, selection.
ORDER_DEPENDENT = frozenset(TIE_SENSITIVE) | {"Greater", "Less", "Max", "Min", "Med", "Quantile", "If", "And", "Or",
                                              "Not", "Abs"}


FLOAT_ORDER_REL = 1e-9


def order_dependent(node) -> bool:
    """True when the tree contains an operator where floating-point order can change the result (§6.4)."""
    from dsl.ast import walk

    return node is not None and any(not m.is_leaf and m.op in ORDER_DEPENDENT for m in walk(node))


def agreement(a: np.ndarray, b: np.ndarray, node=None, abs_tol: float | None = None,
              rank_corr: float | None = None) -> Agreement:
    """§6.4: max |E1 - E2| <= abs_tol after identical NaN handling; the daily cross-sectional rank-correlation
    rule (>= rank_corr) applies only "where float order matters": when ``node`` contains an
    order-dependent operator (ties, argmax, selection), or when every difference is floating-point
    evaluation-order noise (99th-percentile relative difference <= ``FLOAT_ORDER_REL``, e.g. sums of
    large volumes, or an ill-conditioned division that amplifies that noise in a few cells).
    A genuine discrepancy (a wrong constant, a ddof or weighting bug) fails both conditions."""
    cfg = thresholds()["executors"]
    abs_tol = cfg["abs_tol"] if abs_tol is None else abs_tol
    rank_corr = cfg["rank_corr_fallback"] if rank_corr is None else rank_corr
    na, nb = np.isnan(a), np.isnan(b)
    mismatch = int((na != nb).sum())
    both = ~na & ~nb
    diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    if mismatch == 0 and diff <= abs_tol:
        return Agreement(True, 0, diff, 1.0, int(both.sum()), "abs_tol")
    # float-order evidence: the bulk of cells (99th percentile) agree to floating-point noise; ill-conditioned
    # cells (e.g. division by an R^2 of ~0) may amplify that noise, which the rank rule then has to absorb
    rel = (float(np.quantile(np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-300), 0.99))
           if both.any() else 0.0)
    if not (order_dependent(node) or rel <= FLOAT_ORDER_REL):
        return Agreement(False, mismatch, diff, float("nan"), int(both.sum()), "failed")
    rc = min_daily_rank_corr(a, b)
    ok = mismatch == 0 and rc >= rank_corr
    return Agreement(ok, mismatch, diff, rc, int(both.sum()), "rank_corr" if ok else "failed")


def compare_executors(node, panel, e1=None, e2=None) -> Agreement:
    from . import E1Executor, E2Executor

    e1 = e1 or E1Executor()
    e2 = e2 or E2Executor()
    return agreement(e1.evaluate(node, panel), e2.evaluate(node, panel), node)
