"""Executor E1: Qlib-semantics engine (§6.4).

E1 follows Qlib's expression engine operator by operator (``qlib/data/ops.py``): pandas rolling
windows per instrument, ``Rank(x, N)`` via ``percentileofscore(...)/100``, ``IdxMax`` via
``argmax()+1``, ``Corr``/``Cov`` via pandas rolling pairwise statistics, and a thin
cross-sectional layer (``CSRank``/``CSZScore``/``CSScale``) applied per date.

Two NaN modes:
* ``strict`` (default, used for E1-vs-E2 agreement): the canonical semantics — a window is NaN
  unless all n values are present.
* ``qlib_native``: Qlib's own ``min_periods=1`` behaviour and its ``WMA`` normalization
  (``nanmean(w * x)`` with ``w = (1..n)/sum``), for reproducing numbers from Qlib-based papers.

When the real ``qlib`` package is importable and initialized, :mod:`executors.e1_qlib.qlib_engine`
evaluates every cross-section-free subtree with Qlib itself; this module is the pandas mirror used
otherwise (``pyqlib`` ships no wheels for Python >= 3.13 at the time of writing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from dsl.ast import Node

from ..semantics import TIE_SENSITIVE, exact_sum, exact_zscore, snap


class E1Executor:
    name = "E1-qlib-semantics"

    def __init__(self, qlib_native: bool = False):
        self.qlib_native = qlib_native

    def evaluate(self, node: Node, panel, mask_members: bool = True) -> np.ndarray:
        self._panel = panel
        self._member = pd.DataFrame(panel.member, index=pd.RangeIndex(panel.T), columns=range(panel.N))
        self._cache: dict = {}
        with np.errstate(all="ignore"):
            df = self._eval(node)
        out = df.to_numpy(dtype=np.float64, copy=True)
        out[~np.isfinite(out)] = np.nan
        if mask_members:
            out[~panel.member] = np.nan
        return out

    # ------------------------------------------------------------------
    def _frame(self, arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=pd.RangeIndex(self._panel.T), columns=range(self._panel.N))

    def _mp(self, n: int) -> int:
        return 1 if self.qlib_native else n

    def _eval(self, n: Node) -> pd.DataFrame:
        if n in self._cache:
            return self._cache[n]
        if n.is_const:
            df = self._frame(np.full((self._panel.T, self._panel.N), n.value))
        elif n.is_field:
            df = self._frame(self._panel.get(n.name))
        else:
            kids = [self._eval(c) for c in n.children]
            if n.op in TIE_SENSITIVE:
                kids = [self._frame(snap(k.to_numpy(dtype=np.float64))) for k in kids]
            df = getattr(self, f"_op_{n.op}")(n, *kids)
            df = df.replace([np.inf, -np.inf], np.nan)
        self._cache[n] = df
        return df

    # ------------------------------------------------------------------ pointwise
    @staticmethod
    def _nanpair(a, b, r):
        return r.where(~(a.isna() | b.isna()))

    def _op_Add(self, n, a, b):
        return a + b

    def _op_Sub(self, n, a, b):
        return a - b

    def _op_Mul(self, n, a, b):
        return a * b

    def _op_Div(self, n, a, b):
        return (a / b).where(b != 0)

    def _op_Neg(self, n, a):
        return -a

    def _op_Abs(self, n, a):
        return a.abs()

    def _op_Sign(self, n, a):
        return np.sign(a)

    def _op_Log(self, n, a):
        return np.log(a.where(a > 0))

    def _op_Power(self, n, a):
        p = float(n.params[0])
        r = a.pow(p)
        if not p.is_integer():
            r = r.where(~(a < 0))
        if p < 0:
            r = r.where(a != 0)
        return r

    def _op_SignedPower(self, n, a):
        return np.sign(a) * a.abs().pow(float(n.params[0]))

    def _op_Greater(self, n, a, b):
        return self._nanpair(a, b, a.where(a >= b, b))

    def _op_Less(self, n, a, b):
        return self._nanpair(a, b, a.where(a <= b, b))

    def _cmp(self, a, b, r):
        return self._nanpair(a, b, r.astype(float))

    def _op_Gt(self, n, a, b):
        return self._cmp(a, b, a > b)

    def _op_Ge(self, n, a, b):
        return self._cmp(a, b, a >= b)

    def _op_Lt(self, n, a, b):
        return self._cmp(a, b, a < b)

    def _op_Le(self, n, a, b):
        return self._cmp(a, b, a <= b)

    def _op_Eq(self, n, a, b):
        return self._cmp(a, b, a == b)

    def _op_Ne(self, n, a, b):
        return self._cmp(a, b, a != b)

    def _op_And(self, n, a, b):
        return self._cmp(a, b, (a != 0) & (b != 0))

    def _op_Or(self, n, a, b):
        return self._cmp(a, b, (a != 0) | (b != 0))

    def _op_Not(self, n, a):
        return (a == 0).astype(float).where(a.notna())

    def _op_If(self, n, c, a, b):
        return a.where(c != 0, b).where(c.notna())

    # ------------------------------------------------------------------ time series (per instrument)
    def _op_Ref(self, n, a):
        return a.shift(int(n.params[0]))

    def _op_Delta(self, n, a):
        return a - a.shift(int(n.params[0]))

    def _roll(self, a, n):
        w = int(n.params[0])
        return a.rolling(w, min_periods=self._mp(w))

    def _op_Mean(self, n, a):
        return self._roll(a, n).mean()

    def _op_Sum(self, n, a):
        return self._roll(a, n).sum()

    def _op_Std(self, n, a):
        if self.qlib_native:
            return self._roll(a, n).std()
        return self._exact_var(n, a).pow(0.5)

    def _op_Var(self, n, a):
        if self.qlib_native:
            return self._roll(a, n).var()
        return self._exact_var(n, a)

    def _exact_var(self, n, a):
        """Per-window two-pass sample variance (pandas' online rolling variance loses up to ~1e-8 when the
        within-window spread is tiny); exactly 0 for constant windows (canonical semantics)."""
        w = int(n.params[0])
        var = self._roll(a, n).apply(lambda x: x.var(ddof=1), raw=True)
        return var.where(~self._constant(a, w), 0.0).where(var.notna())

    def _op_Max(self, n, a):
        return self._roll(a, n).max()

    def _op_Min(self, n, a):
        return self._roll(a, n).min()

    def _op_Med(self, n, a):
        return self._roll(a, n).median()

    def _op_Quantile(self, n, a):
        return self._roll(a, n).quantile(float(n.params[1]), interpolation="linear")

    def _op_IdxMax(self, n, a):
        return self._roll(a, n).apply(lambda x: x.argmax() + 1, raw=True)

    def _op_IdxMin(self, n, a):
        return self._roll(a, n).apply(lambda x: x.argmin() + 1, raw=True)

    def _op_TsRank(self, n, a):
        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / 100

        return self._roll(a, n).apply(rank, raw=True)

    def _op_WMA(self, n, a):
        if self.qlib_native:
            def weighted_mean(x):
                w = np.arange(len(x)) + 1
                w = w / w.sum()
                return np.nanmean(w * x)
        else:
            def weighted_mean(x):
                w = np.arange(len(x)) + 1.0
                return float(np.dot(w / w.sum(), x))
        return self._roll(a, n).apply(weighted_mean, raw=True)

    def _time_index(self) -> pd.DataFrame:
        T, N = self._panel.T, self._panel.N
        return self._frame(np.repeat(np.arange(T, dtype=float)[:, None], N, axis=1))

    def _constant(self, a, w):
        r = a.rolling(w, min_periods=self._mp(w))
        return r.max() == r.min()

    # Strict mode evaluates regression / correlation operators window by window with exact
    # per-window formulas (np.polyfit, two-pass moments), like Qlib's Cython rolling loops; pandas'
    # online rolling cov/corr loses up to ~1e-7 relative precision when the within-window variance is
    # small relative to the level.  qlib_native mode keeps pandas' vectorized rolling statistics.
    def _per_window(self, a, w, fn, b=None):
        A = a.to_numpy(dtype=np.float64)
        B = None if b is None else b.to_numpy(dtype=np.float64)
        T, N = A.shape
        out = np.full((T, N), np.nan)
        for j in range(N):
            x = A[:, j]
            ok = np.isfinite(x)
            if B is not None:
                y = B[:, j]
                ok &= np.isfinite(y)
            cnt = pd.Series(ok.astype(float)).rolling(w, min_periods=w).sum().to_numpy()
            for t in np.flatnonzero(cnt == w):
                xs = x[t - w + 1:t + 1]
                out[t, j] = fn(xs) if B is None else fn(xs, y[t - w + 1:t + 1])
        return self._frame(out)

    @staticmethod
    def _fit(xs):
        tt = np.arange(1, len(xs) + 1, dtype=float)
        slope, intercept = np.polyfit(tt, xs, 1)
        return tt, slope, intercept

    def _op_Slope(self, n, a):
        w = int(n.params[0])
        if self.qlib_native:
            cov_tx = a.rolling(w, min_periods=1).cov(self._time_index().where(a.notna()))
            return cov_tx / (w * (w + 1) / 12.0)   # sample variance (ddof=1) of w consecutive integers
        return self._per_window(a, w, lambda xs: self._fit(xs)[1])

    def _op_Rsquare(self, n, a):
        w = int(n.params[0])
        if self.qlib_native:
            corr = a.rolling(w, min_periods=1).corr(self._time_index().where(a.notna()))
            return (corr ** 2).where(~self._constant(a, w))

        def r2(xs):
            if xs.max() == xs.min():
                return np.nan
            tt, slope, intercept = self._fit(xs)
            resid = xs - (slope * tt + intercept)
            sst = ((xs - xs.mean()) ** 2).sum()
            return 1.0 - (resid ** 2).sum() / sst

        return self._per_window(a, w, r2)

    def _op_Resi(self, n, a):
        w = int(n.params[0])
        if self.qlib_native:
            slope = self._op_Slope(n, a)
            mean = a.rolling(w, min_periods=1).mean()
            return a - mean - slope * (w - 1) / 2.0

        def resi(xs):
            tt, slope, intercept = self._fit(xs)
            return xs[-1] - (slope * tt[-1] + intercept)

        return self._per_window(a, w, resi)

    @staticmethod
    def _corr(xs, ys):
        if xs.max() == xs.min() or ys.max() == ys.min():
            return np.nan
        xc, yc = xs - xs.mean(), ys - ys.mean()
        den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
        return np.nan if den == 0 else (xc * yc).sum() / den

    @staticmethod
    def _cov(xs, ys):
        return ((xs - xs.mean()) * (ys - ys.mean())).sum() / (len(xs) - 1)

    def _op_Corr(self, n, a, b):
        w = int(n.params[0])
        if self.qlib_native:
            both = a.notna() & b.notna()
            a2, b2 = a.where(both), b.where(both)
            r = a2.rolling(w, min_periods=1).corr(b2)
            return r.where(~(self._constant(a2, w) | self._constant(b2, w)))
        return self._per_window(a, w, self._corr, b)

    def _op_Cov(self, n, a, b):
        w = int(n.params[0])
        if self.qlib_native:
            both = a.notna() & b.notna()
            return a.where(both).rolling(w, min_periods=1).cov(b.where(both))
        return self._per_window(a, w, self._cov, b)

    # ------------------------------------------------------------------ cross-sectional layer
    def _members(self, a):
        return a.where(self._member)

    def _op_CSRank(self, n, a):
        return self._members(a).rank(axis=1, method="average", pct=True)

    def _op_CSZScore(self, n, a):
        m = self._members(a)

        def row(r: pd.Series) -> pd.Series:
            v = r.dropna()
            z = exact_zscore(v.to_numpy(dtype=np.float64))
            out = pd.Series(np.nan, index=r.index)
            if z is not None:
                out[v.index] = z
            return out

        return m.apply(row, axis=1)

    def _op_CSScale(self, n, a):
        m = self._members(a)
        s = m.abs().apply(lambda row: exact_sum(row.dropna()) if row.notna().any() else np.nan, axis=1)
        return m.mul(float(n.params[0])).div(s.where(s != 0), axis=0)
