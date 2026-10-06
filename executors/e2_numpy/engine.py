"""Executor E2: independent vectorized NumPy implementation of every operator, written from the
operator specification (§6.4).  Arrays are (T dates, N instruments); rolling statistics use
memory-bounded sliding windows with exact two-pass formulas."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

from dsl.ast import Node
from dsl.operators import OPS

from ..semantics import TIE_SENSITIVE, exact_sum, exact_zscore, snap

_CHUNK_ELEMS = 4_000_000


class E2Executor:
    name = "E2-numpy"

    def __init__(self, cache: bool = True):
        self._cache_enabled = cache

    # ----------------------------------------------------------------- public
    def evaluate(self, node: Node, panel, mask_members: bool = True) -> np.ndarray:
        cache: dict = {} if self._cache_enabled else None
        with np.errstate(all="ignore"):
            out = self._eval(node, panel, cache)
            out = np.array(out, dtype=np.float64, copy=True)
            if out.ndim == 0:
                out = np.full((panel.T, panel.N), float(out))
            out[~np.isfinite(out)] = np.nan
            if mask_members:
                out[~panel.member] = np.nan
        return out

    # ----------------------------------------------------------------- recursion
    def _eval(self, n: Node, panel, cache):
        if cache is not None and n in cache:
            return cache[n]
        if n.is_const:
            v = np.full((panel.T, panel.N), n.value)
        elif n.is_field:
            v = panel.get(n.name)
        else:
            kids = [self._eval(c, panel, cache) for c in n.children]
            if n.op in TIE_SENSITIVE:
                kids = [snap(k) for k in kids]
            v = self._apply(n, kids, panel)
            v = np.where(np.isfinite(v), v, np.nan)
        if cache is not None:
            cache[n] = v
        return v

    def _apply(self, n: Node, k, panel):
        op = n.op
        p = n.params
        nanmask = None
        if op in ("Add", "Sub", "Mul", "Div", "Greater", "Less", "Gt", "Ge", "Lt", "Le", "Eq", "Ne", "And", "Or"):
            a, b = k
            nanmask = np.isnan(a) | np.isnan(b)
            if op == "Add":
                r = a + b
            elif op == "Sub":
                r = a - b
            elif op == "Mul":
                r = a * b
            elif op == "Div":
                r = np.where(b == 0, np.nan, a / np.where(b == 0, 1.0, b))
            elif op == "Greater":
                r = np.maximum(a, b)
            elif op == "Less":
                r = np.minimum(a, b)
            elif op == "Gt":
                r = (a > b).astype(float)
            elif op == "Ge":
                r = (a >= b).astype(float)
            elif op == "Lt":
                r = (a < b).astype(float)
            elif op == "Le":
                r = (a <= b).astype(float)
            elif op == "Eq":
                r = (a == b).astype(float)
            elif op == "Ne":
                r = (a != b).astype(float)
            elif op == "And":
                r = ((a != 0) & (b != 0)).astype(float)
            else:
                r = ((a != 0) | (b != 0)).astype(float)
            r = np.where(nanmask, np.nan, r)
            return r
        if op == "Neg":
            return -k[0]
        if op == "Abs":
            return np.abs(k[0])
        if op == "Sign":
            return np.sign(k[0])
        if op == "Log":
            x = k[0]
            return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)
        if op == "Not":
            x = k[0]
            return np.where(np.isnan(x), np.nan, (x == 0).astype(float))
        if op == "Power":
            x, e = k[0], float(p[0])
            bad = (x < 0) & (not float(e).is_integer())
            bad |= (x == 0) & (e < 0)
            return np.where(bad, np.nan, np.power(np.where(bad, 1.0, x), e))
        if op == "SignedPower":
            x, e = k[0], float(p[0])
            return np.sign(x) * np.power(np.abs(x), e)
        if op == "If":
            c, a, b = k
            return np.where(np.isnan(c), np.nan, np.where(c != 0, a, b))
        # ------------------------------------------------------------ time series
        if op == "Ref":
            return _shift(k[0], int(p[0]))
        if op == "Delta":
            return k[0] - _shift(k[0], int(p[0]))
        if op in ("Mean", "Sum", "Std", "Var", "Max", "Min", "IdxMax", "IdxMin", "TsRank", "Quantile", "Med",
                  "WMA", "Slope", "Rsquare", "Resi"):
            return _rolling_unary(op, k[0], int(p[0]), p)
        if op in ("Corr", "Cov"):
            return _rolling_pair(op, k[0], k[1], int(p[0]))
        # ------------------------------------------------------------ cross-section
        if op in ("CSRank", "CSZScore", "CSScale"):
            return _cross_section(op, k[0], panel.member, p)
        raise KeyError(f"E2 has no implementation for {op}")


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    if d == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    if d < x.shape[0]:
        out[d:] = x[:-d]
    return out


def _chunks(T: int, N: int, n: int):
    rows = max(1, _CHUNK_ELEMS // max(1, N * n))
    start = n - 1
    while start < T:
        stop = min(T, start + rows)
        yield start, stop
        start = stop


def _windows(x: np.ndarray, n: int, start: int, stop: int) -> np.ndarray:
    """(stop-start, N, n) windows ending at rows start..stop-1, oldest value first."""
    seg = x[start - n + 1:stop]
    return sliding_window_view(seg, n, axis=0)


def _rolling_unary(op: str, x: np.ndarray, n: int, params) -> np.ndarray:
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if n > T:
        return out
    for s, e in _chunks(T, N, n):
        w = _windows(x, n, s, e)
        valid = ~np.isnan(w).any(axis=2)
        r = _window_stat(op, w, n, params)
        r = np.where(valid, r, np.nan)
        out[s:e] = r
    return out


def _window_stat(op: str, w: np.ndarray, n: int, params) -> np.ndarray:
    if op == "Mean":
        return w.mean(axis=2)
    if op == "Sum":
        return w.sum(axis=2)
    if op in ("Std", "Var"):
        mu = w.mean(axis=2, keepdims=True)
        var = ((w - mu) ** 2).sum(axis=2) / (n - 1)
        var = np.where(w.max(axis=2) == w.min(axis=2), 0.0, var)     # constant window -> exactly 0
        return np.sqrt(var) if op == "Std" else var
    if op == "Max":
        return w.max(axis=2)
    if op == "Min":
        return w.min(axis=2)
    if op == "IdxMax":
        return np.argmax(np.where(np.isnan(w), -np.inf, w), axis=2) + 1.0
    if op == "IdxMin":
        return np.argmin(np.where(np.isnan(w), np.inf, w), axis=2) + 1.0
    if op == "TsRank":
        last = w[..., -1:]
        left = (w < last).sum(axis=2)
        right = (w <= last).sum(axis=2)
        plus1 = left < right
        return (left + right + plus1) * (50.0 / n) / 100
    if op == "Quantile":
        return np.quantile(w, float(params[1]), axis=2)
    if op == "Med":
        return np.median(w, axis=2)
    if op == "WMA":
        wt = np.arange(1, n + 1, dtype=float)
        wt /= wt.sum()
        return (w * wt).sum(axis=2)
    if op in ("Slope", "Rsquare", "Resi"):
        t = np.arange(1, n + 1, dtype=float)
        tc = t - t.mean()
        stt = (tc ** 2).sum()
        mu = w.mean(axis=2, keepdims=True)
        xc = w - mu
        slope = (xc * tc).sum(axis=2) / stt
        if op == "Slope":
            return slope
        if op == "Resi":
            return w[..., -1] - (mu[..., 0] + slope * tc[-1])
        sxx = (xc ** 2).sum(axis=2)
        const = w.max(axis=2) == w.min(axis=2)
        r2 = (slope ** 2) * stt / np.where(sxx == 0, 1.0, sxx)
        return np.where(const, np.nan, r2)
    raise KeyError(op)


def _rolling_pair(op: str, x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if n > T:
        return out
    for s, e in _chunks(T, N, n):
        wx = _windows(x, n, s, e)
        wy = _windows(y, n, s, e)
        valid = ~(np.isnan(wx).any(axis=2) | np.isnan(wy).any(axis=2))
        xc = wx - wx.mean(axis=2, keepdims=True)
        yc = wy - wy.mean(axis=2, keepdims=True)
        sxy = (xc * yc).sum(axis=2)
        if op == "Cov":
            r = sxy / (n - 1)
        else:
            sxx = (xc ** 2).sum(axis=2)
            syy = (yc ** 2).sum(axis=2)
            const = (wx.max(axis=2) == wx.min(axis=2)) | (wy.max(axis=2) == wy.min(axis=2))
            den = np.sqrt(sxx * syy)
            r = np.where(const | (den == 0), np.nan, sxy / np.where(den == 0, 1.0, den))
        out[s:e] = np.where(valid, r, np.nan)
    return out


def _cross_section(op: str, x: np.ndarray, member: np.ndarray, params) -> np.ndarray:
    T, N = x.shape
    out = np.full((T, N), np.nan)
    ok = member & ~np.isnan(x)
    for t in range(T):
        idx = np.flatnonzero(ok[t])
        if len(idx) == 0:
            continue
        v = x[t, idx]
        if op == "CSRank":
            out[t, idx] = _avg_rank(v) / len(v)
        elif op == "CSZScore":
            z = exact_zscore(v)
            if z is not None:
                out[t, idx] = z
        else:
            a = float(params[0])
            s = exact_sum(np.abs(v))
            if s == 0:
                continue
            out[t, idx] = a * v / s
    return out


def _avg_rank(v: np.ndarray) -> np.ndarray:
    """1-based average ranks with ties averaged."""
    return rankdata(v, method="average")
