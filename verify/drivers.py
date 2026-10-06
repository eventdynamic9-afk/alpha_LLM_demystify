"""Driver attribution by variance-based global sensitivity (§10.2; Saltelli et al. 2010).

For each input field, the field is replaced by a block-bootstrapped series transplanted from another
stock-period (level-normalized to the target stock, preserving its marginal behaviour) and the
Jansen total-effect index of the signal's cross-sectional ranks is computed:

    S_T(field) = E[(r(A) - r(A_B^field))^2] / (2 Var(r(A)))

"True drivers" are fields with S_T >= 0.10 (pre-registered); used only for coverage/omission metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsl import Node

from .static import dependency_set


def _transplant(x: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    T, N = x.shape
    out = np.full_like(x, np.nan)
    for i in range(N):
        donors = [j for j in range(N) if j != i and np.isfinite(x[:, j]).sum() > block]
        if not donors:
            continue
        j = donors[rng.integers(len(donors))]
        src = x[:, j][np.isfinite(x[:, j])]
        if len(src) <= block:
            continue
        series = []
        while sum(len(s) for s in series) < T:
            s0 = rng.integers(0, len(src) - block)
            series.append(src[s0:s0 + block])
        y = np.concatenate(series)[:T]
        mi, mj = np.nanmedian(x[:, i]), np.nanmedian(src)
        if np.isfinite(mi) and np.isfinite(mj) and mj != 0:
            y = y * (mi / mj)
        out[:, i] = np.where(np.isfinite(x[:, i]), y, np.nan)
    return out


def _ranks(sig: np.ndarray, member: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.where(member, sig, np.nan)).rank(axis=1, pct=True).to_numpy()


def total_effect_indices(node: Node, ctx, window: str = "train", block: int | None = None,
                         seed: int = 0) -> dict[str, float]:
    block = block or int(ctx.thr["drivers"]["block_length"])
    rng = np.random.default_rng(seed)
    p = ctx.panel
    rows = ctx.rows(window)
    r0 = _ranks(ctx.signal(node), p.member)
    out = {}
    for f in sorted(dependency_set(node)):
        x2 = _transplant(p.get(f), rng, block)
        r1 = _ranks(ctx.executor.evaluate(node, p.with_fields(**{f: x2})), p.member)
        m = rows[:, None] & np.isfinite(r0) & np.isfinite(r1)
        if m.sum() < 10:
            out[f] = float("nan")
            continue
        v = np.var(r0[m])
        out[f] = float(np.mean((r0[m] - r1[m]) ** 2) / (2 * v)) if v > 0 else float("nan")
    return out


def true_drivers(node: Node, ctx, threshold: float | None = None, **kw) -> tuple[list[str], dict]:
    thr = ctx.thr["drivers"]["total_effect_threshold"] if threshold is None else threshold
    st = total_effect_indices(node, ctx, **kw)
    return [f for f, v in st.items() if np.isfinite(v) and v >= thr], st
