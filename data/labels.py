"""Forward-return labels (§5.2) — label hygiene, §6.3 layer 4.

This module is deliberately separate from the expression engine: nothing under ``dsl/`` or
``executors/`` may import it (enforced by ``executors/tests/test_label_hygiene.py``).

Convention (stated once): the signal at day t uses data up to and including t's close; the forward
return runs from t+1 open (default) or t close to t+h close, h in {1, 5, 10, 20}.
"""
from __future__ import annotations

import numpy as np

from .panel import Panel


def forward_returns(panel: Panel, h: int = 1, entry: str = "open_t+1") -> np.ndarray:
    close = panel.get("close")
    T = panel.T
    out = np.full_like(close, np.nan)
    if h < 1 or h >= T:
        return out
    exit_px = close[h:]
    if entry == "close_t":
        entry_px = close[:-h]
    elif entry == "open_t+1":
        op = panel.get("open")
        entry_px = op[1:T - h + 1]
    else:
        raise ValueError(f"unknown entry convention {entry!r}")
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:-h] = exit_px / entry_px - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def daily_returns(panel: Panel) -> np.ndarray:
    """Close-to-close return realized on day t (backward looking, used for portfolio P&L)."""
    close = panel.get("close")
    out = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = close[1:] / close[:-1] - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def limit_locked(panel: Panel, limit: float | None = None, tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """(limit_up, limit_down) masks for China A-shares: a stock closing at its daily limit cannot be
    bought (limit-up) or sold (limit-down) that day (§5.3 rule 4)."""
    limit = limit if limit is not None else panel.meta.get("price_limit")
    r = daily_returns(panel)
    if limit is None:
        z = np.zeros_like(r, dtype=bool)
        return z, z
    up = r >= limit - tol
    down = r <= -limit + tol
    return np.nan_to_num(up, nan=False).astype(bool), np.nan_to_num(down, nan=False).astype(bool)
