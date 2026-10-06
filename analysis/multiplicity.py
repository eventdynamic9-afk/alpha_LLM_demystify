"""Multiplicity control (§12.5): Holm (FWER) for the small confirmatory family; Benjamini-Hochberg /
Benjamini-Yekutieli (FDR) for the labelled exploratory family."""
from __future__ import annotations

import numpy as np


def holm(p: list[float], alpha: float = 0.05) -> dict:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return {"p_adjusted": adj.tolist(), "reject": (adj <= alpha).tolist()}


def benjamini_hochberg(p: list[float], q: float = 0.10) -> dict:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return {"p_adjusted": adj.tolist(), "reject": (adj <= q).tolist()}


def benjamini_yekutieli(p: list[float], q: float = 0.10) -> dict:
    m = len(p)
    c = sum(1.0 / (i + 1) for i in range(m))
    bh = benjamini_hochberg(p, q)
    adj = np.minimum(np.asarray(bh["p_adjusted"]) * c, 1.0)
    return {"p_adjusted": adj.tolist(), "reject": (adj <= q).tolist()}
