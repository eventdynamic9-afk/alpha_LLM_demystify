"""Inter-annotator agreement (§9.3): Cohen's kappa (two raters), Fleiss' kappa, Krippendorff's alpha
(nominal; alpha >= 0.80 "reliable", >= 0.667 "tentative") and Gwet's AC1 (robust to skewed prevalence)."""
from __future__ import annotations

from collections import Counter

import numpy as np


def cohen_kappa(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    cats = sorted({c for p in pairs for c in p}, key=str)
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: (N items, K categories), each row summing to the number of raters n (constant)."""
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return float("nan")
    n = counts.sum(axis=1)
    if np.any(n != n[0]) or n[0] < 2:
        return float("nan")
    n = n[0]
    N = counts.shape[0]
    p_j = counts.sum(axis=0) / (N * n)
    P_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    Pbar, Pe = P_i.mean(), (p_j ** 2).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def krippendorff_alpha_nominal(data: list[list]) -> float:
    """data: raters x units matrix with None for missing values (nominal metric)."""
    units = list(zip(*data))
    coincid = Counter()
    for u in units:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i != j:
                    coincid[(a, b)] += 1.0 / (m - 1)
    n = sum(coincid.values())
    if n == 0:
        return float("nan")
    cats = sorted({a for a, _ in coincid} | {b for _, b in coincid}, key=str)
    n_c = {c: sum(v for (a, _), v in coincid.items() if a == c) for c in cats}
    Do = sum(v for (a, b), v in coincid.items() if a != b) / n
    De = sum(n_c[a] * n_c[b] for a in cats for b in cats if a != b) / (n * (n - 1))
    return float(1 - Do / De) if De > 0 else 1.0


def gwet_ac1(data: list[list]) -> float:
    """Gwet's AC1 for >= 2 raters (raters x units, None = missing)."""
    units = [[v for v in u if v is not None] for u in zip(*data)]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    cats = sorted({v for u in units for v in u}, key=str)
    Q = len(cats)
    if Q < 2:
        return 1.0
    pa = np.mean([sum(c * (c - 1) for c in Counter(u).values()) / (len(u) * (len(u) - 1)) for u in units])
    pi = {c: np.mean([Counter(u)[c] / len(u) for u in units]) for c in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (Q - 1)
    return float((pa - pe) / (1 - pe)) if pe < 1 else 1.0


def interpret_alpha(alpha: float) -> str:
    if alpha >= 0.80:
        return "reliable"
    if alpha >= 0.667:
        return "tentative"
    return "unreliable"
