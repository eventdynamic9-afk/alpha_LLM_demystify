"""Prediction-powered inference for parser-dependent aggregate metrics (§9.4).

For a rationale-level metric f (e.g. claim precision of rationale r):

    theta_PPI = mean_{all r} f(auto_r) + mean_{r in gold} [ f(gold_r) - f(auto_r) ]

with the CI of Angelopoulos et al. (2023); PPI++ (power-tuned lambda, Angelopoulos, Duchi & Zrnic
2023) shrinks toward the classical estimate when the parser is poor.  Boyeau et al. (2025) show this
raises the effective human-labelled sample size.  Both naive and PPI-corrected values are reported.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def ppi_mean(auto_all: np.ndarray, auto_gold: np.ndarray, gold: np.ndarray, level: float = 0.95,
             lam: float | None = 1.0) -> dict:
    """``auto_all``: f(auto) on all N rationales; ``auto_gold``/``gold``: f(auto) and f(gold) on the n
    gold-labelled rationales.  ``lam=None`` selects the PPI++ power-tuned lambda."""
    A = np.asarray(auto_all, float)
    A = A[np.isfinite(A)]
    ag, g = np.asarray(auto_gold, float), np.asarray(gold, float)
    ok = np.isfinite(ag) & np.isfinite(g)
    ag, g = ag[ok], g[ok]
    N, n = len(A), len(g)
    if n < 2 or N < 2:
        return {"estimate": float("nan"), "ci": [float("nan")] * 2}
    if lam is None:
        cov = np.cov(g, ag, ddof=1)[0, 1]
        var_a = np.var(np.concatenate([A, ag]), ddof=1)
        lam = float(np.clip(cov / ((1 + n / N) * var_a), 0.0, 1.0)) if var_a > 0 else 0.0
    rect = g - lam * ag
    est = lam * A.mean() + rect.mean()
    se = math.sqrt(lam ** 2 * A.var(ddof=1) / N + rect.var(ddof=1) / n)
    z = sps.norm.ppf(1 - (1 - level) / 2)
    return {"estimate": float(est), "ci": [float(est - z * se), float(est + z * se)], "se": float(se), "lambda": lam,
            "naive_auto": float(A.mean()), "classical_gold": float(g.mean()), "N": N, "n": n}


def rationale_precision(claims: list[dict], verdict_of: dict[str, str]) -> float:
    """Claim precision of one rationale from a list of (parsed) claims and their verdicts."""
    v = [verdict_of.get(c["claim_id"]) for c in claims]
    d = [x for x in v if x in ("SUPPORTED", "REFUTED")]
    return sum(x == "SUPPORTED" for x in d) / len(d) if d else float("nan")


def ppi_by_condition(rat_auto: dict[str, float], rat_gold: dict[str, float], condition_of: dict[str, str],
                     level: float = 0.95, power_tuned: bool = True) -> dict:
    out = {}
    for cond in sorted(set(condition_of.values())):
        ids = [r for r, c in condition_of.items() if c == cond]
        A = np.array([rat_auto.get(r, np.nan) for r in ids])
        gids = [r for r in ids if r in rat_gold]
        out[cond] = ppi_mean(A, np.array([rat_auto.get(r, np.nan) for r in gids]),
                             np.array([rat_gold[r] for r in gids]), level, None if power_tuned else 1.0)
    return out
