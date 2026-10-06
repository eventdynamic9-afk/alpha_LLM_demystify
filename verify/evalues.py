"""Sequential falsification with e-values (§10.3 Option B; POPPER, Huang et al. ICML 2025).

Anytime-valid alternative for behavioral claims tested adaptively (e.g. many conditional claims from A2
tool-using narrators).  For a bounded daily statistic x_t in [-1, 1] (a daily rank correlation or RankIC)
and the null H0: E[x_t] <= mu0, the betting martingale

    K_t = prod_{s <= t} (1 + lambda_s (x_s - mu0)),   lambda_s predictable, 0 <= lambda_s <= 1 / (1 + mu0)

is a nonnegative supermartingale under H0, so P(sup_t K_t >= 1/alpha) <= alpha (Ville).  lambda_s uses
the aGRAPA plug-in (Waudby-Smith & Ramdas 2023).  Rejecting H0 at 1/alpha is valid at any stopping time.
Serial dependence in x_t violates the martingale assumption; thin the series to every ``stride``-th day
(e.g. the automatic block length) before testing.
"""
from __future__ import annotations

import numpy as np


def betting_evalue(x: np.ndarray, mu0: float, alpha: float = 0.05, stride: int = 1, c: float = 0.5) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)][::max(1, stride)]
    lam_max = c / (1.0 + mu0)
    log_k = 0.0
    path = []
    mean, var, n = 0.0, 0.25, 0
    rejected_at = None
    for t, xt in enumerate(x):
        lam = (mean - mu0) / (var + (mean - mu0) ** 2) if n > 0 else 0.0
        lam = float(np.clip(lam, 0.0, lam_max))
        log_k += np.log1p(lam * (xt - mu0))
        path.append(log_k)
        n += 1
        delta = xt - mean
        mean += delta / n
        var = var + (delta * (xt - mean) - var) / n if n > 1 else var
        if rejected_at is None and log_k >= np.log(1.0 / alpha):
            rejected_at = t
    return {"e_value": float(np.exp(log_k)), "max_e_value": float(np.exp(max(path))) if path else 1.0,
            "reject": rejected_at is not None, "rejected_at": rejected_at, "n": int(len(x)), "mu0": mu0, "alpha": alpha}


def sequential_resemblance(rho_series: np.ndarray, floor: float = 0.30, ceiling: float = 0.10, alpha: float = 0.05,
                           stride: int = 1) -> dict:
    """Two one-sided anytime-valid tests: support (H0: mean <= floor) and refutation (H0: mean >= ceiling)."""
    sup = betting_evalue(rho_series, floor, alpha, stride)
    ref = betting_evalue(-np.asarray(rho_series, dtype=float), -ceiling, alpha, stride)
    verdict = "SUPPORTED" if sup["reject"] else "REFUTED" if ref["reject"] else "UNRESOLVED"
    return {"verdict": verdict, "support": sup, "refute": ref}
