"""Shared, implementation-independent conventions of the canonical operator semantics.

Both executors implement the semantics in ``dsl.operators.OPS[...].semantics`` independently; this
module only fixes the few numeric conventions they must agree on so that "identical NaN handling"
and identical tie handling are well defined (§6.4).
"""
import math

import numpy as np

# A rolling window is "constant" (Corr/Rsquare undefined -> NaN) iff its max equals its min exactly.
CONSTANT_WINDOW_RULE = "max == min"
# Standard deviation / variance / covariance use the sample (ddof=1) estimator.
DDOF = 1
# Cross-sectional operators act on universe members with non-NaN inputs; non-members get NaN.
XS_MEMBERS_ONLY = True
# Tie-sensitive operators (TsRank, CSRank, IdxMax, IdxMin, comparisons, Sign) compare their inputs
# after rounding to SNAP_DIGITS significant digits, so that values equal up to floating-point
# summation order are treated as ties by every implementation.
SNAP_DIGITS = 12
TIE_SENSITIVE = ("TsRank", "CSRank", "IdxMax", "IdxMin", "Gt", "Ge", "Lt", "Le", "Eq", "Ne", "Sign")


def snap(x: np.ndarray, digits: int = SNAP_DIGITS) -> np.ndarray:
    """Round to ``digits`` significant digits (NaN/inf/0 unchanged)."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        ax = np.abs(x)
        ok = np.isfinite(x) & (ax > 0)
        e = np.floor(np.log10(np.where(ok, ax, 1.0)))
        scale = np.power(10.0, digits - 1 - e)
        out = np.where(ok, np.round(x * scale) / scale, x)
    return out


def exact_sum(values) -> float:
    """Correctly rounded sum (used for cross-sectional normalizers by both executors)."""
    return math.fsum(values)
