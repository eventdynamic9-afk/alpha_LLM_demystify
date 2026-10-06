"""Dynamic truncation test — layer 3 of look-ahead impossibility (§6.3), as in AlphaQT-Bench.

The signal is computed on the full panel and on >= 5 truncated prefixes; values on overlapping
dates must be exactly equal (NaN == NaN).  Any discrepancy is a causality violation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from configs import thresholds


@dataclass
class CausalityReport:
    ok: bool
    cuts: list[int]
    violations: list[dict] = field(default_factory=list)


def truncation_cuts(T: int, n: int) -> list[int]:
    lo = max(2, int(T * 0.3))
    cuts = sorted(set(int(round(c)) for c in np.linspace(lo, T - 1, n)))
    return [c for c in cuts if 1 <= c < T]


def truncation_test(node, panel, executor, n_truncations: int | None = None) -> CausalityReport:
    n = n_truncations or thresholds()["causality"]["n_truncations"]
    full = executor.evaluate(node, panel)
    cuts = truncation_cuts(panel.T, n)
    violations = []
    for c in cuts:
        part = executor.evaluate(node, panel.prefix(c))
        ref = full[:c]
        same = np.array_equal(ref, part, equal_nan=True)
        if not same:
            bad = ~((ref == part) | (np.isnan(ref) & np.isnan(part)))
            rows = np.flatnonzero(bad.any(axis=1))
            violations.append({"cut": c, "n_cells": int(bad.sum()), "first_row": int(rows[0]) if len(rows) else -1})
    return CausalityReport(not violations, cuts, violations)


class LeakyExecutor:
    """Deliberately non-causal wrapper (normalizes each series by its full-sample mean) used to show
    that the truncation test catches look-ahead that grammar and AST checks cannot see."""

    def __init__(self, inner):
        self.inner = inner

    def evaluate(self, node, panel, mask_members: bool = True):
        x = self.inner.evaluate(node, panel, mask_members)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(x, axis=0, keepdims=True)
        return x - np.nan_to_num(mu)
