"""Same-panel reference library (§5.5, primary behavioral ground truth).

Computed on exactly the same stocks, dates and adjustments as the audited formulas:
* Appendix C canonical characteristics (DSL-defined, plus IVOL/BETA/SKEW computed with the
  equal-weighted market return), and
* the public formula libraries (Alpha101 / GTJA-191 / Alpha158 entries in ``pools.library``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsl import parse
from pools.library import library

from .stats import lag1_rank_autocorr


class ReferenceLibrary:
    def __init__(self, ctx, include_library: bool = True):
        self.ctx = ctx
        self.defs = dict(ctx.cb["reference_signals"])
        self.market = ctx.panel.market
        self.include_library = include_library
        self._cache: dict[str, np.ndarray] = {}
        self._turnover_pcts: dict[str, dict] = {}

    # ------------------------------------------------------------------ names
    def characteristic_names(self) -> list[str]:
        out = []
        for name, d in self.defs.items():
            mk = d.get("markets")
            if mk and self.market not in mk:
                continue
            out.append(name)
        return out

    def library_names(self) -> list[str]:
        return list(library()) if self.include_library else []

    def names(self) -> list[str]:
        return self.characteristic_names() + self.library_names()

    # ------------------------------------------------------------------ resolution
    def resolve(self, ref: str) -> list[tuple[str, int]]:
        """Codebook term / reference name / library id -> [(reference name, sign)]."""
        if ref in self.defs or ref in library():
            return [(ref, 1)]
        key = ref.strip().lower()
        terms = self.ctx.cb.get("terms", {})
        spec = terms.get(key)
        if spec is None:
            for k in sorted(terms, key=len, reverse=True):
                if k in key:
                    spec = terms[k]
                    break
        if spec is None:
            for name in self.defs:
                if name.lower() == key:
                    return [(name, 1)]
            return []
        refs = list(spec["refs"])
        if self.market == "CN":
            refs += list(spec.get("variants_cn", []))
        return [(r, int(spec.get("sign", 1))) for r in refs]

    # ------------------------------------------------------------------ signals
    def signal(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        if name in self.defs:
            d = self.defs[name]
            if "dsl" in d:
                v = self.ctx.signal(parse(d["dsl"]))
            else:
                v = self._computed(d["computed"], int(d["window"]))
        elif name in library():
            v = self.ctx.signal(library()[name].node)
        else:
            raise KeyError(f"unknown reference signal {name!r}")
        self._cache[name] = v
        return v

    def _computed(self, kind: str, w: int) -> np.ndarray:
        r = self.ctx.daily_returns()
        m = self.ctx.market_returns()
        R = pd.DataFrame(r)
        M = pd.Series(m)
        if kind == "skew":
            out = R.rolling(w, min_periods=w).skew().to_numpy()
        else:
            cov = R.rolling(w, min_periods=w).cov(M)
            var_m = M.rolling(w, min_periods=w).var()
            beta = cov.div(var_m, axis=0)
            if kind == "beta":
                out = beta.to_numpy()
            else:  # ivol: residual std of the 60-day market model, ddof = 2
                var_r = R.rolling(w, min_periods=w).var()
                resid_var = (var_r - beta.pow(2).mul(var_m, axis=0)) * (w - 1) / (w - 2)
                out = np.sqrt(resid_var.clip(lower=0)).to_numpy()
        out = np.where(self.ctx.panel.member, out, np.nan)
        out[~np.isfinite(out)] = np.nan
        return out

    # ------------------------------------------------------------------ turnover reference distribution
    def turnover_percentiles(self, window: str = "train") -> dict:
        """Mean lag-1 rank autocorrelation of every characteristic and base-set library signal (per window)."""
        if window not in self._turnover_pcts:
            rows = self.ctx.rows(window)
            vals = []
            names = self.characteristic_names() + [f.lib_id for f in library().values() if f.base]
            for n in names:
                ac = lag1_rank_autocorr(self.signal(n), rows)
                if np.isfinite(ac).sum() > 10:
                    vals.append(float(np.nanmean(ac)))
            vals = np.array(vals)
            self._turnover_pcts[window] = {
                "values": vals, "window": window,
                "p75": float(np.percentile(vals, self.ctx.thr["behavioral"]["turnover_low_percentile"])),
                "p25": float(np.percentile(vals, self.ctx.thr["behavioral"]["turnover_high_percentile"]))}
        return self._turnover_pcts[window]
