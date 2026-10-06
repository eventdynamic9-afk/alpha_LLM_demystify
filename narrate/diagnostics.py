"""A1 diagnostics table (§8.2), computed by the verifier on the training window: mean RankIC and its
t-stat, decile spread, lag-1 rank autocorrelation, turnover, top-5 correlations with the reference
library and factor loadings."""
from __future__ import annotations

import numpy as np

from dsl import Node
from verify.factors import decile_long_short
from verify.performance import long_short_backtest
from verify.stats import daily_spearman, hac_ols, lag1_rank_autocorr, newey_west_mean


def diagnostics(node: Node, ctx, window: str = "train", top_k: int = 5) -> dict:
    rows = ctx.rows(window)
    f = ctx.signal(node)
    ic = daily_spearman(f, ctx.fwd(1, "open_t+1"), rows)
    m, se, t = newey_west_mean(ic)
    ls = decile_long_short(f, ctx.fwd(1, "close_t"), ctx.panel.member)
    ac = lag1_rank_autocorr(f, rows)
    bt = long_short_backtest(f, ctx, rows, costs_bp={"buy": 0, "sell": 0})
    corrs = []
    for name in ctx.references.characteristic_names():
        r = daily_spearman(f, ctx.references.signal(name), rows)
        if np.isfinite(r).sum() > 20:
            corrs.append((name, float(np.nanmean(r))))
    corrs.sort(key=lambda x: -abs(x[1]))
    ctrl = ctx.factors.controls()
    reg = hac_ols(ls[rows], ctrl.to_numpy()[rows], names=list(ctrl.columns))
    loadings = {k: round(v, 3) for k, v in reg.get("params", {}).items() if k != "const"}
    return {"mean_rank_ic": float(m), "rank_ic_t": float(t), "decile_spread_daily": float(np.nanmean(ls[rows])),
            "lag1_rank_autocorr": float(np.nanmean(ac)), "turnover": float(np.nanmean(bt["turnover"])),
            "top_reference_correlations": corrs[:top_k], "factor_loadings": loadings}


def diagnostics_table(d: dict) -> str:
    lines = [f"mean RankIC: {d['mean_rank_ic']:.4f} (t = {d['rank_ic_t']:.2f})",
             f"decile spread (top minus bottom, daily): {d['decile_spread_daily']:.5f}",
             f"lag-1 rank autocorrelation: {d['lag1_rank_autocorr']:.3f}",
             f"turnover (daily, long-short): {d['turnover']:.3f}",
             "top-5 correlations with reference signals: " + ", ".join(f"{n} {r:+.2f}" for n, r in d["top_reference_correlations"]),
             "factor loadings: " + ", ".join(f"{k} {v:+.3f}" for k, v in d["factor_loadings"].items())]
    return "\n".join(lines)
