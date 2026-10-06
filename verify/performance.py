"""Performance claims (C4, §10.4) — out of sample only (test window and post-cutoff window H_post).

* "significant / robust": HLZ t > 3.0 on the OOS mean RankIC or long-short return (Newey-West)
* Sharpe-type claims: Deflated Sharpe Ratio with the actual number of trials (Bailey & Lopez de Prado 2014)
* families of candidates: PBO via CSCV (S = 16), White (2000) Reality Check, Hansen (2005) SPA,
  Romano-Wolf (2005) stepdown
* transaction costs (CN 5 bp buy / 15 bp sell; US 5 bp sell; 15 bp per trade sensitivity) and
  China limit-locked days treated as untradable.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from data.labels import limit_locked

from .stats import daily_spearman, newey_west_mean, optimal_block_length, stationary_indices
from .verdicts import REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict

EULER_GAMMA = 0.5772156649015329


# ------------------------------------------------------------------------------- backtest
def long_short_backtest(signal: np.ndarray, ctx, rows: np.ndarray, q: float = 0.1, costs_bp: dict | None = None,
                        per_trade_bp: float | None = None) -> dict:
    """Daily-rebalanced equal-weight top-minus-bottom quantile portfolio with costs and limit locks."""
    p = ctx.panel
    if costs_bp is None:
        costs_bp = ctx.thr["performance"]["costs_bp"].get(p.market, {"buy": 0, "sell": 5})
    buy = (per_trade_bp if per_trade_bp is not None else costs_bp["buy"]) / 1e4
    sell = (per_trade_bp if per_trade_bp is not None else costs_bp["sell"]) / 1e4
    ret_next = ctx.fwd(1, "close_t")
    up, down = limit_locked(p)
    s = np.where(p.member & np.isfinite(signal), signal, np.nan)
    pct = pd.DataFrame(s).rank(axis=1, pct=True).to_numpy()
    T, N = s.shape
    w_prev = np.zeros(N)
    gross, net, turn = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    for t in np.flatnonzero(rows):
        top, bot = pct[t] > 1 - q, pct[t] <= q
        if top.sum() == 0 or bot.sum() == 0:
            continue
        w = np.zeros(N)
        w[top] = 1.0 / top.sum()
        w[bot] = -1.0 / bot.sum()
        locked = up[t] | down[t]
        if locked.any():          # cannot trade limit-locked names: keep yesterday's weight
            w[locked] = w_prev[locked]
        dw = w - w_prev
        cost = buy * np.clip(dw, 0, None).sum() + sell * np.clip(-dw, 0, None).sum()
        r = np.nan_to_num(ret_next[t], nan=0.0)
        gross[t] = float(w @ r)
        net[t] = gross[t] - cost
        turn[t] = float(np.abs(dw).sum())
        w_prev = w
    return {"gross": gross, "net": net, "turnover": turn}


def sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


# ------------------------------------------------------------------------------- DSR
def expected_max_sharpe(var_trials: float, n_trials: int) -> float:
    if n_trials <= 1:
        return 0.0
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(var_trials, 0.0)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, var_trials: float) -> dict:
    """DSR = Phi[(SR - SR0) sqrt(T-1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)] with per-period SR."""
    x = returns[np.isfinite(returns)]
    T = len(x)
    if T < 10:
        return {"dsr": float("nan")}
    sr = sharpe(x)
    g3 = float(sps.skew(x))
    g4 = float(sps.kurtosis(x, fisher=False))
    sr0 = expected_max_sharpe(var_trials, n_trials)
    den = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr * sr, 1e-12))
    dsr = float(sps.norm.cdf((sr - sr0) * math.sqrt(T - 1) / den))
    return {"dsr": dsr, "sr": sr, "sr0": sr0, "skew": g3, "kurtosis": g4, "T": T, "n_trials": n_trials}


# ------------------------------------------------------------------------------- PBO / CSCV
def pbo_cscv(M: np.ndarray, S: int = 16, metric: str = "sharpe") -> dict:
    """Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu 2017).

    M: (T, N) per-period returns of N strategy configurations.  Rows are split into S blocks; for every
    half/half combination the in-sample best configuration's out-of-sample relative rank w gives the
    logit lambda = log(w / (1 - w)); PBO = share of lambda <= 0.
    """
    M = M[np.all(np.isfinite(M), axis=1)]
    T, N = M.shape
    if N < 2 or T < 2 * S:
        return {"pbo": float("nan"), "reason": "insufficient data"}
    blocks = np.array_split(np.arange(T), S)
    s1 = np.stack([M[b].sum(axis=0) for b in blocks])
    s2 = np.stack([(M[b] ** 2).sum(axis=0) for b in blocks])
    n = np.array([len(b) for b in blocks])[:, None]
    lambdas = []
    for comb in itertools.combinations(range(S), S // 2):
        is_ = np.zeros(S, dtype=bool)
        is_[list(comb)] = True
        perf = []
        for part in (is_, ~is_):
            k = n[part].sum()
            mu = s1[part].sum(axis=0) / k
            var = s2[part].sum(axis=0) / k - mu ** 2
            perf.append(mu / np.sqrt(np.maximum(var, 1e-18)) if metric == "sharpe" else mu)
        best = int(np.argmax(perf[0]))
        rank = sps.rankdata(perf[1])[best]
        w = rank / (N + 1)
        lambdas.append(math.log(w / (1 - w)))
    lam = np.array(lambdas)
    return {"pbo": float((lam <= 0).mean()), "n_combinations": len(lam), "median_logit": float(np.median(lam))}


# ------------------------------------------------------------------------------- data snooping
def white_reality_check(D: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """D: (T, K) performance differentials vs the benchmark. H0: max_k E[d_k] <= 0."""
    D = D[np.all(np.isfinite(D), axis=1)]
    T, K = D.shape
    dbar = D.mean(axis=0)
    stat = math.sqrt(T) * dbar.max()
    b = float(np.median([optimal_block_length(D[:, k]) for k in range(K)]))
    idx = stationary_indices(T, b, n_boot, np.random.default_rng(seed))
    boots = np.stack([math.sqrt(T) * (D[i].mean(axis=0) - dbar).max() for i in idx])
    return {"stat": float(stat), "p_value": float((boots >= stat).mean()), "block": b}


def hansen_spa(D: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Hansen (2005) SPA test, consistent (c) version with studentized statistics."""
    D = D[np.all(np.isfinite(D), axis=1)]
    T, K = D.shape
    dbar = D.mean(axis=0)
    b = float(np.median([optimal_block_length(D[:, k]) for k in range(K)]))
    idx = stationary_indices(T, b, n_boot, np.random.default_rng(seed))
    bmeans = np.stack([D[i].mean(axis=0) for i in idx])
    omega = np.sqrt(T * ((bmeans - dbar) ** 2).mean(axis=0))
    omega = np.where(omega > 0, omega, 1e-12)
    stat = max(0.0, float(np.max(math.sqrt(T) * dbar / omega)))
    # Z*_k = dbar*_k - g_c(dbar_k), g_c(x) = x * 1{x >= -sqrt(omega^2 / T * 2 loglog T)}
    thresh = -np.sqrt(omega ** 2 / T * 2 * math.log(math.log(T)))
    g_c = np.where(dbar >= thresh, dbar, 0.0)
    z = math.sqrt(T) * (bmeans - g_c) / omega
    boots = np.maximum(0.0, z.max(axis=1))
    return {"stat": stat, "p_value": float((boots >= stat).mean()), "block": b}


def romano_wolf(D: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Romano-Wolf (2005) stepdown adjusted p-values for H0_k: E[d_k] <= 0 (studentized)."""
    D = D[np.all(np.isfinite(D), axis=1)]
    T, K = D.shape
    dbar = D.mean(axis=0)
    se = D.std(axis=0, ddof=1) / math.sqrt(T)
    se = np.where(se > 0, se, 1e-12)
    tstat = dbar / se
    b = float(np.median([optimal_block_length(D[:, k]) for k in range(K)]))
    idx = stationary_indices(T, b, n_boot, np.random.default_rng(seed))
    tb = np.stack([(D[i].mean(axis=0) - dbar) / se for i in idx])
    order = np.argsort(-tstat)
    adj = np.empty(K)
    prev = 0.0
    for r, k in enumerate(order):
        remaining = order[r:]
        maxb = tb[:, remaining].max(axis=1)
        p = float((maxb >= tstat[k]).mean())
        prev = max(prev, p)
        adj[k] = prev
    return {"t": tstat.tolist(), "p_adjusted": adj.tolist()}


# ------------------------------------------------------------------------------- PERF claims
def oos_windows(ctx) -> list[str]:
    out = []
    if ctx.has_window("test"):
        out.append("test")
    if "post" in ctx.windows and ctx.has_window("post"):
        start, end = ctx.windows["post"]
        months = (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44
        if months >= ctx.thr["performance"]["min_post_cutoff_months"]:
            out.append("post")
    return out


def verify_perf(f: np.ndarray, ctx, metric: str = "IC", level: str = "high", n_trials: int = 1,
                var_trials: float = 0.0) -> Verdict:
    cfg = ctx.thr["performance"]
    wins = oos_windows(ctx)
    if not wins:
        return Verdict(UNVERIFIABLE, "oos_performance", {"reason": "no out-of-sample window"})
    metric = (metric or "IC").lower()
    ev: dict = {"metric": metric, "level": level, "windows": wins}
    ic = daily_spearman(f, ctx.fwd(1, "open_t+1"))
    s = np.sign(np.nanmean(ic[ctx.rows("train")])) if ctx.has_window("train") else 1.0
    s = s or 1.0
    if metric in ("ic", "rankic", "rank_ic", "significance"):
        ts = {}
        for w in wins:
            m, se, t = newey_west_mean(s * ic[ctx.rows(w)])
            ts[w] = {"mean_ic": float(s * m), "t": t}
        ev["oos"] = ts
        t_test = ts["test"]["t"] if "test" in ts else ts[wins[0]]["t"]
        if all(v["t"] > cfg["discovery_t"] for v in ts.values()):
            return Verdict(SUPPORTED, "oos_performance", ev)
        if not np.isfinite(t_test) or t_test < cfg["refute_t"]:
            return Verdict(REFUTED, "oos_performance", ev)
        return Verdict(UNRESOLVED, "oos_performance", ev)
    if metric == "stability":
        rows = ctx.rows(wins[0])
        years = ctx.panel.years()
        overall = np.nanmean(s * ic[rows])
        ys = [np.nanmean(s * ic[rows & (years == y)]) for y in np.unique(years[rows])]
        share = float(np.mean([np.sign(v) == np.sign(overall) for v in ys if np.isfinite(v)]))
        ev.update({"yearly_mean_ic": [float(v) for v in ys], "same_sign_share": share})
        if share >= cfg["stability_supported_share"]:
            return Verdict(SUPPORTED, "oos_performance", ev)
        if share <= cfg["stability_refuted_share"]:
            return Verdict(REFUTED, "oos_performance", ev)
        return Verdict(UNRESOLVED, "oos_performance", ev)
    if metric in ("sharpe", "returns", "return"):
        bt = long_short_backtest(s * f, ctx, ctx.rows(wins[0]))
        d = deflated_sharpe_ratio(bt["net"], max(1, n_trials), var_trials)
        m, se, t = newey_west_mean(bt["net"])
        ev.update({"dsr": d, "ls_t": t, "mean_turnover": float(np.nanmean(bt["turnover"]))})
        if metric == "sharpe":
            if d.get("dsr", np.nan) >= cfg["dsr_supported"]:
                return Verdict(SUPPORTED, "oos_performance", ev)
            if not np.isfinite(d.get("dsr", np.nan)) or d["dsr"] < cfg["dsr_refuted"]:
                return Verdict(REFUTED, "oos_performance", ev)
            return Verdict(UNRESOLVED, "oos_performance", ev)
        if t > cfg["discovery_t"]:
            return Verdict(SUPPORTED, "oos_performance", ev)
        if not np.isfinite(t) or t < cfg["refute_t"]:
            return Verdict(REFUTED, "oos_performance", ev)
        return Verdict(UNRESOLVED, "oos_performance", ev)
    return Verdict(UNVERIFIABLE, "oos_performance", {"reason": f"unknown metric {metric!r}"})
