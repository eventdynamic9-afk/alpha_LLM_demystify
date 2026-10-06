"""Performance claims (C4, §10.4) — out of sample only: the test window and, separately, the post-cutoff
window H_post (§5.2; an H_post shorter than 6 months is reported as exploratory).

* "significant / robust": SUPPORTED iff the OOS Newey-West t (mean RankIC or long-short return) exceeds
  the HLZ bar 3.0; REFUTED iff the 95% interval of that t (t +/- 1.96) lies entirely at or below the bar;
  otherwise UNRESOLVED.  Whether t > 2.0 is reported, never used for verdicts (§10.6).
* Sharpe-type claims: Deflated Sharpe Ratio with the recorded number of trials and the variance of the
  logged candidates' Sharpe ratios on their selection window (Bailey & Lopez de Prado 2014)
* families of candidates (P1-mined, P3a GP): PBO via CSCV (S = 16) — ``family_report``; "best of"
  claims: Romano-Wolf (2005) stepdown decides, White (2000) Reality Check / Hansen (2005) SPA reported
* transaction costs (AlphaAgent convention: CN 5 bp buy / 15 bp sell; US 5 bp sell) decide; Alpha
  Jungle's 15 bp per trade is reported as sensitivity; China limit-locked days are untradable.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

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
    return float(math.sqrt(max(var_trials, 0.0)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


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


# ------------------------------------------------------------------------------- recorded trials (§10.4)
SELECTION_WINDOW = {"P1": "valid", "P3a": "train"}      # where each protocol selected its candidates (§7.1)


def trial_family_key(rec: dict) -> str | None:
    """Mining run a record was selected from: ``meta.trial_family``; else one GP run per P3a seed; else the
    record's own refinement log (P1-mined)."""
    meta = rec.get("meta") or {}
    if meta.get("trial_family"):
        return str(meta["trial_family"])
    if rec.get("pool") == "P3a":
        return f"P3a-seed{rec.get('seed')}"
    if meta.get("trial_log"):
        return str(rec["formula_id"])
    return None


def _log_formulas(rows) -> list[str]:
    return [str(t["formula"]) for t in rows or [] if t.get("formula") and t.get("valid", True) is not False]


def read_trial_logs(path: str | Path) -> dict[str, list[str]]:
    """JSONL rows {"family": key (or "formula_id"), "formula": dsl, ...} -> family -> candidate formulas."""
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                key = r.get("family") or r.get("formula_id")
                if key and r.get("formula") and r.get("valid", True) is not False:
                    out.setdefault(str(key), []).append(str(r["formula"]))
    return out


def trial_formulas(rec: dict, trial_logs: dict[str, list[str]] | None = None) -> list[str]:
    """Candidate formulas logged behind a record (deduplicated, the record's own formula included)."""
    meta = rec.get("meta") or {}
    cands = _log_formulas(meta.get("trial_log"))
    key = trial_family_key(rec)
    if trial_logs and key in trial_logs:
        cands += trial_logs[key]
    if not cands and meta.get("trial_log_path") and Path(meta["trial_log_path"]).exists():
        cands = read_trial_logs(meta["trial_log_path"]).get(key, [])
    if cands:
        cands = cands + [rec["dsl"]]
    return list(dict.fromkeys(cands))


def evaluate_uncached(ctx, formula: str) -> np.ndarray:
    """Evaluate a logged candidate without filling the context's signal cache (families can be large)."""
    import warnings

    from dsl import parse

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ctx.executor.evaluate(parse(formula), ctx.panel)


def candidate_returns(formulas: list[str], ctx, rows: np.ndarray, costs_bp: dict | None = None,
                      per_trade_bp: float | None = None,
                      min_coverage: float = 0.8) -> tuple[np.ndarray, list[str], list[str]]:
    """(T_w, N) net daily long-short returns of candidate formulas on ``rows`` (sign-aligned by their mean
    RankIC on the same rows, as the selection did); candidates that do not parse or cover < min_coverage of
    the days are skipped.  Days without a position count as flat (0)."""
    fwd = ctx.fwd(1, "open_t+1")
    cols, kept, skipped = [], [], []
    for fml in formulas:
        try:
            sig = evaluate_uncached(ctx, fml)
        except Exception:                          # noqa: BLE001 - unparsable / invalid logged candidate
            skipped.append(fml)
            continue
        s = np.sign(np.nanmean(daily_spearman(sig, fwd, rows))) if np.isfinite(sig[rows]).any() else 0.0
        if not np.isfinite(s) or s == 0:
            s = 1.0
        net = long_short_backtest(s * sig, ctx, rows, costs_bp=costs_bp, per_trade_bp=per_trade_bp)["net"][rows]
        if np.isfinite(net).mean() < min_coverage:
            skipped.append(fml)
            continue
        cols.append(np.nan_to_num(net, nan=0.0))
        kept.append(fml)
    M = np.column_stack(cols) if cols else np.zeros((int(rows.sum()), 0))
    return M, kept, skipped


def trial_sharpe_stats(formulas: list[str], ctx, window: str = "train") -> dict:
    """Variance of the per-period net Sharpe ratios of the logged candidates on their selection window."""
    if not formulas or not ctx.has_window(window):
        return {"var_trials": None, "n_evaluated": 0, "window": window}
    M, kept, skipped = candidate_returns(formulas, ctx, ctx.rows(window))
    srs = np.array([sharpe(M[:, k]) for k in range(M.shape[1])])
    srs = srs[np.isfinite(srs)]
    var = float(np.var(srs, ddof=1)) if len(srs) >= 2 else None
    return {"var_trials": var, "n_evaluated": int(len(srs)), "n_skipped": len(skipped), "window": window}


def record_trials(rec: dict | None, ctx, trial_logs: dict[str, list[str]] | None = None,
                  cache: dict | None = None) -> dict:
    """The recorded search behind a formula record (§10.4): number of trials (``rec['trials']``), the
    variance of the logged candidates' Sharpe ratios (None when no candidate log is available) and the
    candidate formulas themselves (for "best of" claims).  ``cache`` shares a family's statistics across
    the records selected from it (one GP run -> many P3a formulas)."""
    if not rec:
        return {"n_trials": 1, "var_trials": 0.0, "formulas": [], "source": "no record"}
    cands = trial_formulas(rec, trial_logs)
    n = rec.get("trials")
    n = int(n) if n else max(1, len(cands))
    out = {"n_trials": n, "formulas": cands, "family": trial_family_key(rec), "source": "record"}
    if n <= 1:
        out.update(var_trials=0.0)
    else:
        w = SELECTION_WINDOW.get(rec.get("pool"), "train")
        w = w if ctx.has_window(w) else "train"
        key = (out["family"], w, hash(tuple(sorted(set(cands)))))
        if cache is None or key not in cache:
            st = trial_sharpe_stats(cands, ctx, w)
            if cache is not None:
                cache[key] = st
        st = dict(cache[key]) if cache is not None else st
        out.update(var_trials=st.pop("var_trials"), trial_sharpe=st)
    return out


# ------------------------------------------------------------------------------- PERF claims
def oos_windows(ctx) -> list[str]:
    """Out-of-sample windows: test and H_post (kept even when shorter than 6 months — exploratory)."""
    out = [w for w in ("test", "post") if w in ctx.windows and ctx.has_window(w)]
    return out


def post_exploratory(ctx) -> bool:
    """§5.2: H_post shorter than ``min_post_cutoff_months`` is reported as exploratory."""
    if "post" not in ctx.windows:
        return False
    start, end = ctx.windows["post"]
    months = (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44
    return months < ctx.thr["performance"]["min_post_cutoff_months"]


def hlz_verdict(t: float, cfg: dict) -> str | None:
    """HLZ rule (§10.4, §10.6): SUPPORTED iff t > 3.0; REFUTED iff the 95% interval t +/- z lies entirely
    at or below 3.0; None (-> UNVERIFIABLE) if t is not finite; otherwise UNRESOLVED."""
    if not np.isfinite(t):
        return None
    bar = cfg["discovery_t"]
    if t > bar:
        return SUPPORTED
    z = float(sps.norm.ppf(0.5 + cfg.get("refute_ci_level", 0.95) / 2))
    return REFUTED if t + z <= bar else UNRESOLVED


def _t_evidence(m: float, se: float, t: float, cfg: dict) -> dict:
    z = float(sps.norm.ppf(0.5 + cfg.get("refute_ci_level", 0.95) / 2))
    return {"mean": m, "se": se, "t": t, "t_ci": [t - z, t + z],
            "t_above_reported_bar": bool(np.isfinite(t) and t > cfg["reported_t"])}   # reported, not a verdict input


def _ls_stats(signal: np.ndarray, ctx, rows: np.ndarray, n_trials: int, var_trials: float | None,
              per_trade_bp: float | None = None) -> dict:
    bt = long_short_backtest(signal, ctx, rows, per_trade_bp=per_trade_bp)
    d = deflated_sharpe_ratio(bt["net"], max(1, n_trials), 0.0 if var_trials is None else var_trials)
    m, se, t = newey_west_mean(bt["net"])
    return {"dsr": d, "ls_mean_net": m, "ls_se": se, "ls_t": t, "sharpe_net": sharpe(bt["net"]),
            "mean_turnover": float(np.nanmean(bt["turnover"])) if np.isfinite(bt["turnover"]).any() else float("nan")}


def _perf_window(f: np.ndarray, ic: np.ndarray, s: float, ctx, w: str, metric: str, n_trials: int,
                 var_trials: float | None) -> tuple[str, dict]:
    cfg = ctx.thr["performance"]
    rows = ctx.rows(w)
    if metric in ("ic", "rankic", "rank_ic", "significance"):
        m, se, t = newey_west_mean(s * ic[rows])
        ev = {"mean_ic": float(s * m) if np.isfinite(m) else m, **_t_evidence(float(s * m), se, t, cfg)}
        v = hlz_verdict(t, cfg)
        return (v or UNVERIFIABLE), (ev if v else {**ev, "reason": "OOS t not computable"})
    if metric == "stability":
        years = ctx.panel.years()
        overall = np.nanmean(s * ic[rows])
        ys = [np.nanmean(s * ic[rows & (years == y)]) for y in np.unique(years[rows])]
        ys_ok = [v for v in ys if np.isfinite(v)]
        if not ys_ok or not np.isfinite(overall):
            return UNVERIFIABLE, {"reason": "no finite yearly IC"}
        share = float(np.mean([np.sign(v) == np.sign(overall) for v in ys_ok]))
        ev = {"yearly_mean_ic": [float(v) for v in ys], "same_sign_share": share}
        if share >= cfg["stability_supported_share"]:
            return SUPPORTED, ev
        if share <= cfg["stability_refuted_share"]:
            return REFUTED, ev
        return UNRESOLVED, ev
    # Sharpe / return claims: AlphaAgent cost convention decides; 15 bp per trade is the sensitivity
    main = _ls_stats(s * f, ctx, rows, n_trials, var_trials)
    sens = _ls_stats(s * f, ctx, rows, n_trials, var_trials, per_trade_bp=cfg["costs_bp"]["sensitivity_per_trade"])
    ev = {**main, "costs": ctx.thr["performance"]["costs_bp"].get(ctx.panel.market, {"buy": 0, "sell": 5}),
          "cost_sensitivity": {"per_trade_bp": cfg["costs_bp"]["sensitivity_per_trade"], **sens},
          **{k: v for k, v in _t_evidence(main["ls_mean_net"], main["ls_se"], main["ls_t"], cfg).items()
             if k in ("t_ci", "t_above_reported_bar")}}
    if metric == "sharpe":
        dsr = main["dsr"].get("dsr", float("nan"))
        if not np.isfinite(dsr):
            return UNVERIFIABLE, {**ev, "reason": "DSR not computable (too few OOS days)"}
        if var_trials is None and n_trials > 1:
            # no candidate log: SR0 unknown; the undeflated PSR bounds the DSR from above
            ev["dsr_note"] = f"{n_trials} recorded trials but no candidate log: SR0 unknown, DSR <= PSR = {dsr:.3f}"
            return (REFUTED if dsr < cfg["dsr_refuted"] else UNRESOLVED), ev
        if dsr >= cfg["dsr_supported"]:
            return SUPPORTED, ev
        if dsr < cfg["dsr_refuted"]:
            return REFUTED, ev
        return UNRESOLVED, ev
    v = hlz_verdict(main["ls_t"], cfg)
    return (v or UNVERIFIABLE), (ev if v else {**ev, "reason": "OOS t not computable"})


def _by_window(ctx, fn, method: str, base_ev: dict) -> Verdict:
    """Primary verdict on the test window; H_post decided separately and reported (§10.4)."""
    wins = oos_windows(ctx)
    post_ev = {"post_window": list(ctx.windows["post"]) if "post" in ctx.windows else None,
               "post_status": getattr(ctx, "post_status", None)}
    if not wins:
        return Verdict(UNVERIFIABLE, method, {**base_ev, **post_ev, "reason": "no out-of-sample window"})
    by = {}
    for w in wins:
        v, ev = fn(w)
        by[w] = {"verdict": v, **ev}
        if w == "post":
            by[w]["exploratory"] = post_exploratory(ctx)
    primary = "test" if "test" in by else wins[0]
    ev = {**base_ev, "windows": wins, "primary_window": primary, "by_window": by, **post_ev,
          **{k: v for k, v in by[primary].items() if k != "verdict"}}
    if "post" in by:
        ev["post_verdict"] = by["post"]["verdict"]
        ev["post_exploratory"] = by["post"]["exploratory"]
    return Verdict(by[primary]["verdict"], method, ev)


def verify_perf(f: np.ndarray, ctx, metric: str = "IC", level: str = "high", n_trials: int = 1,
                var_trials: float | None = 0.0, trials: dict | None = None) -> Verdict:
    """``trials`` (``record_trials``) overrides ``n_trials`` / ``var_trials``; ``var_trials=None`` means
    the recorded trials have no candidate log (DSR cannot be deflated: only refutation is decidable)."""
    if trials is not None:
        n_trials, var_trials = int(trials.get("n_trials", 1)), trials.get("var_trials", 0.0)
    metric = (metric or "IC").lower()
    if metric not in ("ic", "rankic", "rank_ic", "significance", "stability", "sharpe", "returns", "return"):
        return Verdict(UNVERIFIABLE, "oos_performance", {"reason": f"unknown metric {metric!r}"})
    ic = daily_spearman(f, ctx.fwd(1, "open_t+1"))
    tr = ic[ctx.rows("train")] if ctx.has_window("train") else ic[:0]
    s = np.sign(np.nanmean(tr)) if np.isfinite(tr).any() else 1.0
    s = s if np.isfinite(s) and s != 0 else 1.0
    base = {"metric": metric, "level": level, "n_trials": n_trials, "var_trials": var_trials,
            "trials_source": (trials or {}).get("source", "not supplied: n_trials / var_trials arguments"),
            **({"trial_sharpe": trials["trial_sharpe"]} if trials and "trial_sharpe" in trials else {})}
    return _by_window(ctx, lambda w: _perf_window(f, ic, s, ctx, w, metric, n_trials, var_trials), "oos_performance", base)


# ------------------------------------------------------------------------------- "best of" claims
def best_of_set(ctx, text: str) -> str | None:
    """Codebook ``best_of_sets``: 'library' (every same-panel reference characteristic) or 'candidates'
    (the formula's logged mining candidates) when the claim's ref / level names a set (exact match)."""
    t = str(text).strip().lower()
    for name, words in (ctx.cb.get("best_of_sets") or {}).items():
        if t == name or t in [w.lower() for w in words]:
            return name
    return None


def _competitors(f: np.ndarray, ctx, which: str, trials: dict | None) -> tuple[dict[str, np.ndarray], str | None]:
    if which == "library":
        return {n: ctx.references.signal(n) for n in ctx.references.characteristic_names()}, None
    forms = (trials or {}).get("formulas") or []
    if not forms:
        return {}, "no candidate log for this formula (trial log not recorded)"
    out = {}
    for fml in forms:
        try:
            g = evaluate_uncached(ctx, fml)
        except Exception:                          # noqa: BLE001
            continue
        if g.shape == f.shape and np.array_equal(np.nan_to_num(g, nan=1e300), np.nan_to_num(f, nan=1e300)):
            continue                               # the formula itself
        out[fml] = g
    return out, None if out else "no evaluable competitor"


def verify_best_of(f: np.ndarray, ctx, which: str, metric: str = "IC", trials: dict | None = None) -> Verdict:
    """"Best of a set" claims (§10.4): daily OOS differentials d_k = perf(f) - perf(g_k) against every
    member g_k of the set (RankIC, or net long-short return for Sharpe/return metrics; each sign-aligned by
    its training-window RankIC).  Romano-Wolf stepdown decides: SUPPORTED iff every H0_k: E[d_k] <= 0 is
    rejected at FWER alpha; REFUTED iff some member is significantly better (stepdown on -d); else
    UNRESOLVED.  White's Reality Check and Hansen's SPA p-values (H0: no member beats f) are reported, and
    for the candidate family the SPA of its best member against a zero benchmark."""
    comps, why = _competitors(f, ctx, which, trials)
    base = {"set": which, "metric": metric, "n_competitors": len(comps)}
    if why:
        return Verdict(UNVERIFIABLE, "best_of", {**base, "reason": why})
    cfg = ctx.thr["performance"]
    alpha = float(cfg.get("best_of_alpha", 0.05))
    fwd = ctx.fwd(1, "open_t+1")
    tr = ctx.rows("train") if ctx.has_window("train") else np.ones(ctx.panel.T, dtype=bool)
    metric_l = (metric or "IC").lower()

    def aligned(x):
        ic = daily_spearman(x, fwd)
        sg = np.sign(np.nanmean(ic[tr]))
        return (sg if np.isfinite(sg) and sg != 0 else 1.0), ic

    sf, ic_f = aligned(f)
    al = {k: aligned(g) for k, g in comps.items()}

    def run(w):
        rows = ctx.rows(w)
        if metric_l in ("sharpe", "returns", "return"):
            perf = lambda x, s: long_short_backtest(s * x, ctx, rows)["net"][rows]   # noqa: E731
            pf = perf(f, sf)
            cols = {k: perf(comps[k], s) for k, (s, _) in al.items()}
        else:
            pf = sf * ic_f[rows]
            cols = {k: s * ic[rows] for k, (s, ic) in al.items()}
        names = [k for k, c in cols.items() if (np.isfinite(c) & np.isfinite(pf)).mean() >= 0.8]
        if not names:
            return UNVERIFIABLE, {"reason": "no competitor covers the window"}
        D = np.column_stack([pf - cols[k] for k in names])
        keep = np.all(np.isfinite(D), axis=1)
        if keep.sum() < 30:
            return UNVERIFIABLE, {"reason": "too few common OOS days"}
        D = D[keep]
        nb = ctx.n_boot()
        rw = romano_wolf(D, nb, ctx.seed)
        rw_neg = romano_wolf(-D, nb, ctx.seed)
        ev = {"competitors": names[:50], "n_used": len(names), "alpha": alpha,
              "rw_p_adjusted_max": float(max(rw["p_adjusted"])), "rw_better_than_all": bool(max(rw["p_adjusted"]) <= alpha),
              "rw_p_member_better_min": float(min(rw_neg["p_adjusted"])),
              "reality_check_p_member_better": white_reality_check(-D, nb, ctx.seed)["p_value"],
              "spa_p_member_better": hansen_spa(-D, nb, ctx.seed)["p_value"],
              "mean_diff": D.mean(axis=0).tolist()[:50]}
        if which == "candidates":
            F = np.column_stack([pf[keep]] + [cols[k][keep] for k in names])
            ev["spa_best_vs_zero_p"] = hansen_spa(F, nb, ctx.seed)["p_value"]
        if max(rw["p_adjusted"]) <= alpha:
            return SUPPORTED, ev
        if min(rw_neg["p_adjusted"]) <= alpha:
            return REFUTED, ev
        return UNRESOLVED, ev

    return _by_window(ctx, run, "best_of", base)


# ------------------------------------------------------------------------------- families (PBO)
def family_report(formulas: list[str], ctx, window: str = "train", S: int | None = None) -> dict:
    """§10.4 family-level statistics for a mining run's logged candidates on its selection window: PBO via
    CSCV (S partitions, net returns under the AlphaAgent cost convention and at 15 bp per trade), and
    White RC / Hansen SPA p-values of the best candidate against a zero benchmark."""
    cfg = ctx.thr["performance"]
    S = int(S or cfg["pbo_partitions"])
    if not ctx.has_window(window):
        return {"status": "not_run", "reason": f"no {window!r} window"}
    rows = ctx.rows(window)
    M, kept, skipped = candidate_returns(formulas, ctx, rows)
    out = {"window": window, "n_logged": len(formulas), "n_evaluated": len(kept), "n_skipped": len(skipped), "S": S}
    if M.shape[1] < 2:
        return {**out, "status": "not_run", "reason": "fewer than 2 evaluable candidates"}
    out["pbo"] = pbo_cscv(M, S=S)
    M15, _, _ = candidate_returns(kept, ctx, rows, per_trade_bp=cfg["costs_bp"]["sensitivity_per_trade"])
    out["pbo_cost_sensitivity"] = {"per_trade_bp": cfg["costs_bp"]["sensitivity_per_trade"], **pbo_cscv(M15, S=S)}
    srs = np.array([sharpe(M[:, k]) for k in range(M.shape[1])])
    nb = ctx.n_boot()
    out.update({"status": "ok", "best_formula": kept[int(np.nanargmax(srs))], "best_sharpe": float(np.nanmax(srs)),
                "var_sharpe": float(np.nanvar(srs, ddof=1)),
                "reality_check_p": white_reality_check(M, nb, ctx.seed)["p_value"],
                "spa_p": hansen_spa(M, nb, ctx.seed)["p_value"]})
    return out
