"""Originality claims (C3, §10.3).

NOVEL is SUPPORTED iff (i) the 95% bootstrap upper bound of max |rho_bar| against all reference
signals is < 0.50 and (ii) the formula's long-short return keeps a HAC t >= 3 alpha after controls
chosen by double-selection LASSO (Feng, Giglio & Xiu 2020) from the reference factor set.  It is
REFUTED if some reference signal has a |rho_bar| CI bound beyond 0.80 or an IDENTITY match exists.
BETTER_THAN is a paired out-of-sample IC comparison.
"""
from __future__ import annotations

import numpy as np

from .factors import decile_long_short, decile_portfolios
from .stats import bootstrap_matrix_means, bootstrap_mean_ci, daily_spearman, hac_ols, newey_west_mean
from .verdicts import REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict


def reference_factor_returns(ctx, rows: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(T, K) long-short returns of every characteristic + control factors, names, and (T, n) test assets."""
    fwd = ctx.fwd(1, "close_t")
    cols, names, assets = [], [], []
    for name in ctx.references.characteristic_names():
        sig = ctx.references.signal(name)
        cols.append(decile_long_short(sig, fwd, ctx.panel.member))
        names.append(f"LS_{name}")
        assets.append(decile_portfolios(sig, fwd, ctx.panel.member, 10))
    ctrl = ctx.factors.controls()
    for c in ctrl.columns:
        cols.append(ctrl[c].to_numpy())
        names.append(c)
    H = np.column_stack(cols)[rows]
    R = np.concatenate(assets, axis=1)[rows]
    return H, names, R


def fgx_double_selection(g: np.ndarray, H: np.ndarray, R: np.ndarray, seed: int = 0) -> dict:
    """Feng-Giglio-Xiu (2020) double selection.

    Cross-section over test assets i: LASSO 1 selects factors h whose covariances explain average
    returns; LASSO 2 selects factors whose covariances explain Cov(R_i, g).  Returns the union.
    """
    from sklearn.linear_model import LassoCV

    ok = np.isfinite(g) & np.all(np.isfinite(H), axis=1)
    keep_assets = np.all(np.isfinite(R[ok]), axis=0)
    g, H, R = g[ok], H[ok], R[ok][:, keep_assets]
    if len(g) < 60 or R.shape[1] < 10:
        return {"selected": list(range(H.shape[1])), "note": "insufficient data: all controls used"}
    Rc = R - R.mean(axis=0)
    cov_g = Rc.T @ (g - g.mean()) / (len(g) - 1)
    cov_h = Rc.T @ (H - H.mean(axis=0)) / (len(g) - 1)
    mean_r = R.mean(axis=0)
    Z = (cov_h - cov_h.mean(axis=0)) / np.where(cov_h.std(axis=0) > 0, cov_h.std(axis=0), 1.0)
    l1 = LassoCV(cv=5, random_state=seed, max_iter=20000).fit(Z, mean_r)
    l2 = LassoCV(cv=5, random_state=seed, max_iter=20000).fit(Z, cov_g)
    sel = sorted(set(np.flatnonzero(np.abs(l1.coef_) > 1e-12)) | set(np.flatnonzero(np.abs(l2.coef_) > 1e-12)))
    # second-stage cross-sectional estimate of lambda_g (reported)
    import statsmodels.api as sm

    X = np.column_stack([cov_g] + ([cov_h[:, sel]] if sel else []))
    res = sm.OLS(mean_r, sm.add_constant(X)).fit()
    return {"selected": [int(s) for s in sel], "lambda_g": float(res.params[1]), "lambda_g_t_ols": float(res.tvalues[1])}


def verify_novel(f: np.ndarray, ctx, node=None, window: str = "train") -> Verdict:
    cfg = ctx.thr["originality"]
    rows = ctx.rows(window)
    names = ctx.references.names()
    series, used = [], []
    for n in names:
        s = daily_spearman(f, ctx.references.signal(n), rows)
        if np.isfinite(s).sum() >= 30:
            series.append(s)
            used.append(n)
    if not series:
        return Verdict(UNVERIFIABLE, "originality", {"reason": "no comparable reference signals"})
    S = np.vstack(series)
    rbar = np.nanmean(S, axis=1)
    order = np.argsort(-np.abs(rbar))
    ev = {"n_refs": len(used), "top_refs": [[used[i], float(rbar[i])] for i in order[:5]]}
    # refutation: duplicate of an existing signal / identity match
    for i in order[:5]:
        if abs(rbar[i]) < cfg["duplicate_floor"] - 0.15:
            break
        x = np.sign(rbar[i]) * S[i]
        _, lo, _, _ = bootstrap_mean_ci(x, 0.95, ctx.n_boot(), ctx.seed)
        if lo >= cfg["duplicate_floor"]:
            ev["duplicate_of"] = used[i]
            return Verdict(REFUTED, "originality", ev)
    if node is not None:
        from .identity import identity_matches

        from pools.library import library

        top = {used[i] for i in order[:10]}
        cands = {k: v for k, v in library().items() if k in top}
        m = identity_matches(node, ctx, cands)
        if m:
            ev["identity_match"] = m
            return Verdict(REFUTED, "originality", ev)
    # (i) joint bootstrap of max |rho_bar| over references (common resampled dates)
    finite_cols = np.all(np.isfinite(S), axis=0)
    if finite_cols.sum() >= 30:
        boots = bootstrap_matrix_means(S[:, finite_cols], ctx.n_boot(), ctx.seed)
        upper = float(np.quantile(np.max(np.abs(boots), axis=1), 0.975))
    else:
        upper = float(np.max(np.abs(rbar)))
    ev["max_abs_rho_upper95"] = upper
    if upper >= cfg["novelty_ceiling"]:
        return Verdict(UNRESOLVED, "originality", {**ev, "reason": "correlation ceiling not met"})
    # (ii) alpha after double-selection controls
    H, hnames, R = reference_factor_returns(ctx, rows)
    g = decile_long_short(f, ctx.fwd(1, "close_t"), ctx.panel.member)[rows]
    try:
        ds = fgx_double_selection(g, H, R, ctx.seed)
    except Exception as exc:  # pragma: no cover - optional dependency path
        ds = {"selected": list(range(H.shape[1])), "note": f"selection failed: {exc}"}
    X = H[:, ds["selected"]] if ds["selected"] else np.zeros((len(g), 0))
    res = hac_ols(g, X if X.shape[1] else np.zeros((len(g), 1)), names=[hnames[i] for i in ds["selected"]] or ["zero"])
    if not res.get("ok"):
        return Verdict(UNRESOLVED, "originality", {**ev, "reason": "alpha regression failed"})
    t_alpha = res["t"]["const"]
    ev.update({"alpha": res["params"]["const"], "alpha_t": t_alpha,
               "controls": [hnames[i] for i in ds["selected"]], "fgx": {k: v for k, v in ds.items() if k != "selected"}})
    if abs(t_alpha) >= cfg["alpha_t"]:
        return Verdict(SUPPORTED, "originality", ev)
    return Verdict(UNRESOLVED, "originality", {**ev, "reason": "alpha t below 3 after controls"})


def verify_better_than(f: np.ndarray, ctx, ref: str, metric: str = "IC") -> Verdict:
    resolved = ctx.references.resolve(ref)
    if not resolved:
        return Verdict(UNVERIFIABLE, "paired_oos", {"reason": f"no reference for {ref!r}"})
    if not ctx.has_window("test"):
        return Verdict(UNVERIFIABLE, "paired_oos", {"reason": "no out-of-sample window"})
    name, s = resolved[0]
    g = s * ctx.references.signal(name)
    fwd = ctx.fwd(1, "open_t+1")
    ic_f, ic_g = daily_spearman(f, fwd), daily_spearman(g, fwd)
    tr = ctx.rows("train")
    sf = np.sign(np.nanmean(ic_f[tr])) or 1.0
    sg = np.sign(np.nanmean(ic_g[tr])) or 1.0
    te = ctx.rows("test")
    d = sf * ic_f[te] - sg * ic_g[te]
    m, se, t = newey_west_mean(d)
    ev = {"ref": name, "metric": metric, "mean_diff": m, "t": t}
    thr = ctx.thr["behavioral"]["comparative_t"]
    if not np.isfinite(t):
        return Verdict(UNVERIFIABLE, "paired_oos", ev)
    if t >= thr:
        return Verdict(SUPPORTED, "paired_oos", ev)
    if t <= -thr:
        return Verdict(REFUTED, "paired_oos", ev)
    return Verdict(UNRESOLVED, "paired_oos", ev)
