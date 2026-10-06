"""Pre-registered confirmatory family (§12.5), Holm-corrected at FWER 0.05.

CF1 (RQ1) CP(P3) < CP(P1), complexity-adjusted
CF2 (RQ2) CP(A2) > CP(A0) for C2-C4 claims
CF3 (RQ3) CP on targeted-property claims lower in SA than in K (paired by base formula)
CF4 (RQ3) label-following rate in NL > its rate in K for the same property (paired)
CF5 (RQ4) holistic judge (B1) AUROC for detecting rationales with >= 1 REFUTED claim < 0.90 (one-sided)
CF6 (RQ5) CP(N) != CP(K), complexity-matched (two-sided)

Primary p-values come from the claim-level GLMM (§12.2) fitted to the contrast's data (posterior z of
the condition effect); the two-stage cluster bootstrap is reported alongside and used if the GLMM fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from configs import thresholds

from .bootstrap import bootstrap_difference, micro_precision
from .metrics import TARGETED, label_following
from .multiplicity import holm

CF_SPECS = {
    "CF1": {"rq": "RQ1", "hypothesis": "CP(P3) < CP(P1), complexity-adjusted", "sided": "less"},
    "CF2": {"rq": "RQ2", "hypothesis": "CP(A2) > CP(A0) for C2-C4 claims", "sided": "greater"},
    "CF3": {"rq": "RQ3", "hypothesis": "CP(targeted claims | SA) < CP(same claims | K), paired", "sided": "less"},
    "CF4": {"rq": "RQ3", "hypothesis": "label-following(NL) > label-following(K), paired", "sided": "greater"},
    "CF5": {"rq": "RQ4", "hypothesis": "AUROC(B1 detects refuted rationales) < 0.90", "sided": "less"},
    "CF6": {"rq": "RQ5", "hypothesis": "CP(N) != CP(K), complexity-matched", "sided": "two"},
}


def _glmm_contrast(d: pd.DataFrame, group_col: str, treat: str, control: str, sided: str, base_effect: bool = False) -> dict:
    """GLMM y ~ treat_indicator + complexity (+ type), random intercepts formula / rationale [/ base]."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    x = d[d["decidable"] & d[group_col].isin([treat, control])].copy()
    if x[group_col].nunique() < 2 or len(x) < 20:
        raise ValueError("insufficient data")
    x["y"] = x["supported"].astype(float)
    x["treat"] = (x[group_col] == treat).astype(float)
    x["complexity"] = (x["nodes"] - x["nodes"].mean()) / (x["nodes"].std() or 1.0)
    terms = ["treat", "complexity"]
    if x["type"].nunique() > 1:
        terms.append("C(type)")
    vc = {"formula": "0 + C(formula_id)", "rationale": "0 + C(rationale_id)"}
    if base_effect:
        vc["base"] = "0 + C(base_id)"
    mod = BinomialBayesMixedGLM.from_formula("y ~ " + " + ".join(terms), vc, x)
    res = mod.fit_vb()
    i = mod.exog_names.index("treat")            # patsy orders categorical terms first: look up by name
    b, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
    z = b / sd
    p = {"less": sps.norm.cdf(z), "greater": sps.norm.sf(z), "two": 2 * sps.norm.sf(abs(z))}[sided]
    return {"coef_logit": b, "sd": sd, "z": z, "p": float(p), "n_claims": len(x)}


def _bootstrap_p(boot: dict, sided: str) -> float:
    return {"less": boot.get("p_less"), "greater": boot.get("p_greater"), "two": boot.get("p_two_sided")}[sided]


def _contrast(d: pd.DataFrame, group_col: str, treat: str, control: str, sided: str, n_boot: int, seed: int,
              paired: bool = False) -> dict:
    boot = bootstrap_difference(d[d[group_col].isin([treat, control])], group_col, treat, control, micro_precision,
                                cluster="base_id" if paired else "formula_id", n_boot=n_boot, seed=seed, paired=paired)
    out = {"cp_treat": micro_precision(d[d[group_col] == treat]), "cp_control": micro_precision(d[d[group_col] == control]),
           "diff_pp": 100 * boot["estimate"], "diff_ci_pp": [100 * c for c in boot["ci"]],
           "p_bootstrap": _bootstrap_p(boot, sided)}
    try:
        g = _glmm_contrast(d, group_col, treat, control, sided, base_effect=paired)
        out.update({"glmm": g, "p": g["p"], "p_source": "glmm"})
    except Exception as exc:
        out.update({"glmm_error": str(exc), "p": out["p_bootstrap"], "p_source": "bootstrap"})
    return out


def cf1(C: pd.DataFrame, n_boot: int, seed: int) -> dict:
    d = C[(C["arm"] == "A") & (C["access"] == "A0") & (C["prompt_variant"] == "guided")].copy()
    d["grp"] = np.where(d["condition"].str.startswith("P1"), "P1", np.where(d["condition"] == "P3", "P3", "other"))
    return _contrast(d, "grp", "P3", "P1", "less", n_boot, seed)


def cf2(C: pd.DataFrame, n_boot: int, seed: int) -> dict:
    d = C[(C["arm"] == "A") & C["type"].isin(["C2", "C3", "C4"]) & (C["prompt_variant"] == "guided")]
    return _contrast(d, "access", "A2", "A0", "greater", n_boot, seed)


def cf3(C: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rows = []
    sa = C[(C["pool"] == "SA") & (C["access"] == "A0")]
    k = C[(C["pool"] == "K") & (C["access"] == "A0")]
    for (base, t), g in sa.groupby(["base_id", "variant_type"]):
        preds = TARGETED.get(t, set())
        rows.append(g[g["predicate"].isin(preds)].assign(grp="SA"))
        rows.append(k[(k["base_id"] == base) & k["predicate"].isin(preds)].assign(grp="K"))
    d = pd.concat(rows) if rows else C.iloc[:0].assign(grp=[])
    d = d.drop_duplicates(subset=["claim_id", "grp"])
    return _contrast(d, "grp", "SA", "K", "less", n_boot, seed, paired=True)


def cf4(C: pd.DataFrame, F: pd.DataFrame, n_boot: int, seed: int) -> dict:
    lf = label_following(C, F)
    if not len(lf):
        return {"p": float("nan"), "reason": "no NL data"}
    piv = lf.pivot_table(index=["base_id", "model"], columns="pool", values="label_following_rate").dropna()
    if not len(piv) or "NL" not in piv or "K" not in piv:
        return {"p": float("nan"), "reason": "no paired NL/K cells"}
    diff = (piv["NL"] - piv["K"]).groupby(level=0).mean()          # per base formula
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(diff.values, len(diff)).mean() for _ in range(n_boot)])
    p_boot = float((boots <= 0).mean())
    w = sps.wilcoxon(diff.values, alternative="greater") if (diff != 0).any() and len(diff) > 1 else None
    return {"rate_nl": float(piv["NL"].mean()), "rate_k": float(piv["K"].mean()), "diff_pp": 100 * float(diff.mean()),
            "diff_ci_pp": [100 * float(np.quantile(boots, 0.025)), 100 * float(np.quantile(boots, 0.975))],
            "p": p_boot, "p_source": "paired_bootstrap", "p_wilcoxon": float(w.pvalue) if w is not None else float("nan"),
            "n_bases": int(len(diff))}


def cf5(b1: list[dict], truth: dict, n_boot: int, seed: int, bound: float | None = None) -> dict:
    from judges.validity import auroc

    bound = thresholds()["judges"]["cf5_auroc_bound"] if bound is None else bound
    rows = [(j["C"], int(truth[j["rationale_id"]]["has_refuted"])) for j in b1
            if j["rationale_id"] in truth and truth[j["rationale_id"]]["decidable"] > 0 and j.get("C") == j.get("C")]
    if len(rows) < 10:
        return {"p": float("nan"), "reason": "insufficient judged rationales"}
    s = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    est = auroc(-s, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        a = auroc(-s[i], y[i])
        if a == a:
            boots.append(a)
    boots = np.array(boots)
    return {"auroc": est, "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], "bound": bound,
            "p": float((boots >= bound).mean()), "p_source": "bootstrap", "n": len(rows)}


def cf6(C: pd.DataFrame, n_boot: int, seed: int) -> dict:
    d = C[C["pool"].isin(["N", "K"]) & (C["access"] == "A0")]
    return _contrast(d, "pool", "N", "K", "two", n_boot, seed)


def run_confirmatory(C: pd.DataFrame, F: pd.DataFrame, b1: list[dict] | None = None, truth: dict | None = None,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    res = {"CF1": cf1(C, n_boot, seed), "CF2": cf2(C, n_boot, seed), "CF3": cf3(C, n_boot, seed),
           "CF4": cf4(C, F, n_boot, seed),
           "CF5": cf5(b1, truth, n_boot, seed) if b1 and truth else {"p": float("nan"), "reason": "no B1 judgements"},
           "CF6": cf6(C, n_boot, seed)}
    keys = list(res)
    ps = [res[k].get("p", float("nan")) for k in keys]
    finite = [i for i, p in enumerate(ps) if p == p]
    alpha = thresholds()["multiplicity"]["confirmatory_fwer"]
    h = holm([ps[i] for i in finite], alpha) if finite else {"p_adjusted": [], "reject": []}
    for j, i in enumerate(finite):
        res[keys[i]]["p_holm"] = h["p_adjusted"][j]
        res[keys[i]]["reject_h0"] = h["reject"][j]
    for k in keys:
        res[k].update(CF_SPECS[k])
    return res
