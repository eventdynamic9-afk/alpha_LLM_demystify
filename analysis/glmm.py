"""Primary claim-level model (§12.2):

  logit P(SUPPORTED) = condition + claim_type + condition x claim_type + complexity + hedge_level + market
                       + (1 | formula) + (1 | rationale) [+ (1 | base_formula) in Arm B]

with model as a fixed effect (few models) or a random effect (>= 8 models).  Python implementation:
statsmodels ``BinomialBayesMixedGLM`` (variational Bayes); marginal effects (percentage points) are
reported, not log-odds.  ``write_r_script`` emits the equivalent lme4 / glmmTMB + marginaleffects code.
A cluster-robust logistic regression is provided as a fast frequentist cross-check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps


def _prep(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    d = df[df["decidable"]].copy()
    d["y"] = d["supported"].astype(float)
    d["complexity"] = (d["nodes"] - d["nodes"].mean()) / (d["nodes"].std() or 1.0)
    for c in ["condition", "type", "hedge"] + extra:
        if c in d:
            d[c] = d[c].astype(str)
    return d


def fit_glmm(df: pd.DataFrame, condition_col: str = "condition", interaction: bool = True,
             model_random: bool | None = None, base_effect: bool = False) -> dict:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = _prep(df, ["model", "market"])
    d = d.rename(columns={condition_col: "cond"}) if condition_col != "cond" else d
    n_models = d["model"].nunique() if "model" in d else 1
    model_random = n_models >= 8 if model_random is None else model_random
    terms = ["C(cond)", "C(type)", "complexity"]
    if interaction and d["type"].nunique() > 1 and d["cond"].nunique() > 1:
        terms.append("C(cond):C(type)")
    if d["hedge"].nunique() > 1:
        terms.append("C(hedge)")
    if "market" in d and d["market"].nunique() > 1:
        terms.append("C(market)")
    if "model" in d and d["model"].nunique() > 1 and not model_random:
        terms.append("C(model)")
    vc = {"formula": "0 + C(formula_id)", "rationale": "0 + C(rationale_id)"}
    if base_effect and "base_id" in d:
        vc["base"] = "0 + C(base_id)"
    if model_random and "model" in d:
        vc["model"] = "0 + C(model)"
    formula = "y ~ " + " + ".join(terms)
    mod = BinomialBayesMixedGLM.from_formula(formula, vc, d)
    res = mod.fit_vb()
    names = mod.exog_names
    fe = pd.DataFrame({"mean": res.fe_mean, "sd": res.fe_sd}, index=names)
    fe["z"] = fe["mean"] / fe["sd"]
    fe["p_two_sided"] = 2 * sps.norm.sf(np.abs(fe["z"]))
    me = marginal_effects(mod, res, d, "cond")
    return {"formula": formula, "vc": list(vc), "fixed_effects": fe, "marginal": me, "n_claims": len(d),
            "n_rationales": d["rationale_id"].nunique(), "n_formulas": d["formula_id"].nunique(), "_mod": mod, "_res": res}


def marginal_effects(mod, res, d: pd.DataFrame, cond: str = "cond", n_draws: int = 400, seed: int = 0) -> pd.DataFrame:
    """Average predicted P(SUPPORTED) per condition (random effects at 0), with posterior draws of the
    fixed effects for intervals; returned in percentage points."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(res.fe_mean, res.fe_sd, size=(n_draws, len(res.fe_mean)))
    rows = []
    levels = sorted(d[cond].unique())
    base_exog = None
    for lev in levels:
        dd = d.copy()
        dd[cond] = lev
        X = _design(mod, dd)
        p_draws = 1 / (1 + np.exp(-(X @ beta.T)))
        avg = p_draws.mean(axis=0)
        p_hat = float((1 / (1 + np.exp(-(X @ res.fe_mean)))).mean())
        rows.append({"condition": lev, "pp": 100 * p_hat, "lo": 100 * np.quantile(avg, 0.025),
                     "hi": 100 * np.quantile(avg, 0.975), "_draws": avg})
        if base_exog is None:
            base_exog = avg
    out = pd.DataFrame(rows)
    out["diff_vs_first_pp"] = [100 * float(np.mean(r - base_exog)) for r in out["_draws"]]
    return out.drop(columns=["_draws"])


def _design(mod, dd: pd.DataFrame) -> np.ndarray:
    from patsy import build_design_matrices

    info = mod.data.design_info if hasattr(mod.data, "design_info") else mod.data.orig_exog.design_info
    return np.asarray(build_design_matrices([info], dd, return_type="dataframe")[0])


def cluster_robust_logit(df: pd.DataFrame, formula: str, cluster: str = "formula_id") -> dict:
    """Fast frequentist cross-check: logistic regression with cluster-robust (by formula) SEs."""
    import statsmodels.formula.api as smf

    d = _prep(df, ["model", "market"])
    groups = pd.factorize(d[cluster])[0]
    res = smf.glm(formula, d, family=__import__("statsmodels.api", fromlist=["families"]).families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    return {"params": res.params.to_dict(), "bse": res.bse.to_dict(), "pvalues": res.pvalues.to_dict(), "n": int(res.nobs)}


R_SCRIPT = r"""# Primary GLMM (§12.2) in R — equivalent of analysis/glmm.py
# install.packages(c("lme4", "glmmTMB", "marginaleffects", "jsonlite", "simr"))
library(lme4); library(glmmTMB); library(marginaleffects); library(jsonlite)
claims <- stream_in(file("{claims_path}"))
d <- subset(claims, verdict %in% c("SUPPORTED", "REFUTED"))
d$y <- as.integer(d$verdict == "SUPPORTED")
d$complexity <- scale(d$nodes)
m <- glmer(y ~ condition * type + complexity + hedge + market + model +
             (1 | formula_id) + (1 | rationale_id){base_term},
           data = d, family = binomial, control = glmerControl(optimizer = "bobyqa"))
summary(m)
print(avg_predictions(m, by = "condition"))          # marginal effects in probability units
print(avg_comparisons(m, variables = "condition"))
# glmmTMB alternative (crossed random effects, faster on large data):
m2 <- glmmTMB(y ~ condition * type + complexity + hedge + (1 | formula_id) + (1 | rationale_id),
              data = d, family = binomial)
summary(m2)
# Power by simulation (Green & MacLeod 2016): simr::powerSim(m, test = fixed("condition"), nsim = 200)
"""


def write_r_script(path: str | Path, claims_path: str, base_effect: bool = False) -> Path:
    p = Path(path)
    p.write_text(R_SCRIPT.replace("{claims_path}", claims_path)
                 .replace("{base_term}", " + (1 | base_id)" if base_effect else ""))
    return p
