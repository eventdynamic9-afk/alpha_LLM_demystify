"""Power analysis (§12.4).

Worked example (replace assumptions with pilot estimates):
  CP 0.70 vs 0.55, power 0.80, two-sided
  1. independent claims per arm, alpha = 0.05 ............................ ~162
  2. with Bonferroni-equivalent alpha = 0.05/12 (z ~ 2.87) ................ ~285
  3. design effect claims-in-rationales (m ~ 6, ICC ~ 0.2): DE = 2.0 ...... ~569 claims ~ 95 rationales
  4. design effect k = 3 samples within formula (ICC ~ 0.3): DE = 1.6 ...... ~152 rationales ~ 51 formulas
  5. planned: 60 formulas per arm
The final N is chosen by simulating the §12.2 GLMM with pilot-estimated m, ICCs and decidability.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def n_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True) -> float:
    """Per-arm n for comparing two independent proportions (pooled-variance normal approximation)."""
    za = sps.norm.ppf(1 - alpha / 2) if two_sided else sps.norm.ppf(1 - alpha)
    zb = sps.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (p1 - p2) ** 2


def design_effect(m: float, icc: float) -> float:
    """Kish (1965): DE = 1 + (m - 1) * ICC."""
    return 1 + (m - 1) * icc


def worked_example(p1: float = 0.70, p2: float = 0.55, power: float = 0.80, n_tests: int = 12,
                   m_claims: float = 6, icc_claims: float = 0.2, k_samples: float = 3, icc_samples: float = 0.3,
                   planned: int = 60) -> dict:
    n1 = n_two_proportions(p1, p2, 0.05, power)
    alpha_b = 0.05 / n_tests
    n2 = n_two_proportions(p1, p2, alpha_b, power)
    de1 = design_effect(m_claims, icc_claims)
    claims = n2 * de1
    rationales = claims / m_claims
    de2 = design_effect(k_samples, icc_samples)
    rationales2 = rationales * de2
    formulas = rationales2 / k_samples
    return {"n_claims_alpha05": n1, "z_bonferroni": float(sps.norm.ppf(1 - alpha_b / 2)), "n_claims_bonferroni": n2,
            "de_claims": de1, "claims_after_de": claims, "rationales": rationales, "de_samples": de2,
            "rationales_after_de": rationales2, "formulas_per_arm": formulas, "formulas_per_arm_ceil": math.ceil(formulas),
            "planned": planned, "margin_ok": planned >= math.ceil(formulas)}


# ----------------------------------------------------------------------------- simulation
def simulate_dataset(n_formulas: int, k: int, m: float, p: float, sd_formula: float, sd_rationale: float,
                     rng: np.random.Generator):
    """Claims nested in rationales nested in formulas, random intercepts on the logit scale."""
    logit = math.log(p / (1 - p))
    fid, rid, y = [], [], []
    r_counter = 0
    for f in range(n_formulas):
        uf = rng.normal(0, sd_formula)
        for _ in range(k):
            ur = rng.normal(0, sd_rationale)
            n_c = max(1, rng.poisson(m))
            pr = 1 / (1 + math.exp(-(logit + uf + ur)))
            y.extend(rng.random(n_c) < pr)
            fid.extend([f] * n_c)
            rid.extend([r_counter] * n_c)
            r_counter += 1
    return np.array(fid), np.array(rid), np.array(y, dtype=float)


def icc_to_sd(icc: float) -> float:
    """Latent-scale random-intercept SD giving the ICC on the logistic latent scale (pi^2/3 residual)."""
    if icc <= 0:
        return 0.0
    return math.sqrt(icc / (1 - icc) * math.pi ** 2 / 3)


def simulate_power(n_formulas: int, p1: float = 0.70, p2: float = 0.55, k: int = 3, m: float = 6,
                   icc_formula: float = 0.3, icc_rationale: float = 0.2, alpha: float = 0.05 / 6,
                   n_sims: int = 200, seed: int = 0, method: str = "cluster_t",
                   sd_formula: float | None = None, sd_rationale: float | None = None) -> dict:
    """Power of the arm contrast under the nested design.

    Latent-scale ICCs are not the observed-scale ICCs of the Kish arithmetic above; for the pre-registered
    N pass the variance components of the GLMM fitted on pilot data (``sd_formula``, ``sd_rationale``).
    method="cluster_t": Welch t-test on formula-level precisions (valid, conservative, fast);
    method="glmm": statsmodels Bayesian mixed GLM (slow; use for the final pre-registered N).
    """
    rng = np.random.default_rng(seed)
    sdf = icc_to_sd(icc_formula) if sd_formula is None else sd_formula
    sdr = icc_to_sd(icc_rationale) if sd_rationale is None else sd_rationale
    hits = 0
    for _ in range(n_sims):
        fa, ra, ya = simulate_dataset(n_formulas, k, m, p1, sdf, sdr, rng)
        fb, rb, yb = simulate_dataset(n_formulas, k, m, p2, sdf, sdr, rng)
        if method == "glmm":
            pval = _glmm_p(fa, ra, ya, fb, rb, yb)
        else:
            ma = np.array([ya[fa == f].mean() for f in np.unique(fa)])
            mb = np.array([yb[fb == f].mean() for f in np.unique(fb)])
            pval = sps.ttest_ind(ma, mb, equal_var=False).pvalue
        hits += pval < alpha
    return {"n_formulas": n_formulas, "power": hits / n_sims, "n_sims": n_sims, "alpha": alpha, "method": method}


def _glmm_p(fa, ra, ya, fb, rb, yb) -> float:  # pragma: no cover - slow path
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    df = pd.DataFrame({"y": np.r_[ya, yb], "arm": np.r_[np.zeros(len(ya)), np.ones(len(yb))],
                       "f": np.r_[fa, fb + fa.max() + 1].astype(str), "r": np.r_[ra, rb + ra.max() + 1].astype(str)})
    mod = BinomialBayesMixedGLM.from_formula("y ~ arm", {"f": "0 + C(f)", "r": "0 + C(r)"}, df)
    res = mod.fit_vb()
    i = mod.exog_names.index("arm")
    z = res.fe_mean[i] / res.fe_sd[i]
    return float(2 * sps.norm.sf(abs(z)))


def required_formulas(target_power: float = 0.80, start: int = 20, stop: int = 120, step: int = 5, **kw) -> dict:
    curve = []
    for n in range(start, stop + 1, step):
        r = simulate_power(n, **kw)
        curve.append(r)
        if r["power"] >= target_power:
            return {"n_formulas": n, "curve": curve}
    return {"n_formulas": None, "curve": curve}
