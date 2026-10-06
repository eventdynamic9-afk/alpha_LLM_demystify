import numpy as np
import pandas as pd
import pytest

from analysis.bootstrap import bootstrap_difference, cluster_bootstrap, macro_precision, micro_precision
from analysis.multiplicity import benjamini_hochberg, benjamini_yekutieli, holm
from analysis.power import design_effect, n_two_proportions, simulate_power, worked_example
from analysis.ppi import ppi_mean
from annotation.agreement import cohen_kappa, fleiss_kappa, gwet_ac1, krippendorff_alpha_nominal


def test_worked_example_matches_section_12_4():
    w = worked_example()
    assert round(w["n_claims_alpha05"]) == 162
    assert round(w["z_bonferroni"], 2) == 2.87
    assert round(w["n_claims_bonferroni"]) == 285
    assert w["de_claims"] == pytest.approx(2.0) and round(w["claims_after_de"]) == 569
    assert round(w["rationales"]) == 95
    assert w["de_samples"] == pytest.approx(1.6) and round(w["rationales_after_de"]) == 152
    assert w["formulas_per_arm_ceil"] == 51 and w["planned"] == 60 and w["margin_ok"]
    assert design_effect(6, 0.2) == pytest.approx(2.0)
    assert n_two_proportions(0.5, 0.5 + 1e-9) > 1e10


def test_power_simulation_monotone():
    lo = simulate_power(20, n_sims=60, seed=1, icc_formula=0.1, icc_rationale=0.1)["power"]
    hi = simulate_power(120, n_sims=60, seed=1, icc_formula=0.1, icc_rationale=0.1)["power"]
    assert hi > lo


def test_multiplicity():
    p = [0.01, 0.04, 0.03, 0.005]
    h = holm(p, 0.05)
    assert h["p_adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert h["reject"] == [True, False, False, True]
    bh = benjamini_hochberg([0.01, 0.02, 0.03, 0.5], 0.10)
    assert bh["p_adjusted"] == pytest.approx([0.04, 0.04, 0.04, 0.5])
    by = benjamini_yekutieli([0.01, 0.02, 0.03, 0.5], 0.10)
    assert by["p_adjusted"][0] == pytest.approx(0.04 * (1 + 1 / 2 + 1 / 3 + 1 / 4))


def test_ppi_unbiased_with_biased_parser():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.4, 0.9, 5000)                    # true rationale-level precision
    auto = np.clip(truth - 0.1 + rng.normal(0, 0.05, 5000), 0, 1)  # parser biased by -0.1
    gold_idx = rng.choice(5000, 300, replace=False)
    r = ppi_mean(auto, auto[gold_idx], truth[gold_idx])
    assert abs(r["naive_auto"] - truth.mean()) > 0.08
    assert r["ci"][0] < truth.mean() < r["ci"][1]
    r2 = ppi_mean(auto, auto[gold_idx], truth[gold_idx], lam=None)
    assert r2["ci"][0] < truth.mean() < r2["ci"][1] and 0 <= r2["lambda"] <= 1
    assert r2["se"] <= r["se"] + 1e-12


def test_agreement_statistics():
    a = ["x", "x", "y", "y", "z", "x"]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(["x", "y"] * 10, ["y", "x"] * 10) < 0
    # Fleiss textbook example (Wikipedia, 10 subjects x 5 categories, 14 raters): kappa = 0.210
    counts = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                       [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert fleiss_kappa(counts) == pytest.approx(0.210, abs=1e-3)
    # Krippendorff (2011) "Computing Krippendorff's alpha-reliability": 4 observers x 12 units with
    # missing values, nominal alpha = 0.743
    data = [[1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
            [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]]
    assert krippendorff_alpha_nominal(data) == pytest.approx(0.743, abs=1e-3)
    assert gwet_ac1([a, a]) == pytest.approx(1.0)


def _claims(n_f=30, p_a=0.8, p_b=0.6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g, p in (("a", p_a), ("b", p_b)):
        for f in range(n_f):
            for r in range(3):
                for c in range(5):
                    rows.append({"formula_id": f"{g}{f}", "base_id": f"base{f}", "rationale_id": f"{g}{f}-{r}", "grp": g,
                                 "decidable": True, "supported": float(rng.random() < p)})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_and_differences():
    df = _claims()
    r = cluster_bootstrap(df[df.grp == "a"], micro_precision, n_boot=200)
    assert r["ci"][0] < 0.8 < r["ci"][1]
    assert abs(macro_precision(df[df.grp == "a"]) - micro_precision(df[df.grp == "a"])) < 1e-9
    d = bootstrap_difference(df, "grp", "a", "b", micro_precision, n_boot=200)
    assert d["estimate"] > 0.1 and d["p_greater"] < 0.05
    dp = bootstrap_difference(df, "grp", "a", "b", micro_precision, cluster="base_id", n_boot=200, paired=True)
    assert dp["p_greater"] < 0.05


def test_glmm_runs_and_recovers_direction():
    from analysis.glmm import fit_glmm

    df = _claims(n_f=25)
    df["condition"] = df["grp"]
    df["type"] = "C1"
    df["hedge"] = "absolute"
    df["nodes"] = 8
    df["model"] = "m"
    out = fit_glmm(df, interaction=False)
    me = out["marginal"].set_index("condition")
    assert me.loc["a", "pp"] > me.loc["b", "pp"]
    assert out["n_formulas"] == 50
