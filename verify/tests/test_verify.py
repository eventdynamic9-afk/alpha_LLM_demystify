import numpy as np
import pytest

from dsl import parse
from verify import REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, verify_claim
from verify.calibration import calibration_context, run_calibration
from verify.performance import (deflated_sharpe_ratio, hansen_spa, long_short_backtest, pbo_cscv, romano_wolf,
                                white_reality_check)
from verify.smt import prove_monotone
from verify.stats import (bootstrap_mean_ci, clopper_pearson, daily_spearman, newey_west_mean,
                          optimal_block_length, tost_equivalent)
from verify.verdicts import Verdict, aggregate_any_all, invert


@pytest.fixture(scope="module")
def ctx():
    return calibration_context(n_stocks=60, n_days=700, seed=4, fast=True)


# ----------------------------------------------------------------------------- statistics
def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187086, abs=1e-5) and hi == pytest.approx(0.812914, abs=1e-5)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0


def test_block_length_grows_with_persistence():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(2000)
    ar = np.zeros(2000)
    for t in range(1, 2000):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    assert optimal_block_length(ar) > 3 * optimal_block_length(e)


def test_bootstrap_and_newey_west():
    rng = np.random.default_rng(1)
    x = 0.3 + rng.standard_normal(500) * 0.1
    m, lo, hi, _ = bootstrap_mean_ci(x, 0.95, 500, 0)
    assert lo < 0.3 < hi and hi - lo < 0.05
    mu, se, t = newey_west_mean(x)
    assert mu == pytest.approx(x.mean()) and t > 30
    assert tost_equivalent((-0.05, 0.05), 0.1) and not tost_equivalent((-0.05, 0.12), 0.1)


def test_daily_spearman_matches_scipy():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((5, 40)), rng.standard_normal((5, 40))
    a[0, :3] = np.nan
    r = daily_spearman(a, b, min_n=5)
    ok = np.isfinite(a[0]) & np.isfinite(b[0])
    assert r[0] == pytest.approx(spearmanr(a[0, ok], b[0, ok]).statistic)
    assert r[3] == pytest.approx(spearmanr(a[3], b[3]).statistic)


def test_verdict_helpers():
    v = Verdict(SUPPORTED, "x")
    assert invert(v).verdict == REFUTED and invert(Verdict(UNRESOLVED, "x")).verdict == UNRESOLVED
    assert aggregate_any_all([Verdict(REFUTED, "a"), Verdict(SUPPORTED, "b")], "m").verdict == SUPPORTED
    assert aggregate_any_all([Verdict(REFUTED, "a"), Verdict(REFUTED, "b")], "m").verdict == REFUTED
    assert aggregate_any_all([Verdict(REFUTED, "a"), Verdict(UNRESOLVED, "b")], "m").verdict == UNRESOLVED


# ----------------------------------------------------------------------------- performance statistics
def test_pbo_dsr_and_snooping_tests():
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((800, 20)) * 0.01
    assert 0.25 < pbo_cscv(noise, S=8)["pbo"] < 0.75                      # pure noise: no skill
    skilled = noise.copy()
    skilled[:, 0] += 0.004
    assert pbo_cscv(skilled, S=8)["pbo"] < 0.1
    assert white_reality_check(noise, 300)["p_value"] > 0.05
    assert white_reality_check(skilled, 300)["p_value"] < 0.05
    assert hansen_spa(skilled, 300)["p_value"] < 0.05
    rw = romano_wolf(skilled, 300)
    assert rw["p_adjusted"][0] < 0.05 and min(rw["p_adjusted"][1:]) > 0.05
    d1 = deflated_sharpe_ratio(skilled[:, 0], n_trials=1, var_trials=0.0)
    d100 = deflated_sharpe_ratio(skilled[:, 0], n_trials=100, var_trials=0.05 ** 2)
    assert d1["dsr"] > d100["dsr"] and d1["dsr"] > 0.95


def test_backtest_costs_reduce_returns(ctx):
    f = ctx.signal(parse("CSRank(-1*($close/Ref($close, 5)-1))"))
    rows = ctx.rows("test")
    a = long_short_backtest(f, ctx, rows, costs_bp={"buy": 0, "sell": 0})
    b = long_short_backtest(f, ctx, rows, costs_bp={"buy": 5, "sell": 15})
    assert np.nanmean(a["net"]) > np.nanmean(b["net"]) and np.nanmean(a["gross"]) > 0


# ----------------------------------------------------------------------------- claims
REV = "CSRank(-1*($close/Ref($close, 5)-1))"


@pytest.mark.parametrize("claim,expected", [
    ({"predicate": "SIGN", "args": {"input": "ret_5d", "direction": "-"}}, SUPPORTED),
    ({"predicate": "SIGN", "args": {"input": "ret_5d", "direction": "+"}}, REFUTED),
    ({"predicate": "SIGN", "args": {"input": "close", "direction": "-"}}, SUPPORTED),
    ({"predicate": "SIGN", "args": {"input": "volume", "direction": "+"}}, REFUTED),
    ({"predicate": "DEPENDS_ON", "args": {"input": "volume"}}, REFUTED),
    ({"predicate": "DEPENDS_ON", "args": {"input": "analyst revisions"}}, REFUTED),
    ({"predicate": "DEPENDS_ON", "args": {"input": "close"}}, SUPPORTED),
    ({"predicate": "DEPENDS_ON", "args": {"input": "volume"}, "polarity": "deny"}, SUPPORTED),
    ({"predicate": "LOOKBACK", "args": {"window": 5}}, SUPPORTED),
    ({"predicate": "LOOKBACK", "args": {"window": 20}}, REFUTED),
    ({"predicate": "HORIZON", "args": {"bin": "short"}}, SUPPORTED),
    ({"predicate": "HORIZON", "args": {"bin": "long"}}, REFUTED),
    ({"predicate": "XSEC", "args": {"value": True}}, SUPPORTED),
    ({"predicate": "RANGE", "args": {"low": 0, "high": 1}}, SUPPORTED),
    ({"predicate": "INVARIANT", "args": {"transform": "scale", "input": "price"}}, SUPPORTED),
    ({"predicate": "STRUCT", "args": {"pattern": "rank"}}, SUPPORTED),
    ({"predicate": "STRUCT", "args": {"pattern": "corr"}}, REFUTED),
    ({"predicate": "RESEMBLES", "args": {"ref": "short-term reversal", "sign": "+"}}, SUPPORTED),
    ({"predicate": "RESEMBLES", "args": {"ref": "STREV_5d", "sign": "-"}}, REFUTED),
    ({"predicate": "PRED_SIGN", "args": {"sign": "+", "horizon": 1}}, SUPPORTED),
    ({"predicate": "PRED_SIGN", "args": {"sign": "-", "horizon": 1}}, REFUTED),
    ({"predicate": "NOVEL", "args": {}}, REFUTED),
    ({"predicate": "IDENTITY", "args": {"library_id": "alpha101_999"}}, REFUTED),
    ({"predicate": "IDENTITY", "args": {"library_id": "alpha101_056"}}, UNVERIFIABLE),
    ({"predicate": "THEORY", "args": {"mechanism": "overreaction"}}, UNVERIFIABLE),
])
def test_claims_on_planted_reversal(ctx, claim, expected):
    assert verify_claim(claim, parse(REV), ctx).verdict == expected


def test_identity_and_variant(ctx):
    from pools.library import get

    lf = get("alpha158_ROC20")
    assert verify_claim({"predicate": "IDENTITY", "args": {"library_id": "alpha158_ROC20"}}, lf.node, ctx).verdict == SUPPORTED
    other = parse("Ref($close, 21)/$close")
    assert verify_claim({"predicate": "IDENTITY", "args": {"library_id": "alpha158_ROC20"}}, other, ctx).verdict == REFUTED
    rsi = parse(ctx.cb["variant_templates"]["RSI"]["dsl"])
    assert verify_claim({"predicate": "VARIANT_OF", "args": {"template": "RSI"}}, parse(f"CSRank({ctx.cb['variant_templates']['RSI']['dsl']})"), ctx).verdict == SUPPORTED
    assert verify_claim({"predicate": "VARIANT_OF", "args": {"template": "RSI"}}, parse("Log($volume+1)"), ctx).verdict == REFUTED
    assert rsi is not None


def test_metamorphic_refutes_non_scale_free(ctx):
    v = verify_claim({"predicate": "INVARIANT", "args": {"transform": "scale", "input": "price"}},
                     parse("$close-Mean($close, 5)"), ctx)
    assert v.verdict == REFUTED


def test_static_ambiguous_goes_to_nudge(ctx):
    v = verify_claim({"predicate": "SIGN", "args": {"input": "volume", "direction": "+"}},
                     parse("$volume/Mean($volume, 20)"), ctx)
    assert v.verdict == SUPPORTED and v.method in ("nudge", "smt")


def test_smt():
    assert prove_monotone(parse("$close/$open"), "close", 0, "+") == "proved"
    assert prove_monotone(parse("$close/$open"), "open", 0, "+") == "counterexample"
    assert prove_monotone(parse("Mean($close, 5) - $close"), "close", 0, "+") == "counterexample"
    assert prove_monotone(parse("Mean($close, 5) - $close"), "close", 0, "-") == "proved"
    assert prove_monotone(parse("Log($close)"), "close", 0, "+") == "unknown"


def test_drivers(ctx):
    from verify.drivers import true_drivers

    drv, st = true_drivers(parse("CSRank($close/Ref($close, 5)) + 0*$volume"), ctx)
    assert drv == ["close"]
    drv2, st2 = true_drivers(parse("Corr($close, $volume, 10)"), ctx)
    assert set(drv2) == {"close", "volume"}


def test_calibration_subset_passes():
    ctx = calibration_context(n_stocks=60, n_days=700, seed=11, fast=True)
    s = run_calibration(ctx, n_target=300)
    assert s["static"]["accuracy_decidable"] >= 0.99, s["failures"][:5]
    assert s["statistical"]["sign_errors"] == 0
    assert s["passed"], s["failures"][:5]


@pytest.mark.slow
def test_full_calibration_set():
    s = run_calibration(calibration_context())
    assert s["passed"] and s["n_items"] == 300


def test_sequential_evalues():
    from verify.evalues import betting_evalue, sequential_resemblance

    rng = np.random.default_rng(5)
    strong = np.clip(0.6 + 0.1 * rng.standard_normal(400), -1, 1)
    null = np.clip(0.0 + 0.1 * rng.standard_normal(400), -1, 1)
    assert sequential_resemblance(strong)["verdict"] == "SUPPORTED"
    assert sequential_resemblance(null)["verdict"] == "REFUTED"
    # type-I error control under H0: mean = mu0 exactly
    rej = sum(betting_evalue(np.clip(0.3 + 0.2 * rng.standard_normal(300), -1, 1), 0.3)["reject"] for _ in range(200))
    assert rej / 200 <= 0.08


# ----------------------------------------------------------------------------- audit fixes (§10.2-§10.4)
def test_direction_glyphs():
    from verify.dispatcher import _dir

    for g in ("\u2212", "\u2013", "\u2193", "-", "negative", "\u22121"):
        assert _dir(g) == "-", g
    for g in ("\u2191", "+", "up", "increasing"):
        assert _dir(g) == "+", g


def test_glyph_direction_claim(ctx):
    for glyph, expected in (("\u2193", SUPPORTED), ("\u2191", REFUTED), ("\u2212", SUPPORTED)):
        claim = {"predicate": "SIGN", "args": {"input": "ret_5d", "direction": glyph}}
        assert verify_claim(claim, parse(REV), ctx).verdict == expected


@pytest.mark.parametrize("formula,bin_,expected", [
    ("-1*($close/Ref($close, 21)-1)", "short", SUPPORTED),       # STREV_21d: L = 21 (a month) is short-term
    ("-1*($close/Ref($close, 21)-1)", "medium", REFUTED),
    ("-1*($close/Ref($close, 5)-1)", "very_short", SUPPORTED),
    ("$close/Ref($close, 126)-1", "medium", SUPPORTED),
    ("$close/Ref($close, 126)-1", "long", REFUTED),
    ("$close/Ref($close, 127)-1", "long", SUPPORTED),
    ("Mean($close, 22)/$close", "short", SUPPORTED),            # L = 21
    ("Mean($close, 23)/$close", "medium", SUPPORTED),           # L = 22
])
def test_horizon_bins_effective_lookback(ctx, formula, bin_, expected):
    v = verify_claim({"predicate": "HORIZON", "args": {"bin": bin_}}, parse(formula), ctx)
    assert v.verdict == expected and "effective_lookback" in v.evidence


def test_lookback_compares_effective_lookback_only(ctx):
    f = parse("Mean(Ref($close, 60), 5)")                       # L = 64
    got = {n: verify_claim({"predicate": "LOOKBACK", "args": {"window": n}}, f, ctx) for n in (5, 60, 64, 65, 66)}
    assert [got[n].verdict for n in (5, 60, 64, 65, 66)] == [REFUTED, REFUTED, SUPPORTED, SUPPORTED, REFUTED]
    assert got[60].evidence["claimed_is_window_param"] and got[60].evidence["effective_lookback"] == 64
    from verify.static import verify_lookback

    assert verify_lookback(parse("Mean($close, 20)"), 20, "L").verdict == REFUTED
    assert verify_lookback(parse("Mean($close, 20)"), 20, "L+1").verdict == SUPPORTED


def test_sign_through_lagged_field(ctx):
    node = parse("Ref($close, 5)")
    v = verify_claim({"predicate": "SIGN", "args": {"input": "close", "direction": "+"}}, node, ctx)
    assert v.verdict == SUPPORTED and v.method == "monotonicity_static" and v.evidence["read_lag"] == 5
    assert verify_claim({"predicate": "SIGN", "args": {"input": "close", "direction": "-"}}, node, ctx).verdict == REFUTED
    assert verify_claim({"predicate": "DEPENDS_ON", "args": {"input": "close"}}, node, ctx).verdict == SUPPORTED
    # current-value rule kept: TsRank is increasing in today's value
    assert verify_claim({"predicate": "SIGN", "args": {"input": "vwap", "direction": "+"}}, parse("TsRank($vwap, 10)"),
                        ctx).verdict == SUPPORTED
    # a field folded away by canonicalization is absent: 0 -> REFUTED
    v0 = verify_claim({"predicate": "SIGN", "args": {"input": "volume", "direction": "+"}},
                      parse("CSRank($close/Ref($close, 5)) + 0*$volume"), ctx)
    assert v0.verdict == REFUTED and v0.evidence["static"] == "0"


def test_nudge_perturbs_the_lag_the_formula_reads(ctx):
    from verify.inputs import normalize_input
    from verify.nudge import nudge_test

    node, spec = parse("Ref($close, 3)"), normalize_input("close")
    assert nudge_test(node, ctx, spec, "+", lag=3).verdict == SUPPORTED
    v0 = nudge_test(node, ctx, spec, "+", lag=0)
    assert v0.verdict == REFUTED and v0.evidence["n_nonzero"] == 0


def test_invariant_scale_uses_one_constant(ctx):
    inv = {"predicate": "INVARIANT", "args": {"transform": "scale", "input": "price"}}
    v = verify_claim(inv, parse("CSRank($close)"), ctx)
    assert v.verdict == SUPPORTED and v.evidence["relation"] == "price_scale_global"
    rep = v.evidence["reported_relations"]["price_scale_per_stock"]
    assert rep["violation_share"] > 0.5                          # per-stock scaling reorders prices: reported only
    inv_v = {"predicate": "INVARIANT", "args": {"transform": "scale", "input": "volume"}}
    vv = verify_claim(inv_v, parse("CSRank($volume)"), ctx)
    assert vv.verdict == SUPPORTED and vv.evidence["relation"] == "volume_scale_global"


def test_independent_uses_preregistered_aggregation(ctx, monkeypatch):
    rng = np.random.default_rng(9)
    f = ctx.references.signal("STREV_5d")
    noise = np.where(np.isfinite(f), rng.standard_normal(f.shape), np.nan)
    monkeypatch.setitem(ctx.references._cache, "NOISE_REF", noise)
    monkeypatch.setattr(ctx.references, "resolve", lambda ref: [("STREV_5d", 1), ("NOISE_REF", 1)])
    v = verify_claim({"predicate": "INDEPENDENT", "args": {"ref": "two refs"}}, None, ctx, signal=f)
    ops = [o["verdict"] for o in v.evidence["operationalizations"]]
    assert ops == [REFUTED, SUPPORTED] and v.verdict == SUPPORTED     # SUPPORTED if any SUPPORTED (Appendix C)
    ev0 = v.evidence["operationalizations"][1]["evidence"]
    assert {"mean", "se", "t"} <= set(ev0["newey_west"]) and "newey_west" in ev0["replication"]


def test_resembles_reports_newey_west(ctx):
    v = verify_claim({"predicate": "RESEMBLES", "args": {"ref": "STREV_21d", "sign": "+"}}, parse(REV), ctx)
    ev = v.evidence["operationalizations"][0]["evidence"]
    assert ev["newey_west"]["t"] > 3 and "newey_west" in ev["replication"]


def test_turnover_reports_absolute_and_window_percentiles(ctx):
    v = verify_claim({"predicate": "TURNOVER", "args": {"level": "high"}}, parse("CSRank($close/Ref($close, 1)-1)"), ctx)
    assert v.verdict == SUPPORTED
    assert v.evidence["absolute"]["mean_daily_ls_turnover"] > 1.0 and "absolute" in v.evidence["replication"]
    tr, te = ctx.references.turnover_percentiles("train"), ctx.references.turnover_percentiles("test")
    assert tr["window"] == "train" and te["window"] == "test" and v.evidence["replication"]["ref_p75"] == te["p75"]


# ----------------------------------------------------------------------------- EXPOSED factor sets
def _ext_table(ctx, cols: dict, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    T = ctx.panel.T
    base = {c: rng.standard_normal(T) * 0.01 for c in ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "ST_Rev", "RF")}
    base.update(cols)
    return pd.DataFrame(base, index=pd.DatetimeIndex(ctx.panel.dates))


@pytest.fixture(scope="module")
def us_ctx():
    from data.synthetic import synthetic_panel
    from verify.context import VerificationContext

    p = synthetic_panel(60, 700, seed=4, market="US")
    split = str(p.dates[420])
    return VerificationContext(p, windows={"train": (str(p.dates[0]), split), "test": (split, str(p.dates[-1]))}, fast=True)


def test_exposed_us_french_mapping(us_ctx):
    from verify.factors import decile_long_short

    ctx = us_ctx
    f = ctx.signal(parse(REV))
    ls = decile_long_short(ctx.references.signal("STREV_5d"), ctx.fwd(1, "close_t"), ctx.panel.member)
    realized = np.r_[np.nan, ls[:-1]]                            # the French table is indexed by realization date
    rng = np.random.default_rng(1)
    ctx.external_factors = _ext_table(ctx, {"ST_Rev": realized + 0.001 * rng.standard_normal(ctx.panel.T)})
    ctx._factors = None
    try:
        v = verify_claim({"predicate": "EXPOSED", "args": {"ref": "short-term reversal", "sign": "+"}}, None, ctx, signal=f)
        assert v.method == "exposure" and v.verdict == SUPPORTED and v.evidence["factor_source"] == "french_daily"
        assert [o["evidence"]["factor"] for o in v.evidence["operationalizations"]] == ["ST_Rev"]
        vm = verify_claim({"predicate": "EXPOSED", "args": {"ref": "momentum", "sign": "+"}}, None, ctx, signal=f)
        assert vm.method == "exposure" and vm.evidence["operationalizations"][0]["evidence"]["factor"] == "Mom"
        vs = verify_claim({"predicate": "EXPOSED", "args": {"ref": "small size", "sign": "+"}}, None, ctx, signal=f)
        assert vs.evidence["operationalizations"][0]["evidence"]["factor"] == "SMB"
        # no French factor for volatility: explicit signal-level fallback
        vf = verify_claim({"predicate": "EXPOSED", "args": {"ref": "low volatility", "sign": "+"}}, None, ctx, signal=f)
        assert vf.method == "exposure_signal_level" and "fallback" in vf.evidence
    finally:
        ctx.external_factors, ctx._factors = None, None
    v2 = verify_claim({"predicate": "EXPOSED", "args": {"ref": "short-term reversal", "sign": "+"}}, None, ctx, signal=f)
    assert v2.evidence["factor_source"] == "self_built" and "deviation" in v2.evidence


def test_load_factor_table_roundtrip(tmp_path):
    from verify.factors import load_factor_table

    p = tmp_path / "ff.csv"
    p.write_text("date,Mkt-RF,SMB,Mom   ,ST_Rev\n2020-01-02,0.86,-0.97,1.2,0.3\n2020-01-03,-0.67,0.3,-0.5,0.1\n")
    df = load_factor_table(p)
    assert list(df.columns) == ["Mkt-RF", "SMB", "Mom", "ST_Rev"] and df.iloc[0, 0] == pytest.approx(0.0086)
    q = tmp_path / "ch3.csv"
    q.write_text("mnthdt,mktrf,VMG,SMB\n201501,0.012,0.003,-0.004\n201502,0.02,-0.01,0.002\n")
    dm = load_factor_table(q)
    assert dm.index[1].month == 2 and dm.loc[dm.index[0], "mktrf"] == pytest.approx(0.012)


def test_exposed_cn_ch3_monthly(ctx):
    import pandas as pd

    from verify.factors import decile_long_short, monthly_compound

    f = ctx.signal(parse("CSRank(-1*Log(Mean($amount, 21)))"))
    ls = decile_long_short(ctx.references.signal("SIZE_PROXY"), ctx.fwd(1, "close_t"), ctx.panel.member)
    m = monthly_compound(np.r_[np.nan, -ls[:-1]], ctx.panel.dates)
    rng = np.random.default_rng(2)
    tab = pd.DataFrame({"mktrf": rng.standard_normal(len(m)) * 0.03, "SMB": m.to_numpy(),
                        "VMG": rng.standard_normal(len(m)) * 0.03}, index=m.index.to_timestamp())
    ctx.external_factors_monthly, ctx._factors = tab.dropna(), None
    try:
        v = verify_claim({"predicate": "EXPOSED", "args": {"ref": "value", "sign": "+"}}, None, ctx, signal=f)
        ev = v.evidence["operationalizations"][0]["evidence"]
        assert v.method == "exposure" and ev["frequency"] == "monthly" and ev["factor"] == "VMG"
        vs = verify_claim({"predicate": "EXPOSED", "args": {"ref": "small size", "sign": "+"}}, None, ctx, signal=f)
        cross = vs.evidence["ch3_monthly"]["operationalizations"][0]["evidence"]
        assert vs.evidence["factor_source"] == "self_built" and cross["frequency"] == "monthly" and cross["factor"] == "SMB"
        assert vs.evidence["ch3_monthly"]["verdict"] in (SUPPORTED, UNRESOLVED) and cross["beta"] > 0.5
    finally:
        ctx.external_factors_monthly, ctx._factors = None, None


# ----------------------------------------------------------------------------- PERF (§10.4)
def test_hlz_rule_ignores_reported_bar():
    from configs import thresholds
    from verify.performance import hlz_verdict

    cfg = thresholds()["performance"]
    assert "refute_t" not in cfg
    assert hlz_verdict(3.2, cfg) == SUPPORTED
    assert hlz_verdict(1.5, cfg) == UNRESOLVED                  # below 2.0 but its 95% interval reaches 3.0
    assert hlz_verdict(2.5, cfg) == UNRESOLVED
    assert hlz_verdict(1.0, cfg) == REFUTED                      # t + 1.96 <= 3.0
    assert hlz_verdict(-2.0, cfg) == REFUTED and hlz_verdict(float("nan"), cfg) is None


def test_perf_nan_t_is_not_refuted(ctx):
    f = np.full((ctx.panel.T, ctx.panel.N), np.nan)
    v = verify_claim({"predicate": "PERF", "args": {"metric": "IC"}}, None, ctx, signal=f)
    assert v.verdict == UNVERIFIABLE


def test_perf_post_window_separate_and_exploratory():
    from verify.context import VerificationContext

    base = calibration_context(n_stocks=60, n_days=700, seed=4, fast=True)
    p = base.panel
    def roster(k):
        return {"models": [{"id": "m1", "family": "a", "roles": ["narrator"], "training_cutoff": str(p.dates[k])[:10]}]}

    long_ = VerificationContext(p, windows=dict(base.windows), fast=True, roster=roster(480))
    vl = verify_claim({"predicate": "PERF", "args": {"metric": "IC"}}, parse(REV), long_)
    assert vl.evidence["post_exploratory"] is False                 # >= 6 months: confirmatory
    ctx2 = VerificationContext(p, windows={"train": base.windows["train"], "test": base.windows["test"]}, fast=True,
                               roster=roster(600))
    assert ctx2.post_status["status"] == "computed" and "post" in ctx2.windows
    v = verify_claim({"predicate": "PERF", "args": {"metric": "IC"}}, parse(REV), ctx2)
    assert v.evidence["primary_window"] == "test" and set(v.evidence["by_window"]) == {"test", "post"}
    assert v.evidence["post_exploratory"] is True and v.evidence["post_verdict"] in (SUPPORTED, REFUTED, UNRESOLVED)
    for m in ("sharpe", "returns", "stability"):
        vm = verify_claim({"predicate": "PERF", "args": {"metric": m}}, parse(REV), ctx2)
        assert "post" in vm.evidence["by_window"]
    vr = verify_claim({"predicate": "PERF", "args": {"metric": "returns"}}, parse(REV), ctx2)
    assert vr.evidence["cost_sensitivity"]["per_trade_bp"] == 15
    assert vr.evidence["cost_sensitivity"]["ls_mean_net"] < vr.evidence["ls_mean_net"]
    undoc = {"models": [{"id": "m1", "family": "a", "roles": ["narrator"], "training_cutoff": "TO_FILL"}]}
    ctx3 = VerificationContext(p, windows=dict(base.windows), fast=True, roster=undoc)
    assert "post" not in ctx3.windows and ctx3.post_status["status"] == "not_run"


TRIAL_LOG = [{"round": 0, "formula": "CSRank($high-$low)"}, {"round": 1, "formula": "CSRank(Mean($volume, 20))"},
             {"round": 2, "formula": "x(", "valid": False}, {"round": 3, "formula": "CSRank(Mean($close, 60)/Mean($close, 120))"},
             {"round": 4, "formula": REV}]


def _p1_record():
    return {"formula_id": "P1-mined-m-000", "arm": "A", "pool": "P1", "stratum": "mined", "dsl": REV, "trials": 5,
            "seed": 1, "meta": {"trial_log": TRIAL_LOG}}


def test_dsr_uses_recorded_trials_not_claim_args(ctx):
    from verify.performance import record_trials

    tr = record_trials(_p1_record(), ctx)
    assert tr["n_trials"] == 5 and tr["var_trials"] > 0 and len(tr["formulas"]) == 4
    claim = {"predicate": "PERF", "args": {"metric": "sharpe", "n_trials": 1, "var_trials": 0.0}}
    v = verify_claim(claim, parse(REV), ctx, trials=tr)
    v1 = verify_claim(claim, parse(REV), ctx)
    assert v.evidence["dsr"]["n_trials"] == 5 and v.evidence["dsr"]["sr0"] > 0
    assert v.evidence["dsr"]["dsr"] < v1.evidence["dsr"]["dsr"]
    # trials counted but not logged: SR0 unknown -> never SUPPORTED
    vn = verify_claim(claim, parse(REV), ctx, trials={"n_trials": 200, "var_trials": None, "formulas": []})
    assert vn.verdict in (UNRESOLVED, REFUTED) and "dsr_note" in vn.evidence


def test_best_of_claims(ctx):
    from verify.performance import record_trials

    tr = record_trials(_p1_record(), ctx)
    v = verify_claim({"predicate": "PERF", "args": {"metric": "IC", "level": "best"}}, parse(REV), ctx, trials=tr)
    assert v.method == "best_of" and v.verdict == SUPPORTED and v.evidence["n_competitors"] == 3
    assert {"reality_check_p_member_better", "spa_p_member_better", "spa_best_vs_zero_p"} <= set(v.evidence)
    weak = verify_claim({"predicate": "BETTER_THAN", "args": {"ref": "other candidates", "metric": "IC"}},
                        parse("CSRank($high-$low)"), ctx, trials=tr)
    assert weak.method == "best_of" and weak.verdict == REFUTED
    lib = verify_claim({"predicate": "BETTER_THAN", "args": {"ref": "all factors"}}, parse(REV), ctx)
    assert lib.method == "best_of" and lib.evidence["set"] == "library"
    none = verify_claim({"predicate": "PERF", "args": {"metric": "IC", "level": "best"}}, parse(REV), ctx)
    assert none.verdict == UNVERIFIABLE
    # an ordinary comparative claim keeps the paired OOS test
    assert verify_claim({"predicate": "BETTER_THAN", "args": {"ref": "momentum"}}, parse(REV), ctx).method == "paired_oos"


def test_family_report_pbo(ctx):
    from verify.performance import family_report, trial_formulas

    rep = family_report(trial_formulas(_p1_record()), ctx, "train", S=8)
    # Mean(.., 120) covers < 80% of this short training window and is skipped (counted)
    assert rep["status"] == "ok" and rep["n_evaluated"] + rep["n_skipped"] == 4 and rep["n_evaluated"] >= 3
    assert 0.0 <= rep["pbo"]["pbo"] <= 1.0
    assert rep["pbo_cost_sensitivity"]["per_trade_bp"] == 15 and 0 <= rep["spa_p"] <= 1


def test_cli_run_and_perf_family(tmp_path):
    import json

    from data.synthetic import synthetic_panel
    from verify.__main__ import main

    p = synthetic_panel(40, 500, seed=3)
    p.save(tmp_path / "panel.npz")
    recs = [_p1_record(), {"formula_id": "P3a-000", "arm": "A", "pool": "P3a", "dsl": "CSRank($close/$open)",
                           "trials": 300, "seed": 7, "meta": {}},
            {"formula_id": "P3a-001", "arm": "A", "pool": "P3a", "dsl": "CSRank($high/$low)", "trials": 300, "seed": 8,
             "meta": {}}]
    (tmp_path / "formulas.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    claims = [{"claim_id": "c1", "formula_id": "P1-mined-m-000", "predicate": "PERF",
               "args": {"metric": "sharpe", "level": "high", "n_trials": 1}},
              {"claim_id": "c2", "formula_id": "P3a-000", "predicate": "PERF", "args": {"metric": "sharpe"}}]
    (tmp_path / "claims.jsonl").write_text("".join(json.dumps(c) + "\n" for c in claims))
    (tmp_path / "trials.jsonl").write_text("".join(json.dumps({"family": "P3a-seed8", "formula": f}) + "\n"
                                                   for f in ("CSRank($volume)", "CSRank(Std($close, 20))", REV)))
    base = ["--panel", str(tmp_path / "panel.npz"), "--fast", "--formulas", str(tmp_path / "formulas.jsonl")]
    assert main(["run", *base, "--claims", str(tmp_path / "claims.jsonl"), "--out", str(tmp_path / "v.jsonl")]) == 0
    rows = {r["claim_id"]: r for r in map(json.loads, (tmp_path / "v.jsonl").read_text().splitlines())}
    assert rows["c1"]["evidence"]["n_trials"] == 5 and rows["c1"]["evidence"]["dsr"]["sr0"] > 0
    assert rows["c2"]["evidence"]["n_trials"] == 300 and rows["c2"]["verdict"] != SUPPORTED
    assert main(["perf-family", *base, "--out", str(tmp_path / "fam.jsonl"),
                 "--trial-logs", str(tmp_path / "trials.jsonl")]) == 0
    fams = {r["family"]: r for r in map(json.loads, (tmp_path / "fam.jsonl").read_text().splitlines())}
    assert fams["P1-mined-m-000"]["status"] == "ok" and fams["P1-mined-m-000"]["window"] == "valid"
    assert fams["P1-mined-m-000"]["pbo"]["n_combinations"] > 0
    assert fams["P3a-seed7"]["status"] == "not_run" and fams["P3a-seed8"]["status"] == "ok"
