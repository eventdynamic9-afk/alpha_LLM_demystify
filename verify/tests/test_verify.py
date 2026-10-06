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
