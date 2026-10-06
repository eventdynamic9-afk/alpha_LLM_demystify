import json
import random
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from data.panel import Panel
from data.synthetic import synthetic_panel
from dsl import parse, to_alpha101, to_qlib
from dsl.random_trees import random_tree
from executors import E1Executor, E2Executor
from executors.causality import LeakyExecutor, truncation_test
from executors.compare import agreement

FIX = Path(__file__).resolve().parents[1] / "fixtures"
E1, E2 = E1Executor(), E2Executor()


def tiny_panel() -> Panel:
    # 6 days x 3 stocks with hand-checkable values
    close = np.array([[10, 20, 30], [11, 19, 30], [12, 18, 30], [11, 21, 31], [13, 22, 29], [14, 20, 28]], float)
    vol = np.array([[100, 200, 300], [110, 190, 300], [120, 180, 300], [130, 170, 310], [90, 160, 290],
                    [100, 150, 280]], float)
    f = {"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close, "vwap": close,
         "volume": vol, "amount": close * vol}
    dates = np.arange("2020-01-01", "2020-01-07", dtype="datetime64[D]")
    return Panel(dates, ["A", "B", "C"], f, np.ones((6, 3), bool), "CN")


@pytest.mark.parametrize("ex", [E1, E2], ids=["E1", "E2"])
def test_hand_computed_values(ex):
    p = tiny_panel()
    m = ex.evaluate(parse("Mean($close, 3)"), p)
    assert np.isnan(m[:2]).all() and m[2, 0] == pytest.approx(11.0) and m[5, 1] == pytest.approx(21.0)
    r = ex.evaluate(parse("TsRank($close, 3)"), p)
    assert r[2, 0] == pytest.approx(1.0) and r[2, 1] == pytest.approx(1 / 3)
    assert r[2, 2] == pytest.approx((0 + (3 + 1) / 2) / 3)          # three-way tie -> average rank
    cs = ex.evaluate(parse("CSRank($close)"), p)
    np.testing.assert_allclose(cs[0], [1 / 3, 2 / 3, 1.0])
    idx = ex.evaluate(parse("IdxMax($close, 3)"), p)
    assert idx[2, 0] == 3 and idx[2, 1] == 1 and idx[2, 2] == 1      # first occurrence on ties
    w = ex.evaluate(parse("WMA($close, 3)"), p)
    assert w[2, 0] == pytest.approx((10 * 1 + 11 * 2 + 12 * 3) / 6)
    s = ex.evaluate(parse("Slope($close, 3)"), p)
    assert s[2, 0] == pytest.approx(1.0) and s[2, 1] == pytest.approx(-1.0)
    rs = ex.evaluate(parse("Resi($close, 3)"), p)
    assert rs[2, 0] == pytest.approx(0.0, abs=1e-9)
    c = ex.evaluate(parse("Corr($close, $volume, 3)"), p)
    assert c[2, 0] == pytest.approx(1.0) and c[2, 1] == pytest.approx(1.0)
    assert np.isnan(c[2, 2])                                          # constant window -> NaN
    d = ex.evaluate(parse("$close/($close-$close)"), p)
    assert np.isnan(d).all()                                          # division by zero -> NaN
    sc = ex.evaluate(parse("CSScale($close)"), p)
    np.testing.assert_allclose(np.abs(sc).sum(axis=1), 1.0)


def test_nan_window_semantics():
    p = tiny_panel().copy()
    p.fields["close"][3, 0] = np.nan
    for ex in (E1, E2):
        m = ex.evaluate(parse("Mean($close, 2)"), p)
        assert np.isnan(m[3, 0]) and np.isnan(m[4, 0]) and np.isfinite(m[5, 0])


def test_membership_mask_and_cross_section_universe():
    p = tiny_panel().copy()
    p.member[1, 2] = False
    for ex in (E1, E2):
        cs = ex.evaluate(parse("CSRank($close)"), p)
        assert np.isnan(cs[1, 2])
        np.testing.assert_allclose(cs[1, :2], [0.5, 1.0])


@pytest.mark.parametrize("panel_file,expected_file", [("synthetic_gbm.npz", "expected_outputs.json"),
                                                      ("real_slice.npz", "expected_outputs_real.json")])
def test_fixture_expected_outputs(panel_file, expected_file):
    p = Panel.load(FIX / panel_file)
    exp = json.loads((FIX / expected_file).read_text())
    for src, e in exp.items():
        for ex in (E1, E2):
            v = ex.evaluate(parse(src), p)
            assert int(np.isfinite(v).sum()) == e["n_finite"], src
            for t, i, val in e["cells"]:
                assert v[t, i] == pytest.approx(val, rel=1e-9, abs=1e-9), (src, t, i)


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(40, 260, seed=11)


def _minimal(name: str):
    """A minimal valid formula for each operator, on inputs with NaNs, ties and constant windows."""
    from dsl.ast import C, F, Node
    from dsl.operators import OPS

    s = OPS[name]
    x, y = F("close"), F("volume")
    params = []
    for kind in s.param_kinds:
        params.append({"lag": 2, "lag1": 2, "window": max(5, s.min_window), "level": 0.8}.get(kind, 2.0))
    if name == "CSScale":
        params = [1.0]
    if s.n_children == 1:
        kids = (Node("Gt", (x, F("open"))),) if name == "Not" else (x,)
    elif s.n_children == 2:
        kids = (Node("Gt", (x, F("open"))), Node("Lt", (y, F("amount")))) if name in ("And", "Or") else (x, y)
    else:
        kids = (Node("Gt", (x, F("open"))), x, y)
    return Node(name, kids, tuple(params))


@pytest.fixture(scope="module")
def edge_panel():
    """Panel with suspensions (NaN runs), exact ties across stocks and constant windows."""
    p = synthetic_panel(30, 160, seed=5)
    f = {k: v.copy() for k, v in p.fields.items()}
    for k in f:
        f[k][40:47, 3] = np.nan                       # suspension
        f[k][:, 7] = f[k][:, 6]                       # exact ties across stocks
        f[k][60:80, 9] = f[k][60, 9]                  # constant window
    return p.with_fields(**f)


@pytest.mark.parametrize("name", sorted(__import__("dsl.operators", fromlist=["OPS"]).OPS))
def test_e1_e2_agree_per_operator(name, edge_panel):
    t = _minimal(name)
    ag = agreement(E1.evaluate(t, edge_panel), E2.evaluate(t, edge_panel), t)
    assert ag.ok, (to_qlib(t), ag)


def test_e1_e2_agree_on_random_formulas(panel):
    from dsl.random_trees import AGREEMENT_WEIGHTS

    for i in range(60):
        t = random_tree(random.Random(1000 + i), max_depth=random.Random(i).randint(2, 5), weights=AGREEMENT_WEIGHTS)
        ag = agreement(E1.evaluate(t, panel), E2.evaluate(t, panel), t)
        assert ag.ok, (to_qlib(t), ag)


def test_rank_rule_only_where_float_order_matters(panel):
    t = parse("Mean($close, 5)")
    x = E2.evaluate(t, panel)
    assert not agreement(x, 3 * x + 7, t).ok                       # no order-dependent operator: abs rule only
    r = parse("CSRank(Mean($close, 5))")
    y = E2.evaluate(r, panel)
    assert agreement(y, y + 1e-6 * np.isfinite(y), r).ok          # order-dependent: rank rule allowed


@pytest.mark.slow
def test_e1_e2_agree_on_500_random_formulas(panel):
    from dsl.random_trees import AGREEMENT_WEIGHTS

    fails = []
    used = set()
    for i in range(500):
        t = random_tree(random.Random(i), max_depth=random.Random(i + 7).randint(2, 5), weights=AGREEMENT_WEIGHTS)
        used |= {m.op for m in __import__("dsl").walk(t) if not m.is_leaf}
        ag = agreement(E1.evaluate(t, panel), E2.evaluate(t, panel), t)
        if not ag.ok:
            fails.append((to_qlib(t), ag))
    assert not fails, fails[:5]
    assert set(AGREEMENT_WEIGHTS) <= used | {"SignedPower"}


def test_alpha101_serialization_is_numerically_equivalent(panel):
    for i in range(40):
        t = random_tree(random.Random(5000 + i), max_depth=4)
        a = E2.evaluate(t, panel)
        b = E2.evaluate(parse(to_alpha101(t), "alpha101"), panel)
        assert agreement(a, b, t).ok, to_qlib(t)


def test_truncation_test(panel):
    for src in ["CSRank(Corr($close, $volume, 10)) - TsRank($close, 5)", "Resi($close, 20)/Std($close, 20)",
                "Slope(CSZScore($volume), 10)"]:
        for ex in (E1, E2):
            rep = truncation_test(parse(src), panel, ex)
            assert rep.ok and len(rep.cuts) >= 5
    assert not truncation_test(parse("$close"), panel, LeakyExecutor(E2)).ok   # look-ahead is caught


# ------------------------------------------------------------------ property-based operator laws
def _rand_panel(seed, T=60, N=12):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0)) * rng.uniform(5, 50, N)
    vol = rng.lognormal(10, 0.5, (T, N))
    f = {"open": close * rng.uniform(0.98, 1.02, (T, N)), "high": close * 1.03, "low": close * 0.97,
         "close": close, "vwap": close * rng.uniform(0.99, 1.01, (T, N)), "volume": vol, "amount": close * vol}
    return Panel(np.datetime64("2020-01-01") + np.arange(T),
                 [f"I{i}" for i in range(N)], f, np.ones((T, N), bool), "CN")


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10**6), st.sampled_from(["CSRank", "TsRank"]))
def test_rank_invariant_to_monotone_transform(seed, opname):
    p = _rand_panel(seed)
    wrap = (lambda x: f"CSRank({x})") if opname == "CSRank" else (lambda x: f"TsRank({x}, 10)")
    base = E2.evaluate(parse(wrap("$close")), p)
    for tr in ("Log($close)", "$close^3", "2*$close+1"):
        np.testing.assert_allclose(E2.evaluate(parse(wrap(tr)), p), base, equal_nan=True)
        np.testing.assert_allclose(E1.evaluate(parse(wrap(tr)), p), base, equal_nan=True)


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10**6), st.integers(1, 15), st.floats(-3, 3), st.floats(-3, 3))
def test_mean_linearity(seed, n, a, b):
    p = _rand_panel(seed)
    lhs = E2.evaluate(parse(f"Mean({a!r}*$close+{b!r}*$volume, {n})".replace("--", "+")), p)
    rhs = E2.evaluate(parse(f"{a!r}*Mean($close, {n})+{b!r}*Mean($volume, {n})".replace("--", "+")), p)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-6, equal_nan=True)


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10**6), st.integers(0, 10), st.integers(0, 10))
def test_ref_composition(seed, a, b):
    p = _rand_panel(seed)
    for ex in (E1, E2):
        np.testing.assert_array_equal(ex.evaluate(parse(f"Ref(Ref($close, {a}), {b})"), p),
                                      ex.evaluate(parse(f"Ref($close, {a + b})"), p))


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 10**6), st.integers(2, 20))
def test_window_edges_and_identities(seed, n):
    p = _rand_panel(seed)
    for ex in (E1, E2):
        v = ex.evaluate(parse(f"Std($close, {n})"), p)
        assert np.isnan(v[: n - 1]).all() and np.isfinite(v[n - 1:]).all()
        np.testing.assert_allclose(ex.evaluate(parse("Sum($close, 1)"), p), p.get("close"))
        c = ex.evaluate(parse(f"Corr($close, 2*$close+3, {max(n, 3)})"), p)
        np.testing.assert_allclose(c[max(n, 3) - 1:], 1.0, rtol=1e-9)
        np.testing.assert_allclose(np.nansum(np.abs(ex.evaluate(parse("CSScale($close-$open)"), p)), axis=1), 1.0)
