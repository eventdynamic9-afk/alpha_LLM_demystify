import numpy as np
import pandas as pd

from data.labels import daily_returns, forward_returns, limit_locked
from data.membership import from_intervals, from_snapshots
from data.panel import Panel
from data.qlib_bin import load_qlib_panel, write_qlib_bins
from data.rebuild.references import parse_french_csv
from data.synthetic import synthetic_panel
from data.validation import coverage_table, cross_source_mismatch, qa_checklist, shumway_delisting_bound


def test_synthetic_is_deterministic_and_has_defects():
    a = synthetic_panel(30, 300, seed=3)
    b = synthetic_panel(30, 300, seed=3)
    np.testing.assert_array_equal(a.get("close"), b.get("close"))
    assert np.isnan(a.get("close")).any()           # listings/delistings/suspensions
    assert (a.get("volume") == 0).any()
    assert not a.member.all()
    assert np.all(np.nan_to_num(a.get("high") - a.get("low"), nan=0) >= 0)


def test_forward_returns_convention():
    p = synthetic_panel(10, 50, seed=1, defects=False)
    f1 = forward_returns(p, 1, "close_t")
    c = p.get("close")
    np.testing.assert_allclose(f1[:-1], c[1:] / c[:-1] - 1)
    assert np.isnan(f1[-1]).all()
    f5 = forward_returns(p, 5, "open_t+1")
    np.testing.assert_allclose(f5[10], c[15] / p.get("open")[11] - 1)
    assert np.isnan(f5[-5:]).all()


def test_limit_locked_cn():
    p = synthetic_panel(10, 60, seed=2, defects=False)
    c = p.get("close").copy()
    c[30, 0] = c[29, 0] * 1.10
    p = p.with_fields(close=c)
    p.meta["price_limit"] = 0.10
    up, down = limit_locked(p)
    assert up[30, 0] and not down[30, 0]


def test_qlib_bin_roundtrip(tmp_path):
    p = synthetic_panel(8, 80, seed=4)
    write_qlib_bins(p, tmp_path, "all")
    q = load_qlib_panel(tmp_path, "all", market="CN")
    ever = np.flatnonzero(p.member.any(axis=0))           # Qlib instrument files list members only
    p = p.take_instruments(ever)
    assert list(q.instruments) == list(p.instruments)
    np.testing.assert_allclose(q.get("close"), p.get("close").astype(np.float32), rtol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(q.member, p.member)


def test_membership_helpers():
    dates = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")
    insts = np.array(["A", "B"])
    iv = pd.DataFrame({"symbol": ["A", "A", "B"], "start": ["2020-01-01", "2020-01-08", "2020-01-05"],
                       "end": ["2020-01-03", "2020-01-10", "2020-01-06"]})
    m = from_intervals(dates, insts, iv)
    assert m[:3, 0].all() and not m[3:7, 0].any() and m[7:, 0].all() and m[4:6, 1].all()
    s = from_snapshots(dates, insts, {"2020-01-01": ["A"], "2020-01-06": ["B"]})
    assert s[:5, 0].all() and not s[5:, 0].any() and s[5:, 1].all()


def test_cross_source_and_coverage():
    a = synthetic_panel(12, 120, seed=5)
    b = a.copy()
    c = b.get("close").copy()
    c[60, 3] *= 1.01                                   # 100 bp disagreement on one day
    b = b.with_fields(close=c)
    flag, rates = cross_source_mismatch(a, b)
    assert flag.sum() == 2                              # day 60 and day 61 returns both differ
    assert rates["flagged"].sum() == 2
    cov = coverage_table(a)
    assert ((cov["coverage"] > 0.8) & (cov["coverage"] <= 1)).all()
    assert np.isfinite(shumway_delisting_bound(a)).any()


def test_qa_checklist_runs():
    rep = qa_checklist(synthetic_panel(10, 100, seed=6))
    assert rep.passed and "coverage" in rep.checks


def test_panel_io(tmp_path):
    p = synthetic_panel(5, 40, seed=7)
    p.save(tmp_path / "p.npz")
    q = Panel.load(tmp_path / "p.npz")
    np.testing.assert_array_equal(q.get("close"), p.get("close"))
    assert q.meta["seed"] == 7 and q.market == p.market
    assert daily_returns(q).shape == (40, 5)


def test_french_parser():
    txt = ("This file was created ...\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n19630701,   -0.67,    0.00,   -0.32,"
           "   -0.01,    0.15,    0.012\n19630702,    0.79,   -0.27,    0.27,   -0.07,   -0.19,    0.012\n\n"
           " Annual Factors\n")
    df = parse_french_csv(txt)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert abs(df.iloc[0, 0] + 0.0067) < 1e-12 and len(df) == 2
