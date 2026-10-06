"""United States panel from the plotly/datasets mirror of the Kaggle "S&P 500 stock data" file (CC0).

Used when the §5.4 primary sources (Yahoo, Tiingo, EODHD, FMP) are unreachable from the run
environment. It is a documented deviation (Appendix E.7):

* coverage 2013-02-08 .. 2018-02-07, the 505 constituents of February 2018 only, so the panel is
  survivorship-biased (names that left the index before 2018 are absent; ``universe_coverage``
  measures the share of point-in-time members that have prices);
* prices are split-adjusted by the vendor but not dividend-adjusted (price returns);
* no post-cutoff window exists for any current narrator (all data predate every training cutoff).

Cleaning is deterministic and every touched row is logged:

1. tickers are normalised to the fja05680 convention (``BRK.B`` -> ``BRK-B``), and six renamed
   tickers are aliased (``TICKER_ALIASES``);
2. no-trade rows (volume <= 3, or open/high/low all missing) become missing;
3. one-day spikes (|close-to-close move| >= 30 % that is reversed the next day to within 10 %)
   become missing;
4. OHLC consistency: an open or close outside [low, high] by at most 1 % widens the range; an open
   outside by more is set missing; a close outside by more drops that day's open/high/low;
5. corporate actions with an overnight gap of 30 % or more were adjudicated one by one
   (``CORPORATE_ACTIONS``); prices before the ex-date are multiplied by the action factor
   (for splits the exact ratio, volume divided by it; for spin-offs and special dividends the
   overnight gap open/previous close, i.e. the whole gap is attributed to the distribution).
   Smaller distributions remain unadjusted and are counted in the QA report.

Usage::

    python -m data rebuild us-plotly
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..manifest import add_entry
from ..membership import from_snapshots, parse_sp500_history
from ..panel import Panel
from .common import PROCESSED, RAW, download

PLOTLY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
PLOTLY_PATH = RAW / "us_plotly" / "all_stocks_5yr.csv"
SPIKE_GAP = 0.30
SPIKE_REVERSAL_TOL = 0.10
OHLC_TOL = 0.01
NO_TRADE_VOLUME = 3

# Overnight gaps of 30 % or more in the raw file, adjudicated against public corporate-action records.
# kind "split": exact ratio applied to prices (volume inverse); "distribution": factor = open / prev close.
CORPORATE_ACTIONS = [
    {"ticker": "DISCA", "date": "2014-08-07", "kind": "split", "ratio": 0.5,
     "note": "one Class C share distributed per Class A share (economically 2-for-1)"},
    {"ticker": "DISCK", "date": "2014-08-07", "kind": "split", "ratio": 0.5,
     "note": "one Class C share distributed per Class C share (economically 2-for-1)"},
    {"ticker": "A", "date": "2014-11-03", "kind": "distribution", "note": "Keysight Technologies spin-off"},
    {"ticker": "BAX", "date": "2015-07-01", "kind": "distribution", "note": "Baxalta spin-off"},
    {"ticker": "NI", "date": "2015-07-02", "kind": "distribution", "note": "Columbia Pipeline Group spin-off"},
    {"ticker": "EBAY", "date": "2015-07-20", "kind": "distribution", "note": "PayPal spin-off"},
    {"ticker": "DHR", "date": "2016-07-05", "kind": "distribution", "note": "Fortive spin-off"},
    {"ticker": "ARNC", "date": "2016-11-01", "kind": "distribution", "note": "Alcoa Corp separation"},
    {"ticker": "YUM", "date": "2016-11-01", "kind": "distribution", "note": "Yum China spin-off"},
    {"ticker": "CAG", "date": "2016-11-10", "kind": "distribution", "note": "Lamb Weston spin-off"},
    {"ticker": "XRX", "date": "2017-01-03", "kind": "distribution", "note": "Conduent spin-off"},
    {"ticker": "HPE", "date": "2017-04-03", "kind": "distribution",
     "note": "Enterprise Services spin-merger into DXC Technology"},
    {"ticker": "BHGE", "date": "2017-07-05", "kind": "distribution",
     "note": "special cash dividend to legacy Baker Hughes holders at the GE combination"},
]
# Remaining >= 30 % gaps in the raw file are genuine price moves (earnings, deals, guidance) and are kept:
# AMD 2016-04-22 and 2017-05-02, BBY 2014-01-16, CHK 2016-02-08 and 2016-04-12, DPS 2018-01-29,
# EVHC 2017-11-01, EXPE 2013-07-26, FL 2017-08-18, ILMN 2016-04-19 and 2016-10-11, INCY 2013-08-21,
# KORS 2015-05-27, MNST 2014-08-15, MOS 2013-07-30, NFLX 2014-10-16, NWL 2017-11-02, PWR 2015-10-16,
# SIG 2017-11-21, STZ 2013-02-14, TRIP 2017-11-07, UA/UAA 2017-01-31, UAA 2017-10-31, VRTX 2013-04-19
# and 2014-06-24, WMB 2016-01-14 and 2016-02-08. CHD 2014-05-19 is a bad open print (rule 4);
# LNT 2016-05-19, MRO 2017-09-14 and NWL 2017-09-14 are one-day spikes (rule 3).

# fja05680/sp500 lists historical members under their later tickers; the 2018 file uses the old ones.
TICKER_ALIASES = {"CBRE": "CBG", "KDP": "DPS", "WELL": "HCN", "JEF": "LUK", "BKNG": "PCLN", "WYND": "WYN"}

WINDOWS = {"train": ("2014-01-01", "2015-12-31"), "valid": ("2016-01-01", "2016-06-30"),
           "test": ("2016-07-01", "2018-02-07")}


def load_raw(path: str | Path = PLOTLY_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        download(PLOTLY_URL, path, "plotly/datasets (Kaggle 'S&P 500 stock data')", license="CC0-1.0",
                 notes="daily OHLCV 2013-02-08..2018-02-07, Feb-2018 S&P 500 constituents")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"Name": "instrument"})
    df["instrument"] = df["instrument"].str.replace(".", "-", regex=False)
    return df.sort_values(["instrument", "date"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply rules 2-5; returns the cleaned frame and a log of every modification."""
    df = df.copy()
    log: list[dict] = []
    px = ["open", "high", "low", "close"]

    def note(rows, rule, detail=""):
        for _, r in df.loc[rows, ["instrument", "date"]].iterrows():
            log.append({"instrument": r["instrument"], "date": str(r["date"].date()), "rule": rule, "detail": detail})

    # rule 2: no-trade rows
    nt = (df["volume"] <= NO_TRADE_VOLUME) | df[["open", "high", "low"]].isna().all(axis=1)
    note(nt, "no_trade")
    df.loc[nt, px + ["volume"]] = np.nan

    # rule 3: one-day spikes reversed next day
    g = df.groupby("instrument")["close"]
    prev, nxt = g.shift(1), g.shift(-1)
    up = df["close"] / prev
    back = nxt / df["close"]
    spike = ((up >= 1 + SPIKE_GAP) | (up <= 1 / (1 + SPIKE_GAP))) & (np.abs(up * back - 1) <= SPIKE_REVERSAL_TOL)
    note(spike, "one_day_spike", "close-to-close move reversed next day")
    df.loc[spike, px + ["volume"]] = np.nan

    # rule 4: OHLC consistency
    hi, lo = df["high"], df["low"]
    for col in ("open", "close"):
        v = df[col]
        above = (v > hi) & (v <= hi * (1 + OHLC_TOL))
        below = (v < lo) & (v >= lo * (1 - OHLC_TOL))
        df.loc[above, "high"] = v[above]
        df.loc[below, "low"] = v[below]
        hi, lo = df["high"], df["low"]
    bad_open = (df["open"] > hi * (1 + OHLC_TOL)) | (df["open"] < lo * (1 - OHLC_TOL))
    note(bad_open, "bad_open", "open outside [low, high] by more than 1%")
    df.loc[bad_open, "open"] = np.nan
    bad_close = (df["close"] > hi * (1 + OHLC_TOL)) | (df["close"] < lo * (1 - OHLC_TOL))
    note(bad_close, "bad_range", "close outside [low, high] by more than 1%: open/high/low dropped")
    df.loc[bad_close, ["open", "high", "low"]] = np.nan

    # rule 5: corporate actions
    for ca in CORPORATE_ACTIONS:
        d = pd.Timestamp(ca["date"])
        sel = df["instrument"] == ca["ticker"]
        if not (sel & (df["date"] == d)).any():
            raise ValueError(f"corporate action {ca} not found in the raw file")
        before = sel & (df["date"] < d)
        if ca["kind"] == "split":
            f = float(ca["ratio"])
            df.loc[before, "volume"] = df.loc[before, "volume"] / f
        else:
            row = df[sel & (df["date"] == d)].iloc[0]
            pc = df.loc[before, "close"].dropna().iloc[-1]
            f = float(row["open"] / pc)
        df.loc[before, px] = df.loc[before, px] * f
        log.append({"instrument": ca["ticker"], "date": ca["date"], "rule": "corporate_action",
                    "detail": f"{ca['kind']} factor {f:.6f}: {ca['note']}"})
    return df, log


def residual_gaps(df: pd.DataFrame, lo: float = 0.10, hi: float = SPIKE_GAP) -> dict:
    """Overnight gaps in [lo, hi) left after cleaning (earnings moves and unadjusted distributions)."""
    pc = df.groupby("instrument")["close"].shift(1)
    gap = (df["open"] / pc - 1).abs()
    big = gap >= hi
    mid = (gap >= lo) & (gap < hi)
    return {"overnight_gap_10_30pct": int(mid.sum()), "overnight_gap_ge_30pct": int(big.sum()),
            "share_of_stock_days_10_30pct": float(mid.mean())}


def universe_coverage(snaps: dict, panel: Panel) -> pd.DataFrame:
    """Share of point-in-time S&P 500 members with a valid close, at each membership snapshot in range."""
    close = panel.get("close")
    col = {s: j for j, s in enumerate(panel.instruments)}
    d0, d1 = panel.dates[0], panel.dates[-1]
    rows = []
    for k in sorted(snaps):
        dk = np.datetime64(pd.Timestamp(k).date(), "D")
        if dk < d0 or dk > d1:
            continue
        i = int(np.searchsorted(panel.dates, dk))
        mem = snaps[k]
        have = [s for s in mem if s in col and np.isfinite(close[i, col[s]])]
        rows.append({"date": str(dk), "members": len(mem), "with_prices": len(have),
                     "coverage": len(have) / len(mem) if mem else float("nan")})
    return pd.DataFrame(rows)


def rebuild_us_plotly(raw_path: str | Path = PLOTLY_PATH, membership_csv: str | Path = RAW / "sp500_history.csv",
                      membership_url: str | None = None) -> dict:
    from ..validation import coverage_table, qa_checklist
    from .us import SP500_HISTORY_URL

    raw = load_raw(raw_path)
    add_entry(raw_path, "plotly/datasets (Kaggle 'S&P 500 stock data')", PLOTLY_URL, "CC0-1.0",
              "daily OHLCV 2013-02-08..2018-02-07, Feb-2018 S&P 500 constituents")
    membership_csv = Path(membership_csv)
    if not membership_csv.exists():
        download(membership_url or SP500_HISTORY_URL, membership_csv, "fja05680/sp500", license="see repository",
                 notes="historical S&P 500 constituents since 1996")
    else:
        add_entry(membership_csv, "fja05680/sp500", membership_url or SP500_HISTORY_URL, "see repository",
                  "historical S&P 500 constituents since 1996")
    snaps = {k: [TICKER_ALIASES.get(t, t) for t in v] for k, v in parse_sp500_history(str(membership_csv)).items()}
    df, log = clean(raw)
    panel = Panel.from_long(df[["date", "instrument", "open", "high", "low", "close", "volume"]], "US")
    member = from_snapshots(panel.dates, panel.instruments, snaps)
    meta = dict(panel.meta)
    meta.update({"source": "plotly/datasets all_stocks_5yr (Kaggle S&P 500, CC0)",
                 "adjusted": "vendor split adjustment + manual corporate actions >= 30% gap; dividends not adjusted",
                 "survivorship": "biased: February-2018 constituents only", "price_adjustment": "split",
                 "windows": {k: list(v) for k, v in WINDOWS.items()},
                 "deviation": "Appendix E.7: §5.4 sources unreachable; window 2013-2018; no post-cutoff window"})
    panel = Panel(panel.dates, panel.instruments, panel.fields, member, "US", meta)
    out = PROCESSED / "us_sp500_plotly.npz"
    panel.save(out)
    cov_dir = PROCESSED.parent / "coverage"
    cov_dir.mkdir(parents=True, exist_ok=True)
    coverage_table(panel).to_csv(cov_dir / "us_sp500_plotly_coverage.csv", index=False)
    ucov = universe_coverage(snaps, panel)
    ucov.to_csv(cov_dir / "us_sp500_plotly_universe_coverage.csv", index=False)
    pd.DataFrame(log).to_csv(cov_dir / "us_sp500_plotly_cleaning_log.csv", index=False)
    qa = qa_checklist(panel, manifest=PROCESSED.parent / "MANIFEST.csv")
    qa.add("residual_unadjusted_gaps", True, residual_gaps(df))
    qa.add("universe_coverage", True, {"first": ucov.iloc[0].to_dict() if len(ucov) else None,
                                       "last": ucov.iloc[-1].to_dict() if len(ucov) else None,
                                       "mean": float(ucov["coverage"].mean()) if len(ucov) else None})
    qa.add("cleaning_log", True, pd.DataFrame(log)["rule"].value_counts().to_dict() if log else {})
    qa.save(cov_dir / "us_sp500_plotly_qa.json")
    return {"panel": str(out), "dates": [str(panel.dates[0]), str(panel.dates[-1])], "T": panel.T, "N": panel.N,
            "member_share": float(member.mean()), "universe_coverage_mean": float(ucov["coverage"].mean()),
            "cleaning": pd.DataFrame(log)["rule"].value_counts().to_dict(), "qa_passed": qa.passed,
            "qa_failed": [k for k, v in qa.checks.items() if not v["ok"]]}
