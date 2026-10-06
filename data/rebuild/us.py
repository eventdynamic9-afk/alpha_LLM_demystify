"""United States free stack (§5.4).

Prices: yfinance for every ticker that was ever an S&P 500 member since 2014 (survivorship-biased:
delisted names are largely missing — measure it); historical membership: fja05680/sp500; removed /
delisted names: Tiingo, EODHD (``delisted=1``; reused tickers carry an ``_old`` suffix) and FMP free
tiers when API keys are present; VWAP proxy (H+L+C)/3, disclosed.

Usage::

    python -m data rebuild us --start 2015-01-01 --end 2024-12-31
"""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..membership import from_snapshots, parse_sp500_history
from ..panel import Panel
from .common import PROCESSED, RAW, download, env_key

SP500_HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                     "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2025).csv")


def fetch_sp500_membership(url: str = SP500_HISTORY_URL) -> dict:
    """The file name in fja05680/sp500 carries a date stamp that changes; pass the current URL."""
    path = download(url, RAW / "sp500_history.csv", "fja05680/sp500", license="see repository",
                    notes="historical S&P 500 constituents since 1996")
    return parse_sp500_history(str(path))


def yfinance_panel(tickers: list[str], start: str, end: str) -> Panel:
    import yfinance as yf

    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, actions=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date", "index": "date"})
        df["instrument"] = t
        frames.append(df[["date", "instrument", "open", "high", "low", "close", "volume"]])
    long = pd.concat(frames, ignore_index=True)
    p = Panel.from_long(long, "US")              # vwap proxy (H+L+C)/3 is added by Panel
    p.meta.update({"source": "yfinance", "adjusted": True, "survivorship": "biased: delisted names missing"})
    return p


def tiingo_daily(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    key = env_key("TIINGO_API_KEY")
    if not key:
        return None
    url = (f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start}&endDate={end}"
           f"&token={key}")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read())
    if not data:
        return None
    df = pd.DataFrame(data)
    return pd.DataFrame({"date": pd.to_datetime(df["date"]).dt.tz_localize(None), "instrument": ticker,
                         "open": df["adjOpen"], "high": df["adjHigh"], "low": df["adjLow"],
                         "close": df["adjClose"], "volume": df["adjVolume"]})


def eodhd_delisted_list(exchange: str = "US") -> pd.DataFrame | None:
    key = env_key("EODHD_API_KEY")
    if not key:
        return None
    url = f"https://eodhd.com/api/exchange-symbol-list/{exchange}?api_token={key}&delisted=1&fmt=json"
    with urllib.request.urlopen(url, timeout=60) as r:
        return pd.DataFrame(json.loads(r.read()))


def fmp_delisted(page: int = 0) -> pd.DataFrame | None:
    key = env_key("FMP_API_KEY")
    if not key:
        return None
    url = f"https://financialmodelingprep.com/api/v3/delisted-companies?page={page}&apikey={key}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return pd.DataFrame(json.loads(r.read()))


def fill_delisted(panel: Panel, missing: list[str], start: str, end: str) -> tuple[Panel, dict]:
    """Try free-tier vendors for tickers absent from Yahoo; returns the merged panel and a log."""
    frames, log = [], {}
    for t in missing:
        df = tiingo_daily(t, start, end)
        log[t] = "tiingo" if df is not None else "missing"
        if df is not None:
            frames.append(df)
    if not frames:
        return panel, log
    extra = Panel.from_long(pd.concat(frames, ignore_index=True), "US")
    long = []
    for p in (panel, extra):
        for j, inst in enumerate(p.instruments):
            long.append(pd.DataFrame({"date": pd.to_datetime(p.dates), "instrument": inst,
                                      **{f: p.get(f)[:, j] for f in ("open", "high", "low", "close", "volume")}}))
    merged = pd.concat(long, ignore_index=True).dropna(subset=["close"])
    merged = merged.drop_duplicates(["date", "instrument"], keep="first")
    return Panel.from_long(merged, "US"), log


def rebuild_us(start: str = "2015-01-01", end: str = "2024-12-31", membership_url: str = SP500_HISTORY_URL) -> dict:
    from ..validation import coverage_table, qa_checklist

    snaps = fetch_sp500_membership(membership_url)
    lo = pd.Timestamp(start) - pd.Timedelta(days=400)
    tickers = sorted({t for d, ts in snaps.items() if pd.Timestamp(d) >= lo for t in ts})
    panel = yfinance_panel(tickers, str(lo.date()), end)
    missing = sorted(set(tickers) - set(panel.instruments))
    panel, log = fill_delisted(panel, missing, str(lo.date()), end)
    universe = np.array(tickers)
    mask_all = from_snapshots(panel.dates, panel.instruments, snaps)
    panel = Panel(panel.dates, panel.instruments, panel.fields, mask_all, "US", panel.meta)
    out = PROCESSED / "us_sp500.npz"
    panel.save(out)
    cov = coverage_table(panel)
    (PROCESSED.parent / "coverage").mkdir(parents=True, exist_ok=True)
    cov.to_csv(PROCESSED.parent / "coverage" / "us_sp500_coverage.csv", index=False)
    qa = qa_checklist(panel)
    qa.save(PROCESSED.parent / "coverage" / "us_sp500_qa.json")
    Path(PROCESSED.parent / "coverage" / "us_delisted_fill_log.json").write_text(json.dumps(log, indent=1))
    return {"panel": str(out), "n_universe": int(len(universe)), "n_with_prices": int(panel.N),
            "missing_after_fill": [t for t, s in log.items() if s == "missing"], "qa_passed": qa.passed}


def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))
