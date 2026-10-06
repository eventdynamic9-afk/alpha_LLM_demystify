"""China A-shares free stack (§5.3).

CN-A (primary): community Qlib bins from chenditc/investment_data (Qlib's README points there while
the official dataset is disabled).  CN-B (cross-check): BaoStock daily K-lines and dated CSI300/CSI500
constituent queries.  CN-C (optional third source): AkShare / Tushare Pro.

Usage::

    python -m data rebuild cn --universe csi500 --start 2015-01-01 --end 2024-12-31
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..membership import from_snapshots
from ..panel import Panel
from ..qlib_bin import load_qlib_panel
from .common import PROCESSED, RAW, download

CHENDITC_URL = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"


def fetch_cn_qlib_bins(dest: Path = RAW / "cn_data") -> Path:
    """Download and extract the community Qlib bins (vendor-derived tables: do not redistribute)."""
    tarball = download(CHENDITC_URL, RAW / "qlib_bin.tar.gz", "chenditc/investment_data",
                       license="repo Apache-2.0; vendor tables (Wind, Caihui) undocumented",
                       notes="community Qlib bins; release manifest + validation script in repo")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            parts = Path(m.name).parts
            if len(parts) <= 1:
                continue
            m.name = str(Path(*parts[1:]))          # --strip-components=1
            if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                continue                              # path traversal guard (untrusted archive)
            members.append(m)
        tf.extractall(dest, members=members, filter="data")
    return dest


def build_cn_panel(universe: str = "csi500", start: str = "2015-01-01", end: str = "2024-12-31",
                   root: Path = RAW / "cn_data") -> Path:
    panel = load_qlib_panel(root, universe, start, end, market="CN")
    panel.meta["price_limit"] = 0.10
    out = PROCESSED / f"cn_{universe}.npz"
    panel.save(out)
    return out


# ------------------------------------------------------------------------------------- BaoStock
def _bs_code(qlib_symbol: str) -> str:
    """SH600000 -> sh.600000"""
    return f"{qlib_symbol[:2].lower()}.{qlib_symbol[2:]}"


def _qlib_symbol(bs_code: str) -> str:
    return bs_code.replace(".", "").upper()


def baostock_constituents(universe: str, dates: list[str]) -> dict:
    """Dated constituent snapshots via BaoStock (check how far back `date` returns history)."""
    import baostock as bs

    fn = {"csi500": bs.query_zz500_stocks, "csi300": bs.query_hs300_stocks}[universe]
    bs.login()
    try:
        snaps = {}
        for d in dates:
            df = fn(date=d).get_data()
            if len(df):
                snaps[d] = [_qlib_symbol(c) for c in df["code"]]
        return snaps
    finally:
        bs.logout()


def baostock_panel(symbols: list[str], start: str, end: str, adjustflag: str = "1") -> Panel:
    """Second-source CN panel. adjustflag: '1' backward-adjusted, '2' forward, '3' none (check docs)."""
    import baostock as bs

    bs.login()
    try:
        frames = []
        for sym in symbols:
            rs = bs.query_history_k_data_plus(_bs_code(sym), "date,open,high,low,close,volume,amount",
                                              start_date=start, end_date=end, frequency="d",
                                              adjustflag=adjustflag)
            df = rs.get_data()
            if df is None or not len(df):
                continue
            df["instrument"] = sym
            frames.append(df)
    finally:
        bs.logout()
    long = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume", "amount"):
        long[c] = pd.to_numeric(long[c], errors="coerce")
    long["vwap"] = long["amount"] / long["volume"].replace(0, np.nan)
    p = Panel.from_long(long, "CN")
    p.meta.update({"source": "baostock", "adjustflag": adjustflag, "adjusted": adjustflag != "3"})
    return p


def baostock_all_stock(day: str) -> list[str]:
    """All securities trading on ``day`` (point-in-time, includes names that later delist)."""
    import baostock as bs

    bs.login()
    try:
        df = bs.query_all_stock(day=day).get_data()
    finally:
        bs.logout()
    return [_qlib_symbol(c) for c in df["code"] if c.startswith(("sh.6", "sz.0", "sz.3"))]


def rebuild_cn(universe: str = "csi500", start: str = "2015-01-01", end: str = "2024-12-31",
               cross_check: bool = True) -> dict:
    from ..validation import cross_source_mismatch, qa_checklist

    root = fetch_cn_qlib_bins()
    out = build_cn_panel(universe, start, end, root)
    panel = Panel.load(out)
    result = {"panel": str(out)}
    if cross_check:
        b = baostock_panel(list(panel.instruments), start, end)
        b.save(PROCESSED / f"cn_{universe}_baostock.npz")
        flag, rates = cross_source_mismatch(panel, b)
        rates.to_csv(PROCESSED.parent / "coverage" / f"cn_{universe}_mismatch_by_year.csv", index=False)
        result["mismatch"] = rates.to_dict(orient="records")
        qa = qa_checklist(panel, b)
    else:
        qa = qa_checklist(panel)
    qa.save(PROCESSED.parent / "coverage" / f"cn_{universe}_qa.json")
    result["qa_passed"] = qa.passed
    return result


def membership_from_baostock(panel: Panel, universe: str, every: str = "M") -> np.ndarray:
    dates = pd.date_range(pd.Timestamp(panel.dates[0]), pd.Timestamp(panel.dates[-1]), freq=every)
    snaps = baostock_constituents(universe, [d.strftime("%Y-%m-%d") for d in dates])
    return from_snapshots(panel.dates, panel.instruments, snaps)
