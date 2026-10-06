"""Point-in-time universe membership (§5.3 rule 3, §5.4): never use today's constituent list for history."""
from __future__ import annotations

import numpy as np
import pandas as pd


def from_intervals(dates: np.ndarray, instruments: np.ndarray, intervals: pd.DataFrame) -> np.ndarray:
    """Intervals table with columns symbol, start, end (inclusive) -> (T, N) bool mask."""
    idx = {s: j for j, s in enumerate(instruments)}
    d = np.asarray(dates, dtype="datetime64[D]")
    mask = np.zeros((len(d), len(instruments)), dtype=bool)
    for row in intervals.itertuples(index=False):
        j = idx.get(row.symbol)
        if j is None:
            continue
        s = np.datetime64(pd.Timestamp(row.start).date(), "D")
        e = np.datetime64(pd.Timestamp(row.end).date(), "D")
        mask[:, j] |= (d >= s) & (d <= e)
    return mask


def from_snapshots(dates: np.ndarray, instruments: np.ndarray, snapshots: dict) -> np.ndarray:
    """Dated constituent snapshots {date: [symbols]} -> mask; each snapshot holds until the next one.

    Works for BaoStock ``query_zz500_stocks(date=...)`` pulls and for the fja05680/sp500 history file.
    """
    d = np.asarray(dates, dtype="datetime64[D]")
    idx = {s: j for j, s in enumerate(instruments)}
    keys = sorted(np.datetime64(pd.Timestamp(k).date(), "D") for k in snapshots)
    by_key = {np.datetime64(pd.Timestamp(k).date(), "D"): v for k, v in snapshots.items()}
    mask = np.zeros((len(d), len(instruments)), dtype=bool)
    for i, k in enumerate(keys):
        nxt = keys[i + 1] if i + 1 < len(keys) else np.datetime64("2262-01-01")
        rows = (d >= k) & (d < nxt)
        cols = [idx[s] for s in by_key[k] if s in idx]
        if cols:
            mask[np.ix_(rows, cols)] = True
    return mask


def parse_sp500_history(csv_path: str) -> dict:
    """fja05680/sp500 'S&P 500 Historical Components & Changes' CSV: columns date, tickers (comma list)."""
    df = pd.read_csv(csv_path)
    col = "tickers" if "tickers" in df.columns else df.columns[1]
    return {row["date"]: [t.strip().replace(".", "-") for t in str(row[col]).split(",") if t.strip()]
            for _, row in df.iterrows()}
