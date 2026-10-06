"""Reader for Qlib's binary data layout (no ``pyqlib`` needed).

Layout (``~/.qlib/qlib_data/cn_data``):
  calendars/day.txt                 one trading date per line
  instruments/<universe>.txt        ``SYMBOL<TAB>start<TAB>end`` membership intervals (repeatable)
  features/<symbol lower>/<field>.day.bin   little-endian float32; element 0 = start index into the
                                            calendar, elements 1.. = values for consecutive dates

Point-in-time membership comes straight from the instrument intervals (never from today's list).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .panel import Panel

QLIB_FIELD_MAP = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
                  "vwap": "vwap", "amount": "amount"}


def read_calendar(root: Path) -> np.ndarray:
    lines = (Path(root) / "calendars" / "day.txt").read_text().split()
    return pd.to_datetime(lines).values.astype("datetime64[D]")


def read_instruments(root: Path, universe: str) -> pd.DataFrame:
    path = Path(root) / "instruments" / f"{universe}.txt"
    df = pd.read_csv(path, sep="\t", header=None, names=["symbol", "start", "end"], dtype=str)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def read_feature(root: Path, symbol: str, field: str, n_cal: int) -> np.ndarray | None:
    path = Path(root) / "features" / symbol.lower() / f"{field}.day.bin"
    if not path.exists():
        return None
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0:
        return None
    start = int(raw[0])
    out = np.full(n_cal, np.nan, dtype=np.float64)
    vals = raw[1:].astype(np.float64)
    end = min(n_cal, start + len(vals))
    out[start:end] = vals[: end - start]
    return out


def load_qlib_panel(root: str | Path, universe: str = "csi500", start: str | None = None,
                    end: str | None = None, market: str = "CN", fields=tuple(QLIB_FIELD_MAP)) -> Panel:
    root = Path(root).expanduser()
    cal = read_calendar(root)
    inst = read_instruments(root, universe)
    symbols = sorted(inst["symbol"].unique())
    T = len(cal)
    data = {f: np.full((T, len(symbols)), np.nan) for f in fields}
    member = np.zeros((T, len(symbols)), dtype=bool)
    cal_ts = cal.astype("datetime64[ns]")
    for j, sym in enumerate(symbols):
        for f in fields:
            v = read_feature(root, sym, QLIB_FIELD_MAP[f], T)
            if v is not None:
                data[f][:, j] = v
        for _, row in inst[inst["symbol"] == sym].iterrows():
            member[:, j] |= (cal_ts >= row["start"].to_datetime64()) & (cal_ts <= row["end"].to_datetime64())
    panel = Panel(cal, np.array(symbols), {k: v for k, v in data.items() if np.isfinite(v).any()}, member,
                  market, {"source": f"qlib_bin:{root}", "universe": universe, "adjusted": True})
    if start or end:
        panel = panel.slice(start, end, warmup=260)
    return panel


def write_qlib_bins(panel: Panel, root: str | Path, universe: str = "all") -> None:
    """Inverse of :func:`load_qlib_panel` (used to build test fixtures in Qlib layout)."""
    root = Path(root)
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(panel.dates)
    (root / "calendars" / "day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in dates) + "\n")
    lines = []
    for j, sym in enumerate(panel.instruments):
        m = panel.member[:, j]
        # contiguous membership intervals
        k = 0
        while k < len(m):
            if m[k]:
                s = k
                while k + 1 < len(m) and m[k + 1]:
                    k += 1
                lines.append(f"{sym}\t{dates[s]:%Y-%m-%d}\t{dates[k]:%Y-%m-%d}")
            k += 1
        fdir = root / "features" / sym.lower()
        fdir.mkdir(parents=True, exist_ok=True)
        for f, arr in panel.fields.items():
            col = arr[:, j]
            idx = np.flatnonzero(np.isfinite(col))
            if len(idx) == 0:
                continue
            s, e = idx[0], idx[-1] + 1
            np.concatenate([[float(s)], col[s:e]]).astype("<f4").tofile(fdir / f"{f}.day.bin")
    (root / "instruments" / f"{universe}.txt").write_text("\n".join(lines) + "\n")
