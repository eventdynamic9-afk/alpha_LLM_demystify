"""In-memory daily stock panel: dates x instruments arrays per field plus point-in-time membership."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FIELDS = ("open", "high", "low", "close", "volume", "vwap", "amount")


@dataclass
class Panel:
    dates: np.ndarray                      # datetime64[D], shape (T,)
    instruments: np.ndarray                # str, shape (N,)
    fields: dict[str, np.ndarray]          # name -> float64 (T, N)
    member: np.ndarray                     # bool (T, N): point-in-time universe membership
    market: str = "CN"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        self.instruments = np.asarray(self.instruments).astype(str)
        T, N = len(self.dates), len(self.instruments)
        for k, v in list(self.fields.items()):
            v = np.asarray(v, dtype=np.float64)
            if v.shape != (T, N):
                raise ValueError(f"field {k} has shape {v.shape}, expected {(T, N)}")
            self.fields[k] = v
        self.member = np.asarray(self.member, dtype=bool)
        if self.member.shape != (T, N):
            raise ValueError("membership mask shape mismatch")
        if T > 1 and not np.all(np.diff(self.dates.astype("int64")) > 0):
            raise ValueError("dates must be strictly increasing")
        if len(set(self.instruments.tolist())) != N:
            raise ValueError("duplicate instruments")
        if "vwap" not in self.fields and {"high", "low", "close"} <= set(self.fields):
            self.fields["vwap"] = (self.fields["high"] + self.fields["low"] + self.fields["close"]) / 3.0
            self.meta["vwap_proxy"] = "hlc3"
        if "amount" not in self.fields and {"close", "volume"} <= set(self.fields):
            self.fields["amount"] = self.fields["vwap"] * self.fields["volume"]
            self.meta["amount_proxy"] = "vwap*volume"

    # ------------------------------------------------------------------ shape
    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.instruments)

    def get(self, name: str) -> np.ndarray:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(f"panel has no field {name!r}") from exc

    # ------------------------------------------------------------------ slicing
    def date_mask(self, start: str | None = None, end: str | None = None) -> np.ndarray:
        m = np.ones(self.T, dtype=bool)
        if start is not None:
            m &= self.dates >= np.datetime64(start, "D")
        if end is not None:
            m &= self.dates <= np.datetime64(end, "D")
        return m

    def take_dates(self, idx) -> "Panel":
        idx = np.asarray(idx)
        return Panel(self.dates[idx], self.instruments.copy(), {k: v[idx] for k, v in self.fields.items()},
                     self.member[idx], self.market, dict(self.meta))

    def slice(self, start: str | None = None, end: str | None = None, warmup: int = 0) -> "Panel":
        """Dates in [start, end], optionally keeping ``warmup`` extra rows before ``start``."""
        m = self.date_mask(start, end)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            raise ValueError(f"no dates in [{start}, {end}]")
        lo = max(0, idx[0] - warmup)
        return self.take_dates(np.arange(lo, idx[-1] + 1))

    def prefix(self, n_rows: int) -> "Panel":
        """First ``n_rows`` dates (used by the dynamic truncation test, §6.3)."""
        return self.take_dates(np.arange(n_rows))

    def take_instruments(self, idx) -> "Panel":
        idx = np.asarray(idx)
        return Panel(self.dates.copy(), self.instruments[idx], {k: v[:, idx] for k, v in self.fields.items()},
                     self.member[:, idx], self.market, dict(self.meta))

    def with_fields(self, **arrays: np.ndarray) -> "Panel":
        new = dict(self.fields)
        for k, v in arrays.items():
            new[k] = np.asarray(v, dtype=np.float64)
        return replace(self, fields=new, meta=dict(self.meta))

    def copy(self) -> "Panel":
        return Panel(self.dates.copy(), self.instruments.copy(), {k: v.copy() for k, v in self.fields.items()},
                     self.member.copy(), self.market, dict(self.meta))

    # ------------------------------------------------------------------ conversions
    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.get(name), index=pd.DatetimeIndex(self.dates), columns=self.instruments)

    def years(self) -> np.ndarray:
        return self.dates.astype("datetime64[Y]").astype(int) + 1970

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dates=self.dates.astype("int64"), instruments=self.instruments,
                            member=self.member, market=np.array(self.market),
                            meta=np.array(repr(self.meta)), **{f"f_{k}": v for k, v in self.fields.items()})

    @classmethod
    def load(cls, path: str | Path) -> "Panel":
        import ast as _ast

        z = np.load(path, allow_pickle=False)
        fields = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
        meta = _ast.literal_eval(str(z["meta"])) if "meta" in z.files else {}
        return cls(z["dates"].astype("datetime64[D]"), z["instruments"], fields, z["member"],
                   str(z["market"]), meta)

    @classmethod
    def from_long(cls, df: pd.DataFrame, market: str, member: pd.DataFrame | None = None,
                  date_col: str = "date", inst_col: str = "instrument") -> "Panel":
        """Build from a long table with one row per (date, instrument) and one column per field."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        if df.duplicated([date_col, inst_col]).any():
            raise ValueError("duplicate (date, instrument) rows")
        dates = np.sort(df[date_col].unique())
        insts = np.sort(df[inst_col].unique())
        fields = {}
        for f in BASE_FIELDS:
            if f in df.columns:
                wide = df.pivot(index=date_col, columns=inst_col, values=f).reindex(index=dates, columns=insts)
                fields[f] = wide.to_numpy(dtype=np.float64)
        if member is None:
            mem = np.isfinite(fields["close"])
        else:
            mem = member.reindex(index=dates, columns=insts).fillna(False).to_numpy(dtype=bool)
        return cls(dates.astype("datetime64[D]"), insts, fields, mem, market)
