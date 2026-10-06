"""Public factor / portfolio libraries for return-level behavioral checks (§5.5).

US: Ken French daily FF5 + momentum + short/long-term reversal; Open Source Asset Pricing predictor
portfolio returns (portfolio level only — firm-level merges need a CRSP crosswalk); Hou-Xue-Zhang
q-factors; JKP factors (non-commercial license; load the CSV you downloaded).
CN: Liu-Stambaugh-Yuan CH-3 / CH-4 (load the file from Stambaugh's data page).
All returns are converted from percent to decimal and indexed by date.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from .common import RAW, download

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_DAILY = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
    "st_rev": "F-F_ST_Reversal_Factor_daily_CSV.zip",
    "lt_rev": "F-F_LT_Reversal_Factor_daily_CSV.zip",
}


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the first daily table of a Ken French CSV (header row, YYYYMMDD rows, percent values)."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        cells = [c.strip() for c in ln.split(",")]
        if len(cells) > 1 and cells[0] == "" and any(c for c in cells[1:]):
            start = i
            break
    if start is None:
        raise ValueError("no table header found")
    header = ["date"] + [c.strip() for c in lines[start].split(",")[1:]]
    rows = []
    for ln in lines[start + 1:]:
        cells = [c.strip() for c in ln.split(",")]
        if len(cells) < 2 or not cells[0].isdigit() or len(cells[0]) != 8:
            if rows:
                break
            continue
        rows.append(cells)
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df.set_index("date")


def french_daily(name: str) -> pd.DataFrame:
    fname = FRENCH_DAILY[name]
    path = download(FRENCH_BASE + fname, RAW / "french" / fname, "Ken French Data Library",
                    license="free academic use; cite", notes=f"daily {name}")
    with zipfile.ZipFile(path) as zf:
        inner = zf.namelist()[0]
        return parse_french_csv(zf.read(inner).decode("latin-1"))


def us_factor_returns() -> pd.DataFrame:
    """Daily FF5 + MOM + ST-reversal + LT-reversal (decimal), the US control set of §10.3."""
    ff5 = french_daily("ff5")
    out = ff5.join(french_daily("mom"), how="left").join(french_daily("st_rev"), how="left")
    out = out.join(french_daily("lt_rev"), how="left")
    out.columns = [c.strip().replace(" ", "_") for c in out.columns]
    return out


def osap_portfolios(save_to: Path = RAW / "osap_portfolios.parquet") -> pd.DataFrame:
    """OSAP predictor portfolio returns via the `openassetpricing` package (pip install openassetpricing)."""
    import openassetpricing as oap

    openap = oap.OpenAP()
    df = openap.dl_port("op", "pandas")
    save_to.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(save_to)
    except Exception:
        df.to_csv(save_to.with_suffix(".csv"), index=False)
    return df


def q_factors(url: str) -> pd.DataFrame:
    """Hou-Xue-Zhang q5 daily factors; the file name on global-q.org carries the vintage year."""
    path = download(url, RAW / "q5_factors_daily.csv", "global-q.org", license="free academic use; cite")
    df = pd.read_csv(path)
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
    df = df.set_index(date_col)
    return df.select_dtypes("number") / 100.0


def jkp_factors(csv_path: str) -> pd.DataFrame:
    """JKP Global Factor Data CSV downloaded from jkpfactors.com (non-commercial license)."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.pivot_table(index="date", columns="name", values="ret")


def ch_factors(path_or_url: str) -> pd.DataFrame:
    """Liu-Stambaugh-Yuan CH-3/CH-4 file (monthly CH-3 updated; daily CH-3/CH-4 through 2021)."""
    if path_or_url.startswith("http"):
        path = download(path_or_url, RAW / Path(path_or_url).name, "Stambaugh data page",
                        license="free academic use; cite LSY (2019)")
    else:
        path = Path(path_or_url)
    if str(path).endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    date_col = df.columns[0]
    s = df[date_col].astype(str)
    fmt = "%Y%m%d" if s.str.len().eq(8).all() else "%Y%m"
    df[date_col] = pd.to_datetime(s, format=fmt)
    df = df.set_index(date_col).select_dtypes("number")
    return df / 100.0 if df.abs().mean().mean() > 0.05 else df


def binance_daily(symbol: str, months: list[str]) -> pd.DataFrame:
    """Optional crypto robustness universe (§5.6) from Binance's public data archive."""
    frames = []
    for m in months:
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{m}.zip"
        path = download(url, RAW / "binance" / f"{symbol}-1d-{m}.zip", "Binance public data")
        with zipfile.ZipFile(path) as zf:
            raw = zf.read(zf.namelist()[0])
        df = pd.read_csv(io.BytesIO(raw), header=None).iloc[:, :8]
        df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "amount"]
        unit = "us" if df["open_time"].max() > 1e14 else "ms"
        df["date"] = pd.to_datetime(df["open_time"], unit=unit).dt.normalize()
        df["instrument"] = symbol
        frames.append(df[["date", "instrument", "open", "high", "low", "close", "volume", "amount"]])
    return pd.concat(frames, ignore_index=True)


def jpx_panel(stock_prices_csv: str):
    """Kaggle JPX Tokyo Stock Exchange Prediction ``stock_prices.csv`` -> Panel (check competition rules)."""
    from ..panel import Panel

    df = pd.read_csv(stock_prices_csv)
    df = df.rename(columns={"Date": "date", "SecuritiesCode": "instrument", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume"})
    # cumulative adjustment from AdjustmentFactor (applied backward)
    df = df.sort_values(["instrument", "date"])
    adj = df.groupby("instrument")["AdjustmentFactor"].transform(lambda s: s[::-1].cumprod()[::-1].shift(-1).fillna(1))
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] / adj
    df["volume"] = df["volume"] * adj
    df["instrument"] = df["instrument"].astype(str)
    p = Panel.from_long(df[["date", "instrument", "open", "high", "low", "close", "volume"]], "JP")
    p.meta.update({"source": "kaggle_jpx", "adjusted": True})
    return p
