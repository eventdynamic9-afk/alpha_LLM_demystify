"""Cross-source validation, coverage tables and the pre-narration data QA checklist (§5.3, §5.4, §5.9)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import daily_returns
from .panel import Panel

MISMATCH_BP = 5.0


def align(a: Panel, b: Panel) -> tuple[Panel, Panel]:
    dates = np.intersect1d(a.dates, b.dates)
    insts = np.intersect1d(a.instruments, b.instruments)
    ia = np.searchsorted(a.dates, dates)
    ib = np.searchsorted(b.dates, dates)
    ja = np.array([np.flatnonzero(a.instruments == s)[0] for s in insts])
    jb = np.array([np.flatnonzero(b.instruments == s)[0] for s in insts])
    return a.take_dates(ia).take_instruments(ja), b.take_dates(ib).take_instruments(jb)


def cross_source_mismatch(a: Panel, b: Panel, bp: float = MISMATCH_BP) -> tuple[np.ndarray, pd.DataFrame]:
    """Flag stock-days with |r_A - r_B| > ``bp`` basis points; returns (flag mask on a's grid of the
    aligned panels, mismatch rate by year).  Flagged days are excluded from behavioral statistics and
    kept for mechanistic contexts (§5.3 rule 2)."""
    a2, b2 = align(a, b)
    ra, rb = daily_returns(a2), daily_returns(b2)
    both = np.isfinite(ra) & np.isfinite(rb)
    flag = both & (np.abs(ra - rb) > bp / 1e4)
    years = a2.years()
    rows = []
    for y in np.unique(years):
        m = years == y
        n = int(both[m].sum())
        rows.append({"year": int(y), "compared": n, "flagged": int(flag[m].sum()),
                     "mismatch_rate": float(flag[m].sum() / n) if n else float("nan")})
    return flag, pd.DataFrame(rows)


def coverage_table(panel: Panel) -> pd.DataFrame:
    """Coverage = share of point-in-time member-days with a valid close, by year (§5.4)."""
    valid = np.isfinite(panel.get("close"))
    years = panel.years()
    rows = []
    for y in np.unique(years):
        m = years == y
        md = panel.member[m]
        n = int(md.sum())
        rows.append({"year": int(y), "member_days": n, "valid_days": int((valid[m] & md).sum()),
                     "coverage": float((valid[m] & md).sum() / n) if n else float("nan")})
    return pd.DataFrame(rows)


def shumway_delisting_bound(panel: Panel, delisting_return: float = -0.30) -> np.ndarray:
    """Pessimistic delisting-return imputation (Shumway 1997): the last member day before a stock's
    data ends inside the universe gets ``delisting_return`` (US survivorship sensitivity, §5.4)."""
    r = daily_returns(panel)
    close = panel.get("close")
    out = r.copy()
    for j in range(panel.N):
        idx = np.flatnonzero(np.isfinite(close[:, j]))
        if len(idx) == 0:
            continue
        last = idx[-1]
        if last + 1 < panel.T and panel.member[last, j]:
            out[last + 1, j] = delisting_return
    return out


@dataclass
class QAReport:
    checks: dict = field(default_factory=dict)
    passed: bool = True

    def add(self, name: str, ok: bool, detail) -> None:
        self.checks[name] = {"ok": bool(ok), "detail": detail}
        self.passed &= bool(ok)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1, default=str))


def qa_checklist(panel: Panel, second: Panel | None = None, manifest: str | Path | None = None) -> QAReport:
    """Run the §5.9 checklist (items needing external files are reported as 'not run' when absent)."""
    rep = QAReport()
    rep.add("adjustment_identified", "adjusted" in panel.meta or panel.meta.get("source") == "synthetic",
            {"adjusted": panel.meta.get("adjusted", "synthetic")})
    dup_dates = len(np.unique(panel.dates)) != panel.T
    dup_inst = len(np.unique(panel.instruments)) != panel.N
    rep.add("no_duplicates", not (dup_dates or dup_inst), {"dup_dates": dup_dates, "dup_instruments": dup_inst})
    weekend = int(np.isin(pd.DatetimeIndex(panel.dates).dayofweek, [5, 6]).sum())
    rep.add("trading_calendar", weekend == 0, {"weekend_dates": weekend})
    price = np.stack([panel.get(f) for f in ("open", "high", "low", "close")])
    nonpos = int((np.nan_to_num(price, nan=1.0) <= 0).sum())
    zero_vol = int((panel.get("volume") == 0).sum())
    susp = int((np.isnan(panel.get("close")) & panel.member).sum())
    hl_bad = int((np.nan_to_num(panel.get("high") - panel.get("low"), nan=0.0) < 0).sum())
    rep.add("prices_positive_and_consistent", nonpos == 0 and hl_bad == 0,
            {"nonpositive_prices": nonpos, "high_below_low": hl_bad})
    rep.add("suspensions_flagged", True, {"zero_volume_days": zero_vol, "member_days_without_price": susp})
    r = daily_returns(panel)
    jumps = int((np.abs(np.nan_to_num(r)) > 0.5).sum())
    rep.add("large_jumps", True, {"abs_daily_return_gt_50pct": jumps})
    if second is not None:
        _, rates = cross_source_mismatch(panel, second)
        rep.add("cross_source_mismatch", True, rates.to_dict(orient="records"))
    else:
        rep.add("cross_source_mismatch", True, "not run (no second source supplied)")
    rep.add("point_in_time_membership", not panel.member.all(), {"member_share": float(panel.member.mean())})
    rep.add("coverage", True, coverage_table(panel).to_dict(orient="records"))
    if manifest is not None and Path(manifest).exists():
        from .manifest import verify_manifest

        bad = verify_manifest(manifest)
        rep.add("manifest_hashes", not bad, {"mismatched_or_missing": bad})
    else:
        rep.add("manifest_hashes", True, "not run (no manifest supplied)")
    rep.add("missing_fields", all(f in panel.fields for f in ("open", "high", "low", "close", "volume")),
            sorted(panel.fields))
    return rep
