"""Build the committed fixture panels and their expected outputs (§6.4).

* ``synthetic_gbm.npz``  — deterministic synthetic GBM panel (30 stocks x 160 days) with defects.
* ``expected_outputs.json`` — E2 outputs of a fixed formula list on that panel (rounded checksum +
  a sample of cells), so any future change of either executor's semantics is caught.
* ``real_slice.npz`` + ``expected_outputs_real.json`` — a real 50-stock x 250-day slice. The committed
  slice comes from the US fallback panel (CC0 source, redistributable; ``python -m data rebuild
  us-plotly``) and deliberately includes names with cleaning events (suspension-like gaps, a reversed
  spike, a bad open print, a 2-for-1 distribution): ``python -m executors.fixtures.make_fixtures --real
  data/processed/us_sp500_plotly.npz``. A CN slice is committed only where the data license allows it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.panel import Panel
from data.synthetic import synthetic_panel
from dsl import parse

HERE = Path(__file__).resolve().parent

FIXTURE_FORMULAS = [
    "Mean($close, 5)/$close",
    "CSRank(-1*($close/Ref($close, 5)-1))",
    "Corr($close, Log($volume+1), 10)",
    "TsRank($volume, 10)",
    "IdxMax($high, 10)/10",
    "WMA($close, 6)",
    "Slope($close, 10)",
    "Rsquare($close, 10)",
    "Resi($close, 10)",
    "Quantile($close, 10, 0.8)",
    "CSZScore(Std($close/Ref($close, 1)-1, 20))",
    "CSScale($close-$open)",
    "If($close > $open, $high-$close, $open-$low)",
    "Cov($high, $low, 8)",
    "SignedPower($close-Mean($close, 10), 2)",
]


def build_synthetic() -> Panel:
    return synthetic_panel(n_stocks=30, n_days=160, seed=7)


REAL_MUST_INCLUDE = ("DISCA", "LNT", "CHD", "MRO", "NWL", "BBY", "AMD")   # cleaning events in 2014-2017


def build_real_slice(full: Panel, n_stocks: int = 50, n_days: int = 250, start: str = "2014-05-01") -> Panel:
    rows = np.flatnonzero(full.dates >= np.datetime64(start))[:n_days]
    p = full.take_dates(rows)
    have = [s for s in REAL_MUST_INCLUDE if s in set(p.instruments.tolist())]
    rng = np.random.default_rng(20261006)
    rest = [s for s in p.instruments.tolist() if s not in have and np.isfinite(p.get("close")[:, list(p.instruments).index(s)]).all()]
    pick = have + sorted(rng.choice(rest, size=n_stocks - len(have), replace=False).tolist())
    idx = np.sort([list(p.instruments).index(s) for s in pick])
    return p.take_instruments(idx)


def expected_outputs(panel: Panel) -> dict:
    from executors import E2Executor

    e2 = E2Executor()
    rng = np.random.default_rng(0)
    out = {}
    for src in FIXTURE_FORMULAS:
        v = e2.evaluate(parse(src), panel)
        finite = np.argwhere(np.isfinite(v))
        pick = finite[rng.choice(len(finite), size=min(12, len(finite)), replace=False)]
        out[src] = {
            "n_finite": int(np.isfinite(v).sum()),
            "nansum": float(np.nansum(v)),
            "cells": [[int(t), int(i), float(v[t, i])] for t, i in pick],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="path to a rebuilt CN panel .npz to slice 50 stocks from")
    args = ap.parse_args()
    p = build_synthetic()
    p.save(HERE / "synthetic_gbm.npz")
    (HERE / "expected_outputs.json").write_text(json.dumps(expected_outputs(p), indent=1))
    if args.real:
        real = build_real_slice(Panel.load(args.real))
        real.save(HERE / "real_slice.npz")
        (HERE / "expected_outputs_real.json").write_text(json.dumps(expected_outputs(real), indent=1))
    print("fixtures written to", HERE)


if __name__ == "__main__":
    main()
