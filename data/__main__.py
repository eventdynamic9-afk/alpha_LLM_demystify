"""Data-layer CLI.

    python -m data synthetic --out data/processed/synthetic.npz [--stocks 200 --days 2600]
    python -m data rebuild cn --universe csi500 [--no-cross-check]
    python -m data rebuild us [--membership-url URL]
    python -m data rebuild us-plotly          # documented fallback when §5.4 sources are unreachable
    python -m data refs us
    python -m data qa --panel data/processed/cn_csi500.npz [--second ...]
    python -m data coverage --panel data/processed/us_sp500.npz
    python -m data verify-manifest
"""
from __future__ import annotations

import argparse
import json
import sys

from .panel import Panel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--out", required=True)
    s.add_argument("--stocks", type=int, default=120)
    s.add_argument("--days", type=int, default=2600)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--market", default="CN")
    s.add_argument("--start", default="2014-01-02")
    r = sub.add_parser("rebuild")
    r.add_argument("market", choices=["cn", "us", "us-plotly"])
    r.add_argument("--universe", default="csi500")
    r.add_argument("--start", default="2015-01-01")
    r.add_argument("--end", default="2024-12-31")
    r.add_argument("--no-cross-check", action="store_true")
    r.add_argument("--membership-url", default=None)
    f = sub.add_parser("refs")
    f.add_argument("market", choices=["us"])
    q = sub.add_parser("qa")
    q.add_argument("--panel", required=True)
    q.add_argument("--second")
    q.add_argument("--out")
    c = sub.add_parser("coverage")
    c.add_argument("--panel", required=True)
    sub.add_parser("verify-manifest")
    a = ap.parse_args(argv)

    if a.cmd == "synthetic":
        from .synthetic import synthetic_panel

        p = synthetic_panel(a.stocks, a.days, a.seed, a.start, a.market)
        p.save(a.out)
        print(f"wrote {a.out}: {p.T} dates x {p.N} instruments")
    elif a.cmd == "rebuild":
        if a.market == "cn":
            from .rebuild.cn import rebuild_cn

            print(json.dumps(rebuild_cn(a.universe, a.start, a.end, not a.no_cross_check), indent=1, default=str))
        elif a.market == "us-plotly":
            from .rebuild.us_plotly import rebuild_us_plotly

            print(json.dumps(rebuild_us_plotly(membership_url=a.membership_url), indent=1, default=str))
        else:
            from .rebuild.us import SP500_HISTORY_URL, rebuild_us

            print(json.dumps(rebuild_us(a.start, a.end, a.membership_url or SP500_HISTORY_URL), indent=1, default=str))
    elif a.cmd == "refs":
        from .rebuild.common import PROCESSED
        from .rebuild.references import us_factor_returns

        df = us_factor_returns()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED / "us_factor_returns_daily.csv")
        print(df.tail())
    elif a.cmd == "qa":
        from .validation import qa_checklist

        p = Panel.load(a.panel)
        second = Panel.load(a.second) if a.second else None
        rep = qa_checklist(p, second)
        if a.out:
            rep.save(a.out)
        print(json.dumps(rep.checks, indent=1, default=str))
        return 0 if rep.passed else 1
    elif a.cmd == "coverage":
        from .validation import coverage_table

        print(coverage_table(Panel.load(a.panel)).to_string(index=False))
    elif a.cmd == "verify-manifest":
        from .manifest import verify_manifest

        bad = verify_manifest()
        print("all hashes match" if not bad else f"mismatched/missing: {bad}")
        return 0 if not bad else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
