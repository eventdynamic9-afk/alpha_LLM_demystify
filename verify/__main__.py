"""Verifier CLI.

    python -m verify run --formulas runs/x/formulas.jsonl --claims runs/x/claims.jsonl \
        --panel data/processed/cn_csi500.npz --out runs/x/verdicts.jsonl [--fast]
    python -m verify calibrate --out runs/calibration [--fast]
    python -m verify drivers --formulas runs/x/formulas.jsonl --panel ... --out runs/x/drivers.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from configs import study
from data.panel import Panel
from dsl import parse


def _ctx(panel_path: str, fast: bool, windows: str | None = None):
    from .context import VerificationContext

    p = Panel.load(panel_path)
    w = json.loads(windows) if windows else None
    if w is None and p.meta.get("source") == "synthetic":
        cut = str(p.dates[int(p.T * 0.6)])
        w = {"train": (str(p.dates[0]), cut), "test": (cut, str(p.dates[-1]))}
    return VerificationContext(p, windows=w or {}, fast=fast)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(args) -> int:
    from .dispatcher import verify_claim

    ctx = _ctx(args.panel, args.fast, args.windows)
    formulas = {f["formula_id"]: f for f in _read_jsonl(args.formulas)}
    rationale_formula = {}
    if args.rationales:
        rationale_formula = {r["rationale_id"]: r["formula_id"] for r in _read_jsonl(args.rationales)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for c in _read_jsonl(args.claims):
            fid = c.get("formula_id") or rationale_formula.get(c.get("rationale_id"))
            rec = formulas.get(fid)
            node = parse(rec["dsl"]) if rec else None
            v = verify_claim(c, node, ctx)
            fh.write(json.dumps({"claim_id": c["claim_id"], "rationale_id": c.get("rationale_id"),
                                 "formula_id": fid, "market": ctx.panel.market, **v.to_dict()}, default=float) + "\n")
            n += 1
    print(f"wrote {n} verdicts to {out}")
    return 0


def drivers(args) -> int:
    from .drivers import true_drivers

    ctx = _ctx(args.panel, args.fast, args.windows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for rec in _read_jsonl(args.formulas):
            node = parse(rec["dsl"])
            drv, st = true_drivers(node, ctx)
            fh.write(json.dumps({"formula_id": rec["formula_id"], "true_drivers": drv, "total_effect": st}) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--formulas", required=True)
    r.add_argument("--claims", required=True)
    r.add_argument("--rationales")
    r.add_argument("--panel", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--windows", help='JSON, e.g. {"train": ["2015-01-01", "2019-12-31"], "test": [...]}')
    r.add_argument("--fast", action="store_true")
    c = sub.add_parser("calibrate")
    c.add_argument("--out", required=True)
    c.add_argument("--fast", action="store_true")
    d = sub.add_parser("drivers")
    d.add_argument("--formulas", required=True)
    d.add_argument("--panel", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--windows")
    d.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "drivers":
        return drivers(a)
    from .calibration import calibration_context, run_calibration

    s = run_calibration(calibration_context(fast=a.fast), a.out)
    fails = s.pop("failures")
    print(json.dumps(s, indent=1))
    print(f"{len(fails)} failures")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
