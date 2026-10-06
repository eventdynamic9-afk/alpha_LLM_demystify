"""Verifier CLI.

    python -m verify run --formulas runs/x/formulas.jsonl --claims runs/x/claims.jsonl \
        --panel data/processed/cn_csi500.npz --out runs/x/verdicts.jsonl [--fast]
    python -m verify calibrate --out runs/calibration [--fast] [--panel data/processed/us_sp500_plotly.npz]
    python -m verify drivers --formulas runs/x/formulas.jsonl --panel ... --out runs/x/drivers.jsonl
    python -m verify exposures --formulas runs/x/formulas.jsonl --panel ... --out runs/x/exposures.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.panel import Panel
from dsl import parse


def _ctx(panel_path: str, fast: bool, windows: str | None = None):
    from .context import VerificationContext

    from dsl.fields import set_price_adjustment

    p = Panel.load(panel_path)
    set_price_adjustment(p.meta.get("price_adjustment"))
    w = json.loads(windows) if windows else None
    if w is None and p.meta.get("windows"):          # windows declared by the panel builder (deviation log)
        w = {k: tuple(v) for k, v in p.meta["windows"].items()}
    if w is None and p.meta.get("source") == "synthetic":
        cut_v, cut = str(p.dates[int(p.T * 0.5)]), str(p.dates[int(p.T * 0.6)])
        w = {"train": (str(p.dates[0]), cut_v), "valid": (cut_v, cut), "test": (cut, str(p.dates[-1]))}
    return VerificationContext(p, windows=w or {}, fast=fast)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _targeted(rec: dict) -> set:
    from analysis.metrics import TARGETED

    t = (rec.get("perturbation") or {}).get("type", "").replace("sa_", "")
    return TARGETED.get(t, set())


def _base_records(formulas: dict) -> dict:
    """base_id -> parsed base formula (the K record, else the library entry)."""
    from pools.library import by_short_id

    out = {}
    for f in formulas.values():
        if f.get("pool") == "K" and f.get("base_id"):
            out[f["base_id"]] = parse(f["dsl"])
    for f in formulas.values():
        b = f.get("base_id")
        if b and b not in out:
            lf = by_short_id(b)
            if lf is not None:
                out[b] = lf.node
    return out


def run(args) -> int:
    from .dispatcher import verify_claim

    ctx = _ctx(args.panel, args.fast, args.windows)
    formulas = {f["formula_id"]: f for f in _read_jsonl(args.formulas)}
    rationale_formula = {}
    if args.rationales:
        rationale_formula = {r["rationale_id"]: r["formula_id"] for r in _read_jsonl(args.rationales)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_of = _base_records(formulas)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for c in _read_jsonl(args.claims):
            fid = c.get("formula_id") or rationale_formula.get(c.get("rationale_id"))
            rec = formulas.get(fid)
            node = parse(rec["dsl"]) if rec else None
            v = verify_claim(c, node, ctx)
            row = {"claim_id": c["claim_id"], "rationale_id": c.get("rationale_id"), "formula_id": fid,
                   "market": ctx.panel.market, **v.to_dict()}
            # recall anchoring (§11): targeted-property claims of SA rationales are also checked on the base formula
            if rec and rec.get("pool") == "SA" and c["predicate"] in _targeted(rec):
                base = base_of.get(rec.get("base_id"))
                if base is not None:
                    row["base_verdict"] = verify_claim(c, base, ctx).verdict
            fh.write(json.dumps(row, default=float) + "\n")
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


def exposures(args) -> int:
    """Per formula: reference characteristics with rho_bar CI lower >= 0.30 (exposure coverage, §11) and
    the OOS mean RankIC (fidelity-performance link, exploratory)."""
    import numpy as np

    from .stats import bootstrap_mean_ci, daily_spearman, newey_west_mean

    ctx = _ctx(args.panel, args.fast, args.windows)
    floor = ctx.thr["behavioral"]["resemblance_floor"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_tr = ctx.rows("train")
    with open(out, "w", encoding="utf-8") as fh:
        for rec in _read_jsonl(args.formulas):
            f = ctx.signal(parse(rec["dsl"]))
            ex = []
            for name in ctx.references.characteristic_names():
                r = daily_spearman(f, ctx.references.signal(name), rows_tr)
                if np.isfinite(r).sum() < 30:
                    continue
                m, lo, hi, _ = bootstrap_mean_ci(np.sign(np.nanmean(r)) * r, 0.95, ctx.n_boot(), ctx.seed)
                if lo >= floor:
                    ex.append((name, float(m)))
            ex.sort(key=lambda x: -x[1])
            oos = None
            if ctx.has_window("test"):
                ic = daily_spearman(f, ctx.fwd(1, "open_t+1"), ctx.rows("test"))
                oos = newey_west_mean(ic)[0]
            fh.write(json.dumps({"formula_id": rec["formula_id"], "exposures": [e[0] for e in ex],
                                 "exposure_rho": dict(ex), "oos_rank_ic": oos}, default=float) + "\n")
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
    c.add_argument("--panel", help="run the planted set on a real panel (truth is by construction either way)")
    c.add_argument("--windows")
    d = sub.add_parser("drivers")
    d.add_argument("--formulas", required=True)
    d.add_argument("--panel", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--windows")
    d.add_argument("--fast", action="store_true")
    e = sub.add_parser("exposures")
    e.add_argument("--formulas", required=True)
    e.add_argument("--panel", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--windows")
    e.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "drivers":
        return drivers(a)
    if a.cmd == "exposures":
        return exposures(a)
    from .calibration import calibration_context, run_calibration

    ctx = _ctx(a.panel, a.fast, a.windows) if a.panel else calibration_context(fast=a.fast)
    s = run_calibration(ctx, a.out)
    fails = s.pop("failures")
    print(json.dumps(s, indent=1))
    print(f"{len(fails)} failures")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
