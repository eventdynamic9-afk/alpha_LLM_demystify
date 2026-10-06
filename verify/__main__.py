"""Verifier CLI.

    python -m verify run --formulas runs/x/formulas.jsonl --claims runs/x/claims.jsonl \
        --panel data/processed/cn_csi500.npz --out runs/x/verdicts.jsonl [--fast] [--trial-logs runs/x/trials.jsonl]
    python -m verify calibrate --out runs/calibration [--fast] [--panel data/processed/us_sp500_plotly.npz]
    python -m verify drivers --formulas runs/x/formulas.jsonl --panel ... --out runs/x/drivers.jsonl
    python -m verify exposures --formulas runs/x/formulas.jsonl --panel ... --out runs/x/exposures.jsonl
    python -m verify perf-family --formulas runs/x/formulas.jsonl --panel ... --out runs/x/perf_family.jsonl

Common context options: --windows (JSON), --roster (narrator roster for H_post, default configs/models.yaml),
--factors (US Ken French daily table; default data/processed/us_factor_returns_daily.csv when present),
--factors-monthly (CN CH-3/CH-4 monthly table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.panel import Panel
from dsl import parse


def _ctx(panel_path: str, fast: bool, windows: str | None = None, factors: str | None = None,
         factors_monthly: str | None = None, roster: str | None = None, warn: bool = False):
    from .context import VerificationContext
    from .factors import default_us_factor_path, load_factor_table

    from dsl.fields import set_price_adjustment

    p = Panel.load(panel_path)
    set_price_adjustment(p.meta.get("price_adjustment"))
    w = json.loads(windows) if windows else None
    if w is None and p.meta.get("windows"):          # windows declared by the panel builder (deviation log)
        w = {k: tuple(v) for k, v in p.meta["windows"].items()}
    if w is None and p.meta.get("source") == "synthetic":
        cut_v, cut = str(p.dates[int(p.T * 0.5)]), str(p.dates[int(p.T * 0.6)])
        w = {"train": (str(p.dates[0]), cut_v), "valid": (cut_v, cut), "test": (cut, str(p.dates[-1]))}
    ext = ext_m = None
    if factors and not Path(factors).exists():
        raise FileNotFoundError(f"factor table {factors} not found")
    if p.market == "US":
        path = Path(factors) if factors else default_us_factor_path()
        if path.exists():
            ext = load_factor_table(path)
        elif warn:
            print(f"WARNING: US factor table {path} not found (python -m data refs): EXPOSED/NOVEL fall back to "
                  "self-built proxies and are flagged as a deviation", file=sys.stderr)
    if factors_monthly:
        ext_m = load_factor_table(factors_monthly)
    elif p.market == "CN" and warn:
        print("WARNING: no CH-3 monthly table (--factors-monthly): CN EXPOSED uses the self-built daily factors only",
              file=sys.stderr)
    ctx = VerificationContext(p, windows=w or {}, fast=fast, external_factors=ext, external_factors_monthly=ext_m,
                              roster=roster)
    if warn and ctx.post_status.get("status") == "not_run":
        print(f"WARNING: H_post not evaluated: {ctx.post_status.get('reason')}", file=sys.stderr)
    return ctx


def _ctx_args(a):
    return _ctx(a.panel, a.fast, a.windows, getattr(a, "factors", None), getattr(a, "factors_monthly", None),
                getattr(a, "roster", None), warn=True)


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


_TRIAL_PREDICATES = {"PERF", "BETTER_THAN"}


def run(args) -> int:
    from .dispatcher import verify_claim
    from .performance import read_trial_logs, record_trials

    ctx = _ctx_args(args)
    formulas = {f["formula_id"]: f for f in _read_jsonl(args.formulas)}
    logs = read_trial_logs(args.trial_logs) if getattr(args, "trial_logs", None) else None
    trials_of: dict = {}                         # formula_id -> recorded search (DSR / "best of", §10.4)
    fam_cache: dict = {}                         # family statistics shared by records of one mining run
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
            tr = None
            if str(c.get("predicate", "")).upper() in _TRIAL_PREDICATES:
                if fid not in trials_of:
                    trials_of[fid] = record_trials(rec, ctx, logs, fam_cache)
                tr = trials_of[fid]
            v = verify_claim(c, node, ctx, trials=tr)
            row = {"claim_id": c["claim_id"], "rationale_id": c.get("rationale_id"), "formula_id": fid,
                   "market": ctx.panel.market, **v.to_dict()}
            if "post_verdict" in v.evidence:      # H_post decided separately (§10.4)
                row["post_verdict"] = v.evidence["post_verdict"]
                row["post_exploratory"] = v.evidence["post_exploratory"]
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

    ctx = _ctx_args(args)
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

    ctx = _ctx_args(args)
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


def perf_family(args) -> int:
    """§10.4 family-level statistics: for every mining run (P1-mined refinement log, P3a GP run) with a
    candidate log, PBO via CSCV (S = 16) on the selection window, with the 15 bp per-trade sensitivity, and
    RC / SPA of the best candidate vs zero.  Families whose candidates were counted but not logged are
    written with status "not_run"."""
    from .performance import SELECTION_WINDOW, family_report, read_trial_logs, trial_family_key, trial_formulas

    ctx = _ctx_args(args)
    logs = read_trial_logs(args.trial_logs) if args.trial_logs else None
    fams: dict = {}
    for rec in _read_jsonl(args.formulas):
        if rec.get("pool") not in SELECTION_WINDOW or (rec.get("pool") == "P1" and rec.get("stratum") != "mined"):
            continue
        key = trial_family_key(rec)
        if key is None:
            continue
        f = fams.setdefault(key, {"family": key, "pool": rec["pool"], "formula_ids": [], "formulas": [],
                                  "n_trials_recorded": 0})
        f["formula_ids"].append(rec["formula_id"])
        f["formulas"] += [x for x in trial_formulas(rec, logs) if x not in f["formulas"]]
        f["n_trials_recorded"] = max(f["n_trials_recorded"], int(rec.get("trials") or 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_run = 0
    with open(out, "w", encoding="utf-8") as fh:
        for key, f in fams.items():
            forms = f.pop("formulas")
            if len(forms) < 2:
                rep = {"status": "not_run", "n_logged": len(forms),
                       "reason": "candidate log not available (only the trial count was recorded)"}
                print(f"WARNING: family {key}: {rep['reason']}", file=sys.stderr)
            else:
                w = args.window or SELECTION_WINDOW[f["pool"]]
                rep = family_report(forms, ctx, w if ctx.has_window(w) else "train")
                n_run += rep.get("status") == "ok"
            fh.write(json.dumps({**f, **rep}, default=float) + "\n")
    print(f"wrote {len(fams)} families ({n_run} with PBO) to {out}")
    return 0


def _common(p, panel_required: bool = True) -> None:
    p.add_argument("--panel", required=panel_required)
    p.add_argument("--windows", help='JSON, e.g. {"train": ["2015-01-01", "2019-12-31"], "test": [...]}')
    p.add_argument("--fast", action="store_true")
    p.add_argument("--roster", help="narrator roster for H_post (default configs/models.yaml)")
    p.add_argument("--factors", help="US Ken French daily factor CSV (decimal); default "
                                     "data/processed/us_factor_returns_daily.csv when present")
    p.add_argument("--factors-monthly", dest="factors_monthly", help="CN CH-3/CH-4 monthly factor CSV")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--formulas", required=True)
    r.add_argument("--claims", required=True)
    r.add_argument("--rationales")
    r.add_argument("--out", required=True)
    r.add_argument("--trial-logs", dest="trial_logs", help="JSONL {family, formula} candidate logs (P3a GP runs)")
    _common(r)
    c = sub.add_parser("calibrate")
    c.add_argument("--out", required=True)
    c.add_argument("--panel", help="run the planted set on a real panel (truth is by construction either way)")
    c.add_argument("--windows")
    c.add_argument("--fast", action="store_true")
    d = sub.add_parser("drivers")
    d.add_argument("--formulas", required=True)
    d.add_argument("--out", required=True)
    _common(d)
    e = sub.add_parser("exposures")
    e.add_argument("--formulas", required=True)
    e.add_argument("--out", required=True)
    _common(e)
    pf = sub.add_parser("perf-family")
    pf.add_argument("--formulas", required=True)
    pf.add_argument("--out", required=True)
    pf.add_argument("--trial-logs", dest="trial_logs", help="JSONL {family, formula} candidate logs (P3a GP runs)")
    pf.add_argument("--window", help="override the selection window (default: P1 valid, P3a train)")
    _common(pf)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "drivers":
        return drivers(a)
    if a.cmd == "exposures":
        return exposures(a)
    if a.cmd == "perf-family":
        return perf_family(a)
    from .calibration import calibration_context, run_calibration

    ctx = _ctx(a.panel, a.fast, a.windows) if a.panel else calibration_context(fast=a.fast)
    s = run_calibration(ctx, a.out)
    fails = s.pop("failures")
    print(json.dumps(s, indent=1))
    print(f"{len(fails)} failures")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
