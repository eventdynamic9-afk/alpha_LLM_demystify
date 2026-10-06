"""Narration CLI.

    python -m narrate check-roster [--models configs/models.yaml] [--pilot]
    python -m narrate plan --formulas runs/main/formulas.jsonl [--models ...] [--reference-t0]
    python -m narrate run --formulas runs/main/formulas.jsonl --panel data/processed/cn_csi500.npz \
        --run-dir runs/main [--models configs/models.yaml] [--limit N]
    python -m narrate probe --model-id X --bank probes.yaml
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    from configs import models as load_models

    ap = argparse.ArgumentParser(prog="python -m narrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-roster")
    c.add_argument("--models", default="models.yaml")
    c.add_argument("--pilot", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("--formulas", required=True)
    p.add_argument("--models", default="models.yaml")
    p.add_argument("--reference-t0", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--formulas", required=True)
    r.add_argument("--panel", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--models", default="models.yaml")
    r.add_argument("--limit", type=int)
    r.add_argument("--k", type=int)
    r.add_argument("--reference-t0", action="store_true")
    r.add_argument("--windows")
    r.add_argument("--fast", action="store_true")
    q = sub.add_parser("probe")
    q.add_argument("--models", default="models.yaml")
    q.add_argument("--model-id", required=True)
    q.add_argument("--bank", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "check-roster":
        from .roster import validate_roster

        rep = validate_roster(load_models(a.models), main_run=not a.pilot)
        print(json.dumps({"ok": rep.ok, "problems": rep.problems, **rep.summary}, indent=1))
        return 0 if rep.ok else 1
    if a.cmd == "plan":
        from pools.records import read_jsonl

        from .plan import plan_cells, plan_summary
        from .roster import novel_pool_allowed, with_role

        cfg = load_models(a.models)
        cells = plan_cells(read_jsonl(a.formulas), with_role(cfg, "narrator"), reference_t0=a.reference_t0,
                           novel_allowed=novel_pool_allowed)
        print(json.dumps(plan_summary(cells), indent=1))
        return 0
    if a.cmd == "run":
        from pools.records import read_jsonl
        from verify.__main__ import _ctx

        from .plan import plan_cells
        from .roster import novel_pool_allowed, with_role
        from .runner import NarrationRunner

        cfg = load_models(a.models)
        recs = read_jsonl(a.formulas)
        ctx = _ctx(a.panel, a.fast, a.windows)
        cells = plan_cells(recs, with_role(cfg, "narrator"), k=a.k, reference_t0=a.reference_t0,
                           novel_allowed=novel_pool_allowed)
        if a.limit:
            cells = cells[: a.limit]
        runner = NarrationRunner(cfg, {x["formula_id"]: x for x in recs}, a.run_dir, ctx, ctx.panel.market)
        res = runner.run(cells)
        print(f"narrated {len(res)} cells into {a.run_dir}/rationales.jsonl")
        return 0
    if a.cmd == "probe":
        from .clients import get_client
        from .cutoff_probe import run_probe

        cfg = load_models(a.models)
        m = next(m for m in cfg["models"] if m["id"] == a.model_id)
        print(json.dumps(run_probe(get_client(m), a.bank), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
