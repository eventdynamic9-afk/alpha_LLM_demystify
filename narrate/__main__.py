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
from pathlib import Path


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
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--pilot-n", type=int)
    p.add_argument("--k", type=int)
    r = sub.add_parser("run")
    r.add_argument("--formulas", required=True)
    r.add_argument("--panel", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--models", default="models.yaml")
    r.add_argument("--limit", type=int)
    r.add_argument("--k", type=int)
    r.add_argument("--reference-t0", action="store_true")
    r.add_argument("--pilot", action="store_true", help="§17.1 pilot plan instead of the main plan")
    r.add_argument("--pilot-n", type=int)
    r.add_argument("--windows")
    r.add_argument("--fast", action="store_true")
    rl = sub.add_parser("relay", help="list or summarise pending relay requests")
    rl.add_argument("action", choices=["pending", "stats", "import-journal"])
    rl.add_argument("--dir", required=True)
    rl.add_argument("--journal", help="workflow journal.jsonl: replies returned as text instead of written")
    rl.add_argument("--out", help="write the pending work list as JSON")
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
        if a.pilot:
            from .plan import plan_pilot_cells

            cells = plan_pilot_cells(recs, with_role(cfg, "narrator"), a.pilot_n, a.k, novel_allowed=novel_pool_allowed)
        else:
            cells = plan_cells(recs, with_role(cfg, "narrator"), k=a.k, reference_t0=a.reference_t0,
                               novel_allowed=novel_pool_allowed)
        if a.limit:
            cells = cells[: a.limit]
        runner = NarrationRunner(cfg, {x["formula_id"]: x for x in recs}, a.run_dir, ctx, ctx.panel.market)
        res = runner.run(cells)
        print(f"narrated {len(res)} cells into {a.run_dir}/rationales.jsonl"
              + (f"; {runner.pending} cells pending relay answers" if runner.pending else ""))
        return 0
    if a.cmd == "relay":
        from collections import Counter

        from .relay import Relay

        rel = Relay(a.dir)
        if a.action == "import-journal":
            from .relay import import_journal

            print(json.dumps(import_journal(rel, a.journal), indent=1))
            return 0
        pend = rel.pending()
        if a.action == "stats" or not a.out:
            n_resp = len(list(rel.resp.glob("*.txt"))) if rel.resp.exists() else 0
            print(json.dumps({"pending": len(pend), "answered": n_resp,
                              "pending_by_model": dict(Counter(p["model"] for p in pend))}, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps([{"key": p["key"], "tier": p.get("agent_tier"), "request": p["request"],
                                                "response": p["response"]} for p in pend], indent=0))
            print(f"wrote {len(pend)} pending requests to {a.out}")
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
