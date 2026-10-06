"""Judge-baseline CLI (Arm C re-scores existing rationales; B2-B4 judge calls on a stratified sample).

    python -m judges run --run-dir runs/main --panel data/processed/cn_csi500.npz \
        [--models configs/models.yaml] [--judge-id judge-f] [--sample 0.2] [--judges B1,B2,B3,B4,B5]
    python -m judges human-export --run-dir runs/main --out annotation/b6 [--n 100]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratified_sample(rationales: list[dict], frac: float, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in rationales:
        cells[(r["model"], r["pool"], r["access"])].append(r)
    out = []
    for k in sorted(cells):
        v = cells[k]
        rng.shuffle(v)
        out += v[: max(1, int(round(frac * len(v))))]
    return out


def run(a) -> int:
    from configs import models
    from dsl import parse
    from narrate.clients import get_client
    from narrate.relay import PendingResponse
    from verify.__main__ import _ctx

    from .b5_nli import get_nli, judge_b5
    from .llm_judges import judge_b1, judge_b2, judge_b3, judge_b4

    rd = Path(a.run_dir)
    rationales = [r for r in _read(rd / "rationales.jsonl") if r.get("status", "ok") == "ok"]
    formulas = {f["formula_id"]: f for f in _read(rd / "formulas.jsonl")}
    claims = _read(rd / "claims.jsonl")
    by_r = defaultdict(list)
    for c in claims:
        by_r[c["rationale_id"]].append(c)
    cfg = models(a.models)
    jid = a.judge_id or next(m["id"] for m in cfg["models"] if "judge" in m.get("roles", []))
    client = get_client(next(m for m in cfg["models"] if m["id"] == jid))
    ctx = _ctx(a.panel, a.fast, a.windows)
    which = set(a.judges.split(","))
    nli = get_nli(a.transformers_nli)
    out = rd / "judgements.jsonl"
    sample_ids = {r["rationale_id"] for r in stratified_sample(rationales, a.sample, a.seed)}
    n = pending = 0
    with open(out, "w", encoding="utf-8") as fh:
        for r in rationales:
            rec = formulas[r["formula_id"]]
            rid = r["rationale_id"]
            jobs = []
            if "B1" in which and (a.b1_scope == "all" or rid in sample_ids):   # holistic judge
                jobs.append(lambda: judge_b1(client, rec, r["text"], rec.get("hypothesis"), ctx.panel.market))
            if rid in sample_ids:
                if "B4" in which:
                    jobs.append(lambda: judge_b4(client, rec, r["text"], ctx, ctx.panel.market))
                for c in by_r.get(rid, []):
                    if "B2" in which:
                        jobs.append(lambda c=c: judge_b2(client, rec, c, ctx.panel.market))
                    if "B3" in which:
                        jobs.append(lambda c=c: judge_b3(client, rec, c, ctx, ctx.panel.market))
            if "B5" in which:
                for c in by_r.get(rid, []):
                    jobs.append(lambda c=c: judge_b5(nli, parse(rec["dsl"]), c))
            for job in jobs:
                try:
                    j = job()
                except PendingResponse:              # relay request written; re-run after it is answered
                    pending += 1
                    continue
                fh.write(json.dumps({**j, "rationale_id": rid}, default=str) + "\n")
                n += 1
    print(f"wrote {n} judgements to {out}" + (f"; {pending} pending relay answers" if pending else ""))
    return 0


def human_export(a) -> int:
    from .b6_human import export_packets

    rd = Path(a.run_dir)
    verdicts = {v["claim_id"]: v for v in _read(rd / "verdicts.jsonl")}
    res = export_packets(_read(rd / "rationales.jsonl"), _read(rd / "claims.jsonl"),
                         {f["formula_id"]: f for f in _read(rd / "formulas.jsonl")}, verdicts, a.out, a.n)
    print(res)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m judges")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--panel", required=True)
    r.add_argument("--models", default="models.yaml")
    r.add_argument("--judge-id")
    r.add_argument("--sample", type=float, default=0.2)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--judges", default="B1,B2,B3,B4,B5")
    r.add_argument("--b1-scope", choices=["all", "sample"], default="all",
                   help="B1 on every rationale (default) or only on the stratified sample (§17.1 phase 3)")
    r.add_argument("--transformers-nli", action="store_true")
    r.add_argument("--windows")
    r.add_argument("--fast", action="store_true")
    h = sub.add_parser("human-export")
    h.add_argument("--run-dir", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--n", type=int, default=100)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else human_export(a)


if __name__ == "__main__":
    sys.exit(main())
