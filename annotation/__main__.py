"""Annotation CLI.

    python -m annotation sample --run-dir runs/main --out annotation/batch1 [--n 200]
    python -m annotation refresh --run-dir runs/main --out annotation/batch2 --exclude annotation/batch1 [--n 50]
    python -m annotation agreement --a sheet_A.csv --b sheet_B.csv
    python -m annotation adjudicate --a sheet_A.csv --b sheet_B.csv [--c sheet_C.csv] --out gold_claims.jsonl
    python -m annotation spotcheck --run-dir runs/main --out annotation/spotcheck.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def main(argv=None) -> int:
    from .tools import (adjudicate, agreement_report, export_verdict_spotcheck, read_sheet, stratified_rationale_sample,
                        write_annotation_sheets)

    ap = argparse.ArgumentParser(prog="python -m annotation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "refresh"):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--n", type=int, default=200 if name == "sample" else 50)
        s.add_argument("--exclude")
        s.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("agreement")
    g.add_argument("--a", required=True)
    g.add_argument("--b", required=True)
    j = sub.add_parser("adjudicate")
    j.add_argument("--a", required=True)
    j.add_argument("--b", required=True)
    j.add_argument("--c")
    j.add_argument("--out", required=True)
    sp = sub.add_parser("spotcheck")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n", type=int, default=100)
    a = ap.parse_args(argv)

    if a.cmd in ("sample", "refresh"):
        excl = None
        if a.exclude:
            excl = {r["rationale_id"] for r in _read(Path(a.exclude) / "rationales_to_annotate.jsonl")}
        sample = stratified_rationale_sample(_read(Path(a.run_dir) / "rationales.jsonl"), a.n, a.seed, excl)
        write_annotation_sheets(sample, a.out)
        print(f"{len(sample)} rationales written to {a.out}")
    elif a.cmd == "agreement":
        print(json.dumps(agreement_report(read_sheet(a.a), read_sheet(a.b)), indent=1))
    elif a.cmd == "adjudicate":
        gold = adjudicate(read_sheet(a.a), read_sheet(a.b), read_sheet(a.c) if a.c else None)
        with open(a.out, "w", encoding="utf-8") as fh:
            for c in gold:
                fh.write(json.dumps(c) + "\n")
        print(f"{len(gold)} gold claims, {sum(1 for c in gold if c.get('needs_adjudication'))} need adjudication")
    elif a.cmd == "spotcheck":
        rd = Path(a.run_dir)
        claims = {c["claim_id"]: c for c in _read(rd / "claims.jsonl")}
        n = export_verdict_spotcheck(_read(rd / "verdicts.jsonl"), claims, a.out, a.n)
        print(f"{n} verdicts exported for spot-check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
