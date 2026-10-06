"""Executor CLI (§6.4): E1-vs-E2 agreement and the dynamic look-ahead test on a real panel.

    python -m executors agree --panel data/processed/us_sp500_plotly.npz --set library \
        [--start 2015-01-01 --end 2015-12-31] --out runs/executors/agreement_library.jsonl
    python -m executors agree --panel ... --set random --n 500 --out ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


def agree(a) -> int:
    from data.panel import Panel
    from dsl import parse, to_qlib
    from dsl.operators import warmup
    from dsl.random_trees import random_tree

    from . import E1Executor, E2Executor
    from .causality import truncation_test
    from .compare import agreement

    p = Panel.load(a.panel)
    if a.start or a.end:
        p = p.slice(a.start, a.end, warmup=a.warmup)
    if a.set == "library":
        from pools.library import library

        items = [(k, f.node) for k, f in library().items()]
    else:
        items = [(f"random_{i}", random_tree(random.Random(a.seed + i), max_depth=a.max_depth)) for i in range(a.n)]
    e1, e2 = E1Executor(), E2Executor()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_trunc_ok = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for fid, node in items:
            x1, x2 = e1.evaluate(node, p), e2.evaluate(node, p)
            ag = agreement(x1, x2)
            tr = truncation_test(node, p, e2, a.n_cut) if a.truncation else None
            rec = {"formula_id": fid, "qlib": to_qlib(node), "warmup": warmup(node), **ag.to_dict(),
                   "finite_share": float(np.isfinite(x2).mean())}
            if tr is not None:
                rec["truncation_ok"] = bool(tr.ok)
                n_trunc_ok += rec["truncation_ok"]
            n_ok += ag.ok
            fh.write(json.dumps(rec, default=float) + "\n")
    summary = {"panel": a.panel, "set": a.set, "n": len(items), "agree": n_ok, "T": p.T, "N": p.N,
               "seconds": round(time.time() - t0, 1)}
    if a.truncation:
        summary["truncation_ok"] = n_trunc_ok
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if n_ok == len(items) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m executors")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("agree")
    g.add_argument("--panel", required=True)
    g.add_argument("--set", choices=["library", "random"], default="library")
    g.add_argument("--n", type=int, default=500)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-depth", type=int, default=4)
    g.add_argument("--start")
    g.add_argument("--end")
    g.add_argument("--warmup", type=int, default=260)
    g.add_argument("--truncation", action="store_true", help="also run the dynamic look-ahead test (§6.3)")
    g.add_argument("--n-cut", type=int, default=5)
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return agree(a)


if __name__ == "__main__":
    sys.exit(main())
