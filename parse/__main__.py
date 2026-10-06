"""Claim-parser CLI.

    python -m parse run --rationales runs/main/rationales.jsonl --out runs/main/claims.jsonl \
        [--models configs/models.yaml] [--parser-id parser-e] [--rules-only] [--second-parser judge-f]
    python -m parse validate --pred runs/main/claims.jsonl --gold annotation/gold_claims.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(a) -> int:
    from narrate.logger import CallLogger

    from .ensemble import ensemble, merge_parser_and_rules
    from .llm_parser import llm_parse
    from .normalize import normalize_claim
    from .rules import extract_claims

    rationales = _read(a.rationales)
    client = client2 = None
    if not a.rules_only:
        from configs import models
        from narrate.clients import get_client

        cfg = models(a.models)
        ms = {m["id"]: m for m in cfg["models"]}
        pid = a.parser_id or next(m["id"] for m in cfg["models"] if "parser" in m.get("roles", []))
        client = get_client(ms[pid])
        if a.second_parser:
            client2 = get_client(ms[a.second_parser])
    logger = CallLogger(Path(a.out).with_name("parser_calls.jsonl"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    adjud = []
    with open(out, "w", encoding="utf-8") as fh:
        for r in rationales:
            rid, text = r["rationale_id"], r.get("text", "")
            if r.get("status", "ok") != "ok":
                continue
            if a.rules_only:
                claims = [normalize_claim({**c, "parser": "rules"}) for c in extract_claims(text, rid)]
            else:
                res = llm_parse(client, rid, text, log=logger.log)
                if client2 is not None:
                    res2 = llm_parse(client2, rid, text, log=logger.log)
                    e = ensemble(res["claims"], res2["claims"], text, rid)
                    claims = e["accepted"]
                    adjud += e["adjudicate"]
                else:
                    claims = merge_parser_and_rules(res["claims"], text, rid)
            for c in claims:
                c.update({"rationale_id": rid, "formula_id": r.get("formula_id")})
                fh.write(json.dumps(c) + "\n")
                n += 1
    if adjud:
        with open(out.with_name("claims_to_adjudicate.jsonl"), "w", encoding="utf-8") as fh:
            for c in adjud:
                fh.write(json.dumps(c) + "\n")
    print(f"wrote {n} claims from {len(rationales)} rationales to {out}")
    return 0


def validate(a) -> int:
    from .validate import extraction_metrics, go_no_go

    pred, gold = defaultdict(list), defaultdict(list)
    for c in _read(a.pred):
        pred[c["rationale_id"]].append(c)
    for c in _read(a.gold):
        gold[c["rationale_id"]].append(c)
    pred = {k: v for k, v in pred.items() if k in gold}
    m = extraction_metrics(pred, gold)
    g = go_no_go(m, a.alpha)
    print(json.dumps({"metrics": m, "decision": g}, indent=1, default=float))
    return 0 if g["go"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m parse")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rationales", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--models", default="models.yaml")
    r.add_argument("--parser-id")
    r.add_argument("--second-parser")
    r.add_argument("--rules-only", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--pred", required=True)
    v.add_argument("--gold", required=True)
    v.add_argument("--alpha", type=float)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else validate(a)


if __name__ == "__main__":
    sys.exit(main())
