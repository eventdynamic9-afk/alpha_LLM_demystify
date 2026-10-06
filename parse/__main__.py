"""Claim-parser CLI.

    python -m parse run --rationales runs/main/rationales.jsonl --out runs/main/claims.jsonl \
        [--models configs/models.yaml] [--parser-id parser-e] [--rules-only] [--second-parser judge-f] \
        [--add-rule-claims] [--skip-novel]
    python -m parse validate --pred runs/main/claims.jsonl --gold annotation/gold_claims.jsonl \
        (--sheet-a sheet_A.csv --sheet-b sheet_B.csv | --alpha 0.71)

Parser responses are cached in ``parser_cache.sqlite`` next to the output (never regenerated silently,
§4.3).  Novel-pool rationales are only sent to parsers allowed to see the Novel pool (local, or terms
that exclude training on inputs: §4.1, §7.2); otherwise the run refuses unless ``--skip-novel``.
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
    from narrate.logger import CallLogger, ResponseCache
    from narrate.relay import PendingResponse
    from narrate.roster import novel_pool_allowed

    from .ensemble import ensemble, merge_parser_and_rules
    from .llm_parser import llm_parse
    from .normalize import normalize_claim
    from .rules import extract_claims

    rationales = _read(a.rationales)
    client = client2 = None
    blocked: list[str] = []
    if not a.rules_only:
        from configs import models
        from narrate.clients import get_client

        cfg = models(a.models)
        ms = {m["id"]: m for m in cfg["models"]}
        pid = a.parser_id or next(m["id"] for m in cfg["models"] if "parser" in m.get("roles", []))
        client = get_client(ms[pid])
        if a.second_parser:
            client2 = get_client(ms[a.second_parser])
        blocked = [i for i in [pid, a.second_parser] if i and not novel_pool_allowed(ms[i])]
        n_novel = sum(1 for r in rationales if r.get("pool") == "N" and r.get("status", "ok") == "ok")
        if blocked and n_novel and not a.skip_novel:
            print(f"refusing: {n_novel} Novel-pool rationales would be sent to parser(s) {blocked} whose endpoint "
                  "may train on inputs (§4.1, §7.2); use a local / no-training parser or pass --skip-novel",
                  file=sys.stderr)
            return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger = CallLogger(out.with_name("parser_calls.jsonl"))
    cache = None if a.rules_only else ResponseCache(out.with_name("parser_cache.sqlite"))
    n = pending = 0
    adjud, skipped = [], []
    with open(out, "w", encoding="utf-8") as fh:
        for r in rationales:
            rid, text = r["rationale_id"], r.get("text", "")
            if r.get("status", "ok") != "ok":
                continue
            if a.rules_only:
                claims = [normalize_claim({**c, "parser": "rules"}) for c in extract_claims(text, rid)]
            else:
                if blocked and r.get("pool") == "N":         # §7.2 item 4: never sent; logged, not parsed
                    skipped.append({"rationale_id": rid, "formula_id": r.get("formula_id"), "pool": "N",
                                    "reason": f"parser(s) {blocked} may train on inputs"})
                    continue
                try:
                    res = llm_parse(client, rid, text, log=logger.log, cache=cache)
                    res2 = llm_parse(client2, rid, text, log=logger.log, cache=cache) if client2 is not None else None
                except PendingResponse:              # relay request written; re-run after it is answered
                    pending += 1
                    continue
                if client2 is not None:
                    e = ensemble(res["claims"], res2["claims"], text, rid)
                    claims = e["accepted"]
                    adjud += e["adjudicate"]
                else:
                    claims = merge_parser_and_rules(res["claims"], text, rid, add_rule_claims=a.add_rule_claims)
            for c in claims:
                c.update({"rationale_id": rid, "formula_id": r.get("formula_id")})
                fh.write(json.dumps(c) + "\n")
                n += 1
    if adjud:
        with open(out.with_name("claims_to_adjudicate.jsonl"), "w", encoding="utf-8") as fh:
            for c in adjud:
                fh.write(json.dumps(c) + "\n")
    if skipped:
        with open(out.with_name("parser_skipped_novel.jsonl"), "w", encoding="utf-8") as fh:
            for x in skipped:
                fh.write(json.dumps(x) + "\n")
    print(f"wrote {n} claims from {len(rationales)} rationales to {out}"
          + (f"; {pending} rationales pending relay answers" if pending else "")
          + (f"; {len(skipped)} Novel-pool rationales NOT parsed (data-use terms)" if skipped else ""))
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
    agreement, alpha = None, a.alpha
    if a.sheet_a and a.sheet_b:                      # §9.3: agreement computed from the two annotators' sheets
        from annotation.tools import agreement_report, read_sheet

        agreement = agreement_report(read_sheet(a.sheet_a), read_sheet(a.sheet_b))
        alpha = agreement["krippendorff_alpha"]
    g = go_no_go(m, alpha)
    print(json.dumps({"metrics": m, "agreement": agreement, "decision": g}, indent=1, default=float))
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
    r.add_argument("--add-rule-claims", action="store_true",
                   help="also add rule-layer IDENTITY/LOOKBACK/DEPENDS_ON claims (beyond the first-run E2 + E1 slots)")
    r.add_argument("--skip-novel", action="store_true",
                   help="skip (and log) Novel-pool rationales when a parser may train on inputs instead of refusing")
    v = sub.add_parser("validate")
    v.add_argument("--pred", required=True)
    v.add_argument("--gold", required=True)
    v.add_argument("--sheet-a", help="annotator A sheet (CSV); with --sheet-b computes Krippendorff alpha")
    v.add_argument("--sheet-b")
    v.add_argument("--alpha", type=float, help="precomputed Krippendorff alpha (used when no sheets are given)")
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else validate(a)


if __name__ == "__main__":
    sys.exit(main())
