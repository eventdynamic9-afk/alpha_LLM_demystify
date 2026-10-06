"""Annotation workflow (§9.3, §10.7): stratified sampling of rationales for gold extraction, annotator
sheets, adjudication, agreement report, and the 100-verdict human spot-check with evidence bundles."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .agreement import cohen_kappa, gwet_ac1, interpret_alpha, krippendorff_alpha_nominal

SHEET_COLS = ["rationale_id", "claim_no", "span_start", "span_end", "text", "type", "predicate", "args_json", "scope",
              "hedge", "polarity", "horizon", "ambiguous", "note"]


def stratified_rationale_sample(rationales: list[dict], n: int = 200, seed: int = 0, exclude: set | None = None) -> list[dict]:
    """>= 200 rationales stratified by model x pool/protocol x access (refresh: n=50 with ``exclude``)."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in rationales:
        if exclude and r["rationale_id"] in exclude:
            continue
        if r.get("status", "ok") != "ok":
            continue
        cells[(r["model"], r.get("pool"), r["access"])].append(r)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells, key=str)
    out = []
    i = 0
    while len(out) < n and any(cells[k] for k in keys):
        k = keys[i % len(keys)]
        if cells[k]:
            out.append(cells[k].pop())
        i += 1
    return out


def write_annotation_sheets(sample: list[dict], out_dir: str | Path, annotators=("A", "B")) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rationales_to_annotate.jsonl", "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps({"rationale_id": r["rationale_id"], "text": r["text"]}) + "\n")
    for a in annotators:
        with open(out / f"sheet_{a}.csv", "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=SHEET_COLS).writeheader()


def read_sheet(path: str | Path) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append({"claim_id": f"{row['rationale_id']}-g{row['claim_no']}", "rationale_id": row["rationale_id"],
                        "span": [int(row["span_start"]), int(row["span_end"])], "text": row["text"], "type": row["type"],
                        "predicate": row["predicate"], "args": json.loads(row["args_json"] or "{}"),
                        "scope": row["scope"] or None, "hedge": row["hedge"] or "absolute",
                        "polarity": row["polarity"] or "affirm", "horizon": row["horizon"] or None,
                        "ambiguous": str(row["ambiguous"]).lower() in ("1", "true", "yes")})
    return out


def agreement_report(sheet_a: list[dict], sheet_b: list[dict]) -> dict:
    """Agreement on the predicate assigned to each overlapping span (unit = claim span of annotator A)."""
    from parse.ensemble import spans_overlap

    by_b = defaultdict(list)
    for c in sheet_b:
        by_b[c["rationale_id"]].append(c)
    a_lab, b_lab = [], []
    for c in sheet_a:
        match = next((d for d in by_b[c["rationale_id"]] if spans_overlap(c["span"], d["span"])), None)
        a_lab.append(c["predicate"])
        b_lab.append(match["predicate"] if match else "NONE")
    alpha = krippendorff_alpha_nominal([a_lab, b_lab])
    return {"n_units": len(a_lab), "cohen_kappa": cohen_kappa(a_lab, b_lab), "krippendorff_alpha": alpha,
            "alpha_interpretation": interpret_alpha(alpha), "gwet_ac1": gwet_ac1([a_lab, b_lab])}


def adjudicate(sheet_a: list[dict], sheet_b: list[dict], sheet_c: list[dict] | None = None) -> list[dict]:
    """Gold = claims both annotators extracted with the same predicate; disagreements resolved by the third
    annotator's sheet when supplied (otherwise listed for adjudication with ``needs_adjudication``)."""
    from parse.ensemble import spans_overlap
    from parse.normalize import normalize_claim, predicate_key

    gold, used = [], set()
    for c in sheet_a:
        cn = normalize_claim(c)
        m = next((i for i, d in enumerate(sheet_b) if i not in used and d["rationale_id"] == c["rationale_id"]
                  and spans_overlap(c["span"], d["span"]) and predicate_key(normalize_claim(d)) == predicate_key(cn)), None)
        if m is not None:
            used.add(m)
            gold.append({**cn, "adjudication": "agreed"})
        elif sheet_c is not None:
            if any(d["rationale_id"] == c["rationale_id"] and spans_overlap(c["span"], d["span"])
                   and predicate_key(normalize_claim(d)) == predicate_key(cn) for d in sheet_c):
                gold.append({**cn, "adjudication": "third_annotator"})
        else:
            gold.append({**cn, "needs_adjudication": True})
    return gold


def export_verdict_spotcheck(verdicts: list[dict], claims: dict[str, dict], out_path: str | Path, n: int = 100,
                             seed: int = 0) -> int:
    """§10.7 item 3: 100 random verdicts with their evidence bundle for human review (disagreements are
    investigated, not overruled)."""
    rng = random.Random(seed)
    pool = [v for v in verdicts if v["verdict"] in ("SUPPORTED", "REFUTED", "UNRESOLVED")]
    rng.shuffle(pool)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim_id", "claim_text", "predicate", "args", "verdict", "method",
                                           "evidence", "reviewer_agrees(Y/N)", "comment"])
        w.writeheader()
        for v in pool[:n]:
            c = claims.get(v["claim_id"], {})
            w.writerow({"claim_id": v["claim_id"], "claim_text": c.get("text", ""), "predicate": c.get("predicate"),
                        "args": json.dumps(c.get("args", {})), "verdict": v["verdict"], "method": v.get("method"),
                        "evidence": json.dumps(v.get("evidence", {}), default=str)[:6000],
                        "reviewer_agrees(Y/N)": "", "comment": ""})
    return min(n, len(pool))
