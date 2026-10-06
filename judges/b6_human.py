"""B6 human experts (§13): 100 rationales, claim level, first without and then with the evidence bundle
(static report, nudge histogram, correlation statistics), recording verdicts and time per item.

``export_packets`` writes two CSVs (phase 1 without evidence, phase 2 with evidence); ``import_verdicts``
reads the completed sheets back into judge records comparable with B1-B5.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path


def export_packets(rationales: list[dict], claims: list[dict], formulas: dict[str, dict], verdicts: dict[str, dict],
                   out_dir: str | Path, n: int = 100, seed: int = 0) -> dict:
    rng = random.Random(seed)
    by_r: dict[str, list[dict]] = {}
    for c in claims:
        by_r.setdefault(c["rationale_id"], []).append(c)
    pool = [r for r in rationales if r["rationale_id"] in by_r]
    rng.shuffle(pool)
    pick = pool[:n]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols1 = ["packet_id", "rationale_id", "claim_id", "formula", "rationale_text", "claim_text", "predicate", "args",
             "verdict(TRUE/FALSE/CANT_TELL)", "minutes"]
    cols2 = cols1[:-2] + ["evidence_bundle"] + cols1[-2:]
    rows1, rows2 = [], []
    for k, r in enumerate(pick):
        f = formulas[r["formula_id"]]
        for c in by_r[r["rationale_id"]]:
            base = {"packet_id": f"H{k:03d}", "rationale_id": r["rationale_id"], "claim_id": c["claim_id"],
                    "formula": f["presented"], "rationale_text": r["text"], "claim_text": c.get("text", ""),
                    "predicate": c["predicate"], "args": json.dumps(c.get("args", {})),
                    "verdict(TRUE/FALSE/CANT_TELL)": "", "minutes": ""}
            rows1.append(base)
            ev = verdicts.get(c["claim_id"], {})
            rows2.append({**base, "evidence_bundle": json.dumps({"method": ev.get("method"), "evidence": ev.get("evidence")},
                                                                default=str)[:4000]})
    for name, cols, rows in (("phase1_without_evidence.csv", cols1, rows1), ("phase2_with_evidence.csv", cols2, rows2)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return {"rationales": len(pick), "claims": len(rows1)}


def import_verdicts(path: str | Path, annotator: str, phase: str) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = (row.get("verdict(TRUE/FALSE/CANT_TELL)") or "").strip().upper()
            if v not in ("TRUE", "FALSE", "CANT_TELL"):
                continue
            mins = row.get("minutes") or ""
            out.append({"judge": "B6", "model": f"human:{annotator}:{phase}", "claim_id": row["claim_id"], "verdict": v,
                        "minutes": float(mins) if mins.replace(".", "", 1).isdigit() else None})
    return out
