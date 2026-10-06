"""Empirical training-cutoff probe (§4.1): ask dated questions and record the latest month the model
answers correctly.  The question bank is a YAML list the team fills on the run date with verifiable,
dated public facts (one or more per month):

    - month: "2025-06"
      question: "..."
      accept: ["keyword", "alternative keyword"]

The probe result is stored in the roster (``cutoff_probe``) next to the model card's stated cutoff and
is used to set the post-cutoff window and the Novel-pool timing.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml


def run_probe(client, bank_path: str | Path, temperature: float = 0.0) -> dict:
    bank = yaml.safe_load(Path(bank_path).read_text()) or []
    by_month = defaultdict(list)
    for item in bank:
        msgs = [{"role": "user", "content": item["question"] + " Answer briefly."}]
        r = client.complete(msgs, temperature=temperature, max_tokens=80, seed=0)
        ok = any(a.lower() in (r.text or "").lower() for a in item["accept"])
        by_month[str(item["month"])].append(ok)
    months = sorted(by_month)
    acc = {m: sum(v) / len(v) for m, v in by_month.items()}
    known = [m for m in months if acc[m] >= 0.5]
    return {"accuracy_by_month": acc, "last_known_month": known[-1] if known else None, "n_questions": len(bank)}
