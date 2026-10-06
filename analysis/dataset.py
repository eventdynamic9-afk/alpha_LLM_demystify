"""Assemble the nested analysis tables (§3.1): model x formula x condition -> rationale -> claim -> verdict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DECIDABLE = ("SUPPORTED", "REFUTED")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sa_type(f: dict) -> str | None:
    pert = f.get("perturbation") or {}
    t = pert.get("type", "")
    if f.get("pool") == "SA" and t.startswith("sa_"):
        return t[3:]
    if f.get("pool") == "SP" and t.startswith("sp_"):
        return t[3:]
    return None


def formula_table(formulas: list[dict]) -> pd.DataFrame:
    rows = []
    for f in formulas:
        cx = f.get("complexity") or {}
        rows.append({"formula_id": f["formula_id"], "arm": f["arm"], "pool": f["pool"], "variant_type": _sa_type(f),
                     "base_id": f.get("base_id") or f["formula_id"], "nodes": cx.get("nodes"), "depth": cx.get("depth"),
                     "max_lookback": cx.get("max_lookback"), "dim_consistent": cx.get("dim_consistent"),
                     "author_model": f.get("author_model"), "stratum": f.get("stratum"), "label": f.get("label"),
                     "label_claim": (f.get("perturbation") or {}).get("target_property") if f.get("pool") == "NL" else None,
                     "label_terms": (f.get("perturbation") or {}).get("label_terms") if f.get("pool") == "NL" else None,
                     "a2_subset": bool((f.get("meta") or {}).get("a2_subset"))})
    return pd.DataFrame(rows)


def build_tables(run_dir: str | Path, market: str | None = None) -> dict[str, pd.DataFrame]:
    rd = Path(run_dir)
    formulas = read_jsonl(rd / "formulas.jsonl")
    rationales = read_jsonl(rd / "rationales.jsonl")
    claims = read_jsonl(rd / "claims.jsonl")
    verdicts = read_jsonl(rd / "verdicts.jsonl")
    F = formula_table(formulas)
    R = pd.DataFrame(rationales)
    if len(R):
        R = R.drop(columns=[c for c in ("pool", "arm") if c in R.columns]).merge(F, on="formula_id", how="left")
        R["condition"] = R.apply(condition_label, axis=1)
    C = pd.DataFrame(claims)
    V = pd.DataFrame(verdicts)
    if len(V):
        keep = ["claim_id", "verdict", "method", "regime_dependent"] + [c for c in ("market", "base_verdict") if c in V.columns]
        V = V[keep]
    if len(C):
        C = C.merge(V, on="claim_id", how="left") if len(V) else C.assign(verdict=np.nan)
        rcols = ["rationale_id", "model", "family", "access", "prompt_variant", "sample_idx", "temperature", "arm", "pool",
                 "variant_type", "base_id", "nodes", "depth", "max_lookback", "author_model", "stratum", "condition",
                 "cross_narration", "label_claim", "a2_subset"]
        C = C.drop(columns=[c for c in ("formula_id",) if c in C.columns]).merge(
            R[[c for c in rcols + ["formula_id"] if c in R.columns]], on="rationale_id", how="left")
        C["decidable"] = C["verdict"].isin(DECIDABLE)
        C["supported"] = (C["verdict"] == "SUPPORTED").astype(float)
        if market and "market" not in C.columns:
            C["market"] = market
    return {"formulas": F, "rationales": R, "claims": C}


def condition_label(r) -> str:
    pool = r.get("pool")
    if pool == "SA":
        return f"SA-{r.get('variant_type')}"
    if pool == "P1":
        return f"P1-{r.get('stratum') or 'raw'}"
    if pool in ("P3a", "P3b"):
        return "P3"
    return str(pool)
