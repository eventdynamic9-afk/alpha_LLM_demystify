"""Robustness battery (§12.7): threshold sensitivity; second parser; minimal vs guided prompt; T = 0 vs
T = 0.7; complexity-matched subsets; per market; train- vs test-window verdicts; absolute-only claims;
excluding A1 items answered by the diagnostics table; leave-one-model-out."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import cluster_bootstrap, micro_precision


def _cp_by_condition(C: pd.DataFrame, variant: str, n_boot: int, seed: int) -> list[dict]:
    rows = []
    for cond, g in C.groupby("condition"):
        r = cluster_bootstrap(g, micro_precision, "formula_id", "rationale_id", n_boot, seed) if len(g) else {}
        rows.append({"variant": variant, "condition": cond, "cp": r.get("estimate", np.nan), "ci": r.get("ci"),
                     "n_decidable": int(g["decidable"].sum())})
    return rows


def complexity_matched(C: pd.DataFrame, conditions: list[str] | None = None) -> pd.DataFrame:
    """Keep formulas whose node count lies in the overlap of every condition's node-count range."""
    conds = conditions or list(C["condition"].dropna().unique())
    sub = C[C["condition"].isin(conds)]
    lo = max(sub[sub["condition"] == c]["nodes"].quantile(0.05) for c in conds)
    hi = min(sub[sub["condition"] == c]["nodes"].quantile(0.95) for c in conds)
    return C[(C["nodes"] >= lo) & (C["nodes"] <= hi)]


def battery(C: pd.DataFrame, C_parser2: pd.DataFrame | None = None, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    rows = []
    prim = C[(C["sample_idx"] >= 0) & (C["prompt_variant"] == "guided")]
    rows += _cp_by_condition(prim, "primary", n_boot, seed)
    if C_parser2 is not None and len(C_parser2):
        rows += _cp_by_condition(C_parser2[(C_parser2["sample_idx"] >= 0)], "second_parser", n_boot, seed)
    rows += _cp_by_condition(C[C["prompt_variant"] == "minimal"], "minimal_prompt", n_boot, seed)
    rows += _cp_by_condition(C[C["sample_idx"] < 0], "temperature_0", n_boot, seed)
    rows += _cp_by_condition(complexity_matched(prim), "complexity_matched", n_boot, seed)
    if "market" in prim and prim["market"].nunique() > 1:
        for m, g in prim.groupby("market"):
            rows += _cp_by_condition(g, f"market={m}", n_boot, seed)
    if "regime_dependent" in prim:
        rows += _cp_by_condition(prim[prim["regime_dependent"] != True], "excluding_regime_dependent", n_boot, seed)  # noqa: E712
    rows += _cp_by_condition(prim[prim["hedge"] == "absolute"], "absolute_only", n_boot, seed)
    rows += _cp_by_condition(prim[~((prim["access"] == "A1") & prim["type"].isin(["C2", "C4"]))],
                             "excluding_A1_table_items", n_boot, seed)
    for model in sorted(prim["model"].dropna().unique()):
        rows += _cp_by_condition(prim[prim["model"] != model], f"leave_out={model}", n_boot, seed)
    return pd.DataFrame(rows)


def threshold_variants(thr: dict) -> dict[str, dict]:
    """Pre-registered sensitivity settings from configs/thresholds.yaml (§10.6)."""
    out = {}
    s = thr["nudge"]["sensitivity"]
    for n in s["n_contexts"]:
        t = copy.deepcopy(thr)
        t["nudge"]["n_contexts"] = n
        out[f"nudge_N={n}"] = t
    for d in s["delta_sigma"]:
        t = copy.deepcopy(thr)
        t["nudge"]["delta_sigma"] = d
        out[f"nudge_delta={d}"] = t
    for b in s["bounds"]:
        t = copy.deepcopy(thr)
        t["nudge"]["supported_lower"], t["nudge"]["refuted_upper"] = b, 1 - b
        out[f"nudge_bounds={b}"] = t
    bs = thr["behavioral"]["sensitivity"]
    for v in bs["resemblance_floor"]:
        t = copy.deepcopy(thr)
        t["behavioral"]["resemblance_floor"] = v
        out[f"resemblance_floor={v}"] = t
    for v in bs["independence_margin"]:
        t = copy.deepcopy(thr)
        t["behavioral"]["independence_margin"] = v
        out[f"independence_margin={v}"] = t
    for v in thr["originality"]["sensitivity"]["novelty_ceiling"]:
        t = copy.deepcopy(thr)
        t["originality"]["novelty_ceiling"] = v
        out[f"novelty_ceiling={v}"] = t
    for v in thr["identity"]["sensitivity"]["equivalence_rho"]:
        t = copy.deepcopy(thr)
        t["identity"]["equivalence_rho"] = v
        out[f"equivalence_rho={v}"] = t
    for v in thr["drivers"]["sensitivity"]["total_effect_threshold"]:
        t = copy.deepcopy(thr)
        t["drivers"]["total_effect_threshold"] = v
        out[f"driver_threshold={v}"] = t
    return out


def reverify(run_dir: str | Path, ctx, thr_override: dict, predicates: set | None = None, limit: int | None = None) -> pd.DataFrame:
    """Re-run verification of (a subset of) claims under modified thresholds."""
    import json

    from dsl import parse
    from verify.dispatcher import verify_claim

    rd = Path(run_dir)
    formulas = {}
    with open(rd / "formulas.jsonl", encoding="utf-8") as fh:
        for line in fh:
            f = json.loads(line)
            formulas[f["formula_id"]] = f
    old = ctx.thr
    ctx.thr = thr_override
    rows = []
    try:
        with open(rd / "claims.jsonl", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    break
                c = json.loads(line)
                if predicates and c["predicate"] not in predicates:
                    continue
                f = formulas.get(c.get("formula_id"))
                v = verify_claim(c, parse(f["dsl"]) if f else None, ctx)
                rows.append({"claim_id": c["claim_id"], "verdict": v.verdict})
    finally:
        ctx.thr = old
    return pd.DataFrame(rows)
