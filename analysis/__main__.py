"""Analysis CLI — every number in the paper is produced here, from run outputs only.

    python -m analysis run --run-dir runs/main [--n-boot 1000] [--parser2 runs/main/claims_parser2.jsonl]
    python -m analysis power [--pilot runs/pilot]
    python -m analysis r-script --run-dir runs/main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


def run(a) -> int:
    from judges.validity import claim_agreement, holistic_validity, rationale_truth, reconstruction_table

    from . import metrics as M
    from .confirmatory import run_confirmatory
    from .dataset import build_tables, read_jsonl
    from .figures import forest_plot, verdict_composition
    from .multiplicity import benjamini_hochberg, benjamini_yekutieli
    from .robustness import battery

    rd = Path(a.run_dir)
    out = rd / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    T = build_tables(rd)
    C, R, F = T["claims"], T["rationales"], T["formulas"]
    results: dict = {"n": {"formulas": len(F), "rationales": len(R), "claims": len(C)}}
    prim = C[(C["sample_idx"] >= 0) & (C["prompt_variant"] == "guided")]
    results["precision_by_condition"] = M.precision_summary(prim, ["condition"], a.n_boot)
    results["precision_by_condition_model"] = M.precision_summary(prim, ["condition", "model"], a.n_boot)
    results["precision_by_type"] = M.precision_summary(prim, ["type"], a.n_boot)
    results["error_rate_by_type"] = M.error_rate_by_type(prim)
    results["sa_metrics"] = M.sa_metrics(C)
    results["sp_invariance"] = M.sp_invariance(C)
    results["label_following"] = M.label_following(C, F)
    results["identity_claims"] = M.identity_claims(C)
    results["counter_recall_gaps"] = M.counter_recall_gaps(prim, a.n_boot)
    results["tool_check_rate"] = M.tool_check_rate(R, C)
    results["evidence_use"] = M.evidence_use(C)
    results["sample_stability"] = M.sample_stability(C)
    drivers = {d["formula_id"]: d for d in read_jsonl(rd / "drivers.jsonl")}
    if drivers:
        dc = M.driver_coverage(R, C, drivers)
        results["driver_coverage"] = dc.groupby(["condition", "model"])[["driver_coverage", "dominant_driver_omitted"]].mean().reset_index()
    verdicts = read_jsonl(rd / "verdicts.jsonl")
    claims = read_jsonl(rd / "claims.jsonl")
    truth = rationale_truth(verdicts, claims)
    judg = read_jsonl(rd / "judgements.jsonl")
    b1 = [j for j in judg if j["judge"] == "B1"]
    if judg:
        results["judge_validity_b1"] = holistic_validity(b1, truth)
        results["judge_claim_agreement"] = claim_agreement([j for j in judg if j["judge"] in ("B2", "B3", "B5", "B6")],
                                                           {v["claim_id"]: v for v in verdicts}, {c["claim_id"]: c for c in claims})
        results["reconstruction_table"] = reconstruction_table([j for j in judg if j["judge"] == "B4"], truth)
    conf = run_confirmatory(C, F, b1, truth, a.n_boot)
    results["confirmatory"] = conf
    # exploratory family: per-model x condition contrasts against the pooled precision, BH / BY
    expl = results["precision_by_condition_model"].dropna(subset=["cp_micro"])
    pooled = M.micro_precision(prim) if len(prim) else np.nan
    pvals = []
    for _, r in expl.iterrows():
        g = prim[(prim["condition"] == r["condition"]) & (prim["model"] == r["model"]) & prim["decidable"]]
        k, n = int(g["supported"].sum()), len(g)
        from scipy.stats import binomtest

        pvals.append(binomtest(k, n, pooled).pvalue if n and pooled == pooled else np.nan)
    fin = [p for p in pvals if p == p]
    if fin:
        results["exploratory"] = {"tests": "per model x condition precision vs pooled precision (labelled exploratory)",
                                  "bh": benjamini_hochberg(fin), "by": benjamini_yekutieli(fin)}
    C2 = None
    if a.parser2 and Path(a.parser2).exists():
        alt = pd.DataFrame(read_jsonl(a.parser2))
        V = pd.DataFrame(verdicts)[["claim_id", "verdict"]]
        C2 = alt.merge(V, on="claim_id", how="left").merge(
            R[["rationale_id", "condition", "sample_idx", "formula_id", "model", "nodes", "access", "prompt_variant", "hedge"]
              if "hedge" in R else ["rationale_id", "condition", "sample_idx", "formula_id", "model", "nodes", "access", "prompt_variant"]],
            on="rationale_id", how="left")
        C2["decidable"] = C2["verdict"].isin(["SUPPORTED", "REFUTED"])
        C2["supported"] = (C2["verdict"] == "SUPPORTED").astype(float)
    results["robustness"] = battery(C, C2, n_boot=max(100, a.n_boot // 4))
    if len(results["precision_by_condition_model"]):
        forest_plot(results["precision_by_condition_model"], out / "forest_cp_by_model.png")
    if len(C):
        verdict_composition(C, out / "verdict_composition.png")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=_jsonable))
    print(json.dumps({k: (v.get("p_holm"), v.get("reject_h0")) for k, v in conf.items()}, indent=1, default=_jsonable))
    print(f"analysis written to {out}")
    return 0


def power(a) -> int:
    from .power import required_formulas, worked_example

    print(json.dumps(worked_example(), indent=1))
    if a.simulate:
        print(json.dumps(required_formulas(n_sims=a.n_sims), indent=1, default=_jsonable))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--parser2")
    p = sub.add_parser("power")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--n-sims", type=int, default=200)
    s = sub.add_parser("r-script")
    s.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "power":
        return power(a)
    from .glmm import write_r_script

    p = write_r_script(Path(a.run_dir) / "analysis" / "glmm.R", str(Path(a.run_dir) / "claims.jsonl"))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
