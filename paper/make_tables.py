"""Paper tables generated from analysis outputs only (§16, §12.8): every rate with its CI and raw counts,
effect sizes before p-values, pre-registered vs exploratory results separated, data tier stated.

    python -m paper tables --run-dir runs/main [--data-tier FREE]
"""
from __future__ import annotations

import json
from pathlib import Path


def _fmt_ci(ci) -> str:
    if not ci or ci[0] != ci[0]:
        return "–"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _p(x) -> str:
    if x is None or x != x:
        return "–"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def table_precision(rows: list[dict], key: str = "condition") -> str:
    out = [f"| {key} | claims | decidable | CP (micro) | 95% CI | CP (macro) | refuted | unresolved | unverifiable | ambiguous |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.get(key)} | {r['n_claims']} | {r['n_decidable']} | {r['cp_micro']:.3f} | {_fmt_ci(r['cp_micro_ci'])} | "
                   f"{r['cp_macro']:.3f} | {r['share_refuted']:.3f} | {r['share_unresolved']:.3f} | "
                   f"{r['share_unverifiable']:.3f} | {r['share_ambiguous']:.3f} |")
    return "\n".join(out)


def table_confirmatory(conf: dict) -> str:
    out = ["| test | RQ | hypothesis | effect | 95% CI | p (raw) | source | p (Holm) | reject H0 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, v in conf.items():
        if "diff_pp" in v:
            eff, ci = f"{v['diff_pp']:+.1f} pp", v.get("diff_ci_pp")
            ci = f"[{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci and ci[0] == ci[0] else "–"
        elif "auroc" in v:
            eff, ci = f"AUROC {v['auroc']:.3f}", _fmt_ci(v.get("ci"))
        else:
            eff, ci = "–", "–"
        out.append(f"| {k} | {v['rq']} | {v['hypothesis']} | {eff} | {ci} | {_p(v.get('p'))} | {v.get('p_source', '–')} | "
                   f"{_p(v.get('p_holm'))} | {v.get('reject_h0', '–')} |")
    return "\n".join(out)


def make_tables(run_dir: str | Path, data_tier: str = "FREE") -> Path:
    rd = Path(run_dir)
    res = json.loads((rd / "analysis" / "results.json").read_text())
    parts = [f"# Results tables (generated; data tier: `[{data_tier}]`)\n",
             f"Units: {res['n']['formulas']} formulas, {res['n']['rationales']} rationales, {res['n']['claims']} claims.\n",
             "## Pre-registered confirmatory family (Holm, FWER 0.05)\n", table_confirmatory(res["confirmatory"]), "",
             "## Claim precision by condition (primary: guided prompt, k samples at T = 0.7)\n",
             table_precision(res["precision_by_condition"]), "",
             "## Claim precision by claim type\n", table_precision(res["precision_by_type"], "type"), "",
             "## Exploratory (labelled exploratory; BH / BY at FDR 0.10)\n",
             "Per model × condition precision: see `results.json` → `precision_by_condition_model`, `exploratory`.\n"]
    if "judge_validity_b1" in res:
        j = res["judge_validity_b1"]
        parts += ["## Judge validity (RQ4)\n",
                  f"B1 holistic judge: AUROC for rationales with ≥ 1 refuted claim = {j.get('auroc_detect_refuted', float('nan')):.3f}; "
                  f"false-accept rate at score ≥ {j.get('accept_threshold')} = {j.get('false_accept_rate', float('nan')):.3f}; "
                  f"Spearman(score, verified precision) = {j.get('spearman_score_precision', float('nan')):.3f} (n = {j.get('n')}).\n"]
    if "reconstruction_table" in res:
        t = res["reconstruction_table"]
        parts += ["### B4 reconstruction × truth\n", "| | contains refuted | clean |", "|---|---|---|",
                  f"| reconstructable | {t['recon_true_refuted']} | {t['recon_true_clean']} |",
                  f"| not reconstructable | {t['recon_false_refuted']} | {t['recon_false_clean']} |", ""]
    parts += ["## Deviations from the pre-registration\n", "List every deviation here (Appendix E.7).\n"]
    out = rd / "analysis" / "tables.md"
    out.write_text("\n".join(parts))
    return out
