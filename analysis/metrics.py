"""Metrics (§11), each reported with cluster-bootstrap (or PPI) confidence intervals and raw counts."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .bootstrap import cluster_bootstrap, macro_precision, micro_precision

# claim predicates that address the property each SA perturbation targets (§11 counterfactual metrics)
TARGETED = {"sign": {"SIGN", "MONO", "PRED_SIGN"}, "window": {"LOOKBACK", "HORIZON"},
            "field": {"DEPENDS_ON", "RESEMBLES"}}
TOOL_FOR = {"RESEMBLES": "corr_with", "INDEPENDENT": "corr_with", "EXPOSED": "corr_with", "SIGN": "perturb",
            "MONO": "perturb", "PRED_SIGN": "compute_signal", "PERF": "compute_signal", "RANGE": "describe",
            "TURNOVER": "compute_signal"}
FIELD_NAMES = ("open", "high", "low", "close", "vwap", "volume", "amount")


def _ci(df, stat, n_boot, seed, cluster="formula_id"):
    r = cluster_bootstrap(df, stat, cluster, "rationale_id", n_boot, seed) if len(df) else {"estimate": float("nan"), "ci": [float("nan")] * 2}
    r.pop("boot", None)
    return r


def precision_summary(C: pd.DataFrame, by: list[str] | None = None, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Claim precision (micro, macro), decidability and verdict shares, claim density — overall or by group."""
    groups = [((), C)] if not by else list(C.groupby(by))
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        n_r = g["rationale_id"].nunique()
        mic = _ci(g, micro_precision, n_boot, seed)
        mac = _ci(g, macro_precision, n_boot, seed)
        vc = g["verdict"].value_counts()
        rows.append({**dict(zip(by or [], key)), "n_claims": len(g), "n_rationales": n_r,
                     "n_decidable": int(g["decidable"].sum()), "cp_micro": mic["estimate"], "cp_micro_ci": mic["ci"],
                     "cp_macro": mac["estimate"], "cp_macro_ci": mac["ci"],
                     **{f"share_{v.lower()}": float(vc.get(v, 0) / len(g)) for v in
                        ("SUPPORTED", "REFUTED", "UNRESOLVED", "UNVERIFIABLE", "AMBIGUOUS")},
                     "claims_per_rationale": len(g) / n_r if n_r else float("nan"),
                     "decidable_per_rationale": g["decidable"].sum() / n_r if n_r else float("nan")})
    return pd.DataFrame(rows)


def error_rate_by_type(C: pd.DataFrame) -> pd.DataFrame:
    d = C[C["decidable"]]
    return d.groupby("type").apply(lambda g: pd.Series({"n_decidable": len(g), "error_rate": 1 - g["supported"].mean()}),
                                   include_groups=False).reset_index()


# ---------------------------------------------------------------------------------- coverage
def mentioned_fields(claims: pd.DataFrame) -> dict[str, set]:
    out = defaultdict(set)
    for _, c in claims.iterrows():
        if c["predicate"] in ("DEPENDS_ON", "SIGN", "MONO"):
            inp = str((c.get("args") or {}).get("input", "")).lower()
            if inp in FIELD_NAMES:
                out[c["rationale_id"]].add(inp)
            elif inp.startswith("ret") or inp.startswith("volatil"):
                out[c["rationale_id"]].add("close")
            elif inp == "abn_vol":
                out[c["rationale_id"]].add("volume")
    return out


def driver_coverage(R: pd.DataFrame, C: pd.DataFrame, drivers: dict[str, dict]) -> pd.DataFrame:
    """Driver coverage = |mentioned ∩ true drivers| / |true drivers|; dominant-driver omission."""
    ment = mentioned_fields(C)
    rows = []
    for _, r in R.iterrows():
        d = drivers.get(r["formula_id"])
        if not d or not d.get("true_drivers"):
            continue
        td = set(d["true_drivers"])
        st = {k: v for k, v in d["total_effect"].items() if v == v}
        dom = max(st, key=st.get) if st else None
        m = ment.get(r["rationale_id"], set())
        rows.append({"rationale_id": r["rationale_id"], "formula_id": r["formula_id"], "condition": r.get("condition"),
                     "model": r.get("model"), "driver_coverage": len(m & td) / len(td),
                     "dominant_driver_omitted": float(dom is not None and dom not in m)})
    return pd.DataFrame(rows)


def exposure_coverage(R: pd.DataFrame, C: pd.DataFrame, exposures: dict[str, list[str]], term_refs: dict[str, list[str]]) -> pd.DataFrame:
    """Same as driver coverage for reference signals with rho_bar CI lower >= 0.30."""
    ment = defaultdict(set)
    for _, c in C[C["predicate"].isin(["RESEMBLES", "EXPOSED"])].iterrows():
        ref = str((c.get("args") or {}).get("ref", ""))
        ment[c["rationale_id"]].update(term_refs.get(ref, [ref]))
    rows = []
    for _, r in R.iterrows():
        ex = exposures.get(r["formula_id"]) or []
        if not ex:
            continue
        m = ment.get(r["rationale_id"], set())
        rows.append({"rationale_id": r["rationale_id"], "condition": r.get("condition"), "model": r.get("model"),
                     "exposure_coverage": len(m & set(ex)) / len(ex), "dominant_exposure_omitted": float(ex[0] not in m)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- counter-recall
def sa_metrics(C: pd.DataFrame) -> pd.DataFrame:
    """Counterfactual fidelity and recall-anchoring rate per SA type (and model)."""
    rows = []
    sa = C[C["pool"] == "SA"]
    for (t, model), g in sa.groupby(["variant_type", "model"]):
        tgt = g[g["predicate"].isin(TARGETED.get(t, set()))]
        mention = tgt.groupby("rationale_id")
        n_rat = g["rationale_id"].nunique()
        dec = tgt[tgt["decidable"]]
        fid = dec.groupby("rationale_id")["supported"].mean().mean() if len(dec) else float("nan")
        anch = float("nan")
        if "base_verdict" in tgt.columns:
            flag = (tgt["base_verdict"] == "SUPPORTED") & (tgt["verdict"] == "REFUTED")
            anch = float(flag.groupby(tgt["rationale_id"]).any().reindex(g["rationale_id"].unique(), fill_value=False).mean())
        rows.append({"variant_type": t, "model": model, "n_rationales": n_rat,
                     "share_mentioning_target": mention.ngroups / n_rat if n_rat else float("nan"),
                     "counterfactual_fidelity": float(fid), "recall_anchoring_rate": anch})
    return pd.DataFrame(rows)


def _pred_set(g: pd.DataFrame) -> set:
    from parse.normalize import predicate_key

    return {predicate_key(c) for c in g.to_dict("records")}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def sp_invariance(C: pd.DataFrame) -> pd.DataFrame:
    """Jaccard(K, SP) of normalized predicate sets / within-condition sample-to-sample Jaccard (≈1 = as
    stable as resampling)."""
    rows = []
    sub = C[C["pool"].isin(["K", "SP"]) & (C["prompt_variant"] == "guided")]
    for (model, base), g in sub.groupby(["model", "base_id"]):
        k = {s: _pred_set(x) for s, x in g[g["pool"] == "K"].groupby("sample_idx")}
        sp = {s: _pred_set(x) for s, x in g[g["pool"] == "SP"].groupby("sample_idx")}
        if not k or not sp:
            continue
        between = np.nanmean([jaccard(a, b) for a in k.values() for b in sp.values()])
        within = np.nanmean([jaccard(a, b) for a, b in combinations(list(k.values()) + list(sp.values()), 2)]
                            if len(k) + len(sp) > 1 else [np.nan])
        rows.append({"model": model, "base_id": base, "jaccard_between": between, "jaccard_within": within,
                     "sp_invariance": between / within if within and within == within else float("nan")})
    return pd.DataFrame(rows)


def asserts_label(claim: dict, terms: list[str]) -> bool:
    """A rationale claim asserts the label's property: affirmative resemblance/exposure with a label term."""
    if claim.get("predicate") not in ("RESEMBLES", "EXPOSED") or claim.get("polarity", "affirm") != "affirm":
        return False
    ref = str((claim.get("args") or {}).get("ref", "")).lower()
    return any(ref == t or t in ref for t in terms)


def label_following(C: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """Share of NL rationales asserting the misleading label's property vs the same property in K (same base)."""
    rows = []
    nl = F[F["pool"] == "NL"]
    for _, f in nl.iterrows():
        terms = f.get("label_terms")
        if not isinstance(terms, list) or not terms:
            continue
        for pool in ("NL", "K"):
            g = C[(C["base_id"] == f["base_id"]) & (C["pool"] == pool)]
            for model, gm in g.groupby("model"):
                rats = gm["rationale_id"].unique()
                hit = {c["rationale_id"] for c in gm.to_dict("records") if asserts_label(c, terms)}
                rows.append({"base_id": f["base_id"], "pool": pool, "model": model, "n_rationales": len(rats),
                             "label_following_rate": len(hit) / len(rats) if len(rats) else np.nan})
    return pd.DataFrame(rows)


def identity_claims(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pool, model), g in C.groupby(["pool", "model"]):
        ids = g[g["type"] == "C6"]
        n_r = g["rationale_id"].nunique()
        dec = ids[ids["decidable"]]
        rows.append({"pool": pool, "model": model, "identity_claim_rate": ids["rationale_id"].nunique() / n_r if n_r else np.nan,
                     "identity_precision": dec["supported"].mean() if len(dec) else np.nan, "n_identity_claims": len(ids)})
    return pd.DataFrame(rows)


def counter_recall_gaps(C: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Δ_SA = CP(K) - CP(SA); Δ_N = CP(K) - CP(N); Δ_SP = CP(K) - CP(SP) (expected ≈ 0)."""
    from .bootstrap import bootstrap_difference

    out = {}
    pools = C.assign(pool2=C["pool"])
    for other in ("SA", "N", "SP"):
        sub = pools[pools["pool2"].isin(["K", other])]
        paired = other in ("SA", "SP")
        r = bootstrap_difference(sub, "pool2", "K", other, micro_precision, cluster="base_id" if paired else "formula_id",
                                 n_boot=n_boot, seed=seed, paired=paired)
        out[f"delta_{other}"] = r
    return out


# ---------------------------------------------------------------------------------- access levels
def tool_check_rate(R: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """A2: share of decidable claims preceded by a relevant tool call; precision of checked vs unchecked."""
    calls = {r["rationale_id"]: {t["name"] for t in (r.get("tool_calls") or [])} for _, r in R[R["access"] == "A2"].iterrows()}
    d = C[(C["access"] == "A2") & C["decidable"]].copy()
    if not len(d):
        return pd.DataFrame()
    d["checked"] = [TOOL_FOR.get(p) in calls.get(r, set()) for p, r in zip(d["predicate"], d["rationale_id"])]
    return d.groupby(["model", "checked"])["supported"].agg(["mean", "count"]).reset_index().rename(
        columns={"mean": "precision", "count": "n"})


def evidence_use(C: pd.DataFrame) -> pd.DataFrame:
    """A1: behavioral/performance claims restated correctly (SUPPORTED) vs contradicting the table (REFUTED)."""
    d = C[(C["access"] == "A1") & C["type"].isin(["C2", "C4"]) & C["decidable"]]
    return d.groupby("model").apply(lambda g: pd.Series({"n": len(g), "restated_correctly": g["supported"].mean(),
                                                         "contradiction_rate": 1 - g["supported"].mean()}),
                                    include_groups=False).reset_index()


def sample_stability(C: pd.DataFrame) -> pd.DataFrame:
    """Fleiss' kappa over the k samples of a cell on predicate presence; verdict variance across samples."""
    from annotation.agreement import fleiss_kappa
    from parse.normalize import predicate_key

    rows = []
    prim = C[C["sample_idx"] >= 0]
    for (model, cond), g in prim.groupby(["model", "condition"]):
        cells = []
        for (fid, acc, var), h in g.groupby(["formula_id", "access", "prompt_variant"]):
            sets = [set(predicate_key(c) for c in x.to_dict("records")) for _, x in h.groupby("sample_idx")]
            if len(sets) < 2:
                continue
            universe = set().union(*sets)
            for key in universe:
                yes = sum(key in s for s in sets)
                cells.append([yes, len(sets) - yes])
        kappa = fleiss_kappa(np.array(cells)) if cells else float("nan")
        prec = g[g["decidable"]].groupby(["formula_id", "access", "prompt_variant", "sample_idx"])["supported"].mean()
        var = prec.groupby(level=[0, 1, 2]).var().mean() if len(prec) else float("nan")
        rows.append({"model": model, "condition": cond, "fleiss_kappa_predicates": kappa, "precision_variance": var})
    return pd.DataFrame(rows)


def fidelity_performance(C: pd.DataFrame, oos_ic: dict[str, float]) -> dict:
    """Exploratory: correlation between formula-level CP and OOS RankIC (HLZ caveats apply)."""
    from scipy.stats import spearmanr

    fp = C[C["decidable"]].groupby("formula_id")["supported"].mean()
    x = [(fp[f], abs(oos_ic[f])) for f in fp.index if f in oos_ic and oos_ic[f] == oos_ic[f]]
    if len(x) < 5:
        return {"n": len(x)}
    a, b = zip(*x)
    r = spearmanr(a, b)
    return {"n": len(x), "spearman": float(r.statistic), "p": float(r.pvalue), "label": "exploratory"}
