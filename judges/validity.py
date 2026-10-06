"""Judge validity metrics (§11 "Judge validity", RQ4 / CF5).

* AUROC of the holistic score for detecting rationales with >= 1 REFUTED claim (score oriented so that
  lower consistency = more suspicious)
* Spearman rho between judge score and verified rationale-level claim precision
* false-accept rate: share of rationales with >= 1 REFUTED claim scored >= the acceptance threshold
* claim-level agreement of B2/B3/B5/B6 with the verifier on decidable claims, by claim type
* 2 x 2 reconstructable x contains-REFUTED table for B4 (the key cell: reconstructable but false)
* cost per audited claim (tokens, minutes)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from configs import thresholds


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (ties count 1/2)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rationale_truth(verdicts: list[dict], claims: list[dict]) -> dict[str, dict]:
    """Per rationale: decidable count, precision, has_refuted."""
    rid_of = {c["claim_id"]: c["rationale_id"] for c in claims}
    agg = defaultdict(lambda: {"S": 0, "R": 0})
    for v in verdicts:
        rid = rid_of.get(v["claim_id"])
        if rid is None:
            continue
        if v["verdict"] == "SUPPORTED":
            agg[rid]["S"] += 1
        elif v["verdict"] == "REFUTED":
            agg[rid]["R"] += 1
    out = {}
    for rid, a in agg.items():
        d = a["S"] + a["R"]
        out[rid] = {"decidable": d, "precision": a["S"] / d if d else float("nan"), "has_refuted": a["R"] > 0}
    return out


def holistic_validity(b1: list[dict], truth: dict[str, dict], accept: float | None = None) -> dict:
    accept = thresholds()["judges"]["b1_accept_threshold"] if accept is None else accept
    rows = [(j["C"], truth[j["rationale_id"]]) for j in b1 if j["rationale_id"] in truth and np.isfinite(j.get("C", np.nan))]
    if not rows:
        return {"n": 0}
    score = np.array([r[0] for r in rows])
    has_ref = np.array([r[1]["has_refuted"] for r in rows], dtype=int)
    prec = np.array([r[1]["precision"] for r in rows])
    ok = np.isfinite(prec)
    rho = spearmanr(score[ok], prec[ok]).statistic if ok.sum() > 2 else float("nan")
    false_acc = float(np.mean(score[has_ref == 1] >= accept)) if has_ref.any() else float("nan")
    return {"n": len(rows), "auroc_detect_refuted": auroc(-score, has_ref), "spearman_score_precision": float(rho),
            "false_accept_rate": false_acc, "accept_threshold": accept, "share_with_refuted": float(has_ref.mean())}


def claim_agreement(judgements: list[dict], verdicts: dict[str, dict], claims: dict[str, dict]) -> dict:
    """Agreement with execution-verified verdicts on decidable claims (TRUE<->SUPPORTED, FALSE<->REFUTED)."""
    by = defaultdict(lambda: {"n": 0, "agree": 0, "cant_tell": 0})
    for j in judgements:
        v = verdicts.get(j["claim_id"])
        c = claims.get(j["claim_id"])
        if v is None or c is None or v["verdict"] not in ("SUPPORTED", "REFUTED"):
            continue
        key = (j["judge"], c["type"])
        by[key]["n"] += 1
        if j["verdict"] == "CANT_TELL":
            by[key]["cant_tell"] += 1
        elif (j["verdict"] == "TRUE") == (v["verdict"] == "SUPPORTED"):
            by[key]["agree"] += 1
    return {f"{k[0]}|{k[1]}": {**d, "accuracy": d["agree"] / d["n"] if d["n"] else float("nan")} for k, d in by.items()}


def reconstruction_table(b4: list[dict], truth: dict[str, dict], rule: str = "reconstructable_alphalogics") -> dict:
    t = {"recon_true_refuted": 0, "recon_true_clean": 0, "recon_false_refuted": 0, "recon_false_clean": 0}
    for j in b4:
        tr = truth.get(j["rationale_id"])
        if tr is None or tr["decidable"] == 0:
            continue
        a = "recon_true" if j.get(rule) else "recon_false"
        b = "refuted" if tr["has_refuted"] else "clean"
        t[f"{a}_{b}"] += 1
    return t


def cost_per_claim(judgements: list[dict]) -> dict:
    agg = defaultdict(lambda: {"claims": 0, "tokens": 0, "minutes": 0.0})
    for j in judgements:
        a = agg[j["judge"]]
        a["claims"] += 1
        tok = j.get("tokens") or {}
        a["tokens"] += (tok.get("in") or 0) + (tok.get("out") or 0)
        a["minutes"] += j.get("minutes") or 0.0
    return {k: {"tokens_per_claim": v["tokens"] / v["claims"], "minutes_per_claim": v["minutes"] / v["claims"]}
            for k, v in agg.items() if v["claims"]}
