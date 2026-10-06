"""Parser validation against human gold extractions (§9.3).

Matching rule: an extracted claim matches a gold claim iff their normalized predicates are equal and
their spans overlap.  Reports extraction precision / recall / F1 by claim type, type accuracy and slot
accuracy, and the go/no-go decision (F1 >= 0.80 for C1/C6 and >= 0.70 for C2/C3, plus agreement at or
above the tentative threshold).
"""
from __future__ import annotations

from collections import defaultdict

from configs import thresholds

from .ensemble import spans_overlap
from .normalize import normalize_claim, predicate_key


def _match_pairs(pred: list[dict], gold: list[dict], key=predicate_key):
    used = set()
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j in used:
                continue
            if key(p) == key(g) and spans_overlap(p["span"], g["span"]):
                used.add(j)
                pairs.append((i, j))
                break
    return pairs


def extraction_metrics(pred_by_rid: dict[str, list[dict]], gold_by_rid: dict[str, list[dict]]) -> dict:
    tp = defaultdict(int)
    n_pred = defaultdict(int)
    n_gold = defaultdict(int)
    type_ok = type_n = 0
    slot_ok = defaultdict(int)
    slot_n = defaultdict(int)
    for rid in set(pred_by_rid) | set(gold_by_rid):
        pred = [normalize_claim(c) for c in pred_by_rid.get(rid, [])]
        gold = [normalize_claim(c) for c in gold_by_rid.get(rid, [])]
        for c in pred:
            n_pred[c["type"]] += 1
        for g in gold:
            n_gold[g["type"]] += 1
        for i, j in _match_pairs(pred, gold):
            tp[gold[j]["type"]] += 1
            for slot in ("hedge", "polarity", "scope", "horizon"):
                slot_n[slot] += 1
                slot_ok[slot] += int(pred[i].get(slot) == gold[j].get(slot))
        # type accuracy: span-overlapping pairs regardless of predicate
        for g in gold:
            for c in pred:
                if spans_overlap(c["span"], g["span"]):
                    type_n += 1
                    type_ok += int(c["type"] == g["type"])
                    break
    out = {"by_type": {}}
    for t in sorted(set(n_pred) | set(n_gold)):
        p = tp[t] / n_pred[t] if n_pred[t] else float("nan")
        r = tp[t] / n_gold[t] if n_gold[t] else float("nan")
        f1 = 2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else float("nan")
        out["by_type"][t] = {"precision": p, "recall": r, "f1": f1, "n_pred": n_pred[t], "n_gold": n_gold[t]}
    out["type_accuracy"] = type_ok / type_n if type_n else float("nan")
    out["slot_accuracy"] = {s: slot_ok[s] / slot_n[s] for s in slot_n}
    return out


def go_no_go(metrics: dict, agreement_alpha: float | None = None) -> dict:
    cfg = thresholds()["parser_go_no_go"]
    reasons = []

    def f1(types):
        vals = [metrics["by_type"][t]["f1"] for t in types if t in metrics["by_type"]]
        vals = [v for v in vals if v == v]
        return min(vals) if vals else float("nan")

    f_c1c6, f_c2c3 = f1(["C1", "C6"]), f1(["C2", "C3"])
    if not (f_c1c6 >= cfg["f1_c1_c6"]):
        reasons.append(f"F1(C1/C6) = {f_c1c6:.3f} < {cfg['f1_c1_c6']}")
    if not (f_c2c3 >= cfg["f1_c2_c3"]):
        reasons.append(f"F1(C2/C3) = {f_c2c3:.3f} < {cfg['f1_c2_c3']}")
    if agreement_alpha is not None and agreement_alpha < cfg["alpha_tentative"]:
        reasons.append(f"Krippendorff alpha = {agreement_alpha:.3f} < {cfg['alpha_tentative']}")
    return {"go": not reasons, "reasons": reasons, "f1_c1_c6": f_c1c6, "f1_c2_c3": f_c2c3}
