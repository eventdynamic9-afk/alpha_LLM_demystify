"""B5 NLI / entailment baseline (§13): premise = template-generated canonical description of the AST,
hypothesis = the claim text; output entail / contradict / neutral.

With ``transformers`` installed an off-the-shelf NLI cross-encoder is used (model name configurable).
Without it, a deterministic surface-matching entailment is used: the premise is parsed with the same
rule layer and the claim is entailed if its normalized predicate appears among the premise's claims,
contradicted if the same predicate appears with the opposite direction / sign, neutral otherwise.
Either way the baseline answers "does surface text matching suffice for C1 claims?".
"""
from __future__ import annotations

from dsl import Node, canonicalize, effective_lookback, walk
from dsl.monotonicity import NEG, POS, current_value_direction
from dsl.operators import OPS
from parse.normalize import normalize_claim, predicate_key
from parse.rules import extract_claims

_NAMES = {"Mean": "moving average", "Sum": "moving sum", "Std": "rolling standard deviation", "Var": "rolling variance",
          "Max": "rolling maximum", "Min": "rolling minimum", "TsRank": "time-series rank", "Corr": "rolling correlation",
          "Cov": "rolling covariance", "Ref": "lagged value", "Delta": "change", "WMA": "weighted moving average",
          "Slope": "trend slope", "Rsquare": "trend R-squared", "Resi": "trend residual", "IdxMax": "position of the maximum",
          "IdxMin": "position of the minimum", "Quantile": "rolling quantile", "Med": "rolling median",
          "CSRank": "cross-sectional rank", "CSZScore": "cross-sectional z-score", "CSScale": "cross-sectional scaling"}
_FIELD = {"close": "the closing price", "open": "the opening price", "high": "the daily high", "low": "the daily low",
          "vwap": "the VWAP", "volume": "trading volume", "amount": "dollar volume"}


def describe_ast(node: Node) -> str:
    """Template natural-language description of a formula (the NLI premise)."""
    canon = canonicalize(node)
    fields = sorted({n.name for n in walk(canon) if n.is_field})
    ops = []
    for n in walk(canon):
        if not n.is_leaf and n.op in _NAMES:
            w = f" over {n.params[0]} days" if OPS[n.op].kind == "ts" else ""
            ops.append(_NAMES[n.op] + w)
    L = effective_lookback(canon)
    parts = [f"The factor is driven by {', '.join(_FIELD[f] for f in fields)}."]
    if ops:
        parts.append("It is built from " + ", ".join(dict.fromkeys(ops)) + ".")
    parts.append(f"It uses a {L + 1}-day window.")
    for f in fields:
        d = current_value_direction(canon, f)
        if d == POS:
            parts.append(f"A higher {f} today raises the factor value.")
        elif d == NEG:
            parts.append(f"A higher {f} today lowers the factor value.")
    if any(not n.is_leaf and OPS[n.op].kind == "xs" for n in walk(canon)):
        parts.append("It ranks stocks against each other.")
    return " ".join(parts)


class SurfaceNLI:
    name = "surface-match"

    def predict(self, premise: str, hypothesis_claim: dict) -> str:
        prem = [normalize_claim(c) for c in extract_claims(premise, "premise")]
        hyp = normalize_claim(hypothesis_claim)
        keys = {predicate_key(c) for c in prem}
        if predicate_key(hyp) in keys:
            return "entailment"
        flip = dict(hyp)
        a = dict(hyp.get("args") or {})
        for k in ("direction", "sign"):
            if k in a:
                a[k] = "-" if a[k] == "+" else "+"
        flip["args"] = a
        if a != hyp.get("args") and predicate_key(flip) in keys:
            return "contradiction"
        return "neutral"


class TransformersNLI:  # pragma: no cover - optional heavy dependency
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        from transformers import pipeline

        self.name = model_name
        self.pipe = pipeline("text-classification", model=model_name, top_k=None)

    def predict(self, premise: str, hypothesis_claim: dict) -> str:
        out = self.pipe({"text": premise, "text_pair": hypothesis_claim.get("text", "")})
        best = max(out[0] if isinstance(out[0], list) else out, key=lambda d: d["score"])
        lab = best["label"].lower()
        return "entailment" if "entail" in lab else "contradiction" if "contra" in lab else "neutral"


def get_nli(prefer_transformers: bool = False):
    if prefer_transformers:
        try:
            return TransformersNLI()
        except Exception:
            pass
    return SurfaceNLI()


def judge_b5(nli, node: Node, claim: dict) -> dict:
    lab = nli.predict(describe_ast(node), claim)
    verdict = {"entailment": "TRUE", "contradiction": "FALSE"}.get(lab, "CANT_TELL")
    return {"judge": "B5", "model": nli.name, "claim_id": claim["claim_id"], "nli_label": lab, "verdict": verdict}
