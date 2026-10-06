"""LLM-based judge baselines (§13).

B1  holistic consistency judge (AlphaAgent-style): c2 = description-expression consistency in [0, 1],
    c1 = hypothesis-description consistency when a hypothesis exists, C = 0.5 c1 + 0.5 c2.
B2  claim-level judge without execution: TRUE / FALSE / CANT_TELL per parsed claim.
B3  claim-level judge with the A2 sandbox tools (is execution access, not our verifier, enough?).
B4  reconstruction test (AlphaLogics): rebuild the formula from the rationale alone, then compare.
Judges come from families not used as narrators in the main comparison; a same-family judge run
measures self-preference.
"""
from __future__ import annotations

import json
import re

import numpy as np

from configs import fill, prompt
from dsl import ParseError, parse, try_parse_any
from dsl.fields import field_glossary
from dsl.operators import operator_glossary
from narrate.prompts import _ops_in
from narrate.sandbox import TOOL_SCHEMAS, Sandbox, tool_message
from verify.identity import numerically_equivalent


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _glossaries(rec: dict, market: str) -> tuple[str, str]:
    node = parse(rec["dsl"])
    return (field_glossary(("open", "high", "low", "close", "vwap", "volume", "amount"), market, "qlib"),
            operator_glossary([o for o in _ops_in(node) if o != "Neg"], "qlib"))


# ------------------------------------------------------------------------------------------- B1
def judge_b1(client, rec: dict, rationale_text: str, hypothesis: str | None = None, market: str = "CN") -> dict:
    fg, og = _glossaries(rec, market)
    hblock = fill(prompt("judge_b1_hypothesis_block"), hypothesis=hypothesis) if hypothesis else ""
    user = fill(prompt("judge_b1_holistic"), hypothesis_block=hblock, description=rationale_text,
                expression=rec["presented"], field_glossary=fg, operator_glossary=og,
                c1_field=', "c1": <score between 0 and 1>' if hypothesis else "")
    r = client.complete([{"role": "user", "content": user}], temperature=0.0, max_tokens=300, seed=0)
    out = _json(r.text)
    c2 = float(out.get("c2", np.nan)) if out.get("c2") is not None else float("nan")
    c1 = float(out["c1"]) if hypothesis and out.get("c1") is not None else None
    score = 0.5 * c1 + 0.5 * c2 if c1 is not None else c2
    return {"judge": "B1", "model": client.model_id, "c2": c2, "c1": c1, "C": score,
            "explanation": out.get("explanation"), "tokens": r.usage}


# ------------------------------------------------------------------------------------------- B2/B3
def _claim_prompt(name: str, rec: dict, claim: dict, market: str, extra: dict | None = None) -> str:
    fg, og = _glossaries(rec, market)
    return fill(prompt(name), expression=rec["presented"], field_glossary=fg, operator_glossary=og,
                claim_text=claim.get("text", ""), predicate=claim["predicate"],
                args=json.dumps(claim.get("args", {})), **(extra or {}))


def judge_b2(client, rec: dict, claim: dict, market: str = "CN") -> dict:
    user = _claim_prompt("judge_b2_claim", rec, claim, market)
    r = client.complete([{"role": "user", "content": user}], temperature=0.0, max_tokens=200, seed=0)
    out = _json(r.text)
    v = str(out.get("verdict", "CANT_TELL")).upper().replace("'", "")
    return {"judge": "B2", "model": client.model_id, "claim_id": claim["claim_id"],
            "verdict": v if v in ("TRUE", "FALSE", "CANT_TELL") else "CANT_TELL", "reason": out.get("reason"),
            "tokens": r.usage}


def judge_b3(client, rec: dict, claim: dict, ctx, market: str = "CN", max_calls: int = 10) -> dict:
    refs = ", ".join(ctx.references.characteristic_names())
    user = _claim_prompt("judge_b3_claim_tools", rec, claim, market, {"reference_names": refs})
    msgs = [{"role": "user", "content": user}]
    sb = Sandbox(ctx, max_calls)
    r = client.complete(msgs, tools=TOOL_SCHEMAS, temperature=0.0, max_tokens=400, seed=0)
    tokens = dict(r.usage)
    while r.tool_calls and not sb.exhausted:
        msgs.append({"role": "assistant", "content": r.text or "", "tool_calls": [
            {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
            for tc in r.tool_calls]})
        for tc in r.tool_calls:
            msgs.append(tool_message(tc["id"], sb.call(tc["name"], tc.get("arguments") or {})))
        r = client.complete(msgs, tools=TOOL_SCHEMAS, temperature=0.0, max_tokens=400, seed=0)
    out = _json(r.text)
    v = str(out.get("verdict", "CANT_TELL")).upper()
    return {"judge": "B3", "model": client.model_id, "claim_id": claim["claim_id"],
            "verdict": v if v in ("TRUE", "FALSE", "CANT_TELL") else "CANT_TELL", "reason": out.get("reason"),
            "tool_calls": sb.calls, "tokens": tokens}


# ------------------------------------------------------------------------------------------- B4
def concordance(a: np.ndarray, b: np.ndarray, min_n: int = 10, max_pairs_stocks: int = 400) -> float:
    """Mean daily share of concordant stock pairs, (1 + Kendall tau) / 2 — our operationalization of
    AlphaLogics' "rankings / trends match in > 90% of cases"."""
    from scipy.stats import kendalltau

    vals = []
    for t in range(a.shape[0]):
        m = np.isfinite(a[t]) & np.isfinite(b[t])
        if m.sum() < min_n:
            continue
        x, y = a[t, m][:max_pairs_stocks], b[t, m][:max_pairs_stocks]
        tau = kendalltau(x, y).statistic
        if np.isfinite(tau):
            vals.append((1 + tau) / 2)
    return float(np.mean(vals)) if vals else float("nan")


def judge_b4(client, rec: dict, rationale_text: str, ctx, market: str = "CN") -> dict:
    fg, og = _glossaries(rec, market)
    user = fill(prompt("judge_b4_reconstruct"), field_glossary=fg, operator_glossary=og, rationale_text=rationale_text)
    r = client.complete([{"role": "user", "content": user}], temperature=0.0, max_tokens=300, seed=0)
    m = re.findall(r"FORMULA:\s*`*([^\n`]+)`*", r.text or "")
    out = {"judge": "B4", "model": client.model_id, "reconstructed": m[-1].strip() if m else None, "tokens": r.usage}
    if not m:
        out.update({"reconstructable_alphalogics": False, "reconstructable_strict": False, "reason": "no formula"})
        return out
    try:
        node = try_parse_any(m[-1], strict=False).node
    except ParseError as exc:
        out.update({"reconstructable_alphalogics": False, "reconstructable_strict": False, "reason": str(exc)})
        return out
    a = ctx.signal(parse(rec["dsl"]))
    b = ctx.signal(node)
    thr = ctx.thr
    conc = concordance(a[ctx.rows("train")], b[ctx.rows("train")])
    strict = numerically_equivalent(a, b, thr["identity"]["equivalence_rho"], thr["identity"]["equivalence_date_share"])
    out.update({"concordance": conc,
                "reconstructable_alphalogics": bool(conc > thr["judges"]["b4_alphalogics_agreement"]),
                "reconstructable_strict": bool(strict["equivalent"])})
    return out
