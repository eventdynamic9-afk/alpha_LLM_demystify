"""LLM authorship protocols (§7.1).

P1 hypothesis-first: field list, operator list and one of 20 fixed research directions; the model
writes a structured hypothesis (observation -> mechanism -> specification, as in AlphaAgent's idea
agent), then a formula.  P1-raw keeps the first valid formula per seed; P1-mined runs <= 5 refinement
rounds guided by validation-window IC and logs every trial (for DSR / PBO).
P2 formula-first: same specification, no hypothesis; the rationale is requested later, in a fresh
context, by the narration runner (plus one cross-narration by a different family).
"""
from __future__ import annotations

import re

import numpy as np
import yaml

from configs import PROMPT_DIR, fill, prompt, study
from dsl import ParseError, parse, to_qlib, try_parse_any
from dsl.fields import field_glossary
from dsl.operators import operator_glossary
from narrate.relay import PendingResponse
from verify.stats import daily_spearman, newey_west_mean

from .records import FormulaRecord
from .validity import PoolDeduper, check_validity

AUTHOR_FIELDS = ("open", "high", "low", "close", "vwap", "volume", "amount")
AUTHOR_OPS = ("Add", "Sub", "Mul", "Div", "Abs", "Sign", "Log", "Greater", "Less", "Ref", "Delta", "Mean", "Sum",
              "Std", "Max", "Min", "IdxMax", "IdxMin", "TsRank", "WMA", "Slope", "Corr", "Cov", "CSRank", "CSZScore")


def research_directions() -> list[str]:
    return yaml.safe_load((PROMPT_DIR / "research_directions.yaml").read_text())["directions"]


def extract_formula(text: str) -> str | None:
    m = re.findall(r"FORMULA:\s*`*([^\n`]+)`*", text or "")
    return m[-1].strip() if m else None


def extract_hypothesis(text: str) -> str | None:
    m = re.search(r"(Observation:.*?)(?:\n\s*FORMULA:|$)", text or "", re.S)
    return m.group(1).strip() if m else None


def _glossaries(market: str) -> tuple[str, str]:
    return field_glossary(AUTHOR_FIELDS, market, "qlib"), operator_glossary(AUTHOR_OPS, "qlib")


def _valid_ic(node, ctx) -> tuple[float, float]:
    if "valid" not in ctx.windows:                 # §7.1: refinement is guided by the validation window only
        raise ValueError("P1-mined needs a 'valid' window; the training window decides behavioral truth")
    rows = ctx.rows("valid")
    ic = daily_spearman(ctx.signal(node), ctx.fwd(1, "open_t+1"), rows)
    m, se, t = newey_west_mean(ic)
    return float(np.nan_to_num(m)), float(np.nan_to_num(t))


def author_p1(client, ctx, n_formulas: int = 60, mined: bool = False, rounds: int | None = None,
              temperature: float = 0.7, log=None) -> list[FormulaRecord]:
    rounds = rounds if rounds is not None else study()["arms"]["arm_a"]["P1_mined"]["max_refinement_rounds"]
    fg, og = _glossaries(ctx.panel.market)
    dirs = research_directions()
    dedup = PoolDeduper(ctx)
    out: list[FormulaRecord] = []
    seed = 0
    stratum = "mined" if mined else "raw"
    pending = 0                                      # relay requests awaiting answers (narrate.relay)
    while len(out) + pending < n_formulas and seed < n_formulas * 6:
        direction = dirs[seed % len(dirs)]
        user = fill(prompt("author_p1_hypothesis"), field_glossary=fg, operator_glossary=og,
                    research_direction=direction)
        msgs = [{"role": "user", "content": user}]
        try:
            r = client.complete(msgs, temperature=temperature, seed=seed)
        except PendingResponse:
            pending += 1
            seed += 1
            continue
        if log:
            log(role="author_p1", model=client.model_id, request=msgs, response=r.text, seed=seed)
        seed += 1
        expr, hyp = extract_formula(r.text), extract_hypothesis(r.text)
        if not expr:
            continue
        try:
            node = try_parse_any(expr, strict=False).node
        except ParseError:
            continue
        trials = [{"round": 0, "formula": to_qlib(node)}]
        if mined:
            best = node
            best_ic, best_t = _valid_ic(node, ctx)
            trials[0]["valid_ic"] = best_ic
            prev = node
            for k in range(1, rounds + 1):
                ic, t = _valid_ic(prev, ctx)
                refine = fill(prompt("author_p1_refine"), previous_formula=to_qlib(prev), valid_ic=ic, valid_t=t)
                msgs2 = msgs + [{"role": "assistant", "content": r.text}, {"role": "user", "content": refine}]
                try:
                    r2 = client.complete(msgs2, temperature=temperature, seed=seed * 100 + k)
                except PendingResponse:
                    trials = None
                    break
                if log:
                    log(role="author_p1_refine", model=client.model_id, request=msgs2, response=r2.text)
                e2 = extract_formula(r2.text)
                try:
                    cand = try_parse_any(e2, strict=False).node if e2 else None
                except ParseError:
                    cand = None
                if cand is None or not check_validity(cand, ctx, truncation=False)["valid"]:
                    trials.append({"round": k, "formula": e2, "valid": False})
                    continue
                cic, _ = _valid_ic(cand, ctx)
                trials.append({"round": k, "formula": to_qlib(cand), "valid_ic": cic})
                prev = cand
                if abs(cic) > abs(best_ic):
                    best, best_ic = cand, cic
            if trials is None:                       # a refinement round is pending
                pending += 1
                continue
            node = best
        rep = check_validity(node, ctx, dedup)
        if not rep["valid"]:
            continue
        fid = f"P1-{stratum}-{client.model_id}-{len(out):03d}"
        rec = FormulaRecord.from_node(fid, "A", "P1", node, validity=rep, stratum=stratum,
                                      author_model=client.model_id, hypothesis=hyp, trials=len(trials),
                                      seed=seed, meta={"research_direction": direction, "trial_log": trials})
        dedup.add(fid, node, ctx.signal(node))
        out.append(rec)
    return out


def author_p2(client, ctx, n_formulas: int = 60, temperature: float = 0.7, log=None) -> list[FormulaRecord]:
    fg, og = _glossaries(ctx.panel.market)
    dedup = PoolDeduper(ctx)
    out: list[FormulaRecord] = []
    seed = 0
    pending = 0
    while len(out) + pending < n_formulas and seed < n_formulas * 6:
        user = fill(prompt("author_p2_formula"), field_glossary=fg, operator_glossary=og)
        msgs = [{"role": "user", "content": user}]
        try:
            r = client.complete(msgs, temperature=temperature, seed=seed)
        except PendingResponse:
            pending += 1
            seed += 1
            continue
        if log:
            log(role="author_p2", model=client.model_id, request=msgs, response=r.text, seed=seed)
        seed += 1
        expr = extract_formula(r.text)
        if not expr:
            continue
        try:
            node = try_parse_any(expr, strict=False).node
        except ParseError:
            continue
        rep = check_validity(node, ctx, dedup)
        if not rep["valid"]:
            continue
        fid = f"P2-{client.model_id}-{len(out):03d}"
        out.append(FormulaRecord.from_node(fid, "A", "P2", node, validity=rep, author_model=client.model_id,
                                           seed=seed, trials=1))
        dedup.add(fid, node, ctx.signal(node))
    return out


__all__ = ["author_p1", "author_p2", "extract_formula", "extract_hypothesis", "research_directions", "parse"]
