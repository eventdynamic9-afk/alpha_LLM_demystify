"""Rationale-elicitation prompts (§8, Appendix A).

Builds the exact messages for access levels A0 / A1 / A2 and prompt variants guided / minimal from the
frozen templates in ``configs/prompts``.  Glossary and field order are randomized per narration
(§8.4); no condition labels, pool names or the word "perturbed" ever appear in prompts.

§14 ablation variants (secondary, exploratory; never in the primary analysis): ``structured`` (claims as
bullet facts), ``cap150`` (150-word cap in the system prompt) and ``guided_effort_<level>`` (guided
prompt at a non-default reasoning effort; the effort itself is a request parameter, see the runner).
"""
from __future__ import annotations

import random

from configs import fill, prompt, template_sha256
from dsl import parse, walk
from dsl.fields import field_description
from dsl.operators import OPS, operator_glossary
from dsl.serialize import anonymize_legend

ALL_FIELDS = ("open", "high", "low", "close", "vwap", "volume", "amount")
# prompt variant -> (system template, user template); guided / minimal are the frozen Appendix A templates
VARIANTS = {"guided": ("narrator_system", "narrator_a0_guided"),
            "minimal": ("narrator_system", "narrator_a0_minimal"),
            "structured": ("narrator_system", "narrator_a0_structured"),
            "cap150": ("narrator_system_cap150", "narrator_a0_guided")}
EFFORT_TAG = "_effort_"


def split_variant(variant: str) -> tuple[str, str | None]:
    """``guided_effort_high`` -> (``guided``, ``high``); plain variants -> (variant, None)."""
    base, _, effort = variant.partition(EFFORT_TAG)
    return base, (effort or None)


def variant_templates(variant: str) -> tuple[str, str]:
    base, _ = split_variant(variant)
    if base not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; expected one of {sorted(VARIANTS)}")
    return VARIANTS[base]


def _ops_in(node) -> list[str]:
    seen = []
    for n in walk(node):
        if not n.is_leaf and n.op not in seen:
            seen.append(n.op)
    return seen


def field_glossary_for(record: dict, market: str, rng: random.Random) -> str:
    names = list(ALL_FIELDS)
    rng.shuffle(names)
    legend = record.get("legend")
    lines = []
    for n in names:
        desc = field_description(n, market)
        if legend:
            label = legend.get(n)
            if label is None:
                continue
        else:
            notation = record.get("notation", "qlib")
            label = {"alpha101": n, "gtja": n.upper()}.get(notation, f"${n}")
            if notation == "math":
                label = n
        lines.append(f"{label}: {desc}")
    return "; ".join(lines)


def operator_glossary_for(record: dict, rng: random.Random) -> str:
    node = parse(record["dsl"])
    ops = [o for o in _ops_in(node) if o != "Neg"]
    rng.shuffle(ops)
    notation = record.get("notation", "qlib")
    gl_notation = {"alpha101": "alpha101", "gtja": "gtja"}.get(notation, "qlib")
    if notation == "program":
        gl_notation = "alpha101" if "delay(" in record["presented"] or "rank(" in record["presented"] else "qlib"
    return operator_glossary(ops, gl_notation)


def build_messages(record: dict, access: str = "A0", variant: str = "guided", market: str = "CN",
                   seed: int = 0, diagnostics: str | None = None, train_window: str | None = None) -> tuple[list[dict], dict]:
    """Return (messages, meta) — meta carries template hashes for the call log."""
    rng = random.Random(seed)
    sys_name, user_name = variant_templates(variant)
    system = prompt(sys_name)
    tmpl = prompt(user_name)
    label = record.get("label") or ""
    user = fill(tmpl, field_glossary=field_glossary_for(record, market, rng),
                operator_glossary=operator_glossary_for(record, rng),
                formula_surface_form=record["presented"], optional_label_line=label)
    if not label:
        user = user.replace("\n\n\n", "\n\n")
    hashes = {"system": template_sha256(system), "user": template_sha256(tmpl)}
    if access in ("A1",):
        add = prompt("narrator_a1_addition")
        user += "\n\n" + fill(add, train_window=train_window or "the training window",
                              diagnostics_table=diagnostics or "(not available)")
        hashes["a1"] = template_sha256(add)
    if access == "A2":
        add = prompt("narrator_a2_addition")
        user += "\n\n" + add
        hashes["a2"] = template_sha256(add)
    for banned in ("perturbed", "pool", "counter-recall", "semantics-altering"):
        assert banned not in user.lower().replace("pooling", ""), f"prompt leaks condition label: {banned}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], {"template_sha256": hashes}


def legend_for_all_fields(node, seed: int) -> dict:
    return anonymize_legend(node, seed, all_fields=True)


__all__ = ["build_messages", "field_glossary_for", "operator_glossary_for", "legend_for_all_fields", "OPS", "VARIANTS",
           "split_variant", "variant_templates"]
