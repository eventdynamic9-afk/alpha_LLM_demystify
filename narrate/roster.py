"""Model roster validation against the §4.1 selection criteria and role separation (§4.2)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from configs import models as load_models


@dataclass
class RosterReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def with_role(cfg: dict, role: str) -> list[dict]:
    return [m for m in cfg["models"] if role in m.get("roles", [])]


def validate_roster(cfg: dict | None = None, main_run: bool = True) -> RosterReport:
    cfg = cfg or load_models()
    crit = cfg.get("criteria", {})
    narr = with_role(cfg, "narrator")
    probs: list[str] = []
    closed = {m["family"] for m in narr if not m.get("open_weight")}
    opened = {m["family"] for m in narr if m.get("open_weight")}
    orgs = {m.get("organization") for m in narr}
    if len(closed) < crit.get("min_closed_families", 2):
        probs.append(f"closed-weight narrator families: {len(closed)} < {crit.get('min_closed_families', 2)}")
    if len(opened) < crit.get("min_open_families", 2):
        probs.append(f"open-weight narrator families: {len(opened)} < {crit.get('min_open_families', 2)}")
    if len(orgs) < crit.get("min_organizations", 3):
        probs.append(f"organizations: {len(orgs)} < {crit.get('min_organizations', 3)}")
    ladder = max((len({m.get('size_b') for m in narr if m["family"] == f and m.get("open_weight")}) for f in opened),
                 default=0)
    if ladder < crit.get("min_sizes_in_ladder", 3):
        probs.append(f"size ladder: {ladder} sizes < {crit.get('min_sizes_in_ladder', 3)}")
    if crit.get("require_reasoning", True) and not any(m.get("reasoning") for m in narr):
        probs.append("no reasoning-mode narrator")
    if crit.get("require_non_reasoning", True) and not any(not m.get("reasoning") for m in narr):
        probs.append("no non-reasoning narrator")
    for m in narr:
        if not m.get("tool_use"):
            probs.append(f"{m['id']}: no tool calling (cannot be used at A2)")
        if main_run:
            if not m.get("pinned"):
                probs.append(f"{m['id']}: not pinned against its model card")
            if str(m.get("training_cutoff", "TO_FILL")).startswith("TO_"):
                probs.append(f"{m['id']}: training cutoff not documented")
            if str(m.get("may_train_on_inputs", "TO_FILL")).startswith("TO_"):
                probs.append(f"{m['id']}: data-use terms not recorded")
    narr_fams = {m["family"] for m in narr}
    for role in ("parser", "judge"):
        for m in with_role(cfg, role):
            if m["family"] in narr_fams:
                probs.append(f"{role} {m['id']} shares family {m['family']} with a narrator (role separation)")
    if not with_role(cfg, "parser"):
        probs.append("no parser model")
    return RosterReport(not probs, probs, {"narrators": [m["id"] for m in narr], "closed_families": sorted(closed),
                                           "open_families": sorted(opened), "organizations": sorted(o for o in orgs if o),
                                           "ladder_sizes": ladder})


def latest_narrator_cutoff(cfg: dict) -> pd.Timestamp | None:
    cuts = []
    for m in with_role(cfg, "narrator"):
        c = str(m.get("training_cutoff", ""))
        if c and not c.startswith("TO_"):
            cuts.append(pd.Timestamp(c))
    return max(cuts) if cuts else None


def post_cutoff_window(cfg: dict, data_end: str) -> tuple[str, str] | None:
    """H_post = [latest narrator cutoff + 1 month, data end] (§5.2)."""
    c = latest_narrator_cutoff(cfg)
    if c is None:
        return None
    start = (c + pd.DateOffset(months=1)).date().isoformat()
    return (start, data_end) if pd.Timestamp(start) < pd.Timestamp(data_end) else None


def novel_pool_allowed(model: dict) -> bool:
    """The Novel pool is never sent to an endpoint that may train on inputs before the study ends."""
    return model.get("provider") in ("local", "mock") or model.get("may_train_on_inputs") is False
