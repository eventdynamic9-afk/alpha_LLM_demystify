"""Run plan (§3.5, §7.3): which (model, formula, access, prompt variant, sample) cells are narrated.

Per narrator model (main study):
  Arm B  K, SP, N (60 each), SA (180), K_named (30), NL (30) at A0; K + SA-sign subset (30 + 30) at A2
  Arm A  P1-raw, P2, P3 (60 each) at A0/A1/A2; P1-mined (60) at A0; P2 cross-narration (60) at A0;
         minimal-prompt subsample (45) at A0
  k = 3 samples at T = 0.7  ->  3,555 narrations per model.  An optional T = 0 reference sample per
  (formula, condition) is planned separately and reported apart from the primary k = 3.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass

from configs import study


@dataclass(frozen=True)
class Cell:
    model: str
    formula_id: str
    access: str
    variant: str
    sample_idx: int
    temperature: float
    arm: str
    pool: str
    cross: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _by_pool(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        key = r["pool"]
        if r["pool"] == "SA":
            key = f"SA_{r['perturbation']['type'].replace('sa_', '')}"
        if r["pool"] == "P1":
            key = f"P1_{r.get('stratum') or 'raw'}"
        out.setdefault(key, []).append(r)
    return out


def plan_cells(records: list[dict], narrators: list[dict], k: int | None = None, seed: int | None = None,
               reference_t0: bool = False, novel_allowed=lambda m: True) -> list[Cell]:
    st = study()
    k = k or st["arms"]["k_samples"]
    T = float(st["arms"]["temperature_primary"])
    seed = st["seed"] if seed is None else seed
    pools = _by_pool(records)
    cells: list[Cell] = []
    fam_of = {m["id"]: m["family"] for m in narrators}
    for mi, m in enumerate(narrators):
        mid = m["id"]
        rng = random.Random(f"{seed}-{mid}")

        def add(recs, accesses, variant="guided", cross=False):
            for r in recs:
                for a in accesses:
                    for s in range(k):
                        cells.append(Cell(mid, r["formula_id"], a, variant, s, T, r["arm"], r["pool"], cross))
                    if reference_t0:
                        cells.append(Cell(mid, r["formula_id"], a, variant, -1, 0.0, r["arm"], r["pool"], cross))

        # ---------------- Arm B (shared formulas)
        for key in ("K", "SP", "SA_sign", "SA_window", "SA_field", "K_named", "NL"):
            add(pools.get(key, []), ["A0"])
        if novel_allowed(m):
            add(pools.get("N", []), ["A0"])
        a2k = [r for r in pools.get("K", []) if r.get("meta", {}).get("a2_subset")]
        a2s = [r for r in pools.get("SA_sign", []) if r.get("meta", {}).get("a2_subset")]
        add(a2k + a2s, ["A2"])
        # ---------------- Arm A (P1/P2 authored by this model; P3 shared)
        own = lambda key: [r for r in pools.get(key, []) if r.get("author_model") in (None, mid)]
        p1raw, p2, p3 = own("P1_raw"), own("P2"), pools.get("P3a", []) + pools.get("P3b", [])
        add(p1raw, ["A0", "A1", "A2"])
        add(own("P1_mined"), ["A0"])
        add(p2, ["A0", "A1", "A2"])
        add(p3, ["A0", "A1", "A2"])
        add(pools.get("P4", []), ["A0"])
        # cross-narration: P2 formulas authored by another family
        others = [r for r in pools.get("P2", []) if r.get("author_model") not in (None, mid)
                  and fam_of.get(r.get("author_model")) != m["family"]]
        rng.shuffle(others)
        add(others[: len(p2) or len(others)], ["A0"], cross=True)
        # minimal-prompt subsample (25% of Arm-A formulas, rotated across models: Latin-square blocks)
        block = mi % 4
        for recs in (p1raw, p2, p3):
            n = len(recs)
            q = max(1, n // 4) if n else 0
            add(recs[block * q:(block + 1) * q], ["A0"], variant="minimal")
    rng = random.Random(seed)
    rng.shuffle(cells)
    return cells


def plan_summary(cells: list[Cell]) -> dict:
    per_model = Counter(c.model for c in cells if c.sample_idx >= 0)
    per_pool = Counter((c.pool, c.access, c.variant) for c in cells if c.sample_idx >= 0)
    return {"per_model": dict(per_model), "per_pool_access_variant": {"|".join(k): v for k, v in per_pool.items()},
            "t0_reference": sum(1 for c in cells if c.sample_idx < 0)}


def plan_pilot_cells(records: list[dict], narrators: list[dict], n: int | None = None, k: int | None = None,
                     extra_pools: tuple = ("NL",), seed: int | None = None, novel_allowed=lambda m: True) -> list[Cell]:
    """§17.1 phase-1 pilot: Arm B K, SP, SA-sign, N x n formulas at A0; Arm A P1, P3 x n at A0/A2; k samples.

    Arm-B formulas share base formulas across pools (paired contrasts); ``extra_pools`` adds pools beyond
    the §17.1 list (default NL, so that every confirmatory contrast can be estimated in the pilot).
    """
    st = study()
    n = n or st["pilot"]["formulas_per_cell"]
    k = k or st["arms"]["k_samples"]
    T = float(st["arms"]["temperature_primary"])
    seed = st["seed"] if seed is None else seed
    pools = _by_pool(records)
    keys = ["K", "SP", "SA_sign", *extra_pools]
    bases = [r["base_id"] for r in pools.get("K", [])]
    shared = [b for b in bases if all(any(r.get("base_id") == b for r in pools.get(p, [])) for p in keys)]
    rest = [b for b in bases if b not in shared]
    chosen = (shared + rest)[:n]
    cells: list[Cell] = []
    for m in narrators:
        mid = m["id"]

        def add(recs, accesses):
            for r in recs:
                for a in accesses:
                    for s in range(k):
                        cells.append(Cell(mid, r["formula_id"], a, "guided", s, T, r["arm"], r["pool"]))

        for key in keys:
            recs = pools.get(key, [])
            add([next(r for r in recs if r.get("base_id") == b) for b in chosen
                 if any(r.get("base_id") == b for r in recs)], ["A0"])
        if novel_allowed(m):
            add(pools.get("N", [])[:n], ["A0"])
        own = [r for r in pools.get("P1_raw", []) if r.get("author_model") in (None, mid)][:n]
        p3 = pools.get("P3a", [])[: (n + 1) // 2] + pools.get("P3b", [])[: n // 2]
        add(own, ["A0", "A2"])
        add(p3, ["A0", "A2"])
    random.Random(seed).shuffle(cells)
    return cells
