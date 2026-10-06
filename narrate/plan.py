"""Run plan (§3.5, §7.3): which (model, formula, access, prompt variant, sample) cells are narrated.

Per narrator model (main study):
  Arm B  K, SP, N (60 each), SA (180), K_named (30), NL (30) at A0; K + SA-sign subset (30 + 30) at A2
  Arm A  P1-raw, P2, P3 (60 each) at A0/A1/A2; P1-mined (60) at A0; P2 cross-narration (60) at A0;
         minimal-prompt subsample (45) at A0
  k = 3 samples at T = 0.7  ->  3,555 narrations per model, plus one T = 0 reference sample per
  (formula, condition) (§8.3; on by default, sample_idx = -1, reported apart from the primary k = 3).

Cross-narration (§4.2, §7.1 P2) is assigned globally: every P2 formula gets exactly one narrator from a
different family, balanced so each narrator receives as many as it authored (max-flow over authors x
narrators; any remainder goes to the least-loaded eligible narrator).

§14 ablations (secondary, exploratory, opt-in): ``structured``, ``cap150`` and ``reasoning_effort`` add
A0 cells for a Latin-square-rotated quarter of each model's Arm-A formulas (no T = 0 reference).
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass

from configs import study

ABLATIONS = ("structured", "cap150", "reasoning_effort")


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
    reasoning_effort: str | None = None      # §14 ablation only; None = the model's recorded default

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def rationale_id(self) -> str:
        return f"R-{self.model}-{self.formula_id}-{self.access}-{self.variant}-{self.sample_idx}"


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


def assign_cross_narrators(p2: list[dict], narrators: list[dict], seed: int) -> dict[str, list[dict]]:
    """§4.2: exactly one other-family narrator per authored P2 formula, balanced so that each narrator
    receives as many formulas as it authored.  Returns {narrator id: [P2 records]}."""
    fam = {m["id"]: m["family"] for m in narrators}
    nids = [m["id"] for m in narrators]
    by_author: dict[str, list[dict]] = defaultdict(list)
    for r in p2:
        if r.get("author_model") is not None:
            by_author[r["author_model"]].append(r)
    authors = sorted(by_author)
    eligible = {a: [n for n in nids if fam[n] != fam.get(a)] for a in authors}
    quota = {n: len(by_author.get(n, [])) for n in nids}
    rng = random.Random(f"{seed}-cross")
    flow: dict[tuple[str, str], int] = defaultdict(int)
    supply = {a: len(by_author[a]) for a in authors}
    recv = {n: 0 for n in nids}

    def augment() -> bool:                        # one unit along a shortest residual path S->a->(n->a'->)*n->T
        prev: dict = {}
        q = deque()
        for a in authors:
            if supply[a] > sum(flow[(a, n)] for n in nids):
                prev[("a", a)] = None
                q.append(("a", a))
        while q:
            kind, x = q.popleft()
            if kind == "a":
                nbrs = sorted(eligible[x], key=lambda n: (flow[(x, n)], rng.random()))
                for n in nbrs:
                    if ("n", n) not in prev:
                        prev[("n", n)] = (kind, x)
                        if recv[n] < quota[n]:
                            node = ("n", n)
                            while prev[node] is not None:
                                pk, px = prev[node]
                                if node[0] == "n":
                                    flow[(px, node[1])] += 1
                                else:
                                    flow[(node[1], px)] -= 1
                                node = (pk, px)
                            recv[n] += 1
                            return True
                        q.append(("n", n))
            else:
                for a in authors:
                    if flow[(a, x)] > 0 and ("a", a) not in prev:
                        prev[("a", a)] = (kind, x)
                        q.append(("a", a))
        return False

    while augment():
        pass
    out: dict[str, list[dict]] = defaultdict(list)
    for a in authors:
        recs = sorted(by_author[a], key=lambda r: r["formula_id"])
        rng.shuffle(recs)
        i = 0
        for n in sorted(eligible[a]):
            out[n] += recs[i:i + flow[(a, n)]]
            i += flow[(a, n)]
        for r in recs[i:]:                         # quotas infeasible: least-loaded eligible narrator
            if eligible[a]:
                n = min(eligible[a], key=lambda n: (len(out[n]) - quota[n], len(out[n]), n))
                out[n].append(r)
    return dict(out)


def plan_cells(records: list[dict], narrators: list[dict], k: int | None = None, seed: int | None = None,
               reference_t0: bool | None = None, novel_allowed=lambda m: True, ablations: tuple | list = ()) -> list[Cell]:
    """Main-study plan (§7.3).  ``reference_t0`` defaults to ``study.yaml`` ``arms.reference_t0`` (true, §8.3)."""
    st = study()
    reference_t0 = st["arms"].get("reference_t0", True) if reference_t0 is None else reference_t0
    k = k or st["arms"]["k_samples"]
    T = float(st["arms"]["temperature_primary"])
    T0 = float(st["arms"].get("temperature_reference", 0.0))
    seed = st["seed"] if seed is None else seed
    bad = [x for x in ablations if x not in ABLATIONS]
    if bad:
        raise ValueError(f"unknown ablation(s) {bad}; expected {ABLATIONS}")
    abl = st.get("ablations", {})
    pools = _by_pool(records)
    cells: list[Cell] = []
    cross_of = assign_cross_narrators(pools.get("P2", []), narrators, seed)
    for mi, m in enumerate(narrators):
        mid = m["id"]

        def add(recs, accesses, variant="guided", cross=False, t0=None, effort=None):
            for r in recs:
                for a in accesses:
                    for s in range(k):
                        cells.append(Cell(mid, r["formula_id"], a, variant, s, T, r["arm"], r["pool"], cross, effort))
                    if reference_t0 if t0 is None else t0:
                        cells.append(Cell(mid, r["formula_id"], a, variant, -1, T0, r["arm"], r["pool"], cross, effort))

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
        # cross-narration: P2 formulas authored by another family, assigned globally (one per formula)
        add(sorted(cross_of.get(mid, []), key=lambda r: r["formula_id"]), ["A0"], cross=True)
        # minimal-prompt subsample (25% of Arm-A formulas, rotated across models: Latin-square blocks)
        block = mi % 4
        for recs in (p1raw, p2, p3):
            n = len(recs)
            q = max(1, n // 4) if n else 0
            add(recs[block * q:(block + 1) * q], ["A0"], variant="minimal")
        # §14 ablations: another rotated quarter per ablation, A0, k samples, no T = 0 reference
        for j, name in enumerate(ablations):
            if name == "reasoning_effort":
                levels = [lv for lv in abl.get("reasoning_effort", {}).get("levels", ["low", "high"])
                          if lv in (m.get("reasoning_effort_levels") or [])]
                variants = [(f"guided_effort_{lv}", lv) for lv in levels]
            else:
                variants = [(name, None)]
            nb = max(1, round(1 / float(abl.get("fraction", 0.25))))
            blk = (mi + j + 1) % nb
            for recs in (p1raw, p2, p3):
                n = len(recs)
                q = max(1, n // nb) if n else 0
                for v, eff in variants:
                    add(recs[blk * q:(blk + 1) * q], ["A0"], variant=v, t0=False, effort=eff)
    rng = random.Random(seed)
    rng.shuffle(cells)
    return cells


def cross_narration_check(cells: list[Cell], records: list[dict] | None = None) -> dict:
    """§4.2 invariant: every authored P2 formula has exactly one cross-narrator (distinct models among its
    primary cross cells).  Without ``records`` only P2 formulas that appear in the cells are checked."""
    by_f: dict[str, set] = defaultdict(set)
    p2 = {c.formula_id for c in cells if c.pool == "P2"}
    for c in cells:
        if c.cross and c.sample_idx >= 0:
            by_f[c.formula_id].add(c.model)
    if records is not None:
        p2 = {r["formula_id"] for r in records if r.get("pool") == "P2" and r.get("author_model") is not None}
    counts = Counter(len(by_f.get(f, ())) for f in p2)
    bad = sorted(f for f in p2 if len(by_f.get(f, ())) != 1)
    return {"p2_formulas": len(p2), "cross_narrators_per_formula": {str(k): v for k, v in sorted(counts.items())},
            "ok": not bad, "violations": bad[:20]}


def plan_summary(cells: list[Cell], records: list[dict] | None = None) -> dict:
    per_model = Counter(c.model for c in cells if c.sample_idx >= 0)
    per_pool = Counter((c.pool, c.access, c.variant) for c in cells if c.sample_idx >= 0)
    return {"per_model": dict(per_model), "per_pool_access_variant": {"|".join(k): v for k, v in per_pool.items()},
            "t0_reference": sum(1 for c in cells if c.sample_idx < 0),
            "ablation_cells": sum(1 for c in cells if c.variant not in ("guided", "minimal")),
            "p2_cross_narration": cross_narration_check(cells, records)}


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
