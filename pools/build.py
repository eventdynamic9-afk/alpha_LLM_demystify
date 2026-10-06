"""Build every formula pool and write ``formulas.jsonl`` (Appendix D records) plus a build report.

Arm B (§7.2): base set of 60 public formulas (20 Alpha101 / 20 GTJA-191 / 20 Alpha158) stratified by
complexity tercile; pools K, K_named (30), SP (60, four SP types), SA (sign / window / field, 60 each),
NL (30), N (60: random grammar + GP, generated after the latest narrator cutoff, complexity-matched,
novelty-checked) and the A2 subset (30 K + their SA-sign variants).
Arm A (§7.1): P3a GP + P3b random grammar (complexity matched to P1/P2 when present, else to the base
set); P1/P2 come from :mod:`pools.llm_authors` when author clients are configured.

Base-set formulas that fail the validity filter on the study panel (e.g. a degenerate constant) are
replaced by a reserve formula from the same library, nearest in complexity, and the swap is reported.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from configs import study
from dsl import Node, canonical_hash, descriptors, parse
from verify.dispatcher import verify_claim
from verify.identity import numerically_equivalent

from .gp import GPConfig, run_gp
from .library import LibraryFormula, base_set, library
from .novelty import novelty_checks
from .perturb import SP_TYPES, k_named_label, nl_label, sa_field, sa_sign, sa_window, sp_variant
from .random_grammar import bin_quota, matched_random_trees
from .records import FormulaRecord, now_iso, write_jsonl
from .validity import PoolDeduper, check_validity

_NOTATION = {"alpha101": "alpha101", "gtja191": "gtja", "alpha158": "qlib"}


def _tercile_labels(formulas: list[LibraryFormula]) -> dict[str, int]:
    sizes = np.array([descriptors(f.node)["nodes"] for f in formulas])
    q = np.quantile(sizes, [1 / 3, 2 / 3])
    return {f.lib_id: int((s > q[0]) + (s > q[1])) for f, s in zip(formulas, sizes)}


def validated_base_set(ctx, report: dict) -> list[LibraryFormula]:
    base = base_set()
    chosen, swaps = [], []
    used = {f.lib_id for f in base}
    dedup = PoolDeduper(ctx)
    for f in base:
        rep = check_validity(f.node, ctx, dedup, truncation=False)
        if rep["valid"]:
            chosen.append(f)
            dedup.add(f.lib_id, f.node, ctx.signal(f.node))
            continue
        size = descriptors(f.node)["nodes"]
        reserves = sorted((r for r in library().values() if r.library == f.library and r.lib_id not in used),
                          key=lambda r: abs(descriptors(r.node)["nodes"] - size))
        for r in reserves:
            rr = check_validity(r.node, ctx, dedup, truncation=False)
            if rr["valid"]:
                chosen.append(r)
                used.add(r.lib_id)
                dedup.add(r.lib_id, r.node, ctx.signal(r.node))
                swaps.append({"dropped": f.lib_id, "reason": {k: rep.get(k) for k in ("xs_std_share", "coverage", "duplicate_of")},
                              "replacement": r.lib_id})
                break
    report["base_swaps"] = swaps
    return chosen


def _stratified_subset(formulas: list[LibraryFormula], terc: dict, n: int, rng: random.Random, offset: int = 0) -> list[LibraryFormula]:
    cells: dict[tuple, list] = {}
    for f in formulas:
        cells.setdefault((f.library, terc[f.lib_id]), []).append(f)
    out = []
    keys = sorted(cells)
    for k in keys:
        rng.shuffle(cells[k])
    i = offset
    while len(out) < min(n, len(formulas)):
        k = keys[i % len(keys)]
        pool = [f for f in cells[k] if f not in out]
        if pool:
            out.append(pool[(i // len(keys)) % len(pool)])
        i += 1
        if i > offset + 50 * len(formulas):
            break
    return out


def _equivalent(a: Node, b: Node, ctx) -> bool:
    c = ctx.thr["identity"]
    return numerically_equivalent(ctx.signal(a), ctx.signal(b), c["equivalence_rho"], c["equivalence_date_share"])["equivalent"]


def build_arm_b(ctx, seed: int | None = None, scale: float = 1.0, search: bool = False, report: dict | None = None) -> list[FormulaRecord]:
    st = study()
    seed = st["seed"] if seed is None else seed
    rng = random.Random(seed)
    report = report if report is not None else {}
    base = validated_base_set(ctx, report)
    n_base = max(3, int(round(len(base) * scale)))
    if n_base < len(base):
        terc0 = _tercile_labels(base)
        base = _stratified_subset(base, terc0, n_base, rng)
    terc = _tercile_labels(base)
    report["base_terciles"] = dict(Counter(f"{f.library}|t{terc[f.lib_id]}" for f in base))
    recs: list[FormulaRecord] = []
    half = max(1, len(base) // 2)

    # ---------------- K
    for f in base:
        recs.append(FormulaRecord.from_node(f"B-{f.short_id}-K", "B", "K", f.node, presented=f.source_text,
                                            notation=_NOTATION[f.library], base_id=f.short_id, stratum=f"t{terc[f.lib_id]}",
                                            seed=seed, meta={"library": f.library, "lib_id": f.lib_id}))
    # ---------------- K_named and NL (complementary stratified halves) and the A2 subset
    named = _stratified_subset(base, terc, half, rng)
    nl_bases = [f for f in base if f not in named] or named
    a2 = set(f.lib_id for f in _stratified_subset(base, terc, half, rng, offset=1))
    for r in recs:
        if r.meta["lib_id"] in a2:
            r.meta["a2_subset"] = True
    for f in named:
        recs.append(FormulaRecord.from_node(f"B-{f.short_id}-Knamed", "B", "K_named", f.node, presented=f.source_text,
                                            notation=_NOTATION[f.library], base_id=f.short_id,
                                            label=k_named_label(f.display_name), stratum=f"t{terc[f.lib_id]}", seed=seed,
                                            meta={"library": f.library, "lib_id": f.lib_id}))
    nl_fail = []
    for f in nl_bases[:half]:
        res = nl_label(f.node, ctx, rng)
        if res is None:
            nl_fail.append(f.lib_id)
            continue
        label, claim, verdict, terms = res
        recs.append(FormulaRecord.from_node(f"B-{f.short_id}-NL", "B", "NL", f.node, presented=f.source_text,
                                            notation=_NOTATION[f.library], base_id=f.short_id, label=label,
                                            perturbation={"type": "misleading_label", "target_property": claim,
                                                          "label_terms": terms, "expected_change": "label property REFUTED",
                                                          "verdict": verdict["verdict"]},
                                            stratum=f"t{terc[f.lib_id]}", seed=seed,
                                            meta={"library": f.library, "lib_id": f.lib_id}))
    report["nl_without_refuted_label"] = nl_fail

    # ---------------- SP (four types, balanced over the base set)
    order = list(base)
    rng.shuffle(order)
    sp_fail = []
    for i, f in enumerate(order):
        ok = False
        for j in range(len(SP_TYPES)):
            t = SP_TYPES[(i + j) % len(SP_TYPES)]
            v = sp_variant(f.node, f.library, t, seed + i)
            if t == "intermediates":
                check = parse(v["presented"], "alpha101" if _NOTATION[f.library] == "alpha101" else "qlib")
            else:
                check = v["node"]
            if _equivalent(check, f.node, ctx):
                recs.append(FormulaRecord.from_node(f"B-{f.short_id}-SP-{t}", "B", "SP", v["node"], presented=v["presented"],
                                                    notation=v["notation"], base_id=f.short_id, legend=v["legend"],
                                                    perturbation={**v["perturbation"], "validated_equivalent": True},
                                                    stratum=f"t{terc[f.lib_id]}", seed=seed + i,
                                                    meta={"library": f.library, "lib_id": f.lib_id}))
                ok = True
                break
        if not ok:
            sp_fail.append(f.lib_id)
    report["sp_failures"] = sp_fail

    # ---------------- SA (sign / window / field)
    sa_dedup = {t: PoolDeduper(ctx) for t in ("sign", "window", "field")}
    shortfall = Counter()
    for i, f in enumerate(base):
        meta = {"library": f.library, "lib_id": f.lib_id}
        notation = _NOTATION[f.library]
        from dsl.serialize import to_notation

        # sign flip: predictive direction must flip
        v = sa_sign(f.node)
        rep = check_validity(v, ctx, sa_dedup["sign"], truncation=False)
        if rep["valid"] and not _equivalent(v, f.node, ctx):
            ic_base = verify_claim({"predicate": "PRED_SIGN", "args": {"sign": "+"}}, f.node, ctx).to_dict()
            ic_var = verify_claim({"predicate": "PRED_SIGN", "args": {"sign": "-"}}, v, ctx).to_dict()
            rec = FormulaRecord.from_node(f"B-{f.short_id}-SA-sign", "B", "SA", v, presented=to_notation(v, notation),
                                          notation=notation, base_id=f.short_id, validity=rep,
                                          perturbation={"type": "sa_sign", "target_property": "PRED_SIGN / SIGN(all inputs)",
                                                        "expected_change": "direction flips", "base_pred_plus": ic_base["verdict"],
                                                        "variant_pred_minus": ic_var["verdict"],
                                                        "confirmed": ic_base["verdict"] == ic_var["verdict"]},
                                          stratum=f"t{terc[f.lib_id]}", seed=seed, meta=dict(meta))
            if f.lib_id in a2:
                rec.meta["a2_subset"] = True
            recs.append(rec)
            sa_dedup["sign"].add(rec.formula_id, v, ctx.signal(v))
        else:
            shortfall["sign"] += 1
        # window change: lookback must change
        w = sa_window(f.node, random.Random(seed + i))
        if w is not None:
            v, old, new = w
            rep = check_validity(v, ctx, sa_dedup["window"], truncation=False)
            if rep["valid"]:
                lb_base = verify_claim({"predicate": "LOOKBACK", "args": {"window": old}}, f.node, ctx).verdict
                lb_var = verify_claim({"predicate": "LOOKBACK", "args": {"window": old}}, v, ctx).verdict
                hz = {"base": descriptors(f.node)["max_lookback"] + 1, "variant": descriptors(v)["max_lookback"] + 1}
                rec = FormulaRecord.from_node(f"B-{f.short_id}-SA-window", "B", "SA", v, presented=to_notation(v, notation),
                                              notation=notation, base_id=f.short_id, validity=rep,
                                              perturbation={"type": "sa_window", "target_property": f"LOOKBACK({old}) / HORIZON",
                                                            "expected_change": f"{old} -> {new}", "span_days": hz,
                                                            "base_lookback_verdict": lb_base, "variant_lookback_verdict": lb_var,
                                                            "confirmed": lb_base == "SUPPORTED" and hz["base"] != hz["variant"]},
                                              stratum=f"t{terc[f.lib_id]}", seed=seed, meta=dict(meta))
                recs.append(rec)
                sa_dedup["window"].add(rec.formula_id, v, ctx.signal(v))
            else:
                shortfall["window"] += 1
        else:
            shortfall["window"] += 1
        # field swap: dependence must change
        placed = False
        for v, old, new in sa_field(f.node, random.Random(seed + i)):
            rep = check_validity(v, ctx, sa_dedup["field"], truncation=False)
            if not rep["valid"] or _equivalent(v, f.node, ctx):
                continue
            dep_var = verify_claim({"predicate": "DEPENDS_ON", "args": {"input": old}}, v, ctx).verdict
            dep_new = verify_claim({"predicate": "DEPENDS_ON", "args": {"input": new}}, v, ctx).verdict
            rec = FormulaRecord.from_node(f"B-{f.short_id}-SA-field", "B", "SA", v, presented=to_notation(v, notation),
                                          notation=notation, base_id=f.short_id, validity=rep,
                                          perturbation={"type": "sa_field", "target_property": f"DEPENDS_ON({old}) / RESEMBLES",
                                                        "expected_change": f"{old} -> {new}", "variant_depends_old": dep_var,
                                                        "variant_depends_new": dep_new, "confirmed": dep_new == "SUPPORTED"},
                                          stratum=f"t{terc[f.lib_id]}", seed=seed, meta=dict(meta))
            recs.append(rec)
            sa_dedup["field"].add(rec.formula_id, v, ctx.signal(v))
            placed = True
            break
        if not placed:
            shortfall["field"] += 1
    # fill SA-window shortfall (formulas without any window) with a different-window variant of other bases
    if shortfall["window"]:
        from dsl.serialize import to_notation

        need = shortfall["window"]
        for attempt in range(1, 6):
            for i, f in enumerate(base):
                if need == 0:
                    break
                w = sa_window(f.node, random.Random(seed + i), rank=attempt)
                if w is None:
                    continue
                v, old_w, new_w = w
                rep = check_validity(v, ctx, sa_dedup["window"], truncation=False)
                if not rep["valid"]:
                    continue
                notation = _NOTATION[f.library]
                fid = f"B-{f.short_id}-SA-window{attempt + 1}"
                recs.append(FormulaRecord.from_node(fid, "B", "SA", v, presented=to_notation(v, notation), notation=notation,
                                                    base_id=f.short_id, validity=rep,
                                                    perturbation={"type": "sa_window", "target_property": f"LOOKBACK({old_w}) / HORIZON",
                                                                  "expected_change": f"{old_w} -> {new_w}", "confirmed": True,
                                                                  "fill_for_shortfall": True},
                                                    stratum=f"t{terc[f.lib_id]}", seed=seed,
                                                    meta={"library": f.library, "lib_id": f.lib_id}))
                sa_dedup["window"].add(fid, v, ctx.signal(v))
                need -= 1
            if need == 0:
                break
        # still short: window variants of reserve formulas from the same libraries (flagged)
        if need:
            used_ids = {f.lib_id for f in base}
            for r in library().values():
                if need == 0:
                    break
                if r.lib_id in used_ids or r.library not in {f.library for f in base}:
                    continue
                w = sa_window(r.node, random.Random(seed))
                if w is None:
                    continue
                v, old_w, new_w = w
                rep = check_validity(v, ctx, sa_dedup["window"], truncation=False)
                if not rep["valid"]:
                    continue
                notation = _NOTATION[r.library]
                fid = f"B-{r.short_id}-SA-window-reserve"
                recs.append(FormulaRecord.from_node(fid, "B", "SA", v, presented=to_notation(v, notation), notation=notation,
                                                    base_id=r.short_id, validity=rep,
                                                    perturbation={"type": "sa_window", "target_property": f"LOOKBACK({old_w}) / HORIZON",
                                                                  "expected_change": f"{old_w} -> {new_w}", "confirmed": True,
                                                                  "fill_for_shortfall": True, "reserve_base": True},
                                                    stratum="reserve", seed=seed, meta={"library": r.library, "lib_id": r.lib_id}))
                sa_dedup["window"].add(fid, v, ctx.signal(v))
                need -= 1
        report["sa_window_filled"] = shortfall["window"] - need
        shortfall["window"] = need
    report["sa_shortfall"] = dict(shortfall)

    # ---------------- N (generated now = after the latest narrator cutoff; complexity matched)
    n_novel = max(2, int(round(st["arms"]["arm_b"]["pools"]["N"]["formulas"] * scale)))
    recs += build_novel(ctx, [f.node for f in base], n_novel, seed + 7, search, report)
    return recs


def _accept_fn(ctx, dedup: PoolDeduper):
    def accept(node):
        rep = check_validity(node, ctx, dedup, truncation=True)
        if not rep["valid"]:
            return None
        dedup.add(canonical_hash(node), node, ctx.signal(node))
        return rep
    return accept


def build_novel(ctx, target: list[Node], n: int, seed: int, search: bool, report: dict) -> list[FormulaRecord]:
    created = now_iso()
    lib_dedup = PoolDeduper(ctx)
    for f in library().values():
        lib_dedup.hashes[canonical_hash(f.node)] = f.lib_id
    n_rand = n // 2
    rand = matched_random_trees(target, n_rand * 4, _accept_fn(ctx, lib_dedup), seed=seed)
    gp = run_gp(ctx, GPConfig(population=60, generations=5, seed=seed), n_best=n * 6)
    out, own = [], set()
    fails = Counter()
    quota = bin_quota(target, n - n_rand)
    from dsl.complexity import complexity_bin

    filled = Counter()
    gp_nodes = []
    for node, fit in gp.best:
        b = complexity_bin(descriptors(node))
        if filled[b] >= quota.get(b, 0):
            continue
        rep = check_validity(node, ctx, lib_dedup)
        if rep["valid"]:
            filled[b] += 1
            gp_nodes.append((node, rep, "gp"))
            lib_dedup.add(canonical_hash(node), node, ctx.signal(node))
    # alternate generators so the pool is half random grammar, half GP while candidates last
    r_list = [(t, rep, "random") for t, rep in rand]
    cands = []
    for k in range(max(len(r_list), len(gp_nodes))):
        if k < len(gp_nodes):
            cands.append(gp_nodes[k])
        if k < len(r_list):
            cands.append(r_list[k])
    for node, rep, src in cands:
        if len(out) >= n:
            break
        nov = novelty_checks(node, ctx, own, search=search, seed=seed, created_at=created)
        if not nov["passed"]:
            fails["rho" if not nov["rho_ok"] else "canonical" if not nov["canonical_unique"] else "search"] += 1
            continue
        own.add(canonical_hash(node))
        out.append(FormulaRecord.from_node(f"B-N-{len(out):03d}", "B", "N", node, validity=rep, novelty=nov,
                                           seed=seed, created_at=created, meta={"generator": src,
                                                                                "gp_trials": len(gp.trials) if src == "gp" else None}))
    report["novel"] = {"requested": n, "built": len(out), "rejections": dict(fails), "gp_trials": len(gp.trials)}
    return out


def build_p3(ctx, target: list[Node], n: int = 60, seed: int = 11, report: dict | None = None) -> list[FormulaRecord]:
    """P3a GP (n/2) + P3b random grammar (n/2), complexity matched to ``target``."""
    report = report if report is not None else {}
    dedup = PoolDeduper(ctx)
    gp = run_gp(ctx, GPConfig(seed=seed), n_best=n * 6)
    recs = []
    from dsl.complexity import complexity_bin

    quota = bin_quota(target, n // 2)
    filled = Counter()
    for node, fit in gp.best:
        b = complexity_bin(descriptors(node))
        if filled[b] >= quota.get(b, 0):
            continue
        rep = check_validity(node, ctx, dedup)
        if rep["valid"]:
            filled[b] += 1
            fid = f"P3a-{len(recs):03d}"
            recs.append(FormulaRecord.from_node(fid, "A", "P3a", node, validity=rep, trials=len(gp.trials), seed=seed,
                                                meta={"gp_fitness": fit}))
            dedup.add(fid, node, ctx.signal(node))
        if sum(filled.values()) >= n // 2:
            break
    if len(recs) < n // 2:                       # bins that GP output cannot match: nearest available, flagged
        for node, fit in gp.best:
            if len(recs) >= n // 2:
                break
            rep = check_validity(node, ctx, dedup)
            if rep["valid"]:
                fid = f"P3a-{len(recs):03d}"
                recs.append(FormulaRecord.from_node(fid, "A", "P3a", node, validity=rep, trials=len(gp.trials), seed=seed,
                                                    meta={"gp_fitness": fit, "bin_relaxed": True}))
                dedup.add(fid, node, ctx.signal(node))
    rand = matched_random_trees(target, n - len(recs), _accept_fn(ctx, dedup), seed=seed + 1)
    for i, (node, rep) in enumerate(rand):
        recs.append(FormulaRecord.from_node(f"P3b-{i:03d}", "A", "P3b", node, validity=rep, seed=seed + 1))
    report["p3"] = {"gp": sum(1 for r in recs if r.pool == "P3a"), "random": sum(1 for r in recs if r.pool == "P3b"),
                    "gp_trials": len(gp.trials)}
    report["gp_trial_log_size"] = len(gp.trials)
    return recs


def build_all(ctx, out_dir: str | Path, scale: float = 1.0, search: bool = False, authors: list | None = None,
              seed: int | None = None) -> dict:
    out_dir = Path(out_dir)
    report: dict = {"created_at": now_iso(), "scale": scale}
    recs = build_arm_b(ctx, seed, scale, search, report)
    target = [parse(r.dsl) for r in recs if r.pool == "K"]
    a_recs = []
    if authors:
        from .llm_authors import author_p1, author_p2

        n = max(2, int(round(60 * scale)))
        for client in authors:
            a_recs += author_p1(client, ctx, n, mined=False)
            a_recs += author_p1(client, ctx, n, mined=True)
            a_recs += author_p2(client, ctx, n)
        if a_recs:
            target = [parse(r.dsl) for r in a_recs if r.pool in ("P1", "P2")]
    a_recs += build_p3(ctx, target, max(2, int(round(60 * scale))), report=report)
    allr = recs + a_recs
    write_jsonl(allr, out_dir / "formulas.jsonl")
    report["counts"] = dict(Counter(r.pool if r.pool != "SA" else f"SA_{r.perturbation['type'][3:]}" for r in allr))
    (out_dir / "pools_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report
