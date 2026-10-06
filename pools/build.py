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
from verify.verdicts import REFUTED, SUPPORTED
from verify.identity import numerically_equivalent

from .gp import GPConfig, run_gp
from .library import LibraryFormula, base_candidates, base_set, complexity_terciles, library
from .novelty import novelty_checks
from .perturb import SP_TYPES, k_named_label, nl_label, sa_field, sa_sign, sa_window, sp_variant
from .random_grammar import bin_quota, matched_random_trees
from .records import FormulaRecord, now_iso, write_jsonl
from .validity import PoolDeduper, check_validity

_NOTATION = {"alpha101": "alpha101", "gtja191": "gtja", "alpha158": "qlib"}


def _tercile_labels(formulas: list[LibraryFormula]) -> dict[str, int]:
    """Pooled complexity terciles of the base-set candidates (the strata used to draw the base set)."""
    terc, q = complexity_terciles(base_candidates())
    out = {}
    for f in formulas:
        if f.lib_id in terc:
            out[f.lib_id] = terc[f.lib_id]
        else:
            n = descriptors(f.node)["nodes"]
            out[f.lib_id] = int(n > q[0]) + int(n > q[1])
    return out


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


def parse_presented(text: str, notation: str, legend: dict | None, source_notation: str) -> Node:
    """Parse the exact text shown to the narrator back into a tree (SP validation, §7.2)."""
    import re

    from dsl.math_parser import parse_math

    if notation == "math":
        return parse_math(text)
    if notation == "anonymized":
        inv = {lab: name for name, lab in (legend or {}).items()}
        return parse(re.sub(r"\bx\d+\b", lambda m: f"${inv[m.group(0)]}", text), "qlib")
    if notation == "program":
        return parse(text, "alpha101" if source_notation == "alpha101" else "qlib")
    return parse(text, {"qlib": "qlib", "alpha101": "alpha101", "gtja": "gtja"}[notation])


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
            try:
                check = parse_presented(v["presented"], v["notation"], v["legend"], _NOTATION[f.library])
            except Exception:                     # the presented text must parse back (§7.2 "must be equivalent")
                continue
            if _equivalent(check, f.node, ctx):
                recs.append(FormulaRecord.from_node(f"B-{f.short_id}-SP-{t}", "B", "SP", v["node"], presented=v["presented"],
                                                    notation=v["notation"], base_id=f.short_id, legend=v["legend"],
                                                    perturbation={**v["perturbation"], "validated_equivalent": True,
                                                                  "validated_on": "presented text"},
                                                    stratum=f"t{terc[f.lib_id]}", seed=seed + i,
                                                    meta={"library": f.library, "lib_id": f.lib_id}))
                ok = True
                break
        if not ok:
            sp_fail.append(f.lib_id)
    report["sp_failures"] = sp_fail

    # ---------------- SA (sign / window / field): one code-generated variant per base formula, kept only when
    # the verifier confirms that the targeted property changed (§7.2); unconfirmed variants are dropped and
    # counted as shortfall.
    from dsl.monotonicity import AMB, sign_map
    from dsl.serialize import to_notation
    from verify.stats import daily_spearman, lag1_rank_autocorr

    sa_dedup = {t: PoolDeduper(ctx) for t in ("sign", "window", "field")}
    shortfall, unconfirmed = Counter(), Counter()
    v_rho = float(ctx.thr["identity"]["variant_supported_rho"])
    for i, f in enumerate(base):
        meta = {"library": f.library, "lib_id": f.lib_id}
        notation = _NOTATION[f.library]
        # sign flip: the predictive direction (decided PRED_SIGN verdicts) or, when that is undecided,
        # every input's statically decided direction must flip
        v = sa_sign(f.node)
        rep = check_validity(v, ctx, sa_dedup["sign"], truncation=False)
        placed = False
        if rep["valid"] and not _equivalent(v, f.node, ctx):
            ic_base = verify_claim({"predicate": "PRED_SIGN", "args": {"sign": "+"}}, f.node, ctx).verdict
            ic_var = verify_claim({"predicate": "PRED_SIGN", "args": {"sign": "-"}}, v, ctx).verdict
            pred_flip = ic_base == ic_var and ic_base in (SUPPORTED, REFUTED)
            sb, sv = sign_map(f.node), sign_map(v)
            decided = {k: d for k, d in sb.items() if d != AMB}
            static_flip = bool(decided) and all(sv.get(k) == -d for k, d in decided.items())
            if pred_flip or static_flip:
                rec = FormulaRecord.from_node(f"B-{f.short_id}-SA-sign", "B", "SA", v, presented=to_notation(v, notation),
                                              notation=notation, base_id=f.short_id, validity=rep,
                                              perturbation={"type": "sa_sign", "target_property": "PRED_SIGN / SIGN(all inputs)",
                                                            "expected_change": "direction flips", "base_pred_plus": ic_base,
                                                            "variant_pred_minus": ic_var, "pred_flip": pred_flip,
                                                            "static_flip": static_flip, "confirmed": True},
                                              stratum=f"t{terc[f.lib_id]}", seed=seed, meta=dict(meta))
                if f.lib_id in a2:
                    rec.meta["a2_subset"] = True
                recs.append(rec)
                sa_dedup["sign"].add(rec.formula_id, v, ctx.signal(v))
                placed = True
            else:
                unconfirmed["sign"] += 1
        if not placed:
            shortfall["sign"] += 1
        # window change: LOOKBACK(old) must be SUPPORTED on the base and REFUTED on the variant, and the span
        # must change; the lag-1 turnover change is recorded
        placed = False
        w = sa_window(f.node, random.Random(seed + i))
        if w is not None:
            v, old, new = w
            rep = check_validity(v, ctx, sa_dedup["window"], truncation=False)
            if rep["valid"]:
                lb_base = verify_claim({"predicate": "LOOKBACK", "args": {"window": old}}, f.node, ctx).verdict
                lb_var = verify_claim({"predicate": "LOOKBACK", "args": {"window": old}}, v, ctx).verdict
                hz = {"base": descriptors(f.node)["max_lookback"] + 1, "variant": descriptors(v)["max_lookback"] + 1}
                rows = ctx.rows("train")
                to_b = float(np.nanmean(lag1_rank_autocorr(ctx.signal(f.node), rows)))
                to_v = float(np.nanmean(lag1_rank_autocorr(ctx.signal(v), rows)))
                ok = lb_base == SUPPORTED and lb_var == REFUTED and hz["base"] != hz["variant"]
                if ok:
                    rec = FormulaRecord.from_node(f"B-{f.short_id}-SA-window", "B", "SA", v, presented=to_notation(v, notation),
                                                  notation=notation, base_id=f.short_id, validity=rep,
                                                  perturbation={"type": "sa_window", "target_property": f"LOOKBACK({old}) / HORIZON",
                                                                "expected_change": f"{old} -> {new}", "span_days": hz,
                                                                "base_lookback_verdict": lb_base, "variant_lookback_verdict": lb_var,
                                                                "lag1_rank_autocorr": {"base": round(to_b, 4),
                                                                                       "variant": round(to_v, 4)},
                                                                "confirmed": True},
                                                  stratum=f"t{terc[f.lib_id]}", seed=seed, meta=dict(meta))
                    recs.append(rec)
                    sa_dedup["window"].add(rec.formula_id, v, ctx.signal(v))
                    placed = True
                else:
                    unconfirmed["window"] += 1
        if not placed:
            shortfall["window"] += 1
        # field swap: DEPENDS_ON(new) must be SUPPORTED on the variant, and either DEPENDS_ON(old) REFUTED or the
        # behaviour (resemblance to the base) changed: mean daily rank correlation with the base < variant rho
        placed = False
        for v, old, new in sa_field(f.node, random.Random(seed + i)):
            rep = check_validity(v, ctx, sa_dedup["field"], truncation=False)
            if not rep["valid"] or _equivalent(v, f.node, ctx):
                continue
            dep_old = verify_claim({"predicate": "DEPENDS_ON", "args": {"input": old}}, v, ctx).verdict
            dep_new = verify_claim({"predicate": "DEPENDS_ON", "args": {"input": new}}, v, ctx).verdict
            rows = ctx.rows("train")
            rc = float(np.nanmean(daily_spearman(ctx.signal(v), ctx.signal(f.node), rows)))
            if not (dep_new == SUPPORTED and (dep_old == REFUTED or abs(rc) < v_rho)):
                unconfirmed["field"] += 1
                continue
            rec = FormulaRecord.from_node(f"B-{f.short_id}-SA-field", "B", "SA", v, presented=to_notation(v, notation),
                                          notation=notation, base_id=f.short_id, validity=rep,
                                          perturbation={"type": "sa_field", "target_property": f"DEPENDS_ON({old}) / RESEMBLES",
                                                        "expected_change": f"{old} -> {new}", "variant_depends_old": dep_old,
                                                        "variant_depends_new": dep_new, "rank_corr_with_base": round(rc, 4),
                                                        "confirmed": True},
                                          stratum=f"t{terc[f.lib_id]}", seed=seed, meta=dict(meta))
            recs.append(rec)
            sa_dedup["field"].add(rec.formula_id, v, ctx.signal(v))
            placed = True
            break
        if not placed:
            shortfall["field"] += 1
    report["sa_shortfall"] = dict(shortfall)
    report["sa_unconfirmed_dropped"] = dict(unconfirmed)

    report["_base_nodes"] = [f.node for f in base]
    return recs


def _accept_fn(ctx, dedup: PoolDeduper):
    def accept(node):
        rep = check_validity(node, ctx, dedup, truncation=True)
        if not rep["valid"]:
            return None
        dedup.add(canonical_hash(node), node, ctx.signal(node))
        return rep
    return accept


def build_novel(ctx, target: list[Node], n: int, seed: int, search: bool, report: dict,
                own_pools: list[FormulaRecord] | None = None, allow_pending_search: bool = False) -> list[FormulaRecord]:
    """N pool, built after every other pool so that novelty check 1 covers all of them (§7.2). Without
    completed searches a formula's novelty status is "pending_search"; such formulas are kept only when
    ``allow_pending_search`` (pilot runs without search access), and the report says so."""
    created = now_iso()
    lib_dedup = PoolDeduper(ctx)
    for f in library().values():
        lib_dedup.hashes[canonical_hash(f.node)] = f.lib_id
    own_hashes = set()
    for r in own_pools or []:
        h = canonical_hash(parse(r.dsl))
        own_hashes.add(h)
        lib_dedup.hashes[h] = r.formula_id
    n_rand = n // 2
    rand = matched_random_trees(target, n_rand * 4, _accept_fn(ctx, lib_dedup), seed=seed)
    gp = run_gp(ctx, GPConfig(population=60, generations=5, seed=seed), n_best=n * 6)
    out, own = [], set(own_hashes)
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
        if not (nov["passed"] or (allow_pending_search and nov["status"] == "pending_search")):
            fails["rho" if not nov["rho_ok"] else "canonical" if not nov["canonical_unique"] else "search"] += 1
            continue
        own.add(canonical_hash(node))
        out.append(FormulaRecord.from_node(f"B-N-{len(out):03d}", "B", "N", node, validity=rep, novelty=nov,
                                           seed=seed, created_at=created, meta={"generator": src,
                                                                                "gp_trials": len(gp.trials) if src == "gp" else None}))
    report["novel"] = {"requested": n, "built": len(out), "rejections": dict(fails), "gp_trials": len(gp.trials),
                       "novelty_status": dict(Counter(r.novelty["status"] for r in out)),
                       "own_pool_formulas_checked": len(own_hashes)}
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
              seed: int | None = None, protocols: tuple = ("P1_raw", "P1_mined", "P2"), n_arm_a: int | None = None,
              allow_pending_search: bool = False) -> dict:
    out_dir = Path(out_dir)
    report: dict = {"created_at": now_iso(), "scale": scale}
    recs = build_arm_b(ctx, seed, scale, search, report)
    target = [parse(r.dsl) for r in recs if r.pool == "K"]
    a_recs = []
    if authors:
        from .llm_authors import author_p1, author_p2

        n = n_arm_a or max(2, int(round(60 * scale)))
        for client in authors:
            if "P1_raw" in protocols:
                a_recs += author_p1(client, ctx, n, mined=False)
            if "P1_mined" in protocols:
                a_recs += author_p1(client, ctx, n, mined=True)
            if "P2" in protocols:
                a_recs += author_p2(client, ctx, n)
        if a_recs:
            target = [parse(r.dsl) for r in a_recs if r.pool in ("P1", "P2")]
        report["arm_a_authored"] = dict(Counter(f"{r.pool}_{r.stratum or ''}|{r.author_model}" for r in a_recs))
    a_recs += build_p3(ctx, target, n_arm_a or max(2, int(round(60 * scale))), report=report)
    # N last: generated now (after the latest narrator cutoff), complexity matched to the base set, and
    # canonically distinct from every formula in every other pool
    n_novel = max(2, int(round(study()["arms"]["arm_b"]["pools"]["N"]["formulas"] * scale)))
    seed_n = (study()["seed"] if seed is None else seed) + 7
    n_recs = build_novel(ctx, report.pop("_base_nodes"), n_novel, seed_n, search, report, own_pools=recs + a_recs,
                         allow_pending_search=allow_pending_search)
    allr = recs + a_recs + n_recs
    write_jsonl(allr, out_dir / "formulas.jsonl")
    report["counts"] = dict(Counter(r.pool if r.pool != "SA" else f"SA_{r.perturbation['type'][3:]}" for r in allr))
    (out_dir / "pools_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report
