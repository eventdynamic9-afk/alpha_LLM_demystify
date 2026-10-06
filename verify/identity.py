"""Identity / provenance claims (C6) and numerical equivalence (§6.5, §10.2)."""
from __future__ import annotations

import numpy as np

from dsl import Node, canonical_equal, canonicalize, parse, walk
from pools.library import get as lib_get
from pools.library import library_id_status

from .stats import daily_spearman
from .verdicts import REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict


def numerically_equivalent(a: np.ndarray, b: np.ndarray, rho: float = 0.999, share: float = 0.99,
                           min_n: int = 10) -> dict:
    """Exact-equal, or daily cross-sectional rank correlation >= rho on >= share of dates."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    if np.array_equal(fa, fb) and np.array_equal(a[fa], b[fb]):
        return {"equivalent": True, "rule": "exact"}
    r = daily_spearman(a, b, min_n=min_n)
    ok = np.isfinite(r)
    if ok.sum() == 0:
        return {"equivalent": False, "rule": "no_overlap"}
    sh = float((r[ok] >= rho).mean())
    return {"equivalent": sh >= share, "rule": "rank_corr", "share_dates": sh, "median_rho": float(np.median(r[ok]))}


def _equiv_cfg(ctx) -> tuple[float, float]:
    c = ctx.thr["identity"]
    return float(c["equivalence_rho"]), float(c["equivalence_date_share"])


def verify_identity(node: Node, ctx, library_id: str) -> Verdict:
    status = library_id_status(library_id)
    if status == "nonexistent":
        return Verdict(REFUTED, "identity", {"reason": f"{library_id} does not exist in the named library"})
    if status != "available":
        return Verdict(UNVERIFIABLE, "identity", {"reason": f"{library_id} is {status} on the OHLCV panel"})
    lf = lib_get(library_id)
    if canonical_equal(node, lf.node):
        return Verdict(SUPPORTED, "identity", {"rule": "canonical", "library_id": library_id})
    rho, share = _equiv_cfg(ctx)
    eq = numerically_equivalent(ctx.signal(node), ctx.signal(lf.node), rho, share)
    return Verdict(SUPPORTED if eq["equivalent"] else REFUTED, "identity", {"library_id": library_id, **eq})


def _contains_subtree(node: Node, template: Node) -> bool:
    t = canonicalize(template)
    return any(n == t for n in walk(canonicalize(node)))


def verify_variant_of(node: Node, ctx, template: str) -> Verdict:
    tmpls = ctx.cb.get("variant_templates", {})
    key = template.upper().replace(" ", "_").replace("%", "")
    spec = tmpls.get(key)
    if spec is None:
        for k, v in tmpls.items():
            if template.lower() in v.get("words", []):
                key, spec = k, v
                break
    if spec is None:
        return Verdict(UNVERIFIABLE, "identity_variant", {"reason": f"no template for {template!r}"})
    tnode = parse(spec["dsl"])
    if _contains_subtree(node, tnode):
        return Verdict(SUPPORTED, "identity_variant", {"template": key, "rule": "canonical_subtree"})
    rows = ctx.rows("train")
    r = daily_spearman(ctx.signal(node), ctx.signal(tnode), rows)
    rbar = float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")
    c = ctx.thr["identity"]
    ev = {"template": key, "rho_bar": rbar}
    if not np.isfinite(rbar):
        return Verdict(UNVERIFIABLE, "identity_variant", ev)
    if abs(rbar) >= c["variant_supported_rho"]:
        return Verdict(SUPPORTED, "identity_variant", ev)
    if abs(rbar) <= c["variant_refuted_rho"]:
        return Verdict(REFUTED, "identity_variant", ev)
    return Verdict(UNRESOLVED, "identity_variant", ev)


def identity_matches(node: Node, ctx, candidates=None) -> list[str]:
    """Library ids the formula is equivalent to (used by NOVEL refutation and novelty checks)."""
    from pools.library import library

    rho, share = _equiv_cfg(ctx)
    sig = ctx.signal(node)
    out = []
    for lid, lf in (candidates or library()).items():
        if canonical_equal(node, lf.node):
            out.append(lid)
            continue
        if numerically_equivalent(sig, ctx.references.signal(lid), rho, share)["equivalent"]:
            out.append(lid)
    return out
