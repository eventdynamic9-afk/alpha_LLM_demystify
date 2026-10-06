"""Common validity filter for every protocol (§7.1):

parses -> passes the AST validator and the dynamic truncation test -> non-degenerate (cross-sectional
std > 0 on >= 95% of dates; coverage >= 80% of member-days after warm-up) -> not equivalent (§6.5) to
another formula already in the same pool.  Formulas are never filtered on performance here.
"""
from __future__ import annotations

import numpy as np

from dsl import Node, canonical_hash, effective_lookback, validate
from executors.causality import truncation_test


def degeneracy_stats(sig: np.ndarray, member: np.ndarray, warmup: int, min_n: int = 5) -> dict:
    s = np.where(member, sig, np.nan)[warmup:]
    m = member[warmup:]
    rows = m.sum(axis=1) >= min_n
    if rows.sum() == 0:
        return {"xs_std_share": 0.0, "coverage": 0.0}
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sd = np.nanstd(s[rows], axis=1)
    xs = float(np.mean(np.nan_to_num(sd) > 0))
    cov = float(np.isfinite(s[rows]).sum() / max(1, m[rows].sum()))
    return {"xs_std_share": xs, "coverage": cov}


class PoolDeduper:
    """Tracks canonical hashes and signals of accepted formulas in one pool."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.hashes: dict[str, str] = {}
        self.signals: list[tuple[str, np.ndarray]] = []

    def duplicate_of(self, node: Node, sig: np.ndarray, numeric: bool = True) -> str | None:
        h = canonical_hash(node)
        if h in self.hashes:
            return self.hashes[h]
        if numeric:
            from verify.identity import numerically_equivalent

            rho = self.ctx.thr["identity"]["equivalence_rho"]
            share = self.ctx.thr["identity"]["equivalence_date_share"]
            for fid, other in self.signals:
                if numerically_equivalent(sig, other, rho, share)["equivalent"]:
                    return fid
        return None

    def add(self, fid: str, node: Node, sig: np.ndarray) -> None:
        self.hashes[canonical_hash(node)] = fid
        self.signals.append((fid, sig))


def check_validity(node: Node, ctx, deduper: PoolDeduper | None = None, truncation: bool = True,
                   numeric_dedup: bool = True) -> dict:
    cfg = ctx.thr["pools"]
    rep = validate(node)
    out = {"ast_ok": rep.ok, "errors": rep.errors}
    if not rep.ok:
        out["valid"] = False
        return out
    if truncation:
        tr = truncation_test(node, ctx.panel, ctx.executor)
        out["truncation_ok"] = tr.ok
        if not tr.ok:
            out["valid"] = False
            return out
    sig = ctx.signal(node)
    out.update(degeneracy_stats(sig, ctx.panel.member, effective_lookback(node)))
    ok = out["xs_std_share"] >= cfg["validity_xs_std_share"] and out["coverage"] >= cfg["validity_coverage"]
    if ok and deduper is not None:
        dup = deduper.duplicate_of(node, sig, numeric_dedup)
        out["duplicate_of"] = dup
        ok = dup is None
    out["valid"] = bool(ok)
    return out
