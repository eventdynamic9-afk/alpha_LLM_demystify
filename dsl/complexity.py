"""Complexity descriptors (§6.6) used for pool matching and as analysis covariates."""
from __future__ import annotations

from .ast import Node, depth, size, walk
from .operators import OPS, TS, XS, warmup
from .units import analyze_units


def effective_lookback(node: Node) -> int:
    """Max over root-to-leaf paths of the summed lags d and (window n - 1) of operators on the path."""
    if node.is_leaf:
        return 0
    return warmup(node) + max((effective_lookback(c) for c in node.children), default=0)


def descriptors(node: Node) -> dict:
    ops = [n for n in walk(node) if not n.is_leaf]
    return {
        "nodes": size(node),
        "depth": depth(node),
        "fields": sorted({n.name for n in walk(node) if n.is_field}),
        "window_params": sum(1 for n in ops if OPS[n.op].kind == TS),
        "max_lookback": effective_lookback(node),
        "xsec_ops": sum(1 for n in ops if OPS[n.op].kind == XS),
        "nonmonotone_ops": sum(1 for n in ops if OPS[n.op].nonmonotone),
        "dim_consistent": analyze_units(node).consistent,
    }


def complexity_bin(desc: dict, node_edges=(5, 9, 15, 25), depth_edges=(3, 5, 7)) -> tuple[int, int]:
    """Node-count x depth bin used for stratified complexity matching (§7.1 P3b, §7.2 N)."""
    nb = sum(desc["nodes"] > e for e in node_edges)
    db = sum(desc["depth"] > e for e in depth_edges)
    return nb, db
