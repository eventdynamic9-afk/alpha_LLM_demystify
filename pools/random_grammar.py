"""P3b random-grammar pool and complexity matching (§7.1, §7.2).

Trees are sampled from the DSL grammar and accepted by stratified sampling on node-count x depth
bins so that the pool matches a target complexity distribution (P1/P2 formulas for Arm A, the
public base set for the Novel pool)."""
from __future__ import annotations

import random
from collections import Counter

from dsl import Node, descriptors
from dsl.complexity import complexity_bin
from dsl.random_trees import random_tree


def bin_quota(target_nodes: list[Node], n: int) -> Counter:
    """Allocate n draws over (node bin, depth bin) proportional to the target distribution."""
    bins = Counter(complexity_bin(descriptors(t)) for t in target_nodes)
    total = sum(bins.values())
    quota = Counter({b: int(round(n * c / total)) for b, c in bins.items()})
    diff = n - sum(quota.values())
    for b, _ in bins.most_common():
        if diff == 0:
            break
        quota[b] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
    return +quota


def matched_random_trees(target_nodes: list[Node], n: int, accept, seed: int = 0, max_draws: int = 200000,
                         max_depth_range=(2, 8)) -> list[tuple[Node, dict]]:
    """Draw random trees until each complexity bin quota is filled with trees passing ``accept``.

    ``accept(node) -> dict | None`` returns a validity report (kept) or None (rejected).
    """
    rng = random.Random(seed)
    quota = bin_quota(target_nodes, n)
    got: list[tuple[Node, dict]] = []
    filled = Counter()
    draws = 0
    while sum(filled.values()) < sum(quota.values()) and draws < max_draws:
        draws += 1
        t = random_tree(rng, max_depth=rng.randint(*max_depth_range))
        b = complexity_bin(descriptors(t))
        if filled[b] >= quota.get(b, 0):
            continue
        rep = accept(t)
        if rep is None:
            continue
        got.append((t, {**rep, "complexity_bin": list(b), "draw": draws}))
        filled[b] += 1
    return got
