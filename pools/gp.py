"""P3a genetic programming on the DSL (§7.1): training-window RankIC fitness with a parsimony penalty.

Every evaluated candidate is logged (formula, fitness, generation) so the number of trials behind any
reported formula is known for Deflated Sharpe Ratios and PBO (§10.4)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from dsl import Node, canonical_hash, to_qlib, validate
from dsl.ast import size
from dsl.ast import paths, replace_at, subtree
from dsl.random_trees import random_tree
from verify.stats import daily_spearman


@dataclass
class GPConfig:
    population: int = 60
    generations: int = 8
    tournament: int = 4
    p_crossover: float = 0.6
    p_mutation: float = 0.3
    max_depth: int = 6
    max_nodes: int = 30
    parsimony: float = 0.0005
    horizon: int = 1
    seed: int = 0


@dataclass
class GPResult:
    best: list[tuple[Node, float]]
    trials: list[dict] = field(default_factory=list)


def fitness_fn(ctx, horizon: int = 1, window: str = "train"):
    rows = ctx.rows(window)
    fwd = ctx.fwd(horizon, "open_t+1")

    def fit(node: Node) -> float:
        if not validate(node).ok or size(node) > 60:
            return -np.inf
        sig = ctx.executor.evaluate(node, ctx.panel)
        ic = daily_spearman(sig, fwd, rows)
        if np.isfinite(ic).sum() < 20:
            return -np.inf
        return float(abs(np.nanmean(ic)))

    return fit


def _crossover(a: Node, b: Node, rng: random.Random) -> Node:
    pa = [p for p, n in paths(a)]
    pb = [p for p, n in paths(b)]
    return replace_at(a, rng.choice(pa), subtree(b, rng.choice(pb)))


def _mutate(a: Node, rng: random.Random, max_depth: int) -> Node:
    pa = [p for p, n in paths(a)]
    return replace_at(a, rng.choice(pa), random_tree(rng, max_depth=rng.randint(1, max(2, max_depth // 2))))


def run_gp(ctx, cfg: GPConfig | None = None, n_best: int = 30) -> GPResult:
    cfg = cfg or GPConfig()
    rng = random.Random(cfg.seed)
    fit = fitness_fn(ctx, cfg.horizon)
    cache: dict[str, float] = {}
    trials: list[dict] = []

    def score(n: Node, gen: int) -> float:
        h = canonical_hash(n)
        if h not in cache:
            raw = fit(n)
            cache[h] = raw - cfg.parsimony * size(n) if np.isfinite(raw) else -np.inf
            trials.append({"formula": to_qlib(n), "fitness": cache[h], "raw_ic": raw, "generation": gen})
        return cache[h]

    pop = [random_tree(rng, max_depth=rng.randint(2, cfg.max_depth)) for _ in range(cfg.population)]
    scores = [score(n, 0) for n in pop]
    for gen in range(1, cfg.generations + 1):
        new = []
        while len(new) < cfg.population:
            def pick():
                cand = rng.sample(range(len(pop)), cfg.tournament)
                return pop[max(cand, key=lambda i: scores[i])]

            r = rng.random()
            if r < cfg.p_crossover:
                child = _crossover(pick(), pick(), rng)
            elif r < cfg.p_crossover + cfg.p_mutation:
                child = _mutate(pick(), rng, cfg.max_depth)
            else:
                child = pick()
            if size(child) <= cfg.max_nodes and validate(child).ok:
                new.append(child)
        pop = new
        scores = [score(n, gen) for n in pop]
    ranked = sorted(((canonical_hash(n), n, s) for n, s in zip(pop, scores)), key=lambda x: -x[2])
    best, seen = [], set()
    # best distinct individuals across the whole run
    allc = {}
    for t in trials:
        allc[t["formula"]] = t["fitness"]
    for h, n, s in ranked:
        if h not in seen and np.isfinite(s):
            seen.add(h)
            best.append((n, s))
    if len(best) < n_best:
        from dsl import parse

        for f, s in sorted(allc.items(), key=lambda kv: -kv[1]):
            n = parse(f)
            h = canonical_hash(n)
            if h not in seen and np.isfinite(s):
                seen.add(h)
                best.append((n, s))
            if len(best) >= n_best:
                break
    return GPResult(best[:n_best], trials)
