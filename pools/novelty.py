"""Novelty checks for the Novel (N) pool (§7.2) — replaces the unprovable "absent from training data".

1. Canonical form differs from every public-library formula and every formula in our own pools.
2. Max |time-average cross-sectional Spearman rho| with every reference-library signal < 0.7.
3. Normalized-string search (whitespace / variable-name insensitive) returns no hits in GitHub code
   search and in an n-gram index of open pretraining corpora (infini-gram); results are logged, and a
   search that could not run is recorded as "not_run" (never silently treated as "no hit"); a formula
   passes only when every search completed with zero hits, otherwise its status is "pending_search".
4. Generation timestamp and seed are logged; formulas stay private until the study ends and are
   narrated only by local models or endpoints whose terms exclude training on inputs.
5. On release every file embeds a canary GUID (BIG-bench practice), see :mod:`pools.canary`.

Paper wording: "absent from searchable public sources at generation time".
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np

from dsl import Node, canonical_hash, canonicalize, to_qlib
from pools.library import library
from verify.stats import daily_spearman

INFINIGRAM_URL = "https://api.infini-gram.io/"
INFINIGRAM_INDEX = "v4_dolma-v1_7_llama"


def normalized_string(node: Node) -> str:
    """Whitespace- and variable-name-insensitive key: canonical form with fields replaced by F1, F2..."""
    s = to_qlib(canonicalize(node))
    names = []
    for m in re.finditer(r"\$([a-z]+)", s):
        if m.group(1) not in names:
            names.append(m.group(1))
    for i, n in enumerate(names):
        s = s.replace(f"${n}", f"F{i + 1}")
    return re.sub(r"\s+", "", s)


def github_code_search(query: str, timeout: int = 20) -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {"status": "not_run", "reason": "GITHUB_TOKEN not set"}
    url = "https://api.github.com/search/code?q=" + urllib.parse.quote(f'"{query}"')
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return {"status": "ok", "hits": int(data.get("total_count", 0))}
    except Exception as exc:  # pragma: no cover - network
        return {"status": "error", "reason": str(exc)}


def infinigram_count(query: str, index: str = INFINIGRAM_INDEX, timeout: int = 20) -> dict:
    body = json.dumps({"index": index, "query_type": "count", "query": query}).encode()
    req = urllib.request.Request(INFINIGRAM_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        if "error" in data:
            return {"status": "error", "reason": data["error"]}
        return {"status": "ok", "hits": int(data.get("count", 0)), "index": index}
    except Exception as exc:  # pragma: no cover - network
        return {"status": "error", "reason": str(exc)}


def novelty_checks(node: Node, ctx, own_hashes: set[str], search: bool = True, seed: int | None = None,
                   created_at: str | None = None) -> dict:
    cfg = ctx.thr["pools"]
    h = canonical_hash(node)
    lib_hashes = {canonical_hash(f.node) for f in library().values()}
    c1 = h not in lib_hashes and h not in own_hashes
    rows = ctx.rows("train")
    sig = ctx.signal(node)
    best, best_ref = 0.0, None
    for name in ctx.references.names():
        r = daily_spearman(sig, ctx.references.signal(name), rows)
        if np.isfinite(r).sum() < 30:
            continue
        m = abs(float(np.nanmean(r)))
        if m > best:
            best, best_ref = m, name
    c2 = best < cfg["novel_max_abs_rho"]
    key = normalized_string(node)
    searches = {}
    if search:
        # several renderings: literal Qlib, Qlib without whitespace, the field-name-insensitive key and
        # the Alpha101 form (code search cannot match whitespace- or name-insensitively by itself)
        from dsl import to_alpha101

        q = to_qlib(node)
        for name, query in (("qlib", q), ("qlib_nospace", re.sub(r"\s+", "", q)), ("normalized", key),
                            ("alpha101", to_alpha101(node))):
            searches[f"github_code:{name}"] = github_code_search(query)
            searches[f"infinigram:{name}"] = infinigram_count(query)
    hits = [s.get("hits", 0) for s in searches.values() if s.get("status") == "ok"]
    completed = bool(searches) and all(s.get("status") == "ok" for s in searches.values())
    c3 = completed and not any(hits)
    status = ("failed" if not (c1 and c2) or any(hits) else "passed" if c3 else "pending_search")
    return {"canonical_unique": c1, "max_abs_rho": best, "max_abs_rho_ref": best_ref, "rho_ok": c2,
            "normalized_key": key, "searches": searches, "search_clean": c3, "searches_completed": completed,
            "status": status, "passed": status == "passed", "seed": seed, "created_at": created_at,
            "wording": "absent from searchable public sources at generation time"}
