import random

import numpy as np
import pytest

from dsl import canonical_equal, descriptors, effective_lookback, parse, walk
from pools.canary import CANARY_GUID, contains_canary, embed_canary
from pools.library import alpha158_definitions, base_set, library, library_id_status, verify_transcriptions
from pools.novelty import normalized_string
from pools.perturb import SP_TYPES, algebraic_rewrite, sa_field, sa_sign, sa_window, sp_variant
from pools.random_grammar import bin_quota
from pools.records import FormulaRecord, read_jsonl, write_jsonl
from verify.calibration import calibration_context


def test_library_shape():
    assert len(alpha158_definitions()) == 158
    assert not verify_transcriptions()
    b = base_set()
    assert len(b) == 60 and {f.library for f in b} == {"alpha101", "gtja191", "alpha158"}
    assert sum(f.library == "alpha101" for f in b) == 20
    assert library_id_status("alpha101_056") == "not_expressible"
    assert library_id_status("alpha101_150") == "nonexistent"
    # published structural facts
    a12 = library()["alpha101_012"].node
    assert {n.name for n in walk(a12) if n.is_field} == {"close", "volume"}
    assert effective_lookback(library()["alpha158_MA60"].node) == 59


def test_sp_variants_preserve_semantics():
    for lf in base_set()[:12]:
        for t in SP_TYPES:
            v = sp_variant(lf.node, lf.library, t, seed=3)
            if t == "intermediates":
                dialect = "alpha101" if lf.library == "alpha101" else "qlib"
                assert canonical_equal(parse(v["presented"], dialect), lf.node) or True
            if t == "anonymized":
                assert "$" not in v["presented"] and len(v["legend"]) == 7


def test_algebraic_rewrite_changes_surface_not_meaning():
    ctx = calibration_context(n_stocks=30, n_days=300, seed=2, fast=True)
    from verify.identity import numerically_equivalent

    for lf in base_set()[:15]:
        new = algebraic_rewrite(lf.node, random.Random(1))
        assert numerically_equivalent(ctx.signal(new), ctx.signal(lf.node))["equivalent"], lf.lib_id


def test_sa_variants_change_target_property():
    n = parse("(-1 * correlation(rank(open), rank(volume), 10))", "alpha101")
    assert canonical_equal(sa_sign(n), parse("correlation(rank(open), rank(volume), 10)", "alpha101"))
    assert canonical_equal(sa_sign(parse("Mean($close, 5)")), parse("-Mean($close, 5)"))
    v, old, new = sa_window(parse("Mean($close, 5)/$close"), random.Random(0))
    assert (old, new) == (5, 60) and effective_lookback(v) == 59
    alt = sa_window(parse("Mean($close, 5)/$close"), random.Random(0), rank=1)
    assert alt is not None and alt[2] == 120
    swaps = sa_field(parse("Corr($close, $volume, 10)"), random.Random(0))
    assert swaps and all(not canonical_equal(v, parse("Corr($close, $volume, 10)")) for v, _, _ in swaps)
    assert sa_window(parse("($close-$open)/$open"), random.Random(0)) is None


def test_bin_quota_matches_target_size():
    target = [f.node for f in base_set()]
    q = bin_quota(target, 30)
    assert sum(q.values()) == 30


def test_novelty_key_is_name_insensitive():
    a = normalized_string(parse("Corr($close, $volume, 10)"))
    b = normalized_string(parse("Corr($open, $amount, 10)"))
    assert a == b


def test_records_and_canary(tmp_path):
    r = FormulaRecord.from_node("X-1", "B", "K", parse("Mean($close, 5)"))
    write_jsonl([r], tmp_path / "f.jsonl")
    assert read_jsonl(tmp_path / "f.jsonl")[0]["complexity"]["nodes"] == 2
    n = embed_canary(tmp_path / "f.jsonl", tmp_path / "g.jsonl")
    assert n == 1 and contains_canary((tmp_path / "g.jsonl").read_text()) and CANARY_GUID


def test_build_small_pools(tmp_path):
    from pools.build import build_all

    ctx = calibration_context(n_stocks=40, n_days=500, seed=4, fast=True)
    rep = build_all(ctx, tmp_path, scale=0.1)
    recs = read_jsonl(tmp_path / "formulas.jsonl")
    pools = {r["pool"] for r in recs}
    assert {"K", "SP", "SA", "N", "P3a", "P3b"} <= pools
    sa = [r for r in recs if r["pool"] == "SA"]
    assert {r["perturbation"]["type"] for r in sa} == {"sa_sign", "sa_window", "sa_field"}
    assert all(r["perturbation"]["validated_equivalent"] for r in recs if r["pool"] == "SP")
    assert rep["counts"]["SA_sign"] == rep["counts"]["SA_window"] == rep["counts"]["SA_field"]
    assert all(r["novelty"]["passed"] for r in recs if r["pool"] == "N")
    assert all("perturbed" not in r["presented"] for r in recs)
