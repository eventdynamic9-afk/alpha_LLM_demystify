import random

import pytest
from hypothesis import given, settings, strategies as st

from dsl import (C, F, Node, ParseError, UnsupportedOperator, canonical_equal, canonical_string, canonicalize,
                 descriptors, effective_lookback, parse, to_alpha101, to_math, to_program, to_qlib, validate)
from dsl.monotonicity import AMB, NEG, POS, current_value_direction, field_direction, sign_map
from dsl.random_trees import random_tree
from dsl.ranges import interval
from dsl.serialize import to_anonymized
from dsl.units import analyze_units
from dsl.validate import time_offsets


CORPUS = [
    ("Mean($close, 5)/$close", "qlib"),
    ("Corr($close, Log($volume+1), 20)", "qlib"),
    ("($close-Min($low, 20))/(Max($high, 20)-Min($low, 20)+1e-12)", "qlib"),
    ("Sum(Greater($close-Ref($close, 1), 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)", "qlib"),
    ("Quantile($close, 20, 0.8)/$close", "qlib"),
    ("IdxMax($high, 30)/30", "qlib"),
    ("Rank($close, 10)", "qlib"),
    ("(-1 * correlation(rank(open), rank(volume), 10))", "alpha101"),
    ("(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))", "alpha101"),
    ("(-1 * Ts_Rank(rank(low), 9))", "alpha101"),
    ("scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))", "alpha101"),
    ("((close - open) / ((high - low) + .001))", "alpha101"),
    ("(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))", "gtja"),
    ("MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)", "gtja"),
    ("(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME", "gtja"),
]


@pytest.mark.parametrize("src,dialect", CORPUS)
def test_corpus_parses_and_validates(src, dialect):
    n = parse(src, dialect)
    assert validate(n).ok
    # every serialization re-parses to a canonically identical tree
    assert canonical_equal(parse(to_qlib(n)), n)
    # Alpha101 has no Mean/Var/elementwise-max-with-constant: those serialize to numerically
    # equivalent (not structurally equal) forms, checked on data in executors/tests
    if not any(k in to_qlib(n) for k in ("Mean", "Var", "Greater", "Less")):
        assert canonical_equal(parse(to_alpha101(n), "alpha101"), n)
    assert to_math(n)


def test_alpha101_semantics():
    n = parse("ts_rank(close, 9.91)", "alpha101")       # windows floored as in the paper
    assert n == Node("TsRank", (F("close"),), (9,))
    assert parse("max(close, 5)", "alpha101").op == "Max"           # ts_max
    assert parse("max(close, open)", "alpha101").op == "Greater"    # elementwise
    assert parse("MAX(CLOSE, 0)", "gtja").op == "Greater"           # GTJA MAX is elementwise
    assert parse("TSMAX(CLOSE, 5)", "gtja").op == "Max"
    adv = parse("adv20", "alpha101")
    assert adv.op == "Mean" and adv.params == (20,)
    ret = parse("returns", "alpha101")
    assert to_alpha101(ret) == "returns"


def test_qlib_rank_is_time_series():
    assert parse("Rank($close, 10)").op == "TsRank"
    assert parse("CSRank($close)").op == "CSRank"
    with pytest.raises(ParseError):
        parse("Max($close, $open)")                               # strict Qlib: Max needs a window
    assert parse("Max($close, $open)", strict=False).op == "Greater"


def test_unsupported_and_errors():
    with pytest.raises(UnsupportedOperator):
        parse("indneutralize(close, industry)", "alpha101")
    with pytest.raises(UnsupportedOperator):
        parse("SMA(CLOSE, 15, 2)", "gtja")
    with pytest.raises(ParseError):
        parse("Mean($close)")
    with pytest.raises(ParseError):
        parse("$notafield")


def test_program_form_roundtrip():
    n = parse("($close - Mean($close, 5)) / Std($close, 20)")
    prog = to_program(n, min_size=2)
    assert "factor =" in prog
    assert canonical_equal(parse(prog), n)


def test_anonymized_has_legend():
    n = parse("Corr($close, $volume, 10)")
    txt, legend = to_anonymized(n, seed=3)
    assert set(legend) == {"close", "volume"}
    assert "$" not in txt and all(v in txt for v in legend.values())


def test_canonical_identities():
    pairs = [
        ("$close + $open", "$open + $close"),
        ("$close - $open", "-($open - $close)"),
        ("Ref(Ref($close, 2), 3)", "Ref($close, 5)"),
        ("-(-$close)", "$close"),
        ("Delta($close, 5)", "$close - Ref($close, 5)"),
        ("$close < $open", "$open > $close"),
        ("Corr($close, $volume, 10)", "Corr($volume, $close, 10)"),
        ("Ref(Mean($close, 5), 2)", "Mean(Ref($close, 2), 5)"),
        ("2 * 3 * $close", "6 * $close"),
        ("$close * -1", "-$close"),
        ("Ref($close, 0)", "$close"),
    ]
    for a, b in pairs:
        assert canonical_string(parse(a)) == canonical_string(parse(b)), (a, b)
    assert not canonical_equal(parse("Mean($close, 5)"), parse("Mean($close, 10)"))
    # Ref is not pushed through cross-sectional operators
    assert "Ref(CSRank" in canonical_string(parse("Ref(CSRank($close), 1)"))


def test_validator_rejects_lookahead():
    bad = Node("Ref", (F("close"),), (-1,))
    rep = validate(bad)
    assert not rep.ok and any("look-ahead" in e for e in rep.errors)
    assert not validate(Node("Std", (F("close"),), (1,))).ok
    assert not validate(Node("Mean", (F("close"),), (5.5,))).ok
    assert time_offsets(parse("Mean(Ref($close, 5), 20)")) == (5, 24)


def test_lookback_path_sum():
    assert effective_lookback(parse("Mean(Ref($close, 5), 20)")) == 24
    assert effective_lookback(parse("Corr($close, Ref($volume, 3), 10)")) == 12
    assert effective_lookback(parse("Delta($close, 7) + Mean($open, 3)")) == 7


def test_units():
    assert analyze_units(parse("$close/$open")).dimensionless
    assert not analyze_units(parse("Log($close) + $volume")).consistent        # Alpha Jungle example
    assert analyze_units(parse("Log($close) - Log(Ref($close, 1))")).dimensionless
    assert not analyze_units(parse("$close + $volume")).consistent
    assert analyze_units(parse("$close * $volume")).unit == analyze_units(parse("$amount")).unit


def test_ranges():
    assert interval(parse("CSRank($close)")).within(0, 1)
    assert interval(parse("Corr($close, $volume, 10)")).within(-1, 1)
    assert interval(parse("$close/$open")).strictly_positive
    assert interval(parse("Std($close, 20)")).nonneg
    assert interval(parse("Sign($close - $open)")).within(-1, 1)


def test_monotonicity():
    n = parse("CSRank(-($close/Ref($close, 5) - 1))")
    m = sign_map(n)
    assert m[("close", 0)] == NEG and m[("close", 5)] == POS
    assert field_direction(parse("Mean($close, 5)"), "close") == POS
    assert field_direction(parse("Std($close, 5)"), "close") == AMB
    d = parse("$close / $volume")
    assert current_value_direction(d, "close") == POS and current_value_direction(d, "volume") == NEG
    t = sign_map(parse("TsRank($close, 10)"))
    assert t[("close", 0)] == POS and all(t[("close", k)] == NEG for k in range(1, 10))
    assert field_direction(parse("$open"), "close") == 0
    s = sign_map(parse("Slope($close, 5)"))
    assert s[("close", 0)] == POS and s[("close", 4)] == NEG and ("close", 2) not in s


def test_descriptors():
    d = descriptors(parse("CSRank(Corr($close, $volume, 10))"))
    assert d["nodes"] == 4 and d["depth"] == 3 and d["xsec_ops"] == 1 and d["nonmonotone_ops"] == 1
    assert d["fields"] == ["close", "volume"] and d["max_lookback"] == 9


@settings(max_examples=150, deadline=None)
@given(st.integers(min_value=0, max_value=10**9), st.integers(min_value=2, max_value=6))
def test_random_trees_roundtrip(seed, depth):
    t = random_tree(random.Random(seed), max_depth=depth)
    assert validate(t).ok
    assert parse(to_qlib(t)) == t                      # exact structural round trip
    c = canonicalize(t)
    assert canonicalize(c) == c                         # idempotent
    assert canonical_string(parse(to_qlib(c))) == canonical_string(c)
    assert canonical_equal(parse(to_alpha101(t), "alpha101"), t) or any(
        k in to_qlib(t) for k in ("Mean", "Var", "Greater", "Less", "Neg", "-"))
