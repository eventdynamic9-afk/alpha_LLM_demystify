"""Reference library completeness and id status (§5.5): every Alpha101 / GTJA-191 id the DSL can express
is transcribed; 'not_expressible' and 'unknown' only for ids listed with a reason."""
from collections import Counter

import pytest

from dsl import ParseError, canonical_equal, canonical_hash, effective_lookback, parse, walk
from pools import library as L
from pools.library import library_id_reason, library_id_status, verify_transcriptions


def _fields(lid):
    return {n.name for n in walk(L.get(lid).node) if n.is_field}


def test_every_id_is_in_exactly_one_status_set():
    assert verify_transcriptions() == []
    a101 = Counter(library_id_status(f"alpha101_{k:03d}") for k in range(1, 102))
    gtja = Counter(library_id_status(f"gtja191_{k:03d}") for k in range(1, 192))
    assert a101 == {"available": 76, "not_expressible": 25}
    assert gtja == {"available": 121, "not_expressible": 53, "unknown": 17}


@pytest.mark.parametrize("lib,size,nx,pending", [
    ("alpha101", 101, L.A101_NOT_EXPRESSIBLE_REASONS, L.A101_PENDING),
    ("gtja191", 191, L.GTJA_NOT_EXPRESSIBLE_REASONS, L.GTJA_PENDING)])
def test_unknown_and_not_expressible_only_from_explicit_sets(lib, size, nx, pending):
    for k in range(1, size + 1):
        lid = f"{lib}_{k:03d}"
        st = library_id_status(lid)
        assert (st == "not_expressible") == (k in nx), lid
        assert (st == "unknown") == (k in pending), lid
        assert (st == "available") == (L.get(lid) is not None), lid
        assert bool(library_id_reason(lid)) == (st != "available"), lid
    assert all("pending source access" in why for why in pending.values())


def test_ids_the_audit_found_expressible_are_available():
    for k in (21, 24, 25, 37, 43, 57, 83):
        lid = f"alpha101_{k:03d}"
        assert library_id_status(lid) == "available"
        f = L.get(lid)
        assert f.dialect == "alpha101" and canonical_equal(parse(f.source_text, "alpha101"), f.node)


def test_late_entries_carry_the_unchecked_note():
    for f in L.library().values():
        k = int(f.lib_id.split("_")[1]) if f.library != "alpha158" else None
        late = (f.library == "alpha101" and k in L._A101_LATE) or (f.library == "gtja191" and k in L._GTJA_LATE)
        assert f.notes == (L.LATE_NOTE if late else ""), f.lib_id
    assert L.LATE_NOTE == "transcribed without access to the source PDF; re-check before the main run"
    assert not (L._A101_LATE & L._A101_BASE) and not (L._GTJA_LATE & L._GTJA_BASE)


@pytest.mark.parametrize("src,dialect", [
    ("IndNeutralize(close, close)", "alpha101"),                 # IndNeutralize
    ("rank((returns * cap))", "alpha101"),                        # cap
    ("product(rank(close), 5)", "alpha101"),                     # product
    ("(rank(close)^rank(volume))", "alpha101"),                  # non-constant exponent
    ("SignedPower(close, delta(close, 4.96796))", "alpha101"),
    ("SMA(CLOSE-DELAY(CLOSE,5),5,1)", "gtja"),
    ("WMA(CLOSE,12)", "gtja"),
    ("SMEAN(CLOSE,12,1)", "gtja"),
    ("REGBETA(MEAN(CLOSE,6),SEQUENCE(6))", "gtja"),
    ("COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)", "gtja"),
    ("SUMIF(CLOSE,20,CLOSE<DELAY(CLOSE,1))", "gtja"),
    ("SUMAC(CLOSE-MEAN(CLOSE,48))", "gtja"),
    ("PROD(RANK(CLOSE),1)", "gtja"),
    ("CLOSE>DELAY(CLOSE,1)?CLOSE*SELF:SELF", "gtja"),
    ("RANK(DELTA(VWAP, 1))^TSRANK(CLOSE, 18)", "gtja"),
])
def test_not_expressible_reasons_really_fail_in_the_dsl(src, dialect):
    with pytest.raises(ParseError):
        parse(src, dialect)


def test_published_structural_facts_of_new_entries():
    assert _fields("alpha101_025") == {"close", "high", "volume", "vwap"}  # returns, adv20, vwap, high - close
    assert effective_lookback(L.get("alpha101_024").node) == 199           # delta of a 100-day mean over 100 days
    assert effective_lookback(L.get("alpha101_032").node) == 234           # correlation(vwap, delay(close, 5), 230)
    assert effective_lookback(L.get("alpha101_061").node) == 195           # adv180 inside a 17-day correlation
    assert _fields("gtja191_070") == _fields("gtja191_132") == {"amount"}
    assert effective_lookback(L.get("gtja191_055").node) == 20 and effective_lookback(L.get("gtja191_137").node) == 1
    # GTJA formulas borrowed from Alpha101 must coincide in canonical form (two transcriptions agree)
    for g, a in ((16, 50), (33, 52), (86, 46), (98, 24), (113, 45), (114, 83), (184, 37), (5, 26), (104, 22)):
        assert canonical_equal(L.get(f"gtja191_{g:03d}").node, L.get(f"alpha101_{a:03d}").node), (g, a)


def test_id_spellings_and_out_of_range():
    assert library_id_status("alpha101_12") == "available" and L.get("alpha101_12").lib_id == "alpha101_012"
    for lid in ("alpha101_000", "alpha101_102", "gtja191_192", "alpha158_FOO"):
        assert library_id_status(lid) == "nonexistent" and library_id_reason(lid)
    assert library_id_status("osap_Mom12m") == "unknown"
    assert "IndNeutralize" in library_id_reason("alpha101_048") and "SMA" in library_id_reason("gtja191_009")
    assert library_id_reason("alpha101_101") == ""


def test_partition_errors_are_reported_and_fail_loudly(monkeypatch):
    size, avail, nx, pending = L._NUMBERED["alpha101"]
    monkeypatch.setitem(L._NUMBERED, "alpha101", (size, avail, {**nx, 1: "dup"}, {}))
    bad = verify_transcriptions()
    assert any(b.startswith("alpha101_001: in ['available', 'not_expressible']") for b in bad)
    monkeypatch.setitem(L._NUMBERED, "alpha101", (size, avail, {k: v for k, v in nx.items() if k != 29}, {}))
    assert any(b.startswith("alpha101_029: in no status set") for b in verify_transcriptions())
    with pytest.raises(LookupError):
        library_id_status("alpha101_029")


def test_frozen_base_draw_is_unaffected_by_late_entries():
    late = {f"alpha101_{k:03d}" for k in L._A101_LATE} | {f"gtja191_{k:03d}" for k in L._GTJA_LATE}
    assert not late & {f.lib_id for f in L.base_candidates()}
    with_late = {f.lib_id: canonical_hash(f.node) for f in L.base_candidates(include_late=True)}
    for lid in late - set(with_late):  # left out only as a canonical duplicate of an earlier entry
        assert canonical_hash(L.get(lid).node) in with_late.values(), lid
    sel = L.select_base_set()
    assert sel["alpha101"] == sorted(f"alpha101_{k:03d}" for k in L._A101_BASE)
    assert sel["gtja191"] == sorted(f"gtja191_{k:03d}" for k in L._GTJA_BASE)
    assert sel["alpha158"] == sorted(f"alpha158_{k}" for k in L._A158_BASE)


def test_check_library_cli_passes():
    from pools.__main__ import main

    assert main(["check-library"]) == 0
