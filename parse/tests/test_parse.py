import json

from configs import models as load_models
from narrate.clients import MockClient
from parse.ensemble import ensemble, merge_parser_and_rules
from parse.llm_parser import llm_parse
from parse.normalize import horizon_bin_of_days, normalize_claim, predicate_key
from parse.rules import extract_claims, slot_hedge, slot_polarity, slot_scope
from parse.schema import validate_output
from parse.validate import extraction_metrics, go_no_go

TEXT = ("This is WorldQuant Alpha#12. The factor is driven by trading volume and the closing price. "
        "A higher close today lowers the factor value. It uses a 20-day window. It is a short-term signal. "
        "It ranks stocks against each other. It captures short-term reversal. "
        "High values predict higher future returns over the next 5 days. "
        "It may work best in volatile markets. It is novel and not explained by existing factors. "
        "It is independent of momentum. It also uses analyst revisions. "
        "This is because investors overreact to recent news.")


def _preds(claims):
    return {(c["predicate"], json.dumps(c["args"], sort_keys=True)) for c in claims}


def test_rule_extraction_covers_taxonomy():
    cs = extract_claims(TEXT, "r1")
    preds = {c["predicate"] for c in cs}
    assert {"IDENTITY", "DEPENDS_ON", "SIGN", "LOOKBACK", "HORIZON", "XSEC", "RESEMBLES", "PRED_SIGN", "REGIME",
            "NOVEL", "INDEPENDENT", "THEORY"} <= preds
    got = _preds(cs)
    assert ("IDENTITY", '{"library_id": "alpha101_012"}') in got
    assert ("SIGN", '{"direction": "-", "input": "close"}') in got
    assert ("LOOKBACK", '{"window": 20}') in got
    assert ("DEPENDS_ON", '{"input": "analyst_revisions"}') in got
    pred = [c for c in cs if c["predicate"] == "PRED_SIGN"][0]
    assert pred["args"] == {"sign": "+", "horizon": 5} and pred["horizon"] == "very_short"
    reg = [c for c in cs if c["predicate"] == "REGIME"][0]
    assert reg["hedge"] == "possible" and reg["args"]["regime"] == "high_vol"
    nov = [c for c in cs if c["predicate"] == "NOVEL"][0]
    assert nov["polarity"] == "affirm"
    for c in cs:
        assert TEXT[c["span"][0]:c["span"][1]] == c["text"]
    assert not validate_output({"rationale_id": "r1", "claims": cs})


def test_slots_and_normalization():
    assert slot_hedge("It might capture reversal.") == "possible"
    assert slot_hedge("It typically captures reversal.") == "typical"
    assert slot_polarity("It does not depend on volume.") == "deny"
    assert slot_scope("The effect is stronger in small caps.") == "small caps"
    assert horizon_bin_of_days(5) == "very_short" and horizon_bin_of_days(60) == "medium"
    c = normalize_claim({"predicate": "resembles", "args": {"ref": "Short-term reversal effect", "sign": "positive"}})
    assert c["args"] == {"ref": "short-term reversal", "sign": "+"} and c["type"] == "C2" and not c["ambiguous"]
    amb = normalize_claim({"predicate": "RESEMBLES", "args": {"ref": "the quality anomaly"}})
    assert amb["ambiguous"]
    assert normalize_claim({"predicate": "IDENTITY", "args": {"library_id": "Alpha#7"}})["args"]["library_id"] == "alpha101_007"
    assert normalize_claim({"predicate": "HORIZON", "args": {"bin": "medium-term"}})["args"]["bin"] == "medium"


def test_llm_parser_with_mock_and_merge():
    client = MockClient(next(m for m in load_models("models.mock.yaml")["models"] if m["id"] == "mock-parser"))
    res = llm_parse(client, "r1", TEXT)
    assert not res["errors"] and len(res["claims"]) >= 10
    merged = merge_parser_and_rules(res["claims"][:3], TEXT, "r1")
    assert len(merged) > 3
    e = ensemble(res["claims"], res["claims"], TEXT, "r1")
    assert len(e["accepted"]) == len(res["claims"]) and not e["adjudicate"]


def test_validation_metrics_and_gate():
    gold = {"r1": [normalize_claim(c) for c in extract_claims(TEXT, "r1")]}
    perfect = extraction_metrics(gold, gold)
    assert all(v["f1"] == 1.0 for v in perfect["by_type"].values())
    assert go_no_go(perfect)["go"]
    half = {"r1": gold["r1"][::2]}
    m = extraction_metrics(half, gold)
    assert any(v["recall"] < 1 for v in m["by_type"].values())
    assert predicate_key(gold["r1"][0]) == predicate_key(dict(gold["r1"][0]))
