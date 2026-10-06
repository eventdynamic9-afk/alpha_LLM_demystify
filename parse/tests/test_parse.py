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
    assert len(merged) == 3                                  # first run: E2 + E1 slots only, no rule additions
    added = merge_parser_and_rules(res["claims"][:3], TEXT, "r1", add_rule_claims=True)
    assert len(added) > 3
    assert len(merge_parser_and_rules(res["claims"], TEXT, "r1", add_rule_claims=True)) == len(res["claims"])
    e = ensemble(res["claims"], res["claims"], TEXT, "r1")
    assert len(e["accepted"]) == len(res["claims"]) and not e["adjudicate"]


def test_validation_metrics_and_gate():
    gold = {"r1": [normalize_claim(c) for c in extract_claims(TEXT, "r1")]}
    perfect = extraction_metrics(gold, gold)
    assert all(v["f1"] == 1.0 for v in perfect["by_type"].values())
    assert not go_no_go(perfect)["go"]                       # §9.3: agreement is required, not optional
    assert go_no_go(perfect, 0.85)["go"] and not go_no_go(perfect, 0.5)["go"]
    half = {"r1": gold["r1"][::2]}
    m = extraction_metrics(half, gold)
    assert any(v["recall"] < 1 for v in m["by_type"].values())
    assert predicate_key(gold["r1"][0]) == predicate_key(dict(gold["r1"][0]))


# ------------------------------------------------------------------ audit fixes (§9.1-§9.5, §4.3, §7.2)
class _StubParser:
    """Returns a fixed parser reply; counts calls."""

    def __init__(self, reply, model_id="stub-parser"):
        self.reply, self.model_id, self.model_string, self.n = reply, model_id, f"{model_id}-v1", 0

    def complete(self, messages, tools=None, temperature=0.0, max_tokens=700, seed=None):
        from narrate.clients import LLMResponse

        self.n += 1
        return LLMResponse(self.reply if isinstance(self.reply, str) else json.dumps(self.reply), [],
                           {"in": 10, "out": 5}, 0.01, "fp-1", self.model_string, "stop")


def test_identity_ids_keep_library_case_and_alpha101_is_anchored():
    from pools.library import library_id_status

    for raw in ("alpha158_KMID", "alpha158_kmid", "Alpha158 KMID"):
        c = normalize_claim({"predicate": "IDENTITY", "args": {"library_id": raw}})
        assert c["args"]["library_id"] == "alpha158_KMID" and not c["ambiguous"]
    assert library_id_status("alpha158_KMID") == "available"
    for txt, want in (("This factor is Alpha158 KMID.", ["alpha158_KMID"]),
                      ("This is WorldQuant Alpha#12.", ["alpha101_012"]),
                      ("This is Alpha 101 number 12.", []), ("This is GTJA191.", []),
                      ("It is alpha101_012 in disguise.", ["alpha101_012"])):
        got = [normalize_claim(c)["args"]["library_id"] for c in extract_claims(txt) if c["predicate"] == "IDENTITY"]
        assert got == want, (txt, got)
    assert normalize_claim({"predicate": "IDENTITY", "args": {"library_id": "Alpha 101"}})["ambiguous"]
    assert normalize_claim({"predicate": "IDENTITY", "args": {"library_id": "GTJA-191 Alpha5"}})["args"]["library_id"] == "gtja191_005"
    # E2 + rules never re-introduces a spurious alpha101_158
    reply = {"rationale_id": "r", "claims": [{"text": "This factor is Alpha158 KMID.", "type": "C6", "predicate": "IDENTITY",
                                              "args": {"library_id": "alpha158_KMID"}}]}
    res = llm_parse(_StubParser(reply), "r", "This factor is Alpha158 KMID.")
    ids = [c["args"]["library_id"] for c in merge_parser_and_rules(res["claims"], "This factor is Alpha158 KMID.", "r",
                                                                    add_rule_claims=True)]
    assert ids == ["alpha158_KMID"]


def test_refs_map_only_on_exact_codebook_match():
    from configs import codebook
    from parse.normalize import _canon_ref

    assert _canon_ref("low-volatility") == ("low volatility", False) and codebook()["terms"]["low volatility"]["sign"] == -1
    assert _canon_ref("Low-Beta") == ("low beta", False)
    assert _canon_ref("STREV_5d") == ("STREV_5d", False) and _canon_ref("strev_5d") == ("STREV_5d", False)
    assert _canon_ref("reversals") == ("reversal", False)
    for vague in ("liquidity-adjusted momentum", "emphasize", "detrended price", "small-cap"):
        assert _canon_ref(vague)[1], vague
    ex = [c for c in extract_claims("It has a low-volatility tilt.") if c["predicate"] == "EXPOSED"]
    assert ex and ex[0]["args"] == {"ref": "low volatility", "sign": "+"} and not ex[0]["ambiguous"]
    assert not [c for c in extract_claims("It loads on high-volatility names.") if c["predicate"] == "EXPOSED"]
    amb = [c for c in extract_claims("It has exposure to less momentum.") if c["predicate"] == "EXPOSED"]
    assert amb and amb[0]["ambiguous"]
    bt = normalize_claim({"predicate": "BETTER_THAN", "args": {"ref": "All factors", "metric": "IC"}})
    assert bt["args"]["ref"] == "all factors" and not bt["ambiguous"]


def test_signs_levels_and_ranges_are_never_guessed():
    c = normalize_claim({"predicate": "SIGN", "args": {"input": "close", "direction": "−"}})
    assert c["args"]["direction"] == "-" and not c["ambiguous"]
    assert normalize_claim({"predicate": "MONO", "args": {"input": "abn_vol", "direction": "↑"}})["args"]["direction"] == "+"
    assert normalize_claim({"predicate": "PRED_SIGN", "args": {"sign": "↓"}})["args"]["sign"] == "-"
    assert normalize_claim({"predicate": "EXPOSED", "args": {"ref": "volatility", "sign": "−"}})["args"]["sign"] == "-"
    for c in ({"predicate": "SIGN", "args": {"input": "close"}}, {"predicate": "SIGN", "args": {"input": "close", "direction": "sideways"}},
              {"predicate": "PRED_SIGN", "args": {}}, {"predicate": "RESEMBLES", "args": {"ref": "momentum"}},
              {"predicate": "TURNOVER", "args": {"level": "moderate"}}, {"predicate": "TURNOVER", "args": {}},
              {"predicate": "RANGE", "args": {"low": -1}}):
        assert normalize_claim(c)["ambiguous"], c
    assert normalize_claim({"predicate": "TURNOVER", "args": {"level": "slow-moving"}})["args"]["level"] == "low"
    assert normalize_claim({"predicate": "RANGE", "args": {"low": "−1", "high": 1}})["args"] == {"low": -1.0, "high": 1.0}
    for txt in ("It is bounded between −1 and 1.", "It is bounded between -1 and 1."):
        r = [c for c in extract_claims(txt) if c["predicate"] == "RANGE"]
        assert r[0]["args"] == {"low": -1.0, "high": 1.0}


def test_llm_spans_are_located_from_copied_text():
    from parse.llm_parser import locate_span

    text = "The factor is driven by trading volume. It does not use the closing price. It uses a 20-day window."
    assert locate_span(text, "driven by trading volume", None) == ([14, 38], "exact")
    assert locate_span(text, "driven by trading volume", [0, 5])[1] == "exact"
    assert locate_span(text, "driven  by TRADING volume", None) == ([14, 38], "normalized")
    sp, src = locate_span(text, "it doesn't use the closing price", None)
    assert src == "fuzzy" and sp[0] >= 40 and sp[1] <= 74
    assert locate_span(text, "uses analyst revisions heavily", None) == ([0, 0], "unlocated")
    assert locate_span(text, "", [40, 74]) == ([40, 74], "llm_offsets")
    reply = {"rationale_id": "r", "claims": [
        {"text": "driven by trading volume", "type": "C1", "predicate": "DEPENDS_ON", "args": {"input": "volume"}},
        {"span": [999, 1005], "text": "a 20-day window", "type": "C1", "predicate": "LOOKBACK", "args": {"window": 20}}]}
    res = llm_parse(_StubParser(reply), "r", text)
    assert not res["errors"]
    for c in res["claims"]:
        assert text[c["span"][0]:c["span"][1]] == c["text"] and c["span_source"] == "exact"
    merged = merge_parser_and_rules(res["claims"], text, "r", add_rule_claims=True)
    assert sum(1 for c in merged if c["predicate"] == "DEPENDS_ON" and c["args"]["input"] == "volume") == 1
    assert sum(1 for c in merged if c["predicate"] == "LOOKBACK") == 1


def test_llm_claims_get_rule_slots_before_defaults():
    text = "The factor might be driven by trading volume. It does not use the closing price."
    reply = {"rationale_id": "r", "claims": [
        {"text": "The factor might be driven by trading volume.", "type": "C1", "predicate": "DEPENDS_ON", "args": {"input": "volume"}},
        {"text": "It does not use the closing price.", "type": "C1", "predicate": "DEPENDS_ON", "args": {"input": "close"},
         "hedge": None},
        {"text": "It does not use the closing price.", "type": "C1", "predicate": "DEPENDS_ON", "args": {"input": "open"},
         "hedge": "typical", "polarity": "affirm"}]}
    res = llm_parse(_StubParser(reply), "r", text)
    vol, close, kept = res["claims"]
    assert vol["hedge"] == "possible" and vol["polarity"] == "affirm"
    assert close["polarity"] == "deny" and close["hedge"] == "absolute"
    assert kept["hedge"] == "typical" and kept["polarity"] == "affirm"     # parser-supplied slots are kept


def test_parser_calls_are_cached_and_fully_logged(tmp_path):
    from configs import prompt, template_sha256
    from narrate.logger import CallLogger, ResponseCache

    stub = _StubParser({"rationale_id": "r", "claims": []})
    cache, log = ResponseCache(tmp_path / "pc.sqlite"), CallLogger(tmp_path / "pc.jsonl")
    a = llm_parse(stub, "r", "It uses a 20-day window.", log=log.log, cache=cache)
    b = llm_parse(stub, "r", "It uses a 20-day window.", log=log.log, cache=cache)
    assert stub.n == 1 and not a["cached"] and b["cached"]
    rec = [json.loads(x) for x in (tmp_path / "pc.jsonl").read_text().splitlines()]
    assert len(rec) == 1
    r = rec[0]
    assert r["template_sha256"]["parser_instruction"] == template_sha256(prompt("parser_instruction"))
    assert r["params"]["temperature"] == 0.0 and r["params"]["seed"] == 0 and r["params"]["max_tokens"] == 2000
    assert r["response"]["latency_s"] == 0.01 and r["response"]["usage"] == {"in": 10, "out": 5}
    assert r["response"]["fingerprint"] == "fp-1" and r["prompt_hash"] and r["model_string"] == "stub-parser-v1"


def test_polarity_is_a_slot_in_parser_validation():
    from parse.validate import _match_pairs

    g = normalize_claim({"predicate": "DEPENDS_ON", "args": {"input": "close"}, "span": [0, 30], "polarity": "deny"})
    p = normalize_claim({"predicate": "DEPENDS_ON", "args": {"input": "close"}, "span": [0, 30], "polarity": "affirm"})
    assert _match_pairs([p], [g]) == [(0, 0)]
    m = extraction_metrics({"r": [p]}, {"r": [g]})
    assert m["by_type"]["C1"]["f1"] == 1.0 and m["slot_accuracy"]["polarity"] == 0.0
    assert predicate_key(p) != predicate_key(g)                    # downstream key keeps deny/affirm distinct


def test_validate_cli_requires_agreement(tmp_path, capsys):
    import csv

    import parse.__main__ as pm
    from annotation.tools import SHEET_COLS

    gold = [normalize_claim(c) for c in extract_claims(TEXT, "r1")]
    for c in gold:
        c["rationale_id"] = "r1"
    gp = tmp_path / "gold.jsonl"
    gp.write_text("\n".join(json.dumps(c) for c in gold))
    assert pm.main(["validate", "--pred", str(gp), "--gold", str(gp)]) == 1          # no agreement -> no-go
    assert "not supplied" in capsys.readouterr().out
    for who in ("A", "B"):
        with open(tmp_path / f"sheet_{who}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SHEET_COLS)
            w.writeheader()
            for i, c in enumerate(gold):
                w.writerow({"rationale_id": "r1", "claim_no": i, "span_start": c["span"][0], "span_end": c["span"][1],
                            "text": c["text"], "type": c["type"], "predicate": c["predicate"],
                            "args_json": json.dumps(c["args"]), "scope": "", "hedge": c["hedge"],
                            "polarity": c["polarity"], "horizon": "", "ambiguous": "", "note": ""})
    assert pm.main(["validate", "--pred", str(gp), "--gold", str(gp), "--sheet-a", str(tmp_path / "sheet_A.csv"),
                    "--sheet-b", str(tmp_path / "sheet_B.csv")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["agreement"]["krippendorff_alpha"] >= 0.667 and out["decision"]["go"]


def test_parse_run_keeps_novel_pool_away_from_training_endpoints(tmp_path, capsys):
    import yaml

    import parse.__main__ as pm

    cfg = {"models": [{"id": "p-api", "family": "x", "provider": "relay", "relay_dir": str(tmp_path / "relay"),
                       "roles": ["parser"], "may_train_on_inputs": True}]}
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(cfg))
    rats = [{"rationale_id": "R-N", "formula_id": "N1", "pool": "N", "text": "It uses a 20-day window.", "status": "ok"},
            {"rationale_id": "R-K", "formula_id": "K1", "pool": "K", "text": "It uses a 5-day window.", "status": "ok"}]
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rats))
    args = ["run", "--rationales", str(tmp_path / "r.jsonl"), "--out", str(tmp_path / "c.jsonl"), "--models", str(tmp_path / "m.yaml")]
    assert pm.main(args) == 2 and not (tmp_path / "relay").exists()             # refused before any call
    assert pm.main(args + ["--skip-novel"]) == 0
    reqs = list((tmp_path / "relay" / "requests").glob("*.txt"))
    assert len(reqs) == 1 and "5-day" in reqs[0].read_text() and "20-day" not in reqs[0].read_text()
    skipped = [json.loads(x) for x in (tmp_path / "parser_skipped_novel.jsonl").read_text().splitlines()]
    assert [s["rationale_id"] for s in skipped] == ["R-N"]
