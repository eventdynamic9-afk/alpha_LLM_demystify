import json

import pytest

from configs import models as load_models
from configs import prompt
from dsl import parse
from narrate.clients import MockClient, get_client
from narrate.logger import ResponseCache, prompt_hash
from narrate.plan import plan_cells, plan_summary
from narrate.prompts import build_messages
from narrate.roster import novel_pool_allowed, post_cutoff_window, validate_roster, with_role
from narrate.runner import NarrationRunner, classify_output
from narrate.sandbox import Sandbox
from pools.records import FormulaRecord
from verify.calibration import calibration_context


def _rec(fid="B-A101-012-K", src="(sign(delta(volume, 1)) * (-1 * delta(close, 1)))", pool="K", **kw):
    r = FormulaRecord.from_node(fid, "B", pool, parse(src, "alpha101"), presented=src, notation="alpha101", **kw)
    return r.to_dict()


def test_prompt_is_verbatim_template():
    msgs, meta = build_messages(_rec(), "A0", "guided", "CN", seed=1)
    assert msgs[0]["content"] == prompt("narrator_system")
    u = msgs[1]["content"]
    assert "Factor:      (sign(delta(volume, 1))" in u
    assert u.rstrip().endswith("and under what conditions it should work.")
    assert "Name:" not in u and "perturbed" not in u.lower()
    m2, _ = build_messages(_rec(label="Name: WorldQuant Alpha#12"), "A0", "minimal", "CN", seed=1)
    assert "Name: WorldQuant Alpha#12" in m2[1]["content"]
    assert m2[1]["content"].rstrip().endswith("Explain the rationale of this alpha factor.")
    m3, _ = build_messages(_rec(), "A2", "guided", "CN", seed=1)
    assert "At most 10 tool calls." in m3[1]["content"]
    assert "template_sha256" in meta


def test_field_order_randomized_but_deterministic():
    a, _ = build_messages(_rec(), "A0", "guided", "CN", seed=1)
    b, _ = build_messages(_rec(), "A0", "guided", "CN", seed=1)
    c, _ = build_messages(_rec(), "A0", "guided", "CN", seed=2)
    assert a == b and a != c


def test_roster_validation():
    rep = validate_roster(load_models("models.yaml"), main_run=True)
    assert not rep.ok                                       # placeholders must be pinned on the run date
    assert any("not pinned" in p for p in rep.problems)
    mock = load_models("models.mock.yaml")
    assert validate_roster(mock, main_run=True).ok
    assert post_cutoff_window(mock, "2026-09-30") == ("2026-01-31", "2026-09-30")
    assert novel_pool_allowed({"provider": "local"}) and not novel_pool_allowed({"provider": "api", "may_train_on_inputs": True})


def test_plan_counts_match_spec_shape():
    recs = []
    for i in range(6):
        recs.append(_rec(f"K{i}", meta={"a2_subset": i < 3}))
        recs.append(_rec(f"SAs{i}", pool="SA", perturbation={"type": "sa_sign"}, meta={"a2_subset": i < 3}))
        recs.append(_rec(f"P3a{i}", pool="P3a"))
    for r in recs:
        if r["pool"] == "P3a":
            r["arm"] = "A"
    narr = with_role(load_models("models.mock.yaml"), "narrator")
    cells = plan_cells(recs, narr, k=3)
    s = plan_summary(cells)
    # per model: K 6x3 + SA 6x3 + A2 (3+3)x3 + P3 6x3x3 + minimal 1x3
    assert s["per_model"]["mock-careful"] == 18 + 18 + 18 + 54 + 3


def test_cache_and_classify(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    h = prompt_hash([{"role": "user", "content": "x"}])
    c.put("m", h, 0, 0.7, {"text": "a"})
    c.put("m", h, 0, 0.7, {"text": "b"})                  # never overwritten
    assert c.get("m", h, 0, 0.7)["text"] == "a" and c.get("m", h, 1, 0.7) is None
    assert classify_output("") == "empty" and classify_output("I'm sorry, I cannot help.") == "refusal"


def test_mock_narration_and_runner(tmp_path):
    ctx = calibration_context(n_stocks=30, n_days=400, seed=3, fast=True)
    cfg = load_models("models.mock.yaml")
    recs = {r["formula_id"]: r for r in [_rec(), _rec("P3a-0", "Mean($close, 5)/$close", pool="P3a")]}
    recs["P3a-0"]["arm"] = "A"
    runner = NarrationRunner(cfg, recs, tmp_path, ctx)
    cells = plan_cells(list(recs.values()), with_role(cfg, "narrator"), k=1)
    out = runner.run(cells)
    assert len(out) == len(cells) and all(o["status"] == "ok" for o in out)
    again = runner.run(cells)                              # resumable: nothing re-narrated
    assert again == []
    lines = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert lines and "template_sha256" in json.loads(lines[0])
    a1 = [o for o in out if o["access"] == "A1"]
    assert a1


def test_sandbox_tools():
    ctx = calibration_context(n_stocks=30, n_days=400, seed=3, fast=True)
    sb = Sandbox(ctx, max_calls=3)
    r = sb.call("compute_signal", {"expr": "-1*($close/Ref($close, 5)-1)"})
    assert r["rank_ic_t"] > 0
    assert "mean_rank_corr" in sb.call("corr_with", {"expr": "$close/Ref($close, 5)", "reference_name": "STREV_5d"})
    assert "coverage" in sb.call("describe", {"expr": "$close"})
    assert "error" in sb.call("describe", {"expr": "$close"})            # budget exhausted


def test_get_client_requires_endpoint():
    with pytest.raises(ValueError):
        get_client({"id": "x", "provider": "openai_compatible", "base_url": "TO_FILL"})
    assert isinstance(get_client({"id": "m", "provider": "mock"}), MockClient)


def test_full_plan_is_3555_per_model():
    """§7.3: 1,440 Arm-B + 2,115 Arm-A narrations per model with full-size pools."""
    narr = with_role(load_models("models.mock.yaml"), "narrator")
    recs = []

    def mk(fid, pool, arm="B", **kw):
        d = {"formula_id": fid, "pool": pool, "arm": arm, "meta": kw.pop("meta", {}), **kw}
        recs.append(d)

    for i in range(60):
        mk(f"K{i}", "K", meta={"a2_subset": i < 30})
        mk(f"SP{i}", "SP")
        mk(f"N{i}", "N")
        for t in ("sign", "window", "field"):
            mk(f"SA{t}{i}", "SA", perturbation={"type": f"sa_{t}"}, meta={"a2_subset": t == "sign" and i < 30})
    for i in range(30):
        mk(f"KN{i}", "K_named")
        mk(f"NL{i}", "NL")
        mk(f"P3a{i}", "P3a", "A")
        mk(f"P3b{i}", "P3b", "A")
    for m in narr:
        for i in range(60):
            mk(f"P1r-{m['id']}-{i}", "P1", "A", stratum="raw", author_model=m["id"])
            mk(f"P1m-{m['id']}-{i}", "P1", "A", stratum="mined", author_model=m["id"])
            mk(f"P2-{m['id']}-{i}", "P2", "A", author_model=m["id"])
    s = plan_summary(plan_cells(recs, narr, k=3))
    assert s["per_model"] == {m["id"]: 3555 for m in narr}


def test_relay_roundtrip(tmp_path):
    from narrate.clients import get_client
    from narrate.relay import PendingResponse, Relay, parse_reply, render_request
    from narrate.sandbox import TOOL_SCHEMAS

    cfg = {"id": "relay-x", "provider": "relay", "relay_dir": str(tmp_path), "agent_tier": "small"}
    client = get_client(cfg)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "explain Mean($close, 5)"}]
    with pytest.raises(PendingResponse) as e:
        client.complete(msgs, temperature=0.7, seed=1)
    rel = Relay(tmp_path)
    pend = rel.pending()
    assert len(pend) == 1 and pend[0]["model"] == "relay-x" and "explain Mean" in open(pend[0]["request"]).read()
    with pytest.raises(PendingResponse):                     # a different seed is a different request
        client.complete(msgs, temperature=0.7, seed=2)
    rel.put(e.value.key, "A five-day moving average of the close.")
    assert client.complete(msgs, temperature=0.7, seed=1).text == "A five-day moving average of the close."
    assert len(rel.pending()) == 1
    txt, calls = parse_reply('TOOL_CALL: {"name": "describe", "arguments": {"expr": "$close"}}\n'
                             'TOOL_CALL: {"name": "perturb", "arguments": {"expr": "$close", "field": "close"}}')
    assert txt == "" and [c["name"] for c in calls] == ["describe", "perturb"] and calls[1]["id"] == "call_1"
    assert parse_reply("plain answer") == ("plain answer", [])
    r = render_request(msgs + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "describe", "arguments": '{"expr": "$close"}'}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": '{"coverage": 1.0}'}], TOOL_SCHEMAS, 700)
    assert "TOOL_CALL" in r and "=== tool (call_0) ===" in r and "compute_signal(expr: string)" in r


def test_pilot_plan_counts():
    from narrate.plan import plan_pilot_cells, plan_summary
    from pools.records import read_jsonl

    recs = []
    for pool, n in (("K", 14), ("SP", 14), ("NL", 14), ("N", 14)):
        recs += [{"formula_id": f"{pool}-{i}", "pool": pool, "arm": "B", "base_id": f"b{i}"} for i in range(n)]
    recs += [{"formula_id": f"SA-{i}", "pool": "SA", "arm": "B", "base_id": f"b{i}",
              "perturbation": {"type": "sa_sign"}} for i in range(14)]
    recs += [{"formula_id": f"P1-{m}-{i}", "pool": "P1", "arm": "A", "stratum": "raw", "author_model": m}
             for m in ("m1", "m2") for i in range(14)]
    recs += [{"formula_id": f"P3a-{i}", "pool": "P3a", "arm": "A"} for i in range(8)]
    recs += [{"formula_id": f"P3b-{i}", "pool": "P3b", "arm": "A"} for i in range(8)]
    cells = plan_pilot_cells(recs, [{"id": "m1", "family": "f"}, {"id": "m2", "family": "f"}], n=12, k=3)
    s = plan_summary(cells)
    # (K, SP, SA-sign, NL, N) x 12 x A0 + (P1, P3) x 12 x (A0, A2) = 108 cells, x 3 samples
    assert s["per_model"] == {"m1": 324, "m2": 324}
    assert all(c.formula_id.split("-")[1] == c.model for c in cells if c.pool == "P1")
    assert read_jsonl is not None


# ------------------------------------------------------------------ audit fixes (§4.2, §4.3, §8.2, §8.3, §14)
def _p2_records(fams, n=60):
    narr = [{"id": f"m{i}", "family": f} for i, f in enumerate(fams)]
    recs = [{"formula_id": f"P2-m{i}-{j}", "pool": "P2", "arm": "A", "author_model": f"m{i}"}
            for i in range(len(fams)) for j in range(n)]
    return narr, recs


@pytest.mark.parametrize("fams", [("a", "a", "b", "b", "c", "c"), ("a", "a", "a", "b", "c", "d"), ("a", "b")])
def test_one_cross_narration_per_p2_formula(fams):
    from collections import Counter

    narr, recs = _p2_records(fams)
    cells = plan_cells(recs, narr, k=3, reference_t0=False)
    cross = [c for c in cells if c.cross]
    fam = {m["id"]: m["family"] for m in narr}
    per_f = {f: {c.model for c in cross if c.formula_id == f} for f in {r["formula_id"] for r in recs}}
    assert all(len(v) == 1 for v in per_f.values())
    assert all(fam[c.model] != fam[c.formula_id.split("-")[1]] for c in cross)
    assert set(Counter(c.model for c in cross if c.sample_idx == 0).values()) == {60}    # balanced: 60 per narrator
    s = plan_summary(cells, recs)["p2_cross_narration"]
    assert s["ok"] and s["cross_narrators_per_formula"] == {"1": len(recs)}
    single, srecs = _p2_records(("a", "a"), n=3)
    assert not plan_summary(plan_cells(srecs, single, k=1), srecs)["p2_cross_narration"]["ok"]


def test_reference_t0_is_planned_by_default():
    narr, recs = _p2_records(("a", "b"), n=4)
    cells = plan_cells(recs, narr, k=3)
    t0 = [c for c in cells if c.sample_idx < 0]
    assert t0 and all(c.temperature == 0.0 for c in t0)
    assert len(t0) == len({(c.model, c.formula_id, c.access, c.variant) for c in cells if c.sample_idx >= 0})
    assert plan_summary(cells)["t0_reference"] == len(t0)
    assert not [c for c in plan_cells(recs, narr, k=3, reference_t0=False) if c.sample_idx < 0]


def test_ablation_cells_and_templates():
    from configs import file_sha256
    from configs import PROMPT_DIR

    # the pre-registered primary templates are untouched
    assert file_sha256(PROMPT_DIR / "narrator_a0_guided.txt") == "a400a536d5b061e4a1355a4d358af434267c39c35ca45c6a102d7759e4a848f2"
    assert file_sha256(PROMPT_DIR / "narrator_a0_minimal.txt") == "095483645d6624f67d213f786476539dae3e45154241029d14959409b3c32666"
    assert file_sha256(PROMPT_DIR / "narrator_system.txt") == "2b5cc6df1073ce46db70cc150734d26783271e0c2e5b7c0273ab9740aa90c5c7"
    ms, _ = build_messages(_rec(), "A0", "structured", "CN", seed=1)
    assert "bullet facts" in ms[1]["content"] and ms[0]["content"] == prompt("narrator_system")
    mc, meta = build_messages(_rec(), "A0", "cap150", "CN", seed=1)
    g, gmeta = build_messages(_rec(), "A0", "guided", "CN", seed=1)
    assert "at most 150 words" in mc[0]["content"] and mc[1] == g[1] and meta["template_sha256"]["user"] == gmeta["template_sha256"]["user"]
    me, _ = build_messages(_rec(), "A0", "guided_effort_high", "CN", seed=1)
    assert me == g
    with pytest.raises(ValueError):
        build_messages(_rec(), "A0", "bogus", "CN", seed=1)
    narr = with_role(load_models("models.mock.yaml"), "narrator")
    recs = [{"formula_id": f"P3a{i}", "pool": "P3a", "arm": "A"} for i in range(8)]
    base = plan_cells(recs, narr, k=3, reference_t0=False)
    cells = plan_cells(recs, narr, k=3, reference_t0=False, ablations=("structured", "cap150", "reasoning_effort"))
    extra = [c for c in cells if c not in base]
    assert {c.variant for c in extra} == {"structured", "cap150", "guided_effort_low", "guided_effort_high"}
    assert all(c.access == "A0" and c.sample_idx >= 0 for c in extra)
    eff = [c for c in extra if c.reasoning_effort]
    assert eff and {c.model for c in eff} == {"mock-sloppy"} and {c.reasoning_effort for c in eff} == {"low", "high"}
    assert len({c.rationale_id for c in cells}) == len(cells)
    assert plan_summary(cells)["ablation_cells"] == len(extra)
    with pytest.raises(ValueError):
        plan_cells(recs, narr, ablations=("bogus",))


class _ToolHappyClient:
    """Calls a tool on every turn; answers only when tools are disabled (unless ``stubborn``)."""

    def __init__(self, stubborn=False):
        from narrate.clients import LLMClient

        LLMClient.__init__(self, {"id": "mock-careful", "family": "mock_a"})
        self.stubborn, self.calls = stubborn, []

    provenance = lambda self: {"model_id": self.model_id}

    def complete(self, messages, tools=None, temperature=0.7, max_tokens=700, seed=None, tool_choice=None,
                 reasoning_effort=None):
        from narrate.clients import LLMResponse

        self.calls.append(tool_choice)
        if tool_choice == "none" and not self.stubborn:
            return LLMResponse("The factor is a short-term reversal signal built from the five-day return of the "
                               "closing price; high values predict higher returns over the next week.", [], {"in": 1, "out": 1})
        n = len(self.calls)
        return LLMResponse("", [{"id": f"c{n}a", "name": "describe", "arguments": {"expr": "$close"}},
                                {"id": f"c{n}b", "name": "describe", "arguments": {"expr": "$open"}}],
                           {"in": 1, "out": 1}, finish_reason="tool_calls")


@pytest.mark.parametrize("stubborn", [False, True])
def test_a2_budget_exhaustion_gets_a_final_answer(tmp_path, stubborn):
    from narrate.plan import Cell

    ctx = calibration_context(n_stocks=30, n_days=400, seed=3, fast=True)
    cfg = load_models("models.mock.yaml")
    recs = {"K1": _rec("K1")}
    runner = NarrationRunner(cfg, recs, tmp_path, ctx)
    stub = _ToolHappyClient(stubborn)
    runner.clients["mock-careful"] = stub
    r = runner.narrate(Cell("mock-careful", "K1", "A2", "guided", 0, 0.7, "B", "K"))
    assert r["tool_budget_exhausted"] and len(r["tool_calls"]) == 10
    log = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    final = [x for x in log if x["role"] == "narrator_final_turn"]
    assert final and final[0]["params"]["tool_choice"] == "none"
    assert "a2_budget_exhausted" in final[0]["template_sha256"]
    assert final[0]["request"]["messages"][-1]["content"] == prompt("narrator_a2_budget_exhausted")
    if not stubborn:
        assert r["status"] == "ok" and "reversal" in r["text"] and len(r["attempts"]) == 1
    else:
        assert r["status"] == "tool_budget_exhausted" and len(r["attempts"]) == 2
        assert all(a["status"] == "tool_budget_exhausted" and len(a["tool_calls"]) == 10 for a in r["attempts"])


def test_reasoning_setting_and_provenance_logged(tmp_path):
    from narrate.plan import Cell

    cfg = load_models("models.mock.yaml")
    recs = {"P3a-0": {**_rec("P3a-0", "Mean($close, 5)/$close", pool="P3a"), "arm": "A"}}
    runner = NarrationRunner(cfg, recs, tmp_path)
    out = runner.run([Cell("mock-sloppy", "P3a-0", "A0", "guided", 0, 0.7, "A", "P3a"),
                      Cell("mock-sloppy", "P3a-0", "A0", "guided_effort_high", 0, 0.7, "A", "P3a", reasoning_effort="high"),
                      Cell("mock-careful", "P3a-0", "A0", "guided", 0, 0.7, "A", "P3a")])
    assert [o["reasoning_setting"] for o in out] == ["default", "high", None]
    log = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(log) == 3 and len({x["prompt_hash"] for x in log[:2]}) == 2       # effort is part of the cache key
    assert [x["params"]["reasoning_setting"] for x in log] == ["default", "high", None]
    assert log[1]["request_params"]["reasoning_effort"] == "high"
    assert log[0]["provenance"]["model_string"] == "mock-sloppy-v1" and "reasoning" in log[0]["response"]
    assert validate_roster(cfg, main_run=True).ok
    bad = {**cfg, "models": [{**m, "reasoning_setting": None} for m in cfg["models"]]}
    assert any("reasoning setting" in p for p in validate_roster(bad, main_run=True).problems)


def test_openai_client_sends_and_reports_request_params(monkeypatch):
    import io

    from narrate.clients import OpenAICompatibleClient

    sent = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        sent.update(json.loads(req.data))
        return _Resp(json.dumps({"model": "served-x", "system_fingerprint": "fp", "usage": {"prompt_tokens": 3,
                                 "completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 2}},
                                 "choices": [{"finish_reason": "stop", "message": {"content": "ok",
                                                                                     "reasoning_content": "thinking"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = OpenAICompatibleClient({"id": "x", "base_url": "http://h/v1", "model_string": "x-1", "reasoning": True,
                                "extra_body": {"top_p": 1.0}})
    r = c.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}], temperature=0.0, seed=1,
                   tool_choice="none", reasoning_effort="low")
    assert sent["tool_choice"] == "none" and sent["reasoning_effort"] == "low" and sent["top_p"] == 1.0
    assert r.reasoning == "thinking" and r.usage["reasoning"] == 2
    assert r.params == {k: v for k, v in sent.items() if k not in ("messages", "tools")}
    assert c.reasoning_setting == "default" and c.provenance()["reasoning_setting"] == "default"
    sent.clear()
    c.complete([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in sent and "tool_choice" not in sent


def test_relay_final_turn_and_effort_keys(tmp_path):
    from narrate.relay import PendingResponse, Relay
    from narrate.sandbox import TOOL_SCHEMAS

    client = get_client({"id": "relay-x", "provider": "relay", "relay_dir": str(tmp_path)})
    msgs = [{"role": "user", "content": "explain"}]
    keys = []
    for kw in ({"tools": TOOL_SCHEMAS}, {"tools": TOOL_SCHEMAS, "tool_choice": "none"}, {"reasoning_effort": "high"}, {}):
        with pytest.raises(PendingResponse) as e:
            client.complete(msgs, temperature=0.7, seed=1, **kw)
        keys.append(e.value.key)
    assert len(set(keys)) == 3 and keys[1] == keys[3]           # tools disabled = a plain request
    final = Relay(tmp_path).req / f"{keys[1]}.txt"
    assert "TOOL_CALL" not in final.read_text()
