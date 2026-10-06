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
