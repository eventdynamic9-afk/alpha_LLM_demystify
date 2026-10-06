import numpy as np

from configs import models as load_models
from dsl import parse
from judges.b5_nli import SurfaceNLI, describe_ast, judge_b5
from judges.b6_human import export_packets, import_verdicts
from judges.llm_judges import concordance, judge_b1, judge_b2, judge_b3, judge_b4
from judges.validity import auroc, claim_agreement, cost_per_claim, holistic_validity, rationale_truth, reconstruction_table
from narrate.clients import MockClient
from parse.normalize import normalize_claim
from parse.rules import extract_claims
from pools.records import FormulaRecord
from verify.calibration import calibration_context


def _client():
    return MockClient(next(m for m in load_models("models.mock.yaml")["models"] if m["id"] == "mock-parser"))


def _rec():
    return FormulaRecord.from_node("F1", "B", "K", parse("CSRank(-1*($close/Ref($close, 5)-1))")).to_dict()


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.standard_normal(200) + y
    assert abs(auroc(s, y) - roc_auc_score(y, s)) < 1e-12


def test_llm_judges_with_mock():
    ctx = calibration_context(n_stocks=30, n_days=400, seed=3, fast=True)
    c = _client()
    rec = _rec()
    text = "A higher close today lowers the factor value. It uses a 6-day window. It captures short-term reversal."
    b1 = judge_b1(c, rec, text)
    assert 0 <= b1["c2"] <= 1 and b1["C"] == b1["c2"]
    b1h = judge_b1(c, rec, text, hypothesis="Observation: ...")
    assert b1h["c1"] is not None
    claim = normalize_claim(extract_claims(text, "r")[0])
    assert judge_b2(c, rec, claim)["verdict"] in ("TRUE", "FALSE", "CANT_TELL")
    assert judge_b3(c, rec, claim, ctx)["verdict"] in ("TRUE", "FALSE", "CANT_TELL")
    b4 = judge_b4(c, rec, text, ctx)
    assert "reconstructable_alphalogics" in b4


def test_b5_surface_nli():
    node = parse("CSRank(-1*($close/Ref($close, 5)-1))")
    prem = describe_ast(node)
    assert "lowers the factor value" in prem and "6-day window" in prem
    nli = SurfaceNLI()
    yes = normalize_claim(extract_claims("A higher close today lowers the factor value.", "r")[0])
    no = normalize_claim(extract_claims("A higher close today raises the factor value.", "r")[0])
    assert judge_b5(nli, node, yes)["verdict"] == "TRUE"
    assert judge_b5(nli, node, no)["verdict"] == "FALSE"


def test_validity_metrics():
    claims = [{"claim_id": "a1", "rationale_id": "A", "type": "C1"}, {"claim_id": "a2", "rationale_id": "A", "type": "C2"},
              {"claim_id": "b1", "rationale_id": "B", "type": "C1"}]
    verdicts = [{"claim_id": "a1", "verdict": "SUPPORTED"}, {"claim_id": "a2", "verdict": "REFUTED"},
                {"claim_id": "b1", "verdict": "SUPPORTED"}]
    truth = rationale_truth(verdicts, claims)
    assert truth["A"]["has_refuted"] and truth["B"]["precision"] == 1.0
    hv = holistic_validity([{"rationale_id": "A", "C": 0.9}, {"rationale_id": "B", "C": 0.95}], truth)
    assert hv["false_accept_rate"] == 1.0 and hv["auroc_detect_refuted"] == 1.0
    ag = claim_agreement([{"judge": "B2", "claim_id": "a1", "verdict": "TRUE"},
                          {"judge": "B2", "claim_id": "a2", "verdict": "TRUE"}],
                         {v["claim_id"]: v for v in verdicts}, {c["claim_id"]: c for c in claims})
    assert ag["B2|C1"]["accuracy"] == 1.0 and ag["B2|C2"]["accuracy"] == 0.0
    t = reconstruction_table([{"rationale_id": "A", "reconstructable_alphalogics": True}], truth)
    assert t["recon_true_refuted"] == 1
    assert cost_per_claim([{"judge": "B2", "tokens": {"in": 10, "out": 5}}])["B2"]["tokens_per_claim"] == 15


def test_concordance():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((20, 30))
    assert concordance(a, a) == 1.0 and concordance(a, -a) == 0.0


def test_human_packets_roundtrip(tmp_path):
    rec = _rec()
    rationales = [{"rationale_id": "R1", "formula_id": "F1", "text": "It captures short-term reversal."}]
    claims = [{"claim_id": "R1-c0", "rationale_id": "R1", "predicate": "RESEMBLES", "args": {"ref": "short-term reversal"},
               "text": "It captures short-term reversal."}]
    res = export_packets(rationales, claims, {"F1": rec}, {"R1-c0": {"method": "signal_corr", "evidence": {}}}, tmp_path, n=5)
    assert res == {"rationales": 1, "claims": 1}
    p = tmp_path / "phase1_without_evidence.csv"
    txt = p.read_text().replace(",,\n", ",TRUE,3.5\n", 1).replace(",,\r\n", ",TRUE,3.5\r\n", 1)
    p.write_text(txt)
    v = import_verdicts(p, "ann1", "phase1")
    assert v and v[0]["verdict"] == "TRUE" and v[0]["minutes"] == 3.5
