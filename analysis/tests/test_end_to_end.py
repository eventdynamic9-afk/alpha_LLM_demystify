"""Offline end-to-end run of every stage with mock LLMs (the `make smoke` pipeline at minimal size)."""
import json

import pytest


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    import analysis.__main__ as an
    import data.__main__ as dm
    import judges.__main__ as jm
    import narrate.__main__ as nm
    import parse.__main__ as pm
    import paper.__main__ as pp
    import pools.__main__ as po
    import verify.__main__ as vm

    rd = tmp_path / "run"
    panel = str(rd / "panel.npz")
    assert dm.main(["synthetic", "--out", panel, "--stocks", "40", "--days", "500"]) == 0
    assert po.main(["build", "--panel", panel, "--out", str(rd), "--scale", "0.05", "--fast",
                    "--authors", "models.mock.yaml"]) == 0
    assert nm.main(["run", "--formulas", str(rd / "formulas.jsonl"), "--panel", panel, "--run-dir", str(rd),
                    "--models", "models.mock.yaml", "--k", "1", "--fast"]) == 0
    assert pm.main(["run", "--rationales", str(rd / "rationales.jsonl"), "--out", str(rd / "claims.jsonl"),
                    "--models", "models.mock.yaml"]) == 0
    assert vm.main(["run", "--formulas", str(rd / "formulas.jsonl"), "--claims", str(rd / "claims.jsonl"),
                    "--rationales", str(rd / "rationales.jsonl"), "--panel", panel,
                    "--out", str(rd / "verdicts.jsonl"), "--fast"]) == 0
    assert jm.main(["run", "--run-dir", str(rd), "--panel", panel, "--models", "models.mock.yaml",
                    "--judges", "B1,B5", "--fast"]) == 0
    assert an.main(["run", "--run-dir", str(rd), "--n-boot", "50"]) == 0
    assert pp.main(["tables", "--run-dir", str(rd)]) == 0
    res = json.loads((rd / "analysis" / "results.json").read_text())
    assert set(res["confirmatory"]) == {"CF1", "CF2", "CF3", "CF4", "CF5", "CF6"}
    pools = {json.loads(line)["pool"] for line in open(rd / "formulas.jsonl")}
    assert {"K", "SA", "SP", "N", "P1", "P2", "P3a"} <= pools
    assert (rd / "analysis" / "tables.md").read_text().startswith("# Results tables")
