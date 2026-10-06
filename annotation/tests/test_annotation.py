import csv

from annotation.tools import (adjudicate, agreement_report, export_verdict_spotcheck, read_sheet,
                              stratified_rationale_sample, write_annotation_sheets)


def _rationales(n=40):
    out = []
    for i in range(n):
        out.append({"rationale_id": f"R{i}", "model": f"m{i % 2}", "pool": ["K", "SA", "P3"][i % 3],
                    "access": ["A0", "A1"][i % 2], "text": "It captures short-term reversal.", "status": "ok"})
    return out


def test_stratified_sample_covers_cells(tmp_path):
    s = stratified_rationale_sample(_rationales(), n=12)
    assert len(s) == 12 and len({(r["model"], r["pool"], r["access"]) for r in s}) == 6
    refresh = stratified_rationale_sample(_rationales(), n=6, exclude={r["rationale_id"] for r in s})
    assert not {r["rationale_id"] for r in refresh} & {r["rationale_id"] for r in s}
    write_annotation_sheets(s, tmp_path)
    assert (tmp_path / "sheet_A.csv").exists() and (tmp_path / "rationales_to_annotate.jsonl").exists()


def _sheet(path, rows):
    cols = ["rationale_id", "claim_no", "span_start", "span_end", "text", "type", "predicate", "args_json", "scope",
            "hedge", "polarity", "horizon", "ambiguous", "note"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**{c: "" for c in cols}, **r})


def test_agreement_and_adjudication(tmp_path):
    base = {"rationale_id": "R1", "span_start": 0, "span_end": 30, "text": "It captures short-term reversal.",
            "type": "C2", "predicate": "RESEMBLES", "args_json": '{"ref": "short-term reversal", "sign": "+"}'}
    _sheet(tmp_path / "a.csv", [{**base, "claim_no": 0}, {**base, "claim_no": 1, "span_start": 40, "span_end": 60,
                                                          "predicate": "THEORY", "type": "C5", "args_json": "{}"}])
    _sheet(tmp_path / "b.csv", [{**base, "claim_no": 0}])
    a, b = read_sheet(tmp_path / "a.csv"), read_sheet(tmp_path / "b.csv")
    rep = agreement_report(a, b)
    assert rep["n_units"] == 2 and rep["krippendorff_alpha"] < 1
    gold = adjudicate(a, b)
    assert sum(1 for g in gold if g.get("adjudication") == "agreed") == 1
    assert sum(1 for g in gold if g.get("needs_adjudication")) == 1


def test_spotcheck_export(tmp_path):
    verdicts = [{"claim_id": f"c{i}", "verdict": "SUPPORTED", "method": "static", "evidence": {"x": i}} for i in range(5)]
    n = export_verdict_spotcheck(verdicts, {"c0": {"text": "t", "predicate": "SIGN", "args": {}}}, tmp_path / "s.csv", n=3)
    assert n == 3 and len((tmp_path / "s.csv").read_text().splitlines()) == 4
