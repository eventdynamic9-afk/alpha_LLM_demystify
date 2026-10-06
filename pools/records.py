"""Formula record schema (Appendix D) and JSONL I/O."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dsl import Node, canonical_hash, descriptors, to_alpha101, to_math, to_qlib


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class FormulaRecord:
    formula_id: str
    arm: str                                   # "A" (authorship) | "B" (counter-recall)
    pool: str                                  # K, K_named, SP, SA, NL, N | P1, P2, P3a, P3b, P4
    dsl: str                                   # canonical surface (extended Qlib DSL) of the executed tree
    presented: str                             # exact string the narrator sees
    notation: str                              # notation of `presented`: qlib | alpha101 | gtja | math | program | anonymized
    base_id: str | None = None                 # library short id for Arm B variants
    perturbation: dict | None = None           # {"type", "target_property", "expected_change"}
    label: str | None = None                   # "Name: ..." line (K_named, NL)
    legend: dict | None = None                 # field anonymization legend (SP-iv)
    surface: dict = field(default_factory=dict)
    canonical_hash: str = ""
    complexity: dict = field(default_factory=dict)
    validity: dict = field(default_factory=dict)
    stratum: str | None = None                 # P1: raw | mined ; base tercile for Arm B
    author_model: str | None = None            # P1 / P2
    hypothesis: str | None = None              # P1 structured hypothesis
    provided_rationale: str | None = None      # P4 in-the-wild rationale text
    novelty: dict | None = None                # N pool checks
    trials: int | None = None                  # candidates behind the formula (DSR / PBO)
    created_at: str = field(default_factory=now_iso)
    seed: int | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_node(cls, formula_id: str, arm: str, pool: str, node: Node, presented: str | None = None,
                  notation: str = "qlib", **kw) -> "FormulaRecord":
        q = to_qlib(node)
        rec = cls(formula_id=formula_id, arm=arm, pool=pool, dsl=q, presented=presented or q, notation=notation, **kw)
        rec.surface = {"qlib": q, "alpha101": to_alpha101(node), "math": to_math(node)}
        rec.canonical_hash = canonical_hash(node)
        rec.complexity = descriptors(node)
        return rec

    def to_dict(self) -> dict:
        return asdict(self)


def write_jsonl(records, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict() if hasattr(r, "to_dict") else r, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
