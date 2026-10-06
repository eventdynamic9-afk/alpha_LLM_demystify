"""Verdict vocabulary (§10.1)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
UNRESOLVED = "UNRESOLVED"
UNVERIFIABLE = "UNVERIFIABLE"
AMBIGUOUS = "AMBIGUOUS"
VERDICTS = (SUPPORTED, REFUTED, UNRESOLVED, UNVERIFIABLE, AMBIGUOUS)
DECIDABLE = (SUPPORTED, REFUTED)

VERIFIER_VERSION = "v1.0.0"


@dataclass
class Verdict:
    verdict: str
    method: str
    evidence: dict = field(default_factory=dict)
    regime_dependent: bool = False

    @property
    def decidable(self) -> bool:
        return self.verdict in DECIDABLE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verifier_version"] = VERIFIER_VERSION
        return d


def invert(v: Verdict) -> Verdict:
    """Polarity 'deny': the negated claim is verified by inverting a decided verdict."""
    flip = {SUPPORTED: REFUTED, REFUTED: SUPPORTED}
    return Verdict(flip.get(v.verdict, v.verdict), v.method, {**v.evidence, "polarity": "deny"}, v.regime_dependent)


def aggregate_any_all(verdicts: list[Verdict], method: str) -> Verdict:
    """Pre-registered aggregation over a codebook term's operationalizations (Appendix C "report both"):
    SUPPORTED if any SUPPORTED, REFUTED if all REFUTED, otherwise UNRESOLVED."""
    if not verdicts:
        return Verdict(UNVERIFIABLE, method, {"reason": "no operationalization available"})
    ev = {"operationalizations": [v.to_dict() for v in verdicts]}
    if any(v.verdict == SUPPORTED for v in verdicts):
        out = SUPPORTED
    elif all(v.verdict == REFUTED for v in verdicts):
        out = REFUTED
    else:
        out = UNRESOLVED
    return Verdict(out, method, ev, any(v.regime_dependent for v in verdicts))


def interval_verdict(lo: float, hi: float, true_region: tuple[float, float], false_regions: list[tuple[float, float]]) -> str:
    """SUPPORTED if [lo, hi] lies entirely in the true region; REFUTED if entirely inside one false region."""
    if true_region[0] <= lo and hi <= true_region[1]:
        return SUPPORTED
    for a, b in false_regions:
        if a <= lo and hi <= b:
            return REFUTED
    return UNRESOLVED
