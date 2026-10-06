"""Ground-truth verification engine (§10)."""
from .context import VerificationContext
from .dispatcher import verify_claim
from .verdicts import AMBIGUOUS, REFUTED, SUPPORTED, UNRESOLVED, UNVERIFIABLE, Verdict

__all__ = ["VerificationContext", "verify_claim", "Verdict", "SUPPORTED", "REFUTED", "UNRESOLVED",
           "UNVERIFIABLE", "AMBIGUOUS"]
