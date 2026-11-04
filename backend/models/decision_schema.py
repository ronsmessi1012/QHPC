"""
=============================================================================
TEMPORARY PLACEHOLDER — DELETE WHEN THE REAL FILE ARRIVES
=============================================================================
This is NOT the authoritative ExecutionDecision. It was reconstructed from
the field table in docs/sampling.md so that Steps 6 and 7 could be written
and tested while backend/models/ was missing from the dev branch.

When the real backend/models/ lands, this file is overwritten by the
genuine one — same filename, so a plain copy replaces it. Afterwards,
re-run backend/tests/test_resource_estimator.py and
backend/tests/test_hardware_mapper.py: they exercise the exact field names
Steps 6 and 7 depend on, so a mismatch surfaces immediately as a test
failure rather than silently later.

Fields Steps 6 and 7 actually read: selected_family, selected_algorithm.
=============================================================================
"""

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


class ExecutionDecision(BaseModel):
    selected_family: Literal["classical", "quantum", "dual", "borderline"]
    selected_algorithm: str
    routing_mode: Literal[
        "classical_only", "quantum_only", "dual_execution", "borderline_review"
    ]
    decision_confidence: float = Field(..., ge=0.55, le=0.95)
    borderline_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    reasoning_trace: Dict[str, Any] = Field(default_factory=dict)
