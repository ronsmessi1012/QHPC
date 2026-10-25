"""
=============================================================================
TEMPORARY PLACEHOLDER — DELETE WHEN THE REAL FILE ARRIVES
=============================================================================
This is NOT the authoritative ProblemSpecification. It was reconstructed
from the field table in docs/sampling.md so that Steps 6 and 7 could be
written and tested while backend/models/ was missing from the dev branch
(the bare `models/` pattern in .gitignore excluded it).

When the real backend/models/ lands, this file is overwritten by the
genuine one — same filename, so a plain copy replaces it. Afterwards,
re-run backend/tests/test_resource_estimator.py and
backend/tests/test_hardware_mapper.py: they exercise the exact field names
Steps 6 and 7 depend on, so a mismatch surfaces immediately as a test
failure rather than silently later.

Fields Steps 6 and 7 actually read: dimensionality, dataset_size.
=============================================================================
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProblemSpecification(BaseModel):
    user_query: str
    problem_class: Literal[
        "classification", "regression", "optimization",
        "simulation", "search", "unknown",
    ]
    confidence: float = Field(..., ge=0.0, le=1.0)
    dimensionality: int = Field(..., ge=1)
    dataset_size: int = Field(..., ge=1)
    linearity: Literal["linear", "nonlinear", "unknown"] = "unknown"
    sparsity: Literal["dense", "sparse", "unknown"] = "unknown"
    constraint_structure: Literal[
        "unconstrained", "constrained", "mixed", "unknown"
    ] = "unknown"
    quantum_candidate: bool = False
    reasoning: Optional[str] = None
