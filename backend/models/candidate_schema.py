"""
Step 3 — AlgorithmCandidate & CandidatePool Schemas

=============================================================================
RECONSTRUCTED — replace with the original when available
=============================================================================
Rebuilt from the "Schema (CandidatePool & AlgorithmCandidate)" table in
docs/sampling.md, because backend/models/ is excluded from the repository
by a bare `models/` pattern in .gitignore (intended for the Ollama model
cache, it also matches backend/models/).

Validation is deliberately permissive: bounds come from the documented
ranges only, so that a legitimate value produced by the existing agents is
never rejected by a reconstruction. If the original is stricter, replacing
this file will simply tighten it.
=============================================================================
"""

from typing import List, Literal

from pydantic import BaseModel, Field


class AlgorithmCandidate(BaseModel):
    name: str = Field(..., description="Formal algorithm name")
    family: Literal["classical", "quantum"]
    algorithm_type: str = Field(..., description="Problem class addressed")
    prior_score: float = Field(..., ge=0.0, le=1.0,
                               description="Expert prior suitability score")
    complexity: str = Field(..., description="Asymptotic time or circuit scaling")
    justification: str = Field(..., description="Reason for candidate inclusion")


class CandidatePool(BaseModel):
    # Both lists are sorted descending by prior_score (Step 3's ordering
    # rule). The schema does not enforce it — that is the generator's job,
    # and a contract that silently reordered data would hide its bugs.
    classical_candidates: List[AlgorithmCandidate] = Field(default_factory=list)
    quantum_candidates: List[AlgorithmCandidate] = Field(default_factory=list)
