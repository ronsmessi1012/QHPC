"""
Step 5.4 — BorderlineResult Schema

=============================================================================
RECONSTRUCTED — replace with the original when available
=============================================================================
Rebuilt from the "Schema (BorderlineResult)" table in docs/sampling.md,
because backend/models/ is excluded from the repository by a bare `models/`
pattern in .gitignore.
=============================================================================

Routing thresholds that produce `routing_recommendation`:
    dS >= 0.25           classical_only / quantum_only
    0.15 <= dS < 0.25    dual_execution
    dS < 0.15            borderline_review
"""

from typing import Literal

from pydantic import BaseModel, Field


class BorderlineResult(BaseModel):
    classical_score: float = Field(..., ge=0.0, le=1.0, description="S_C")
    quantum_score: float = Field(..., ge=0.0, le=1.0, description="S_Q")
    score_margin: float = Field(..., ge=0.0, le=1.0,
                                description="dS = |S_C - S_Q|")
    borderline_score: float = Field(..., ge=0.0, le=1.0,
                                    description="B = 1 - dS")
    routing_recommendation: Literal[
        "classical_only", "quantum_only", "dual_execution", "borderline_review"
    ]
    winner_family: Literal["classical", "quantum", "tie"]
    explanation: str
