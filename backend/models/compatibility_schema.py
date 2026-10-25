"""
Step 5 — CompatibilityResult Schema

=============================================================================
RECONSTRUCTED — replace with the original when available
=============================================================================
Rebuilt from the "Schema (CompatibilityResult)" table in docs/sampling.md,
because backend/models/ is excluded from the repository by a bare `models/`
pattern in .gitignore.
=============================================================================
"""

from typing import Dict

from pydantic import BaseModel, Field


class CompatibilityResult(BaseModel):
    # Scores come from S_compat = 0.45*P + 0.25*C + 0.20*K + 0.10*D_bonus.
    # The weights sum to 1 and every input is in [0, 1], so a score cannot
    # legitimately leave [0, 1] either.
    algorithm_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Algorithm name -> compatibility score, sorted descending",
    )
    best_classical_algorithm: str
    best_quantum_algorithm: str
    best_classical_score: float = Field(..., ge=0.0, le=1.0, description="S_C")
    best_quantum_score: float = Field(..., ge=0.0, le=1.0, description="S_Q")
