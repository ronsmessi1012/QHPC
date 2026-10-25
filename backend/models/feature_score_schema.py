"""
Step 4 — FeatureScore Schema

=============================================================================
RECONSTRUCTED — replace with the original when available
=============================================================================
Rebuilt from the "Schema (FeatureScore)" table in docs/sampling.md, because
backend/models/ is excluded from the repository by a bare `models/` pattern
in .gitignore.
=============================================================================

Documented value ranges (from the Step 4 scoring lookups):
    dimension_score    0.30 / 0.55 / 0.80 / 0.95
    constraint_score   0.30 / 0.50 / 0.60 / 0.95
    complexity_score   0.45 / 0.50 / 0.55 / 0.65 / 0.90 / 0.95

Bounds are [0, 1] rather than those exact sets: a schema is a data
contract, and pinning it to today's lookup tables would make any future
tuning of the scoring engine fail validation instead of simply working.
"""

from pydantic import BaseModel, Field


class FeatureScore(BaseModel):
    dimension_score: float = Field(..., ge=0.0, le=1.0,
                                   description="Difficulty from dimensionality")
    constraint_score: float = Field(..., ge=0.0, le=1.0,
                                    description="Difficulty from constraint structure")
    complexity_score: float = Field(..., ge=0.0, le=1.0,
                                    description="Difficulty from problem class")
    classical_affinity: float = Field(..., ge=0.0, le=1.0,
                                      description="max prior score over classical candidates")
    quantum_affinity: float = Field(..., ge=0.0, le=1.0,
                                    description="max prior score over quantum candidates")
