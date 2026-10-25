"""
Step 5.5.1 — ArbitrationContext Schema

=============================================================================
RECONSTRUCTED — replace with the original when available
=============================================================================
Rebuilt from the "Schema (ArbitrationContext)" table in docs/sampling.md,
because backend/models/ is excluded from the repository by a bare `models/`
pattern in .gitignore.

This one is corroborated by a second source: backend/agents/agent_selector.py
reads context.borderline_score, .routing_recommendation, .best_classical_score,
.best_quantum_score, .best_classical_algorithm, .best_quantum_algorithm,
.confidence, .dimension_score, .constraint_score, .complexity_score and
.score_margin, and calls context.model_dump_json(). Every one of those is
present below, so the field names are confirmed against real consuming code
rather than documentation alone.
=============================================================================
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ArbitrationContext(BaseModel):
    """
    Immutable snapshot of all deterministic evidence, serialized straight
    into the LLM prompt.

    Typed loosely where the upstream schemas type strictly (problem_class,
    constraint_structure and routing_recommendation are plain str here):
    this is a flattened view assembled for prompting, and a Literal that
    disagreed with an upstream enum would break the builder for no benefit.
    """

    # --- From ProblemSpecification ---------------------------------------
    user_query: str
    problem_class: str
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Problem Analyzer confidence")
    dimensionality: int = Field(..., ge=1)
    dataset_size: int = Field(..., ge=1)
    constraint_structure: str

    # --- From FeatureScore ------------------------------------------------
    dimension_score: float = Field(..., ge=0.0, le=1.0)
    constraint_score: float = Field(..., ge=0.0, le=1.0)
    complexity_score: float = Field(..., ge=0.0, le=1.0)

    # --- From CompatibilityResult ----------------------------------------
    best_classical_algorithm: str
    best_classical_score: float = Field(..., ge=0.0, le=1.0)
    best_quantum_algorithm: str
    best_quantum_score: float = Field(..., ge=0.0, le=1.0)

    # --- From BorderlineResult -------------------------------------------
    score_margin: float = Field(..., ge=0.0, le=1.0)
    borderline_score: float = Field(..., ge=0.0, le=1.0)
    routing_recommendation: str

    # --- Deterministic prose summary of the evidence ----------------------
    summary: str
