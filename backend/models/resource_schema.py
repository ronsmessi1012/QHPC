"""
Step 6 — ResourceEstimate Schema

Produced by: backend/agents/resource_estimator.py
Consumed by: backend/engine/hardware_mapper.py (Step 7)
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ResourceEstimate(BaseModel):
    estimated_memory_gb: float = Field(..., ge=0.0)
    estimated_runtime_seconds: float = Field(..., ge=0.0)

    # Quantum-only fields (None for classical algorithms)
    circuit_depth: Optional[int] = Field(default=None, ge=0)
    qubit_count: Optional[int] = Field(default=None, ge=0)
    simulation_strategy: Optional[Literal["statevector", "mps"]] = None

    # >1.0 means parallel speedup applied (MPI/CUDA); 1.0 = serial
    parallelization_factor: float = Field(default=1.0, ge=1.0)

    resource_tier: Literal["light", "moderate", "heavy", "extreme"]

    # False when the profile's constants are reasoned defaults rather than
    # measured benchmarks. Surfaced so downstream reports never present an
    # uncalibrated estimate as an empirical one.
    calibrated: bool = False

    # True when the estimate assumed a declared target deployment rather
    # than hardware verified present on this host (planning mode "target").
    hardware_is_hypothetical: bool = False

    # Full formula inputs/outputs for audit — mirrors the reasoning_trace
    # pattern established by ExecutionDecision (Step 5.5).
    reasoning_trace: Dict[str, Any]
