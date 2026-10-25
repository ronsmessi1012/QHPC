"""
Step 9 — CriticFinding & CriticReport Schemas

Produced by: backend/agents/critic_agent.py
Consumed by: Final Synthesis & Report Generator (Step 10), and the Step 6
             calibration loop.

The critic's job is to judge the FRAMEWORK, not the workload. Steps 6 and 7
make two claims about every run — this will cost X, and this backend is the
right one — and Step 8 measures what actually happened. Step 9 is where
those claims are checked against the measurement.

A finding is therefore always about the framework's own behaviour: a cost
model that mispredicted, a backend choice that a measurement contradicts, a
constant that was never calibrated. A slow algorithm is not a finding; a
slow algorithm the framework predicted would be fast is.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low", "info"]

FindingCategory = Literal[
    "prediction_error",        # Step 6 cost model was wrong
    "uncalibrated_estimate",   # the prediction rested on reasoned constants
    "correctness_failure",     # the workload produced a wrong answer
    "suboptimal_routing",      # Step 7 chose a backend a measurement contradicts
    "execution_failure",       # the run did not complete
    "missing_implementation",  # the algorithm has no executable workload
    "unverified_hardware",     # execution was refused as hypothetical
]


class CriticFinding(BaseModel):
    """One specific, evidenced judgement about the framework's behaviour."""

    category: FindingCategory
    severity: Severity
    algorithm: str
    backend_type: str

    summary: str = Field(..., description="One sentence stating the problem")
    evidence: str = Field(
        ..., description="The measured numbers that support it"
    )
    recommendation: str = Field(
        ..., description="What to change, concretely"
    )

    # Populated for prediction_error so Step 10 and the calibration loop can
    # work with the numbers rather than re-parsing prose.
    predicted: Optional[float] = None
    actual: Optional[float] = None
    ratio: Optional[float] = None


class CalibrationSuggestion(BaseModel):
    """
    A concrete constant adjustment derived from measurement.

    This is the feedback loop: Step 6 predicts, Step 8 measures, Step 9
    computes the correction. `suggested_base_runtime_s` is the existing
    value multiplied by the median observed ratio, so applying it would
    have made the median prediction exact.
    """

    algorithm: str
    observed_runs: int = Field(..., ge=1)
    median_ratio: float
    current_base_runtime_s: Optional[float] = None
    suggested_base_runtime_s: Optional[float] = None
    calibration_host: str
    confidence: Literal["high", "moderate", "low"] = "low"
    note: str = ""


class CriticReport(BaseModel):
    """Aggregate judgement over one or more executions."""

    runs_analysed: int = Field(..., ge=0)
    runs_succeeded: int = Field(..., ge=0)

    findings: List[CriticFinding] = Field(default_factory=list)
    calibration_suggestions: List[CalibrationSuggestion] = Field(
        default_factory=list
    )

    # --- Aggregate cost-model accuracy -----------------------------------
    # None when no run produced a usable ratio, rather than 0.0, which would
    # read as "perfectly wrong" instead of "not measured".
    median_runtime_ratio: Optional[float] = None
    runs_within_2x: int = 0
    runs_with_usable_ratio: int = 0

    # --- Provenance -------------------------------------------------------
    all_estimates_uncalibrated: bool = True
    telemetry_hosts: List[str] = Field(default_factory=list)

    verdict: str = Field(
        ..., description="Deterministic summary of the framework's performance"
    )

    # Present only when an LLM narrative was requested and succeeded. The
    # deterministic fields above never depend on it, so the report is fully
    # usable without a model available.
    llm_narrative: Optional[str] = None
    llm_available: bool = False

    reasoning_trace: Dict[str, Any] = Field(default_factory=dict)

    def findings_by_severity(self, severity: Severity) -> List[CriticFinding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def has_blocking_findings(self) -> bool:
        return any(f.severity in ("critical", "high") for f in self.findings)
