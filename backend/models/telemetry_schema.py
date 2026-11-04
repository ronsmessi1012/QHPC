"""
Step 8 — ExecutionTelemetry Schema

Produced by: backend/engine/performance_engine.py
Consumed by: Critic Agent (Step 9), Report Generator (Step 10),
             and the Step 6 calibration loop.

This schema is where the framework stops asserting and starts measuring.
Steps 6 and 7 PREDICT a cost and CHOOSE a backend; this records what the
run actually cost on the backend actually used, and the gap between the
two.

That gap is the point. A framework that routes workloads is only as good
as its cost model, and `runtime_error_ratio` is the number that says
whether the model works. It is also what Step 9's critic agent needs in
order to have anything to criticise.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

BackendType = Literal["cpu", "gpu_cuda", "mpi_cluster", "qpu_simulator"]


class ExecutionTelemetry(BaseModel):
    # ---- What ran, and where -------------------------------------------
    algorithm: str
    backend_type: BackendType
    device_detail: str = Field(
        ..., description="Concrete device the work actually ran on"
    )
    telemetry_host: str = Field(
        ..., description="Machine identity; runtimes do not transfer between hosts"
    )

    # ---- Did it work ----------------------------------------------------
    success: bool
    error_message: Optional[str] = None

    # ---- Measured ------------------------------------------------------
    actual_runtime_seconds: float = Field(..., ge=0.0)
    actual_peak_memory_gb: float = Field(..., ge=0.0)

    # ---- Predicted, carried over from Step 6 ---------------------------
    predicted_runtime_seconds: float = Field(..., ge=0.0)
    predicted_memory_gb: float = Field(..., ge=0.0)

    # ---- The comparison that matters -----------------------------------
    # ratio = actual / predicted.
    #   1.0  -> the model was right
    #   >1.0 -> the model was optimistic (ran slower/heavier than predicted)
    #   <1.0 -> the model was pessimistic
    # None when the prediction was zero, which would make the ratio
    # undefined; a placeholder number there would be a fabricated result.
    runtime_error_ratio: Optional[float] = None
    memory_error_ratio: Optional[float] = None

    # ---- Evidence the computation was real ------------------------------
    # Without this, telemetry only proves something consumed time. A
    # correctness metric proves the algorithm actually solved its problem.
    result_metric_name: Optional[str] = None
    result_metric_value: Optional[float] = None

    # ---- Provenance ------------------------------------------------------
    # True when the estimate this is compared against used uncalibrated
    # constants, so a large error ratio reflects an unmeasured model
    # rather than a genuine mis-prediction.
    compared_against_uncalibrated_estimate: bool = True

    raw_metrics: Dict[str, Any] = Field(default_factory=dict)

    def runtime_within(self, factor: float = 2.0) -> bool:
        """Whether the runtime prediction landed within `factor`x of reality."""
        if self.runtime_error_ratio is None:
            return False
        return (1.0 / factor) <= self.runtime_error_ratio <= factor

    def memory_within(self, factor: float = 2.0) -> bool:
        """Whether the memory prediction landed within `factor`x of reality."""
        if self.memory_error_ratio is None:
            return False
        return (1.0 / factor) <= self.memory_error_ratio <= factor
