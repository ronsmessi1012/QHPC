"""
Step 7 — HardwareMapping Schema
Consumed by: Runtime Execution & Telemetry Engine (Step 8)
Produced by: Hardware Mapper (backend/agents/hardware_mapper.py)
"""

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

BackendType = Literal["cpu", "gpu_cuda", "mpi_cluster", "qpu_simulator"]


class HardwareMapping(BaseModel):
    backend_type: BackendType
    device_spec: str = Field(..., description="Concrete device/cluster descriptor")
    justification: str
    fallback_backend: Optional[BackendType] = None

    # True when this mapping targets a DECLARED deployment rather than
    # hardware verified present on the host (planning mode "target"). A
    # report must not present a hypothetical placement as an observed one.
    is_hypothetical: bool = False

    reasoning_trace: Dict[str, Any]
