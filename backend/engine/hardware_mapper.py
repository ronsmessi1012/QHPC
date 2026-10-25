"""
Step 7 — Hardware Mapper

Input:  ResourceEstimate, ExecutionDecision
Output: HardwareMapping

Deterministic threshold routing across this deployment's available backends:
CPU, CUDA GPU, MPI cluster, or quantum simulator. No physical QPU is
targeted — quantum work runs on a classical simulator.
"""

from typing import Any, Dict, Optional

from backend.models.decision_schema import ExecutionDecision
from backend.models.hardware_schema import HardwareMapping
from backend.models.resource_schema import ResourceEstimate
from backend.utils.hardware_detect import HardwareInventory, detect_hardware
from backend.utils.resource_constants import (
    CLASSICAL_RESOURCE_PROFILES,
    DEFAULT_CLASSICAL_PROFILE,
    GPU_MEMORY_THRESHOLD_GB,
    MPI_RUNTIME_THRESHOLD_S,
    MPS_QUBIT_LIMIT,
    STATEVECTOR_QUBIT_LIMIT,
)


def _qubit_limit_for(strategy: str) -> int:
    """Classical-simulation ceiling depends on the simulation method."""
    return STATEVECTOR_QUBIT_LIMIT if strategy == "statevector" else MPS_QUBIT_LIMIT


def _map_quantum(
    estimate: ResourceEstimate, decision: ExecutionDecision,
    trace: Dict[str, Any], inventory: HardwareInventory
) -> HardwareMapping:
    strategy = estimate.simulation_strategy or "statevector"
    limit = _qubit_limit_for(strategy)
    qubits = estimate.qubit_count or 0

    trace["simulation_strategy"] = strategy
    trace["qubit_limit_for_strategy"] = limit

    if qubits > limit:
        return HardwareMapping(
            backend_type="qpu_simulator",
            device_spec=(
                f"{strategy} simulator ({qubits} qubits — EXCEEDS practical "
                f"limit of {limit})"
            ),
            justification=(
                f"{decision.selected_algorithm} requires {qubits} qubits, beyond "
                f"the practical {strategy} simulation limit of {limit}. The "
                f"circuit is not classically simulable at this scale with the "
                f"configured method; a larger bond dimension, an approximate "
                f"method, or real quantum hardware would be required."
            ),
            fallback_backend="cpu",
            is_hypothetical=inventory.is_hypothetical,
            reasoning_trace=trace,
        )

    detail = (
        f"{strategy} simulator ({qubits} qubits"
        + (
            f", bond dimension {estimate.reasoning_trace.get('bond_dimension')}"
            if strategy == "mps"
            else ""
        )
        + ")"
    )
    return HardwareMapping(
        backend_type="qpu_simulator",
        device_spec=detail,
        justification=(
            f"{decision.selected_algorithm} is a quantum algorithm and no physical "
            f"QPU is available in this deployment, so it runs on a classical "
            f"{strategy} simulator at {qubits} qubits "
            f"(within the {limit}-qubit limit for this method)."
        ),
        fallback_backend="cpu",
        is_hypothetical=inventory.is_hypothetical,
        reasoning_trace=trace,
    )


def _map_classical(
    estimate: ResourceEstimate, decision: ExecutionDecision,
    trace: Dict[str, Any], inventory: HardwareInventory
) -> HardwareMapping:
    profile = CLASSICAL_RESOURCE_PROFILES.get(
        decision.selected_algorithm, DEFAULT_CLASSICAL_PROFILE
    )
    gpu_capable = profile["gpu_capable"]
    mpi_capable = profile["mpi_capable"]

    # A backend is assignable only when the algorithm supports it AND the
    # deployment has it. Without the second condition the mapper recommends
    # hardware that is not there.
    gpu_assignable = gpu_capable and inventory.cuda_available
    mpi_assignable = mpi_capable and inventory.mpi_available

    trace.update(
        {
            "gpu_capable": gpu_capable,
            "mpi_capable": mpi_capable,
            "cuda_available": inventory.cuda_available,
            "mpi_available": inventory.mpi_available,
            "mpi_slots": inventory.mpi_slots,
            "gpu_assignable": gpu_assignable,
            "mpi_assignable": mpi_assignable,
            "gpu_memory_threshold_gb": GPU_MEMORY_THRESHOLD_GB,
            "mpi_runtime_threshold_s": MPI_RUNTIME_THRESHOLD_S,
        }
    )

    # Record capability the host cannot satisfy, so the gap is visible in
    # the audit trail rather than silently absent from the decision.
    unavailable = []
    if gpu_capable and not inventory.cuda_available:
        unavailable.append("CUDA (algorithm supports it; no CUDA device present)")
    if mpi_capable and not inventory.mpi_available:
        unavailable.append("MPI (algorithm supports it; no MPI runtime present)")
    if unavailable:
        trace["unavailable_backends"] = unavailable

    if gpu_assignable and estimate.estimated_memory_gb >= GPU_MEMORY_THRESHOLD_GB:
        return HardwareMapping(
            backend_type="gpu_cuda",
            device_spec=inventory.gpu_description,
            justification=(
                f"{decision.selected_algorithm} supports GPU acceleration, a CUDA "
                f"device is present ({inventory.gpu_description}), and the "
                f"estimated memory footprint ({estimate.estimated_memory_gb} GB) "
                f"meets the CUDA threshold ({GPU_MEMORY_THRESHOLD_GB} GB)."
            ),
            fallback_backend="cpu",
            is_hypothetical=inventory.is_hypothetical,
            reasoning_trace=trace,
        )

    if mpi_assignable and estimate.estimated_runtime_seconds >= MPI_RUNTIME_THRESHOLD_S:
        topology = (
            f"{inventory.mpi_slots} slots"
            if inventory.planning_mode.value == "detected"
            else f"{inventory.mpi_slots} slots (declared target)"
        )
        return HardwareMapping(
            backend_type="mpi_cluster",
            device_spec=f"MPI ({topology})",
            justification=(
                f"{decision.selected_algorithm} supports MPI parallelism, an MPI "
                f"runtime is available with {inventory.mpi_slots} slots, and the "
                f"estimated runtime ({estimate.estimated_runtime_seconds}s) meets "
                f"the distributed-execution threshold ({MPI_RUNTIME_THRESHOLD_S}s)."
            ),
            fallback_backend="cpu",
            is_hypothetical=inventory.is_hypothetical,
            reasoning_trace=trace,
        )

    # CPU is the floor. Be explicit about WHY, since "no GPU on this host"
    # and "too small to need one" are very different conclusions.
    if unavailable:
        reason = (
            f"the accelerated backends it supports are not available on this "
            f"deployment ({'; '.join(unavailable)})"
        )
    else:
        reason = (
            f"its estimated footprint (memory="
            f"{estimate.estimated_memory_gb} GB, runtime="
            f"{estimate.estimated_runtime_seconds}s) does not cross the GPU or "
            f"MPI thresholds, or the algorithm does not support that form of "
            f"parallelism"
        )

    return HardwareMapping(
        backend_type="cpu",
        device_spec=(
            f"CPU ({inventory.cpu_logical_cores} logical cores, "
            f"{inventory.total_ram_gb:.1f} GB RAM)"
        ),
        justification=f"{decision.selected_algorithm} runs on CPU because {reason}.",
        fallback_backend=None,
        is_hypothetical=inventory.is_hypothetical,
        reasoning_trace=trace,
    )


def map_hardware(
    estimate: ResourceEstimate,
    decision: ExecutionDecision,
    inventory: Optional[HardwareInventory] = None,
) -> HardwareMapping:
    """
    Assign an execution backend for Step 7.

    `inventory` defaults to the detected host hardware. Pass one explicitly
    to map against a different deployment, or to keep a test deterministic
    regardless of the machine it runs on.
    """
    inventory = inventory or detect_hardware()

    trace: Dict[str, Any] = {
        "selected_family": decision.selected_family,
        "selected_algorithm": decision.selected_algorithm,
        "estimated_memory_gb": estimate.estimated_memory_gb,
        "estimated_runtime_seconds": estimate.estimated_runtime_seconds,
        "qubit_count": estimate.qubit_count,
        "estimate_calibrated": estimate.calibrated,
        "planning_mode": inventory.planning_mode.value,
        "hardware_is_hypothetical": inventory.is_hypothetical,
        "detection_notes": list(inventory.detection_notes),
    }

    if decision.selected_family == "quantum":
        return _map_quantum(estimate, decision, trace, inventory)
    return _map_classical(estimate, decision, trace, inventory)
