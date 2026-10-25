"""
Step 6 — Resource Estimator

Input:  ExecutionDecision (+ ProblemSpecification for dimensionality/dataset_size)
Output: ResourceEstimate

Purely deterministic — no LLM call. Mirrors the Feature Scoring Engine
(Step 4) in spirit: table lookups plus closed-form formulas, fully traced.
"""

import math
from typing import Any, Dict, Optional

from backend.models.decision_schema import ExecutionDecision
from backend.models.problem_schema import ProblemSpecification
from backend.models.resource_schema import ResourceEstimate
from backend.utils.hardware_detect import HardwareInventory, detect_hardware
from backend.utils.resource_constants import (
    BYTES_PER_GB,
    CLASSICAL_RESOURCE_PROFILES,
    DEFAULT_CLASSICAL_PROFILE,
    DEFAULT_QUANTUM_PROFILE,
    GPU_SPEEDUP_FACTOR,
    MAX_MPI_SPEEDUP,
    MAX_REPORTED_MEMORY_GB,
    MAX_STATEVECTOR_EXPONENT,
    MPI_PARALLEL_EFFICIENCY,
    MPS_BYTES_PER_ELEMENT,
    MPS_TENSORS_PER_SITE,
    QUANTUM_RESOURCE_PROFILES,
    RESOURCE_TIER_THRESHOLDS,
    SIM_FLOPS_PER_SECOND,
    STATEVECTOR_BYTES_PER_AMPLITUDE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resource_tier(memory_gb: float) -> str:
    if memory_gb <= RESOURCE_TIER_THRESHOLDS["light"]:
        return "light"
    if memory_gb <= RESOURCE_TIER_THRESHOLDS["moderate"]:
        return "moderate"
    if memory_gb <= RESOURCE_TIER_THRESHOLDS["heavy"]:
        return "heavy"
    return "extreme"


def _statevector_memory_gb(n_qubits: int) -> float:
    """2^n complex128 amplitudes, clamped so an absurd n cannot overflow."""
    if n_qubits > MAX_STATEVECTOR_EXPONENT:
        return MAX_REPORTED_MEMORY_GB
    raw = (2 ** n_qubits) * STATEVECTOR_BYTES_PER_AMPLITUDE / BYTES_PER_GB
    return min(raw, MAX_REPORTED_MEMORY_GB)


def _mps_memory_gb(n_qubits: int, bond_dimension: int) -> float:
    """n sites, each ~2*chi^2 complex128 elements. Polynomial, not exponential."""
    elements = n_qubits * MPS_TENSORS_PER_SITE * (bond_dimension ** 2)
    return min(elements * MPS_BYTES_PER_ELEMENT / BYTES_PER_GB,
               MAX_REPORTED_MEMORY_GB)


def _two_qubit_gate_count(circuit_depth: int, n_qubits: int) -> int:
    """Rough two-qubit gate count: depth * n / 2 (each layer pairs qubits)."""
    return max(1, (circuit_depth * n_qubits) // 2)


# ---------------------------------------------------------------------------
# Classical estimation
# ---------------------------------------------------------------------------

def _estimate_classical(
    algorithm: str, dimensionality: int, dataset_size: int,
    inventory: HardwareInventory
) -> Dict[str, Any]:
    profile = CLASSICAL_RESOURCE_PROFILES.get(algorithm, DEFAULT_CLASSICAL_PROFILE)

    driver_value = (
        dataset_size if profile["size_driver"] == "dataset_size" else dimensionality
    )
    scale = max(driver_value, 1) / profile["ref_size"]

    memory_gb = profile["base_memory_gb"] * (scale ** profile["memory_exp"])
    runtime_s = profile["base_runtime_s"] * (scale ** profile["runtime_exp"])

    # Speedup is claimed only for parallelism the algorithm supports AND the
    # host actually has. A GPU-capable algorithm on a machine with no CUDA
    # device runs serially, and the estimate must say so.
    parallelization_factor = 1.0
    applied = []
    if profile["gpu_capable"] and inventory.cuda_available:
        parallelization_factor = max(parallelization_factor, GPU_SPEEDUP_FACTOR)
        applied.append("cuda")
    if profile["mpi_capable"] and inventory.mpi_available:
        mpi_factor = 1.0 + (inventory.mpi_slots - 1) * MPI_PARALLEL_EFFICIENCY
        mpi_factor = min(mpi_factor, MAX_MPI_SPEEDUP)
        if mpi_factor > parallelization_factor:
            parallelization_factor = mpi_factor
            applied = ["mpi"]
        elif mpi_factor > 1.0:
            applied.append("mpi")

    return {
        "parallelism_applied": applied or ["none"],
        "mpi_slots_used": inventory.mpi_slots if "mpi" in applied else 0,
        "profile_used": profile,
        "size_driver": profile["size_driver"],
        "driver_value": driver_value,
        "ref_size": profile["ref_size"],
        "scale": scale,
        "raw_memory_gb": memory_gb,
        "raw_runtime_s": runtime_s,
        "parallelization_factor": parallelization_factor,
        "qubit_count": None,
        "circuit_depth": None,
        "simulation_strategy": None,
        "calibrated": profile["calibrated"],
    }


# ---------------------------------------------------------------------------
# Quantum estimation
# ---------------------------------------------------------------------------

def _estimate_quantum(
    algorithm: str, dimensionality: int, dataset_size: int
) -> Dict[str, Any]:
    profile = QUANTUM_RESOURCE_PROFILES.get(algorithm, DEFAULT_QUANTUM_PROFILE)

    if profile["qubit_mode"] == "log2":
        qubit_count = max(
            1, math.ceil(profile["qubit_coeff"] * math.log2(max(dataset_size, 2)))
        )
    else:  # "linear"
        qubit_count = max(1, math.ceil(profile["qubit_coeff"] * max(dimensionality, 1)))

    circuit_depth = max(
        1,
        math.ceil(profile["depth_layers"] * qubit_count * profile["depth_multiplier"]),
    )
    gate_count = _two_qubit_gate_count(circuit_depth, qubit_count)

    strategy = profile["sim_strategy"]
    if strategy == "statevector":
        memory_gb = _statevector_memory_gb(qubit_count)
        # Each gate sweeps all 2^n amplitudes.
        amplitudes = (
            float("inf")
            if qubit_count > MAX_STATEVECTOR_EXPONENT
            else float(2 ** qubit_count)
        )
        runtime_s = (
            MAX_REPORTED_MEMORY_GB
            if amplitudes == float("inf")
            else gate_count * amplitudes / SIM_FLOPS_PER_SECOND
        )
        bond_dimension = None
    else:  # "mps"
        bond_dimension = profile["bond_dimension"]
        memory_gb = _mps_memory_gb(qubit_count, bond_dimension)
        # Each two-qubit MPS gate costs O(chi^3).
        runtime_s = gate_count * (bond_dimension ** 3) / SIM_FLOPS_PER_SECOND

    return {
        "profile_used": profile,
        "raw_memory_gb": memory_gb,
        "raw_runtime_s": runtime_s,
        # Simulation here is modeled as a single simulator process; no
        # MPI/CUDA decomposition of the circuit is assumed.
        "parallelization_factor": 1.0,
        "qubit_count": qubit_count,
        "circuit_depth": circuit_depth,
        "gate_count": gate_count,
        "simulation_strategy": strategy,
        "bond_dimension": bond_dimension,
        "calibrated": profile["calibrated"],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def estimate_resources(
    decision: ExecutionDecision,
    spec: ProblemSpecification,
    inventory: Optional[HardwareInventory] = None,
) -> ResourceEstimate:
    """
    Deterministic resource estimation for Step 6.

    `inventory` defaults to the detected host hardware. Pass one explicitly
    to estimate against a different deployment, or to keep a test
    deterministic regardless of the machine it runs on.

    Note: for `routing_mode == "dual_execution"` (or `selected_family ==
    "dual"`) this estimates the SELECTED algorithm only. Callers wanting a
    side-by-side classical/quantum comparison should invoke this twice,
    against best_classical_algorithm and best_quantum_algorithm.
    """
    inventory = inventory or detect_hardware()
    algorithm = decision.selected_algorithm

    if decision.selected_family == "quantum":
        result = _estimate_quantum(algorithm, spec.dimensionality, spec.dataset_size)
    else:
        result = _estimate_classical(
            algorithm, spec.dimensionality, spec.dataset_size, inventory)

    final_memory_gb = result["raw_memory_gb"]
    final_runtime_s = result["raw_runtime_s"] / result["parallelization_factor"]

    reasoning_trace: Dict[str, Any] = {
        "selected_family": decision.selected_family,
        "selected_algorithm": algorithm,
        "dimensionality": spec.dimensionality,
        "dataset_size": spec.dataset_size,
        "profile_used": result["profile_used"],
        "raw_memory_gb": round(result["raw_memory_gb"], 6),
        "raw_runtime_s": round(result["raw_runtime_s"], 6),
        "parallelization_factor": result["parallelization_factor"],
        "final_memory_gb": round(final_memory_gb, 6),
        "final_runtime_s": round(final_runtime_s, 6),
        "calibrated": result["calibrated"],
        "calibration_host": result["profile_used"].get("calibration_host"),
        "planning_mode": inventory.planning_mode.value,
        "hardware_is_hypothetical": inventory.is_hypothetical,
    }
    for key in ("size_driver", "driver_value", "ref_size", "scale",
                "gate_count", "simulation_strategy", "bond_dimension",
                "parallelism_applied", "mpi_slots_used"):
        if key in result:
            value = result[key]
            reasoning_trace[key] = round(value, 6) if isinstance(value, float) else value

    return ResourceEstimate(
        estimated_memory_gb=round(final_memory_gb, 6),
        estimated_runtime_seconds=round(final_runtime_s, 6),
        circuit_depth=result["circuit_depth"],
        qubit_count=result["qubit_count"],
        simulation_strategy=result["simulation_strategy"],
        parallelization_factor=result["parallelization_factor"],
        resource_tier=_resource_tier(final_memory_gb),
        calibrated=result["calibrated"],
        hardware_is_hypothetical=inventory.is_hypothetical,
        reasoning_trace=reasoning_trace,
    )
